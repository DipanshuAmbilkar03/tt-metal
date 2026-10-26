// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "call_sites.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>

#include "hostdev/profiler_zone_id.h"
#include "hostdev/streaming_profiler_common.h"
#include "tt_elffile.hpp"

namespace tt::llrt {

namespace {

namespace sp = tt::tt_metal::experimental::streaming_profiler;
namespace spd = sp::detail;

// The fields each struct name has, as kernels and the host show it. The host matches a type by its name, so every
// showing of a name must have the same fields. A long, unsigned long or size_t field is 32 bits on the device and 64 on
// the host, which neither compiler can tell from a fixed-width type, so it shows as a difference here.
class TypeLayouts {
public:
    void add_device_type(std::string_view name, std::span<const uint16_t> codes, std::span<const FieldBits> bits) {
        const auto [shown, added] = device_.try_emplace(std::string(name));
        if (added) {
            shown->second = {{codes.begin(), codes.end()}, {bits.begin(), bits.end()}};
        } else if (!std::ranges::equal(shown->second.codes, codes) || !std::ranges::equal(shown->second.bits, bits)) {
            TT_THROW("streaming profiler: two kernels send different types named {}; give each its own name", name);
        }
        if (const auto host = host_.find(name); host != host_.end()) {
            check_same_fields(name, host->second, codes);
        }
    }
    void add_host_type(std::string_view name, std::span<const uint16_t> codes) {
        host_.try_emplace(std::string(name), codes.begin(), codes.end());
        if (const auto device = device_.find(name); device != device_.end()) {
            check_same_fields(name, device->second.codes, codes);
        }
    }

private:
    struct Layout {
        std::vector<uint16_t> codes;
        std::vector<FieldBits> bits;
    };
    static void check_same_fields(
        std::string_view name, std::span<const uint16_t> device_codes, std::span<const uint16_t> host_codes) {
        if (!std::ranges::equal(device_codes, host_codes)) {
            TT_THROW(
                "streaming profiler: {} has different fields in kernels than on the host; rebuild the host, and use "
                "fixed-width types instead of long or size_t",
                name);
        }
    }

    std::map<std::string, Layout, std::less<>> device_;
    std::map<std::string, std::vector<uint16_t>, std::less<>> host_;
};

// The stall call site, which kernel_profiler::write_pending_stall() writes without an entry in any ELF: a zone named
// STALL_ZONE_NAME whose records all take the medium form, its end the record's time and its start a 32-bit distance
// before its end. Its name is the String with hash 0 of its own strings.
CallSite make_stall_site() {
    auto strings = std::make_shared<CallSiteStrings>();
    strings->text.assign(sp::STALL_ZONE_NAME.begin(), sp::STALL_ZONE_NAME.end());
    strings->by_hash.emplace(0, std::string_view(strings->text.data(), strings->text.size()));
    return {
        .metadata_type = spd::type_name<sp::ZoneMetadata>,
        .data_codes = {spd::kTimestampCode, spd::kTimestampCode},
        .metadata_codes = {spd::kStringCode},
        .metadata_bits = {{0, 64}},
        .values = {{.type = spd::type_name<sp::Interval>, .first_field = 0, .field_bits = {{0, 64}, {64, 64}}}},
        .value_fields = {2},
        .forms =
            {{.time_form = spd::TimeForm::Medium,
              .body_words = kernel_profiler::ppfmt::STALL_BODY_WORDS,
              .data_bits = {{0, 32}, {0, 0}}}},
        .timestamps_as_distances = true,
        .published_field = 1,
        .metadata = std::vector<std::byte>(sizeof(uint64_t)),
        .strings = std::move(strings)};
}

struct Registry {
    Registry() { by_id[TT_ZONE_STALL_ID].store(&sites.emplace_back(make_stall_site()), std::memory_order_release); }

    std::mutex mutex;
    uint32_t next_id = 0;
    std::deque<CallSite> sites;  // a record may hold its site's address for the life of the process
    std::array<std::atomic<const CallSite*>, TT_ZONE_ID_COUNT> by_id{};
    TypeLayouts types;
};

Registry& call_site_registry() {
    static ttsl::Indestructible<Registry> instance;
    return instance.get();
}

struct Section {
    Section(const ll_api::ElfFile& elf, std::string_view name) { bytes = elf.GetSectionContents(name, vma); }

    template <typename T>
    std::vector<T> read_table(uint64_t address, uint64_t count) const {
        std::vector<T> out(count);
        if (count != 0) {
            TT_FATAL(
                address >= vma && address - vma + count * sizeof(T) <= bytes.size(),
                "streaming profiler: a call site's table lies outside its section");
            std::memcpy(out.data(), bytes.data() + (address - vma), count * sizeof(T));
        }
        return out;
    }

    uint64_t vma = 0;
    std::span<const std::byte> bytes;
};

// Each field of struct `type`, of `struct_size` bytes, from its samples at `samples_at`, one per field with only that
// field set. A field fits its type's width, one as wide as its type starts on a byte, and no two fields share a bit.
std::vector<FieldBits> read_field_bits(
    const Section& data,
    uint64_t samples_at,
    uint32_t struct_size,
    std::span<const uint16_t> codes,
    std::string_view type) {
    const std::vector<std::byte> samples = data.read_table<std::byte>(samples_at, uint64_t{struct_size} * codes.size());
    std::vector<FieldBits> bits;
    for (size_t field = 0; field < codes.size(); field++) {
        const FieldBits found = find_set_bits(std::span(samples).subspan(field * struct_size, struct_size));
        const uint32_t type_bits = 8 * spd::code_bytes(codes[field]);
        TT_FATAL(
            found.width < type_bits || (found.width == type_bits && found.bit % 8 == 0),
            "streaming profiler: field {} of {} takes bits {} to {} of its sample, which its {}-bit type can't",
            field,
            type,
            found.bit,
            found.bit + found.width,
            type_bits);
        bits.push_back(found);
    }
    std::vector<FieldBits> by_bit = bits;
    std::ranges::sort(by_bit, {}, &FieldBits::bit);
    for (size_t k = 1; k < by_bit.size(); k++) {
        TT_FATAL(
            by_bit[k - 1].bit + by_bit[k - 1].width <= by_bit[k].bit,
            "streaming profiler: two fields of {} share bit {} in their samples",
            type,
            by_bit[k].bit);
    }
    return bits;
}

// An image's Strings by hash. Each translation unit has its own entry for a text it uses, so a text repeats; two
// different texts with one hash fail the load.
void read_strings(CallSiteStrings& strings, const Section& table, const std::string& elf_path) {
    for (const spd::RecordString& entry :
         table.read_table<spd::RecordString>(table.vma, table.bytes.size() / sizeof(spd::RecordString))) {
        const uint64_t offset = uint64_t{entry.text} - strings.vma;
        TT_FATAL(
            offset < strings.text.size() && entry.size < strings.text.size() - offset,
            "streaming profiler: '{}' has a string outside its .tt_record_data",
            elf_path);
        const std::string_view text(strings.text.data() + offset, entry.size);
        const auto [held, added] = strings.by_hash.try_emplace(entry.hash, text);
        TT_FATAL(
            added || held->second == text,
            "streaming profiler: '{}' has two strings with one hash, \"{}\" and \"{}\"; change either text",
            elf_path,
            held->second,
            text);
    }
}

// Every String in a call site's metadata has its text in the image.
void check_metadata_strings(const CallSite& site, const std::string& elf_path) {
    for (size_t field = 0; field < site.metadata_codes.size(); field++) {
        if (site.metadata_codes[field] != spd::kStringCode) {
            continue;
        }
        uint64_t hash = 0;
        std::memcpy(&hash, site.metadata.data() + site.metadata_bits[field].bit / 8, sizeof(hash));
        TT_FATAL(
            site.strings->by_hash.contains(hash),
            "streaming profiler: '{}': {}:{} names a string that isn't in the image's .tt_record_strings",
            elf_path,
            site.location.file,
            site.location.line);
    }
}

// A form sends a Timestamp's distance in the copy that starts where the Timestamp does.
FieldBits find_in_payload(std::span<const spd::RecordCopy> copies, uint32_t value, FieldBits field, bool as_distance) {
    for (const spd::RecordCopy& copy : copies) {
        const bool holds =
            as_distance ? 8u * copy.from == field.bit
                        : 8u * copy.from <= field.bit && field.bit + field.width <= 8u * (copy.from + copy.bytes);
        if (copy.value == value && holds) {
            return {field.bit - 8u * copy.from + 8u * copy.to, as_distance ? 8u * copy.bytes : field.width};
        }
    }
    return {};
}

// A data field: the value holding it, and its bits in that value.
struct ValueField {
    uint32_t value;
    FieldBits bits;
};

// Adds the call site's values, and returns each data field where its value holds it, an array value's fields once per
// element.
std::vector<ValueField> read_values(CallSite& site, std::span<const spd::RecordValue> values, const Section& data) {
    std::vector<ValueField> fields;
    uint32_t first_field = 0;
    for (const spd::RecordValue& value : values) {
        const auto codes = std::span(site.data_codes).subspan(first_field, value.field_count);
        std::vector<FieldBits> bits =
            value.samples != 0
                ? read_field_bits(data, value.samples, value.size, codes, site.strings->string_at(value.type))
                : std::vector{FieldBits{0, 8 * spd::code_bytes(codes[0])}};
        for (uint32_t element = 0; element < value.count; element++) {
            for (const FieldBits& field : bits) {
                fields.push_back({uint32_t(site.values.size()), {field.bit + 8 * element * value.size, field.width}});
            }
        }
        site.values.push_back(
            {.type = site.strings->string_at(value.type),
             .first_field = first_field,
             .is_array = value.is_array != 0,
             .field_bits = std::move(bits)});
        site.value_fields.push_back(value.field_count * value.count);
        first_field += value.field_count * value.count;
    }
    return fields;
}

// Each data field's bits in a form's payload, which the form's copies give. A struct travels as its bytes up to where
// its field types alone would end it, which misses a field that alignas moved past that end. The field a record is
// published into isn't sent.
std::vector<FieldBits> find_payload_bits(
    const CallSite& site, std::span<const ValueField> fields, std::span<const spd::RecordCopy> copies) {
    std::vector<FieldBits> bits;
    for (uint32_t field = 0; field < fields.size(); field++) {
        const auto [value, bits_in_value] = fields[field];
        const bool as_distance = site.timestamps_as_distances && site.data_codes[field] == spd::kTimestampCode;
        bits.push_back(find_in_payload(copies, value, bits_in_value, as_distance));
        TT_FATAL(
            bits.back().width != 0 || field == site.published_field,
            "streaming profiler: {}:{}: field {} of {} lies past the bytes a record carries because of an alignas; "
            "remove the alignas or move the field earlier",
            site.location.file,
            site.location.line,
            (field - site.values[value].first_field) % site.values[value].field_bits.size(),
            site.values[value].type.empty() ? "a value" : site.values[value].type);
    }
    return bits;
}

std::vector<std::pair<uint32_t, CallSite>> read_call_sites(const std::string& elf_path, const ll_api::ElfFile& elf) {
    const Section meta(elf, ".tt_record_meta"), data(elf, ".tt_record_data"), string_table(elf, ".tt_record_strings");
    auto image_strings = std::make_shared<CallSiteStrings>(
        CallSiteStrings{.text = data.read_table<char>(data.vma, data.bytes.size()), .vma = data.vma});
    read_strings(*image_strings, string_table, elf_path);
    const std::shared_ptr<const CallSiteStrings> strings = std::move(image_strings);
    std::vector<std::pair<uint32_t, CallSite>> sites;
    for (const spd::RecordSiteEntry& entry :
         meta.read_table<spd::RecordSiteEntry>(meta.vma, meta.bytes.size() / sizeof(spd::RecordSiteEntry))) {
        TT_FATAL(
            strings->by_hash.contains(entry.file),
            "streaming profiler: '{}': a call site's file name isn't in the image's .tt_record_strings",
            elf_path);
        CallSite site{
            .location =
                {.file = strings->string_of(entry.file),
                 .function = strings->string_at(entry.function),
                 .line = entry.line},
            .metadata_type = strings->string_at(entry.metadata_type),
            .data_codes = data.read_table<uint16_t>(entry.data_codes, entry.data_field_count),
            .metadata_codes = data.read_table<uint16_t>(entry.metadata_codes, entry.metadata_field_count),
            .published_field = entry.data_published_field,
            .metadata = data.read_table<std::byte>(entry.metadata, entry.metadata_size),
            .strings = strings};
        site.metadata_bits =
            read_field_bits(data, entry.metadata_samples, entry.metadata_size, site.metadata_codes, site.metadata_type);
        check_metadata_strings(site, elf_path);
        const std::vector<ValueField> fields =
            read_values(site, data.read_table<spd::RecordValue>(entry.data_values, entry.data_value_count), data);
        // Only data whose Timestamps travel as distances has a full form.
        const auto forms = data.read_table<spd::RecordForm>(entry.data_forms, entry.data_form_count);
        site.timestamps_as_distances =
            std::ranges::find(forms, spd::TimeForm::Full, &spd::RecordForm::time_form) != forms.end();
        for (const spd::RecordForm& form : forms) {
            site.forms.push_back(
                {.time_form = form.time_form,
                 .body_words = form.body_words,
                 .data_bits =
                     find_payload_bits(site, fields, data.read_table<spd::RecordCopy>(form.copies, form.copy_count))});
        }
        sites.emplace_back(entry.id, std::move(site));
    }
    return sites;
}

}  // namespace

FieldBits find_set_bits(std::span<const std::byte> sample) {
    std::optional<uint32_t> lowest;
    uint32_t highest = 0;
    for (size_t byte = 0; byte < sample.size(); byte++) {
        const auto set = std::to_integer<uint8_t>(sample[byte]);
        if (set != 0) {
            lowest = lowest.value_or(uint32_t(8 * byte) + std::countr_zero(set));
            highest = uint32_t(8 * byte) + 7 - std::countl_zero(set);
        }
    }
    return lowest ? FieldBits{*lowest, highest + 1 - *lowest} : FieldBits{};
}

void load_call_sites(const std::string& elf_path, ll_api::ElfFile& elf) {
    uint64_t ids_vma = 0;
    const size_t count = elf.GetSectionContents(".tt_record_ids", ids_vma).size();
    if (count == 0) {
        return;
    }

    Registry& registry = call_site_registry();
    uint32_t base = 0;
    {
        const std::lock_guard lock(registry.mutex);
        TT_FATAL(
            uint64_t{registry.next_id} + count <= TT_ZONE_STALL_ID,
            "streaming profiler: '{}' has {} call sites, and only {} of the process's {} call-site ids remain",
            elf_path,
            count,
            TT_ZONE_STALL_ID - registry.next_id,
            TT_ZONE_STALL_ID);
        base = std::exchange(registry.next_id, registry.next_id + static_cast<uint32_t>(count));
    }
    elf.RebaseCallSiteIds(base);

    std::vector<std::pair<uint32_t, CallSite>> sites = read_call_sites(elf_path, elf);
    const std::lock_guard lock(registry.mutex);
    for (auto& [id, site] : sites) {
        registry.types.add_device_type(site.metadata_type, site.metadata_codes, site.metadata_bits);
        for (const CallSite::Value& value : site.values) {
            if (!value.type.empty()) {
                registry.types.add_device_type(
                    value.type,
                    std::span(site.data_codes).subspan(value.first_field, value.field_bits.size()),
                    value.field_bits);
            }
        }
        registry.by_id[id].store(&registry.sites.emplace_back(std::move(site)), std::memory_order_release);
    }
    log_debug(tt::LogLLRuntime, "streaming profiler: '{}' -> call site ids [{}, {})", elf_path, base, base + count);
}

const CallSite* find_call_site(uint32_t id) { return call_site_registry().by_id[id].load(std::memory_order_acquire); }

void add_host_type(std::string_view name, std::span<const uint16_t> codes) {
    Registry& registry = call_site_registry();
    const std::lock_guard lock(registry.mutex);
    registry.types.add_host_type(name, codes);
}

}  // namespace tt::llrt
