// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The types test_streaming_profiler_semantics's kernels publish and its callbacks read, and the values both sides fill
// them with. Compiled by the device compiler for the kernels and by the host compiler for the callbacks.
#pragma once

#include <cstdint>
#include <type_traits>

#include <hostdevcommon/streaming_profiler_reflection.hpp>
#include <hostdevcommon/streaming_profiler_types.hpp>

// Memberwise == for the host to compare a struct with the one it expects; kernels are C++17, which has no default ==.
#if defined(KERNEL_BUILD)
#define SEMANTICS_EQ(T)
#else
#define SEMANTICS_EQ(T) bool operator==(const T&) const = default;
#endif

namespace semantics_test {
using tt::tt_metal::experimental::streaming_profiler::String;
using tt::tt_metal::experimental::streaming_profiler::Timestamp;

// Every call's metadata names the call, and Other is a second metadata type with the same field.
struct Call {
    String name;
};
struct Other {
    String name;
};
// The integrity sweep's metadata, which names how a struct was sent.
struct Send {
    String name;
};

enum class Color : uint8_t { Red, Green, Blue };
struct PairA {
    uint32_t x, y;
    SEMANTICS_EQ(PairA)
};
struct PairB {
    uint32_t x, y;
    SEMANTICS_EQ(PairB)
};

// The layouts the integrity sweep sends, each a way a struct's fields can lie.
struct AllScalars {
    bool b;
    int8_t i8;
    uint8_t u8;
    char c;
    int16_t i16;
    uint16_t u16;
    int32_t i32;
    uint32_t u32;
    float f;
    int64_t i64;
    uint64_t u64;
    double d;
    SEMANTICS_EQ(AllScalars)
};
struct Descending {
    uint64_t a;
    uint32_t b;
    uint16_t c;
    uint8_t d;
    SEMANTICS_EQ(Descending)
};
struct Ascending {
    uint8_t a;
    uint16_t b;
    uint32_t c;
    uint64_t d;
    SEMANTICS_EQ(Ascending)
};
struct Bools {
    bool a, b, c;
    uint8_t d;
    bool e;
    SEMANTICS_EQ(Bools)
};
struct WordBits {
    uint32_t a : 3;
    uint32_t b : 13;
    uint32_t c : 16;
    SEMANTICS_EQ(WordBits)
};
struct CrossingBits {
    uint8_t a;
    uint16_t b : 12;
    uint16_t c : 4;
    uint32_t d : 20;
    SEMANTICS_EQ(CrossingBits)
};
struct SignedBits {
    int32_t a : 5;
    int32_t b : 11;
    int16_t c : 7;
    SEMANTICS_EQ(SignedBits)
};
struct __attribute__((packed)) Packed {
    uint8_t a;
    uint32_t b;
    uint16_t c;
    uint64_t d;
    SEMANTICS_EQ(Packed)
};
struct AlignedMember {
    uint8_t a;
    alignas(8) uint32_t b;
    uint8_t c;
    SEMANTICS_EQ(AlignedMember)
};
struct Inner {
    uint16_t x;
    uint8_t y;
    SEMANTICS_EQ(Inner)
};
struct Nested {
    Inner in;
    uint32_t z;
    Inner more[2];
    SEMANTICS_EQ(Nested)
};
struct Base {
    uint32_t a;
    SEMANTICS_EQ(Base)
};
struct Derived : Base {
    uint16_t b;
    uint8_t c;
    SEMANTICS_EQ(Derived)
};
struct Tagged {
    char tag[5];
    uint32_t v;
    SEMANTICS_EQ(Tagged)
};
struct Pairs {
    PairA pairs[3];
    SEMANTICS_EQ(Pairs)
};
struct WithEnum {
    Color color;
    uint16_t v;
    Color more[3];
    SEMANTICS_EQ(WithEnum)
};
struct Floats {
    float a;
    double b;
    float c;
    SEMANTICS_EQ(Floats)
};
// Long enough for word runs, and the longest a record can be.
struct Run {
    uint32_t words[40];
    SEMANTICS_EQ(Run)
};
struct Largest {
    uint32_t words[126];
    SEMANTICS_EQ(Largest)
};
// Timestamps whose distances back from the record take each form's width, beside other fields.
struct Stamps {
    Timestamp first;
    uint8_t x;
    Timestamp second;
    Timestamp third;
};
struct StampedRun {
    Timestamp at;
    uint32_t words[12];
};

// The catalog's structs in seed order: a record's seed starts from its struct's index here.
#define SEMANTICS_CATALOG                                                                                        \
    AllScalars, Descending, Ascending, Bools, WordBits, CrossingBits, SignedBits, Packed, AlignedMember, Nested, \
        Derived, Tagged, Pairs, WithEnum, Floats, Run, Largest, Stamps, StampedRun

// splitmix64, so a field's value takes every bit pattern across records.
constexpr uint64_t mix(uint64_t x) {
    x += 0x9e3779b97f4a7c15;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9;
    x = (x ^ (x >> 27)) * 0x94d049bb133111eb;
    return x ^ (x >> 31);
}
// A record's seed: which call sent it, from which processor, in which iteration.
constexpr uint64_t seed_of(uint32_t call, uint32_t processor, uint32_t iteration) {
    return mix(uint64_t{call} << 40 | uint64_t{processor} << 32 | iteration);
}
// Fills a struct's fields in declaration order, each from the seed and its index. A float or double takes an exact,
// finite value, so the host can compare it with ==.
struct FillSource {
    uint64_t seed;
    uint32_t field = 0;
    constexpr void open() {}
    constexpr void close() {}
    template <typename V>
    constexpr V leaf() {
        const uint64_t bits = mix(seed + field++);
        if constexpr (std::is_same_v<V, bool>) {
            return (bits & 1) != 0;
        } else if constexpr (std::is_same_v<V, float>) {
            return float(int32_t(bits >> 40) - (int32_t{1} << 23));
        } else if constexpr (std::is_same_v<V, double>) {
            return double(int64_t(bits >> 11) - (int64_t{1} << 52));
        } else if constexpr (std::is_enum_v<V>) {
            return V(std::underlying_type_t<V>(bits));
        } else {
            return V(bits);
        }
    }
};
template <typename T>
constexpr T filled(uint64_t seed) {
    FillSource source{seed};
    return tt::tt_metal::experimental::streaming_profiler::detail::build<T>(source);
}

// How far back from the record's first Timestamp its others lie in iteration i: within a short form's 16 bits, a medium
// form's 32, or past them.
constexpr uint64_t stamp_distance(uint32_t iteration, uint32_t which) {
    constexpr uint64_t kWidths[] = {uint64_t{1} << 12, uint64_t{1} << 28, uint64_t{1} << 34};
    return kWidths[iteration % 3] - 1000 * which;
}

}  // namespace semantics_test
