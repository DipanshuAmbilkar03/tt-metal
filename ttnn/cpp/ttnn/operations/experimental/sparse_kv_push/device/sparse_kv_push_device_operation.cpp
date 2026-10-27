// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push_device_operation.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/ccl_topology_utils.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

namespace {

// The cache is sharded over its rows across the whole mesh; the compact buffer holds rows too.
constexpr uint32_t kRowDim = 2;

void validate_semaphores(const SparseKvPushParams& args, const Tensor& input_tensor) {
    TT_FATAL(
        args.ready_semaphore.has_value() && args.data_valid_semaphore.has_value(),
        "sparse_kv_push needs caller-owned ready_semaphore / data_valid_semaphore");
    const auto& ready = *args.ready_semaphore;
    const auto& data_valid = *args.data_valid_semaphore;
    TT_FATAL(
        ready.address() != data_valid.address(), "sparse_kv_push ready_semaphore and data_valid_semaphore must differ");
    auto* mesh_device = input_tensor.device();
    TT_FATAL(
        ready.device() == mesh_device && data_valid.device() == mesh_device,
        "sparse_kv_push semaphores must be created on the input tensor's mesh device");
    const auto ready_buffer_type = std::get<1>(ready.attribute_values());
    const auto data_valid_buffer_type = std::get<1>(data_valid.attribute_values());
    const bool has_l1_small = mesh_device->allocator()->get_bank_size(BufferType::L1_SMALL) > 0;
    const auto expected_buffer_type = has_l1_small ? BufferType::L1_SMALL : BufferType::L1;
    TT_FATAL(
        ready_buffer_type == expected_buffer_type && data_valid_buffer_type == expected_buffer_type,
        "sparse_kv_push semaphores must use buffer type {}, got ready={} and data_valid={}",
        expected_buffer_type,
        ready_buffer_type,
        data_valid_buffer_type);
}

// Bounds of the runtime cache slot; the value itself is hash-excluded, so this runs on cache hits too.
void validate_batch_slot(const SparseKvPushParams& args, const SparseKvPushInputs& tensor_args) {
    const uint32_t cache_batch = tensor_args.input_tensor.padded_shape()[0];
    if (tensor_args.batch_index_tensor.has_value()) {
        TT_FATAL(
            args.batch_slot_layer_idx < args.batch_slot_num_layers,
            "sparse_kv_push batch_slot_layer_idx {} must be < batch_slot_num_layers {}",
            args.batch_slot_layer_idx,
            args.batch_slot_num_layers);
        TT_FATAL(
            args.batch_slot_num_layers <= cache_batch && cache_batch % args.batch_slot_num_layers == 0,
            "sparse_kv_push batch_slot_num_layers {} must divide the cache batch {}",
            args.batch_slot_num_layers,
            cache_batch);
    } else {
        TT_FATAL(
            args.batch_index < cache_batch,
            "sparse_kv_push batch_index {} must be < cache batch {}",
            args.batch_index,
            cache_batch);
    }
}

void validate_batch_index_tensor(const Tensor& meta) {
    TT_FATAL(meta.storage_type() == StorageType::DEVICE, "sparse_kv_push batch_index_tensor must be on device");
    // Its TensorAccessorArgs are compile-time args while only its presence is hashed, so pin the placement.
    TT_FATAL(
        meta.dtype() == DataType::UINT32 && meta.layout() == Layout::ROW_MAJOR && meta.logical_volume() == 1 &&
            meta.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            meta.memory_config().buffer_type() == BufferType::DRAM,
        "sparse_kv_push batch_index_tensor must be a 1-element UINT32 ROW_MAJOR interleaved DRAM tensor");
}

}  // namespace

ttsl::hash::hash_t SparseKvPushDeviceOperation::compute_program_hash(
    const SparseKvPushParams& args, const SparseKvPushInputs& tensor_args) {
    // Routing is resolved from the fabric node ids, so they are part of the key.
    auto* device = tensor_args.input_tensor.device();
    std::vector<tt::tt_fabric::FabricNodeId> fabric_nodes;
    fabric_nodes.reserve(device->shape().mesh_size());
    for (const auto& coord : tt::tt_metal::distributed::MeshCoordinateRange(device->shape())) {
        fabric_nodes.push_back(device->get_fabric_node_id(coord));
    }
    return tt::tt_metal::operation::hash_operation<SparseKvPushDeviceOperation>(
        args.stripe_chunk,
        args.max_rows,
        args.part,
        args.num_links,
        args.core_grid,
        args.fabric_config,
        device->shape(),
        fabric_nodes,
        tensor_args);
}

void SparseKvPushDeviceOperation::validate_on_program_cache_miss(
    const SparseKvPushParams& args, const SparseKvPushInputs& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& output_tensor = tensor_args.output_tensor;
    const auto& table = tensor_args.table;
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE && input_tensor.buffer() != nullptr,
        "sparse_kv_push input must be allocated on device");
    TT_FATAL(
        output_tensor.storage_type() == StorageType::DEVICE && output_tensor.buffer() != nullptr,
        "sparse_kv_push output must be allocated on device");
    auto* mesh_device = input_tensor.device();
    TT_FATAL(output_tensor.device() == mesh_device, "sparse_kv_push input and output must be on the same mesh device");
    const auto mesh_shape = mesh_device->shape();
    const uint32_t num_devices = mesh_shape.mesh_size();
    TT_FATAL(mesh_shape.dims() == 2, "sparse_kv_push needs a 2D device mesh, got {}", mesh_shape);
    TT_FATAL(
        args.fabric_config == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY,
        "sparse_kv_push needs a 2D torus fabric (FABRIC_2D_TORUS_XY), got {}",
        args.fabric_config);
    validate_semaphores(args, input_tensor);

    TT_FATAL(
        args.stripe_chunk > 0 && args.max_rows > 0, "sparse_kv_push needs stripe_chunk and max_rows greater than 0");
    TT_FATAL(args.part == 0 || args.part == 2, "sparse_kv_push part must be 0 (all rows) or 2 (remainder)");
    TT_FATAL(
        table.dtype() == DataType::UINT32 && table.layout() == Layout::ROW_MAJOR &&
            table.memory_config().buffer_type() == BufferType::DRAM,
        "sparse_kv_push table must be a UINT32 ROW_MAJOR DRAM tensor");

    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output_tensor.padded_shape();
    TT_FATAL(input_shape.rank() == 4, "sparse_kv_push input must be rank 4, got {}", input_shape);
    TT_FATAL(input_tensor.buffer()->is_dram(), "sparse_kv_push requires a DRAM input");
    TT_FATAL(
        ttnn::operations::ccl::common::has_row_major_mesh_coordinates(input_tensor),
        "sparse_kv_push requires row-major tensor mesh coordinates");
    TT_FATAL(
        ttnn::operations::ccl::common::tensor_dim_shard_factor(input_tensor, kRowDim) == num_devices,
        "sparse_kv_push requires the input to be sharded over dim {} across all {} devices",
        kRowDim,
        num_devices);
    if (tensor_args.batch_index_tensor.has_value()) {
        validate_batch_index_tensor(*tensor_args.batch_index_tensor);
    }
    validate_batch_slot(args, tensor_args);

    TT_FATAL(
        output_tensor.layout() == input_tensor.layout() && output_tensor.dtype() == input_tensor.dtype() &&
            output_tensor.tensor_spec().page_config() == input_tensor.tensor_spec().page_config(),
        "sparse_kv_push output layout, dtype and page config must match the input");
    TT_FATAL(
        output_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            output_tensor.memory_config().buffer_type() == BufferType::DRAM,
        "sparse_kv_push requires an interleaved DRAM output");
    // No row-count floor: slots stop at the requester's distinct rows plus per-owner padding, which the caller sizes
    // for (SparseKvPushPlanner.buffer_rows), so the buffer may be far shorter than the gathered slot.
    TT_FATAL(
        output_shape.rank() == 4 && output_shape[0] == 1 && output_shape[1] == input_shape[1] &&
            output_shape[3] == input_shape[3],
        "sparse_kv_push output must be [1, {}, rows, {}], got {}",
        input_shape[1],
        input_shape[3],
        output_shape);
}

void SparseKvPushDeviceOperation::validate_on_program_cache_hit(
    const SparseKvPushParams& args, const SparseKvPushInputs& tensor_args) {
    TT_FATAL(
        ttnn::operations::ccl::common::has_row_major_mesh_coordinates(tensor_args.input_tensor),
        "sparse_kv_push requires row-major tensor mesh coordinates");
    TT_FATAL(tensor_args.output_tensor.buffer() != nullptr, "sparse_kv_push output must be allocated on device");
    validate_semaphores(args, tensor_args.input_tensor);
    validate_batch_slot(args, tensor_args);
}

SparseKvPushDeviceOperation::spec_return_value_t SparseKvPushDeviceOperation::compute_output_specs(
    const SparseKvPushParams&, const SparseKvPushInputs& tensor_args) {
    return tensor_args.output_tensor.tensor_spec();
}

SparseKvPushDeviceOperation::topology_return_value_t SparseKvPushDeviceOperation::compute_output_topologies(
    const SparseKvPushParams&, const SparseKvPushInputs& tensor_args) {
    // Like a full-mesh gather of the rows: every placement sharding them becomes replicated.
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& input_topology = input_tensor.tensor_topology();
    auto placements = input_topology.placements();
    const uint32_t rank = input_tensor.logical_shape().rank();
    for (auto& placement : placements) {
        if (ttnn::operations::ccl::common::placement_shards_tensor_dim(placement, kRowDim, rank)) {
            placement = tt::tt_metal::distributed::MeshMapperConfig::Replicate{};
        }
    }
    return {tt::tt_metal::TensorTopology(
        input_topology.distribution_shape(), std::move(placements), input_topology.mesh_coords())};
}

SparseKvPushDeviceOperation::tensor_return_value_t SparseKvPushDeviceOperation::create_output_tensors(
    const SparseKvPushParams&, const SparseKvPushInputs& tensor_args) {
    return tensor_args.output_tensor;
}

SparseKvPushDeviceOperation::program_factory_t SparseKvPushDeviceOperation::select_program_factory(
    const SparseKvPushParams&, const SparseKvPushInputs&) {
    return SparseKvPushProgramFactory{};
}

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::prim {

Tensor sparse_kv_push(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& table,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    const GlobalSemaphore& ready_semaphore,
    const GlobalSemaphore& data_valid_semaphore,
    uint32_t part,
    uint32_t num_links,
    uint32_t batch_index,
    const std::optional<Tensor>& batch_index_tensor,
    uint32_t batch_slot_num_layers,
    uint32_t batch_slot_layer_idx,
    const std::optional<CoreRangeSet>& core_grid) {
    using namespace ttnn::operations::experimental::sparse_kv_push;
    auto* mesh_device = input_tensor.device();
    TT_FATAL(mesh_device != nullptr, "sparse_kv_push input must be on device");
    // Default: every worker core of the first sub-device.
    const auto grid = core_grid.value_or(mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, mesh_device->get_sub_device_ids().at(0)));
    return ttnn::device_operation::launch<SparseKvPushDeviceOperation>(
        SparseKvPushParams{
            .stripe_chunk = stripe_chunk,
            .max_rows = max_rows,
            .part = part,
            .num_links = num_links,
            .core_grid = grid,
            .fabric_config = tt::tt_fabric::GetFabricConfig(),
            .ready_semaphore = ready_semaphore,
            .data_valid_semaphore = data_valid_semaphore,
            .batch_index = batch_index,
            .batch_slot_num_layers = batch_slot_num_layers,
            .batch_slot_layer_idx = batch_slot_layer_idx},
        SparseKvPushInputs{
            .input_tensor = input_tensor,
            .output_tensor = output_tensor,
            .table = table,
            .batch_index_tensor = batch_index_tensor});
}

}  // namespace ttnn::prim
