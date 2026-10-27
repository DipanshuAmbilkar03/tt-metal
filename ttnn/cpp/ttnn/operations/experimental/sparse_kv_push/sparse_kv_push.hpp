// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

// Each owner chip pushes the rows of `input_tensor` (the KVPE cache, sharded over dim 2 across a 2D torus) that a
// requester's top-k references into that requester's compact `output_tensor`, as described by `table`: the
// all-gathered sparse_push_plan tables of a cache of max_rows rows (stripe_chunk rows per block-cyclic block).
// part: 0 = every needed row; 2 = only the rows from the chunk start on (the prefix was pushed by a fused sparse_sdpa).
Tensor sparse_kv_push(
    const Tensor& input_tensor,
    const Tensor& output_tensor,
    const Tensor& table,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    const GlobalSemaphore& ready_semaphore,
    const GlobalSemaphore& data_valid_semaphore,
    uint32_t part = 0,
    uint32_t num_links = 2,
    uint32_t batch_index = 0,
    // Trace-safe slot: batch_index_tensor[0] * batch_slot_num_layers + batch_slot_layer_idx, read on device
    // (batch_index is then ignored).
    const std::optional<Tensor>& batch_index_tensor = std::nullopt,
    uint32_t batch_slot_num_layers = 1,
    uint32_t batch_slot_layer_idx = 0,
    const std::optional<CoreRangeSet>& core_grid = std::nullopt);

// On-device plan for the sparse KV push: returns {plan table, indices rewritten to compact slots}
// (see device/sparse_push_plan_device_operation.hpp).
std::vector<Tensor> sparse_push_plan(
    const Tensor& indices,
    uint32_t num_devices,
    uint32_t stripe_chunk,
    uint32_t max_rows,
    uint32_t pages_per_packet,
    uint32_t banks,
    const std::optional<Tensor>& chunk_start = std::nullopt);

}  // namespace ttnn::operations::experimental::sparse_kv_push

namespace ttnn::experimental {
using operations::experimental::sparse_kv_push::sparse_kv_push;
using operations::experimental::sparse_kv_push::sparse_push_plan;
}  // namespace ttnn::experimental
