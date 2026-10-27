// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental::sparse_kv_push {

struct SparseKvPushParams {
    // The plan table's geometry and which rows to push. Structural.
    uint32_t stripe_chunk = 0;
    uint32_t max_rows = 0;
    uint32_t part = 0;  // 0 every needed row, 2 only the rows from the chunk start on
    uint32_t num_links = 2;
    CoreRangeSet core_grid;
    tt::tt_fabric::FabricConfig fabric_config = tt::tt_fabric::FabricConfig::DISABLED;

    // Runtime resources, re-patched on every dispatch.
    std::optional<GlobalSemaphore> ready_semaphore;
    std::optional<GlobalSemaphore> data_valid_semaphore;

    // Cache slot pushed: batch_index, or with batch_index_tensor (trace-safe)
    // batch_index_tensor[0] * batch_slot_num_layers + batch_slot_layer_idx read on device. Runtime args, not hashed.
    uint32_t batch_index = 0;
    uint32_t batch_slot_num_layers = 1;
    uint32_t batch_slot_layer_idx = 0;
};

struct SparseKvPushInputs {
    Tensor input_tensor;   // the KVPE cache, sharded over its rows (dim 2) across the whole mesh
    Tensor output_tensor;  // the requester's compact buffer
    Tensor table;          // the all-gathered sparse_push_plan tables (uint32 [1, 1, pages * devices, 1024])
    std::optional<Tensor> batch_index_tensor;
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
