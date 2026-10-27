// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sparse_kv_push_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

#include <tt-metalium/global_semaphore.hpp>

namespace ttnn::operations::experimental::sparse_kv_push {

// Sparse KV push: instead of a full-mesh gather of the whole cache onto every chip, each owner pushes only the rows a
// requester's top-k references, packed into the requester's compact buffer at the slots the plan op rewrote its
// indices to (see sparse_push_plan_device_operation.hpp and kernels/sparse_push_table_reader.cpp). sparse_sdpa then
// reads that buffer through the rewritten indices.
struct SparseKvPushProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
    };

    using cached_mesh_workload_t = ttnn::device_operation::AdaptedCachedMeshWorkload<shared_variables_t>;

    static cached_mesh_workload_t create_mesh_workload(
        const SparseKvPushParams& operation_attributes,
        const ttnn::MeshCoordinateRangeSet& tensor_coords,
        const SparseKvPushInputs& tensor_args,
        Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_mesh_workload_t& cached_workload,
        const SparseKvPushParams& operation_attributes,
        const SparseKvPushInputs& tensor_args,
        Tensor& output_tensor);
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
