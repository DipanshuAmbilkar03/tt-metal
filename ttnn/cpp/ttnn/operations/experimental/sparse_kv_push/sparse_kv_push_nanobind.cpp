// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sparse_kv_push_nanobind.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include "sparse_kv_push.hpp"
#include "ttnn-nanobind/bind_function.hpp"

namespace ttnn::operations::experimental::sparse_kv_push::detail {

void bind_experimental_sparse_kv_push_operations(nb::module_& mod) {
    ttnn::bind_function<"sparse_kv_push", "ttnn.experimental.">(
        mod,
        R"doc(
            Each owner chip pushes only the cache rows a requester's top-k references into that requester's
            compact ``output_tensor``, at the slots ``sparse_push_plan`` rewrote its indices to.
            Needs a FABRIC_2D_TORUS_XY fabric.

            Args:
                input_tensor: The KVPE cache, sharded over its rows (dim 2) across every chip.
                output_tensor: The requester's compact buffer, [1, 1, rows, width] DRAM interleaved; it may be
                    taller than the gathered slot (each owner's stream is padded to whole bank-lane units).
                table: The all-gathered ``sparse_push_plan`` tables of every chip
                    (uint32 [1, 1, pages * devices, 1024]).
                stripe_chunk: Rows per block-cyclic block of the cache the plan was built for.
                max_rows: Row capacity the plan tables were built for.
                ready_semaphore, data_valid_semaphore: Caller-owned persistent semaphores.
                part: 0 pushes every needed row; 2 only the rows from the plan's chunk start on (the prefix rows
                    were pushed earlier by a ``sparse_sdpa`` with a fused push).
                num_links: Fabric links per direction (1 or 2).
                batch_index: Cache slot pushed.
                batch_index_tensor: Trace-safe slot: a 1-element uint32 DRAM tensor holding the user id; the slot is
                    read on device as ``batch_index_tensor[0] * batch_slot_num_layers + batch_slot_layer_idx``
                    and ``batch_index`` is ignored.
                core_grid: Cores for the push's senders and fabric muxes (default: the first sub-device's cores).
        )doc",
        &sparse_kv_push,
        nb::arg("input_tensor").noconvert(),
        nb::kw_only(),
        nb::arg("output_tensor").noconvert(),
        nb::arg("table").noconvert(),
        nb::arg("stripe_chunk"),
        nb::arg("max_rows"),
        nb::arg("ready_semaphore"),
        nb::arg("data_valid_semaphore"),
        nb::arg("part") = 0,
        nb::arg("num_links") = 2,
        nb::arg("batch_index") = 0,
        nb::arg("batch_index_tensor") = nb::none(),
        nb::arg("batch_slot_num_layers") = 1,
        nb::arg("batch_slot_layer_idx") = 0,
        nb::arg("core_grid") = nb::none());

    ttnn::bind_function<"sparse_push_plan", "ttnn.experimental.">(
        mod,
        R"doc(
            On-device plan for the sparse KV push, from one requester's top-k indices (per device uint32
            row-major [1, 1, rows, topk], 0xFFFFFFFF = masked). Returns ``[table, compact_indices]``: the plan table
            (uint32 [1, 1, pages, 1024]) to all-gather for the owners, and the indices rewritten to slots in the
            requester's compact buffer.
        )doc",
        &sparse_push_plan,
        nb::arg("indices").noconvert(),
        nb::kw_only(),
        nb::arg("num_devices"),
        nb::arg("stripe_chunk"),
        nb::arg("max_rows"),
        nb::arg("pages_per_packet"),
        nb::arg("banks"),
        nb::arg("chunk_start") = nb::none());
}

}  // namespace ttnn::operations::experimental::sparse_kv_push::detail
