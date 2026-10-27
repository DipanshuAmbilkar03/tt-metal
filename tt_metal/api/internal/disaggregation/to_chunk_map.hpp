// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric_types.hpp>
#include <tt-metalium/experimental/distributed_tensor/topology/tensor_topology.hpp>

#include <internal/disaggregation/kv_chunk_address_table.hpp>
#include <internal/disaggregation/cache_tensor_layout_spec.hpp>

namespace tt::tt_metal::internal::disaggregation {

// Runtime extents the chunk map is enumerated over. Mirrors the device-free reference factory's
// `Geometry` (kv_manager/tests/kv_layout_spec_smoke/kv_layout_spec.py): the (layer, slot, position)
// grid. The DRAM base address is per-cache (CacheConfig::base_addr); the fabric mesh is a call arg.
struct MapGeometry {
    uint32_t num_layers = 0;
    uint32_t num_slots = 0;
    uint32_t max_seq_len = 0;         // in tokens
    uint32_t position_step = tt::constants::TILE_WIDTH;  // token stride between enumerated positions
};

// One co-resident cache to address: the residence-agnostic spec, the mesh distribution the tensor was
// allocated with (TensorTopology), the op/engine generation policy, and the allocated buffer address.
// The addresser reads shape/dtype/shard-spec off `spec.tensor`, the seq/head shard axes + device coords off
// `topology`, the DRAM bank count + bank order off `spec.arch` (num_dram_banks / optimal_bank_order),
// `mesh_id` from the call args, and everything else from `policy`.
//
// A tensor whose batch axis (axis 0) holds num_slots * num_layers rows folds the layers into it,
// user-major: row = slot * num_layers + layer (the prefill write layout). Otherwise every layer of the
// cache resolves to the same rows.
struct CacheConfig {
    KvLayoutSpec spec;
    TensorTopology topology;
    GenerationPolicy policy;
    uint64_t base_addr = 0;
    // The table config's name — the key the KV manager pairs a source and a destination config by.
    // Empty names the config by its decimal index; names must be all empty or all set.
    std::string name;
    // The layers this cache publishes, ascending; empty publishes every layer. A layer left out keeps its
    // addresses (they are arithmetic in the layer) but gets no table rows, for a cache allocated on every
    // layer that only some layers write.
    std::vector<uint32_t> layers;
};

// The commonized factory. `configs` is a model expressed as an ordered LIST of co-resident caches
// (K/V + indexer, per layer-type). ONE dispatch over bank_scheme x temporal x distribution fills a
// KvChunkAddressTable: each (config, layer, slot, position[, head]) resolves to a KvCacheLocation whose
// noc_addr encodes (bank_id << 32) | per_bank_offset and whose device_group_index references the replica
// set of devices the topology implies, as chip ids linearized row-major over `mesh_shape`.
//
// One config per cache; a per-head cache (BLOCK/CYCLIC GQA) fans its heads into ADDITIONAL slots so
// heads and slots share one flat slot axis: config slot = head * num_slots + slot.
KvChunkAddressTable to_chunk_map(
    const std::vector<CacheConfig>& configs,
    tt::tt_fabric::MeshId mesh_id,
    const tt::tt_metal::distributed::MeshShape& mesh_shape,
    const MapGeometry& geometry);

// Everything to_chunk_map reads: a model's caches and the mesh and geometry they are enumerated over. A
// model exports this so a consumer can derive the same table itself.
struct CacheLayoutModel {
    std::vector<CacheConfig> caches;
    tt::tt_fabric::MeshId mesh_id{0};
    tt::tt_metal::distributed::MeshShape mesh_shape{1};
    MapGeometry geometry;
};

KvChunkAddressTable to_chunk_map(const CacheLayoutModel& model);

}  // namespace tt::tt_metal::internal::disaggregation
