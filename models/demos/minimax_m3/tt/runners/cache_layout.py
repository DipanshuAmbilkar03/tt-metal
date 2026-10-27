# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""The MiniMax-M3 prefill caches declared as a ``CacheLayoutModel`` (the cache tensor layout spec).

Each cache is a ``CacheConfig``: the tensor's ``KvLayoutSpec`` (its logical shape, dtype and DRAM ND-shard
spec, and how its sequence is retained), the mesh distribution its chunks really have, and the writer's
``GenerationPolicy``. ``ttnn.experimental.disaggregation.to_chunk_map`` derives the KV chunk address table
from the model, so ``kv_chunk_table.py`` does not have to replay the writer's bank walk itself.

The allocated tensors carry a 1-D Replicate topology (which head or sequence block a chip holds is decided
at write time), so the distribution is declared here instead:

    configs 00..N-1   k head h:   the sequence sharded over the SP rows of TP column h
    configs N..2N-1   v head h:   likewise
    config  2N        index_k:    the sequence sharded over the SP rows, replicated over the TP columns

Every cache is ``[num_users * num_layers, 1, seq_len, head_dim]``, user-major, so ``to_chunk_map`` folds
the layers into its batch axis. The writer's layout is the Natural scheme with the Identity bank order:
32-token blocks round-robin over the banks, and SP row r owning ``[n*chunk + r*chunk/sp, +chunk/sp)`` of
each prefill chunk n (chunk-modulo ownership with ``device_chunk_size = chunk/sp``).
"""

import ttnn
from models.demos.minimax_m3.tt.attention.kv_cache import NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK, cache_memory_config

disaggregation = ttnn.experimental.disaggregation

# The model is written next to the table it derives: ``<table path><LAYOUT_MODEL_SUFFIX>``.
LAYOUT_MODEL_SUFFIX = ".cache_layout.pb"


def _spec(tensor, *, num_users, num_layers, seq_len, head_dim, arch):
    shape = ttnn.Shape([num_users * num_layers, 1, seq_len, head_dim])
    tensor_spec = ttnn.TensorSpec(shape, tensor.dtype, ttnn.TILE_LAYOUT, cache_memory_config(head_dim))
    return disaggregation.KvLayoutSpec(tensor_spec, disaggregation.TemporalDense(), arch)


def _head_topology(sp, head):
    """K/V head ``head``: the sequence sharded over the SP rows of TP column ``head``."""
    coords = [ttnn.MeshCoordinate(row, head) for row in range(sp)]
    return ttnn.TensorTopology(ttnn.MeshShape([sp]), [ttnn.PlacementShard(2)], coords)


def _replicated_topology(sp, cols):
    """index_k: the sequence sharded over the SP rows, replicated over the TP columns."""
    shape = ttnn.MeshShape(sp, cols)
    coords = [ttnn.MeshCoordinate(row, col) for row in range(sp) for col in range(cols)]
    return ttnn.TensorTopology(shape, [ttnn.PlacementShard(2), ttnn.PlacementReplicate()], coords)


def m3_cache_layout_model(
    *,
    kv_cache,
    base_addrs,
    mesh_id,
    mesh_shape,
    arch,
    num_users,
    num_layers,
    seq_len,
    chunk_size,
    num_kv_heads,
    head_dim,
    index_k_layers=None,
):
    """The ``CacheLayoutModel`` of ``kv_cache`` on one mesh with the sequence on its rows (``sp_axis == 0``)
    and index_k TP-replicated. ``base_addrs`` is the (k, v, index_k) buffer address, identical on every
    chip; ``index_k_layers`` limits index_k's rows to those layers (the MSA layers), as the decode peer
    holds index_k only there. Config names are the zero-padded positions the decode table uses."""
    from models.demos.minimax_m3.tt.runners.kv_chunk_table import config_name

    sp, cols = mesh_shape
    assert num_kv_heads == cols, f"num_kv_heads ({num_kv_heads}) must equal the TP column count ({cols})"
    policy = disaggregation.GenerationPolicy(
        bank_scheme=disaggregation.BankScheme.Natural,
        bank_order=disaggregation.BankOrder.Identity,
        # Natural/Identity pages the sequence at k_chunk_size tokens: one DRAM block.
        k_chunk_size=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
        device_chunk_size=chunk_size // sp,
    )
    shape_args = dict(num_users=num_users, num_layers=num_layers, seq_len=seq_len, head_dim=head_dim, arch=arch)
    k_base, v_base, index_k_base = base_addrs
    num_configs = 2 * num_kv_heads + 1

    caches = []
    for tensor, base in ((kv_cache.k, k_base), (kv_cache.v, v_base)):
        spec = _spec(tensor, **shape_args)
        for head in range(num_kv_heads):
            caches.append(
                disaggregation.CacheConfig(
                    spec, _head_topology(sp, head), policy, base, config_name(len(caches), num_configs)
                )
            )
    caches.append(
        disaggregation.CacheConfig(
            _spec(kv_cache.index_k, **shape_args),
            _replicated_topology(sp, cols),
            policy,
            index_k_base,
            config_name(len(caches), num_configs),
            sorted(index_k_layers) if index_k_layers is not None else [],
        )
    )
    geometry = disaggregation.MapGeometry(
        num_layers, num_users, seq_len, position_step=NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    )
    return disaggregation.CacheLayoutModel(caches, mesh_id, ttnn.MeshShape(sp, cols), geometry)
