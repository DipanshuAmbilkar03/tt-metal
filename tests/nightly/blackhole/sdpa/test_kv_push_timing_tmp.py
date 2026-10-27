# Temporary: per-layer timing of the productised sparse KV push at a 1M prefix (not for commit).

import os

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from tests.nightly.blackhole.sdpa.test_sparse_sdpa_kv_push import (
    _DRAM_BANKS,
    _KV_FORMATS,
    _TORUS_XY_GLM,
    _TRACE_1M,
    _TRACE_1M_START,
    H,
    K_DIM,
    TOPK,
    TP,
    V_DIM,
    _cols,
    _kv_cache,
    _sparse_sdpa,
    _traced_us,
    _zero_kv_buffer,
)


def _rows(first, last, cols=12):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, first), ttnn.CoreCoord(cols - 1, last))])


@pytest.mark.timeout(0)
@pytest.mark.parametrize("device_params", [_TORUS_XY_GLM], indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_kv_push_timing(mesh_device):
    from safetensors.numpy import load_file

    from models.demos.deepseek_v3_d_p.tt.mla.sparse_kv_push import _shard_over_mesh

    kv_fmt, chunk_global, links = "fp8", 5120, 2
    n_dev = mesh_device.get_num_devices()
    stripe_chunk = s_local = chunk_global // n_dev
    cache_tokens = _TRACE_1M_START + chunk_global
    pages_per_packet = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // 704)
    buffer_rows = min(cache_tokens, s_local * TOPK) + n_dev * pages_per_packet * _DRAM_BANKS
    layers = [int(x) for x in os.environ.get("TOPK_LAYERS", "14,58,62").split(",")]
    from models.demos.deepseek_v3_d_p.tt.tt_ccl import get_tt_ccl

    profiles = {
        "rows2": (_rows(2, 9), _rows(0, 1)),
        "derived": get_tt_ccl(mesh_device).get_sparse_kv_push_grids(links, s_local),
    }

    common = dict(layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    shard_seq = ttnn.ShardTensorToMesh(mesh_device, dim=2)
    replicate = ttnn.ReplicateTensorToMesh(mesh_device)
    cache = _kv_cache(mesh_device, 2, cache_tokens, kv_fmt, shard_seq, seed=2)
    dense_out = _zero_kv_buffer(mesh_device, cache_tokens, kv_fmt)
    compact = [_zero_kv_buffer(mesh_device, buffer_rows, kv_fmt) for _ in range(2)]
    q = ttnn.from_torch(
        torch.randn(1, H, n_dev * s_local, K_DIM).to(torch.bfloat16), dtype=ttnn.bfloat16, mesh_mapper=shard_seq, **common
    )
    start = ttnn.from_torch(
        torch.tensor([[[[_TRACE_1M_START]]]], dtype=torch.int32), dtype=ttnn.uint32, mesh_mapper=replicate, **common
    )
    sems = [
        ttnn.create_global_semaphore(mesh_device, _cols(0, 11), 0, ttnn.BufferType.L1_SMALL) for _ in range(2)
    ]
    sdpa_kw = dict(kv_format=_KV_FORMATS[kv_fmt][0], scale=K_DIM**-0.5, k_chunk_size=128)

    lines = [
        f"GLM-5.3 real top-k fp8, chunk {chunk_global}, prefix {_TRACE_1M_START}, 8x4 TORUS_XY, traced us",
        f"{'layer':>5} {'profile':>10} | {'dense: sdpa+gather':>20} | {'plan':>5} {'push':>5} {'sdpa':>5} "
        f"{'fused':>6} {'rem':>5} | {'plan/3+fused+rem':>16} {'overhang':>8} {'saved':>6}",
    ]
    tables = None
    for layer in layers:
        f = _TRACE_1M / "dsa" / f"dsa_topk_indices_layer_{layer}" / "rows_00000000_00005120.safetensors"
        natural = next(iter(load_file(str(f)).values())).astype(np.int64).reshape(n_dev, s_local, TOPK)
        idx_natural = ttnn.from_torch(
            torch.from_numpy(natural.reshape(1, 1, n_dev * s_local, TOPK).astype(np.int32)),
            dtype=ttnn.uint32,
            mesh_mapper=shard_seq,
            **common,
        )

        def plan():
            table, compact_idx = ttnn.experimental.sparse_push_plan(
                idx_natural,
                num_devices=n_dev,
                stripe_chunk=stripe_chunk,
                max_rows=cache_tokens,
                pages_per_packet=pages_per_packet,
                banks=_DRAM_BANKS,
                chunk_start=start,
            )
            _shard_over_mesh(table, mesh_device)
            ttnn.experimental.high_bw_all_gather(table, dim=2, output_tensor=tables, cluster_axis=None, num_links=links)
            return compact_idx

        if tables is None:
            t0, _ = ttnn.experimental.sparse_push_plan(
                idx_natural,
                num_devices=n_dev,
                stripe_chunk=stripe_chunk,
                max_rows=cache_tokens,
                pages_per_packet=pages_per_packet,
                banks=_DRAM_BANKS,
                chunk_start=start,
            )
            tables = ttnn.from_torch(
                torch.zeros(1, 1, t0.shape[2] * n_dev, t0.shape[3], dtype=torch.int32),
                dtype=ttnn.uint32,
                mesh_mapper=replicate,
                **common,
            )
        compact_idx = plan()
        t_plan = _traced_us(mesh_device, plan)

        def push(buf, slot, part):
            return ttnn.experimental.sparse_kv_push(
                cache,
                output_tensor=buf,
                table=tables,
                stripe_chunk=stripe_chunk,
                max_rows=cache_tokens,
                part=part,
                num_links=links,
                batch_index=slot,
                ready_semaphore=sems[0],
                data_valid_semaphore=sems[1],
            )

        def dense_gather():
            ttnn.experimental.high_bw_all_gather(
                cache, dim=2, output_tensor=dense_out, cluster_axis=None, num_links=links, input_batch_index=1
            )

        t_gather = _traced_us(mesh_device, dense_gather)
        t_dense_sdpa = _traced_us(mesh_device, lambda: _sparse_sdpa(q, dense_out, idx_natural, s_local * TP, kv_fmt))
        dense = t_gather + t_dense_sdpa
        push(compact[0], 0, 0)
        t_push = _traced_us(mesh_device, lambda: push(compact[1], 1, 0))

        for name, (sdpa_grid, push_grid) in profiles.items():
            sdpa_alone = lambda: ttnn.transformer.sparse_sdpa(
                q, compact[0], compact_idx, V_DIM, **sdpa_kw, sub_core_grids=sdpa_grid
            )
            fused = lambda: ttnn.transformer.sparse_sdpa(
                q,
                compact[0],
                compact_idx,
                V_DIM,
                **sdpa_kw,
                sub_core_grids=sdpa_grid,
                push_input=cache,
                push_table=tables,
                push_output=compact[1],
                push_stripe_chunk=stripe_chunk,
                push_max_rows=cache_tokens,
                push_batch_index=1,
                push_num_links=links,
                push_ready_semaphore=sems[0],
                push_data_valid_semaphore=sems[1],
                push_core_grid=push_grid,
            )
            t_sdpa = _traced_us(mesh_device, sdpa_alone)
            t_fused = _traced_us(mesh_device, fused)
            t_rem = _traced_us(mesh_device, lambda: push(compact[1], 1, 2))
            total = t_plan / 3 + t_fused + t_rem
            lines.append(
                f"{layer:>5} {name:>10} | {t_dense_sdpa:>6.0f} + {t_gather:>5.0f} = {dense:>5.0f} | "
                f"{t_plan:>5.0f} {t_push:>5.0f} {t_sdpa:>5.0f} {t_fused:>6.0f} {t_rem:>5.0f} | "
                f"{total:>16.0f} {t_fused + t_rem - t_sdpa:>8.0f} {100 * (dense - total) / dense:>5.1f}%"
            )
            logger.info(lines[-1])
    logger.info("\n" + "\n".join(lines))
