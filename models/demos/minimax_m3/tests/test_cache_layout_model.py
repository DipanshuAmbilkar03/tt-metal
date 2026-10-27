# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Device-free checks that the M3 KV chunk table derived from the caches' declared layout
(``CacheLayoutModel``, ``to_chunk_map``) is the table the prefill writer's bank walk gives.

``build_and_serialize_kv_chunk_table`` derives the single-mesh table from the layout model; handing it the
same mesh as a one-stage ``stage_layouts`` makes it walk instead. The two tables must agree on every
(config, layer, slot, position): address, size and device group. The layout model written next to the
table must derive the same table again.
"""

import os
import socket
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.minimax_m3.tt.attention.kv_cache import BH_NUM_DRAM_BANKS, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
from models.demos.minimax_m3.tt.runners.cache_layout import LAYOUT_MODEL_SUFFIX
from models.demos.minimax_m3.tt.runners.kv_chunk_table import build_and_serialize_kv_chunk_table

disaggregation = ttnn.experimental.disaggregation

SP = 2
COLS = 4  # == num_kv_heads
CHUNK_SIZE = 128  # chunk_local = 64: two DRAM blocks per SP row per prefill chunk
SEQ_LEN = 256
NUM_USERS = 2
NUM_LAYERS = 4
HEAD_DIM = 128
INDEX_K_LAYERS = [1, 3]
BASES = (0x10000, 0x80000, 0xF0000)  # k, v, index_k


def _mesh_device(chip_id):
    return SimpleNamespace(
        get_fabric_node_id=lambda coord: ttnn.FabricNodeId(ttnn.MeshId(0), chip_id(coord[0], coord[1])),
        arch=lambda: ttnn.device.Arch.BLACKHOLE,
    )


def _stub_cache():
    def t(base):
        return SimpleNamespace(
            shape=(NUM_USERS * NUM_LAYERS, 1, SEQ_LEN, HEAD_DIM), dtype=ttnn.bfloat8_b, buffer_address=lambda: base
        )

    return SimpleNamespace(k=t(BASES[0]), v=t(BASES[1]), index_k=t(BASES[2]), index_k_tp_axis=None)


def _build(path, mesh_device, *, walked=False):
    stage_layouts = None
    if walked:
        fnids = [[mesh_device.get_fabric_node_id((r, c)) for c in range(COLS)] for r in range(SP)]
        stage_layouts = [
            [
                {
                    "first_layer": 0,
                    "count": NUM_LAYERS,
                    "base_addr": base,
                    "num_banks": BH_NUM_DRAM_BANKS,
                    "host_name": socket.gethostname(),
                    "fnids": fnids,
                }
            ]
            for base in BASES
        ]
    build_and_serialize_kv_chunk_table(
        mesh_device=mesh_device,
        kv_cache=_stub_cache(),
        seq_len=SEQ_LEN,
        num_layers=NUM_LAYERS,
        mesh_shape=(SP, COLS),
        sp_axis=0,
        num_users=NUM_USERS,
        chunk_size=CHUNK_SIZE,
        num_kv_heads=COLS,
        head_dim=HEAD_DIM,
        path=path,
        stage_layouts=stage_layouts,
        index_k_layers=INDEX_K_LAYERS,
    )
    return disaggregation.import_from_protobuf_file(path)


def _cells(table):
    """Every (config, layer, slot, position) -> (noc_addr, size_bytes, device group)."""
    cells = {}
    for config_id in range(table.num_configs()):
        name = table.config_name(config_id)
        for layer in range(NUM_LAYERS):
            for slot in range(NUM_USERS):
                for position in range(0, SEQ_LEN, NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK):
                    location = table.lookup(layer, position, slot, config_id)
                    group = table.get_device_group(location.device_group_index).fabric_node_ids
                    if location.size_bytes == 0:
                        group = []
                    cells[(name, layer, slot, position)] = (
                        location.noc_addr,
                        location.size_bytes,
                        sorted((int(f.mesh_id), int(f.chip_id)) for f in group),
                    )
    return cells


@pytest.fixture(scope="module")
def tables(tmp_path_factory):
    directory = tmp_path_factory.mktemp("m3_layout")
    mesh_device = _mesh_device(lambda r, c: r * COLS + c)
    derived_path = os.path.join(str(directory), "derived.pb")
    derived = _build(derived_path, mesh_device)
    walked = _build(os.path.join(str(directory), "walked.pb"), mesh_device, walked=True)
    return SimpleNamespace(derived=derived, walked=walked, model_path=derived_path + LAYOUT_MODEL_SUFFIX)


def test_the_derived_table_is_the_walked_table(tables):
    assert [tables.derived.config_name(i) for i in range(tables.derived.num_configs())] == [
        f"{i:02d}" for i in range(2 * COLS + 1)
    ]
    derived = _cells(tables.derived)
    assert derived == _cells(tables.walked)
    published = {layer for (name, layer, _, _), (_, size, _) in derived.items() if name == "08" and size}
    assert published == set(INDEX_K_LAYERS)


def test_the_layout_model_next_to_the_table_derives_it(tables):
    model = disaggregation.import_cache_layout_model_from_protobuf_file(tables.model_path)
    assert [cache.name for cache in model.caches] == [f"{i:02d}" for i in range(2 * COLS + 1)]
    assert model.caches[-1].layers == INDEX_K_LAYERS
    assert all(cache.layers == [] for cache in model.caches[:-1])
    assert [cache.base_addr for cache in model.caches] == [BASES[0]] * COLS + [BASES[1]] * COLS + [BASES[2]]
    assert (model.geometry.num_layers, model.geometry.num_slots, model.geometry.max_seq_len) == (
        NUM_LAYERS,
        NUM_USERS,
        SEQ_LEN,
    )
    assert model.caches[0].policy.device_chunk_size == CHUNK_SIZE // SP
    assert _cells(disaggregation.to_chunk_map(model)) == _cells(tables.derived)


def test_a_mesh_numbered_column_major_keeps_the_walk(tmp_path):
    mesh_device = _mesh_device(lambda r, c: c * SP + r)
    path = os.path.join(str(tmp_path), "column_major.pb")
    walked = _build(path, mesh_device)
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)
    assert _cells(walked) == _cells(_build(os.path.join(str(tmp_path), "walked.pb"), mesh_device, walked=True))
