"""The physical pool is shared by groups, with whole-block flooring."""

import json
from types import SimpleNamespace as NS

import pytest

from dattn.contracts import DattnError
from dattn.engine.assertions import check_pool_record, pool_geometry


def cache(monkeypatch):
    import vllm.v1.kv_cache_interface as interface

    monkeypatch.setattr(interface, "get_kv_cache_spec_kind", lambda s: NS(value=s.kind))
    return NS(
        num_blocks=10,
        kv_cache_tensors=[NS(size=2000), NS(size=2000)],
        kv_cache_groups=[
            NS(layer_names=["a", "b"], kv_cache_spec=NS(kind="full_attention", block_size=16, page_size_bytes=100)),
            NS(layer_names=["c"], kv_cache_spec=NS(kind="mamba", block_size=1024, page_size_bytes=150)),
        ],
    )


def test_pool_aliases_flooring_and_null_reservation(monkeypatch):
    result = pool_geometry(cache(monkeypatch), 2050)
    assert result["physical_bytes_per_block"] == 200
    assert result["pool_bytes"] == result["backing_allocation_bytes"] == 2000
    assert result["null_block_reservation"] == 200 and result["usable_pool_bytes"] == 1800
    assert result["rounding_remainder"] == 50
    assert [g["logical_full_capacity_bytes"] for g in result["groups"]] == [2000, 1500]
    assert sum(g["logical_full_capacity_bytes"] for g in result["groups"]) != result["pool_bytes"]


@pytest.mark.parametrize("budget", [1999, 2200])
def test_pool_refuses_wrong_budget(monkeypatch, budget):
    with pytest.raises(AssertionError, match="pool budget"):
        pool_geometry(cache(monkeypatch), budget)


def test_pool_refuses_backing_mismatch(monkeypatch):
    c = cache(monkeypatch)
    c.kv_cache_tensors[0].size = 1900
    with pytest.raises(AssertionError, match="backing"):
        pool_geometry(c, 2050)


def test_recorded_pool_refuses_geometry_change(monkeypatch, tmp_path):
    pool = pool_geometry(cache(monkeypatch), 2050)
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool))
    monkeypatch.setenv("DATTN_POOL_RECORD", str(path))
    check_pool_record(pool)
    with pytest.raises(DattnError, match="geometry changed"):
        check_pool_record(pool | {"num_gpu_blocks": 11})
