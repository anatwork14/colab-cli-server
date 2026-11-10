"""Block accounting and reproducible, immutable, without-replacement draws."""

import json

import pytest

from dattn.bench.protocol import Protocol
from dattn.bench.sizing import draw_batch, pin_batches, sequence_bytes, sizing_validation

GROUPS = [
    {"kind": "full_attention", "block_size": 16, "page_bytes": 100},
    {"kind": "sliding_window", "block_size": 8, "sliding_window": 24, "page_bytes": 50},
    {"kind": "mamba", "block_size": 32768, "page_bytes": 200},
]


def test_cost_rounds_each_sequence_and_group_before_summing():
    assert sequence_bytes(33, GROUPS, 200) == (3 + 3 + 1) * 200
    assert sequence_bytes(9, GROUPS, 200) == (1 + 2 + 1) * 200
    assert sum(sequence_bytes(n, GROUPS, 200) for n in [9, 9]) != sequence_bytes(18, GROUPS, 200)


def draw(lengths, **kwargs):
    return draw_batch(
        "task",
        lengths,
        **(
            {
                "seed": 0,
                "batch_index": 0,
                "groups": GROUPS,
                "physical_bytes_per_block": 200,
                "pool_bytes": 19000,
                "max_num_seqs": 64,
            }
            | kwargs
        ),
    )


def test_band_reached_without_duplicates():
    result = draw([32] * 20)
    assert 0.77 <= result["modelled_occupancy"] <= 0.83
    assert not result["shortfall"]
    assert len(set(result["indices"])) == result["concurrency"]


def test_overshoot_skipped_and_exhaustion_records_shortfall():
    result = draw([100000, 32, 32], pool_bytes=20000)
    assert 0 not in result["indices"]
    assert result["shortfall"] and result["concurrency"] == 2
    assert draw([100000])["indices"] == []


def test_limit_stops_draw_and_records_shortfall():
    result = draw([32] * 20, max_num_seqs=1)
    assert result["shortfall"] and result["concurrency"] == 1


def test_seed_task_and_batch_define_reproducible_shuffle():
    lengths = list(range(100, 140))
    base = draw(lengths)
    assert base == draw(lengths)
    assert base != draw(lengths, seed=1)
    assert base != draw(lengths, batch_index=1)
    assert base != draw_batch(
        "other",
        lengths,
        seed=0,
        batch_index=0,
        groups=GROUPS,
        physical_bytes_per_block=200,
        pool_bytes=10000,
        max_num_seqs=64,
    )


def test_pin_records_every_draw_and_refuses_identity_or_limit_drift(tmp_path):
    path = tmp_path / "task.json"
    kwargs = {
        "task": "t",
        "lengths": [32] * 20,
        "protocol": Protocol(model_id="m"),
        "groups": GROUPS,
        "physical_bytes_per_block": 200,
        "pool_bytes": 10000,
        "max_num_seqs": 64,
        "identity": {"engine_kwargs": {"enable_prefix_caching": True, "max_num_batched_tokens": 16384}},
    }
    pin = pin_batches(path, **kwargs)
    assert len(pin["batches"]) == 5 and pin["CAP"] == 512
    assert pin["enable_prefix_caching"] and pin["target"] == 0.8 and pin["tolerance"] == 0.03
    assert pin_batches(path, **kwargs) == pin
    with pytest.raises(ValueError, match="changed"):
        pin_batches(path, **(kwargs | {"pool_bytes": 11000}))
    pin["batches"][0]["concurrency"] = 65
    path.write_text(json.dumps(pin))
    with pytest.raises(ValueError, match="above max_num_seqs"):
        pin_batches(path, **kwargs)


@pytest.mark.parametrize(
    "caching, measured, passed", [(True, 0.57, True), (False, 0.57, False), (True, 0.84, False), (False, 0.81, True)]
)
def test_resident_validation_sharing_only_relaxes_the_lower_side(caching, measured, passed):
    pool = {
        "groups": [{"kind": "full_attention", "block_size": 16}],
        "physical_bytes_per_block": 100,
        "pool_bytes": 1000,
    }
    snapshot = {"requests": [{"resident_tokens": 128}], "kv_usage": measured, "n_running": 1, "C": 2, "step_index": 3}
    result = sizing_validation(snapshot, pool, enable_prefix_caching=caching)
    assert result["passed"] is passed
    assert result["predicted"] == 0.8 and result["measured"] == measured
    assert result["difference"] == pytest.approx(measured - 0.8)
    assert result["enable_prefix_caching"] is caching
    assert result["gate"] == ("one_sided" if caching else "two_sided")
