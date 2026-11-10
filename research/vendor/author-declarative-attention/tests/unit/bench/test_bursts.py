"""Burst reporting leaves the original paired measurements and macro intact."""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from dattn.bench.aggregate import macro_report
from dattn.bench.analysis import analyse_cell, burst_statistics
from dattn.bench.protocol import Protocol


def kept(n=20):
    return pd.DataFrame({"rep_index": 0, "step_index": range(n), "da_ms": 10.0, "vanilla_ms": 30.0, "mask_ratio": 0.0})


@pytest.mark.parametrize("arm", ["da_ms", "vanilla_ms"])
def test_medians_are_per_arm_and_threshold_is_strict(arm):
    frame = kept()
    baseline = frame[arm].iloc[0]
    frame.loc[18, arm] = 2 * baseline
    frame.loc[19, arm] = 2.4 * baseline
    stats = burst_statistics(frame)
    assert stats == {
        "step_time_median_ms": {"da": 10.0, "vanilla": 30.0},
        "burst_fraction": 0.05,
        "burst_longest_run": 1,
        "burst_affected": True,
    }


def test_clean_cell_and_fraction_below_threshold():
    frame = kept(21)
    assert burst_statistics(frame)["burst_longest_run"] == 0
    assert burst_statistics(frame)["burst_fraction"] == 0
    assert not burst_statistics(frame)["burst_affected"]
    frame.loc[0, "vanilla_ms"] *= 2.4
    assert burst_statistics(frame)["burst_fraction"] == pytest.approx(1 / 21)
    assert not burst_statistics(frame)["burst_affected"]


def test_run_breaks_at_filtered_indices_and_rep_boundaries_and_counts_overlap_once():
    frame = kept(30)
    frame.loc[0:4, ["da_ms", "vanilla_ms"]] *= 2.4
    frame.loc[2:3, "step_index"] = [3, 4]  # step 2 did not survive filtering
    frame.loc[4:, "step_index"] = range(5, 31)
    frame.loc[4:, "rep_index"] = 1  # numeric adjacency must not join different boots
    stats = burst_statistics(frame.sample(frac=1, random_state=7))
    assert stats["burst_longest_run"] == 2
    assert stats["burst_fraction"] == pytest.approx(5 / 30)
    assert stats["burst_affected"]


def cell_summary(path, task, batch, frame):
    cell = SimpleNamespace(
        rect=lambda: frame.copy(),
        protocol=Protocol(model_id="m", concurrency=2, n_resamples=10),
        task=task,
        meta={"batch_index": batch},
        concurrency=2,
    )
    return analyse_cell(cell, out_dir=path)


def test_exclusion_counts_cells_per_bin_and_keeps_original_macro(tmp_path):
    clean = kept()
    burst = kept()
    burst.loc[0:1, "vanilla_ms"] *= 2.4
    burst.loc[10:, "mask_ratio"] = 0.5  # exclude the whole cell, including its clean bin
    flagged = cell_summary(tmp_path / "a0", "a", 0, burst)
    cell_summary(tmp_path / "a1", "a", 1, clean)
    cell_summary(tmp_path / "b0", "b", 0, clean)  # same batch index in a different task
    assert flagged["burst_affected"] and flagged["burst_longest_run"] == 2
    assert flagged["publishable"] and flagged["n_kept"] == 20
    assert len(pd.read_csv(tmp_path / "a0/paired_steps.csv")) == 20
    report = macro_report([tmp_path], allow_partial=True)
    original = {row["bin_index"]: row for row in report["macro"]}
    retained = report["macro_excl_burst"]
    assert original[0]["n_excluded"] == original[5]["n_excluded"] == 1
    assert original[0]["n_batches"] == 3 and original[0]["n_kept"] == 50
    assert original[5]["n_batches"] == 1 and original[5]["n_kept"] == 10
    assert len(retained) == 1 and retained[0]["bin_index"] == 0
    assert retained[0]["n_excluded"] == 1 and retained[0]["n_batches"] == 2
    assert retained[0]["n_tasks"] == 2 and retained[0]["n_kept"] == 40
    assert retained[0]["saving"] == pytest.approx(2 / 3)
    assert original[0]["saving"] != retained[0]["saving"]


@pytest.mark.parametrize("affected", [False, True])
def test_none_or_all_affected_has_no_gate_and_no_lost_exclusion_count(tmp_path, affected):
    for batch in range(2):
        frame = kept()
        if affected:
            frame.loc[0, "da_ms"] *= 2.4
        cell_summary(tmp_path / str(batch), "t", batch, frame)
    report = macro_report([tmp_path], allow_partial=True)
    assert report["macro"][0]["n_excluded"] == (2 if affected else 0)
    if affected:
        assert report["macro_excl_burst"] == []
    else:
        assert json.dumps(report["macro_excl_burst"]) == json.dumps(report["macro"])
