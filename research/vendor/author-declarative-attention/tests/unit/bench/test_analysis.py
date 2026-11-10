"""The walltime statistic and filtering invariants, independent of engine execution."""

from dataclasses import replace

import pandas as pd
import pytest

from dattn.bench.analysis import Cell, binned_savings, closed_rectangle, derive, macro_average
from dattn.bench.protocol import Protocol
from dattn.contracts import StepRecord, StepRows


def records(n=20, *, ms=10, seq=(17, 17), att=(16, 16), arm="da"):
    return [
        StepRecord(
            step_id=i,
            engine_step=i + 100,
            t_sched_ns=1_000_000 + int(i * ms * 1e6),
            num_reqs=2,
            num_reqs_padded=2,
            dattn_block_size=16,
            cudagraph_mode="FULL",
            rows=StepRows(seq_len=list(seq), att_blk=list(att) if arm == "da" else [], att_tok=list(att)),
        )
        for i in range(n)
    ]


def proto(**kwargs):
    return Protocol(**({"model_id": "m", "concurrency": 2, "leading_trim": 0, "n_resamples": 20} | kwargs))


def test_period_belongs_to_current_step_and_rounding_is_per_request():
    rs = records(3)
    rs[2].t_sched_ns += 20_000_000
    frame = derive(rs)
    assert frame.wall_ms[:2].tolist() == [10, 30]
    assert pd.isna(frame.wall_ms.iloc[-1])
    assert frame.mask_ratio.tolist() == [0.5] * 3  # ceil(17/16)+ceil(17/16), not ceil(34/16)


def test_contiguity_is_computed_before_filtering_and_never_bridges_a_gap():
    a, b = records(8), records(8)
    a[2].engine_step += 10
    a[5].step_id += 10
    frame = closed_rectangle(a, b, protocol=proto(), concurrency=2)
    assert frame.step_index.tolist() == [0, 3, 6]
    assert frame.da_ms.tolist() == [10] * 3


@pytest.mark.parametrize(
    "change", [{"kind": "prefill"}, {"num_ctx_tokens": 1}, {"num_reqs": 1}, {"num_reqs_padded": 4}]
)
def test_kept_filter_intersects_both_arms(change):
    a, b = records(4), records(4)
    b[1] = replace(b[1], **change)
    assert closed_rectangle(a, b, protocol=proto(), concurrency=2).step_index.tolist() == [0, 2]


@pytest.mark.parametrize(
    "change,match", [({"num_preempted": 1}, "preempted"), ({"cudagraph_mode": "PIECEWISE"}, "FULL")]
)
def test_preemption_and_non_full_kept_steps_are_refused(change, match):
    a, b = records(4), records(4)
    b[1] = replace(b[1], **change)
    with pytest.raises(AssertionError, match=match):
        closed_rectangle(a, b, protocol=proto(), concurrency=2)


def test_trim_is_per_rep_and_no_duration_crosses_process_boundaries():
    a, b = records(6), records(9, ms=20)
    c = Cell(proto(leading_trim=1), {}, concurrency=2, sets={"da": [a, a], "vanilla": [b, b]})
    frame = c.rect()
    assert len(frame) == 8
    assert frame.da_ms.tolist() == [10] * 8
    assert frame.vanilla_ms.tolist() == [20] * 8


def test_bin_boundaries_and_ratio_of_means():
    frame = pd.DataFrame({"mask_ratio": [0.3, 0.3], "da_ms": [1, 9], "vanilla_ms": [2, 10]})
    bins = binned_savings(frame, protocol=proto(), task="t", batch=0)
    assert bins.bin_index.tolist() == [3]
    assert bins.saving.iloc[0] == pytest.approx(1 - 10 / 12)


def test_macro_weights_tasks_equally_and_weights_steps_within_task():
    frame = pd.DataFrame(
        [
            {"task": "a", "batch": 0, "bin_index": 0, "n": 100, "da_sum": 100, "vanilla_sum": 200},
            {"task": "a", "batch": 1, "bin_index": 0, "n": 1, "da_sum": 9, "vanilla_sum": 10},
            {"task": "b", "batch": 0, "bin_index": 0, "n": 1, "da_sum": 8, "vanilla_sum": 10},
        ]
    )
    row = macro_average(frame, n_resamples=20)[0]
    assert row["saving"] == pytest.approx(((1 - 109 / 210) + 0.2) / 2)
    assert row["n_tasks"] == 2 and not row["ci_defined"]


def test_bootstrap_uses_batches_not_repetitions_or_steps():
    rows = [
        {"task": t, "batch": b, "bin_index": 0, "n": 100, "da_sum": 100, "vanilla_sum": 200}
        for t in ("a", "b")
        for b in (0, 1)
    ]
    row = macro_average(pd.DataFrame(rows), n_resamples=20)[0]
    assert row["ci_defined"] and row["ci_lo"] == row["ci_hi"] == 0.5
    # Thousands of steps in one batch cannot create replication.
    row = macro_average(pd.DataFrame(rows[:1]), n_resamples=20)[0]
    assert not row["ci_defined"]


def test_tasks_are_fixed_and_bin_contribution_counts_are_explicit():
    frame = pd.DataFrame(
        [
            {"task": t, "batch": b, "bin_index": i, "n": 10, "da_sum": d, "vanilla_sum": 100}
            for t, d in (("a", 10), ("b", 90))
            for b in (0, 1)
            for i in ((0, 1) if t == "a" else (0,))
        ]
    )
    rows = macro_average(frame, n_resamples=30)
    assert rows[0]["n_tasks"] == 2 and rows[1]["n_tasks"] == 1
    assert rows[0]["ci_lo"] == rows[0]["ci_hi"] == 0.5
    assert rows[1]["saving"] == 0.9


def test_five_by_one_analysis_counts_batch_intervals_and_pass_order(tmp_path):
    from dattn.bench.aggregate import macro_report
    from dattn.bench.analysis import analyse_cell

    protocol = proto(repetitions=1, batches_per_task=5)
    for batch in range(5):
        first = batch % 2 == 0
        # Vary within each order group so the bootstrap must sample batches.
        ms = 4 + batch
        da = records(5, ms=ms, seq=(32, 32), att=(32, 32))
        vanilla = records(5, ms=10, seq=(32, 32), att=(32, 32), arm="vanilla")
        if batch == 0:
            da[2].rows.att_blk = [16, 16]  # bin 5 has only one contributing batch
        cell = Cell(
            protocol,
            {"da": da, "vanilla": vanilla},
            meta={"batch_index": batch, "pass_order": ["da_first" if first else "vanilla_first"]},
            concurrency=2,
            task="t",
        )
        summary = analyse_cell(cell, out_dir=tmp_path / f"b{batch}")
        assert summary["bin0_by_pass_order"][0]["pass_order"] == cell.meta["pass_order"][0]
        assert (
            pd.read_csv(tmp_path / f"b{batch}" / "paired_steps.csv").pass_order.unique().tolist()
            == cell.meta["pass_order"]
        )
    report = macro_report([tmp_path], allow_partial=True)
    bins = {row["bin_index"]: row for row in report["macro"]}
    assert bins[0]["n_batches"] == bins[0]["n_batches_min"] == 5
    assert bins[0]["n_tasks"] == 1 and bins[0]["ci_defined"]
    assert bins[0]["ci_lo"] < bins[0]["ci_hi"]
    assert bins[5]["n_batches"] == bins[5]["n_tasks"] == 1
    assert not bins[5]["ci_defined"]
    split = report["bin0_by_pass_order"]
    assert split["da_first"] == {
        "saving": pytest.approx(1 - (3 * 4 + 4 * 6 + 4 * 8) / 110),
        "n_tasks": 1,
        "n_batches": 3,
        "n_kept": 11,
    }
    assert split["vanilla_first"] == {
        "saving": pytest.approx(1 - (5 + 7) / 20),
        "n_tasks": 1,
        "n_batches": 2,
        "n_kept": 8,
    }


@pytest.mark.parametrize("ms", [0, -1])
@pytest.mark.parametrize("arm", ["da", "vanilla"])
def test_nonpositive_period_fault_names_cell_and_step_before_filtering(ms, arm):
    arms = {"da": records(4), "vanilla": records(7)}
    target = arms[arm]
    # Exercise both a prefill row that would be filtered and a trimmed vanilla tail.
    step = 1 if arm == "da" else 5
    target[step].kind = "prefill"
    target[step + 1].t_sched_ns = target[step].t_sched_ns + ms * 1_000_000
    cell = Cell(proto(), arms, meta={"batch_index": 2}, concurrency=2, task="task")
    with pytest.raises(ValueError, match=rf"m/task/b2/r0/{arm}: step_index={step}, .*nonpositive wall_ms="):
        cell.rect()


def test_unequal_repetition_lengths_keep_pooled_step_weighting():
    import pandas as pd

    from dattn.bench.analysis import binned_savings

    frame = pd.DataFrame(
        {
            "rep_index": [0] * 100 + [1],
            "da_ms": [10.0] * 100 + [20.0],
            "vanilla_ms": [20.0] * 101,
            "mask_ratio": [0.0] * 101,
        }
    )
    row = binned_savings(frame, protocol=Protocol(model_id="m"), task="t", batch=0).iloc[0]
    assert row.n == 101 and row.n_reps == 2
    assert row.saving == pytest.approx(1 - 1020 / 2020)
