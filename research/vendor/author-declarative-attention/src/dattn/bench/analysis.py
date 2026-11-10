"""Paired step periods, block-rounded bins and a fixed-task batch bootstrap."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from dattn.bench.protocol import Protocol


def derive(records, *, cell_id="unknown"):
    """Compute successors before any filtering. Duration belongs to row k."""
    frame = pd.DataFrame([r.to_dict() for r in records])
    if frame.empty:
        return frame
    frame["step_index"] = np.arange(len(frame))
    frame["wall_ms"] = (frame.t_sched_ns.shift(-1) - frame.t_sched_ns) / 1e6
    bad = frame.loc[frame.wall_ms.le(0)]
    if not bad.empty:
        row = bad.iloc[0]
        raise ValueError(
            f"{cell_id}: step_index={row.step_index}, engine_step={row.engine_step}, "
            f"step_id={row.step_id}: nonpositive wall_ms={row.wall_ms}"
        )
    frame["contiguous"] = frame.engine_step.shift(-1).eq(frame.engine_step + 1) & frame.step_id.shift(-1).eq(
        frame.step_id + 1
    )
    seq, att = [], []
    for row in frame.itertuples():
        bs = row.dattn_block_size
        lengths = row.rows["seq_len"]
        attended = row.rows["att_blk"] or row.rows["att_tok"]
        if bs <= 0 or len(lengths) != len(attended):
            raise ValueError("missing block size or misaligned attended lengths")
        # Round EACH request, then sum. att_blk stores block-rounded TOKEN lengths.
        seq.append(sum((x + bs - 1) // bs for x in lengths))
        att.append(sum((x + bs - 1) // bs for x in attended))
    frame["mask_ratio"] = 1 - pd.Series(att, dtype=float) / pd.Series(seq, dtype=float)
    return frame


def closed_rectangle(da, vanilla, *, protocol, concurrency=0, cell_id="unknown"):
    """Pair within one repetition before applying the common kept-step mask."""
    assert not any(r.num_preempted for r in [*da, *vanilla]), "preempted batch"
    n = min(len(da), len(vanilla))
    left = derive(da, cell_id=f"{cell_id}/da").iloc[:n]
    right = derive(vanilla, cell_id=f"{cell_id}/vanilla").iloc[:n]
    if not n:
        return pd.DataFrame(columns=["step_index", "da_ms", "vanilla_ms", "mask_ratio"])
    keep = pd.Series(True, index=left.index)
    for frame in (left, right):
        keep &= (
            frame.kind.eq("decode")
            & frame.contiguous
            & frame.num_ctx_tokens.eq(0)
            & frame.num_reqs.eq(concurrency)
            & frame.num_reqs_padded.eq(concurrency)
            & frame.num_preempted.eq(0)
        )
    indices = left.index[keep][protocol.leading_trim :]
    for frame in (left.loc[indices], right.loc[indices]):
        assert frame.num_reqs.eq(concurrency).all() and frame.num_reqs_padded.eq(concurrency).all(), "padded batch"
        assert frame.cudagraph_mode.eq("FULL").all(), "kept step did not use FULL graph"
    return pd.DataFrame(
        {
            "step_index": left.loc[indices, "step_index"],
            "da_ms": left.loc[indices, "wall_ms"],
            "vanilla_ms": right.loc[indices, "wall_ms"],
            "mask_ratio": left.loc[indices, "mask_ratio"],
        }
    ).reset_index(drop=True)


def binned_savings(frame, *, protocol, task, batch):
    """Sufficient statistics per batch/bin. Repetitions are not bootstrap units."""
    if frame.empty:
        return pd.DataFrame(columns=["task", "batch", "bin_index", "n", "da_sum", "vanilla_sum"])
    frame = frame.assign(
        bin_index=np.floor(frame.mask_ratio / protocol.bin_width + 1e-9).clip(0, protocol.n_bins - 1).astype(int)
    )
    bins = frame.groupby("bin_index", as_index=False).agg(
        n=("da_ms", "size"), da_sum=("da_ms", "sum"), vanilla_sum=("vanilla_ms", "sum")
    )
    if "rep_index" in frame:
        bins["n_reps"] = bins.bin_index.map(frame.groupby("bin_index").rep_index.nunique())
    bins["task"], bins["batch"] = task, batch
    bins["mean_da"], bins["mean_vanilla"] = bins.da_sum / bins.n, bins.vanilla_sum / bins.n
    bins["saving"] = 1 - bins.da_sum / bins.vanilla_sum
    return bins


def _task_bins(frame):
    sums = frame.groupby(["task", "bin_index"], as_index=False)[["n", "da_sum", "vanilla_sum"]].sum()
    return sums.assign(saving=1 - sums.da_sum / sums.vanilla_sum)


def cluster_bootstrap(frame, *, n_resamples=2000, ci=0.95, seed=0):
    """Resample whole batches within each fixed task, never steps or tasks."""
    rng = np.random.default_rng(seed)
    tasks = [(task, [batch for _, batch in rows.groupby("batch")]) for task, rows in frame.groupby("task")]
    draws = []
    for _ in range(n_resamples):
        sampled = pd.concat(
            [batches[i] for _, batches in tasks for i in rng.integers(len(batches), size=len(batches))],
            ignore_index=True,
        )
        draws.append(_task_bins(sampled).groupby("bin_index").saving.mean())
    distribution = pd.DataFrame(draws)
    alpha = (1 - ci) / 2
    return distribution.quantile([alpha, 1 - alpha])


def macro_average(frame, *, n_resamples=2000, ci=0.95, seed=0):
    """Ratio of means within each task/bin, then an equal-weight task macro."""
    if frame.empty:
        return []
    per_task = _task_bins(frame)
    macro = per_task.groupby("bin_index").agg(saving=("saving", "mean"), n_tasks=("task", "nunique"))
    counts = frame.groupby(["bin_index", "task"]).batch.nunique()
    enough = counts.groupby("bin_index").min().ge(2)
    limits = cluster_bootstrap(frame, n_resamples=n_resamples, ci=ci, seed=seed) if enough.any() else None
    macro["ci_lo"] = limits.iloc[0].where(enough) if limits is not None else np.nan
    macro["ci_hi"] = limits.iloc[1].where(enough) if limits is not None else np.nan
    macro["ci_defined"] = macro.ci_lo.notna() & macro.ci_hi.notna()
    macro["n_batches_min"] = counts.groupby("bin_index").min()
    macro["n_batches"] = counts.groupby("bin_index").sum()
    return macro.reset_index().to_dict("records")


@dataclass
class Cell:
    protocol: Protocol
    arms: dict
    meta: dict = field(default_factory=dict)
    concurrency: int = 0
    task: str = ""
    sets: dict = field(default_factory=dict)
    source_protocol: dict = field(default_factory=dict)

    def rect(self):
        assert not self.meta.get("preempted_cell", 0), "preempted batch"
        sets = self.sets or {k: [v] for k, v in self.arms.items()}
        if len(sets["da"]) != len(sets["vanilla"]):
            raise ValueError("unmatched repetitions")
        return pd.concat(
            [
                closed_rectangle(
                    a,
                    b,
                    protocol=self.protocol,
                    concurrency=self.concurrency,
                    cell_id=f"{self.protocol.model_id}/{self.task}/b{self.meta.get('batch_index', 0)}/r{rep}",
                ).assign(rep_index=rep)
                for rep, (a, b) in enumerate(zip(sets["da"], sets["vanilla"], strict=True))
            ],
            ignore_index=True,
        )


def burst_statistics(frame):
    """Flag paired steps if either arm exceeds twice its own cell-wide kept median.

    Report medians by arm. Fractions count the union of flagged paired steps, once
    per step. Runs never cross a missing step index or a repetition boundary.
    """
    steps = frame.sort_values(["rep_index", "step_index"])
    periods = steps[["da_ms", "vanilla_ms"]]
    medians = periods.median()
    burst = periods.gt(2 * medians).any(axis=1)
    contiguous = steps.rep_index.eq(steps.rep_index.shift()) & steps.step_index.eq(steps.step_index.shift() + 1)
    groups = (~(burst & burst.shift(fill_value=False) & contiguous)).cumsum()
    longest = int(burst.groupby(groups).sum().max())
    fraction = float(burst.mean())
    return {
        "step_time_median_ms": {"da": float(medians.da_ms), "vanilla": float(medians.vanilla_ms)},
        "burst_fraction": fraction,
        "burst_longest_run": longest,
        "burst_affected": fraction >= 0.05,
    }


def repetition_diagnostics(frame):
    """Vanilla drift uses common kept step positions, never the full 512-token tail."""
    from itertools import combinations

    reps = {int(rep): rows.set_index("step_index") for rep, rows in frame.groupby("rep_index")}
    drift = []
    for (left, a), (right, b) in combinations(reps.items(), 2):
        common = a.index.intersection(b.index)
        if not len(common):
            continue
        x, y = float(a.loc[common, "vanilla_ms"].mean()), float(b.loc[common, "vanilla_ms"].mean())
        drift.append(
            {
                "left_rep": left,
                "right_rep": right,
                "n_kept": len(common),
                "left_mean_ms": x,
                "right_mean_ms": y,
                "relative_change": y / x - 1,
            }
        )
    return {"per_rep_n_kept": {str(rep): len(rows) for rep, rows in reps.items()}, "vanilla_rep_pair_drift": drift}


def analyse_cell(cell, *, out_dir, task=""):
    frame = cell.rect()
    if frame.empty:
        raise ValueError("no kept paired steps")
    bins = binned_savings(frame, protocol=cell.protocol, task=task or cell.task, batch=cell.meta.get("batch_index", 0))
    order_bins = []
    if orders := cell.meta.get("pass_order"):
        frame["pass_order"] = frame.rep_index.map(dict(enumerate(orders)))
        for order, steps in frame.groupby("pass_order"):
            rows = binned_savings(
                steps, protocol=cell.protocol, task=task or cell.task, batch=cell.meta.get("batch_index", 0)
            )
            order_bins.extend(rows[rows.bin_index.eq(0)].assign(pass_order=order).to_dict("records"))
    summary = {
        "protocol": getattr(cell, "source_protocol", {}) or cell.protocol.to_dict(),
        "task": task or cell.task,
        "meta": cell.meta,
        "concurrency": cell.concurrency,
        "n_kept": len(frame),
        "bin0_by_pass_order": order_bins,
        "headline_axis": "mask_ratio",
        "bins": {"mask_ratio": bins.to_dict("records")},
        "publishable": True,
        "repetition_weighting": "within-batch pooled steps over repetitions",
        **burst_statistics(frame),
        **repetition_diagnostics(frame),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "paired_steps.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
