"""Equal-task macro over the paper's 15 tasks, bootstrapping whole batches."""

import json
from pathlib import Path

import pandas as pd

from dattn.bench.analysis import macro_average
from dattn.bench.protocol import MACRO_TASKS
from dattn.eval.datasets import HEADLINE_SUBSETS


def macro_report(dirs, *, allow_partial=False, plot=None):
    docs = [json.loads(p.read_text()) for root in dirs for p in sorted(Path(root).rglob("summary.json"))]
    if not docs:
        raise ValueError("no paired summaries")
    models = {d["protocol"]["model_id"] for d in docs}
    if len(models) != 1:
        raise ValueError("macro requires one model")
    tasks = {d["task"] for d in docs}
    if not allow_partial and (len(tasks) != MACRO_TASKS or tasks != set(HEADLINE_SUBSETS)):
        raise ValueError("macro requires all 15 headline tasks")
    keys = [(d["task"], d["meta"]["batch_index"]) for d in docs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate task/batch summary")
    frame = pd.DataFrame([row for d in docs for row in d["bins"]["mask_ratio"]])
    proto = docs[0]["protocol"]
    rows = macro_average(frame, n_resamples=proto["n_resamples"], ci=proto["ci"], seed=proto["seed"])
    kept = frame.groupby("bin_index").n.sum()
    for row in rows:
        row["n_kept"] = int(kept.loc[row["bin_index"]])
        if "n_reps" in frame:
            row["n_reps"] = int(frame.loc[frame.bin_index.eq(row["bin_index"]), "n_reps"].sum())
        row["thin"] = row["n_kept"] < 20 or row["n_tasks"] < 3
    affected = {(d["task"], d["meta"]["batch_index"]) for d in docs if d.get("burst_affected", False)}
    excluded = pd.Series([(task, batch) in affected for task, batch in zip(frame.task, frame.batch, strict=True)])
    counts = frame.loc[excluded, ["bin_index", "task", "batch"]].drop_duplicates().groupby("bin_index").size()
    retained = frame.loc[~excluded]
    rows_excl_burst = macro_average(retained, n_resamples=proto["n_resamples"], ci=proto["ci"], seed=proto["seed"])
    for row in rows:
        row["n_excluded"] = int(counts.get(row["bin_index"], 0))
    if rows_excl_burst:
        kept_excl_burst = retained.groupby("bin_index").n.sum()
        for row in rows_excl_burst:
            row["n_kept"] = int(kept_excl_burst.loc[row["bin_index"]])
            if "n_reps" in retained:
                row["n_reps"] = int(retained.loc[retained.bin_index.eq(row["bin_index"]), "n_reps"].sum())
            row["thin"] = row["n_kept"] < 20 or row["n_tasks"] < 3
            row["n_excluded"] = int(counts.get(row["bin_index"], 0))
    by_order = pd.DataFrame([row for d in docs for row in d.get("bin0_by_pass_order", [])])
    diagnostic = {}
    if not by_order.empty:
        for order, rows_for_order in by_order.groupby("pass_order"):
            sums = rows_for_order.groupby("task")[["da_sum", "vanilla_sum"]].sum()
            diagnostic[order] = {
                "saving": float((1 - sums.da_sum / sums.vanilla_sum).mean()),
                "n_tasks": len(sums),
                "n_batches": len(rows_for_order[["task", "batch"]].drop_duplicates()),
                "n_kept": int(rows_for_order.n.sum()),
            }
    report = {
        "model": next(iter(models)),
        "n_tasks": len(tasks),
        "partial": len(tasks) != MACRO_TASKS,
        "macro": rows,
        "repetition_weighting": "within-batch pooled steps over repetitions",
        "bin0_by_pass_order": diagnostic,
        "cells": docs,
        "macro_excl_burst": rows_excl_burst,
    }
    if plot:
        from dattn.bench.plots import saving_plot

        saving_plot(rows, plot)
    return report
