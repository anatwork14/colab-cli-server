"""Five-sample accuracy with explicit partial-task coverage and no rep0 writes."""

from __future__ import annotations

from dattn.campaign import compare as c
from dattn.campaign.population import ARMS, HEADLINE_SUBSETS, MODELS, SLUGS, digest, identity, indexed, reference
from dattn.campaign.samples import readonly
from dattn.eval.persistence import failed


def ready(row, judgments):
    return not failed(row) and (
        row.get("format_correct") is False
        or (row.get("format_correct") is True and identity(row) in judgments and not failed(judgments[identity(row)]))
    )


def report(root, repeats, dac_root, tasks, *, stage, macro_pp, task_pp, task_se, per_task_stop, ablation=False):
    if repeats != ["rep0"] or ablation:
        raise ValueError("--samples uses rep0 plus samples and cannot be combined with repeats or ablation")
    coverage, refs, runs, hashes, counts, nulls = {}, {}, {}, {}, {}, {}
    for model in MODELS:
        for arm in ARMS:
            label = f"{model}/{arm}"
            rr, counts[label] = reference(model, arm, dac_root)
            rr = [r for r in rr if r["subset"] in tasks]
            rj = [r for r in readonly(dac_root / SLUGS[model] / ARMS[arm] / "judgments.jsonl") if r["subset"] in tasks]
            refs[label] = c.summarize(rr, rj, tasks)
            expected = indexed(rr)
            all_responses, all_judgments = [], []
            for tree, allowed in (("rep0", {0}), ("samples", {1, 2, 3, 4})):
                path = root / tree / SLUGS[model] / arm
                responses = readonly(path / "responses.jsonl", committed=tree == "samples")
                judgments = readonly(path / "judgments.jsonl", committed=tree == "samples")
                if any(
                    identity(r)[2] not in allowed or r["subset"] not in HEADLINE_SUBSETS for r in responses + judgments
                ):
                    raise ValueError(f"{path}: foreign response index or task")
                responses = [r for r in responses if r["subset"] in tasks]
                judgments = [r for r in judgments if r["subset"] in tasks]
                ri, ji = indexed(responses), indexed(judgments)
                if not ji.keys() <= ri.keys():
                    raise ValueError(f"{path}: orphan judgments")
                for key, row in ri.items():
                    base_key = (*key[:2], 0)
                    if base_key not in expected:
                        raise ValueError(f"{path}: unexpected row {key}")
                    if not failed(row) and any(
                        row.get(f) != expected[base_key].get(f) for f in ("question", "answer", "rubric")
                    ):
                        raise ValueError(f"{path}: content mismatch {key}")
                for key, row in ri.items():
                    if not failed(row) and type(row.get("format_correct")) is not bool:
                        raise ValueError(f"{path}: missing boolean format_correct {key}")
                    if key in ji and not row.get("format_correct"):
                        raise ValueError(f"{path}: judgment for a format failure {key}")
                for key, judgment in ji.items():
                    if not failed(judgment):
                        for field in ("correct", "dac_correct"):
                            if field not in judgment or (
                                judgment[field] is not None and type(judgment[field]) is not bool
                            ):
                                raise ValueError(f"{path}: invalid judgment {field} {key}")
                if tree == "rep0":
                    c.validate_population(responses, rr, path)
                    c.summarize(responses, judgments, tasks, candidate=True)
                hashes[f"{tree}/{label}"] = digest(sorted(responses, key=identity) + sorted(judgments, key=identity))
                all_responses.extend(responses)
                all_judgments.extend(judgments)
            js = indexed(all_judgments)
            coverage[label], runs[label] = {}, [{} for _ in range(5)]
            for task in tasks:
                n = refs[label][task]["n"]
                by_sample = []
                for i in range(5):
                    selected = [r for r in all_responses if r["subset"] == task and identity(r)[2] == i]
                    scored = [r for r in selected if ready(r, js)]
                    by_sample.append(
                        {
                            "response_index": i,
                            "generated": sum(not failed(r) for r in selected),
                            "scored": len(scored),
                            "expected": n,
                        }
                    )
                    if len(scored) == n:
                        judgments = [js[identity(r)] for r in scored if identity(r) in js]
                        runs[label][i][task] = c.summarize(scored, judgments, (task,), candidate=True)[task]
                coverage[label][task] = {"by_sample": by_sample, "complete": all(x["scored"] == n for x in by_sample)}
            nulls[label] = {
                "reference": refs[label]["macro"]["n_unparsable"],
                "by_sample": {
                    str(i): {
                        "n_unparsable": sum(
                            j.get("correct") is None for j in all_judgments if identity(j)[2] == i and not failed(j)
                        ),
                        "n_dac_unparsable": sum(
                            j.get("dac_correct") is None for j in all_judgments if identity(j)[2] == i and not failed(j)
                        ),
                    }
                    for i in range(5)
                },
            }
    common = [t for t in tasks if all(coverage[label][t]["complete"] for label in coverage)]
    cells = {}
    for label in coverage:
        cells[label] = {}
        for task in tasks:
            if coverage[label][task]["complete"]:
                # compare_cell also computes a macro. Only retain this task here.
                cells[label][task] = c.compare_cell(
                    {**refs[label], "macro": refs[label][task]},
                    [{**r, "macro": r[task]} for r in runs[label]],
                    (task,),
                    macro_pp=macro_pp,
                    task_pp=task_pp,
                    task_se=task_se,
                    per_task_stop=per_task_stop,
                )[task]
            else:
                cells[label][task] = {
                    "n": refs[label][task]["n"],
                    "dac_pct": 100 * refs[label][task]["correct"],
                    "dac_correct_pct": None,
                    "correct_pct": None,
                    "run_sd_pp": None,
                    "delta_pp": None,
                    "delta_se": None,
                    "stop": False,
                    "flag": False,
                }
        if common:
            ref = {
                **refs[label],
                "macro": {
                    "correct": c.mean(refs[label][t]["correct"] for t in common),
                    "n": sum(refs[label][t]["n"] for t in common),
                },
            }
            candidates = [
                {**run, "macro": {f: c.mean(run[t][f] for t in common) for f in ("correct", "dac_correct")}}
                for run in runs[label]
            ]
            cells[label]["macro"] = c.compare_cell(
                ref,
                candidates,
                common,
                macro_pp=macro_pp,
                task_pp=task_pp,
                task_se=task_se,
                per_task_stop=per_task_stop,
            )["macro"]
        else:
            cells[label]["macro"] = {
                "n": 0,
                "dac_pct": None,
                "dac_correct_pct": None,
                "correct_pct": None,
                "run_sd_pp": None,
                "delta_pp": None,
                "delta_se": None,
                "stop": False,
                "flag": False,
            }
    for label, cell in cells.items():
        by_sample = nulls[label]["by_sample"]
        nulls[label]["by_repeat"] = {
            "rep0": by_sample["0"],
            "samples": {
                f: sum(by_sample[str(i)][f] for i in range(1, 5)) for f in ("n_unparsable", "n_dac_unparsable")
            },
        }
        for task, item in cell.items():
            for field in ("correct_run_sd_pp", "correct_delta_pp", "correct_delta_se", "se_pp", "threshold_pp"):
                item.setdefault(field, None)
            if task != "macro":
                item.setdefault("rows", refs[label][task]["keys"])
                item.setdefault("parser_disagreements", [[] for _ in range(5)])
    complete = len(common) == len(tasks)
    stops = [{"cell": label, "task": t} for label, cell in cells.items() for t, item in cell.items() if item["stop"]]
    flags = [{"cell": label, "task": t} for label, cell in cells.items() for t, item in cell.items() if item["flag"]]
    verdict = {
        "schema": c.SCHEMA,
        "stage": stage,
        "status": ("fail" if stops else "pass") if complete else "incomplete",
        "complete": complete,
        "root": str(root.resolve()),
        "repeats": ["rep0"],
        "tasks": list(tasks),
        "thresholds": {"macro_pp": macro_pp, "task_pp": task_pp, "task_se": task_se, "per_task_stop": per_task_stop},
        "se_convention": "single dac run, observed rate, macro quadrature / task count",
        "gate_parser": "dac_correct",
        "cells": cells,
        "stops": stops,
        "flags": flags,
        "reference_counts": counts,
        "n_unparsable": nulls,
        "input_sha256": hashes,
        "samples": {
            "n": 5,
            "response_indices": list(range(5)),
            "sd_ddof": 1,
            "coverage": coverage,
            "macro_tasks": common,
        },
    }
    lines = [
        "# Step 3 five-sample comparison",
        "",
        "Mean accuracy and sample SD (ddof=1) across response indices 0 through 4, in percent and pp. "
        "SE uses the single dac reference run. Format failures count as wrong. "
        "Incomplete tasks have no five-sample estimate. The macro uses only tasks complete in all six cells.",
        "",
        "| cell / task | scored / expected per sample 0..4 | mean % | SD pp | dac % | delta pp | delta SE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, cell in cells.items():
        for task, item in cell.items():
            covered = (
                ", ".join(f"{v['scored']}/{v['expected']}" for v in coverage[label][task]["by_sample"])
                if task != "macro"
                else f"{len(common)}/{len(tasks)} tasks"
            )
            lines.append(
                f"| {label}/{task} | {covered} | {c.fmt(item['dac_correct_pct'])} | "
                f"{c.fmt(item['run_sd_pp'])} | {c.fmt(item['dac_pct'])} | {c.fmt(item['delta_pp'])} | "
                f"{c.fmt(item['delta_se'])} |"
            )
    lines += ["", f"Verdict: **{verdict['status']}**. Macro tasks: {', '.join(common) or 'none'}."]
    return verdict, "\n".join(lines) + "\n"
