"""CPU comparison with stable, fail-closed step3.verdict.v1 gate records.

All arithmetic traverses sorted row keys and fixed task order, using math.fsum.
Neither shard assignment nor response append order can affect aggregate bytes.
Gate accuracy uses dac_correct to reproduce dac's parser. correct is shown alongside.
SE is a SINGLE reference run's binomial SE at dac's observed rate, not a difference
SE. The reference rate anchors the scale independently of candidate results.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from dattn.campaign.population import (
    ARMS,
    DAC_ROOT,
    DEFAULT_ROOT,
    HEADLINE_SUBSETS,
    MODELS,
    PILOT,
    SLUGS,
    digest,
    identity,
    indexed,
    reference,
)
from dattn.eval.datasets import resolve_subset_key
from dattn.eval.persistence import atomic_json, failed, read_records
from dattn.eval.schema import iter_jsonl

SCHEMA = "step3.verdict.v1"


def mean(values):
    values = list(values)
    return math.fsum(values) / len(values)


def se_pp(rate, n):
    return 100 * math.sqrt(rate * (1 - rate) / n)


def standardized(delta, se):
    return delta / se if se else (0.0 if delta == 0 else None)


def summarize(responses, judgments, tasks, *, candidate=False):
    rs, js = indexed(responses), indexed(judgments)
    if not set(js) <= set(rs):
        raise ValueError("judgments have keys absent from responses")
    result = {}
    for task in tasks:
        keys = sorted(k for k in rs if k[0] == task)
        if not keys:
            raise ValueError(f"no rows for {task}")
        rows = [rs[k] for k in keys]
        if any(failed(r) for r in rows):
            raise ValueError(f"unfinished rollout transports in {task}")
        if candidate:
            if any(type(r.get("format_correct")) is not bool for r in rows):
                raise ValueError("candidate responses lack boolean format_correct")
            if any(k in js and not rs[k]["format_correct"] for k in keys):
                raise ValueError("candidate judgments include format-failed responses")
            missing = [k for k in keys if rs[k]["format_correct"] and (k not in js or failed(js[k]))]
            if missing:
                raise ValueError(f"unfinished judgments in {task}: {missing}")
            for k in keys:
                if k in js:
                    for field in ("correct", "dac_correct"):
                        if field not in js[k] or (js[k][field] is not None and type(js[k][field]) is not bool):
                            raise ValueError(f"candidate judgment has missing or invalid {field}: {k}")
        item = {"n": len(keys), "keys": [list(k) for k in keys]}
        for field in ("correct", "dac_correct"):
            source = field if candidate else "correct"
            item[field] = mean(js.get(k, {}).get(source) is True for k in keys)
            count_field = "n_unparsable" if field == "correct" else "n_dac_unparsable"
            item[count_field] = sum(k in js and js[k].get(source) is None for k in keys)
        item["parser_disagreements"] = (
            [list(k) for k in keys if k in js and js[k].get("correct") != js[k].get("dac_correct")] if candidate else []
        )
        for field in ("total_steps", "attended_tokens"):
            values = [r.get("dac_metrics", {}).get(field) for r in rows]
            item[field] = mean(values) if all(v is not None for v in values) else None
        for field in ("rendered_prompt_tokens", "n_bookmarks", "n_semantic_segments"):
            values = [r.get(field) for r in rows]
            item[field] = mean(values) if all(v is not None for v in values) else None
        result[task] = item
    fields = (
        "correct",
        "dac_correct",
        "total_steps",
        "attended_tokens",
        "rendered_prompt_tokens",
        "n_bookmarks",
        "n_semantic_segments",
    )
    result["macro"] = {
        field: mean(result[t][field] for t in tasks) if all(result[t][field] is not None for t in tasks) else None
        for field in fields
    }
    for field in ("n", "n_unparsable", "n_dac_unparsable"):
        result["macro"][field] = sum(result[t][field] for t in tasks)
    return result


def compare_cell(ref, repeats, tasks, *, macro_pp=2, task_pp=5, task_se=0, per_task_stop=False):
    out = {}
    for task in (*tasks, "macro"):
        reference_rate = ref[task]["correct"]
        se = (
            math.sqrt(math.fsum(se_pp(ref[t]["correct"], ref[t]["n"]) ** 2 for t in tasks)) / len(tasks)
            if task == "macro"
            else se_pp(reference_rate, ref[task]["n"])
        )
        threshold = macro_pp if task == "macro" else max(task_pp, task_se * se)
        strict = [r[task]["dac_correct"] * 100 for r in repeats]
        tolerant = [r[task]["correct"] * 100 for r in repeats]
        delta = mean(strict) - reference_rate * 100
        over = abs(delta) > threshold + 1e-10
        item = {
            "n": ref[task]["n"],
            "dac_pct": reference_rate * 100,
            "dac_correct_pct": mean(strict),
            "correct_pct": mean(tolerant),
            "run_sd_pp": statistics.stdev(strict) if len(strict) > 1 else None,
            "correct_run_sd_pp": statistics.stdev(tolerant) if len(tolerant) > 1 else None,
            "delta_pp": delta,
            "delta_se": standardized(delta, se),
            "correct_delta_pp": mean(tolerant) - reference_rate * 100,
            "correct_delta_se": standardized(mean(tolerant) - reference_rate * 100, se),
            "se_pp": se,
            "threshold_pp": threshold,
            "stop": over and (task == "macro" or per_task_stop),
            "flag": over and task != "macro" and not per_task_stop,
        }
        if task != "macro":
            item["rows"] = ref[task]["keys"]
            item["parser_disagreements"] = [r[task]["parser_disagreements"] for r in repeats]
        out[task] = item
    return out


def validate_verdict(doc, *, repeat, root):
    """A permissive JSON object is not evidence of a pass."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise ValueError("missing or malformed pass1 verdict schema")
    required_cells = {f"{m}/{a}" for m in MODELS for a in ARMS}
    if (
        doc.get("stage") != "full"
        or doc.get("status") != "pass"
        or doc.get("repeats") != [repeat]
        or doc.get("root") != str(root.resolve())
        or doc.get("tasks") != list(HEADLINE_SUBSETS)
        or doc.get("thresholds") != {"macro_pp": 2, "task_pp": 5, "task_se": 0, "per_task_stop": False}
        or set(doc.get("cells", {})) != required_cells
        or doc.get("stops") != []
        or doc.get("complete") is not True
    ):
        raise ValueError("pass2 requires a complete canonical pass1 macro verdict of pass")
    for cell in doc["cells"].values():
        if set(cell) != {*HEADLINE_SUBSETS, "macro"}:
            raise ValueError("malformed verdict cell")
        for name, item in cell.items():
            if not isinstance(item, dict) or item.get("stop") is not False:
                raise ValueError("malformed or stopped verdict task")
            for field in ("delta_pp", "se_pp", "threshold_pp", "n"):
                if not isinstance(item.get(field), (int, float)) or not math.isfinite(item[field]):
                    raise ValueError(f"malformed verdict {field}")
            if name == "macro" and (abs(item["delta_pp"]) > 2 + 1e-10 or item["n"] != 1611):
                raise ValueError("invalid macro pass")
    return doc


def require_pass(path, *, repeat, root, override=False, dac_root=DAC_ROOT):
    if override:
        return {"override": True, "reason": "explicit operator override"}
    try:
        recorded = validate_verdict(json.loads(path.read_text()), repeat=repeat, root=root)
        current, _ = report(
            root,
            [repeat],
            dac_root,
            HEADLINE_SUBSETS,
            stage="full",
            macro_pp=2,
            task_pp=5,
            task_se=0,
            per_task_stop=False,
        )
        if recorded != current:
            raise ValueError("recorded verdict is malformed or no longer matches the input files")
        return recorded
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"Gate B REFUSED: {path}: {exc}") from exc


def fmt(value):
    return "N/A" if value is None else f"{value:.3f}"


def validate_population(responses, expected, path):
    ref_index = indexed(expected)
    if set(indexed(responses)) != set(ref_index):
        raise ValueError(f"{path}: population mismatch, expected {len(expected)}, got {len(responses)}")
    for row in responses:
        if any(row.get(f) != ref_index[identity(row)].get(f) for f in ("question", "answer", "rubric")):
            raise ValueError(f"{path}: content mismatch {identity(row)}")


def report(root, repeats, dac_root, tasks, *, stage, macro_pp, task_pp, task_se, per_task_stop, ablation=False):
    cells, summaries, refs, counts, hashes, unparsable = {}, {}, {}, {}, {}, {}
    for model in MODELS:
        for arm in ARMS:
            label = f"{model}/{arm}"
            rr, count = reference(model, arm, dac_root)
            rr = [r for r in rr if r["subset"] in tasks]
            rj = [
                r for r in iter_jsonl(dac_root / SLUGS[model] / ARMS[arm] / "judgments.jsonl") if r["subset"] in tasks
            ]
            refs[label] = summarize(rr, rj, tasks)
            counts[label] = count
            candidates = []
            for repeat in repeats:
                path = root / repeat / SLUGS[model] / arm
                all_responses = read_records(path / "responses.jsonl")
                if any(r["subset"] not in HEADLINE_SUBSETS for r in all_responses):
                    raise ValueError(f"{path}: unexpected non-headline rows")
                responses = [r for r in all_responses if r["subset"] in tasks]
                validate_population(responses, rr, path)
                judgments = [r for r in read_records(path / "judgments.jsonl") if r["subset"] in tasks]
                candidates.append(summarize(responses, judgments, tasks, candidate=True))
                hashes[f"{repeat}/{label}"] = digest(sorted(responses, key=identity) + sorted(judgments, key=identity))
            unparsable[label] = {
                "reference": refs[label]["macro"]["n_unparsable"],
                "by_repeat": {
                    repeat: {f: run["macro"][f] for f in ("n_unparsable", "n_dac_unparsable")}
                    for repeat, run in zip(repeats, candidates, strict=True)
                },
            }
            summaries[label] = candidates
            cells[label] = compare_cell(
                refs[label],
                candidates,
                tasks,
                macro_pp=macro_pp,
                task_pp=task_pp,
                task_se=task_se,
                per_task_stop=per_task_stop,
            )
    stops = [{"cell": label, "task": t} for label, cell in cells.items() for t, x in cell.items() if x["stop"]]
    flags = [{"cell": label, "task": t} for label, cell in cells.items() for t, x in cell.items() if x["flag"]]
    verdict = {
        "schema": SCHEMA,
        "stage": stage,
        "status": "fail" if stops else "pass",
        "complete": True,
        "root": str(root.resolve()),
        "repeats": repeats,
        "tasks": list(tasks),
        "thresholds": {"macro_pp": macro_pp, "task_pp": task_pp, "task_se": task_se, "per_task_stop": per_task_stop},
        "se_convention": "single dac run, observed dac rate, sqrt(p*(1-p)/n), macro quadrature / task count",
        "gate_parser": "dac_correct",
        "cells": cells,
        "stops": stops,
        "flags": flags,
        "reference_counts": counts,
        "n_unparsable": unparsable,
        "input_sha256": hashes,
    }
    lines = [
        "# Step 3 comparison",
        "",
        "Acceptance rule: macro accuracy within 2 pp of dac per model and arm, "
        "no task beyond 5 pp, anything outside listed with its rows. The full-run "
        "macro is decisive and its task overshoots are flags. "
        + (
            "Pilot task stops are disabled by explicit operator override."
            if stage == "pilot" and not per_task_stop
            else "Pilot task overshoots remain stops."
        ),
        "",
        "SE legend: SINGLE dac run binomial SE, sqrt(p(1-p)/n), using observed dac accuracy. "
        "This anchors the scale to the reference. It is NOT the SE of a difference. "
        "Macro SE is sqrt(sum(task SE squared)) / task count. Zero SE with a nonzero delta prints N/A. "
        "Every accuracy delta has pp and SE columns. Metric ratios have no binomial SE.",
        "Noise floor near p=0.5: n=128 gives 4.4 pp per task and an equal-size 15-task macro about 1.1 pp. "
        "n=18 gives 11.8 pp. Actual unequal-count macro SE is printed below. dac is one run.",
        (
            f"{len(repeats)} repeats: mean and sample run-to-run SD (ddof=1), not SEM."
            if len(repeats) > 1
            else "Single run: SD is unavailable, not zero."
        ),
        "Gate parser is dac_correct. correct is the tolerant parser. Format failures stay in the denominator. "
        "Transport gaps refuse a verdict. P-CAP, P-ANS, E8-E13, E19/E20 and P-REPLAY are detailed in CONTRACT.md.",
        "",
        "| reference cell | total rows | headline rows | dropped |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [f"| {k} | {v['total']} | {v['headline']} | {v['dropped']} |" for k, v in counts.items()]
    lines += [
        "",
        "| cell / task | n | dac % | dac_correct % ± SD | correct % ± SD | strict Δ pp | "
        "strict Δ SE | tolerant Δ pp | tolerant Δ SE | SE pp | threshold pp | verdict |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    row_details = []
    for label, cell in cells.items():
        for task, x in cell.items():
            state = "STOP" if x["stop"] else ("FLAG" if x["flag"] else "ok")
            lines.append(
                f"| {label} / {task} | {x['n']} | {fmt(x['dac_pct'])} | "
                f"{fmt(x['dac_correct_pct'])} ± {fmt(x['run_sd_pp'])} | "
                f"{fmt(x['correct_pct'])} ± {fmt(x['correct_run_sd_pp'])} | "
                f"{fmt(x['delta_pp'])} | {fmt(x['delta_se'])} | "
                f"{fmt(x['correct_delta_pp'])} | {fmt(x['correct_delta_se'])} | "
                f"{fmt(x['se_pp'])} | {fmt(x['threshold_pp'])} | {state} |"
            )
            if task != "macro" and (x["stop"] or x["flag"] or any(x["parser_disagreements"])):
                row_details.append(
                    f"\n{label}/{task} rows: "
                    + json.dumps(x["rows"])
                    + ". Parser disagreements by repeat: "
                    + json.dumps(x["parser_disagreements"])
                    + "\n"
                )
    lines += row_details
    lines += [
        "",
        "Metrics use per-task means then an unweighted task macro, with arm / vanilla computed after aggregation.",
        "P-VANILLA-SYSTEM: campaign vanilla = historical vanilla (no system turn) != "
        "dattn default vanilla (system turn). The known Qwen row retains its +18 tokenizer residual. "
        "DA text replay may omit EOS. "
        "Qwen stock-template inflation is 315 per GLOBAL step on 1610/1611 DA rows (~0.45% DA, ~0.73% DA-nm). "
        "A conservative template-only upper bound is 333 * total_steps (315 plus the exceptional 18-token row). "
        "dac excludes document <pad> tokens. The known Gemma row has two pads and 146 attended-count difference. "
        "Gemma's three-token first-span offset does not cancel here: attended-token deltas are raw exact counts, "
        "attended_tokens_blocks is None, and MASKCHECK.md's 32-token block rounding does not apply. "
        "The bound is 3 spans times 3 tokens per step, about 4k against the Gemma DA mean of 6.45M "
        "(under 0.07 percent, ours up). "
        "These are historical biased metrics, not certified equal measurement conventions.",
        "",
        "| cell / task | steps ours / dac | attended ours / dac | steps % vanilla ours / dac "
        "| attended % vanilla ours / dac | Qwen template upper bound | comparability |",
        "| --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for label, runs in summaries.items():
        model, arm = label.split("/")
        for task in (*tasks, "macro"):
            a = {
                f: mean(r[task][f] for r in runs) if all(r[task][f] is not None for r in runs) else None
                for f in ("total_steps", "attended_tokens")
            }
            b = refs[label][task]
            base = summaries[f"{model}/vanilla"]

            def ratios(field, a=a, b=b, base=base, model=model, task=task):
                if a[field] is None or b[field] is None:
                    return "N/A"
                ownbase = mean(r[task][field] for r in base)
                refbase = refs[f"{model}/vanilla"][task][field]
                return (
                    f"{fmt(100 * a[field] / ownbase)} / {fmt(100 * b[field] / refbase)}"
                    if ownbase and refbase
                    else "N/A"
                )

            bound = 333 * b["total_steps"] if model == "qwen" and arm != "vanilla" and b["total_steps"] else None
            sds = {
                field: statistics.stdev(r[task][field] for r in runs)
                if len(runs) > 1 and a[field] is not None
                else None
                for field in ("total_steps", "attended_tokens")
            }
            lines.append(
                f"| {label}/{task} | {fmt(a['total_steps'])} ± {fmt(sds['total_steps'])} / "
                f"{fmt(b['total_steps'])} | {fmt(a['attended_tokens'])} ± {fmt(sds['attended_tokens'])} / "
                f"{fmt(b['attended_tokens'])} | {ratios('total_steps')} | {ratios('attended_tokens')} | "
                f"{fmt(bound)} | historical conventions differ, EOS/pad residual not corrected |"
            )
    if ablation:
        lines += [
            "",
            "| ablation rep0 / task / baseline | strict accuracy Δ pp | Δ SE | correct Δ pp | Δ "
            "SE | steps % baseline | attended % baseline | prompt overhead % | semantic N | "
            "emitted N |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        if "rep0" not in repeats:
            raise ValueError("ablation needs rep0")
        for model in MODELS:
            path = root / "rep0" / SLUGS[model] / "da_eqseg"
            rr = read_records(path / "responses.jsonl")
            expected, _ = reference(model, "da", dac_root)
            validate_population(rr, expected, path)
            eq = summarize(rr, read_records(path / "judgments.jsonl"), tasks, candidate=True)
            unparsable[f"{model}/da_eqseg"] = {
                "reference": None,
                "by_repeat": {"rep0": {f: eq["macro"][f] for f in ("n_unparsable", "n_dac_unparsable")}},
            }
            for baseline in ("da", "vanilla"):
                base = summaries[f"{model}/{baseline}"][repeats.index("rep0")]
                for task in (*tasks, "macro"):
                    a, b = eq[task], base[task]
                    se = (
                        math.sqrt(math.fsum(se_pp(base[t]["dac_correct"], base[t]["n"]) ** 2 for t in tasks))
                        / len(tasks)
                        if task == "macro"
                        else se_pp(b["dac_correct"], b["n"])
                    )
                    strict, tolerant = 100 * (a["dac_correct"] - b["dac_correct"]), 100 * (a["correct"] - b["correct"])

                    def ratio(field, subtract=0, a=a, b=b):
                        return 100 * (a[field] / b[field] - subtract) if a[field] is not None and b[field] else None

                    lines.append(
                        f"| {model}/{task}/{baseline} | {fmt(strict)} | {fmt(standardized(strict, se))} | "
                        f"{fmt(tolerant)} | {fmt(standardized(tolerant, se))} | "
                        f"{fmt(ratio('total_steps'))} | {fmt(ratio('attended_tokens'))} | "
                        f"{fmt(ratio('rendered_prompt_tokens', 1))} | "
                        f"{fmt(a['n_semantic_segments'])} | {fmt(a['n_bookmarks'])} |"
                    )
        lines.append("\nAblation SE uses the observed single-run baseline rate with the same binomial convention.")
    lines += [
        "",
        "Unparsable verdicts count as wrong. Counts include terminal null judgments, not unjudged format failures. "
        "n_unparsable uses correct, n_dac_unparsable uses dac_correct. Both-None judgments appear in both counts.",
        "",
        "| cell | repeat | reference n_unparsable | n_unparsable | n_dac_unparsable |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for label, counts in unparsable.items():
        for repeat, values in counts["by_repeat"].items():
            reference_count = counts["reference"] if counts["reference"] is not None else "N/A"
            lines.append(
                f"| {label} | {repeat} | {reference_count} | {values['n_unparsable']} | {values['n_dac_unparsable']} |"
            )
    lines.append(f"\nVerdict: **{verdict['status']}**, {len(stops)} stops, {len(flags)} flags.\n")
    return verdict, "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--dac-root", type=Path, default=DAC_ROOT)
    p.add_argument("--repeats", nargs="+", choices=("rep0", "rep1", "rep2"), default=["rep0"])
    p.add_argument("--stage", choices=("pilot", "full"), default="full")
    p.add_argument("--tasks", nargs="+")
    p.add_argument("--macro-pp", type=float)
    p.add_argument("--task-pp", type=float)
    p.add_argument("--task-se", type=float)
    p.add_argument("--per-task-stop", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--samples", action="store_true")
    p.add_argument("--verdict", type=Path, required=True)
    p.add_argument("--markdown", type=Path, required=True)
    args = p.parse_args(argv)
    pilot = args.stage == "pilot"
    tasks = tuple(resolve_subset_key(t) for t in args.tasks) if args.tasks else (PILOT if pilot else HEADLINE_SUBSETS)
    if len(set(tasks)) != len(tasks) or not set(tasks) <= set(HEADLINE_SUBSETS):
        p.error("tasks must be unique headline tasks")
    if len(set(args.repeats)) != len(args.repeats):
        p.error("repeat indices must be unique")
    for value in (args.macro_pp, args.task_pp, args.task_se):
        if value is not None and (value < 0 or not math.isfinite(value)):
            p.error("thresholds must be finite and nonnegative")
    reporter = report
    if args.samples:
        from dattn.campaign.sample_compare import report as reporter

        for path in (args.verdict, args.markdown):
            if path.resolve().is_relative_to((args.root / "rep0").resolve()):
                p.error("--samples cannot write verdicts under rep0")
    try:
        verdict, markdown = reporter(
            args.root,
            args.repeats,
            args.dac_root,
            tasks,
            stage=args.stage,
            macro_pp=args.macro_pp if args.macro_pp is not None else (8 if pilot else 2),
            task_pp=args.task_pp if args.task_pp is not None else (15 if pilot else 5),
            task_se=args.task_se if args.task_se is not None else (3 if pilot else 0),
            per_task_stop=args.per_task_stop if args.per_task_stop is not None else pilot,
            ablation=args.ablation,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        atomic_json(
            args.verdict,
            {"schema": SCHEMA, "stage": args.stage, "status": "fail", "complete": False, "error": str(exc)},
        )
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(f"# Comparison refused\n\n{exc}\n")
        print(f"REFUSED: {exc}")
        return 2
    atomic_json(args.verdict, verdict)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(markdown)
    print(markdown)
    return int(verdict["status"] != "pass")


if __name__ == "__main__":
    raise SystemExit(main())
