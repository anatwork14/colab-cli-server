"""Four additional responses per rep0 prompt, with a barrier between tasks."""

from __future__ import annotations

import json
import math
import signal
import threading
from dataclasses import replace
from pathlib import Path

from dattn.campaign.population import ARMS, HEADLINE_SUBSETS, HOURS, MODELS, SLUGS, digest, identity, indexed
from dattn.eval.persistence import atomic_json, failed, lock
from dattn.eval.schema import iter_jsonl

N = 4


def readonly(path, *, committed=False):
    """Never create a lock or repair a torn line in rep0."""
    if committed and path.exists():
        lines = path.read_bytes().splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            lines.pop()
        return [json.loads(line) for line in lines if line.strip()]
    return list(iter_jsonl(path)) if path.exists() else []


def safe_root(root):
    root = root.resolve()
    if "rep0" in root.parts:
        raise ValueError("samples output root must not be under rep0")
    base = root / "samples"
    if base.is_symlink() or any(p.is_symlink() for p in base.rglob("*")):
        raise ValueError("samples output tree must not contain symlinks")
    return base


def sample_config(model, arm):
    from dattn.campaign.driver import config

    return replace(config(model, arm), n=N)


def sample_provenance(model, arm, tok):
    from dattn.campaign.driver import provenance

    return {**provenance(model, arm, tok), "rollout": sample_config(model, arm).to_dict()}


def expanded_keys(rows):
    return [[r["subset"], r["row_index"], i] for r in rows for i in range(1, N + 1)]


def task_order(records):
    """Mean rendered length across the six rep0 cells, with a stable tie break."""
    totals = {t: [] for t in HEADLINE_SUBSETS}
    for rows in records.values():
        for row in rows:
            count = row.get("rendered_prompt_tokens")
            if type(count) is not int or count <= 0:
                raise ValueError("rep0 lacks positive rendered prompt token counts")
            totals[row["subset"]].append(count)
    if any(not values for values in totals.values()):
        raise ValueError("rep0 does not cover every headline task")
    means = {t: math.fsum(v) / len(v) for t, v in totals.items()}
    return sorted(means, key=lambda t: (means[t], t)), means


def prepare(args):
    from dattn.campaign import driver
    from dattn.campaign.compare import validate_population

    base = safe_root(args.root)
    populations, records, provenances = {}, {}, {}
    # Finish all rep0 checks before creating samples artifacts.
    for model in MODELS:
        tok = driver.tokenizer(MODELS[model])
        for arm in ARMS:
            cell = args.root / "rep0" / SLUGS[model] / arm
            manifest = json.loads((cell / "campaign.json").read_text())
            rows = json.loads(Path(manifest["population"]).read_text())
            expected, _ = driver.compare.reference(model, arm, args.dac_root)
            validate_population(rows, expected, cell)
            if digest(rows) != manifest["selection_sha256"] or manifest["row_keys"] != [
                list(identity(r)) for r in rows
            ]:
                raise ValueError(f"{cell}: rep0 population provenance mismatch")
            provenance = driver.provenance(model, arm, tok)
            if any(manifest[k] != v for k, v in provenance.items()):
                raise ValueError(f"{cell}: rep0 rendering or sampling changed")
            rr = readonly(cell / "responses.jsonl")
            validate_population(rr, rows, cell)
            if any(failed(r) for r in rr):
                raise ValueError(f"{cell}: rep0 has unfinished responses")
            if model in populations and populations[model] != rows:
                raise ValueError("rep0 arms disagree about population")
            populations[model] = rows
            records[model, arm] = rr
            provenances[model, arm] = sample_provenance(model, arm, tok)
    order, means = task_order(records)
    driver.ensure_json(base / "order.json", {"tasks": order, "mean_rendered_prompt_tokens": means})
    jobs = []
    for model in MODELS:
        rows = populations[model]
        source = base / "population" / f"{model}.json"
        driver.ensure_json(source, rows)
        for arm in ARMS:
            cell = base / SLUGS[model] / arm
            manifest = {
                "schema": "step3.cell.v1",
                "repeat": "samples",
                "model": model,
                "arm": arm,
                "population": str(source),
                "selection_sha256": digest(rows),
                "row_keys": expanded_keys(rows),
                "profile": "parity_prefix",
                "response_index_start": 1,
                **provenances[model, arm],
            }
            if (cell / "responses.jsonl").exists() and not (cell / "campaign.json").exists():
                raise ValueError(f"{cell}: responses without campaign metadata")
            driver.ensure_json(cell / "campaign.json", manifest)
            driver.check_cell(cell, manifest)
            recorded = indexed(records[model, arm])
            total = sum(r["rendered_prompt_tokens"] for r in recorded.values())
            for task in order:
                part = [r for r in rows if r["subset"] == task]
                weight = sum(recorded[identity(r)]["rendered_prompt_tokens"] for r in part)
                minutes = HOURS[model][arm] * 60 * 1.6 * weight / total
                count = min(4, len(part), max(1, math.floor((minutes + 1) / 20)))
                for i in range(count):
                    selected = part[i::count]
                    name = f"{task}-{i}"
                    shard = cell / "shards" / name
                    atomic_json(shard / "keys.json", expanded_keys(selected))
                    jobs.append(
                        {
                            "cell": str(cell),
                            "shard": str(shard),
                            "name": name,
                            "task": task,
                            "model": model,
                            "arm": arm,
                            "minutes": minutes / count,
                        }
                    )
    return order, jobs


def generate_tasks(args, order, jobs, stop):
    from dattn.campaign import driver

    for task in order:
        task_jobs = [j for j in jobs if j["task"] == task]
        driver.generate(args, task_jobs, stop)
        if stop.is_set() or any(driver.pending(j) for j in task_jobs):
            raise RuntimeError(f"samples task incomplete: {task}")
        atomic_json(args.root / "samples" / "tasks" / f"{task}.json", {"task": task, "status": "complete"})


def _run(args, stop):
    from dattn.campaign import driver

    safe_root(args.root)
    with lock(args.root / ".driver.lock", blocking=False):
        try:
            order, jobs = prepare(args)
            driver.check_pidfds()
            driver.wait_all(args.gpus)
            if args.command == "judge":
                cells = sorted(
                    {Path(j["cell"]) for j in jobs if readonly(Path(j["cell"]) / "responses.jsonl", committed=True)}
                )
                if cells:
                    driver.judge(args, cells, stop)
            else:
                generate_tasks(args, order, jobs, stop)
                atomic_json(args.root / "samples" / "generation.complete.json", {"status": "complete", "n": N})
        except (Exception, KeyboardInterrupt) as exc:
            stop.set()
            incident = {"command": args.command, "error": str(exc)}
            if isinstance(exc, driver.CampaignIncident):
                incident["failures"] = exc.failures
            atomic_json(args.root / "samples" / "incident.json", incident)
            print(f"SAMPLES STOP: {exc}", flush=True)
            return 1
    return 0


def run(args):
    stop = threading.Event()

    def interrupt(signum, frame):
        stop.set()
        raise KeyboardInterrupt

    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in previous:
            signal.signal(sig, interrupt)
        return _run(args, stop)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
