"""Self-verifying, bounded GPU identity check for the optional samples profile.

This command is for the later authorized GPU run. Its selection and comparison
functions are CPU-testable. Each profile boots in a separate child process.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from dattn.campaign.population import ARMS, DEFAULT_ROOT, MODELS, PILOT, SLUGS, digest, identity, indexed
from dattn.campaign.samples import readonly
from dattn.eval.persistence import atomic_json

SCHEMA = "step3.prefix_validation.v1"
DECODE_TOKENS = 128


def select_rows(population, recorded):
    """Select extremes by rendered DA prompt length and a shared-document pair."""
    records = indexed(recorded)
    candidates = [r for r in population if r["subset"] in PILOT]
    if len(candidates) < 6:
        raise ValueError("need at least six pilot rows")
    if any(identity(r) not in records or not records[identity(r)].get("rendered_prompt_tokens") for r in candidates):
        raise ValueError("rep0 pilot prompt lengths are incomplete")
    ordered = sorted(candidates, key=lambda r: (records[identity(r)]["rendered_prompt_tokens"], identity(r)))
    selected = ordered[:3] + ordered[-3:]
    groups = defaultdict(list)
    for row in population:
        groups[row["context"]].append(row)
    pairs = []
    for group in groups.values():
        group = sorted(group, key=identity)
        for a in group:
            b = next((b for b in group if b["question"] != a["question"]), None)
            if b is not None:
                pairs.append((a, b))
                break
    if not pairs:
        raise ValueError("no distinct questions sharing exactly one document")
    pair = min(pairs, key=lambda p: (len(p[0]["context"]), identity(p[0]), identity(p[1])))
    return {"rows": selected, "pair": list(pair)}


def first_difference(a, b):
    for i, (left, right) in enumerate(zip(a, b, strict=False)):
        if left != right:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else None


def compare_outputs(plain, cached):
    """Compare actual tokens, replay metrics and live worker trace at each position."""
    for field in ("token_ids", "mode_trace", "attended_tokens_exact", "live_trace", "text", "finish_reason"):
        if plain[field] == cached[field]:
            continue
        pos = first_difference(plain["token_ids"], cached["token_ids"])
        trace_pos = None
        if field in ("mode_trace", "live_trace"):
            trace_pos = first_difference(plain[field], cached[field])
            if field == "live_trace" and trace_pos is not None:
                trace = plain[field] if trace_pos < len(plain[field]) else cached[field]
                trace_pos = trace[trace_pos][0]
        return {
            "field": field,
            "first_differing_token_position": pos,
            "first_differing_trace_position": trace_pos,
            "plain": plain[field] if field == "attended_tokens_exact" else None,
            "cached": cached[field] if field == "attended_tokens_exact" else None,
        }
    return None


def verdict(model, checks, *, error=None, selection=None):
    failure = next((c for c in checks if c.get("difference") is not None), None)
    passed = bool(checks) and failure is None and error is None
    return {
        "schema": SCHEMA,
        "model": model,
        "status": "PASS" if passed else "FAIL",
        "first_differing_token_position": failure["difference"].get("first_differing_token_position")
        if failure
        else None,
        "failure": failure,
        "error": error,
        "checks": checks,
        "selection": selection,
        "decode_tokens": DECODE_TOKENS,
        "token_positions_zero_based": True,
        "scope": "selected greedy prefixes only, no full-population or sampled-output identity claim",
    }


def check_artifacts(plain, prefix, selection):
    checks = []
    expected = {repr(identity(r)) for r in selection["rows"] + selection["pair"]}
    for arm in ARMS:
        if set(plain["arms"][arm]) != expected or set(prefix["arms"][arm]) != expected:
            raise ValueError("validation row coverage incomplete")
        for key, baseline in plain["arms"][arm].items():
            candidate = prefix["arms"][arm][key]
            if set(candidate) != {"cold", "warm", "children"} or len(candidate["children"]) != 4:
                raise ValueError("incomplete prefix cases")
            for variant, outputs in candidate.items():
                for i, out in enumerate(outputs):
                    checks.append(
                        {
                            "arm": arm,
                            "row": key,
                            "case": variant,
                            "child": i,
                            "difference": compare_outputs(baseline["cold"][0], out),
                        }
                    )
        b = repr(identity(selection["pair"][1]))
        checks.append(
            {
                "arm": arm,
                "row": b,
                "case": "shared_document",
                "child": 0,
                "difference": compare_outputs(plain["arms"][arm][b]["cold"][0], prefix["pair"][arm]),
            }
        )
    # Identity without any DA mask transitions would not exercise the feature.
    masked = any(step[1] != 0 for case in plain["arms"]["da"].values() for step in case["cold"][0]["live_trace"])
    if not masked:
        raise ValueError("greedy rows never exercised a DA mask transition")
    return checks


def run(args):
    from dattn.campaign import driver

    started = time.monotonic()
    plan = None
    checks = []
    try:
        driver.assert_free(args.gpu)
        cell = args.root / "rep0" / SLUGS[args.model] / "da"
        manifest = json.loads((cell / "campaign.json").read_text())
        rows = json.loads(Path(manifest["population"]).read_text())
        if digest(rows) != manifest["selection_sha256"]:
            raise ValueError("rep0 population digest changed")
        plan = select_rows(rows, readonly(cell / "responses.jsonl"))
        directory = args.out.parent / (args.out.stem + ".artifacts")
        directory.mkdir(parents=True, exist_ok=True)
        plan_path = directory / "selection.json"
        atomic_json(plan_path, plan)
        artifacts = []
        for profile in ("parity", "parity_prefix"):
            driver.assert_free(args.gpu)
            artifact = directory / f"{profile}.json"
            artifact.unlink(missing_ok=True)
            command = [
                sys.executable,
                "-m",
                "dattn.campaign.validate_prefix",
                "--worker",
                "--gpu",
                args.gpu,
                "--model",
                args.model,
                "--profile",
                profile,
                "--plan",
                str(plan_path),
                "--artifact",
                str(artifact),
                "--out",
                str(args.out),
            ]
            env = {**driver.cpu_env(), "CUDA_VISIBLE_DEVICES": args.gpu, "VLLM_PLUGINS": "dattn"}
            with (directory / f"{profile}.log").open("wb") as log:
                process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                owner = driver.Owned(process)
                try:
                    while process.poll() is None:
                        if time.monotonic() - started >= 1200:
                            raise TimeoutError("20-minute validation budget exhausted")
                        owner.scan()
                        driver.assert_free(args.gpu, owner.fds)
                        time.sleep(0.5)
                    if process.returncode:
                        raise RuntimeError(f"{profile} worker failed, see {log.name}")
                finally:
                    owner.close()
            artifacts.append(json.loads(artifact.read_text()))
        checks = check_artifacts(*artifacts, plan)
        doc = verdict(args.model, checks, selection=plan)
    except (Exception, KeyboardInterrupt) as exc:
        doc = verdict(args.model, checks, error=f"{type(exc).__name__}: {exc}", selection=plan)
    doc["elapsed_seconds"] = time.monotonic() - started
    atomic_json(args.out, doc)
    print(f"{doc['status']} {args.model}: {args.out}", flush=True)
    return 0 if doc["status"] == "PASS" else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", required=True)
    p.add_argument("--model", choices=tuple(MODELS), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--profile", choices=("parity", "parity_prefix"), help=argparse.SUPPRESS)
    p.add_argument("--plan", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--artifact", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if not args.gpu.isdigit():
        p.error("--gpu requires one physical GPU index")
    if args.out.resolve().is_relative_to((args.root / "rep0").resolve()):
        p.error("validation output must not be under rep0")
    if args.worker:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu:
            p.error("worker must see exactly the requested GPU")
        from dattn.cli import _prefix_validation_worker

        return _prefix_validation_worker(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
