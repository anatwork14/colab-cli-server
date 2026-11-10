"""Pair separate arm processes by task, batch and repetition before step filtering."""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

from dattn.bench.analysis import Cell, analyse_cell
from dattn.bench.driver import identity_faults, identity_hash
from dattn.bench.protocol import Protocol
from dattn.metrics.record import read_table


def config_faults(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    """Refuse missing facts, wrong live bindings, or any non-DA config difference."""
    faults, configs = [], []
    for meta in (left, right):
        arm = meta.get("arm")
        if arm not in {"da", "vanilla"}:
            faults.append(f"unknown arm {arm!r}")
        for key in ("dattn_builder_installed", "dattn_lp_installed"):
            if meta.get(key) is not (arm == "da"):
                faults.append(f"{arm}: wrong or missing {key}")
        cfg = json.loads(json.dumps(meta.get("resolved_config") or {}))
        for key in (
            "engine_args",
            "attention_backend",
            "block_sizes",
            "cudagraph_mode",
            "capture_sizes",
            "max_num_batched_tokens",
            "num_gpu_blocks",
        ):
            if cfg.get(key) in (None, [], {}, ""):
                faults.append(f"{arm}: missing resolved {key}")
        args = cfg.get("engine_args", {})
        args.pop("logits_processors", None)
        attention = args.get("attention_config")
        if isinstance(attention, dict):
            attention.pop("backend_per_kind", None)
            if not attention:
                args.pop("attention_config")
        configs.append(cfg)
    if configs[0] != configs[1]:
        faults.append("resolved configurations differ beyond the DA bindings")
    return faults


def _runs(root: Path) -> dict[tuple[str, str, int, int], tuple[Path, dict[str, Any]]]:
    runs = {}
    paths = [root] if root.is_file() else sorted(root.rglob("arm.json"))
    for path in paths:
        doc = json.loads(path.read_text())
        key = (doc["protocol"]["model_id"], doc["task"], doc["batch_index"], doc["rep_index"])
        if key in runs:
            raise ValueError(f"duplicate process for {key} under {root}")
        if int(doc["meta"].get("preempted_cell", 0)):
            raise ValueError(f"{path}: preempted capture")
        if doc.get("faults"):
            raise ValueError(f"{path}: capture faults: {doc['faults']}")
        if not doc.get("identity") or identity_hash(doc["identity"]) != doc.get("identity_hash"):
            raise ValueError(f"{path}: missing or corrupt batch identity")
        runs[key] = path, doc
    return runs


def pair_runs(left: Path | str, right: Path | str, *, out_dir: Path | str) -> list[dict[str, Any]]:
    """Both inputs use the same keys, including for vanilla-versus-vanilla A/A."""
    left_runs, right_runs = _runs(Path(left)), _runs(Path(right))
    if not left_runs or left_runs.keys() != right_runs.keys():
        raise ValueError("arm processes have missing or unmatched task/batch/repetition keys")
    grouped: dict[tuple[str, str, int], list[Any]] = {}
    for key in sorted(left_runs):
        lp, a = left_runs[key]
        rp, b = right_runs[key]
        if (a["arm"], b["arm"]) not in {("da", "vanilla"), ("vanilla", "vanilla")}:
            raise ValueError("inputs must be DA/vanilla or vanilla/vanilla, regardless of execution order")
        faults = config_faults(a["meta"], b["meta"])
        faults += identity_faults(a["identity"], b["identity"], cell_id=str(key))
        if a["protocol"] != b["protocol"]:
            faults.append("protocols differ")
        if faults:
            raise ValueError(f"{key}: {' | '.join(faults)}")
        grouped.setdefault(key[:3], []).append((key[3], lp, a, rp, b))
    summaries = []
    for (model, task, batch), pairs in grouped.items():
        proto = pairs[0][2]["protocol"]
        protocol = Protocol(**{f.name: proto[f.name] for f in fields(Protocol) if f.name in proto})
        aa = pairs[0][2]["arm"] == "vanilla"
        if aa and (protocol.repetitions < 2 or protocol.repetitions % 2):
            raise ValueError("vanilla/vanilla A/A requires even repetitions, at least two")
        if [p[0] for p in pairs] != list(range(protocol.repetitions)):
            raise ValueError(f"{task}/{batch}: expected {protocol.repetitions} matched repetitions")
        sets: dict[str, list[Any]] = {"da": [], "vanilla": []}
        audits, order = [], []
        meta = dict(pairs[0][2]["meta"])
        meta.update(task=task, batch_index=batch, arm_kinds={})
        preempted = 0
        previous_end = float("-inf")
        in_boot = "boot_id" in meta
        if in_boot:
            for offset in (2, 4):
                docs = [p[offset] for p in pairs]
                boot = docs[0]["meta"]["boot_id"]
                previous = docs[0]["meta"]["process_start_unix"]
                for doc in docs:
                    m = doc["meta"]
                    if (
                        m.get("boot_id") != boot
                        or m.get("task_index") != meta["task_index"]
                        or m.get("process_start_unix") != docs[0]["meta"]["process_start_unix"]
                        or m.get("process_end_unix") != docs[0]["meta"]["process_end_unix"]
                    ):
                        raise ValueError("repetitions changed boot identity or task index")
                    if boot["config_hash"] != identity_hash(m["resolved_config"]):
                        raise ValueError("boot config hash changed")
                    if not previous <= doc["t_pass_start"] <= doc["t_pass_end"] <= m["process_end_unix"]:
                        raise ValueError("repetitions overlap or are out of order")
                    previous = doc["t_pass_end"]
            first_boot, second_boot = sorted((pairs[0][2], pairs[0][4]), key=lambda d: d["meta"]["process_start_unix"])
            if first_boot["meta"]["process_end_unix"] > second_boot["meta"]["process_start_unix"]:
                raise ValueError("arm boots overlapped")
            if not aa and (first_boot is pairs[0][2]) != ((meta["task_index"] + batch) % 2 == 0):
                raise ValueError("expected DA first on even task_index + batch_index, vanilla first on odd")
        for rep, lp, a, rp, b in pairs:
            first, second = sorted((a, b), key=lambda d: d["meta"]["process_start_unix"])
            if first["t_pass_end"] > second["meta"]["process_start_unix"]:
                raise ValueError(f"{task}/{batch}/{rep}: arm processes overlapped")
            if not in_boot and first["meta"]["process_start_unix"] < previous_end:
                raise ValueError("repetitions overlap or are out of order")
            previous_end = second["t_pass_end"]
            if a["identity"] != pairs[0][2]["identity"]:
                raise ValueError("batch identity changed between repetitions")
            order.append(first is a)
            if not in_boot and protocol.repetitions == 1 and order[-1] != (batch % 2 == 0):
                raise ValueError(f"{task}/{batch}: expected DA first on even batches, vanilla first on odd batches")
            if not in_boot and rep and order[-1] == order[-2]:
                raise ValueError(f"{task}/{batch}: process order did not alternate AB then BA")
            for label, path, doc in (("da", lp, a), ("vanilla", rp, b)):
                if rep and meta["arm_kinds"][label] != doc["arm"]:
                    raise ValueError("arm kind changed between repetitions")
                meta["arm_kinds"][label] = doc["arm"]
                faults = config_faults(pairs[0][2]["meta"], doc["meta"])
                if faults:
                    raise ValueError(f"configuration changed between repetitions: {faults}")
                if len(doc["windows"]) != 1:
                    raise ValueError(f"{path}: expected one window per process")
                wid, window = next(iter(doc["windows"].items()))
                records = read_table(path.parent / f"steps_{wid}.jsonl")
                if not 0 <= window["start"] < window["stop"] <= len(records):
                    raise ValueError(f"{path}: invalid or empty window")
                sets[label].append(records[window["start"] : window["stop"]])
                preempted += int(doc["meta"].get("preempted_cell", 0))
                audit = doc["meta"].get("audit", {})
                if doc["arm"] == "da":
                    audit = audit.get("recorder", audit)
                    if not audit.get("checked", 0) or audit.get("mismatches", 0) or audit.get("groups_dropped", 0):
                        raise ValueError(f"{path}: DA audit missing, empty or failed")
                    audits.append(audit)
            if a["arm"] == "da" and len(sets["vanilla"][-1]) < len(sets["da"][-1]):
                raise ValueError(f"{task}/{batch}/{rep}: vanilla window is shorter than DA")
        meta["audit"] = {k: sum(int(a.get(k, 0)) for a in audits) for k in ("checked", "mismatches", "groups_dropped")}
        meta["preempted_cell"] = preempted
        labels = ("left_first", "right_first") if aa else ("da_first", "vanilla_first")
        meta["pass_order"] = [labels[0] if first else labels[1] for first in order]
        if in_boot or protocol.repetitions == 1:
            batches = [b for m, t, b in grouped if (m, t) == (model, task)]
            da_first = sum((meta.get("task_index", 0) + b) % 2 == 0 for b in batches)
            meta["task_pass_order"] = {
                "da_first": da_first,
                "vanilla_first": len(batches) - da_first,
                "imbalance": 2 * da_first - len(batches),
                "planned_batches": protocol.batches_per_task,
            }
        meta["per_rep_n_steps"] = {
            label: [len(records) for records in repetitions] for label, repetitions in sets.items()
        }
        meta["pair_sources"] = [[str(p[1]), str(p[3])] for p in pairs]
        meta["pair_key"] = ["task", "batch_index", "rep_index", "step_index"]
        arms = {label: [r for rep in reps for r in rep] for label, reps in sets.items()}
        cell = Cell(protocol, arms, meta, protocol.concurrency, task, sets, source_protocol=proto)
        path = Path(out_dir) / model.replace("/", "_") / task / f"b{batch}"
        summaries.append(analyse_cell(cell, out_dir=path, task=task))
    return summaries
