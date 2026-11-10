"""Deterministic, without-replacement batch draws from resolved KV group costs."""

import hashlib
import json
import math
import random
from pathlib import Path

CAP = 512
TARGET = 0.80
TOLERANCE = 0.03
MODEL_VERSION = "block-accounting-v1"


def sequence_bytes(length, groups, physical_bytes_per_block):
    """Each group-owned block consumes one shared physical pool block."""
    total = 0
    for group in groups:
        kind = group["kind"]
        if kind in {"mamba", "gdn", "gdn_attn"}:
            blocks = 1
        elif kind == "full_attention":
            blocks = math.ceil(length / group["block_size"])
        elif kind == "sliding_window":
            blocks = math.ceil(min(length, group["sliding_window"]) / group["block_size"])
        else:
            raise ValueError(f"unsupported sizing group: {kind}")
        total += blocks * physical_bytes_per_block
    return total


def draw_seed(seed, task, batch_index):
    payload = json.dumps([seed, task, batch_index], separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")


def draw_batch(task, lengths, *, seed, batch_index, groups, physical_bytes_per_block, pool_bytes, max_num_seqs):
    if pool_bytes <= 0 or max_num_seqs <= 0 or not groups or any(n <= 0 for n in lengths):
        raise ValueError("sizing requires positive pool, limit, prompt lengths and resolved groups")
    seed = draw_seed(seed, task, batch_index)
    order = list(range(len(lengths)))
    random.Random(seed).shuffle(order)
    indices, cost = [], 0
    lower, upper = (TARGET - TOLERANCE) * pool_bytes, (TARGET + TOLERANCE) * pool_bytes
    for i in order:
        candidate = cost + sequence_bytes(lengths[i] + CAP, groups, physical_bytes_per_block)
        if candidate > upper:
            continue
        indices.append(i)
        cost = candidate
        if cost >= lower or len(indices) == max_num_seqs:
            break
    return {
        "indices": indices,
        "concurrency": len(indices),
        "seed": seed,
        "modelled_bytes": cost,
        "modelled_occupancy": cost / pool_bytes,
        "shortfall": cost < lower,
    }


def pin_batches(path, *, task, lengths, protocol, groups, physical_bytes_per_block, pool_bytes, max_num_seqs, identity):
    """Persist all five draws once. Reuse only an exactly reproducible task pin."""
    doc = {
        "sizing_model_version": MODEL_VERSION,
        "groups": groups,
        "physical_bytes_per_block": physical_bytes_per_block,
        "pool_bytes": pool_bytes,
        "target": TARGET,
        "tolerance": TOLERANCE,
        "CAP": CAP,
        "enable_prefix_caching": identity["engine_kwargs"]["enable_prefix_caching"],
        "max_num_batched_tokens": identity["engine_kwargs"]["max_num_batched_tokens"],
        "identity": identity,
        "batches": [
            draw_batch(
                task,
                lengths,
                seed=protocol.seed,
                batch_index=b,
                groups=groups,
                physical_bytes_per_block=physical_bytes_per_block,
                pool_bytes=pool_bytes,
                max_num_seqs=max_num_seqs,
            )
            for b in range(protocol.batches_per_task)
        ],
    }
    path = Path(path)
    if path.exists():
        previous = json.loads(path.read_text())
        if any(b["concurrency"] > max_num_seqs for b in previous.get("batches", [])):
            raise ValueError("pinned concurrency above max_num_seqs")
        if previous != doc:
            raise ValueError("sizing pin identity or draw changed")
        return previous
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(doc, f, indent=2)
    return doc


def sizing_validation(snapshot, pool, *, enable_prefix_caching):
    """Compare resident load at maximum concurrency, preferring all-decode state."""
    if snapshot is None:
        raise ValueError("warm pass ran no scheduler steps")
    predicted_bytes = sum(
        sequence_bytes(r["resident_tokens"], pool["groups"], pool["physical_bytes_per_block"])
        for r in snapshot["requests"]
    )
    predicted = predicted_bytes / pool["pool_bytes"]
    measured = snapshot["kv_usage"]
    return {
        "snapshot": snapshot,
        "pool": pool,
        "predicted_bytes": predicted_bytes,
        "predicted": predicted,
        "measured": measured,
        "difference": measured - predicted,
        "enable_prefix_caching": enable_prefix_caching,
        "gate": "one_sided" if enable_prefix_caching else "two_sided",
        "passed": measured <= predicted + TOLERANCE
        if enable_prefix_caching
        else abs(predicted - measured) <= TOLERANCE,
    }
