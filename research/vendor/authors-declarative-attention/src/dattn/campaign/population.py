"""Pinned campaign rows, validated against every historical headline cell.

Use the cached selected-row artifact used by the full-population prompt comparator.
Selection precedes pilot ordering and sharding. No repeat index enters selection.
The artifact includes contexts that historical response files do not retain.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from dattn.eval.datasets import HEADLINE_128CAP_COUNTS, HEADLINE_SUBSETS, resolve_subset_key
from dattn.eval.schema import iter_jsonl

#: Campaign workspace root. Override with `DATTN_WORKSPACE`; every path below is relative to it.
NG = Path(os.environ.get("DATTN_WORKSPACE", "workspace"))
DEFAULT_ROOT = NG / "results/step3"
DAC_ROOT = NG / "reference_outputs"
ROWS_ROOT = NG / "results/phase9f_matrix"
MODELS = {"gemma": "google/gemma-4-31B-it", "qwen": "Qwen/Qwen3.6-27B"}
SLUGS = {"gemma": "gemma_4_31b_it", "qwen": "qwen3.6_27b"}
ARMS = {"da": "dac_chunk_multi_turn_tool_v1", "da_nm": "dac_chunk_multi_turn_tool_v1_no_mask", "vanilla": "vanilla_v3"}
PILOT = tuple(
    resolve_subset_key(s)
    for s in ("ruler_niah_multikey_2", "longbench_v2_multidoc_qa", "qmsum", "longbench_v2_code_repo")
)
HOURS = {
    "gemma": {"da": 2.5, "da_nm": 2.5, "vanilla": 1.75, "da_eqseg": 2.5},
    "qwen": {"da": 1.5, "da_nm": 1.5, "vanilla": 1.2, "da_eqseg": 1.5},
}


def identity(row):
    return row["subset"], row["row_index"], row.get("response_index", 0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def indexed(rows):
    out = {}
    for row in rows:
        k = identity(row)
        if k in out:
            raise ValueError(f"duplicate key {k}")
        out[k] = row
    return out


def reference(model, arm, root=DAC_ROOT):
    cell = root / SLUGS[model] / ARMS[arm]
    all_rows = list(iter_jsonl(cell / "responses.jsonl"))
    rows = [r for r in all_rows if r["subset"] in HEADLINE_SUBSETS]
    indexed(rows)
    counts = dict(Counter(r["subset"] for r in rows))
    if counts != HEADLINE_128CAP_COUNTS or any(identity(r)[2] != 0 for r in rows):
        raise ValueError(f"{cell}: invalid reference population: {counts}")
    return rows, {"total": len(all_rows), "headline": len(rows), "dropped": len(all_rows) - len(rows)}


def load_population(model, rows_root=ROWS_ROOT, dac_root=DAC_ROOT):
    cached = [
        r
        for r in iter_jsonl(rows_root / model / "rows.jsonl")
        if "_selection_spec" not in r and r["subset"] in HEADLINE_SUBSETS
    ]
    rows_by_key = indexed(cached)
    for arm in ARMS:
        ref, _ = reference(model, arm, dac_root)
        refs = indexed(ref)
        if set(rows_by_key) != set(refs):
            raise ValueError(f"{model}/{arm}: cached selection does not match reference IDs")
        for k, row in rows_by_key.items():
            if any(row[field] != refs[k][field] for field in ("question", "answer", "rubric")):
                raise ValueError(f"{model}/{arm}: row content differs at {k}")
    # row_index is the post-shuffle prefix index in dac, so this retains selection order.
    return sorted(cached, key=lambda r: (HEADLINE_SUBSETS.index(r["subset"]), r["row_index"]))


def pilot_order(rows):
    order = (*PILOT, *(s for s in HEADLINE_SUBSETS if s not in PILOT))
    return sorted(rows, key=lambda r: order.index(r["subset"]))


def shards(rows, hours):
    """Pilot plus up to four remaining shards, sized from estimated GPU minutes.

    Use context-token weights to approximate within-cell work. Pilot membership is
    fixed. A one-minute tolerance avoids dropping a shard for rounding just
    below a multiple of the approximate 20-minute minimum.
    """
    ordered = pilot_order(rows)
    pilot = [r for r in ordered if r["subset"] in PILOT]
    rest = [r for r in ordered if r["subset"] not in PILOT]

    def weight(r):
        return max(1, r.get("raw_context_tokens") or len(r["context"]) // 4)

    total = sum(map(weight, ordered))
    pilot_minutes = hours * 60 * sum(map(weight, pilot)) / total
    remaining_minutes = hours * 60 - pilot_minutes
    count = max(1, min(4, math.floor((remaining_minutes + 1) / 20)))
    # Contiguous weighted partitions retain within-task order.
    result = [("pilot", pilot, pilot_minutes)]
    at = 0
    for i in range(count):
        target = sum(map(weight, rest[at:])) / (count - i)
        end, work = at, 0
        while end < len(rest) - (count - i - 1) and (work < target or end == at):
            work += weight(rest[end])
            end += 1
        if i == count - 1:
            end = len(rest)
            work = sum(map(weight, rest[at:end]))
        result.append((f"rest{i}", rest[at:end], hours * 60 * work / total))
        at = end
    return result
