"""CPU-only actual-population equal-length segmentation audit, cached tokenizers only."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dattn.campaign.population import MODELS, digest, load_population
from dattn.eval.persistence import atomic_json
from dattn.prompt.segment import normalize_context, segment_context, segment_equal_length

_TOKENIZER = None


def initialize(model):
    global _TOKENIZER
    from transformers import AutoTokenizer

    _TOKENIZER = AutoTokenizer.from_pretrained(model, local_files_only=True)


def audit_row(row):
    text = normalize_context(row["context"])
    semantic = segment_context(_TOKENIZER, text)
    equal = segment_equal_length(_TOKENIZER, text, len(semantic))
    lengths = [s.token_count for s in equal]
    return {
        "subset": row["subset"],
        "row_index": row["row_index"],
        "semantic": len(semantic),
        "equal": len(equal),
        "delta": len(semantic) - len(equal),
        "spread": max(lengths) - min(lengths),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError('audit requires CUDA_VISIBLE_DEVICES=""')
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    result = {}
    for name, model in MODELS.items():
        rows = load_population(name)
        records = []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(model,)) as pool:
            for i, record in enumerate(pool.map(audit_row, rows, chunksize=8)):
                records.append(record)
                if i % 200 == 0:
                    print(f"{name}: {i}/{len(rows)}", flush=True)
        result[name] = {
            "n": len(records),
            "selection_sha256": digest(rows),
            "count_delta_distribution": dict(sorted(Counter(r["delta"] for r in records).items())),
            "different_fraction": sum(r["delta"] != 0 for r in records) / len(records),
            "worst_count_delta": max(r["delta"] for r in records),
            "spread_distribution": dict(sorted(Counter(r["spread"] for r in records).items())),
            "worst_spread": max(r["spread"] for r in records),
            "rows": records,
        }
        atomic_json(args.out, result)
        print(f"{name}: " + str({k: v for k, v in result[name].items() if k != "rows"}), flush=True)


if __name__ == "__main__":
    main()
