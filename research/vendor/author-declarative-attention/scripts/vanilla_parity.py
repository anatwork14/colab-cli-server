"""CPU audit of campaign vanilla prompts against dac's recorded token counts.

Uses the real campaign renderer and population validator. No dac code is imported.
The no-system and stock-template renders are diagnostic controls only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from dattn.campaign.population import (
    DAC_ROOT,
    MODELS,
    NG,
    ROWS_ROOT,
    SLUGS,
    digest,
    identity,
    indexed,
    load_population,
    reference,
)
from dattn.prompt.formats.base import CHAT_TEMPLATE_KWARGS, DEFAULT_SYSTEM
from dattn.prompt.render import render
from dattn.prompt.template_policy import resolve_template

_STATE = {}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def initialize(model):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
    _STATE.update(tok=tok, bundled=resolve_template(tok, model, "dac_bundled"))


def audit_row(row):
    tok = _STATE["tok"]
    built = render(
        tok,
        context=row["context"],
        question=row["question"],
        fmt="vanilla",
        chat_template=_STATE["bundled"],
    )
    messages = list(built.messages)
    assert messages[0] == {"role": "system", "content": DEFAULT_SYSTEM}
    assert built.tools is None
    counts = {}
    for name, turns, template in (
        ("no_system", messages[1:], _STATE["bundled"]),
        ("stock", messages, tok.get_chat_template()),
    ):
        text = tok.apply_chat_template(
            turns,
            tools=None,
            chat_template=template,
            tokenize=False,
            add_generation_prompt=True,
            **CHAT_TEMPLATE_KWARGS,
        )
        counts[name] = len(tok(text, add_special_tokens=False)["input_ids"])
    recorded = row["recorded_tokens"]
    return {
        "subset": row["subset"],
        "row_index": row["row_index"],
        "recorded_tokens": recorded,
        "current_tokens": len(built.ids),
        "no_system_tokens": counts["no_system"],
        "stock_tokens": counts["stock"],
        "current_minus_recorded": len(built.ids) - recorded,
        "no_system_minus_recorded": counts["no_system"] - recorded,
        "current_minus_no_system": len(built.ids) - counts["no_system"],
        "stock_minus_bundled": counts["stock"] - len(built.ids),
        "prompt_sha256": sha(built.text.encode()),
        "token_ids_sha256": digest(list(built.ids)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError('audit requires CUDA_VISIBLE_DEVICES=""')
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    old_body = NG / "reference_templates/vanilla_v3.md"
    new_body = Path(__file__).resolve().parents[1] / "src/dattn/prompt/templates/vanilla.md"
    assert old_body.read_bytes() == new_body.read_bytes()
    result = {
        "policy": "dac_bundled",
        "instruction_body_sha256": sha(new_body.read_bytes()),
        "versions": {p: importlib.metadata.version(p) for p in ("transformers", "tokenizers")},
        "models": {},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for name, model in MODELS.items():
        rows = load_population(name)
        refs = indexed(reference(name, "vanilla")[0])
        assert len(rows) == len(refs) == 1611
        for row in rows:
            n = refs[identity(row)]["rendered_prompt_tokens"]
            assert type(n) is int and n > 0
            row["recorded_tokens"] = n
        records = []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(model,)) as pool:
            for i, record in enumerate(pool.map(audit_row, rows, chunksize=4), 1):
                records.append(record)
                if i % 200 == 0:
                    print(f"{name}: {i}/1611", flush=True)
        source = DAC_ROOT / SLUGS[name] / "vanilla_v3/responses.jsonl"
        contexts = ROWS_ROOT / name / "rows.jsonl"
        summary = {
            "n": len(records),
            "source": str(source),
            "source_sha256": sha(source.read_bytes()),
            "contexts": str(contexts),
            "contexts_sha256": sha(contexts.read_bytes()),
            "distributions": {
                field: dict(sorted(Counter(r[field] for r in records).items()))
                for field in (
                    "current_minus_recorded",
                    "no_system_minus_recorded",
                    "current_minus_no_system",
                    "stock_minus_bundled",
                )
            },
        }
        result["models"][name] = {**summary, "rows": records}
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"{name}: {json.dumps(summary, sort_keys=True)}", flush=True)


if __name__ == "__main__":
    main()
