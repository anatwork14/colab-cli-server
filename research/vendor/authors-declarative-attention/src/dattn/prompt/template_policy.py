"""Explicit rollout rendering policy, also consumed from run metadata by replay.

The two data files in prompt/templates are byte copies from dac fb1ef99,
src/dac/templates/{gemma4,qwen3_5}_chat_template_with_assistant_mask.jinja2.
dac installs these at server boot (scripts/260507_live_dac_datagen/serve.py:1120).
No dac import, tokenizer mutation, or network lookup is involved here.
"""

import hashlib
from pathlib import Path
from typing import Any

TEMPLATE_SHA256 = {
    "gemma4": "341d3242476c37e31d0d88d7fb294428d8220234da7c02307fe310a73b7a417a",
    "qwen3_5": "8d783e2caa28fe7fddd1d968603b9ff628b8d61527b1c6a825702e7bc7b0c357",
}


def resolve_template(tokenizer: Any, model_id: str, policy: str) -> str:
    """Resolve one explicit template, failing closed on unknown bundled models."""
    if policy == "stock":
        template = tokenizer.get_chat_template()
    elif policy == "dac_bundled":
        if model_id.startswith("google/gemma-4"):
            family = "gemma4"
        elif model_id.startswith(("Qwen/Qwen3.5-", "Qwen/Qwen3.6-")):
            family = "qwen3_5"
        else:
            raise ValueError(f"no dac_bundled template for {model_id!r}, choose stock explicitly")
        path = Path(__file__).parents[1] / "prompt" / "templates" / f"{family}_chat_template_with_assistant_mask.jinja2"
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != TEMPLATE_SHA256[family]:
            raise ValueError(f"bundled template sha256 mismatch: {path}")
        template = raw.decode("utf-8")
    else:
        raise ValueError(f"unknown template policy {policy!r}")
    if not isinstance(template, str) or not template:
        raise ValueError("rendering requires a nonempty chat template")
    return template


def template_metadata(template: str, policy: str) -> dict[str, str]:
    return {"policy": policy, "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest()}
