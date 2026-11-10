"""Tokenizer fixtures for the prompt tests.

The two shipping models only. Loading both costs about two seconds once per session, and
every render test reuses them, which is what keeps `tests/unit` inside the 60 second merge
gate of DESIGN.md 7.2. A box without the tokenizers in its Hugging Face cache skips rather
than fails, so the FSM half of the suite still runs anywhere.
"""

import os
from typing import Any

import pytest

GEMMA = "google/gemma-4-31B-it"
QWEN = "Qwen/Qwen3.6-27B"
MODELS = (GEMMA, QWEN)

_CACHE: dict[str, Any] = {}


def load_tokenizer(model_id: str) -> Any:
    """Load one fast tokenizer, or skip the test that asked for it."""
    if model_id in _CACHE:
        return _CACHE[model_id]
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"{model_id} is not in the local Hugging Face cache ({exc})")
    if not getattr(tokenizer, "is_fast", False):  # pragma: no cover
        pytest.skip(f"{model_id} has no fast tokenizer")
    _CACHE[model_id] = tokenizer
    return tokenizer


@pytest.fixture(scope="session")
def gemma() -> Any:
    return load_tokenizer(GEMMA)


@pytest.fixture(scope="session")
def qwen() -> Any:
    return load_tokenizer(QWEN)


@pytest.fixture(scope="session", params=MODELS, ids=["gemma-4-31B", "Qwen3.6-27B"])
def tokenizer(request: pytest.FixtureRequest) -> Any:
    return load_tokenizer(request.param)
