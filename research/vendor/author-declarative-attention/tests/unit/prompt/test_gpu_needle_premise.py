"""The long-prefill GPU test's premise is satisfiable, checked WITHOUT a GPU.

`tests/gpu/test_long_prompt_prefill.py` is the hardware proof for review item 1, the
no-token watchdog crash. It needs one filler rung that is simultaneously past
`GRACE_CHUNKS * MAX_NUM_BATCHED_TOKENS`, or the old watchdog would never have fired and a
pass proves nothing, and inside `MAX_MODEL_LEN` less the sampled tokens, or vLLM's
renderer rejects it before a single engine step.

It shipped with `MAX_MODEL_LEN = 40_960` and no upper bound on the rung search, and the
band it needed was therefore EMPTY on BOTH shipping models: the ladder's only rung past
four chunks is 49,789 tokens on Gemma and 50,941 on Qwen, and neither fits 40,960. The
failure surfaced 90 seconds into an engine boot as a renderer message about output tokens,
which names neither real cause, and it was read as a Qwen-specific problem because Qwen is
where it was first seen.

Where a rung lands is a property of the TOKENIZER, so it cannot be checked on one model
and assumed for the other, and on hardware it costs a GPU slot to discover. Here it costs
nothing: these tokenizers are already loaded once per session for the render tests.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from conftest import MODELS

pytestmark = pytest.mark.slow

GPU_DIR = Path(__file__).resolve().parents[2] / "gpu"


def _load(name: str, path: Path) -> Any:
    """Import a module by file path, so `tests/gpu/conftest.py` does not collide with this one."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered BEFORE execution: `tests/gpu/conftest.py` defines dataclasses, and
    # `dataclasses` resolves a field's module through `sys.modules[cls.__module__]`.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _constants(path: Path) -> dict[str, Any]:
    """The module-level constants of a test file, WITHOUT importing it.

    `tests/gpu/test_long_prompt_prefill.py` does `from conftest import ...`, which under
    this directory's own `conftest.py` resolves to the wrong module. Nothing here needs
    that file's imports, only its numbers, so the assignments are executed on their own.
    """

    def is_constant(node: ast.stmt) -> bool:
        """An UPPER_CASE module-level assignment. `pytestmark` and friends are not."""
        if not isinstance(node, ast.Assign):
            return False
        return all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets)

    tree = ast.parse(path.read_text())
    keep = [node for node in tree.body if is_constant(node)]
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
    return namespace


def test_the_long_prefill_gpu_test_has_a_usable_filler_rung_on_every_model(tokenizer: Any) -> None:
    """One rung must land in the band, on THIS model, or that GPU slot is wasted."""
    t = _constants(GPU_DIR / "test_long_prompt_prefill.py")
    gpu_conftest = _load("_dattn_gpu_conftest", GPU_DIR / "conftest.py")
    lo, hi = t["MIN_PROMPT_TOKENS"], t["MAX_PROMPT_TOKENS"]

    sizes = [
        (cr, len(gpu_conftest.build_needle_prompt(tokenizer, n_chunks=24, chunk_repeats=cr, needle_chunk=12).ids))
        for cr in t["CHUNK_REPEATS"]
    ]
    usable = [(cr, n) for cr, n in sizes if lo <= n <= hi]
    assert usable, (
        f"no filler rung lands in [{lo}, {hi}] for this tokenizer. Rungs (chunk_repeats, tokens): "
        f"{sizes}. The GPU test cannot pass and will burn an engine boot to say so"
    )
    _cr, n = usable[0]
    assert -(-n // t["MAX_NUM_BATCHED_TOKENS"]) > t["GRACE_CHUNKS"], "the rung must be past the old grace"
    assert n + t["MAX_OUTPUT_TOKENS"] <= t["MAX_MODEL_LEN"], "and it must still render"


def test_the_gpu_test_covers_both_models_rather_than_quietly_one() -> None:
    """`MODELS[:1]` gave Gemma while the test id said `model_id0`.

    Getting any Qwen coverage needed a pytest plugin that reordered `MODELS`. A
    parametrised GPU test that hands you one model while you believe you chose between
    them is worth making impossible rather than documenting, so the parametrisation is
    pinned here, where it costs no GPU time to check.
    """
    source = (GPU_DIR / "test_long_prompt_prefill.py").read_text()
    decorators = [line.strip() for line in source.splitlines() if "parametrize" in line and "model_id" in line]
    assert decorators == ['@pytest.mark.parametrize("model_id", MODELS, ids=MODELS)'], decorators
    assert len(MODELS) == 2
