"""Lane B contract smoke test: vLLM is pinned to 0.29.0.

Every line number in DESIGN.md and in the real contract tests of WP5 is pinned to this
version, so a mismatch skips the whole directory with a clear message rather than
failing twenty tests obscurely (A01).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import pytest

from dattn.engine.assertions import PINNED_VLLM_VERSION

pytestmark = pytest.mark.contract


def _installed() -> str | None:
    try:
        return version("vllm")
    except PackageNotFoundError:
        return None


def test_vllm_is_installed_and_pinned() -> None:
    got = _installed()
    if got is None:
        pytest.skip("vllm is not installed; contract tests need vllm==0.29.0")
    assert got == PINNED_VLLM_VERSION, (
        f"contract tests are pinned to vllm {PINNED_VLLM_VERSION}, found {got}. "
        "Re-read the line numbers in DESIGN.md before touching anything."
    )


def test_the_removed_attention_backend_env_var_is_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A15: VLLM_ATTENTION_BACKEND does not exist in 0.29.0, so its presence is a stale script."""
    import os

    assert os.environ.get("VLLM_ATTENTION_BACKEND") is None
