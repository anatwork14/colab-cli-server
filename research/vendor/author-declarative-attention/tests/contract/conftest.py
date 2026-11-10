"""Contract tests: every vLLM internal Lane B touches, pinned to 0.29.0.

The whole directory is skipped with one clear message on any other version, rather than
failing twenty tests obscurely. Each test asserts on behaviour where behaviour is
reachable on a CPU box, and on the installed source where it is not; the expected line
numbers live in the docstrings, never in the assertions, because line numbers rot and
behaviour does not.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import pytest

from dattn.engine.assertions import PINNED_VLLM_VERSION


def _installed() -> str | None:
    try:
        return version("vllm")
    except PackageNotFoundError:
        return None


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    got = _installed()
    if got == PINNED_VLLM_VERSION:
        return
    reason = (
        f"contract tests are pinned to vllm {PINNED_VLLM_VERSION}, found {got}. "
        "Re-read the line numbers in DESIGN.md before touching anything."
    )
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "contract" in item.keywords and item.nodeid.startswith("tests/contract/"):
            item.add_marker(skip)
