"""Regenerate the test fixtures. Fixtures are REGENERATED, never copied (PROVENANCE.md).

Reads dac's recorded `responses.jsonl` (data only) and writes `cases.jsonl` with a header
recording the dac sha, the tokenizer revisions and the generation date. No test file, no
fixture file and no source file is copied from either reference repository.

Owned by WP1 (`cases.jsonl`) and WP2 (`tool_schema.json`).
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent

CASES = HERE / "cases.jsonl"
TOOL_SCHEMA = HERE / "tool_schema.json"


def main() -> int:
    raise NotImplementedError("WP1/WP2: tests/unit/fixtures/make.py")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
