"""Persisted rollout records, strict JSONL and reproducible run metadata."""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ResponseRecord", "RunMeta", "iter_jsonl", "json_safe", "write_jsonl"]


def json_safe(obj: Any) -> Any:
    """Convert nonfinite floats to null recursively before strict JSON encoding."""
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, Mapping):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [json_safe(v) for v in obj]
    return obj


@dataclass(slots=True)
class ResponseRecord:
    """One rollout. `(subset, row_index, response_index)` is the key, as in dac."""

    # -- identity (dac) ----------------------------------------------------------------
    subset: str
    row_index: int
    response_index: int = 0

    # -- the dataset row (dac) ---------------------------------------------------------
    question: str = ""
    answer: str = ""
    rubric: str = ""
    raw_context_tokens: int | None = None

    # -- the prompt (dac has only `rendered_prompt_tokens` and `n_bookmarks`) ----------
    rendered_prompt_tokens: int | None = None
    n_bookmarks: int | None = None
    n_semantic_segments: int | None = None
    segmenter: str = "semantic"
    prompt_sha256: str = ""
    prompt_format: str = ""
    format_error: str = ""

    # -- the rollout (dac) -------------------------------------------------------------
    response: str = ""
    parsed_answer: str = ""
    format_correct: bool = False

    # -- what dac lost -----------------------------------------------------------------
    finish_reason: str = ""
    stop_reason: str | int | None = None
    n_generated_tokens: int | None = None
    answer_policy: str = "last"
    n_answer_blocks: int = 0
    first_answer: str | None = None
    #: The text the OTHER answer policy would have graded, so one run prices
    #: `--last-answer` without a second pass. `first_answer != last_answer` is
    #: `Extraction.policies_differ` persisted. Defaults to None so a row written
    #: before this field existed still loads.
    last_answer: str | None = None
    error: str = ""
    transport_error: bool = False

    # -- the declarative-attention trace ----------------------------------------------
    dac_metrics: dict[str, Any] | None = None
    mode_trace: list[int] = field(default_factory=list)
    attended_tokens_exact: int | None = None
    attended_tokens_blocks: int | None = None
    total_attendable: int | None = None

    @property
    def id(self) -> str:
        return f"{self.subset}#{self.row_index}"

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.subset, self.row_index, self.response_index)

    @property
    def truncated(self) -> bool:
        """`finish_reason == "length"`. False when the field is dac's hard-coded "stop"."""
        return self.finish_reason == "length"

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> ResponseRecord:
        """Load a dattn record without raw-row coercions or discarded unknown fields."""
        row = cls(**d)
        if any(not isinstance(getattr(row, k), str) for k in ("subset", "response", "rubric", "parsed_answer")):
            raise ValueError("record text fields must be strings")
        return row


@dataclass(slots=True)
class RunMeta:
    """The header of an output tree. Everything needed to re-run it, in one file."""

    model_id: str
    arm: str
    created: str = ""
    dattn_version: str = ""
    vllm_version: str = ""
    transformers_version: str = ""
    datasets_version: str = ""
    tokenizer_revision: str = ""
    selection: dict[str, Any] = field(default_factory=dict)
    rollout: dict[str, Any] = field(default_factory=dict)
    rendering: dict[str, str] = field(default_factory=dict)
    judge: dict[str, Any] = field(default_factory=dict)
    profile: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RunMeta:
        kw = {k: v for k, v in d.items() if k in set(cls.__slots__)}
        return cls(**kw)  # type: ignore[arg-type]

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Stream JSONL, skipping blank lines and refusing malformed records."""
    with path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno} is not valid JSON") from exc


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", buffering=1) as fh:
        for row in rows:
            fh.write(json.dumps(json_safe(row), separators=(",", ":"), allow_nan=False) + "\n")
