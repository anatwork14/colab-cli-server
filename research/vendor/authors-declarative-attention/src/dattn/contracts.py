"""Shared contracts for every dattn package.

STDLIB ONLY. This module is importable in an environment with no third-party package
installed, and `tests/unit/test_import_discipline.py` enforces that by walking the AST.

Changes to these shared types must preserve the contracts used across packages.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Final

__all__ = [
    "INF",
    "MAX_IDS_PER_FOCUS",
    "MAX_INTERVALS",
    "NULL_BLOCK_ID",
    "SCAN_CHARS",
    "SINK_TOKENS",
    "BackendProfile",
    "DattnBootError",
    "DattnError",
    "DattnInvariantError",
    "DattnRequest",
    "Interval",
    "Intervals",
    "LayoutError",
    "Mask",
    "Mode",
    "PromptLayout",
    "RenderError",
    "Rendered",
    "Segment",
    "StepRecord",
    "StepRows",
    "Tensor",
]

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

INF: Final[int] = 1 << 30
MAX_INTERVALS: Final[int] = 5
SINK_TOKENS: Final[int] = 16
MAX_IDS_PER_FOCUS: Final[int] = 3
SCAN_CHARS: Final[int] = 500
NULL_BLOCK_ID: Final[int] = 0

# --------------------------------------------------------------------------------------
# Type aliases
# --------------------------------------------------------------------------------------

#: A half-open token interval [start, end).
type Interval = tuple[int, int]
#: Sorted, disjoint, non-touching, non-empty half-open intervals.
type Intervals = tuple[Interval, ...]
#: `torch.Tensor` at runtime. Spelled as an alias so this module stays stdlib only.
type Tensor = Any


class Mode(IntEnum):
    GLOBAL = 0
    FOCUS = 1
    LOCAL = 2


# --------------------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------------------


class DattnError(Exception):
    """Base class for every error dattn raises on purpose."""


# Compatibility spellings used by the unchanged FSM and prompt packages.
LayoutError = RenderError = DattnBootError = DattnInvariantError = DattnError


# --------------------------------------------------------------------------------------
# Mask
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Mask:
    """A set of token positions, as sorted disjoint non-touching half-open intervals.

    The open end is encoded as `INF`. At most `MAX_INTERVALS` intervals survive in
    practice (sink, up to three focus spans, open tail), but `Mask` itself does not
    enforce that bound: `PromptLayout` does, at admission (DESIGN.md section 1.1).
    """

    intervals: Intervals

    # -- construction ------------------------------------------------------------------

    @classmethod
    def of(cls, *ivals: Interval) -> Mask:
        """Normalise: drop empty intervals, sort, merge overlapping and touching ones.

        Raises `ValueError` on a negative start.
        """
        cleaned: list[Interval] = []
        for s, e in ivals:
            s, e = int(s), int(e)
            if s < 0:
                raise ValueError(f"negative interval start: ({s}, {e})")
            if e > s:
                cleaned.append((s, e))
        if not cleaned:
            return cls(())
        cleaned.sort()
        merged: list[Interval] = [cleaned[0]]
        for s, e in cleaned[1:]:
            ls, le = merged[-1]
            if s <= le:  # overlapping or touching
                merged[-1] = (ls, max(le, e))
            else:
                merged.append((s, e))
        return cls(tuple(merged))

    # -- operations --------------------------------------------------------------------

    def union(self, other: Mask) -> Mask:
        return Mask.of(*self.intervals, *other.intervals)

    def clip(self, now: int) -> Mask:
        """Intersect with `[0, now)`."""
        return Mask.of(*((s, min(e, now)) for s, e in self.intervals if s < now))

    def area(self, now: int = INF) -> int:
        """Number of positions below `now` that this mask selects."""
        return sum(min(e, now) - s for s, e in self.intervals if s < now)

    def round_to_blocks(self, bs: int) -> Mask:
        """Round every interval outward to a multiple of `bs`. `INF` stays `INF`."""
        if bs < 1:
            raise ValueError(f"block size must be positive: {bs}")
        return Mask.of(*((s // bs * bs, e if e >= INF else -(-e // bs) * bs) for s, e in self.intervals))

    def contains(self, pos: int) -> bool:
        return any(s <= pos < e for s, e in self.intervals)

    def to_list(self) -> list[list[int]]:
        return [[s, e] for s, e in self.intervals]

    @classmethod
    def from_list(cls, rows: Sequence[Sequence[int]]) -> Mask:
        return cls.of(*((int(r[0]), int(r[1])) for r in rows))

    # -- properties --------------------------------------------------------------------

    @property
    def dense(self) -> bool:
        """True iff this mask is `(0, >= INF)` and nothing else (DESIGN.md section 4.2)."""
        return len(self.intervals) == 1 and self.intervals[0][0] == 0 and self.intervals[0][1] >= INF

    @property
    def open_ended(self) -> bool:
        return bool(self.intervals) and self.intervals[-1][1] >= INF


# --------------------------------------------------------------------------------------
# Prompt side
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Segment:
    id: int
    text: str
    token_count: int
    char_start: int
    char_end: int


@dataclass(frozen=True, slots=True)
class PromptLayout:
    """Everything the engine needs about one request's prompt, computed on the client.

    The engine never re-derives a layout: it round-trips through
    `SamplingParams.extra_args["dattn"]` (docs/DESIGN.md section 2).
    """

    n_prompt_tokens: int
    question_start: int
    segments: Intervals = ()
    sink: int = SINK_TOKENS
    tag_keyword: str = "magic_chunks"
    prompt_format: str = "chunk_multi_turn_tool"
    prompt_tail: str = ""
    max_ids_per_focus: int = MAX_IDS_PER_FOCUS
    format_error: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", tuple((int(s), int(e)) for s, e in self.segments))
        if not 0 <= self.sink <= self.question_start < self.n_prompt_tokens:
            raise LayoutError(
                "require 0 <= sink <= question_start < n_prompt_tokens, got "
                f"sink={self.sink}, question_start={self.question_start}, "
                f"n_prompt_tokens={self.n_prompt_tokens}"
            )
        if len(self.segments) > 256:
            raise LayoutError(f"at most 256 segments, got {len(self.segments)}")
        prev_end = 0
        for i, (s, e) in enumerate(self.segments):
            if not prev_end <= s < e <= self.question_start:
                raise LayoutError(
                    f"segment {i} = ({s}, {e}) violates prev_end <= s < e <= question_start "
                    f"(prev_end={prev_end}, question_start={self.question_start})"
                )
            prev_end = e
        # Prevent a request-controlled engine crash: sink + tail + focus spans
        # must fit in MAX_INTERVALS.
        if not 0 <= self.max_ids_per_focus <= MAX_INTERVALS - 2:
            raise LayoutError(f"max_ids_per_focus must be in [0, {MAX_INTERVALS - 2}], got {self.max_ids_per_focus}")

    def to_dict(self) -> dict[str, object]:
        return {
            "n_prompt_tokens": self.n_prompt_tokens,
            "question_start": self.question_start,
            "segments": [[s, e] for s, e in self.segments],
            "sink": self.sink,
            "tag_keyword": self.tag_keyword,
            "prompt_format": self.prompt_format,
            "prompt_tail": self.prompt_tail,
            "max_ids_per_focus": self.max_ids_per_focus,
            "format_error": self.format_error,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> PromptLayout:
        raw = d.get("segments") or ()
        segments = tuple((int(r[0]), int(r[1])) for r in raw)  # type: ignore[index, union-attr]
        return cls(
            n_prompt_tokens=int(d["n_prompt_tokens"]),  # type: ignore[arg-type]
            question_start=int(d["question_start"]),  # type: ignore[arg-type]
            segments=segments,
            sink=int(d.get("sink", SINK_TOKENS)),  # type: ignore[arg-type]
            tag_keyword=str(d.get("tag_keyword", "magic_chunks")),
            prompt_format=str(d.get("prompt_format", "chunk_multi_turn_tool")),
            prompt_tail=str(d.get("prompt_tail", "")),
            max_ids_per_focus=int(d.get("max_ids_per_focus", MAX_IDS_PER_FOCUS)),  # type: ignore[arg-type]
            format_error=str(d.get("format_error", "")),
        )


@dataclass(frozen=True, slots=True)
class Rendered:
    text: str
    ids: tuple[int, ...]
    layout: PromptLayout
    messages: tuple[Mapping[str, object], ...]
    tools: tuple[Mapping[str, object], ...] | None
    segments: tuple[Segment, ...]
    format_error: str
    n_semantic_segments: int | None = None


# --------------------------------------------------------------------------------------
# Engine side
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class BackendProfile:
    """Engine kwargs and native planner choice for one model/use."""

    model_id: str
    capability: str
    use: str
    base_backend: str
    dattn_block_size: int
    needs_compacted_seq_lens_cpu: bool
    engine_args: Mapping[str, object]
    env: Mapping[str, str]
    bench: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class DattnRequest:
    """The per-request payload carried in `SamplingParams.extra_args["dattn"]`."""

    layout: PromptLayout
    enabled: bool = True
    forced_mask: Mask | None = None

    def to_extra_args(self) -> dict[str, object]:
        """The value that belongs under `SamplingParams.extra_args["dattn"]`."""
        return {
            "layout": self.layout.to_dict(),
            "enabled": self.enabled,
            "forced_mask": None if self.forced_mask is None else self.forced_mask.to_list(),
        }

    @classmethod
    def from_extra_args(cls, d: Mapping[str, object]) -> DattnRequest:
        fm = d.get("forced_mask")
        return cls(
            layout=PromptLayout.from_dict(d["layout"]),  # type: ignore[arg-type]
            enabled=bool(d.get("enabled", True)),
            forced_mask=None if fm is None else Mask.from_list(fm),  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


@dataclass(slots=True)
class StepRows:
    """Parallel fixed-width arrays indexed by batch slot, scheduled rows only."""

    req: list[str] = field(default_factory=list)
    mode: list[int] = field(default_factory=list)
    seq_len: list[int] = field(default_factory=list)
    att_blk: list[int] = field(default_factory=list)
    att_tok: list[int] = field(default_factory=list)
    out_pos: list[int] = field(default_factory=list)
    since_adm: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "req": list(self.req),
            "mode": [int(m) for m in self.mode],
            "seq_len": [int(v) for v in self.seq_len],
            "att_blk": [int(v) for v in self.att_blk],
            "att_tok": [int(v) for v in self.att_tok],
            "out_pos": [int(v) for v in self.out_pos],
            "since_adm": [int(v) for v in self.since_adm],
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> StepRows:
        return cls(
            req=[str(x) for x in d.get("req", ())],  # type: ignore[union-attr]
            mode=[int(x) for x in d.get("mode", ())],  # type: ignore[union-attr]
            seq_len=[int(x) for x in d.get("seq_len", ())],  # type: ignore[union-attr]
            att_blk=[int(x) for x in d.get("att_blk", ())],  # type: ignore[union-attr]
            att_tok=[int(x) for x in d.get("att_tok", ())],  # type: ignore[union-attr]
            out_pos=[int(x) for x in d.get("out_pos", ())],  # type: ignore[union-attr]
            since_adm=[int(x) for x in d.get("since_adm", ())],  # type: ignore[union-attr]
        )


@dataclass(slots=True)
class StepRecord:
    """One engine iteration. Plain scalars and lists only: no numpy, no torch.

    Exactly the fields of DESIGN.md section 5.2, plus the two flags section 5.3 and
    section 5.4 require the recorder to carry (`drained_early`, `host_bracket_valid`).
    """

    step_id: int = 0
    engine_step: int = 0
    t_sched_ns: int = 0
    t_ready_ns: int = 0
    gpu_forward_ms: float = 0.0
    gpu_sample_ms: float = 0.0
    lp_host_us: float = 0.0
    compact_host_us: float = 0.0
    #: The device time of the block-table rewrite, from its own CUDA-event pair. It is a
    #: DECOMPOSITION of the host bracket `(t_ready_ns - t_sched_ns)`, which already
    #: contains it, so nothing may add it to `gpu_forward_ms + gpu_sample_ms`: that would
    #: double count the same work (main.tex:843, owner decision #11).
    gpu_compact_ms: float = 0.0
    kind: str = "decode"
    num_reqs: int = 0
    num_reqs_padded: int = 0
    cudagraph_replayed: bool = False
    cudagraph_mode: str = ""
    runner_generation: str = "V1"
    dattn_block_size: int = 0
    num_new_reqs: int = 0
    num_ctx_tokens: int = 0
    num_gen_tokens: int = 0
    num_preempted: int = 0
    kv_usage: float = 0.0
    n_full_attn_groups: int = 0
    n_compacted_groups: int = 0
    fsm_lag: int = 1
    host_bracket_valid: bool = True
    drained_early: bool = False
    rows: StepRows = field(default_factory=StepRows)

    def to_dict(self) -> dict[str, object]:
        d: dict[str, object] = {}
        for name in self.__slots__:
            if name == "rows":
                continue
            d[name] = getattr(self, name)
        d["rows"] = self.rows.to_dict()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> StepRecord:
        kw = {k: v for k, v in d.items() if k in cls.__slots__ and k != "rows"}
        return cls(rows=StepRows.from_dict(d.get("rows") or {}), **kw)  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, s: str) -> StepRecord:
        return cls.from_dict(json.loads(s))
