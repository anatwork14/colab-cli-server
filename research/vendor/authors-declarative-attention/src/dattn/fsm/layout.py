"""PromptLayout validation.

`PromptLayout` itself, and its structural `__post_init__` invariants, are in
`dattn.contracts`: `0 <= sink <= question_start < n_prompt_tokens`, at most 256 segments,
segments increasing and disjoint and ending at or before `question_start`, and
`max_ids_per_focus <= MAX_INTERVALS - 2` (which prevents a request-controlled
engine-core crash).

This module owns the checks that need a serving fact the dataclass cannot see, namely
`max_model_len`.
"""

from dattn.contracts import (
    INF,
    MAX_IDS_PER_FOCUS,
    MAX_INTERVALS,
    SINK_TOKENS,
    LayoutError,
    PromptLayout,
    Segment,
)

#: Hard cap on segments, enforced by `PromptLayout.__post_init__`. Restated here so the
#: prompt side can refuse a context before it builds a layout it cannot ship.
MAX_SEGMENTS: int = 256

__all__ = [
    "MAX_IDS_PER_FOCUS",
    "MAX_INTERVALS",
    "MAX_SEGMENTS",
    "SINK_TOKENS",
    "PromptLayout",
    "Segment",
    "validate_layout",
]


def validate_layout(layout: PromptLayout, *, max_model_len: int) -> None:
    """Raise `LayoutError` unless `layout` is admissible at this `max_model_len`.

    Also asserts `max_model_len < INF` (DESIGN.md section 1.1, A19): a position at or
    above `INF` is indistinguishable from the open end, and a `(0, e >= INF)` mask would
    be misclassified as dense.
    """
    if max_model_len < 1:
        raise LayoutError(f"max_model_len must be positive, got {max_model_len}")
    if max_model_len >= INF:
        raise LayoutError(
            f"max_model_len {max_model_len} >= INF {INF}: positions at or above INF are "
            "indistinguishable from the open interval end"
        )
    if layout.n_prompt_tokens > max_model_len:
        raise LayoutError(f"prompt of {layout.n_prompt_tokens} tokens exceeds max_model_len {max_model_len}")
    # Restated rather than trusted: `from_dict` on a hand-written extra_args payload runs
    # `__post_init__`, but a caller can still hand us a mutated tuple through `dataclasses`.
    if len(layout.segments) > MAX_SEGMENTS:
        raise LayoutError(f"at most {MAX_SEGMENTS} segments, got {len(layout.segments)}")
    if not 0 <= layout.max_ids_per_focus <= MAX_INTERVALS - 2:
        raise LayoutError(f"max_ids_per_focus must be in [0, {MAX_INTERVALS - 2}], got {layout.max_ids_per_focus}")
