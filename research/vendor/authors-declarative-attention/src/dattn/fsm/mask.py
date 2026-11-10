"""Mask operations and the FSM-facing mask compositions.

`Mask`, `INF` and `MAX_INTERVALS` live in `dattn.contracts` because the engine, metrics
and eval packages depend on them too (docs/DESIGN.md section 1). This module re-exports them and owns the
derivations the state machine needs.

The composition is the paper's, `main.tex:1354-1358`:

* a fixed 16-token attention sink at the head of the prompt,
* a local window that runs from the question through the end of the prompt,
* the entire response so far,
* in FOCUS, the token spans of the named segments.

The last two terms collapse into one open-ended interval `(question_start, INF)`, because
`question_start < n_prompt_tokens` always holds, so the window and the response so far are
contiguous. That is why `MAX_INTERVALS` is 5 and not 6: sink, at most three focus spans,
and one open tail.

Block rounding is a free function rather than a method so that engine can call exactly the
same pure code the property tests pin, with no `Mask` instance in the way.
"""

from collections.abc import Iterable
from typing import Final

from dattn.contracts import INF, MAX_INTERVALS, Interval, Mask, PromptLayout

#: The mask that admits everything. Returned when either tag closes (DESIGN.md 1.1).
DENSE: Mask = Mask.of((0, INF))

#: The empty mask. Never handed to the engine, but it is what `compose_mask` degenerates
#: to for a layout with no sink and no tail, and the property tests exercise it.
EMPTY: Mask = Mask.of()

#: dac's `DAC_LOCAL_WINDOW_SIZE`. Used only by `local_window` when a layout carries no
#: question marker, which the shipped templates never produce (`F D6`).
LOCAL_WINDOW_TOKENS: Final[int] = 1024

__all__ = [
    "DENSE",
    "EMPTY",
    "INF",
    "LOCAL_WINDOW_TOKENS",
    "MAX_INTERVALS",
    "Mask",
    "base_mask",
    "compose_mask",
    "focus_mask",
    "local_window",
    "round_to_blocks",
]


def local_window(n_prompt_tokens: int, question_start: int | None = None) -> Interval:
    """The local window, `[question_start, n_prompt_tokens)`.

    `question_start` is the first prompt token at or after the last occurrence of the
    template's question marker, and `dattn.prompt.render` always finds one. When a layout
    was recovered without a marker, fall back to the last `LOCAL_WINDOW_TOKENS` prompt
    tokens, which is what dac does with `local_window_range=None` (`F D6`).

    Raises `ValueError` on a non-positive `n_prompt_tokens`, because there is no window in
    an empty prompt and a silent `(0, 0)` would be read as "sink only".
    """
    if n_prompt_tokens <= 0:
        raise ValueError(f"n_prompt_tokens must be positive: {n_prompt_tokens}")
    if question_start is None:
        return (max(0, n_prompt_tokens - LOCAL_WINDOW_TOKENS), n_prompt_tokens)
    if not 0 <= question_start < n_prompt_tokens:
        raise ValueError(f"question_start {question_start} outside [0, {n_prompt_tokens})")
    return (question_start, n_prompt_tokens)


def compose_mask(layout: PromptLayout, spans: Iterable[Interval] = ()) -> Mask:
    """Sink, local window, response so far and `spans`, as one normalised mask.

    The local window and the response so far are written as the single open interval
    `(question_start, INF)`: the window ends exactly where the response begins, and `Mask.of`
    would merge them anyway. Keeping them as one interval is what holds the FOCUS mask to
    `MAX_INTERVALS` when three non-adjacent segments are cited.
    """
    window_start, _ = local_window(layout.n_prompt_tokens, layout.question_start)
    return Mask.of((0, layout.sink), *spans, (window_start, INF))


def base_mask(layout: PromptLayout) -> Mask:
    """`Mask.of((0, sink), (question_start, INF))` -- the GLOBAL-fallback and LOCAL mask.

    The open tail subsumes both the paper's "local window" and its "response so far"
    terms, because `question_start < n_prompt_tokens` always (`F 3.4`).
    """
    return compose_mask(layout)


def focus_mask(layout: PromptLayout, ids: Iterable[int]) -> Mask:
    """The FOCUS mask: `base_mask(layout)` plus the spans of the cited segment ids.

    Ids are 1-based and must already have passed `dattn.fsm.tags.parse_focus_ids`, so they
    are in `1..len(layout.segments)`. Duplicates are accepted and contribute their area
    once, which is what both references do (`F 3.3`).
    """
    return compose_mask(layout, (layout.segments[i - 1] for i in ids))


def round_to_blocks(mask: Mask, block_size: int) -> Mask:
    """Widen every interval outward to a multiple of `block_size`. `INF` stays `INF`.

    Pure, total on a positive `block_size`, and a superset of `mask` by construction. This
    is the function engine's compaction plan mirrors on the device (DESIGN.md 4.2), and the
    property tests here are what pin it (`main.tex:1372-1375`).
    """
    if block_size < 1:
        raise ValueError(f"block size must be positive: {block_size}")
    return mask.round_to_blocks(block_size)
