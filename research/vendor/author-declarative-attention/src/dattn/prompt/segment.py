"""Context segmentation.

A five-level delimiter ladder, greedy packing, `bisect_left` token counting, a
fast-tokenizer requirement and `<empty_context>` normalisation. Both reference
implementations agree byte for byte here (`F 3.2`) and match `main.tex:483,2419`.

The context is tokenized once with an offset mapping and everything after that happens in
character-offset space, so every segment's `text` is a real substring of the context, the
char ranges are an exact partition of `[0, len(context))`, and concatenating the segment
texts reproduces the context. There is no `decode` of a token-id slice, hence no U+FFFD.

A piece that already fits under `max_tokens` is kept whole. Only an oversized piece is
descended one level finer, so cuts land on the coarsest boundary the cap allows. A
whitespace-free run (a base64 blob, say) is atomic: it becomes its own over-cap segment
rather than being cut mid-word.
"""

import re
from bisect import bisect_left
from typing import Any, Final

from dattn.contracts import Segment

DELIMITERS: Final[tuple[str, ...]] = (r"(\n\s*\n)", r"(\n)", r"([.!?]+\s+)", r"([;:,]\s+)", r"(\s+)")
EMPTY_CONTEXT: Final[str] = "<empty_context>"

_LEVELS: Final[tuple[re.Pattern[str], ...]] = tuple(re.compile(p) for p in DELIMITERS)

SEGMENTERS: Final[tuple[str, ...]] = ("semantic", "equal_length")

__all__ = ["DELIMITERS", "EMPTY_CONTEXT", "SEGMENTERS", "normalize_context", "segment_context", "segment_equal_length"]


def normalize_context(context: str) -> str:
    """`EMPTY_CONTEXT` for an empty or whitespace-only context, else the context.

    Every prompt then has at least one segment, so the FSM never sees an empty segment map
    for a context that merely happened to be blank.
    """
    return EMPTY_CONTEXT if not context or context.isspace() else context


def segment_context(
    tokenizer: Any, context: str, *, target_tokens: int = 2048, max_tokens: int = 2560
) -> list[Segment]:
    """Split `context` into segments with ids contiguous from 1.

    Invariants, each with a named test:

    * `[]` only for an empty context, otherwise at least one segment, and a context whose
      token length is at most `max_tokens` comes back as exactly one.
    * `char_start` / `char_end` form an exact half-open partition of `[0, len(context))`.
    * `token_count <= max_tokens` for every segment except a single atomic word.
    * ids are the contiguous range `1..len(result)`.
    """
    if max_tokens < target_tokens:
        raise ValueError(f"max_tokens ({max_tokens}) must be >= target_tokens ({target_tokens})")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("segment_context needs a fast tokenizer for offset mapping")
    if not context:
        return []
    starts = [s for s, _ in tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]]
    if not starts:
        return []
    n_chars = len(context)

    def tlen(a: int, b: int) -> int:
        """Tokens in the char span `[a, b)`. A token belongs to the piece its start is in."""
        return bisect_left(starts, b) - bisect_left(starts, a)

    def children(a: int, b: int, rx: re.Pattern[str]) -> list[tuple[int, int]]:
        """Split `[a, b)` at `rx`, keeping each delimiter on the piece that precedes it."""
        text = context[a:b]
        out: list[tuple[int, int]] = []
        prev = 0
        for m in rx.finditer(text):
            if m.end() > prev:
                out.append((a + prev, a + m.end()))
                prev = m.end()
        if prev < len(text):
            out.append((a + prev, b))
        return out

    def split(a: int, b: int, level: int) -> list[tuple[int, int]]:
        if tlen(a, b) <= max_tokens:
            return [(a, b)]
        if level == len(_LEVELS):
            return [(a, b)]  # an indivisible word over the cap stays whole
        kids = children(a, b, _LEVELS[level])
        if len(kids) <= 1:
            return split(a, b, level + 1)  # this delimiter did not split it
        return [span for ca, cb in kids for span in split(ca, cb, level + 1)]

    # Greedy pack toward `target_tokens` with a hard ceiling at `max_tokens`: flush before
    # a piece that would overflow, and flush again once the buffer reaches the target.
    spans: list[tuple[int, int]] = []
    buf_start: int | None = None
    buf_tokens = 0
    for cs, ce in split(0, n_chars, 0):
        n = tlen(cs, ce)
        if buf_start is None:
            buf_start = cs
        elif buf_tokens + n > max_tokens:
            spans.append((buf_start, cs))
            buf_start, buf_tokens = cs, 0
        buf_tokens += n
        if buf_tokens >= target_tokens:
            spans.append((buf_start, ce))
            buf_start, buf_tokens = None, 0
    if buf_start is not None:
        spans.append((buf_start, n_chars))

    return [Segment(i + 1, context[a:b], tlen(a, b), a, b) for i, (a, b) in enumerate(spans)]


def segment_equal_length(tokenizer: Any, context: str, n_segments: int) -> list[Segment]:
    """Cut at token starts ``k * (T // N)``, leaving the remainder in the last segment.

    Tokenize once, never decode. Counts use the semantic segmenter's ownership rule:
    a token belongs to the character span containing its start, not a retokenized slice.
    Empty input returns []. For nonempty input, clamp N to [1, max(1, T)], so N > T
    yields one segment per token when offsets are distinct. Tokenless text stays whole.

    Byte-fallback tokens can share a character start. Such a character cannot be split
    losslessly. Use the requested offsets without snapping to semantic boundaries,
    coalescing coincident cuts and cuts at the endpoints to avoid empty segments.
    In that exceptional case the realized count can be smaller than N and token counts
    can differ from T // N. Callers must report requested and realized segment counts.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("segment_equal_length needs a fast tokenizer for offset mapping")
    if not context:
        return []
    starts = [s for s, _ in tokenizer(context, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]]
    n = max(1, min(n_segments, len(starts)))
    width = len(starts) // n
    cuts = [0]
    for k in range(1, n):
        cut = starts[k * width]
        if cuts[-1] < cut < len(context):
            cuts.append(cut)
    cuts.append(len(context))
    return [
        Segment(i, context[a:b], bisect_left(starts, b) - bisect_left(starts, a), a, b)
        for i, (a, b) in enumerate(zip(cuts, cuts[1:], strict=False), 1)
    ]
