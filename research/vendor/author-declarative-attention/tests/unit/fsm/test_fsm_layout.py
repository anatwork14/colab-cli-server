"""Layout admission (WP1). The counts DESIGN.md 7 names: 0, 1, 3, 40, 255, 256, 257."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dattn.contracts import INF, MAX_INTERVALS, LayoutError, PromptLayout
from dattn.fsm.layout import MAX_SEGMENTS, validate_layout


def segments(n: int, width: int = 3) -> tuple[tuple[int, int], ...]:
    return tuple((i * width, i * width + width - 1) for i in range(n))


def layout_with(n: int) -> PromptLayout:
    q = max(16, n * 3)
    return PromptLayout(n_prompt_tokens=q + 1, question_start=q, segments=segments(n))


@pytest.mark.parametrize("n", [0, 1, 3, 40, 255, 256])
def test_segment_counts_up_to_the_cap_are_admissible(n: int) -> None:
    layout = layout_with(n)
    assert len(layout.segments) == n
    validate_layout(layout, max_model_len=8192)


def test_two_hundred_and_fifty_seven_segments_is_refused() -> None:
    with pytest.raises(LayoutError, match="at most 256 segments"):
        layout_with(257)
    assert MAX_SEGMENTS == 256


def test_question_start_before_the_sink_is_refused() -> None:
    with pytest.raises(LayoutError, match="sink <= question_start"):
        PromptLayout(n_prompt_tokens=100, question_start=8, sink=16)


def test_question_start_at_or_past_the_prompt_end_is_refused() -> None:
    with pytest.raises(LayoutError, match="question_start < n_prompt_tokens"):
        PromptLayout(n_prompt_tokens=100, question_start=100)


def test_a_segment_ending_past_question_start_is_refused() -> None:
    with pytest.raises(LayoutError, match="violates"):
        PromptLayout(n_prompt_tokens=100, question_start=50, segments=((20, 60),))


def test_overlapping_or_unsorted_segments_are_refused() -> None:
    with pytest.raises(LayoutError, match="violates"):
        PromptLayout(n_prompt_tokens=100, question_start=90, segments=((20, 40), (30, 50)))
    with pytest.raises(LayoutError, match="violates"):
        PromptLayout(n_prompt_tokens=100, question_start=90, segments=((30, 50), (10, 20)))


def test_an_empty_segment_is_refused() -> None:
    with pytest.raises(LayoutError, match="violates"):
        PromptLayout(n_prompt_tokens=100, question_start=90, segments=((20, 20),))


def test_max_ids_per_focus_of_four_is_refused() -> None:
    """`F D8`: a request-controlled 4 produces a six-interval mask and crashes the engine
    core in the other clean-room implementation. Here it never gets past admission."""
    assert MAX_INTERVALS - 2 == 3
    with pytest.raises(LayoutError, match="max_ids_per_focus"):
        PromptLayout(n_prompt_tokens=100, question_start=90, max_ids_per_focus=4)
    with pytest.raises(LayoutError, match="max_ids_per_focus"):
        PromptLayout(n_prompt_tokens=100, question_start=90, max_ids_per_focus=-1)


def test_validate_layout_refuses_a_prompt_longer_than_max_model_len() -> None:
    with pytest.raises(LayoutError, match="exceeds max_model_len"):
        validate_layout(layout_with(3), max_model_len=8)


def test_validate_layout_refuses_a_max_model_len_at_or_above_inf() -> None:
    """A19. A position at or above `INF` is indistinguishable from the open end, and a
    `(0, e >= INF)` mask would be misclassified as dense."""
    with pytest.raises(LayoutError, match="indistinguishable"):
        validate_layout(layout_with(1), max_model_len=INF)


@pytest.mark.parametrize("bad", [0, -1])
def test_validate_layout_refuses_a_non_positive_max_model_len(bad: int) -> None:
    with pytest.raises(LayoutError, match="must be positive"):
        validate_layout(layout_with(1), max_model_len=bad)


@given(
    n_segments=st.integers(0, 6),
    sink=st.integers(0, 16),
    tail=st.integers(1, 100),
    keyword=st.sampled_from(["magic_chunks", "chunks", ""]),
    error=st.sampled_from(["", "no_segments_detected"]),
)
@settings(max_examples=200, deadline=None)
def test_layout_round_trips_through_extra_args(n_segments: int, sink: int, tail: int, keyword: str, error: str) -> None:
    """`from_dict(to_dict(x)) == x` after a JSON round trip, which is what carries the
    layout through `SamplingParams.extra_args` without the engine re-deriving it."""
    import json

    q = max(sink, n_segments * 3)
    layout = PromptLayout(
        n_prompt_tokens=q + tail,
        question_start=q,
        segments=segments(n_segments),
        sink=sink,
        tag_keyword=keyword,
        prompt_tail="tail text",
        format_error=error,
    )
    assert PromptLayout.from_dict(json.loads(json.dumps(layout.to_dict()))) == layout
