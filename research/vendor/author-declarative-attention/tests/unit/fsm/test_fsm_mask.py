"""Mask composition and block rounding (WP1).

The property tests are the ones DESIGN.md 7 asks for: a random mask and a random `now`
against the naive set cardinality, and block rounding against the set it is supposed to
widen to.
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dattn.contracts import INF, MAX_INTERVALS, Mask, PromptLayout
from dattn.fsm.mask import (
    DENSE,
    LOCAL_WINDOW_TOKENS,
    base_mask,
    compose_mask,
    focus_mask,
    local_window,
    round_to_blocks,
)

LAYOUT = PromptLayout(
    n_prompt_tokens=1000,
    question_start=900,
    segments=((20, 120), (200, 300), (400, 500), (600, 700)),
)


# --------------------------------------------------------------------------------------
# Composition
# --------------------------------------------------------------------------------------


def test_base_mask_is_sink_plus_open_tail() -> None:
    assert base_mask(LAYOUT).intervals == ((0, 16), (900, INF))


def test_the_open_tail_subsumes_the_window_and_the_response_so_far() -> None:
    """`question_start < n_prompt_tokens` always, so the two terms are contiguous."""
    mask = base_mask(LAYOUT)
    assert mask.contains(LAYOUT.question_start)
    assert mask.contains(LAYOUT.n_prompt_tokens - 1)
    assert mask.contains(LAYOUT.n_prompt_tokens)  # the response so far
    assert mask.contains(LAYOUT.n_prompt_tokens + 10_000)


def test_focus_mask_adds_exactly_the_cited_spans() -> None:
    assert focus_mask(LAYOUT, (2,)).intervals == ((0, 16), (200, 300), (900, INF))
    assert focus_mask(LAYOUT, (1, 3)).intervals == ((0, 16), (20, 120), (400, 500), (900, INF))


def test_duplicate_ids_contribute_their_area_once() -> None:
    once = focus_mask(LAYOUT, (2,))
    twice = focus_mask(LAYOUT, (2, 2, 2))
    assert once == twice
    assert twice.area(1000) == once.area(1000)


def test_three_non_adjacent_spans_still_fit_in_max_intervals() -> None:
    mask = focus_mask(LAYOUT, (1, 2, 3))
    assert len(mask.intervals) == MAX_INTERVALS
    assert mask.open_ended


def test_a_span_touching_the_sink_merges_rather_than_adding_an_interval() -> None:
    layout = PromptLayout(n_prompt_tokens=100, question_start=80, segments=((16, 30),))
    assert focus_mask(layout, (1,)).intervals == ((0, 30), (80, INF))


def test_compose_mask_with_no_spans_is_base_mask() -> None:
    assert compose_mask(LAYOUT) == base_mask(LAYOUT)


def test_dense_is_the_everything_mask_and_nothing_else_is() -> None:
    assert DENSE.dense
    assert not base_mask(LAYOUT).dense
    assert not Mask.of((0, INF - 1)).dense


# --------------------------------------------------------------------------------------
# Local window detection and its fallback
# --------------------------------------------------------------------------------------


def test_local_window_is_the_question_region() -> None:
    assert local_window(1000, 900) == (900, 1000)


def test_local_window_falls_back_to_the_last_1024_prompt_tokens() -> None:
    """dac's `local_window_range=None` path (`F D6`). Unreachable with the shipped
    templates, kept so a layout recovered without a marker still has a window."""
    assert local_window(5000) == (5000 - LOCAL_WINDOW_TOKENS, 5000)
    assert local_window(100) == (0, 100)


@pytest.mark.parametrize("bad", [0, -1])
def test_local_window_refuses_an_empty_prompt(bad: int) -> None:
    with pytest.raises(ValueError, match="n_prompt_tokens"):
        local_window(bad)


def test_local_window_refuses_a_question_start_past_the_prompt() -> None:
    with pytest.raises(ValueError, match="outside"):
        local_window(100, 100)


# --------------------------------------------------------------------------------------
# Block rounding
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
def test_round_to_blocks_widens_outward_and_keeps_inf(block_size: int) -> None:
    mask = Mask.of((5, 37), (900, INF))
    rounded = round_to_blocks(mask, block_size)
    assert rounded.open_ended
    assert rounded.intervals[-1][1] == INF
    for s, e in rounded.intervals:
        assert s % block_size == 0
        if e < INF:
            assert e % block_size == 0


@pytest.mark.parametrize("bad", [0, -1])
def test_round_to_blocks_refuses_a_non_positive_block(bad: int) -> None:
    with pytest.raises(ValueError, match="block size"):
        round_to_blocks(DENSE, bad)


def test_rounding_collapses_the_recorded_gemma_first_span_lead() -> None:
    """`F D1`: on the layout the reference diff recorded, the two Gemma first spans are
    147 and 150, and rounding at any shipped block size makes them identical."""
    for block_size in (16, 32, 64, 128):
        assert round_to_blocks(Mask.of((147, 2317)), block_size) == round_to_blocks(Mask.of((150, 2317)), block_size)


@pytest.mark.parametrize("block_size", [16, 32, 64, 128])
def test_a_three_token_span_lead_costs_at_most_one_block_after_rounding(block_size: int) -> None:
    """The general form of `F D1`, which holds whether or not the two starts straddle a
    block boundary: `render_golden.json` records 175 against 178, which do straddle one."""
    ours = round_to_blocks(Mask.of((175, 2291)), block_size)
    theirs = round_to_blocks(Mask.of((178, 2291)), block_size)
    assert abs(ours.area(4096) - theirs.area(4096)) <= block_size


# --------------------------------------------------------------------------------------
# Properties
# --------------------------------------------------------------------------------------

_pairs = st.lists(
    st.tuples(st.integers(0, 4000), st.integers(0, 4000)).map(lambda p: (min(p), max(p))),
    max_size=8,
)


def _naive(mask: Mask, now: int) -> set[int]:
    return {p for s, e in mask.intervals for p in range(s, min(e, now))}


@given(pairs=_pairs, now=st.integers(0, 4000))
@settings(max_examples=300, deadline=None)
def test_clip_and_area_agree_with_the_naive_set(pairs: list[tuple[int, int]], now: int) -> None:
    mask = Mask.of(*pairs)
    naive = _naive(mask, now)
    assert mask.area(now) == len(naive)
    assert mask.clip(now).area() == len(naive)
    assert _naive(mask.clip(now), 4000) == naive


@given(pairs=_pairs, block_size=st.sampled_from([1, 2, 16, 32, 64, 128, 784]))
@settings(max_examples=300, deadline=None)
def test_round_to_blocks_is_a_superset_and_is_idempotent(pairs: list[tuple[int, int]], block_size: int) -> None:
    mask = Mask.of(*pairs)
    rounded = round_to_blocks(mask, block_size)
    assert _naive(mask, 8000) <= _naive(rounded, 8000)
    assert rounded.area(8000) >= mask.area(8000)
    assert round_to_blocks(rounded, block_size) == rounded
    for s, e in rounded.intervals:
        assert s % block_size == 0 and e % block_size == 0


@given(pairs=_pairs, block_size=st.sampled_from([16, 32, 64, 128]))
@settings(max_examples=200, deadline=None)
def test_rounding_an_open_ended_mask_keeps_it_open_ended(pairs: list[tuple[int, int]], block_size: int) -> None:
    mask = Mask.of(*pairs, (3000, INF))
    rounded = round_to_blocks(mask, block_size)
    assert rounded.open_ended
    assert rounded.intervals[-1][1] == INF


@given(
    sink=st.integers(0, 64),
    question_start=st.integers(1, 2000),
    n_extra=st.integers(0, 40),
    ids=st.lists(st.integers(1, 3), min_size=0, max_size=3),
)
@settings(max_examples=300, deadline=None)
def test_composition_is_sorted_disjoint_non_touching_and_bounded(
    sink: int, question_start: int, n_extra: int, ids: list[int]
) -> None:
    assume(sink <= question_start)
    # Three well separated segments inside the question region.
    step = max(1, question_start // 8)
    segments = tuple((i * step, i * step + max(1, step // 2)) for i in (1, 3, 5))
    assume(all(s < e <= question_start for s, e in segments))
    layout = PromptLayout(
        n_prompt_tokens=question_start + 1 + n_extra,
        question_start=question_start,
        segments=segments,
        sink=sink,
    )
    mask = focus_mask(layout, ids)
    assert len(mask.intervals) <= MAX_INTERVALS
    for (s1, e1), (s2, _e2) in zip(mask.intervals, mask.intervals[1:], strict=False):
        assert s1 < e1 < s2, "sorted, disjoint and non-touching"
    assert mask.open_ended
    for i in ids:
        s, e = layout.segments[i - 1]
        assert mask.contains(s) and mask.contains(e - 1)
    if sink:
        assert mask.contains(0)
