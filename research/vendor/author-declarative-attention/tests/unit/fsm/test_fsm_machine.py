"""The state machine (WP1): transitions, counters and the sharp edges of `F 5`."""

import math

import pytest

from dattn.contracts import INF, DattnInvariantError, Mask, Mode, PromptLayout
from dattn.fsm.machine import DeclarativeFSM
from dattn.fsm.metrics import METRIC_KEYS

LAYOUT = PromptLayout(
    n_prompt_tokens=1000,
    question_start=900,
    segments=((20, 120), (200, 300), (400, 500)),
)


def drive(text: str, layout: PromptLayout = LAYOUT) -> DeclarativeFSM:
    """Feed `text` one character at a time. One character per token is the harshest
    tokenisation: every `>` lands in its own token, so no tag can hide in a tail."""
    fsm = DeclarativeFSM(layout)
    for char in text:
        fsm.step(1, char)
    return fsm


def modes(text: str, layout: PromptLayout = LAYOUT) -> list[Mode]:
    fsm = DeclarativeFSM(layout)
    seen = []
    for char in text:
        fsm.step(1, char)
        seen.append(fsm.mode)
    return seen


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------


def test_a_fresh_machine_is_global_and_dense() -> None:
    fsm = DeclarativeFSM(LAYOUT)
    assert fsm.mode is Mode.GLOBAL
    assert fsm.mask.dense
    assert fsm.n_seen == LAYOUT.n_prompt_tokens


def test_step_returns_the_new_mask_only_when_it_changed() -> None:
    fsm = DeclarativeFSM(LAYOUT)
    returned = [fsm.step(1, c) for c in '<focus magic_chunks="1">x']
    assert sum(r is not None for r in returned) == 1
    assert returned[-2] is not None  # the `>` token
    assert returned[-1] is None


def test_step_raises_on_a_negative_token_id() -> None:
    """vLLM's async-scheduling placeholder. A silent wrong answer becomes a crash."""
    fsm = DeclarativeFSM(LAYOUT)
    with pytest.raises(DattnInvariantError, match="negative token id"):
        fsm.step(-1, "x")


def test_a_focus_tag_in_the_prompt_tail_fires_on_the_first_generated_token() -> None:
    """`F 5.1`: the scan window is seeded with the decoded prompt tail, so a prompt that
    ends in a tag puts the machine into FOCUS before the model has written anything."""
    layout = PromptLayout(
        n_prompt_tokens=1000,
        question_start=900,
        segments=((20, 120), (200, 300)),
        prompt_tail='filler filler <focus magic_chunks="2"',
    )
    fsm = DeclarativeFSM(layout)
    assert fsm.step(1, ">") is not None
    assert fsm.mode is Mode.FOCUS
    assert fsm.ledger.focus_attempts == 1


# --------------------------------------------------------------------------------------
# Transitions
# --------------------------------------------------------------------------------------


def test_focus_opens_closes_and_returns_to_dense() -> None:
    fsm = DeclarativeFSM(LAYOUT)
    for char in '<focus magic_chunks="2">':
        fsm.step(1, char)
    assert fsm.mode is Mode.FOCUS
    assert fsm.mask.intervals == ((0, 16), (200, 300), (900, INF))
    for char in "v</focus>":
        fsm.step(1, char)
    assert fsm.mode is Mode.GLOBAL
    assert fsm.mask.dense


def test_local_and_answer_are_aliases_in_both_directions() -> None:
    for open_tag, close_tag in [
        ("<local>", "</local>"),
        ("<answer>", "</answer>"),
        ("<local>", "</answer>"),
        ("<answer>", "</local>"),
    ]:
        fsm = drive(f"{open_tag}body{close_tag}")
        assert fsm.mode is Mode.GLOBAL
        assert fsm.ledger.steps_by_mode.get(Mode.LOCAL, 0) > 0


def test_local_mask_is_the_base_mask_with_no_spans() -> None:
    fsm = drive("<local>")
    assert fsm.mode is Mode.LOCAL
    assert fsm.mask.intervals == ((0, 16), (900, INF))


def test_global_tags_are_inert() -> None:
    fsm = drive("<global>reasoning</global> more")
    assert fsm.mode is Mode.GLOBAL
    assert fsm.ledger.focus_attempts == 0
    assert fsm.ledger.steps_by_mode == {Mode.GLOBAL: len("<global>reasoning</global> more")}


def test_a_stray_close_in_global_moves_nothing() -> None:
    for text in ("</focus>", "</local>", "</answer>"):
        fsm = drive(text)
        assert fsm.mode is Mode.GLOBAL
        assert fsm.ledger.focus_attempts == 0
        assert fsm.ledger.focus_successes == 0


def test_nesting_collapses_on_the_inner_close() -> None:
    """`F 5.3`: the inner `</focus>` ends the outer block and the outer close is a stray."""
    fsm = drive('<focus magic_chunks="1">a<focus magic_chunks="2">b</focus>c</focus>d')
    assert fsm.mode is Mode.GLOBAL
    assert fsm.ledger.focus_attempts == 1, "the inner open is not counted"
    assert fsm.ledger.focus_successes == 1


def test_an_unclosed_focus_stays_in_focus_to_the_end() -> None:
    fsm = drive('<focus magic_chunks="1">never closed')
    assert fsm.mode is Mode.FOCUS
    assert fsm.ledger.focus_attempts == 1
    assert fsm.ledger.focus_successes == 0


def test_a_focus_tag_inside_local_is_invisible_even_to_the_counters() -> None:
    """`F 5.4`: LOCAL only ever looks for its own close, so the attempt is never counted
    and `focus_success_rate` over-reports adherence for a model that nests."""
    fsm = drive('<local>a <focus magic_chunks="1">value</focus> b</local>')
    assert fsm.mode is Mode.GLOBAL
    assert fsm.ledger.focus_attempts == 0
    assert fsm.ledger.fail_log == []


def test_a_local_tag_inside_focus_is_ignored() -> None:
    fsm = drive('<focus magic_chunks="1"><local>ignored</local>')
    assert fsm.mode is Mode.FOCUS


def test_a_second_local_open_while_local_is_ignored() -> None:
    fsm = drive("<local>outer<local>inner")
    assert fsm.mode is Mode.LOCAL
    assert len(fsm.ledger.local_areas) == 1


def test_a_declined_focus_leaves_the_machine_in_global_with_the_dense_mask() -> None:
    fsm = drive('<focus magic_chunks="99">x')
    assert fsm.mode is Mode.GLOBAL
    assert fsm.mask.dense
    assert fsm.ledger.focus_attempts == 1
    assert fsm.ledger.fail_log == ["out_of_range_index"]


def test_the_scan_only_runs_on_a_token_carrying_a_close_bracket() -> None:
    """A tag delivered in one token without a `>` in the final token is never seen."""
    fsm = DeclarativeFSM(LAYOUT)
    assert fsm.step(1, '<focus magic_chunks="1">') is not None
    fsm2 = DeclarativeFSM(LAYOUT)
    assert fsm2.step(1, '<focus magic_chunks="1"') is None
    assert fsm2.step(1, "x") is None, "no `>` in this token, so no scan"
    assert fsm2.mode is Mode.GLOBAL


def test_one_token_carrying_the_bracket_plus_nine_characters_hides_the_tag() -> None:
    """`F 5.2`: Qwen3-4B token 69171 is `">Description"`. Both references miss it."""
    fsm = DeclarativeFSM(LAYOUT)
    for chunk in ['<focus magic_chunks="1', '">Description']:
        fsm.step(1, chunk)
    assert fsm.mode is Mode.GLOBAL
    assert fsm.ledger.focus_attempts == 0


def test_the_same_token_with_eight_characters_is_seen() -> None:
    fsm = DeclarativeFSM(LAYOUT)
    for chunk in ['<focus magic_chunks="1', '">Descript']:
        fsm.step(1, chunk)
    assert fsm.mode is Mode.FOCUS


def test_a_tag_that_scrolled_out_of_the_window_no_longer_matches() -> None:
    fsm = drive('<focus magic_chunks="1">' + "x" * 600)
    assert fsm.mode is Mode.FOCUS  # it opened when the `>` arrived
    assert fsm.ledger.focus_attempts == 1


# --------------------------------------------------------------------------------------
# Accounting
# --------------------------------------------------------------------------------------


def test_the_ledger_records_before_the_new_mask_applies() -> None:
    """dac's convention, and the one the published `attention_ratio` numbers use: the
    token that triggers the tag is itself accounted under the OLD mask."""
    text = '<focus magic_chunks="1">'
    fsm = drive(text)
    n = len(text)
    assert fsm.ledger.total_steps == n
    # Every step ran under the dense mask, so attended equals attendable.
    assert sum(fsm.ledger.attended) == fsm.ledger.total_attendable
    assert fsm.ledger.total_attendable == sum(1000 + t for t in range(1, n + 1))


def test_focus_area_is_measured_at_the_transition_not_at_the_end() -> None:
    fsm = drive('<focus magic_chunks="2">')
    # sink 16 + span 100 + tail from 900 to n_seen at the transition.
    n_seen = 1000 + len('<focus magic_chunks="2">') - 1
    assert fsm.ledger.focus_areas == [16 + 100 + (n_seen - 900)]
    assert fsm.ledger.focus_span_areas == [100]


def test_metrics_return_exactly_the_contract_key_set() -> None:
    fsm = drive('<focus magic_chunks="1">v</focus><local>c</local>')
    assert tuple(fsm.metrics()) == METRIC_KEYS


def test_focus_success_is_counted_on_the_close_not_on_the_open() -> None:
    assert drive('<focus magic_chunks="1">v').metrics()["focus_successes"] == 0
    assert drive('<focus magic_chunks="1">v</focus>').metrics()["focus_successes"] == 1


def test_global_focus_and_local_steps_partition_total_steps() -> None:
    m = drive('<focus magic_chunks="1">v</focus> mid <local>c</local> end').metrics()
    assert m["global_steps"] + m["focus_steps"] + m["local_steps"] == m["total_steps"]


def test_duplicate_ids_count_twice_in_mean_segments_per_focus_but_once_in_area() -> None:
    once = drive('<focus magic_chunks="2">')
    twice = drive('<focus magic_chunks="2,2">')
    assert once.ledger.focus_span_areas == twice.ledger.focus_span_areas
    assert once.metrics()["mean_segments_per_focus"] == 1.0
    assert twice.metrics()["mean_segments_per_focus"] == 2.0
    assert twice.metrics()["n_unique_segments_focused"] == 1


def test_an_empty_run_gives_nan_focus_ratio_but_zero_local_ratio() -> None:
    """dac's documented asymmetry, reproduced on purpose (`F 3.5`)."""
    m = DeclarativeFSM(LAYOUT).metrics()
    assert m["total_steps"] == 0
    assert math.isnan(m["focus_ratio"])  # type: ignore[arg-type]
    assert m["local_ratio"] == 0.0
    assert math.isnan(m["focus_success_rate"])  # type: ignore[arg-type]
    assert math.isnan(m["attention_ratio"])  # type: ignore[arg-type]


def test_a_layout_carrying_a_format_error_declines_every_focus_tag() -> None:
    layout = PromptLayout(
        n_prompt_tokens=1000,
        question_start=900,
        segments=(),
        format_error="no_segments_detected",
    )
    m = drive('<focus magic_chunks="1">v</focus>', layout).metrics()
    assert m["segment_format_error"] == 1
    assert m["segment_format_error_reason"] == "no_segments_detected"
    assert m["focus_fail_segment_format_error"] == 1
    assert m["focus_steps"] == 0
    assert m["n_segments"] == 0


def test_attended_never_exceeds_attendable() -> None:
    fsm = drive('<focus magic_chunks="1">v</focus><local>c</local>done')
    for attended, step in zip(fsm.ledger.attended, range(1, fsm.ledger.total_steps + 1), strict=True):
        assert 0 < attended <= 1000 + step


def test_the_mask_is_always_open_ended_so_the_response_stays_visible() -> None:
    fsm = DeclarativeFSM(LAYOUT)
    for char in '<focus magic_chunks="1">v</focus><local>c':
        fsm.step(1, char)
        assert fsm.mask.open_ended
        assert isinstance(fsm.mask, Mask)
