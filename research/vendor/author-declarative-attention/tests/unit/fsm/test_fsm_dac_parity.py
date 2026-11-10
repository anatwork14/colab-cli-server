"""Parity against the dac reference state machine, replayed from `tests/fixtures/fsm`.

Every case in `cases.jsonl` was produced by driving dac's own `DacState` at `fb1ef99` over
a synthetic prompt and one response. This module replays the same response through
`DeclarativeFSM` and asserts that, at every step where dac changed state:

* the mode is the same, and
* the kept key set is the same.

dac's `custom_mask_ranges` are unsorted, unmerged, possibly overlapping and clipped to
`boundary`, with every index at or above `boundary` attended unconditionally. dattn's
`Mask` is the merged open-ended form, so `Mask.of(*ranges, (boundary, INF))` is the
translation, and equality of those two objects is exact set equality.

The metric dict is then compared key by key, with the two documented deltas skipped:
dac has no `global_steps`, and dac's three search-method `focus_fail_*` counters have no
dattn counterpart. `total_attendable` is compared exactly, which is only sound because no
response in the fixture contains a pad token (`F D15`, `P5`).

The replay needs no tokenizer: the fixture carries each response's decoded token texts,
which is exactly what the FSM consumes.
"""

import json
import math
from pathlib import Path
from typing import Any

import pytest

from dattn.contracts import INF, Mask, Mode, PromptLayout
from dattn.fsm.machine import DeclarativeFSM

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "fsm" / "cases.jsonl"

#: dac's metric keys that dattn deliberately does not emit, and the one dattn adds.
DAC_ONLY_KEYS = frozenset(
    {
        "mean_focus_start_marker_length",
        "mean_focus_end_marker_length",
        "focus_fail_marker_shorter_than_three_words",
        "focus_fail_more_than_three_matches",
        "focus_fail_imperfect_pairs",
    }
)
DATTN_ONLY_KEYS = frozenset({"global_steps"})


def _load() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not FIXTURE.is_file():  # pragma: no cover - the fixture is committed
        pytest.skip(f"{FIXTURE} is missing; regenerate it with tests/fixtures/fsm/make.py")
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    return header, [json.loads(line) for line in lines[1:]]


HEADER, CASES = _load()
LAYOUTS = HEADER["layouts_by_id"]


def _layout(case: dict[str, Any]) -> PromptLayout:
    return PromptLayout.from_dict(LAYOUTS[case["layout_id"]])


def _dac_mask(event: dict[str, Any]) -> Mask:
    """dac's state as a dattn `Mask`. GLOBAL carries no mask, which means dense."""
    if event["ranges"] is None:
        return Mask.of((0, INF))
    return Mask.of(*(tuple(r) for r in event["ranges"]), (event["boundary"], INF))


def _ids() -> list[str]:
    return [case["case_id"] for case in CASES]


# --------------------------------------------------------------------------------------
# The fixture itself
# --------------------------------------------------------------------------------------


def test_the_fixture_records_the_dac_commit_it_was_generated_from() -> None:
    assert HEADER["dac_sha"] == "fb1ef99"
    assert HEADER["focus_method"] == "index"
    assert HEADER["segment_format"] == "chunk_multi_turn_tool"
    assert "provenance" in HEADER and "ranges_semantics" in HEADER


def test_the_fixture_covers_at_least_56_responses_across_several_layouts() -> None:
    assert HEADER["n_adversarial"] == 56
    assert len(LAYOUTS) >= 4
    assert len(CASES) >= 56 * len(LAYOUTS)
    assert {len(PromptLayout.from_dict(v).segments) for v in LAYOUTS.values()} >= {0, 1, 5, 41}


def test_the_fixture_exercises_every_mode_and_every_fail_reason() -> None:
    modes = {event["mode"] for case in CASES for event in case["events"]}
    assert modes == {"GLOBAL", "FOCUS", "LOCAL"}
    reasons = {
        key for case in CASES for key, value in case["metrics"].items() if key.startswith("focus_fail_") and value
    }
    assert reasons == {
        "focus_fail_syntax_error",
        "focus_fail_empty_index_list",
        "focus_fail_out_of_range_index",
        "focus_fail_more_than_three_indices",
        "focus_fail_segment_format_error",
    }


# --------------------------------------------------------------------------------------
# The replay
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=_ids())
def test_modes_and_kept_intervals_match_dac(case: dict[str, Any]) -> None:
    fsm = DeclarativeFSM(_layout(case))
    expected = {event["step"]: event for event in case["events"]}
    for step, token_text in enumerate(case["token_texts"]):
        fsm.step(1, token_text)
        event = expected.get(step)
        if event is None:
            continue
        assert fsm.mode.name == event["mode"], f"mode diverged at step {step}"
        assert fsm.mask == _dac_mask(event), f"kept set diverged at step {step}"
    assert fsm.mode.name == case["final_mode"]


@pytest.mark.parametrize("case", CASES, ids=_ids())
def test_state_only_changes_where_dac_changed_it(case: dict[str, Any]) -> None:
    """The converse of the replay: dattn must not transition on a step dac ignored."""
    fsm = DeclarativeFSM(_layout(case))
    changed_at = {event["step"] for event in case["events"]}
    for step, token_text in enumerate(case["token_texts"]):
        new_mask = fsm.step(1, token_text)
        assert (new_mask is not None) == (step in changed_at), f"spurious or missing change at step {step}"


@pytest.mark.parametrize("case", CASES, ids=_ids())
def test_metrics_match_dac(case: dict[str, Any]) -> None:
    fsm = DeclarativeFSM(_layout(case))
    for token_text in case["token_texts"]:
        fsm.step(1, token_text)
    ours = fsm.metrics()
    theirs = case["metrics"]

    assert set(theirs) - DAC_ONLY_KEYS == set(ours) - DATTN_ONLY_KEYS
    for key, expected in theirs.items():
        if key in DAC_ONLY_KEYS:
            continue
        got = ours[key]
        if expected is None:  # the fixture writes NaN as null
            assert isinstance(got, float) and math.isnan(got), f"{key}: expected NaN, got {got!r}"
        elif isinstance(expected, float):
            assert got == pytest.approx(expected, rel=1e-12, abs=0), key
        else:
            assert got == expected, key

    assert ours["global_steps"] == theirs["total_steps"] - theirs["focus_steps"] - theirs["local_steps"]


@pytest.mark.parametrize("case", CASES[::7], ids=_ids()[::7])
def test_the_mask_never_leaves_more_than_max_intervals(case: dict[str, Any]) -> None:
    from dattn.contracts import MAX_INTERVALS

    fsm = DeclarativeFSM(_layout(case))
    for token_text in case["token_texts"]:
        fsm.step(1, token_text)
        assert len(fsm.mask.intervals) <= MAX_INTERVALS
        assert fsm.mask.open_ended


def test_the_no_segment_layout_declines_every_focus_attempt() -> None:
    broken = [c for c in CASES if c["layout_id"] == "qwen_nseg00"]
    assert broken, "the fixture should carry a zero-segment layout"
    for case in broken:
        metrics = case["metrics"]
        assert metrics["segment_format_error"] == 1
        assert metrics["focus_steps"] == 0
        assert metrics["focus_fail_segment_format_error"] == metrics["focus_attempts"]


def test_real_recorded_responses_are_replayed_too() -> None:
    real = [c for c in CASES if c["kind"] == "real"]
    assert len(real) >= 8, "recorded dac responses are part of the parity set"
    assert any(c["metrics"]["focus_steps"] > 0 for c in real)
    assert any(Mode[c["final_mode"]] is Mode.GLOBAL for c in real)
