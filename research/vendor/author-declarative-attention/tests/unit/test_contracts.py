"""Every WP0 dataclass round-trips through its `to_*` / `from_*` pair (DESIGN.md 7)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from dattn.contracts import (
    INF,
    MAX_IDS_PER_FOCUS,
    MAX_INTERVALS,
    NULL_BLOCK_ID,
    SCAN_CHARS,
    SINK_TOKENS,
    DattnRequest,
    LayoutError,
    Mask,
    Mode,
    PromptLayout,
    StepRecord,
    StepRows,
)

# --------------------------------------------------------------------------------------
# Constants and Mode
# --------------------------------------------------------------------------------------


def test_constants() -> None:
    assert INF == 2**30
    assert MAX_INTERVALS == 5
    assert SINK_TOKENS == 16
    assert MAX_IDS_PER_FOCUS == 3
    assert SCAN_CHARS == 500
    assert NULL_BLOCK_ID == 0
    assert MAX_IDS_PER_FOCUS <= MAX_INTERVALS - 2


def test_mode_values() -> None:
    assert (Mode.GLOBAL, Mode.FOCUS, Mode.LOCAL) == (0, 1, 2)


# --------------------------------------------------------------------------------------
# Mask
# --------------------------------------------------------------------------------------


def test_mask_of_normalises() -> None:
    assert Mask.of((5, 9), (0, 3)).intervals == ((0, 3), (5, 9))
    assert Mask.of((0, 3), (3, 7)).intervals == ((0, 7),)  # touching merges
    assert Mask.of((0, 9), (2, 4)).intervals == ((0, 9),)  # overlapping merges
    assert Mask.of((4, 4)).intervals == ()  # empty dropped


def test_mask_of_rejects_a_negative_start() -> None:
    with pytest.raises(ValueError, match="negative interval start"):
        Mask.of((-1, 4))


def test_mask_area_and_clip() -> None:
    m = Mask.of((0, 16), (100, INF))
    assert m.area(200) == 16 + 100
    assert m.area() == 16 + (INF - 100)
    assert m.clip(120).intervals == ((0, 16), (100, 120))
    assert m.clip(16).intervals == ((0, 16),)


def test_mask_round_to_blocks_keeps_inf() -> None:
    m = Mask.of((3, 17), (100, INF)).round_to_blocks(16)
    assert m.intervals == ((0, 32), (96, INF))
    assert m.open_ended


def test_mask_dense_and_contains() -> None:
    assert Mask.of((0, INF)).dense
    assert not Mask.of((0, 16), (32, INF)).dense
    assert not Mask.of((1, INF)).dense
    assert Mask.of((0, 16)).contains(15)
    assert not Mask.of((0, 16)).contains(16)


def test_mask_union() -> None:
    assert Mask.of((0, 4)).union(Mask.of((4, 9))).intervals == ((0, 9),)


_ivals = st.lists(
    st.tuples(st.integers(0, 4096), st.integers(0, 4096)).map(lambda t: (min(t), max(t))),
    max_size=8,
)


@settings(max_examples=200, deadline=None)
@given(_ivals)
def test_mask_list_round_trip(ivals: list[tuple[int, int]]) -> None:
    m = Mask.of(*ivals)
    assert Mask.from_list(m.to_list()) == m


@settings(max_examples=200, deadline=None)
@given(_ivals, st.integers(0, 4096))
def test_mask_area_matches_the_naive_set(ivals: list[tuple[int, int]], now: int) -> None:
    m = Mask.of(*ivals)
    naive = len({p for s, e in ivals for p in range(s, min(e, now))})
    assert m.clip(now).area() == naive
    assert m.area(now) == naive


# --------------------------------------------------------------------------------------
# PromptLayout
# --------------------------------------------------------------------------------------


def _layout(**kw: object) -> PromptLayout:
    base: dict[str, object] = {"n_prompt_tokens": 1000, "question_start": 900, "segments": ((16, 400), (400, 900))}
    base.update(kw)
    return PromptLayout(**base)  # type: ignore[arg-type]


def test_layout_round_trip() -> None:
    lay = _layout()
    assert PromptLayout.from_dict(lay.to_dict()) == lay


def test_layout_rejects_question_start_past_the_prompt() -> None:
    with pytest.raises(LayoutError):
        _layout(question_start=1000)


def test_layout_rejects_a_sink_past_question_start() -> None:
    with pytest.raises(LayoutError):
        _layout(sink=901)


def test_layout_rejects_more_than_256_segments() -> None:
    with pytest.raises(LayoutError):
        _layout(segments=tuple((i, i + 1) for i in range(257)))


def test_layout_rejects_a_segment_past_question_start() -> None:
    with pytest.raises(LayoutError):
        _layout(segments=((16, 901),))


def test_layout_rejects_overlapping_segments() -> None:
    with pytest.raises(LayoutError):
        _layout(segments=((16, 400), (300, 900)))


def test_layout_rejects_four_ids_per_focus() -> None:
    """`F D8`: sink + tail + focus spans must fit in MAX_INTERVALS."""
    with pytest.raises(LayoutError):
        _layout(max_ids_per_focus=4)


def test_layout_accepts_no_segments() -> None:
    assert _layout(segments=()).segments == ()


# --------------------------------------------------------------------------------------
# DattnRequest
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mask", [None, Mask.of((0, 16), (900, INF))])
def test_request_round_trip(mask: Mask | None) -> None:
    req = DattnRequest(layout=_layout(), enabled=False, forced_mask=mask)
    assert DattnRequest.from_extra_args(req.to_extra_args()) == req


def test_request_extra_args_is_json_serialisable() -> None:
    import json

    payload = {"dattn": DattnRequest(layout=_layout()).to_extra_args()}
    assert json.loads(json.dumps(payload)) == payload


# --------------------------------------------------------------------------------------
# StepRecord
# --------------------------------------------------------------------------------------


def test_step_record_json_round_trip() -> None:
    rec = StepRecord(
        step_id=7,
        engine_step=1201,
        t_sched_ns=10,
        t_ready_ns=20,
        gpu_forward_ms=1.5,
        kind="decode",
        num_reqs=2,
        dattn_block_size=32,
        rows=StepRows(
            req=["a", "b"],
            mode=[int(Mode.FOCUS), int(Mode.LOCAL)],
            seq_len=[100, 200],
            att_blk=[64, 128],
            att_tok=[60, 120],
            out_pos=[3, 4],
            since_adm=[3, 4],
        ),
    )
    assert StepRecord.from_json(rec.to_json()) == rec


def test_step_record_defaults_round_trip() -> None:
    rec = StepRecord()
    assert StepRecord.from_json(rec.to_json()) == rec
