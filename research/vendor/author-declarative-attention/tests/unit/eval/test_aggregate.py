"""Accuracy weighting and strict response/judgment joins."""

import math
from pathlib import Path

import pytest

from dattn.eval.aggregate import Cell, Outcome, aggregate, load_cell

from .conftest import judgment_row, response_row, write_cell


def cell(*specs: tuple[str, int, int]) -> Cell:
    """`(subset, n_rows, n_correct)` triples -> a Cell with those outcomes."""
    outcomes = []
    for subset, n, n_correct in specs:
        for i in range(n):
            outcomes.append(
                Outcome(id=f"{subset}#{i}", subset=subset, correct=i < n_correct, format_correct=True, judged=True)
            )
    return Cell(model="m", arm="a", outcomes=outcomes)


def test_macro_is_unweighted_over_subsets_and_micro_is_pooled() -> None:
    """A big saturated subset and a small hard one: the two numbers must disagree."""
    c = cell(("big", 128, 128), ("small", 18, 9))
    agg = aggregate(c)
    assert agg.macro == pytest.approx((1.0 + 0.5) / 2)
    assert agg.micro == pytest.approx(137 / 146)
    assert agg.micro > agg.macro
    assert agg.n == 146
    assert agg.n_subsets == 2


def test_format_failures_and_unjudged_rows_stay_in_the_denominator() -> None:
    """The published recipe. `accuracy <= format_rate` holds by construction."""
    c = Cell(
        model="m",
        arm="a",
        outcomes=[
            Outcome("s#0", "s", True, True, True),
            Outcome("s#1", "s", None, False, False),  # format failure, never judged
            Outcome("s#2", "s", None, True, True),  # judged, unparsable
            Outcome("s#3", "s", False, True, True),
        ],
    )
    agg = aggregate(c)
    assert agg.n == 4
    assert agg.macro == pytest.approx(0.25)
    assert agg.micro == pytest.approx(0.25)
    assert agg.format_rate == pytest.approx(0.75)
    assert agg.n_unparsable == 1
    assert agg.n_format_fail == 1
    assert agg.macro <= agg.format_rate


def test_an_empty_subset_does_not_poison_the_macro() -> None:
    assert math.isnan(aggregate(Cell("m", "a", [])).macro)


def test_load_cell_joins_on_the_dac_key(tmp_path: Path) -> None:
    root = write_cell(
        tmp_path / "cell",
        [response_row("s", 0), response_row("s", 1, format_correct=False, parsed_answer="<invalid_response>")],
        [judgment_row("s", 0, True)],
    )
    c = load_cell(root / "responses.jsonl", root / "judgments.jsonl", model="m", arm="a")
    assert [o.id for o in c.outcomes] == ["s#0", "s#1"]
    assert c.outcomes[0].correct is True and c.outcomes[0].judged
    assert c.outcomes[1].correct is None and not c.outcomes[1].judged
    assert aggregate(c).macro == pytest.approx(0.5)


def test_an_orphan_judgment_is_an_error_not_a_free_extra_row(tmp_path: Path) -> None:
    """dac's figure scripts count judgments with no join, which lets accuracy exceed 100%."""
    js = [judgment_row("s", 0, True), judgment_row("s", 9, True)]
    root = write_cell(tmp_path / "cell", [response_row("s", 0)], js)
    with pytest.raises(ValueError, match="no response"):
        load_cell(root / "responses.jsonl", root / "judgments.jsonl")


def test_a_duplicate_judgment_is_an_error(tmp_path: Path) -> None:
    js = [judgment_row("s", 0, True), judgment_row("s", 0, False)]
    root = write_cell(tmp_path / "cell", [response_row("s", 0)], js)
    with pytest.raises(ValueError, match="duplicate judgment"):
        load_cell(root / "responses.jsonl", root / "judgments.jsonl")


def test_transport_events_do_not_double_count_a_retried_answer(tmp_path):
    root = write_cell(
        tmp_path,
        [
            response_row("s", 0, error="connection failed", transport_error=True),
            response_row("s", 0),
        ],
        [judgment_row("s", 0, True)],
    )
    result = aggregate(load_cell(root / "responses.jsonl", root / "judgments.jsonl"))
    assert result.n == result.n_correct == result.n_errored == 1
    assert result.micro == result.macro == 1


def test_duplicate_successful_responses_are_refused(tmp_path):
    root = write_cell(tmp_path, [response_row("s", 0)] * 2)
    with pytest.raises(ValueError, match="duplicate response"):
        load_cell(root / "responses.jsonl")
