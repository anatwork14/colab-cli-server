"""`run_rollouts` fills the declarative-attention trace on every row (review blocker 3).

`dattn eval` recorded none of `dac_metrics`, `mode_trace`, `attended_tokens_exact`,
`attended_tokens_blocks` or `total_attendable` before this: `eval/replay.py` could compute
them and nothing called it. dac writes 29 metric keys per row and the paper's
attended-tokens and decode-steps headline numbers come from them.

No server and no tokenizer: the client is faked, the renderer is injected, and the
tokenizer is one character per token, which is enough to drive the tag scanner.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeClient
from fakes import FakeTokenizer as CharTokenizer

from dattn.contracts import PromptLayout, Rendered
from dattn.eval import runner as run_mod
from dattn.eval.datasets import EvalRow
from dattn.eval.runner import RunPaths, run_rollouts
from dattn.eval.sampling import RolloutConfig
from dattn.eval.schema import ResponseRecord

LAYOUT = PromptLayout(n_prompt_tokens=100, question_start=80, segments=((16, 40), (40, 60)))
DA_RESPONSE = '<focus magic_chunks="1">x</focus><answer>4</answer>'
VANILLA_RESPONSE = "just an answer"


def _renderer(tokenizer: Any, **kw: Any) -> Rendered:
    del tokenizer, kw
    return Rendered(
        text="PROMPT",
        ids=tuple(range(100)),
        layout=LAYOUT,
        messages=(),
        tools=None,
        segments=(object(), object()),  # type: ignore[arg-type]
        format_error="",
    )


@pytest.fixture
def rows() -> list[EvalRow]:
    return [
        EvalRow(subset="s", row_index=i, split="test", context="c", question="q", answer="4", rubric="r")
        for i in range(3)
    ]


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[EvalRow],
    *,
    response: str = DA_RESPONSE,
    errors: frozenset[int] = frozenset(),
    **kw: Any,
) -> list[ResponseRecord]:
    import importlib

    monkeypatch.setattr(importlib.import_module("dattn.prompt.render"), "render", _renderer)
    monkeypatch.setattr(run_mod, "ChatClient", functools.partial(FakeClient, response=response, errors=errors))
    paths = RunPaths(tmp_path / "cell")
    run_rollouts(
        rows,
        tokenizer=CharTokenizer(response),
        config=RolloutConfig(model_id="Qwen/Qwen3.6-27B", arm="da"),
        server=object(),  # type: ignore[arg-type]
        paths=paths,
        **kw,
    )
    return [ResponseRecord.from_dict(json.loads(line)) for line in paths.responses.read_text().splitlines()]


def test_every_row_carries_the_replay_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows) -> None:
    written = _run(tmp_path, monkeypatch, rows)
    assert len(written) == 3
    for rec in written:
        assert rec.dac_metrics is not None
        assert rec.dac_metrics["focus_attempts"] == 1
        assert rec.dac_metrics["focus_successes"] == 1
        assert rec.mode_trace and len(rec.mode_trace) == rec.dac_metrics["total_steps"]
        assert rec.attended_tokens_exact == rec.dac_metrics["attended_tokens"]
        assert rec.total_attendable == rec.dac_metrics["total_attendable"]
        # The conservative branch: only a recorder-armed run may fill this (P8).
        assert rec.attended_tokens_blocks is None


def test_replayed_total_steps_is_within_one_of_the_generated_token_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows
) -> None:
    """dac's own property: replay reads the TEXT, so the final EOS never reaches it."""
    (rec, *_) = _run(tmp_path, monkeypatch, rows)
    assert rec.n_generated_tokens is not None and rec.dac_metrics is not None
    assert abs(int(rec.dac_metrics["total_steps"]) - rec.n_generated_tokens) <= 1


def test_the_metric_key_set_is_dacs_minus_the_search_method_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows
) -> None:
    from dattn.fsm.metrics import METRIC_KEYS

    (rec, *_) = _run(tmp_path, monkeypatch, rows)
    assert rec.dac_metrics is not None
    assert set(rec.dac_metrics) == set(METRIC_KEYS)
    assert len(METRIC_KEYS) == 25  # dac's 29, minus five search-method keys, plus global_steps


def test_a_transport_failure_is_flagged_without_a_trace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows) -> None:
    """A failed request stays visible without becoming a model answer."""
    written = _run(tmp_path, monkeypatch, rows, errors=frozenset({1}))
    assert [r.row_index for r in written] == [0, 1, 2]
    assert written[1].transport_error and written[1].dac_metrics is None
    assert all(r.dac_metrics is not None for r in (written[0], written[2]))


def test_a_nan_metric_survives_the_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows) -> None:
    """Major 19's hard ordering constraint, executed end to end through `run_rollouts`.

    A vanilla-shaped response has no focus attempt, so `focus_success_rate` and the four
    `_mean` keys are NaN. Without `json_safe` the first row would kill the writer.
    """
    rec = _run(tmp_path, monkeypatch, rows, response=VANILLA_RESPONSE)[0]
    assert rec.dac_metrics is not None
    assert rec.dac_metrics["focus_attempts"] == 0
    assert rec.dac_metrics["focus_success_rate"] is None  # NaN on the wire is null
    assert rec.dac_metrics["mean_focus_area"] is None
    assert rec.dac_metrics["local_ratio"] == 0.0  # dac's asymmetry, not a NaN


def test_replay_metrics_can_be_switched_off_for_a_smoke_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows
) -> None:
    written = _run(tmp_path, monkeypatch, rows, record_replay_metrics=False)
    assert all(r.dac_metrics is None and r.mode_trace == [] for r in written)


# --------------------------------------------------------------------------------------
# The cross-package patches the integrator applied: the answer policy is forwarded, the
# other policy's text is persisted, and the replay can be turned off.
# --------------------------------------------------------------------------------------

TWO_ANSWERS = '<focus magic_chunks="1">x</focus><answer>4</answer> then <answer>5</answer>'


def test_the_default_policy_grades_the_first_answer_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows
) -> None:
    """dac's and the paper's policy, and the default since FIX_WPA item 33."""
    (rec, *_) = _run(tmp_path, monkeypatch, rows, response=TWO_ANSWERS)
    assert rec.answer_policy == "first"
    assert rec.parsed_answer == "4"
    assert rec.n_answer_blocks == 2


def test_last_answer_reaches_the_runner_and_changes_the_graded_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows
) -> None:
    """`--last-answer` parsed but was forwarded nowhere before this (FIX_WPA handoff H2)."""
    (rec, *_) = _run(tmp_path, monkeypatch, rows, response=TWO_ANSWERS, last_answer=True)
    assert rec.answer_policy == "last"
    assert rec.parsed_answer == "5"


def test_both_policies_texts_are_persisted_so_one_run_prices_the_deviation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows
) -> None:
    """`Extraction.policies_differ` is now readable off a written row (FIX_WPA handoff H5)."""
    (rec, *_) = _run(tmp_path, monkeypatch, rows, response=TWO_ANSWERS)
    assert rec.first_answer == "4"
    assert rec.last_answer == "5"
    assert rec.first_answer != rec.last_answer  # the row moved under the other policy

    (same, *_) = _run(tmp_path / "single", monkeypatch, rows, response=DA_RESPONSE)
    assert same.first_answer == same.last_answer == "4"


def test_a_row_written_before_last_answer_existed_still_loads(tmp_path: Path) -> None:
    """Absent, not empty: an old cell must not turn into a row claiming the policies agree."""
    rec = ResponseRecord.from_dict({"subset": "s", "row_index": 0, "first_answer": "4"})
    assert rec.last_answer is None
    assert "last_answer" in rec.to_dict()


def test_no_replay_metrics_leaves_the_trace_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows) -> None:
    """wpB's escape hatch for a rollout-only smoke run. Not a publishable cell."""
    written = _run(tmp_path, monkeypatch, rows, record_replay_metrics=False)
    assert len(written) == 3
    for rec in written:
        assert rec.dac_metrics is None
        assert rec.mode_trace == []
        assert rec.attended_tokens_exact is None
        assert rec.attended_tokens_blocks is None


# --------------------------------------------------------------------------------------
# P8, P9, P10: the denominator
# --------------------------------------------------------------------------------------


def test_a_non_da_arm_gets_its_denominator_without_the_fsm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows
) -> None:
    """D5 plus P10: no replay on a non-DA arm, but the baseline number survives.

    review major 18 divides the DA arm by its own model's vanilla cell, so a vanilla arm
    with no `attended_tokens` makes every headline ratio uncomputable. A non-DA arm
    attends everything, so the number is arithmetic rather than replayed.
    """
    import importlib

    monkeypatch.setattr(importlib.import_module("dattn.prompt.render"), "render", _renderer)
    monkeypatch.setattr(run_mod, "ChatClient", functools.partial(FakeClient, response=VANILLA_RESPONSE))

    def no_replay(*_a: Any, **_kw: Any) -> dict[str, Any]:
        raise AssertionError("the FSM must not be replayed on a non-DA arm (D5)")

    monkeypatch.setattr(run_mod, "replay_record", no_replay)

    paths = RunPaths(tmp_path / "vanilla")
    run_rollouts(
        rows,
        tokenizer=CharTokenizer(VANILLA_RESPONSE),
        config=RolloutConfig(model_id="Qwen/Qwen3.6-27B", arm="vanilla", prompt_format="vanilla", dattn_enabled=False),
        server=object(),  # type: ignore[arg-type]
        paths=paths,
    )
    written = [ResponseRecord.from_dict(json.loads(line)) for line in paths.responses.read_text().splitlines()]
    assert len(written) == 3
    for rec in written:
        assert rec.dac_metrics is not None
        assert rec.attended_tokens_exact == rec.total_attendable, "a non-DA arm attends everything"
        assert rec.dac_metrics["attention_ratio"] == 1.0
        assert rec.dac_metrics["focus_steps"] == 0
        assert set(rec.mode_trace) == {0}  # GLOBAL throughout
        assert rec.attended_tokens_blocks is None


def test_the_da_arm_still_replays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows) -> None:
    """The other half of the branch: `dattn_enabled` still means a real replay."""
    (rec, *_) = _run(tmp_path, monkeypatch, rows)
    assert rec.dac_metrics is not None
    assert rec.dac_metrics["focus_attempts"] == 1, "the FSM really ran on the DA arm"
