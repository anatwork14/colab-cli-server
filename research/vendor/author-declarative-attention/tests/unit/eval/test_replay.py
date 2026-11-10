"""Offline replay: the per-step traces and the metric dict, with no tokenizer.

`replay_ids` is the tokenizer-free core, so the state machine's contract with Lane D can
be pinned on a CPU with no transformers import. The one property that matters for parity
is the ORDER: the mode and mask recorded for a step are the ones that were in force for
that forward, and a tag takes effect on the NEXT step.
"""

from __future__ import annotations

import pytest
from fakes import FakeTokenizer as _CharTokenizer

from dattn.contracts import Mode, PromptLayout
from dattn.eval.replay import SKIP_SPECIAL_TOKENS, replay, replay_ids, replay_record


@pytest.fixture
def layout() -> PromptLayout:
    return PromptLayout(n_prompt_tokens=100, question_start=80, segments=((16, 40), (40, 60)))


def _chars(text: str) -> tuple[list[int], list[str]]:
    """One character per token, which is enough to drive the tag scanner."""
    return list(range(1, len(text) + 1)), list(text)


def test_a_response_with_no_tags_stays_global(layout: PromptLayout) -> None:
    ids, texts = _chars("hello world")
    modes, masks, metrics = replay_ids(ids, texts, layout)
    assert set(modes) == {Mode.GLOBAL}
    assert len(modes) == len(masks) == len(ids)
    assert metrics["total_steps"] == len(ids)
    assert metrics["focus_attempts"] == 0
    assert all(m.dense for m in masks)


def test_a_focus_tag_takes_effect_on_the_next_step(layout: PromptLayout) -> None:
    text = '<focus magic_chunks="1">abc</focus>xyz'
    modes, masks, metrics = replay_ids(*_chars(text), layout)
    close_at = text.index(">")
    assert modes[close_at] is Mode.GLOBAL  # the `>` step itself still ran under GLOBAL
    assert modes[close_at + 1] is Mode.FOCUS
    assert not masks[close_at + 1].dense
    assert metrics["focus_attempts"] == 1
    assert metrics["focus_successes"] == 1
    assert metrics["n_unique_segments_focused"] == 1


def test_the_local_alias_and_its_close(layout: PromptLayout) -> None:
    text = "<answer>4</answer>."
    modes, _, metrics = replay_ids(*_chars(text), layout)
    assert modes[text.index(">") + 1] is Mode.LOCAL
    assert modes[-1] is Mode.GLOBAL
    assert metrics["local_steps"] > 0
    assert metrics["focus_attempts"] == 0


def test_a_failed_focus_is_counted_and_leaves_the_request_global(layout: PromptLayout) -> None:
    modes, _, metrics = replay_ids(*_chars('<focus magic_chunks="99">x'), layout)
    assert set(modes) == {Mode.GLOBAL}
    assert metrics["focus_attempts"] == 1
    assert metrics["focus_successes"] == 0
    assert metrics["focus_fail_out_of_range_index"] == 1


def test_the_metric_dict_has_dacs_key_set(layout: PromptLayout) -> None:
    from dattn.fsm.metrics import METRIC_KEYS

    _, _, metrics = replay_ids(*_chars("plain text"), layout)
    assert set(metrics) == set(METRIC_KEYS)


def test_mismatched_ids_and_texts_are_rejected(layout: PromptLayout) -> None:
    with pytest.raises(ValueError, match="ids but"):
        replay_ids([1, 2], ["a"], layout)


def test_an_empty_response_replays_to_zero_steps(layout: PromptLayout) -> None:
    modes, masks, metrics = replay_ids([], [], layout)
    assert modes == [] and masks == []
    assert metrics["total_steps"] == 0
    assert metrics["local_ratio"] == 0.0  # dac's asymmetry, preserved


def test_replay_row_refuses_without_the_context() -> None:
    from dattn.eval.replay import replay_row

    with pytest.raises(ValueError, match="tokenizer"):
        replay_row({"response": "x"})
    with pytest.raises(KeyError, match="context"):
        replay_row({"response": "x"}, tokenizer=object())


def test_replay_decodes_under_the_engines_convention(layout: PromptLayout) -> None:
    """review major 16. A replayed metric IS the recorded metric, so the two must agree."""
    from dattn.engine import processor

    assert SKIP_SPECIAL_TOKENS is processor.SKIP_SPECIAL_TOKENS is True

    seen: list[bool] = []

    class Tok:
        def __call__(self, text, add_special_tokens=False):  # noqa: ANN001, ANN204, FBT002
            return {"input_ids": list(range(1, len(text) + 1))}

        def decode(self, ids, skip_special_tokens=False):  # noqa: ANN001, ANN201, FBT002
            seen.append(skip_special_tokens)
            return "x"

    replay("abc", layout, Tok())
    assert seen == [True, True, True]


# ---------------------------------------------------------------------------------------
# replay_record: the five schema fields (review blocker 3)
# ---------------------------------------------------------------------------------------


def test_replay_record_fills_every_schema_field(layout: PromptLayout) -> None:
    text = '<focus magic_chunks="1">abc</focus>xyz'
    out = replay_record(text, layout, _CharTokenizer(text))
    assert set(out) == {
        "dac_metrics",
        "mode_trace",
        "attended_tokens_exact",
        "attended_tokens_blocks",
        "total_attendable",
    }
    metrics = out["dac_metrics"]
    assert len(out["mode_trace"]) == metrics["total_steps"] == len(text)
    assert all(isinstance(m, int) for m in out["mode_trace"])
    assert out["attended_tokens_exact"] == metrics["attended_tokens"]
    assert out["total_attendable"] == metrics["total_attendable"]
    assert out["attended_tokens_blocks"] is None  # the conservative branch (P8)
    assert metrics["focus_attempts"] == 1 and metrics["focus_successes"] == 1


def test_block_rounding_is_never_smaller_than_the_exact_count(layout: PromptLayout) -> None:
    text = '<focus magic_chunks="1">abc</focus>xyz'
    tok = _CharTokenizer(text)
    assert replay_record(text, layout, tok, block_size=0)["attended_tokens_blocks"] is None
    rounded = replay_record(text, layout, tok, block_size=16)["attended_tokens_blocks"]
    assert rounded >= replay_record(text, layout, tok)["attended_tokens_exact"]


def test_the_metric_dict_is_a_copy_the_caller_may_own(layout: PromptLayout) -> None:
    out = replay_record("abc", layout, _CharTokenizer("abc"))
    out["dac_metrics"]["total_steps"] = -1
    assert replay_record("abc", layout, _CharTokenizer("abc"))["dac_metrics"]["total_steps"] == 3


# -- P10: the non-DA denominator, arithmetically, with no FSM ---------------------------


def test_the_dense_trace_equals_a_real_replay_of_a_response_with_no_focus_tag() -> None:
    """The whole claim of `dense_trace`, checked against the FSM rather than asserted.

    D5 forbids running the FSM on a non-DA arm. `dense_trace` replaces it with a closed
    form. The two must agree exactly on a response the FSM would keep GLOBAL throughout,
    or the vanilla denominator the paper's ratios divide by silently changes meaning.
    """
    from dattn.eval.replay import dense_trace, replay_ids

    layout = PromptLayout(n_prompt_tokens=100, question_start=80, segments=((16, 40), (40, 60)))
    texts = ["just", " an", " answer", " here"]
    ids = [1, 2, 3, 4]
    _modes, _masks, metrics = replay_ids(ids, texts, layout)

    trace = dense_trace(len(ids), layout)
    assert trace["total_attendable"] == int(metrics["total_attendable"])
    assert trace["attended_tokens_exact"] == int(metrics["attended_tokens"])
    assert trace["dac_metrics"]["total_steps"] == int(metrics["total_steps"])
    assert trace["dac_metrics"]["attention_ratio"] == metrics["attention_ratio"] == 1.0


def test_the_dense_trace_is_all_global_and_fills_no_block_count() -> None:
    from dattn.eval.replay import dense_trace

    layout = PromptLayout(n_prompt_tokens=100, question_start=80, segments=())
    trace = dense_trace(3, layout)
    assert trace["mode_trace"] == [int(Mode.GLOBAL)] * 3
    assert trace["attended_tokens_blocks"] is None  # PROVENANCE P8, as everywhere on this path
    assert trace["dac_metrics"]["focus_steps"] == 0
    assert trace["dac_metrics"]["global_steps"] == 3


def test_the_dense_trace_of_an_empty_rollout_is_empty_not_negative() -> None:
    from dattn.eval.replay import dense_trace

    layout = PromptLayout(n_prompt_tokens=100, question_start=80, segments=())
    trace = dense_trace(0, layout)
    assert trace["total_attendable"] == 0
    assert trace["attended_tokens_exact"] == 0
    assert trace["mode_trace"] == []
    assert trace["dac_metrics"]["attention_ratio"] == 0.0
