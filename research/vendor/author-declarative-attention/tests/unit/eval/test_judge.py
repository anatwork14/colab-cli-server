"""The judge: a byte-identical rubric prompt, and a parser that never lies.

Two halves. The prompt half asserts the template is the exact bytes the published numbers
were produced under, and that dattn's substitution agrees with real Jinja. The parser half
walks every failure in `B` E10-E13 and asserts (a) dattn reads the verdict, and (b) dattn
still reports what dac would have said, so the deviation is always priceable.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from dattn.eval.judge import (
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    JUDGE_TEMPLATE_SHA256,
    PARSE_STAGES,
    JudgeConfig,
    Judgment,
    dac_parse_verdict,
    judge_template_path,
    judge_template_text,
    parse_verdict,
    render_judge_prompt,
)

# --------------------------------------------------------------------------------------
# The prompt
# --------------------------------------------------------------------------------------


def test_the_shipped_template_is_the_pinned_bytes() -> None:
    raw = judge_template_path().read_bytes()
    assert hashlib.sha256(raw).hexdigest() == JUDGE_TEMPLATE_SHA256
    # md5 `2deed1ad3979200eb42690c7370a6fbf`, the value `E 4` verified on both trees.
    assert hashlib.md5(raw, usedforsecurity=False).hexdigest() == "2deed1ad3979200eb42690c7370a6fbf"


def test_the_template_has_exactly_the_three_fields_the_judge_may_see() -> None:
    """Never the context. The judge grades question + rubric + parsed answer only."""
    text = judge_template_text()
    assert text.count("{{ question }}") == 1
    assert text.count("{{ rubric }}") == 1
    assert text.count("{{ response }}") == 1
    assert "{{ context" not in text
    assert "{%" not in text  # no control structures, so plain substitution is exact


def test_rendering_agrees_with_real_jinja() -> None:
    jinja2 = pytest.importorskip("jinja2")
    q, r, a = "What is 2+2?", "- must say 4", "four"
    mine = render_judge_prompt(q, r, a)
    theirs = jinja2.Template(judge_template_text()).render(question=q, rubric=r, response=a)
    assert mine == theirs


def test_rendering_places_the_answer_inside_the_model_response_tags() -> None:
    out = render_judge_prompt("q", "rubric text", "my answer")
    assert "<model_response>\nmy answer\n</model_response>" in out
    assert out.startswith("You are an AI assistant specializing in language model evaluation.")
    assert not out.endswith("\n")  # Jinja's keep_trailing_newline=False


def test_an_empty_answer_still_renders(pytestconfig: pytest.Config) -> None:
    """`B` E17: `<answer></answer>` reaches the judge as an empty body. Reproduced."""
    assert "<model_response>\n\n</model_response>" in render_judge_prompt("q", "r", "")


def test_the_judge_config_pins_the_model_and_the_recipe() -> None:
    cfg = JudgeConfig()
    assert cfg.model_id == JUDGE_MODEL == "Qwen/Qwen3.5-4B"
    assert cfg.thinking is True
    assert cfg.max_tokens == JUDGE_MAX_TOKENS == 32768
    assert cfg.sampling() == {
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 1.5,
        "repetition_penalty": 1.0,
    }


# --------------------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------------------


def test_the_clean_case_parses_as_json_and_agrees_with_dac() -> None:
    for text, want in (('{"correct": true}', True), ('{"correct": false}', False)):
        v = parse_verdict(text)
        assert v.correct is want
        assert v.stage == "json"
        assert not v.differs_from_dac


def test_a_think_preamble_is_stripped_like_dac() -> None:
    v = parse_verdict('<think>weighing it up</think>\n{"correct": true}')
    assert v.correct is True
    assert not v.differs_from_dac
    assert not v.think_truncated


def test_e12_a_reopened_think_erases_the_verdict_exactly_as_dac_does() -> None:
    """dac splits on the FIRST `</think>` and keeps the empty tail, and so does dattn.

    This pins a CHANGE OF MIND, so it is worth stating plainly. dattn used to split on
    the last close tag, which recovered the verdict this judge stated before reopening
    its scratchpad, and `B` E12 recorded that as an improvement on dac. It was one, and
    it was dropped anyway: this parser exists to be comparable with dac's recorded
    judgments, and an improvement we cannot evidence is still an undisclosed divergence.
    The two rules differ only on a response carrying more than one close tag, which 0 of
    the 500 sampled real rows do, so nothing already recorded moves. See
    `judge._verdict_regions` and `run_eval.py:93-102`.
    """
    text = '{"correct": true}<think>wait, no</think>'
    assert dac_parse_verdict(text) is None
    v = parse_verdict(text)
    assert v.correct is None
    assert v.stage == "unparsable"
    assert not v.differs_from_dac, "the whole point of the change is that dattn now agrees here"


def test_e10_a_truncated_think_is_recorded_rather_than_scored_wrong() -> None:
    """141 of 142 null verdicts on the legacy data were exactly this shape."""
    text = "<think>I am still reasoning and the cap arrives"
    v = parse_verdict(text)
    assert v.correct is None
    assert v.stage == "unparsable"
    assert v.think_truncated
    assert dac_parse_verdict(text) is None


def test_e10_a_truncated_think_that_did_state_a_verdict_is_read() -> None:
    """dac's regex fallback recovers this one too. Both agree, and dattn says so."""
    text = '<think>the rubric wants 4, the answer says 4, so {"correct": true} and now I ramble'
    v = parse_verdict(text)
    assert v.correct is True
    assert v.stage in {"json_block", "regex"}
    assert v.think_truncated
    assert dac_parse_verdict(text) is True
    assert not v.differs_from_dac


def test_e10_plus_e13_a_truncated_think_with_a_python_bool_is_only_read_by_dattn() -> None:
    """`{'correct': True}` is not JSON and does not match dac's lowercase-only regex."""
    text = "<think>so {'correct': True} and then the cap arrives"
    assert dac_parse_verdict(text) is None
    v = parse_verdict(text)
    assert v.correct is True
    assert v.think_truncated
    assert v.differs_from_dac


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ('{"correct": "true"}', True),
        ('{"correct": 1}', True),
        ('{"correct": 0}', False),
        ("{'correct': True}", True),
        ('Here is my assessment:\n{"correct": false}\nThanks.', False),
    ],
)
def test_e13_bool_like_verdicts_parse_where_dac_returned_none(text: str, want: bool) -> None:
    v = parse_verdict(text)
    assert v.correct is want
    assert v.stage in PARSE_STAGES
    assert v.stage != "unparsable"


def test_e11_a_self_correcting_judge_is_scored_on_its_last_verdict() -> None:
    text = '{"correct": false}\nOn reflection: {"correct": true}'
    assert dac_parse_verdict(text) is False
    v = parse_verdict(text)
    assert v.correct is True
    assert v.n_candidates == 2
    assert v.differs_from_dac


def test_a_verdict_inside_the_think_block_is_never_returned() -> None:
    """dac strips through `</think>` and so must dattn (`run_eval.py:93`).

    The scratchpad said true, the judge then answered that the response was wrong and
    stated no machine-readable verdict. Reading true out of the thinking scores the row
    on a thought the judge talked itself out of.
    """
    text = 'I weigh it up and {"correct": true} seems right.</think> The response is wrong.'
    assert dac_parse_verdict(text) is None
    v = parse_verdict(text)
    assert v.correct is None
    assert v.stage == "unparsable"
    assert not v.differs_from_dac


def test_a_close_tag_with_no_opener_is_the_real_world_shape() -> None:
    """Qwen's template injects `<think>` into the PROMPT, so responses carry only the
    close tag. All 500 sampled rows of dac's recorded judgments look like this."""
    text = 'reasoning about the rubric</think>\n{"correct": false}'
    assert "<think>" not in text
    v = parse_verdict(text)
    assert v.correct is False
    assert v.stage == "json"
    assert not v.think_truncated
    assert not v.differs_from_dac


def test_only_the_tail_after_the_first_close_tag_is_ever_read() -> None:
    """Everything before and including the first `</think>` is dropped, at every stage.

    The tail here holds only a bare `"correct": true`, which is the `regex` stage. The
    text that was dropped holds `{"correct": false}`, which is the stronger `json_block`
    stage. Reading the ladder stage major rather than region major is what let the
    dropped text win.
    """
    text = 'my draft was {"correct": false}<think>on reflection</think>"correct": true'
    v = parse_verdict(text)
    assert v.correct is True
    assert v.stage == "regex"


def test_genuinely_unreadable_output_stays_unparsable() -> None:
    for text in ("", "I cannot decide.", "correctness is a spectrum"):
        v = parse_verdict(text)
        assert v.correct is None
        assert v.stage == "unparsable"
        assert not v.parsed


def test_every_stage_name_is_declared() -> None:
    samples = ['{"correct": true}', '{"correct": 1}', 'x "correct": false x', "correct = yes", "nothing here"]
    assert {parse_verdict(s).stage for s in samples} <= set(PARSE_STAGES)


def test_judgment_round_trips_and_keys_like_dac() -> None:
    j = Judgment(subset="s", row_index=3, response_index=0, correct=None, stage="unparsable")
    assert j.key == ("s", 3, 0)
    assert Judgment.from_dict(j.to_dict()) == j
    # A raw dac judgment row, which has neither `stage` nor `dac_correct`.
    raw = {"subset": "s", "row_index": 3, "response_index": 0, "correct": True, "evaluation_text": "x"}
    assert Judgment.from_dict(raw).correct is True


# --------------------------------------------------------------------------------------
# The request shape (Blocker 2)
# --------------------------------------------------------------------------------------


def test_judge_request_body_is_dacs_recipe_plus_enable_thinking() -> None:
    """dac `model_specs.QWEN3_5_THINKING_GENERAL_SAMPLING_PARAMS` and `full_eval.sh:79`."""
    body = JudgeConfig().request_body()
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["max_tokens"] == JUDGE_MAX_TOKENS == 32768
    assert body["n"] == 1
    assert body["temperature"] == 1.0
    assert body["top_p"] == 0.95
    assert body["presence_penalty"] == 1.5
    # vLLM extensions travel in `extra_body`; `ChatClient.chat` flattens them.
    assert body["extra_body"] == {"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0}
    assert "stop" not in body
    assert "seed" not in body
    assert "tools" not in body


def test_thinking_off_is_carried_into_chat_template_kwargs() -> None:
    """A deliberate non-paper configuration must still reach the wire, not be dropped."""
    assert JudgeConfig(thinking=False).request_body()["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_judge_sends_exactly_one_user_turn_and_no_system_turn() -> None:
    msgs = JudgeConfig().messages("Q", "R", "A")
    assert [m["role"] for m in msgs] == ["user"]
    assert msgs[0]["content"] == render_judge_prompt("Q", "R", "A")


# --------------------------------------------------------------------------------------
# The parser against real dac judgments
# --------------------------------------------------------------------------------------


def test_the_real_dac_judgment_shape_parses() -> None:
    """Qwen's thinking template injects `<think>\\n` into the PROMPT.

    So a recorded dac `evaluation_text` carries the closing `</think>` with no opener.
    Verified on the first 200 rows of
    `outputs_128cap_backup_260703/gemma_4_31b_it/dac_chunk_multi_turn_tool_v1/judgments.jsonl`:
    200/200 have `</think>`, 0/200 have `<think>`. A parser that keys off the opener would
    read every one of them as unparsable.
    """
    text = 'The user wants me to evaluate...\nOutput JSON.\n{\n  "correct": true\n}\n</think>\n\n{"correct": true}'
    v = parse_verdict(text)
    assert v.correct is True
    assert v.stage == "json"
    assert v.dac_correct is True
    assert not v.think_truncated


def test_dattn_and_dac_parsers_agree_on_recorded_dac_judgments(dac_outputs: Path) -> None:
    """Opt-in. Every one of the first 200 real verdicts must parse the same way.

    Run with `DATTN_DAC_OUTPUTS=<.../260507_live_dac_datagen>`. Executed at the time this
    test was written: 200 rows, 0 disagreements between `dac_parse_verdict` and dac's
    `parse_evaluation_text`, 0 between `parse_verdict` and dac, and the recorded `correct`
    field reproduced on all 200. Stages were 197 `json` and 3 `json_block`.
    """
    path = (
        dac_outputs
        / "outputs_128cap_backup_260703"
        / "gemma_4_31b_it"
        / "dac_chunk_multi_turn_tool_v1"
        / "judgments.jsonl"
    )
    if not path.is_file():
        pytest.skip(f"no recorded judgments at {path}")
    n = 0
    with path.open() as fh:
        for line in fh:
            if n >= 200:
                break
            row = json.loads(line)
            text = row["evaluation_text"]
            n += 1
            assert dac_parse_verdict(text) == row["correct"], row["row_index"]
            assert parse_verdict(text).correct == row["correct"], row["row_index"]
            assert "</think>" in text
    assert n == 200
