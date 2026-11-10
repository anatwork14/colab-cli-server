"""The decode convention, as an executed regression (review major 16).

`dattn.engine.processor` decoded runtime tokens with `skip_special_tokens=False` until
2026-09-17, while dac decodes with `True` (`src/dac/dac_state.py:715`, `:540-547`) and both
`dattn.prompt.render` and the fixture generator already used `True`. The stop token lands in
`output_token_ids` on essentially every request, so the divergence fired constantly and the
328-case dac parity suite never exercised the production convention.

These two tests are the minimal pair. They also specify the `-special-token` fixture case in
`tests/fixtures/fsm/cases.jsonl`, which is `kind: "dattn_only"` because dac cannot produce a
case whose tail contains a special token at all.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dattn.contracts import Mode, PromptLayout
from dattn.eval.replay import replay_ids

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "fsm" / "cases.jsonl"
CASE_ID = "qwen_nseg05-special-token"

#: The divergence, verbatim. `X` is the fourth token.
TAG_PREFIX = ["<focus", " magic_chunks", '="1"']


@pytest.fixture
def layout() -> PromptLayout:
    return PromptLayout(n_prompt_tokens=100, question_start=80, segments=((16, 40), (40, 60)))


def _run(x: str, layout: PromptLayout):  # noqa: ANN202
    texts = [*TAG_PREFIX, x, ">", " ok"]
    return replay_ids(list(range(1, len(texts) + 1)), texts, layout)


def test_a_special_token_inside_a_focus_tag_does_not_break_the_tag(layout: PromptLayout) -> None:
    """The new convention: the special token decodes to "" and the tag survives it."""
    modes, _, metrics = _run("", layout)
    assert modes[-1] is Mode.FOCUS
    assert metrics["focus_attempts"] == 1
    assert metrics["focus_fail_syntax_error"] == 0
    assert metrics["focus_successes"] == 0  # a success is counted on `</focus>`, not the open
    assert metrics["n_unique_segments_focused"] == 1
    assert metrics["total_steps"] == 6  # the empty token still costs a forward


def test_the_old_convention_scored_the_same_tag_as_two_syntax_errors(layout: PromptLayout) -> None:
    """What was fixed. `<|im_end|>` carries a `>`, so it fires the scan a step early."""
    modes, _, metrics = _run("<|im_end|>", layout)
    assert modes[-1] is Mode.GLOBAL
    assert metrics["focus_attempts"] == 2
    assert metrics["focus_fail_syntax_error"] == 2
    assert metrics["focus_steps"] == 0


def test_the_fixture_carries_the_special_token_case_and_dac_could_not_have() -> None:
    lines = FIXTURE.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    cases = {json.loads(line)["case_id"]: json.loads(line) for line in lines[1:]}
    case = cases[CASE_ID]
    assert case["kind"] == "dattn_only"
    assert case["token_texts"] == [*TAG_PREFIX, "", ">", " ok"]
    assert case["final_mode"] == "FOCUS"
    assert header["n_dattn_only"] == 1
    assert "skip_special_tokens=True" in header["known_deltas"]["special_tokens"]
    assert sum(1 for c in cases.values() if c["kind"] == "dattn_only") == 1
