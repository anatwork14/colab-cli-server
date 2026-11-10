"""The `<answer>` format gate, and the one policy dattn lets an operator change.

Every case dac's `B` E14-E17 lists is pinned here, including the ones dattn keeps because
they define the measurement, so that "fixing" one is a visible test change and not a
silent shift in every published number.
"""

from __future__ import annotations

import re

from dattn.eval.extract import ANSWER_RE, INVALID_SENTINEL, extract_answer


def test_the_regex_is_dacs_byte_for_byte() -> None:
    assert ANSWER_RE.pattern == r"<answer>(.*?)</answer>"
    assert ANSWER_RE.flags & re.DOTALL


def test_the_first_block_wins_by_default_and_last_answer_selects_the_other() -> None:
    """The owner's ruling, stated as a test: the default is the paper's measurement.

    dattn used to default to the last block. On the recorded dac data that policy moves
    11 of 1611 Qwen vanilla rows, 9 of 1611 Qwen DA rows and 0 Gemma rows, so the flip is
    small, real, and exactly the kind of thing that must not happen silently.
    """
    text = "<answer>WRONG</answer> on reflection <answer>RIGHT</answer>"
    default = extract_answer(text)
    assert default.text == "WRONG"
    assert default.n_blocks == 2
    assert default.policy == "first"
    assert default.first_text == "WRONG"
    assert not default.differs_from_dac

    last = extract_answer(text, last=True)
    assert last.text == "RIGHT"
    assert last.policy == "last"
    assert last.first_text == "WRONG"
    assert last.differs_from_dac


def test_three_blocks_take_the_first_not_the_middle_or_the_last() -> None:
    """The multi-answer fixture. `re.search` semantics, pinned at every arity."""
    text = "<answer>A</answer>\nthinking\n<answer>B</answer>\nmore\n<answer>C</answer>"
    ex = extract_answer(text)
    assert ex.text == "A"
    assert ex.n_blocks == 3
    assert ex.policy == "first"
    assert ex.parsed_answer == "A"
    assert extract_answer(text, last=True).text == "C"


def test_the_default_matches_dacs_regex_on_a_multi_answer_response() -> None:
    """Belt and braces: the default IS `re.search`, not a reimplementation of it."""
    text = "<answer>  42  </answer> no wait <answer>43</answer> sorry <answer>44</answer>"
    dac = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    assert dac is not None
    assert extract_answer(text).text == dac.group(1).strip()


def test_the_first_keyword_is_a_shim_that_never_selects_the_last_block() -> None:
    """The old spelling. `first=False` used to mean the last block; it no longer can."""
    text = "<answer>WRONG</answer> <answer>RIGHT</answer>"
    assert extract_answer(text, first=True).text == "WRONG"
    assert extract_answer(text, first=False).text == "WRONG"
    assert extract_answer(text, first=None).text == "WRONG"
    assert extract_answer(text, first=True, last=True).text == "WRONG"


def test_a_single_block_is_identical_under_both_policies() -> None:
    ex = extract_answer("blah <answer> 42 </answer> blah")
    assert ex.text == "42"  # stripped, as dac does
    assert ex.n_blocks == 1
    assert not ex.differs_from_dac
    assert extract_answer("blah <answer> 42 </answer> blah", last=True).text == "42"


def test_no_block_is_a_format_failure_with_the_dac_sentinel() -> None:
    ex = extract_answer("I refuse to use tags")
    assert ex.text is None
    assert not ex.format_correct
    assert ex.n_blocks == 0
    assert ex.parsed_answer == INVALID_SENTINEL


def test_the_gate_stays_case_sensitive_and_attribute_intolerant() -> None:
    """dac's behaviour, kept: these are format failures and are scored wrong (`B` E14)."""
    for text in ("<Answer>x</Answer>", '<answer type="final">x</answer>', "<answer >x</answer>"):
        assert not extract_answer(text).format_correct, text


def test_an_empty_answer_is_format_correct_with_an_empty_body() -> None:
    """`B` E17. Kept, not fixed: the judge then sees an empty <model_response>."""
    ex = extract_answer("<answer></answer>")
    assert ex.format_correct
    assert ex.text == ""
    assert ex.parsed_answer == ""


def test_a_nested_open_leaks_the_outer_text_under_both_policies() -> None:
    """`B` E16. The lazy group stops at the first close, so the outer text leaks."""
    ex = extract_answer("<answer>outer <answer>inner</answer>")
    assert ex.text == "outer <answer>inner"
    assert ex.n_blocks == 1


def test_dotall_lets_an_answer_span_newlines() -> None:
    assert extract_answer("<answer>line1\nline2</answer>").text == "line1\nline2"


def test_a_truncated_response_has_no_closing_tag_and_fails_the_gate() -> None:
    """The shape of every 8192-token truncation (`B` E6/E7), which is scored wrong."""
    ex = extract_answer("<global>thinking</global><answer>the answer begins and then the cap")
    assert not ex.format_correct
    assert ex.parsed_answer == INVALID_SENTINEL


def test_both_policies_are_recorded_on_every_row() -> None:
    """One run prices `--last-answer` without a second pass."""
    ex = extract_answer("<answer>A</answer> <answer>B</answer>")
    assert ex.first_text == "A"
    assert ex.last_text == "B"
    assert ex.policies_differ
    assert not ex.differs_from_dac  # the default IS dac's, by construction

    same = extract_answer("<answer>A</answer>")
    assert not same.policies_differ
    assert same.first_text == same.last_text == "A"

    none = extract_answer("no tags")
    assert none.first_text is None
    assert none.last_text is None
    assert not none.policies_differ
