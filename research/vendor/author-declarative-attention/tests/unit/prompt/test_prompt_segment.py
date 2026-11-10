"""The segmenter (WP2). Both references agree byte for byte here (`F 3.2`)."""

import pytest
import synth_contexts

from dattn.contracts import Segment
from dattn.prompt.segment import DELIMITERS, EMPTY_CONTEXT, normalize_context, segment_context


class FakeFastTokenizer:
    """One token per four characters, with real offsets. Keeps the invariant tests fast
    and free of a Hub download, and it is a fast tokenizer by the only test that matters."""

    is_fast = True

    def __init__(self, chars_per_token: int = 4) -> None:
        self.chars_per_token = chars_per_token

    def __call__(self, text: str, return_offsets_mapping: bool = False, add_special_tokens: bool = True) -> dict:
        step = self.chars_per_token
        offsets = [(i, min(i + step, len(text))) for i in range(0, len(text), step)]
        out: dict[str, object] = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        return out


@pytest.fixture
def fake() -> FakeFastTokenizer:
    return FakeFastTokenizer()


# --------------------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------------------


def test_a_slow_tokenizer_is_refused(fake: FakeFastTokenizer) -> None:
    fake.is_fast = False
    with pytest.raises(ValueError, match="fast tokenizer"):
        segment_context(fake, "text")
    with pytest.raises(ValueError, match="fast tokenizer"):
        segment_context(None, "text")


def test_max_below_target_is_refused(fake: FakeFastTokenizer) -> None:
    with pytest.raises(ValueError, match="must be >="):
        segment_context(fake, "text", target_tokens=100, max_tokens=99)


def test_the_delimiter_ladder_is_the_specified_one() -> None:
    assert DELIMITERS == (r"(\n\s*\n)", r"(\n)", r"([.!?]+\s+)", r"([;:,]\s+)", r"(\s+)")


# --------------------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("blank", ["", "   ", "\n\n", "\t \n"])
def test_blank_contexts_normalise_to_the_placeholder(blank: str) -> None:
    assert normalize_context(blank) == EMPTY_CONTEXT == "<empty_context>"


def test_a_non_blank_context_is_unchanged() -> None:
    assert normalize_context(" real text ") == " real text "


def test_an_empty_context_segments_to_nothing_but_the_placeholder_does_not(
    fake: FakeFastTokenizer,
) -> None:
    assert segment_context(fake, "") == []
    assert len(segment_context(fake, normalize_context(""))) == 1


# --------------------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------------------


def test_a_short_context_is_one_segment(fake: FakeFastTokenizer) -> None:
    segments = segment_context(fake, "one two three four five.")
    assert len(segments) == 1
    assert segments[0] == Segment(1, "one two three four five.", 6, 0, 24)


@pytest.mark.parametrize("n_paragraphs", [0, 1, 6, 46, 120])
def test_segments_partition_the_context_exactly(fake: FakeFastTokenizer, n_paragraphs: int) -> None:
    context = normalize_context(synth_contexts._paragraphs(3, n_paragraphs))
    segments = segment_context(fake, context, target_tokens=200, max_tokens=250)
    assert [s.id for s in segments] == list(range(1, len(segments) + 1))
    assert "".join(s.text for s in segments) == context
    assert segments[0].char_start == 0
    assert segments[-1].char_end == len(context)
    for previous, following in zip(segments, segments[1:], strict=False):
        assert previous.char_end == following.char_start
    for s in segments:
        assert s.text == context[s.char_start : s.char_end]


def test_no_segment_exceeds_the_cap_unless_it_is_one_atomic_word(fake: FakeFastTokenizer) -> None:
    context = synth_contexts._paragraphs(5, 60)
    for s in segment_context(fake, context, target_tokens=200, max_tokens=250):
        assert s.token_count <= 250 or " " not in s.text.strip()


def test_a_whitespace_free_blob_stays_atomic_even_over_the_cap(fake: FakeFastTokenizer) -> None:
    blob = "x" * 20_000
    segments = segment_context(fake, blob, target_tokens=200, max_tokens=250)
    assert len(segments) == 1
    assert segments[0].token_count == 5000 > 250


def test_the_coarsest_boundary_the_cap_allows_is_used(fake: FakeFastTokenizer) -> None:
    """Paragraph breaks are level one, so a context of exactly-sized paragraphs cuts on
    blank lines and never mid sentence."""
    paragraph = ("word " * 50).strip()
    context = "\n\n".join([paragraph] * 8)
    for s in segment_context(fake, context, target_tokens=64, max_tokens=80):
        assert not s.text.strip().endswith("wor"), "a cut landed inside a word"


def test_target_and_max_defaults_are_the_papers(tokenizer: object) -> None:
    import inspect

    signature = inspect.signature(segment_context)
    assert signature.parameters["target_tokens"].default == 2048
    assert signature.parameters["max_tokens"].default == 2560


def test_segment_counts_on_a_real_tokenizer(tokenizer: object) -> None:
    context = synth_contexts.context_for("five_segments")
    segments = segment_context(tokenizer, context)
    assert len(segments) == 5
    assert "".join(s.text for s in segments) == context
    assert all(s.token_count <= 2560 for s in segments)
    assert sum(s.token_count for s in segments) == len(
        tokenizer(context, add_special_tokens=False)["input_ids"]  # type: ignore[operator]
    )
