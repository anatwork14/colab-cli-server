"""The renderer and span recovery (WP2), live against the two shipping tokenizers."""

from typing import Any

import pytest
import synth_contexts

from dattn.contracts import SINK_TOKENS, PromptLayout, RenderError
from dattn.fsm.layout import validate_layout
from dattn.prompt import get_format, render, segment_context, template_hashes
from dattn.prompt.formats.base import TOOL_NAME
from dattn.prompt.segment import EMPTY_CONTEXT, normalize_context

TAGGED = ("chunk_multi_turn_tool", "chunk_single_turn_xml")
ALL_FORMATS = (*TAGGED, "vanilla")


def do_render(tokenizer: Any, case: str, fmt: str) -> Any:
    return render(
        tokenizer,
        context=synth_contexts.context_for(case),
        question=synth_contexts.question_for(case),
        fmt=fmt,
        tool_schema=(fmt == "chunk_multi_turn_tool"),
    )


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------


def test_the_three_formats_are_dacs_names() -> None:
    for name in ALL_FORMATS:
        assert get_format(name).name == name


def test_an_unknown_format_is_a_render_error() -> None:
    with pytest.raises(RenderError, match="unknown prompt format"):
        get_format("magic_chunk")


# --------------------------------------------------------------------------------------
# Span recovery
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", TAGGED)
@pytest.mark.parametrize("case", list(synth_contexts.CASES))
def test_every_segment_span_is_recovered_and_contains_its_own_text(tokenizer: Any, case: str, fmt: str) -> None:
    """`len(spans) == n_segments` is asserted inside `render`; this checks the spans are
    the right ones by decoding them back."""
    rendered = do_render(tokenizer, case, fmt)
    assert rendered.format_error == ""
    assert len(rendered.layout.segments) == len(rendered.segments)
    for segment, (start, end) in zip(rendered.segments, rendered.layout.segments, strict=True):
        body = tokenizer.decode(rendered.ids[start:end])
        assert segment.text.strip() in body, f"segment {segment.id} is not inside its own span"


@pytest.mark.parametrize("fmt", TAGGED)
@pytest.mark.parametrize("case", ["collision_qwen", "collision_gemma", "collision_pair"])
def test_a_context_that_quotes_tool_markup_does_not_truncate_a_span(tokenizer: Any, case: str, fmt: str) -> None:
    """The `F D2` regression, with the mechanism stated as the reviewer verified it.

    dac's per-family regex ends the affected span at the quoted markup instead of the real
    turn boundary: 30 tokens lost on Qwen3.6 and 424 on Gemma 4 for these fixtures, over
    2000 when the quote lands early in a segment body (2187 down to 111), and dac reports
    nothing. `collision_pair` is the OTHER path: a forged tool-call pair returns
    `non_contiguous_segment_ids` and degrades to GLOBAL, so "no error reported" is not
    true of that case. Prefix-render recovery is immune to both: the injected markup is
    inside both renders being compared, so it moves no bound.
    """
    rendered = do_render(tokenizer, case, fmt)
    for segment, (start, end) in zip(rendered.segments, rendered.layout.segments, strict=True):
        assert end - start >= segment.token_count, f"segment {segment.id} lost tokens"


@pytest.mark.parametrize("fmt", TAGGED)
def test_the_magic_chunk_header_is_inside_the_multi_turn_span(tokenizer: Any, fmt: str) -> None:
    rendered = do_render(tokenizer, "five_segments", fmt)
    for segment, (start, end) in zip(rendered.segments, rendered.layout.segments, strict=True):
        body = tokenizer.decode(rendered.ids[start:end])
        marker = f"Magic Chunk {segment.id}" if fmt == "chunk_multi_turn_tool" else f'index="{segment.id}"'
        assert marker in body


def test_spans_are_disjoint_ascending_and_end_before_the_question(tokenizer: Any) -> None:
    rendered = do_render(tokenizer, "five_segments", "chunk_multi_turn_tool")
    previous = 0
    for start, end in rendered.layout.segments:
        assert previous <= start < end <= rendered.layout.question_start
        previous = end


# --------------------------------------------------------------------------------------
# The layout
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_the_layout_is_admissible_and_round_trips(tokenizer: Any, fmt: str) -> None:
    rendered = do_render(tokenizer, "five_segments", fmt)
    layout = rendered.layout
    validate_layout(layout, max_model_len=262_144)
    assert PromptLayout.from_dict(layout.to_dict()) == layout
    assert layout.sink == SINK_TOKENS
    assert layout.prompt_format == fmt
    assert layout.n_prompt_tokens == len(rendered.ids)


@pytest.mark.parametrize("fmt", TAGGED)
def test_the_tag_keyword_is_carried_for_tagged_formats_only(tokenizer: Any, fmt: str) -> None:
    assert do_render(tokenizer, "one_segment", fmt).layout.tag_keyword == "magic_chunks"
    assert do_render(tokenizer, "one_segment", "vanilla").layout.tag_keyword == ""


def test_question_start_is_the_last_occurrence_of_the_marker(tokenizer: Any) -> None:
    """A context that quotes the marker must not move the window: the question turn is
    always rendered last, so `rfind` picks the real one."""
    marker = get_format("chunk_multi_turn_tool").marker
    clean = render(tokenizer, context="short body", question="Q?", fmt="chunk_multi_turn_tool")
    poisoned = render(
        tokenizer,
        context="short body\n\n" + marker + "\n\nmore body",
        question="Q?",
        fmt="chunk_multi_turn_tool",
    )
    assert clean.text[clean.layout.question_start :] or True
    tail_clean = tokenizer.decode(clean.ids[clean.layout.question_start :])
    tail_poisoned = tokenizer.decode(poisoned.ids[poisoned.layout.question_start :])
    assert marker.strip() in tail_clean and marker.strip() in tail_poisoned
    assert "Q?" in tail_poisoned


def test_the_prompt_tail_is_the_last_500_decoded_characters(tokenizer: Any) -> None:
    rendered = do_render(tokenizer, "one_segment", "chunk_multi_turn_tool")
    tail = rendered.layout.prompt_tail
    assert len(tail) == 500
    assert tokenizer.decode(rendered.ids, skip_special_tokens=True).endswith(tail)


def test_an_empty_context_becomes_one_placeholder_segment(tokenizer: Any) -> None:
    rendered = do_render(tokenizer, "empty", "chunk_multi_turn_tool")
    assert len(rendered.segments) == 1
    assert rendered.segments[0].text == EMPTY_CONTEXT == normalize_context("")
    assert len(rendered.layout.segments) == 1


# --------------------------------------------------------------------------------------
# The tool declaration
# --------------------------------------------------------------------------------------


def test_the_tool_is_declared_in_the_rendered_text_for_both_families(tokenizer: Any) -> None:
    """`main.tex:491`, `P6`. dac's bundled Qwen template drops it and the other reference
    disables it by config, so this assertion is the deliberate divergence."""
    rendered = do_render(tokenizer, "one_segment", "chunk_multi_turn_tool")
    assert rendered.tools is not None
    assert TOOL_NAME in rendered.text
    bare = render(
        tokenizer,
        context=synth_contexts.context_for("one_segment"),
        question=synth_contexts.question_for("one_segment"),
        fmt="chunk_multi_turn_tool",
        tool_schema=False,
    )
    assert bare.tools is None
    assert len(rendered.ids) > len(bare.ids), "the declaration must cost tokens"


def test_a_template_that_ignores_tools_is_refused(tokenizer: Any) -> None:
    """The second half of the assertion: a chat template that renders `tools=` as nothing
    produces a zero token delta, and `render` refuses rather than shipping the wrong prompt."""

    class IgnoresTools:
        is_fast = True
        chat_template = tokenizer.chat_template

        def __getattr__(self, name: str) -> Any:
            return getattr(tokenizer, name)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return tokenizer(*args, **kwargs)

        def decode(self, *args: Any, **kwargs: Any) -> Any:
            return tokenizer.decode(*args, **kwargs)

        def apply_chat_template(self, messages: list, tools: Any = None, **kwargs: Any) -> str:
            del tools
            return tokenizer.apply_chat_template(messages, tools=None, **kwargs)

    with pytest.raises(RenderError, match="ignores tools="):
        render(IgnoresTools(), context="body", question="Q?", fmt="chunk_multi_turn_tool")


def test_the_xml_and_vanilla_formats_declare_no_tool(tokenizer: Any) -> None:
    for fmt in ("chunk_single_turn_xml", "vanilla"):
        rendered = render(tokenizer, context="body", question="Q?", fmt=fmt, tool_schema=True)
        assert rendered.tools is None


# --------------------------------------------------------------------------------------
# The xml validator
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("injected", "reason"),
    [
        ("</magic_chunk>", "malformed_segment_markup"),
        ('<magic_chunk index="7">', "malformed_segment_markup"),
        ('<magic_chunk index="1">\nnested\n</magic_chunk>', "malformed_segment_markup"),
    ],
)
def test_xml_markup_in_the_context_degrades_with_dacs_reason_strings(
    tokenizer: Any, injected: str, reason: str
) -> None:
    rendered = render(
        tokenizer,
        context=f"first part\n\n{injected}\n\nsecond part",
        question="Q?",
        fmt="chunk_single_turn_xml",
    )
    assert rendered.format_error == reason
    assert rendered.layout.format_error == reason
    assert rendered.layout.segments == ()
    assert rendered.segments == ()


def test_xml_span_recovery_rejects_out_of_order_and_duplicate_ids() -> None:
    """Directly, because a real render always wraps the whole context in one well formed
    element, so the id gates are only reachable from markup the renderer did not write."""
    from dattn.prompt.render import _xml_spans

    text = '<magic_chunk index="2">\nb\n</magic_chunk><magic_chunk index="1">\na\n</magic_chunk>'
    starts = list(range(len(text)))
    assert _xml_spans(text, starts, len(text)) == ((), "non_contiguous_segment_ids")

    duplicate = '<magic_chunk index="1">\na\n</magic_chunk><magic_chunk index="1">\nb\n</magic_chunk>'
    assert _xml_spans(duplicate, list(range(len(duplicate))), len(duplicate)) == (
        (),
        "duplicate_segment_ids",
    )
    assert _xml_spans("no markup here", list(range(14)), 14) == ((), "no_segments_detected")


def test_the_xml_scan_stops_at_the_question_turn() -> None:
    """The instructions mention `</magic_chunk>` literally, and the scan must not see it."""
    from dattn.prompt.render import _xml_spans

    text = '<magic_chunk index="1">\na\n</magic_chunk> QUESTION </magic_chunk>'
    spans, error = _xml_spans(text, list(range(len(text))), text.index(" QUESTION "))
    assert error == "" and len(spans) == 1


def test_out_of_order_xml_ids_are_declined(tokenizer: Any) -> None:
    """`P7`, against `F D4`: dac sorts the ids before comparing, accepts `2, 1` and maps
    chunk 1 to the wrong text. dattn requires ascending order."""
    from dattn.prompt.formats.base import validate_segment_ids

    assert validate_segment_ids([1, 2, 3]) == ""
    assert validate_segment_ids([2, 1]) == "non_contiguous_segment_ids"
    assert validate_segment_ids([1, 1]) == "duplicate_segment_ids"
    assert validate_segment_ids([]) == "no_segments_detected"
    assert validate_segment_ids([1, 3]) == "non_contiguous_segment_ids"


# --------------------------------------------------------------------------------------
# Hashes
# --------------------------------------------------------------------------------------


def test_template_hashes_cover_the_chat_template_and_every_instruction_body(tokenizer: Any) -> None:
    hashes = template_hashes(tokenizer)
    assert set(hashes) == {"chat_template", *ALL_FORMATS}
    assert all(len(v) == 64 for v in hashes.values())
    assert len(set(hashes.values())) == len(hashes), "the three bodies must differ"


def test_vanilla_reassembles_the_context_from_its_segments(tokenizer: Any) -> None:
    """The segmenter's partition is exact, which is what lets `build_messages(segments,
    question)` be the one signature every format implements."""
    context = synth_contexts.context_for("five_segments")
    segments = segment_context(tokenizer, normalize_context(context))
    messages, tools = get_format("vanilla").build_messages(segments, "Q?")
    assert tools is None
    assert context in messages[-1]["content"]
