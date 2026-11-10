"""The three prompt formats (WP2): message shape, tool dict and instruction bodies.

No tokenizer here. `build_messages` is pure, and the whole table of DESIGN.md 1.2 is
checkable on the messages alone.
"""

import pytest

from dattn.contracts import RenderError, Segment
from dattn.prompt.formats import ChunkMultiTurnTool, ChunkSingleTurnXml, Vanilla
from dattn.prompt.formats.base import (
    DEFAULT_SYSTEM,
    FORMATS,
    TEMPLATE_DIR,
    TOOL_NAME,
    body_sha256,
    get_format,
    load_template,
    question_marker,
)
from dattn.prompt.formats.chunk_multi_turn_tool import BODY_HEADER, BOOTSTRAP, GET_MAGIC_CHUNK_TOOL

SEGMENTS = [
    Segment(1, "first body", 3, 0, 10),
    Segment(2, "second body", 3, 10, 21),
    Segment(3, "third body", 3, 21, 31),
]
QUESTION = "Which one is it?"


# --------------------------------------------------------------------------------------
# The registry
# --------------------------------------------------------------------------------------


def test_the_registry_holds_exactly_dacs_three_names() -> None:
    get_format("vanilla")  # force registration
    assert set(FORMATS) == {"chunk_multi_turn_tool", "chunk_single_turn_xml", "vanilla"}


def test_get_format_is_idempotent_and_returns_one_instance() -> None:
    assert get_format("vanilla") is get_format("vanilla")


@pytest.mark.parametrize("name", ["", "chunk", "bookmark", "magic_chunk", "declared"])
def test_unknown_names_are_refused(name: str) -> None:
    with pytest.raises(RenderError, match="unknown prompt format"):
        get_format(name)


# --------------------------------------------------------------------------------------
# chunk_multi_turn_tool, the paper headline
# --------------------------------------------------------------------------------------


def test_the_multi_turn_conversation_is_system_bootstrap_pairs_question() -> None:
    messages, tools = ChunkMultiTurnTool().build_messages(SEGMENTS, QUESTION)
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", *["assistant", "tool"] * 3, "user"]
    assert messages[0]["content"] == DEFAULT_SYSTEM == "You are a helpful assistant."
    assert messages[1]["content"] == BOOTSTRAP
    assert TOOL_NAME in BOOTSTRAP
    assert QUESTION in messages[-1]["content"]
    assert tools == [GET_MAGIC_CHUNK_TOOL]


def test_each_pair_is_a_tool_call_and_its_response_with_the_magic_chunk_header() -> None:
    messages, _ = ChunkMultiTurnTool().build_messages(SEGMENTS, QUESTION)
    for segment, call, response in zip(SEGMENTS, messages[2::2], messages[3::2], strict=False):
        assert call["content"] == ""
        function = call["tool_calls"][0]["function"]
        assert function["name"] == TOOL_NAME
        # A mapping, not a json.dumps string: the Qwen templates iterate it.
        assert function["arguments"] == {"id": str(segment.id)}
        assert call["tool_calls"][0]["id"] == response["tool_call_id"] == f"call_{segment.id}"
        assert response["content"] == f"Magic Chunk {segment.id}\n{segment.text}"
    assert BODY_HEADER == "Magic Chunk {id}"


def test_no_segments_means_no_scaffolding_and_no_tool() -> None:
    """A short document delivered inline is the GLOBAL-degrade case: there is nothing to
    retrieve, so declaring the tool would be a lie."""
    messages, tools = ChunkMultiTurnTool().build_messages([], QUESTION)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert tools is None


def test_the_tool_declaration_is_the_openai_shape_the_chat_templates_expect() -> None:
    function = GET_MAGIC_CHUNK_TOOL["function"]
    assert GET_MAGIC_CHUNK_TOOL["type"] == "function"
    assert function["name"] == TOOL_NAME == "get_magic_chunk"
    assert function["parameters"]["required"] == ["id"]
    assert function["parameters"]["properties"]["id"]["type"] == "string"


def test_the_instruction_body_can_be_overridden_without_touching_the_package() -> None:
    """How the byte-equality harness pins dattn against another renderer's template."""
    body = "# Question\n\n{question}\n\n# Instructions\n\nAnswer.\n"
    fmt = ChunkMultiTurnTool(body)
    messages, _ = fmt.build_messages(SEGMENTS, QUESTION)
    assert messages[-1]["content"] == body.format(question=QUESTION)
    assert fmt.marker == "# Question\n"


# --------------------------------------------------------------------------------------
# chunk_single_turn_xml, the ablation
# --------------------------------------------------------------------------------------


def test_the_xml_body_wraps_each_segment_in_its_own_element() -> None:
    messages, tools = ChunkSingleTurnXml().build_messages(SEGMENTS, QUESTION)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert tools is None
    body = messages[-1]["content"]
    for segment in SEGMENTS:
        assert f'<magic_chunk index="{segment.id}">\n{segment.text}\n</magic_chunk>' in body
    assert body.count("<magic_chunk index=") == len(SEGMENTS) + 1, "the template mentions one itself"


def test_the_xml_elements_are_in_ascending_id_order() -> None:
    messages, _ = ChunkSingleTurnXml().build_messages(SEGMENTS, QUESTION)
    body = messages[-1]["content"]
    positions = [body.index(f'<magic_chunk index="{s.id}">') for s in SEGMENTS]
    assert positions == sorted(positions)


# --------------------------------------------------------------------------------------
# vanilla
# --------------------------------------------------------------------------------------


def test_vanilla_puts_the_context_inline_with_no_scaffolding() -> None:
    messages, tools = Vanilla().build_messages(SEGMENTS, QUESTION)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert tools is None
    body = messages[-1]["content"]
    assert "".join(s.text for s in SEGMENTS) in body
    assert "magic_chunk" not in body
    assert TOOL_NAME not in body


def test_vanilla_is_untagged_and_the_other_two_are_tagged() -> None:
    assert Vanilla().tagged is False
    assert Vanilla().tag_keyword == ""
    assert ChunkMultiTurnTool().tag_keyword == ChunkSingleTurnXml().tag_keyword == "magic_chunks"


# --------------------------------------------------------------------------------------
# Templates
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["chunk_multi_turn_tool", "chunk_single_turn_xml", "vanilla"])
def test_every_shipped_template_exists_and_has_a_usable_question_marker(name: str) -> None:
    body = load_template(name)
    assert "{question}" in body
    marker = question_marker(body)
    assert marker.endswith("\n") and marker.strip()
    assert len(body_sha256(body)) == 64


def test_a_missing_template_is_a_render_error() -> None:
    with pytest.raises(RenderError, match="not found"):
        load_template("no_such_template")


def test_the_two_tagged_templates_document_the_tags_the_fsm_reads() -> None:
    for name in ("chunk_multi_turn_tool", "chunk_single_turn_xml"):
        body = load_template(name)
        assert '<focus magic_chunks="K">' in body
        assert "<local>" in body and "<answer>" in body and "<global>" in body


def test_only_the_multi_turn_template_mentions_the_tool() -> None:
    assert TOOL_NAME in load_template("chunk_multi_turn_tool")
    assert TOOL_NAME not in load_template("chunk_single_turn_xml")
    assert TOOL_NAME not in load_template("vanilla")


def test_the_xml_and_vanilla_templates_carry_a_context_placeholder() -> None:
    assert "{context}" in load_template("chunk_single_turn_xml")
    assert "{context}" in load_template("vanilla")
    assert "{context}" not in load_template("chunk_multi_turn_tool")


def test_templates_live_in_the_package_so_they_ship_with_it() -> None:
    assert TEMPLATE_DIR.is_dir()
    assert TEMPLATE_DIR.parent.name == "prompt"
    assert {p.stem for p in TEMPLATE_DIR.glob("*.md")} == {
        "chunk_multi_turn_tool",
        "chunk_single_turn_xml",
        "vanilla",
    }


def test_a_template_that_repeats_its_marker_after_the_question_is_refused() -> None:
    """dac's `validate_question_marker` check: `rfind` would otherwise pick the trailing
    occurrence and start the local window in the wrong place."""
    with pytest.raises(RenderError, match="recurs after"):
        question_marker("# Question\n\n{question}\n\n# Question\n\nmore\n")
    with pytest.raises(RenderError, match="no .question. placeholder"):
        question_marker("no placeholder here")
    with pytest.raises(RenderError, match="no text before"):
        question_marker("{question}\n")
