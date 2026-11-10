"""The render dispatcher, span recovery and the tool-schema check.

Span recovery is by prefix render, never by regex. For `k = 1..n_segments`, render
`messages[:head + 2k] + [question_turn]`, binary-search the longest common prefix against
the `k-1` render, snap to a token boundary, and verify the recovered span contains both
`f"Magic Chunk {id}"` and the segment's own text. Then assert `len(spans) == n_segments`.

This avoids a reference-parser defect where a context
that quotes `"\n</tool_response><|im_end|>"` (Qwen) or `"<|\"|>}<tool_response|>"`
(Gemma 4) ends the affected segment span at the quoted text instead of the real turn
boundary. The collision fixtures here lose 30 tokens on Qwen3.6 and 424 on Gemma 4, and a
quote landing early in a segment body costs over 2000 (measured 2187 down to 111). dac
reports nothing in that case. When the context forges a whole tool-call pair it instead
returns `non_contiguous_segment_ids` and drops every span. No context in dac's evaluated
set triggers either path, so this is a hazard the design removes, not a
defect in a published number. Nothing here reads the rendered text looking for markup: the
bounds come from what two renders of the same conversation have in common, so injected
markup is inside both and changes nothing. Cost is O(n_segments) renders on the client,
paid once and never in the engine.

Every prefix render keeps the final question turn. Templates render a trailing assistant
turn differently from a trailing user turn (Qwen adds an empty think block), so cutting the
conversation short would change the very turns being measured.
"""

from bisect import bisect_left, bisect_right
from typing import Any, Final

from dattn.contracts import SINK_TOKENS, PromptLayout, Rendered, RenderError, Segment
from dattn.prompt.formats.base import (
    CHAT_TEMPLATE_KWARGS,
    TOOL_NAME,
    body_sha256,
    get_format,
    validate_segment_ids,
)
from dattn.prompt.formats.chunk_multi_turn_tool import NAME as CHUNK_MULTI_TURN_TOOL
from dattn.prompt.formats.chunk_single_turn_xml import NAME as CHUNK_SINGLE_TURN_XML
from dattn.prompt.formats.chunk_single_turn_xml import TAG_RE
from dattn.prompt.formats.vanilla import NAME as VANILLA
from dattn.prompt.segment import EMPTY_CONTEXT, SEGMENTERS, normalize_context, segment_context, segment_equal_length

#: How far back `_turn_end` looks for the newline that precedes a turn header.
_HEADER_LOOKBACK: Final[int] = 64

__all__ = ["render", "template_hashes"]


def render(
    tokenizer: Any,
    *,
    context: str,
    question: str,
    fmt: str,
    tool_schema: bool = True,
    tag_keyword: str = "magic_chunks",
    sink: int = SINK_TOKENS,
    chat_template: str | None = None,
    segmenter: str = "semantic",
    vanilla_system: str | None = None,
) -> Rendered:
    """Render one request and recover every segment's token span.

    When `tool_schema` is true, asserts the rendered text contains the tool name and
    that the token delta against a `tools=None` render is non-zero, except for the
    pinned dac Qwen template which deliberately omits the schema for score parity.
    """
    fmt_obj = get_format(fmt)
    if fmt == VANILLA and vanilla_system is not None:
        from dattn.prompt.formats.vanilla import Vanilla

        fmt_obj = Vanilla(system=vanilla_system)
    if segmenter not in SEGMENTERS:
        raise ValueError(f"unknown segmenter {segmenter!r}, expected one of {SEGMENTERS}")
    context = normalize_context(context)
    segments = segment_context(tokenizer, context)
    n_semantic_segments = len(segments)
    if segmenter == "equal_length":
        segments = segment_equal_length(tokenizer, context, n_semantic_segments)
    messages, tools = fmt_obj.build_messages(segments, question)
    if not tool_schema:
        tools = None

    text = _apply(tokenizer, messages, tools, generation_prompt=True, chat_template=chat_template)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    ids = tuple(int(i) for i in enc["input_ids"])
    starts = [s for s, _ in enc["offset_mapping"]]
    if not ids:
        raise RenderError("the chat template produced an empty prompt")

    # dac's bundled Qwen template ignores tools, as verified by prompt parity.
    # Keep passing the tools object unchanged. Only this exact template may omit it.
    from dattn.prompt.template_policy import TEMPLATE_SHA256

    omits_schema = chat_template is not None and body_sha256(chat_template) == TEMPLATE_SHA256["qwen3_5"]
    if tools is not None and not omits_schema:
        _assert_tool_declaration(tokenizer, messages, text, len(ids), chat_template=chat_template)

    question_start = _question_start(text, starts, fmt_obj.marker, len(ids))

    format_error = ""
    spans: tuple[tuple[int, int], ...] = ()
    if fmt == CHUNK_MULTI_TURN_TOOL:
        spans = _tool_turn_spans(tokenizer, messages, tools, text, starts, segments, chat_template=chat_template)
    elif fmt == CHUNK_SINGLE_TURN_XML:
        spans, format_error = _xml_spans(text, starts, text.rfind(fmt_obj.marker))
        if format_error:
            spans, segments = (), []
    elif fmt != VANILLA:  # pragma: no cover - get_format already refused anything else
        raise RenderError(f"no span recovery for prompt format {fmt!r}")

    if fmt != VANILLA and not format_error and len(spans) != len(segments):
        raise RenderError(f"recovered {len(spans)} spans for {len(segments)} segments")

    layout = PromptLayout(
        n_prompt_tokens=len(ids),
        question_start=question_start,
        segments=spans,
        sink=sink,
        tag_keyword=tag_keyword if getattr(fmt_obj, "tagged", False) else "",
        prompt_format=fmt,
        prompt_tail=_decode_tail(tokenizer, ids),
        format_error=format_error,
    )
    return Rendered(
        text=text,
        ids=ids,
        layout=layout,
        messages=tuple(messages),
        tools=None if tools is None else tuple(tools),
        segments=tuple(segments),
        n_semantic_segments=n_semantic_segments,
        format_error=format_error,
    )


def template_hashes(tokenizer: Any) -> dict[str, str]:
    """sha256 of `tokenizer.chat_template` and of each instruction template body.

    Checked at boot by assertion A17. The tokenizer may legitimately carry no chat
    template, which is itself a fact worth pinning, so that case hashes the empty string
    and is reported as `""`.
    """
    out: dict[str, str] = {}
    chat_template = getattr(tokenizer, "chat_template", None)
    out["chat_template"] = "" if chat_template is None else body_sha256(str(chat_template))
    for name in (CHUNK_MULTI_TURN_TOOL, CHUNK_SINGLE_TURN_XML, VANILLA):
        out[name] = body_sha256(get_format(name).template_text)  # type: ignore[attr-defined]
    return out


# --------------------------------------------------------------------------------------
# Rendering helpers
# --------------------------------------------------------------------------------------


def _apply(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None,
    *,
    generation_prompt: bool,
    chat_template: str | None = None,
) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=generation_prompt,
            **CHAT_TEMPLATE_KWARGS,
            **({"chat_template": chat_template} if chat_template is not None else {}),
        )
    except Exception as exc:
        raise RenderError(
            f"the chat template rejects this conversation ({exc}); "
            f"use prompt_format={CHUNK_SINGLE_TURN_XML!r} for this model"
        ) from exc


def _assert_tool_declaration(
    tokenizer: Any, messages: list[dict], text: str, n_tokens: int, *, chat_template: str | None = None
) -> None:
    """The paper's claim, asserted rather than assumed (`main.tex:491`, `P6`).

    Two checks, because either alone is weak: the tool name must appear in the rendered
    text, and the token count must differ from the same conversation rendered with
    `tools=None`. A template that silently ignores `tools=` fails the second one even when
    the conversation happens to mention the tool by name.
    """
    if TOOL_NAME not in text:
        raise RenderError(f"the chat template dropped the {TOOL_NAME!r} declaration from the rendered prompt")
    bare = _apply(tokenizer, messages, None, generation_prompt=True, chat_template=chat_template)
    delta = n_tokens - len(tokenizer(bare, add_special_tokens=False)["input_ids"])
    if delta == 0:
        raise RenderError(
            f"the chat template ignores tools=: the {TOOL_NAME!r} declaration costs 0 tokens, "
            "so the prompt is not the one the paper describes"
        )


def _question_start(text: str, starts: list[int], marker: str, n_tokens: int) -> int:
    """The first token at or after the LAST occurrence of the question marker."""
    pos = text.rfind(marker)
    if pos < 0:
        raise RenderError(f"question marker {marker!r} is not in the rendered prompt")
    q = bisect_left(starts, pos)
    if q >= n_tokens:
        raise RenderError("the question marker resolves past the last prompt token")
    return q


def _decode_tail(tokenizer: Any, ids: tuple[int, ...], n_chars: int = 500) -> str:
    """The last `n_chars` characters of the decoded prompt, without decoding all of it.

    The FSM seeds its scan window with this, so the first generated tokens see the same
    text the reference implementation saw. Doubling out from a small suffix keeps a 264K
    prompt from being decoded in full just to read 500 characters off the end.
    """
    n = 256
    while True:
        tail = tokenizer.decode(ids[-n:], skip_special_tokens=True)
        if len(tail) >= n_chars + 8 or n >= len(ids):
            return tail[-n_chars:]
        n *= 4


# --------------------------------------------------------------------------------------
# Span recovery
# --------------------------------------------------------------------------------------


def _tool_turn_spans(
    tokenizer: Any,
    messages: list[dict],
    tools: list[dict] | None,
    text: str,
    starts: list[int],
    segments: list[Segment],
    *,
    chat_template: str | None = None,
) -> tuple[tuple[int, int], ...]:
    """Segment k is the k-th (tool call, tool response) turn pair.

    Turn ends come from consecutive prefix renders that both keep the question turn. One
    extra render with a dummy pair appended bounds the last real pair, so every bound is
    produced the same way and none is read off the rendered markup.
    """
    if not segments:
        return ()
    head_turns = 2 if messages[0]["role"] == "system" else 1
    question_turn = messages[-1]
    n = len(segments)

    head = _apply(tokenizer, messages[:head_turns], tools, generation_prompt=False, chat_template=chat_template)
    if not text.startswith(head):
        raise RenderError(
            "the chat template rewrites the bootstrap turn when later turns are added; "
            f"use prompt_format={CHUNK_SINGLE_TURN_XML!r} for this model"
        )

    dummy_id = str(n + 1)
    dummy_pair = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": f"call_{dummy_id}",
                    "type": "function",
                    "function": {"name": TOOL_NAME, "arguments": {"id": dummy_id}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": f"call_{dummy_id}", "content": f"Magic Chunk {dummy_id}\n{EMPTY_CONTEXT}"},
    ]
    renders = [
        _apply(
            tokenizer,
            messages[: head_turns + 2 * k] + [question_turn],
            tools,
            generation_prompt=True,
            chat_template=chat_template,
        )
        for k in range(1, n)
    ]
    renders.append(text)
    renders.append(
        _apply(
            tokenizer,
            messages[:-1] + dummy_pair + [question_turn],
            tools,
            generation_prompt=True,
            chat_template=chat_template,
        )
    )

    bounds = [len(head)] + [_turn_end(renders[k], renders[k + 1], starts) for k in range(n)]
    spans: list[tuple[int, int]] = []
    for k, seg in enumerate(segments):
        a = bounds[k]
        b = len(text[: bounds[k + 1]].rstrip())
        body = text[a:b]
        if a >= b or f"Magic Chunk {seg.id}" not in body or seg.text.strip() not in body:
            raise RenderError(
                f"segment {seg.id} is not rendered inside its own tool turn; the chat template "
                f"drops or rewrites tool turns, use prompt_format={CHUNK_SINGLE_TURN_XML!r} for this model"
            )
        ts, te = bisect_left(starts, a), bisect_left(starts, b)
        if ts >= te:
            raise RenderError(f"segment {seg.id} recovered an empty token span")
        spans.append((ts, te))
    return tuple(spans)


def _turn_end(shorter: str, longer: str, starts: list[int]) -> int:
    """The character offset where the turns two renders share come to an end.

    The longest common prefix over-reaches by whatever the two next turns happen to share.
    Two corrections, in this order:

    1. If it lands strictly inside a token, cut at that token's start. A partial token
       (`<` of `<|tool_call>` against `<turn|>`) belongs to neither turn.
    2. If it lands exactly on a token boundary right after a newline, and the characters
       since that newline look like a turn header (no spaces, no digits), give the header
       to the turn that follows.

    Without (2) a Gemma 4 span would start three tokens inside its own `<|turn>model`
    header, the recorded span difference between the two references.
    """
    lo, hi = 0, min(len(shorter), len(longer))
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if shorter[:mid] == longer[:mid]:
            lo = mid
        else:
            hi = mid - 1
    i = bisect_right(starts, lo) - 1
    if i >= 0 and starts[i] < lo:
        return starts[i]
    newline = shorter.rfind("\n", max(0, lo - _HEADER_LOOKBACK), lo)
    header = shorter[newline + 1 : lo]
    if newline >= 0 and header and not any(c.isspace() or c.isdigit() for c in header):
        return newline + 1
    return lo


def _xml_spans(text: str, starts: list[int], scan_end: int) -> tuple[tuple[tuple[int, int], ...], str]:
    """Strict open and close alternation before the question turn.

    A nested open, a stray close or a dangling open means a body collided with the tag
    markup, and the whole request degrades to GLOBAL rather than silently truncating a
    segment. Ids must then be contiguous AND ascending (`P7`).
    """
    found: list[tuple[int, int, int]] = []
    pending: tuple[int, int] | None = None
    for m in TAG_RE.finditer(text, 0, scan_end if scan_end >= 0 else len(text)):
        if m.group(1) is not None:
            if pending is not None:
                return (), "malformed_segment_markup"
            pending = (int(m.group(1)), m.start())
        else:
            if pending is None:
                return (), "malformed_segment_markup"
            cid, cs = pending
            ts, te = bisect_left(starts, cs), bisect_left(starts, m.end())
            if ts < te:
                found.append((cid, ts, te))
            pending = None
    if pending is not None:
        return (), "malformed_segment_markup"

    reason = validate_segment_ids([cid for cid, _, _ in found])
    if reason:
        return (), reason
    return tuple((s, e) for _, s, e in found), ""
