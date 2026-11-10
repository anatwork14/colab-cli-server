"""Render eval requests through the shared prompt implementation and encode HTTP payloads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from dattn.contracts import DattnRequest, PromptLayout

__all__ = [
    "PROMPT_FORMATS",
    "BuiltPrompt",
    "build_prompt",
    "dattn_extra_args",
    "dattn_xargs",
    "prompt_sha256",
]

#: The formats eval drives. `chunk_multi_turn_tool` is the paper's headline arm,
#: `chunk_single_turn_xml` the format ablation, `vanilla` the baseline (`DESIGN.md` 1.2).
#: da_eqseg keeps `chunk_multi_turn_tool` and selects the equal_length segmenter.
PROMPT_FORMATS: tuple[str, ...] = ("chunk_multi_turn_tool", "chunk_single_turn_xml", "vanilla")


@dataclass(frozen=True, slots=True)
class BuiltPrompt:
    """One rendered request, plus the two things the parity tool aligns on."""

    text: str
    ids: tuple[int, ...]
    layout: PromptLayout
    format_error: str
    n_segments: int
    n_semantic_segments: int | None = None

    @property
    def sha256(self) -> str:
        return prompt_sha256(self.text)

    @property
    def n_prompt_tokens(self) -> int:
        return len(self.ids) or self.layout.n_prompt_tokens


def prompt_sha256(text: str) -> str:
    """The rendered-prompt hash the parity report compares.

    Hashing the rendered TEXT rather than the token ids is deliberate: two trees can use
    different tokenizer revisions, and the text is the thing the protocol specifies. The
    token count is compared separately and is allowed to differ by the documented
    tool-schema delta (`PROVENANCE.md` P6).
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompt(
    tokenizer: Any,
    *,
    context: str,
    question: str,
    fmt: str,
    tool_schema: bool = True,
    tag_keyword: str = "magic_chunks",
    chat_template: str | None = None,
    segmenter: str = "semantic",
    vanilla_system: str | None = None,
) -> BuiltPrompt:
    """Render one row through the shared prompt implementation."""
    if fmt not in PROMPT_FORMATS:
        raise ValueError(f"unknown prompt format {fmt!r}, expected one of {PROMPT_FORMATS}")
    from dattn.prompt.render import render

    rendered = render(
        tokenizer,
        context=context,
        question=question,
        fmt=fmt,
        tool_schema=tool_schema,
        tag_keyword=tag_keyword,
        chat_template=chat_template,
        segmenter=segmenter,
        vanilla_system=vanilla_system,
    )
    return BuiltPrompt(
        text=rendered.text,
        ids=tuple(rendered.ids),
        layout=rendered.layout,
        format_error=rendered.format_error,
        n_segments=len(rendered.segments),
        n_semantic_segments=rendered.n_semantic_segments,
    )


def dattn_extra_args(layout: PromptLayout, *, enabled: bool = True) -> dict[str, object]:
    """The offline-API payload: `SamplingParams(extra_args={"dattn": <this>})`."""
    return DattnRequest(layout=layout, enabled=enabled).to_extra_args()


def dattn_xargs(layout: PromptLayout, *, enabled: bool = True) -> dict[str, str]:
    """The HTTP payload: `{"vllm_xargs": <this>}` on a chat-completions request.

    vLLM 0.29.0 types `vllm_xargs` as `dict[str, str | int | float | list[...]]`
    (`vllm/entrypoints/openai/chat_completion/protocol.py:492`), so a nested dict cannot
    be sent as-is and the payload travels as a JSON string. The engine side accepts
    either form so offline and HTTP requests carry the same layout.
    """
    return {"dattn": json.dumps(dattn_extra_args(layout, enabled=enabled), separators=(",", ":"))}
