"""The paper headline format (`main.tex:491,2432`).

`get_magic_chunk` declared through the model's native tool format, one assistant
tool-call turn plus one tool-response turn per segment, body header `Magic Chunk {id}`.

The declaration travels as `tools=` into `tokenizer.apply_chat_template`, so each model
family renders it in its own dialect. That is the paper's claim and it is a deliberate
divergence from both references, one of which drops the declaration on Qwen entirely and
the other of which disables it by config (docs/PROVENANCE.md P6).
"""

from collections.abc import Sequence
from typing import Final

from dattn.contracts import Segment
from dattn.prompt.formats.base import DEFAULT_SYSTEM, TOOL_NAME, load_template, question_marker

NAME = "chunk_multi_turn_tool"
BODY_HEADER = "Magic Chunk {id}"
TAG_KEYWORD = "magic_chunks"

#: The cover story, as the first user turn. A request that the assistant retrieve the
#: document with `get_magic_chunk`, which the synthetic tool turns then visibly fulfil.
BOOTSTRAP: Final[str] = (
    "A question will follow about a long document. First, retrieve the document "
    "using the `get_magic_chunk` tool, one magic chunk at a time, in order from "
    'magic chunk id "1", until every magic chunk has been retrieved.'
)

#: OpenAI-style declaration. Rendered natively per family by the chat template.
GET_MAGIC_CHUNK_TOOL: Final[dict] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Retrieve one magic chunk of the user's document by its 1-indexed "
            "id. The document is split into consecutive magic chunks, which are "
            "arbitrary retrieval units and do not align with the document's own "
            "sections or chapters."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": 'The magic chunk id to retrieve (e.g. "1", "2", ...).',
                },
            },
            "required": ["id"],
        },
    },
}

__all__ = ["BODY_HEADER", "BOOTSTRAP", "GET_MAGIC_CHUNK_TOOL", "NAME", "TAG_KEYWORD", "ChunkMultiTurnTool"]


class ChunkMultiTurnTool:
    name = NAME
    tag_keyword = TAG_KEYWORD
    tagged = True

    def __init__(
        self,
        template_text: str | None = None,
        *,
        tool: dict | None = None,
        system: str = DEFAULT_SYSTEM,
        bootstrap: str = BOOTSTRAP,
    ) -> None:
        self.template_text = load_template(NAME) if template_text is None else template_text
        self.tool = GET_MAGIC_CHUNK_TOOL if tool is None else tool
        self.system = system
        self.bootstrap = bootstrap
        self.marker = question_marker(self.template_text)

    def build_messages(self, segments: Sequence[Segment], question: str) -> tuple[list[dict], list[dict] | None]:
        """System turn, bootstrap turn, one (tool call, tool response) pair per segment,
        then the question turn. `tools` is `None` when there is nothing to retrieve."""
        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        if segments:
            messages.append({"role": "user", "content": self.bootstrap})
            for seg in segments:
                call_id = f"call_{seg.id}"
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        # A dict, not a json.dumps string: the Qwen3.5 and Qwen3.6
                        # templates iterate `arguments` as a mapping.
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": TOOL_NAME, "arguments": {"id": str(seg.id)}},
                            }
                        ],
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": f"{BODY_HEADER.format(id=seg.id)}\n{seg.text}",
                    }
                )
        messages.append({"role": "user", "content": self.template_text.format(question=question)})
        return messages, ([self.tool] if segments else None)
