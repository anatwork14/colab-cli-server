"""The xml ablation format.

Inline `<magic_chunk index="{id}">\\n{text}\\n</magic_chunk>` in one user turn,
`tools=None`. Span recovery walks the open and close tags requiring strict alternation and
then requires the ids to be contiguous AND ascending, which is stricter than dac
(docs/PROVENANCE.md P7). Fail-reason strings stay dac's so the histograms remain comparable.
"""

import re
from collections.abc import Sequence
from typing import Final

from dattn.contracts import Segment
from dattn.prompt.formats.base import DEFAULT_SYSTEM, load_template, question_marker

NAME = "chunk_single_turn_xml"
TAG_KEYWORD = "magic_chunks"

#: One alternative per tag: an open `<magic_chunk index="N">` capturing the id, or a close.
#: The id must be digits, so the template's own literal `index="N"` never matches.
TAG_RE: Final[re.Pattern[str]] = re.compile(r'<magic_chunk index="(\d+)">|</magic_chunk>')

__all__ = ["NAME", "TAG_KEYWORD", "TAG_RE", "ChunkSingleTurnXml", "wrap_segments"]


def wrap_segments(segments: Sequence[Segment]) -> str:
    """The inline body: each segment wrapped in its own element, two newlines apart."""
    return "".join(f'\n\n<magic_chunk index="{s.id}">\n{s.text}\n</magic_chunk>' for s in segments)


class ChunkSingleTurnXml:
    name = NAME
    tag_keyword = TAG_KEYWORD
    tagged = True

    def __init__(self, template_text: str | None = None, *, system: str = DEFAULT_SYSTEM) -> None:
        self.template_text = load_template(NAME) if template_text is None else template_text
        self.system = system
        self.marker = question_marker(self.template_text)

    def build_messages(self, segments: Sequence[Segment], question: str) -> tuple[list[dict], list[dict] | None]:
        """System turn plus one user turn carrying the wrapped context and the question."""
        body = wrap_segments(segments)
        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.template_text.format(context=body, question=question)})
        return messages, None
