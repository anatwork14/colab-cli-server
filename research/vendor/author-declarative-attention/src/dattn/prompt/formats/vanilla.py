"""The vanilla baseline format: context inline, no scaffolding, no tool.

The context is reassembled from the segment texts, which is exact: the segmenter's char
ranges are a half-open partition of the context, so concatenating the texts reproduces it
byte for byte. That keeps `Format.build_messages(segments, question)` as the one signature
every format implements, with no side channel for the raw context.
"""

from collections.abc import Sequence

from dattn.contracts import Segment
from dattn.prompt.formats.base import DEFAULT_SYSTEM, load_template, question_marker

NAME = "vanilla"

__all__ = ["NAME", "Vanilla"]


class Vanilla:
    name = NAME
    #: No tags, so no keyword. The FSM stays GLOBAL for the whole request.
    tag_keyword = ""
    tagged = False

    def __init__(self, template_text: str | None = None, *, system: str = DEFAULT_SYSTEM) -> None:
        self.template_text = load_template(NAME) if template_text is None else template_text
        self.system = system
        self.marker = question_marker(self.template_text)

    def build_messages(self, segments: Sequence[Segment], question: str) -> tuple[list[dict], list[dict] | None]:
        context = "".join(s.text for s in segments)
        messages: list[dict] = []
        if self.system:
            messages.append({"role": "system", "content": self.system})
        messages.append({"role": "user", "content": self.template_text.format(context=context, question=question)})
        return messages, None
