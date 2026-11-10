"""The `Format` protocol and the format registry.

| format | tool / element | tag keyword | tags |
|---|---|---|---|
| `chunk_multi_turn_tool` | `get_magic_chunk`, native tool format, paper headline | `magic_chunks` | yes |
| `chunk_single_turn_xml` | inline `<magic_chunk index="{id}">`, ablation | `magic_chunks` | yes |
| `vanilla` | context inline, no scaffolding | n/a | no |

Names are dac's, verbatim (DESIGN.md O9).

A format owns its instruction body and, where it has one, its tool declaration. Both are
constructor arguments with a shipped default, so a test can pin byte equality against a
reference renderer by handing the same body in rather than by editing the package.
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Final, Protocol, runtime_checkable

from dattn.contracts import RenderError, Segment

TOOL_NAME: Final[str] = "get_magic_chunk"

#: The system turn every format prepends. Sixteen tokens of content-free scaffolding is
#: exactly what the fixed attention sink covers (`main.tex:1354`). This matches dac at
#: `fb1ef99`, but NOT the paper's Table 2 vanilla rollout, which predates dac `c995167`
#: and has no system turn. See `docs/PROVENANCE.md` P11 before quoting a vanilla number.
DEFAULT_SYSTEM: Final[str] = "You are a helpful assistant."

#: Rollouts run with thinking off. A template without the flag ignores it.
CHAT_TEMPLATE_KWARGS: Final[dict[str, object]] = {"enable_thinking": False}

TEMPLATE_DIR: Final[Path] = Path(__file__).resolve().parent.parent / "templates"

#: dac's fail-reason strings, kept so the histograms stay comparable.
DUPLICATE_SEGMENT_IDS: Final[str] = "duplicate_segment_ids"
NON_CONTIGUOUS_SEGMENT_IDS: Final[str] = "non_contiguous_segment_ids"
NO_SEGMENTS_DETECTED: Final[str] = "no_segments_detected"
MALFORMED_SEGMENT_MARKUP: Final[str] = "malformed_segment_markup"

SEGMENT_FAIL_REASONS: Final[tuple[str, ...]] = (
    DUPLICATE_SEGMENT_IDS,
    NON_CONTIGUOUS_SEGMENT_IDS,
    NO_SEGMENTS_DETECTED,
    MALFORMED_SEGMENT_MARKUP,
)

#: The `.format()` placeholder every tagged template carries.
QUESTION_PLACEHOLDER: Final[str] = "{question}"

__all__ = [
    "CHAT_TEMPLATE_KWARGS",
    "DEFAULT_SYSTEM",
    "DUPLICATE_SEGMENT_IDS",
    "FORMATS",
    "MALFORMED_SEGMENT_MARKUP",
    "NON_CONTIGUOUS_SEGMENT_IDS",
    "NO_SEGMENTS_DETECTED",
    "QUESTION_PLACEHOLDER",
    "SEGMENT_FAIL_REASONS",
    "TEMPLATE_DIR",
    "TOOL_NAME",
    "Format",
    "body_sha256",
    "get_format",
    "load_template",
    "question_marker",
    "validate_segment_ids",
]


@runtime_checkable
class Format(Protocol):
    name: str

    def build_messages(self, segments: Sequence[Segment], question: str) -> tuple[list[dict], list[dict] | None]: ...


#: Populated by `_register()` the first time a format is looked up.
FORMATS: Final[dict[str, Format]] = {}


def load_template(name: str) -> str:
    """Read one instruction body out of `dattn/prompt/templates`.

    Raises `RenderError` rather than `FileNotFoundError` so a packaging mistake surfaces as
    a dattn error with the searched path in it.
    """
    path = TEMPLATE_DIR / f"{name}.md"
    if not path.is_file():
        raise RenderError(f"instruction template {name!r} not found at {path}")
    return path.read_text(encoding="utf-8")


def body_sha256(text: str) -> str:
    """sha256 of one instruction body, for the boot-time golden-file assertion (A17)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def question_marker(template: str) -> str:
    """The last non-blank line before `{question}`, with its newline.

    This is the locator `render()` uses to find `question_start`: the first prompt token at
    or after the LAST occurrence of the marker. `rfind` is robust to the marker appearing
    inside the context, because the question turn always renders last. The one way it
    misfires is a template that repeats the marker after `{question}`, which this function
    refuses outright.
    """
    idx = template.rfind(QUESTION_PLACEHOLDER)
    if idx < 0:
        raise RenderError("instruction template has no {question} placeholder")
    lines = [line for line in template[:idx].splitlines(keepends=True) if line.strip()]
    if not lines:
        raise RenderError("instruction template has no text before {question}")
    marker = lines[-1]
    if marker in template[idx + len(QUESTION_PLACEHOLDER) :]:
        raise RenderError(f"question marker {marker!r} recurs after {{question}} in the instruction template")
    return marker


def validate_segment_ids(ids: Sequence[int]) -> str:
    """`""` when `ids` is exactly `1, 2, ... n` in that order, else a dac fail reason.

    Stricter than dac on purpose (docs/PROVENANCE.md P7): dac compares `sorted(ids)` to the range
    (`dac/src/dac/focus/base.py:80`), so `2, 1` passes there. dac then **resolves by id**, not
    by position, through a dict keyed by `segment_id` (`base.py:82`, lookup at `:177`), so
    chunk 1 still maps to the span labelled 1. dac is NOT wrong here. dattn requires ascending
    order because out-of-order ids mean span detection drifted, which is worth refusing rather
    than papering over. Duplicates keep their own reason so the histograms stay readable.
    """
    if not ids:
        return NO_SEGMENTS_DETECTED
    if len(set(ids)) != len(ids):
        return DUPLICATE_SEGMENT_IDS
    if list(ids) != list(range(1, len(ids) + 1)):
        return NON_CONTIGUOUS_SEGMENT_IDS
    return ""


def _register() -> None:
    """Import the three format modules once, on first lookup.

    A local import, because each format module imports this one for `DEFAULT_SYSTEM` and
    `load_template`, and a module-scope import here would be a cycle.
    """
    if FORMATS:
        return
    from dattn.prompt.formats.chunk_multi_turn_tool import ChunkMultiTurnTool
    from dattn.prompt.formats.chunk_single_turn_xml import ChunkSingleTurnXml
    from dattn.prompt.formats.vanilla import Vanilla

    for cls in (ChunkMultiTurnTool, ChunkSingleTurnXml, Vanilla):
        instance = cls()
        FORMATS[instance.name] = instance


def get_format(name: str) -> Format:
    """Look a format up by dac's name. Raises `RenderError` on an unknown name."""
    _register()
    try:
        return FORMATS[name]
    except KeyError:
        raise RenderError(f"unknown prompt format {name!r}, expected one of {sorted(FORMATS)}") from None
