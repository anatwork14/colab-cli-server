"""The four tag regexes and the focus-id parser.

Exactly four end-anchored regexes with an eight-character tail, case sensitive
(DESIGN.md section 1.1). The reference's parsing behavior is carried forward
deliberately, each with a named unit test in `tests/unit/fsm/test_tags.py`, because these
are what the published numbers were produced under:

* `int()` accepts `"+1"`, `"01"`, `"1_0"` (which becomes id 10) and fullwidth digits.
* `"1,"` and `",1"` drop their empty pieces.
* Duplicate ids are accepted.
* A single-quoted attribute, a space before `=`, a trailing space and a self-closing `/>`
  are all `syntax_error`, because `_ID_RE` demands `">` immediately after the value.
* `<FOCUS>` is not seen at all, the regexes are case sensitive.
* `<global>` and `</global>` are inert: no regex mentions them.
* `<answer>` aliases `<local>`, and `<local>...</answer>` closes.

The tail is `.{0,8}$` with no `re.DOTALL`, so a tag followed by nine or more non-newline
characters inside one token is never seen, and a newline between the tag and the end of
the window kills the match. Both references share that blind spot (`F 5.2`).
"""

import re
from typing import Final

from dattn.contracts import SCAN_CHARS

RE_FOCUS_OPEN: Final[re.Pattern[str]] = re.compile(r"<focus[^>]*>.{0,8}$")
RE_FOCUS_CLOSE: Final[re.Pattern[str]] = re.compile(r"</focus>.{0,8}$")
RE_LOCAL_OPEN: Final[re.Pattern[str]] = re.compile(r"<(?:local|answer)>.{0,8}$")
RE_LOCAL_CLOSE: Final[re.Pattern[str]] = re.compile(r"</(?:local|answer)>.{0,8}$")
_ID_SPLIT: Final[re.Pattern[str]] = re.compile(r"[,\s]+")

# The five index-path fail reasons plus the segmenter's own, in gate order.
SEGMENT_FORMAT_ERROR: Final[str] = "segment_format_error"
SYNTAX_ERROR: Final[str] = "syntax_error"
EMPTY_INDEX_LIST: Final[str] = "empty_index_list"
MORE_THAN_THREE_INDICES: Final[str] = "more_than_three_indices"
OUT_OF_RANGE_INDEX: Final[str] = "out_of_range_index"

FOCUS_FAIL_REASONS: Final[tuple[str, ...]] = (
    SEGMENT_FORMAT_ERROR,
    SYNTAX_ERROR,
    EMPTY_INDEX_LIST,
    MORE_THAN_THREE_INDICES,
    OUT_OF_RANGE_INDEX,
)

#: `parse_focus_ids` returns this instead of a reason when the tag resolved.
NO_FAILURE: Final[str] = ""

__all__ = [
    "EMPTY_INDEX_LIST",
    "FOCUS_FAIL_REASONS",
    "MORE_THAN_THREE_INDICES",
    "NO_FAILURE",
    "OUT_OF_RANGE_INDEX",
    "RE_FOCUS_CLOSE",
    "RE_FOCUS_OPEN",
    "RE_LOCAL_CLOSE",
    "RE_LOCAL_OPEN",
    "SCAN_CHARS",
    "SEGMENT_FORMAT_ERROR",
    "SYNTAX_ERROR",
    "id_regex",
    "parse_focus_ids",
    "scan_tail",
]


def id_regex(keyword: str) -> re.Pattern[str]:
    """`_ID_RE` of DESIGN.md section 1.1, built for one tag keyword.

    `\\s+` before the keyword means `<focus  magic_chunks="1">` with two spaces parses,
    while `<focus magic_chunks ="1">` does not. The keyword is escaped so a caller cannot
    smuggle a pattern in through the layout.
    """
    return re.compile(rf'<focus\s+{re.escape(keyword)}="([^"]*)">')


def scan_tail(text: str, scan_chars: int = SCAN_CHARS) -> str:
    """The tag scan window: the last `scan_chars` characters of `text`, stripped.

    The strip is what lets an end-anchored regex fire when the model emits a trailing
    newline in the same token as the `>`. Both references strip, and the parity tests pin
    the combination (the eight-versus-nine trailing-character cases).
    """
    return text[-scan_chars:].strip()


def parse_focus_ids(tail: str, keyword: str, n_segments: int, max_ids: int) -> tuple[tuple[int, ...], str]:
    """Parse the focus ids out of one detected `<focus ...>` tag.

    Returns `(ids, fail_reason)`. `fail_reason` is `NO_FAILURE` on success and otherwise
    one of `FOCUS_FAIL_REASONS`. The ids parsed so far are returned alongside a failure
    when there are any, which is what the metrics need for a declined tag.

    Gate order is pinned by the parity tests and is exactly: no segments ->
    `segment_format_error`, no `_ID_RE` match -> `syntax_error`, blank id list ->
    `empty_index_list`, non-integer piece -> `syntax_error`, more than `max_ids` ids ->
    `more_than_three_indices`, unknown id -> `out_of_range_index`.
    """
    if n_segments == 0:
        return (), SEGMENT_FORMAT_ERROR
    match = id_regex(keyword).search(tail)
    if match is None:
        return (), SYNTAX_ERROR
    raw = match.group(1).strip()
    if not raw:
        return (), EMPTY_INDEX_LIST
    try:
        ids = tuple(int(piece) for piece in _ID_SPLIT.split(raw) if piece)
    except ValueError:
        return (), SYNTAX_ERROR
    if not ids:
        # Every piece was empty, e.g. `","`. Both references land here, not on syntax.
        return (), EMPTY_INDEX_LIST
    if len(ids) > max_ids:
        return ids, MORE_THAN_THREE_INDICES
    if any(not 1 <= i <= n_segments for i in ids):
        return ids, OUT_OF_RANGE_INDEX
    return ids, NO_FAILURE
