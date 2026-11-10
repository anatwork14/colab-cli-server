"""The tag grammar (WP1): one named test per row of the table in `F 3.3`.

These behaviours are surprising on purpose. They are what the published numbers were
produced under, so each one is pinned here by name rather than left to the fixture replay
to catch.
"""

import pytest

from dattn.fsm.tags import (
    EMPTY_INDEX_LIST,
    FOCUS_FAIL_REASONS,
    MORE_THAN_THREE_INDICES,
    NO_FAILURE,
    OUT_OF_RANGE_INDEX,
    RE_FOCUS_CLOSE,
    RE_FOCUS_OPEN,
    RE_LOCAL_CLOSE,
    RE_LOCAL_OPEN,
    SCAN_CHARS,
    SEGMENT_FORMAT_ERROR,
    SYNTAX_ERROR,
    id_regex,
    parse_focus_ids,
    scan_tail,
)

KW = "magic_chunks"


def parse(tag: str, n_segments: int = 5, max_ids: int = 3) -> tuple[tuple[int, ...], str]:
    return parse_focus_ids(tag, KW, n_segments, max_ids)


# --------------------------------------------------------------------------------------
# Accepted
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "ids"),
    [
        ('<focus magic_chunks="1">', (1,)),
        ('<focus magic_chunks="1,2">', (1, 2)),
        ('<focus magic_chunks="1,2,3">', (1, 2, 3)),
        ('<focus magic_chunks="1, 2 , 3">', (1, 2, 3)),
        ('<focus magic_chunks="1 2 3">', (1, 2, 3)),
        ('<focus magic_chunks="1\n2">', (1, 2)),
        ('<focus  magic_chunks="1">', (1,)),
    ],
)
def test_well_formed_tags_resolve(raw: str, ids: tuple[int, ...]) -> None:
    assert parse(raw) == (ids, NO_FAILURE)


def test_int_accepts_a_leading_plus() -> None:
    assert parse('<focus magic_chunks="+1">') == ((1,), NO_FAILURE)


def test_int_accepts_a_leading_zero() -> None:
    assert parse('<focus magic_chunks="01">') == ((1,), NO_FAILURE)


def test_int_accepts_a_python_underscore_literal_and_1_0_becomes_10() -> None:
    """The most surprising of the lot: a model writing `1_0` addresses chunk 10."""
    assert parse('<focus magic_chunks="1_0">', n_segments=20) == ((10,), NO_FAILURE)
    assert parse('<focus magic_chunks="1_0">', n_segments=5) == ((10,), OUT_OF_RANGE_INDEX)


def test_int_accepts_fullwidth_digits() -> None:
    assert parse('<focus magic_chunks="１">') == ((1,), NO_FAILURE)


@pytest.mark.parametrize("raw", ['<focus magic_chunks="1,">', '<focus magic_chunks=",1">'])
def test_empty_pieces_are_dropped(raw: str) -> None:
    assert parse(raw) == ((1,), NO_FAILURE)


def test_duplicate_ids_are_accepted_and_counted_twice() -> None:
    assert parse('<focus magic_chunks="1,1">') == ((1, 1), NO_FAILURE)
    assert parse('<focus magic_chunks="2,2,2">') == ((2, 2, 2), NO_FAILURE)


# --------------------------------------------------------------------------------------
# Declined, and the gate order
# --------------------------------------------------------------------------------------


def test_no_segments_beats_every_other_gate() -> None:
    """Even a syntactically broken tag reports `segment_format_error` when the map is
    empty, because that gate runs first."""
    assert parse("<focus>", n_segments=0) == ((), SEGMENT_FORMAT_ERROR)
    assert parse('<focus magic_chunks="">', n_segments=0) == ((), SEGMENT_FORMAT_ERROR)
    assert parse('<focus magic_chunks="1,2,3,4">', n_segments=0) == ((), SEGMENT_FORMAT_ERROR)


def test_a_blank_list_beats_the_count_gate() -> None:
    assert parse('<focus magic_chunks="">') == ((), EMPTY_INDEX_LIST)
    assert parse('<focus magic_chunks="   ">') == ((), EMPTY_INDEX_LIST)
    assert parse('<focus magic_chunks=",">') == ((), EMPTY_INDEX_LIST)


def test_a_non_integer_piece_beats_the_count_gate() -> None:
    assert parse('<focus magic_chunks="one, two, three, four">') == ((), SYNTAX_ERROR)


def test_the_count_gate_beats_the_range_gate() -> None:
    assert parse('<focus magic_chunks="1,2,3,99">') == ((1, 2, 3, 99), MORE_THAN_THREE_INDICES)


def test_more_than_max_ids() -> None:
    assert parse('<focus magic_chunks="1,2,3,4">')[1] == MORE_THAN_THREE_INDICES
    assert parse('<focus magic_chunks="1,2">', max_ids=1)[1] == MORE_THAN_THREE_INDICES


@pytest.mark.parametrize("value", ["0", "-1", "1,-2", "12", "999999999999999999999"])
def test_unknown_ids(value: str) -> None:
    assert parse(f'<focus magic_chunks="{value}">')[1] == OUT_OF_RANGE_INDEX


@pytest.mark.parametrize("value", ["one, two", "1.0", "1e3", "0x10"])
def test_non_integers_are_a_syntax_error(value: str) -> None:
    assert parse(f'<focus magic_chunks="{value}">') == ((), SYNTAX_ERROR)


@pytest.mark.parametrize(
    "raw",
    [
        "<focus magic_chunks='1'>",  # single quotes
        '<focus magic_chunks ="1">',  # a space before the equals
        '<focus magic_chunks="1" >',  # a trailing space
        '<focus magic_chunks="1"/>',  # self closing
        '<focus id="x" magic_chunks="1">',  # another attribute first
        '<focus magic_chunks="1" id="x">',  # another attribute last
        '<focus bookmarks="1">',  # the wrong keyword
        "<focus>",  # bare
    ],
)
def test_tags_the_parse_regex_refuses(raw: str) -> None:
    """`_ID_RE` demands `">` immediately after the value, so anything between the closing
    quote and the `>` is a syntax error."""
    assert parse(raw) == ((), SYNTAX_ERROR)


def test_every_fail_reason_is_reachable() -> None:
    reached = {
        parse("<focus>", n_segments=0)[1],
        parse("<focus>")[1],
        parse('<focus magic_chunks="">')[1],
        parse('<focus magic_chunks="1,2,3,4">')[1],
        parse('<focus magic_chunks="99">')[1],
    }
    assert reached == set(FOCUS_FAIL_REASONS)


def test_the_keyword_is_escaped_so_a_layout_cannot_smuggle_a_pattern_in() -> None:
    assert parse_focus_ids('<focus a.c="1">', "a.c", 5, 3) == ((1,), NO_FAILURE)
    assert parse_focus_ids('<focus abc="1">', "a.c", 5, 3) == ((), SYNTAX_ERROR)


# --------------------------------------------------------------------------------------
# The four regexes
# --------------------------------------------------------------------------------------


def test_the_regexes_are_case_sensitive_so_uppercase_focus_is_invisible() -> None:
    assert RE_FOCUS_OPEN.search('<focus magic_chunks="1">') is not None
    assert RE_FOCUS_OPEN.search('<FOCUS MAGIC_CHUNKS="1">') is None
    assert RE_FOCUS_CLOSE.search("</FOCUS>") is None


def test_answer_aliases_local_in_both_directions() -> None:
    assert RE_LOCAL_OPEN.search("<local>") is not None
    assert RE_LOCAL_OPEN.search("<answer>") is not None
    assert RE_LOCAL_CLOSE.search("</local>") is not None
    assert RE_LOCAL_CLOSE.search("</answer>") is not None


def test_global_is_inert_no_regex_mentions_it() -> None:
    for pattern in (RE_FOCUS_OPEN, RE_FOCUS_CLOSE, RE_LOCAL_OPEN, RE_LOCAL_CLOSE):
        assert pattern.search("<global>") is None
        assert pattern.search("</global>") is None


def test_the_tail_boundary_is_eight_characters_and_nine_is_invisible() -> None:
    assert RE_FOCUS_OPEN.search('<focus magic_chunks="1">' + "x" * 8) is not None
    assert RE_FOCUS_OPEN.search('<focus magic_chunks="1">' + "x" * 9) is None
    assert RE_FOCUS_CLOSE.search("</focus>" + "y" * 8) is not None
    assert RE_FOCUS_CLOSE.search("</focus>" + "y" * 9) is None


def test_a_newline_inside_the_tail_kills_the_match() -> None:
    """`.{0,8}` without `re.DOTALL`. Both references share this blind spot (`F 5.2`)."""
    assert RE_FOCUS_OPEN.search('<focus magic_chunks="1">\nx') is None
    assert RE_LOCAL_OPEN.search("<local>\nx") is None


def test_a_single_trailing_newline_still_matches_because_dollar_allows_one() -> None:
    """Python's `$` matches before one trailing newline, so a tag that ends the window
    with a single `\n` is still seen. The FSM strips the window anyway."""
    assert RE_LOCAL_OPEN.search("<local>\n") is not None
    assert RE_FOCUS_CLOSE.search("</focus>\n") is not None


def test_the_attribute_region_stops_at_the_first_close_bracket() -> None:
    """`<focus[^>]*>` cannot span a `>`, so the tag the FSM hands to the parser ends at
    the first one. A later `>` is only ever part of the eight-character tail."""
    match = RE_FOCUS_OPEN.search('<focus magic_chunks="1">a > b')
    assert match is not None
    assert match.group(0).startswith('<focus magic_chunks="1">')
    assert parse(match.group(0)) == ((1,), NO_FAILURE)
    # A body with a `>` beyond the eight-character tail hides the tag entirely.
    assert RE_FOCUS_OPEN.search('<focus magic_chunks="1">a longer body > here') is None


def test_id_regex_needs_whitespace_between_focus_and_the_keyword() -> None:
    assert id_regex(KW).search('<focusmagic_chunks="1">') is None
    assert id_regex(KW).search('<focus\tmagic_chunks="1">') is not None


# --------------------------------------------------------------------------------------
# The scan window
# --------------------------------------------------------------------------------------


def test_scan_tail_takes_the_last_500_characters_and_strips() -> None:
    assert SCAN_CHARS == 500
    text = "z" * 600 + "  </focus>  "
    window = scan_tail(text)
    assert len(window) <= SCAN_CHARS
    assert window.endswith("</focus>")
    assert RE_FOCUS_CLOSE.search(window) is not None


def test_a_tag_that_scrolled_out_of_the_window_is_gone() -> None:
    assert RE_FOCUS_OPEN.search(scan_tail('<focus magic_chunks="1">' + "x" * 600)) is None
