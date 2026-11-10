"""Golden files: byte equality against the dac formatter, and the tool-schema rows.

`render_golden.json` was produced by running dac's own `ChunkMultiTurnToolFormatter` and
`ChunkSingleTurnXmlFormatter` at `fb1ef99` (in dac's venv, transformers 5.5.4) and dattn's
`render` (in dattn's venv, transformers 5.17.0) over the same seeded synthetic contexts,
and recording both. `reference_text_equal` is the byte-equality verdict of that run, and
this module re-renders now and reproduces the recorded digests, so a change to the
segmenter, the formats, the templates or the tool dict shows up as a failed hash rather
than as a quietly different prompt.

`tool_schema.json` carries the probe rows of DESIGN.md 1.2, which must reproduce
`research/probe_logs/{gemma,qwen}_default.json` exactly, and one row per (model, tagged
format) with the rendered sha256 and the token delta against a `tools=None` render.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import synth_contexts
from conftest import load_tokenizer

from dattn.prompt import render, template_hashes
from dattn.prompt.formats.base import TOOL_NAME

HERE = Path(__file__).resolve().parent
RENDER_GOLDEN = json.loads((HERE / "render_golden.json").read_text(encoding="utf-8"))
TOOL_GOLDEN = json.loads((HERE / "tool_schema.json").read_text(encoding="utf-8"))

PROBE_MESSAGES = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user", "content": "And of Japan? Answer in one word."},
]
PROBE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_capital",
            "description": "Return the capital city of a country.",
            "parameters": {
                "type": "object",
                "properties": {"country": {"type": "string", "description": "Country name."}},
                "required": ["country"],
            },
        },
    }
]


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any]) -> str:
    return f"{row['model'].split('/')[-1]}-{row['case']}-{row['fmt']}"


ROWS = RENDER_GOLDEN["rows"]


# --------------------------------------------------------------------------------------
# The recorded comparison
# --------------------------------------------------------------------------------------


def test_the_golden_file_records_the_dac_commit_and_the_known_deltas() -> None:
    assert RENDER_GOLDEN["dac_sha"] == "fb1ef99"
    assert set(RENDER_GOLDEN["known_deltas"]) == {"first_span_lead", "d2_truncation"}
    assert RENDER_GOLDEN["chat_template_kwargs"] == {"enable_thinking": False, "add_generation_prompt": True}


def test_the_vanilla_note_records_the_empty_context_delta_not_a_missing_formatter() -> None:
    """review MINOR 46. dac's vanilla formatter exists, and the real delta is named.

    The note used to claim dac has no vanilla formatter. It does, at `serve.py:608-617`
    and `run_eval.py:300-309`. The behaviour is unchanged, only the record.
    """
    note = RENDER_GOLDEN["vanilla_note"]
    assert "dac DOES have a vanilla formatter" in note
    assert "serve.py:608-617" in note
    assert "run_eval.py:300-309" in note
    assert "<empty_context>" in note
    assert "228 vs 223" in note
    assert "byte-identically" in note


def test_the_provenance_table_records_the_vanilla_system_turn() -> None:
    """review MINOR 47. The behaviour matches current dac; the deviation is from the paper.

    Pinned the same way `JUDGE_TEMPLATE_SHA256` is: a documentation fact the repo must
    not lose. The owner ruled dattn keeps the system turn and records what it costs.
    """
    table = (Path(__file__).resolve().parents[3] / "docs" / "PROVENANCE.md").read_text()
    rows = [ln for ln in table.splitlines() if ln.startswith("| P11 |")]
    assert len(rows) == 1
    row = rows[0]
    assert "Y_format_ablation" in row
    assert "fb1ef99" in row  # current dac, which dattn matches
    assert "c995167" in row  # the commit that added the system turn
    assert "11" in row  # tokens, on 1611 of 1611 rows


def test_every_rendered_prompt_was_byte_identical_to_dacs() -> None:
    """24 rows, two models, both tagged formats, six contexts including the three that
    quote tool markup. dac has no vanilla formatter, so those rows carry no reference."""
    compared = [r for r in ROWS if "reference_text_sha256" in r]
    assert len(compared) == 24
    for row in compared:
        assert row["reference_text_equal"] is True, row_id(row)
        assert row["text_sha256"] == row["reference_text_sha256"], row_id(row)
        assert row["n_tokens"] == row["reference_n_tokens"], row_id(row)
        assert row["question_start"] == row["reference_question_start"], row_id(row)


def test_the_transformers_versions_were_different_and_the_bytes_still_matched() -> None:
    """The reference ran on 5.5.4 and dattn on 5.17.0, so the recorded equality is also
    evidence that neither chat template changed across the two pins."""
    assert RENDER_GOLDEN["reference_transformers"] != RENDER_GOLDEN["dattn_transformers"]
    assert RENDER_GOLDEN["dattn_transformers"].startswith("5.")


def test_the_only_span_differences_are_the_two_documented_ones() -> None:
    differing = [r for r in ROWS if r.get("reference_spans_equal") is False]
    for row in differing:
        gemma_header = row["model"].startswith("google/gemma") and row["fmt"] == "chunk_multi_turn_tool"
        collision = row["case"].startswith("collision") and row["fmt"] == "chunk_multi_turn_tool"
        assert gemma_header or collision, f"undocumented span difference at {row_id(row)}"


def test_the_gemma_first_span_lead_is_exactly_three_tokens() -> None:
    """`F D1`. It is the first span only, and every later bound agrees."""
    rows = [
        r
        for r in ROWS
        if r["model"].startswith("google/gemma")
        and r["fmt"] == "chunk_multi_turn_tool"
        and not r["case"].startswith("collision")
    ]
    assert rows
    for row in rows:
        ours, theirs = row["spans"], row["reference_spans"]
        assert theirs[0][0] - ours[0][0] == 3, row_id(row)
        assert [s[1] for s in ours] == [s[1] for s in theirs]
        assert [s[0] for s in ours[1:]] == [s[0] for s in theirs[1:]]


def test_dac_truncates_a_span_on_a_context_that_quotes_tool_markup() -> None:
    """`F D2`, reproduced: at least one recorded reference span is strictly shorter than
    ours on a collision context.

    "No error reported on either side" was an over-claim and is corrected here (the reviewer,
    2026-09-17): dac reports nothing on the QUOTED-markup path, which is the path these
    rows take, but on the path where a context forges a whole tool-call pair it returns
    `non_contiguous_segment_ids` and degrades to GLOBAL. dattn reports no error on either,
    because it gets both right.
    """
    shortfalls = []
    for row in ROWS:
        if not row["case"].startswith("collision") or row["fmt"] != "chunk_multi_turn_tool":
            continue
        for ours, theirs in zip(row["spans"], row["reference_spans"], strict=False):
            shortfalls.append((ours[1] - ours[0]) - (theirs[1] - theirs[0]))
        assert row["format_error"] == "", "dattn reports no error either, it simply gets it right"
    assert max(shortfalls) > 100, "the reference truncation should be hundreds of tokens"


@pytest.mark.parametrize("row", ROWS, ids=[row_id(r) for r in ROWS])
def test_rendering_now_reproduces_the_golden_digest(row: dict[str, Any]) -> None:
    tokenizer = load_tokenizer(row["model"])
    rendered = render(
        tokenizer,
        context=synth_contexts.context_for(row["case"]),
        question=synth_contexts.question_for(row["case"]),
        fmt=row["fmt"],
        tool_schema=(row["fmt"] == "chunk_multi_turn_tool"),
    )
    assert sha256(rendered.text) == row["text_sha256"]
    assert len(rendered.ids) == row["n_tokens"]
    assert rendered.layout.question_start == row["question_start"]
    assert [list(s) for s in rendered.layout.segments] == row["spans"]
    assert rendered.format_error == row["format_error"]
    assert sha256(rendered.layout.prompt_tail) == row["prompt_tail_sha256"]


# --------------------------------------------------------------------------------------
# The tool schema
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("row", TOOL_GOLDEN["probe"], ids=lambda r: r["model"].split("/")[-1])
def test_the_probe_rows_still_hold(row: dict[str, Any]) -> None:
    """Reproduces `research/probe_logs/{gemma,qwen}_default.json`: Gemma 41 -> 99 tokens,
    Qwen 38 -> 305, on the fixed three-message list."""
    tokenizer = load_tokenizer(row["model"])
    assert sha256(tokenizer.chat_template) == row["chat_template_sha256"]
    for key, tools in (("no_tools", None), ("with_tools", PROBE_TOOLS)):
        text = tokenizer.apply_chat_template(PROBE_MESSAGES, tools=tools, tokenize=False, add_generation_prompt=True)
        assert sha256(text) == row[key]["sha256"], key
        assert len(tokenizer(text, add_special_tokens=False)["input_ids"]) == row[key]["n_tokens"], key
    assert row["token_delta"] == row["with_tools"]["n_tokens"] - row["no_tools"]["n_tokens"]
    assert row["token_delta"] > 0


def test_the_recorded_probe_deltas_are_the_designs() -> None:
    by_model = {r["model"]: r for r in TOOL_GOLDEN["probe"]}
    assert by_model["google/gemma-4-31B-it"]["no_tools"]["n_tokens"] == 41
    assert by_model["google/gemma-4-31B-it"]["with_tools"]["n_tokens"] == 99
    assert by_model["Qwen/Qwen3.6-27B"]["no_tools"]["n_tokens"] == 38
    assert by_model["Qwen/Qwen3.6-27B"]["with_tools"]["n_tokens"] == 305


@pytest.mark.parametrize(
    "row", TOOL_GOLDEN["formats"], ids=[f"{r['model'].split('/')[-1]}-{r['fmt']}" for r in TOOL_GOLDEN["formats"]]
)
def test_the_tool_schema_rows_still_hold(row: dict[str, Any]) -> None:
    tokenizer = load_tokenizer(row["model"])
    case = row["case"]
    rendered = render(
        tokenizer,
        context=synth_contexts.context_for(case),
        question=synth_contexts.question_for(case),
        fmt=row["fmt"],
        tool_schema=True,
    )
    assert sha256(rendered.text) == row["rendered_sha256"]
    assert len(rendered.ids) == row["n_tokens"]
    assert (rendered.tools is not None) == row["declares_tool"]
    assert (TOOL_NAME in rendered.text) == row["tool_name_in_text"]
    if row["declares_tool"]:
        assert row["token_delta"] > 0, "the declaration must cost tokens on both families"
    else:
        assert row["token_delta"] == 0


def test_the_template_body_hashes_are_pinned() -> None:
    """Assertion A17's client-side half: the instruction bodies dattn ships are the ones
    the golden file was generated with."""
    for model, expected in TOOL_GOLDEN["template_hashes"].items():
        got = template_hashes(load_tokenizer(model))
        assert got == expected, model
