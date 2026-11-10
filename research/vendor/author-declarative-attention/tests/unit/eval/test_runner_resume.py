"""Resume semantics for `run_rollouts`: provenance ordering, and transport-failure retry.

Two bugs, one function.

**The ordering.** `run_rollouts` wrote the new `meta.json` before it read the resume key.
`RunMeta.write` is a whole-file rewrite, so the only record of which `SelectionSpec`
produced the rows already in `responses.jsonl` was destroyed one statement before the
resume decision. The module docstring claimed the resume key was checked against that
spec. In that order no such check could have been written by anyone.

**The denominator.** A request that failed in transport was persisted as a row with an
`error`, which made its key done forever and scored it WRONG. dac skips the sample and
retries it on resume (`run_eval.py:579-583`). The cost is not a wrong numerator, it is a
wrong denominator: every accuracy number in the cell is diluted by a count of server
hiccups, and no rerun recovers it.

No server and no tokenizer. Each row renders to a prompt whose first id IS its row index,
so the fake client can report exactly which rows each invocation asked for.
"""

from __future__ import annotations

import dataclasses
import functools
import json
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeClient as RecordingClient
from fakes import FakeTokenizer as CharTokenizer

from dattn.contracts import PromptLayout, Rendered
from dattn.eval import runner as run_mod
from dattn.eval.datasets import EvalRow, SelectionSpec
from dattn.eval.runner import RunPaths, run_rollouts
from dattn.eval.sampling import RolloutConfig

LAYOUT = PromptLayout(n_prompt_tokens=100, question_start=80, segments=((16, 40), (40, 60)))
RESPONSE = '<focus magic_chunks="1">x</focus><answer>4</answer>'
SPEC = SelectionSpec(
    model_id="Qwen/Qwen3.6-27B",
    subsets=("s",),
    n_per_subset=4,
    seed=1234,
    min_context_tokens=0,
    max_context_tokens=8192,
)


def _renderer(tokenizer: Any, *, question: str, **kw: Any) -> Rendered:
    """The renderer `eval.prompts.build_prompt` wraps. Row index is smuggled in as id 0."""
    del tokenizer, kw
    return Rendered(
        text="PROMPT",
        ids=(int(question), *range(1, 100)),
        layout=LAYOUT,
        messages=(),
        tools=None,
        segments=(object(), object()),  # type: ignore[arg-type]
        format_error="",
    )


@pytest.fixture
def rows() -> list[EvalRow]:
    return [
        EvalRow(subset="s", row_index=i, split="test", context="c", question=str(i), answer="4", rubric="r")
        for i in range(4)
    ]


def _run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[EvalRow],
    *,
    errors: frozenset[int] = frozenset(),
    empty: frozenset[int] = frozenset(),
    spec: SelectionSpec | None = SPEC,
    n: int = 1,
) -> tuple[RunPaths, list[list[int]]]:
    """One invocation against `tmp_path/cell`. Returns the paths and the rows requested."""
    seen: list[list[int]] = []
    import importlib

    monkeypatch.setattr(importlib.import_module("dattn.prompt.render"), "render", _renderer)
    client = functools.partial(RecordingClient, seen=seen, errors=errors, empty=empty, response=RESPONSE)
    monkeypatch.setattr(run_mod, "ChatClient", client)
    paths = RunPaths(tmp_path / "cell")
    run_rollouts(
        rows,
        tokenizer=CharTokenizer(RESPONSE),
        config=RolloutConfig(model_id="Qwen/Qwen3.6-27B", arm="da", n=n),
        server=object(),  # type: ignore[arg-type]
        paths=paths,
        spec=spec,
    )
    return paths, seen


def _persisted(paths: RunPaths) -> list[dict[str, Any]]:
    return [json.loads(line) for line in paths.responses.read_text().splitlines()]


# --- item 2: transport failures are retried, answers are not ---------------------------


def test_a_second_invocation_retries_exactly_the_transport_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows
) -> None:
    """The headline proof. Rows 1 and 3 fail in transport, rows 0 and 2 answer."""
    paths, first = _run(tmp_path, monkeypatch, rows, errors=frozenset({1, 3}))
    assert first == [[0, 1, 2, 3]]
    assert [r["row_index"] for r in _persisted(paths) if not r["transport_error"]] == [0, 2]
    original_successes = [r for r in _persisted(paths) if not r["transport_error"]]

    _, second = _run(tmp_path, monkeypatch, rows)
    assert second == [[1, 3]], "a resume must ask for the failed rows and only those"
    assert sorted(r["row_index"] for r in _persisted(paths) if not r["transport_error"]) == [0, 1, 2, 3]
    assert all(r in _persisted(paths) for r in original_successes)
    assert len(_persisted(paths)) == 4


def test_a_clean_resume_asks_for_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows) -> None:
    _run(tmp_path, monkeypatch, rows)
    _, second = _run(tmp_path, monkeypatch, rows)
    assert second == [], "no failures means no work, and no duplicated rows"


def test_an_empty_model_answer_is_terminal_and_is_never_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows
) -> None:
    """The distinction the fix must not blur: said nothing is not the same as never asked."""
    paths, _ = _run(tmp_path, monkeypatch, rows, empty=frozenset({2}))
    (row,) = [r for r in _persisted(paths) if r["row_index"] == 2]
    assert row["response"] == "" and row["error"] == ""
    assert row["format_correct"] is False  # It scores WRONG, which is correct.

    _, second = _run(tmp_path, monkeypatch, rows)
    assert second == [], "an empty answer is an answer"


def test_legacy_failure_events_are_replaced_on_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows) -> None:
    """Legacy error-only events remain retryable within a template-audited cell."""
    paths, _ = _run(tmp_path, monkeypatch, rows)
    paths.responses.write_text(
        "\n".join(
            json.dumps({"subset": "s", "row_index": i, "response_index": 0, "response": "", "error": err})
            for i, err in ((0, ""), (1, "HTTPError: 503"), (2, ""))
        )
        + "\n"
    )
    _, seen = _run(tmp_path, monkeypatch, rows)
    assert seen == [[1, 3]]
    assert sorted(r["row_index"] for r in _persisted(paths) if not r["error"]) == [0, 1, 2, 3]
    assert len(_persisted(paths)) == 4


# --- item 1: provenance is read before it is overwritten -------------------------------


def test_the_persisted_selection_spec_survives_long_enough_to_be_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows
) -> None:
    """A changed spec over an existing cell is refused, not silently appended to."""
    paths, _ = _run(tmp_path, monkeypatch, rows)
    other = dataclasses.replace(SPEC, seed=999)
    with pytest.raises(ValueError, match="different SelectionSpec") as exc:
        _run(tmp_path, monkeypatch, rows, spec=other)
    assert "seed" in str(exc.value)
    assert "1234" in str(exc.value) and "999" in str(exc.value), "name both values, at 4am"
    assert json.loads(paths.meta.read_text())["selection"]["seed"] == 1234, "meta was not clobbered"


def test_an_unchanged_spec_resumes_normally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows) -> None:
    paths, _ = _run(tmp_path, monkeypatch, rows, errors=frozenset({0}))
    created = json.loads(paths.meta.read_text())["created"]
    _, second = _run(tmp_path, monkeypatch, rows)
    assert second == [[0]]
    assert json.loads(paths.meta.read_text())["created"] == created, "created is provenance, not a clock"


def test_a_spec_less_call_does_not_blank_the_recorded_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows
) -> None:
    """`meta.json` is rewritten whole every time, so `spec=None` used to erase it."""
    paths, _ = _run(tmp_path, monkeypatch, rows)
    _run(tmp_path, monkeypatch, rows, spec=None)
    assert json.loads(paths.meta.read_text())["selection"] == SPEC.to_dict()


def test_a_fresh_cell_accepts_any_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, rows) -> None:
    """Nothing to mix means nothing to refuse."""
    paths, _ = _run(tmp_path, monkeypatch, rows, spec=dataclasses.replace(SPEC, seed=7))
    assert json.loads(paths.meta.read_text())["selection"]["seed"] == 7


def test_resume_skips_each_completed_response_index(tmp_path, monkeypatch, rows):
    paths, _ = _run(tmp_path, monkeypatch, rows)
    original = RecordingClient.complete_many

    def two(self, prompts, **kwargs):
        return [choices * 2 for choices in original(self, prompts, **kwargs)]

    monkeypatch.setattr(RecordingClient, "complete_many", two)
    _, seen = _run(tmp_path, monkeypatch, rows, n=2)
    assert seen == [[0, 1, 2, 3]]
    keys = [(r["row_index"], r["response_index"]) for r in _persisted(paths)]
    assert len(keys) == len(set(keys)) == 8
    _, seen = _run(tmp_path, monkeypatch, rows, n=2)
    assert seen == []


@pytest.mark.parametrize("legacy", [False, True])
def test_judge_resume_retries_transport_only_and_matches_uninterrupted(tmp_path, monkeypatch, legacy):
    from dattn.eval.aggregate import aggregate, load_cell
    from dattn.eval.client import Completion
    from dattn.eval.runner import run_judge
    from dattn.eval.schema import iter_jsonl, write_jsonl

    from .conftest import response_row

    paths = RunPaths(tmp_path / "resumed")
    write_jsonl(paths.responses, [response_row("s", i) for i in range(3)])
    seen = []
    fail = True

    class JudgeClient:
        def __init__(self, server):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def chat_many(self, messages, **kwargs):
            seen.append(len(messages))
            if fail:
                return [
                    [Completion(text='{"correct": true}')],
                    [Completion(error="timeout", transport_error=True)],
                    [Completion(text="I cannot grade this.", error="refusal", transport_error=False)],
                ]
            return [[Completion(text='{"correct": false}')] for _ in messages]

    monkeypatch.setattr(run_mod, "ChatClient", JudgeClient)
    assert run_judge(paths=paths, server=object()) == 3
    if legacy:
        docs = list(iter_jsonl(paths.judgments))
        docs[1].pop("transport_error")
        write_jsonl(paths.judgments, docs)
    fail = False
    assert run_judge(paths=paths, server=object()) == 1
    original = paths.judgments.read_bytes()
    assert run_judge(paths=paths, server=object()) == 0
    assert seen == [3, 1]
    assert paths.judgments.read_bytes() == original
    docs = list(iter_jsonl(paths.judgments))
    assert len(docs) == len({(d["subset"], d["row_index"], d["response_index"]) for d in docs}) == 3
    resumed = aggregate(load_cell(paths.responses, paths.judgments))
    assert resumed.n == 3 and resumed.n_correct == 1 and resumed.n_unparsable == 1

    clean = RunPaths(tmp_path / "clean")
    write_jsonl(clean.responses, list(iter_jsonl(paths.responses)))

    def uninterrupted(self, messages, **kwargs):
        return [
            [Completion(text='{"correct": true}')],
            [Completion(text='{"correct": false}')],
            [Completion(text="I cannot grade this.", error="refusal", transport_error=False)],
        ]

    monkeypatch.setattr(JudgeClient, "chat_many", uninterrupted)
    assert run_judge(paths=clean, server=object()) == 3
    assert aggregate(load_cell(clean.responses, clean.judgments)).to_dict() == resumed.to_dict()


def test_explicit_nontransport_rollout_error_is_terminal(tmp_path, monkeypatch, rows):
    from dattn.eval.aggregate import aggregate, load_cell
    from dattn.eval.schema import write_jsonl

    paths, _ = _run(tmp_path, monkeypatch, rows)
    records = _persisted(paths)
    records[0]["error"] = "refusal"
    records[0]["transport_error"] = False
    write_jsonl(paths.responses, records)
    _, seen = _run(tmp_path, monkeypatch, rows)
    assert seen == []
    assert aggregate(load_cell(paths.responses)).n == 4
