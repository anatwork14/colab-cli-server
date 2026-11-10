"""Real task rows for the bench, drawn the way `dattn eval` draws them (WP7, item 7/12)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dattn.bench.tasks import (
    HEADLINE_SUBSETS,
    SYNTHETIC_TASK,
    TaskSample,
    load_task,
    load_tasks,
    resolve_subsets,
    samples_from_rows,
    synthetic_task,
)
from dattn.eval.datasets import EvalRow


def row(subset: str, i: int, **kw: Any) -> dict[str, Any]:
    d = {
        "subset": subset,
        "row_index": i,
        "split": "test",
        "context": f"ctx {subset} {i}",
        "question": f"q {i}?",
        "answer": "gold",
        "rubric": "r",
    }
    d.update(kw)
    return d


def test_the_headline_subsets_are_the_papers_fifteen() -> None:
    assert len(HEADLINE_SUBSETS) == 15
    assert resolve_subsets(None) == HEADLINE_SUBSETS
    assert resolve_subsets(["headline"]) == HEADLINE_SUBSETS


def test_a_bare_subset_name_resolves_the_way_dattn_eval_resolves_it() -> None:
    assert resolve_subsets(["qmsum"]) == ("longbench_synthetic_v3_1_qmsum",)
    with pytest.raises(KeyError, match="unknown subset"):
        resolve_subsets(["not_a_subset"])


def test_a_sample_carries_the_real_question_not_a_hardcoded_one() -> None:
    """The hardcoded `Summarise the archive in one sentence.` is why S1 never had a pair."""
    rows = [EvalRow.from_dict(row("s", 0))]
    samples = samples_from_rows(rows)
    assert samples == [TaskSample(subset="s", sample_id=rows[0].id, context="ctx s 0", question="q 0?")]


def test_the_bench_never_carries_the_reference_answer_next_to_a_timing_batch() -> None:
    sample = samples_from_rows([EvalRow.from_dict(row("s", 0))])[0]
    assert not hasattr(sample, "answer")
    assert "gold" not in json.dumps(sample.to_dict())


def test_a_row_with_no_question_is_dropped_rather_than_timed() -> None:
    rows = [EvalRow.from_dict(row("s", 0, question="")), EvalRow.from_dict(row("s", 1))]
    assert [s.sample_id for s in samples_from_rows(rows)] == [rows[1].id]


def test_a_dump_names_its_own_tasks(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(row(s, i)) for s in ("a", "b") for i in range(3)), encoding="utf-8")
    out = load_tasks(None, model_id="m", rows_file=path)
    assert list(out) == ["a", "b"]
    assert [len(v) for v in out.values()] == [3, 3]
    assert load_task("a", model_id="m", rows_file=path)[0].context == "ctx a 0"


def test_a_json_list_dump_is_accepted_too(tmp_path: Path) -> None:
    path = tmp_path / "rows.json"
    path.write_text(json.dumps([row("a", i) for i in range(2)]), encoding="utf-8")
    assert len(load_tasks(None, model_id="m", rows_file=path)["a"]) == 2


def test_a_list_of_bare_contexts_is_refused_because_it_has_no_question(tmp_path: Path) -> None:
    """That is exactly what the old `--prompts` took, and it is why gate S1 never ran."""
    path = tmp_path / "ctx.json"
    path.write_text(json.dumps(["a context", "another"]), encoding="utf-8")
    with pytest.raises(ValueError, match="no question and cannot"):
        load_tasks(None, model_id="m", rows_file=path)


def test_a_task_with_no_rows_is_refused_rather_than_silently_empty(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row("a", 0)), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot draw a batch for a task with no rows"):
        load_tasks(["qmsum"], model_id="m", rows_file=path)


def test_the_selection_is_delegated_to_the_eval_lane_and_never_reimplemented(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second copy of dac's filter order would be a silent divergence."""
    seen: dict[str, Any] = {}

    def fake_select_rows(**kw: Any) -> list[EvalRow]:
        seen.update(kw)
        return [EvalRow.from_dict(row(s, 0)) for s in kw["subsets"]]

    import dattn.eval.datasets as ds

    monkeypatch.setattr(ds, "select_rows", fake_select_rows)
    out = load_tasks(["qmsum"], model_id="google/gemma-4-31B-it", n=64, seed=42)
    assert seen["model_id"] == "google/gemma-4-31B-it"
    assert seen["n_per_subset"] == 64 and seen["seed"] == 42
    assert seen["subsets"] == ("longbench_synthetic_v3_1_qmsum",)
    assert list(out) == ["longbench_synthetic_v3_1_qmsum"]


def test_filler_is_labelled_so_it_can_never_be_read_as_a_task() -> None:
    samples = synthetic_task(3, approx_tokens=128, seed=0)
    assert len(samples) == 3
    assert all(s.subset == SYNTHETIC_TASK for s in samples)
    assert synthetic_task(3, approx_tokens=128, seed=0) == samples, "deterministic"
