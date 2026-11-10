"""Samples orchestration and persistence using synthetic rep0 cells."""

import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeTokenizer

from dattn.campaign import driver, samples
from dattn.campaign.population import ARMS, MODELS, SLUGS, digest, identity
from dattn.eval.persistence import atomic_json


def tokenizer():
    tok = FakeTokenizer()
    tok.backend_tokenizer = SimpleNamespace(to_str=lambda: "fake tokenizer")
    return tok


@pytest.fixture
def sample_campaign(tmp_path, monkeypatch):
    tasks = ("long", "short")
    monkeypatch.setattr(samples, "HEADLINE_SUBSETS", tasks)
    monkeypatch.setattr(driver, "tokenizer", lambda model: tokenizer())
    rows = [{"subset": t, "row_index": 0, "question": t, "answer": "a", "rubric": "r", "context": "doc"} for t in tasks]
    monkeypatch.setattr(driver.compare, "reference", lambda *a: (rows, {}))
    args = SimpleNamespace(root=tmp_path / "campaign", dac_root=tmp_path / "reference")
    for model in MODELS:
        source = args.root / "population" / f"{model}.json"
        atomic_json(source, rows)
        for arm in ARMS:
            cell = args.root / "rep0" / SLUGS[model] / arm
            manifest = {
                "population": str(source),
                "selection_sha256": digest(rows),
                "row_keys": [list(identity(r)) for r in rows],
                **driver.provenance(model, arm, tokenizer()),
            }
            atomic_json(cell / "campaign.json", manifest)
            (cell / "responses.jsonl").write_text(
                "".join(
                    json.dumps({**r, "rendered_prompt_tokens": 100 if r["subset"] == "long" else 10}) + "\n"
                    for r in rows
                )
            )
    return args


def test_prepare_preserves_rep0_bytes_and_creates_only_samples(sample_campaign):
    args = sample_campaign
    before = {p.relative_to(args.root): p.read_bytes() for p in (args.root / "rep0").rglob("*") if p.is_file()}
    order, jobs = samples.prepare(args)
    assert order == ["short", "long"]
    assert len(jobs) == 12
    after = {p.relative_to(args.root): p.read_bytes() for p in (args.root / "rep0").rglob("*") if p.is_file()}
    assert after == before
    for job in jobs:
        keys = json.loads((Path(job["shard"]) / "keys.json").read_text())
        assert {k[2] for k in keys} == {1, 2, 3, 4}
    assert samples.prepare(args) == (order, jobs)


def test_task_barrier_covers_all_six_cells(monkeypatch, tmp_path):
    jobs = [{"task": t, "model": m, "arm": a} for t in ("long", "short") for m in MODELS for a in ARMS]
    seen = []
    monkeypatch.setattr(driver, "generate", lambda args, batch, stop: seen.append(batch))
    monkeypatch.setattr(driver, "pending", lambda job: set())
    samples.generate_tasks(SimpleNamespace(root=tmp_path), ["short", "long"], jobs, threading.Event())
    assert [[j["task"] for j in batch] for batch in seen] == [["short"] * 6, ["long"] * 6]
    assert all({(j["model"], j["arm"]) for j in batch} == {(m, a) for m in MODELS for a in ARMS} for batch in seen)


def test_incomplete_task_does_not_start_next(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(driver, "generate", lambda *a: seen.append(a[1]))
    monkeypatch.setattr(driver, "pending", lambda job: {("short", 0, 4)})
    with pytest.raises(RuntimeError, match="incomplete"):
        samples.generate_tasks(
            SimpleNamespace(root=tmp_path), ["short", "long"], [{"task": "short"}, {"task": "long"}], threading.Event()
        )
    assert len(seen) == 1


def test_refuse_rep0_or_symlink_destination(tmp_path):
    with pytest.raises(ValueError, match="rep0"):
        samples.safe_root(tmp_path / "rep0")
    (tmp_path / "rep0").mkdir()
    (tmp_path / "samples").symlink_to(tmp_path / "rep0", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        samples.safe_root(tmp_path)


def test_sample_recipe_changes_only_n():
    for model in MODELS:
        for arm in ARMS:
            got = samples.sample_config(model, arm).to_dict()
            assert got.pop("n") == 4
            expected = driver.config(model, arm).to_dict()
            expected.pop("n")
            assert got == expected


def test_rollout_indices_partial_resume_and_judge_every_sample(tmp_path, monkeypatch):
    from dattn.contracts import PromptLayout
    from dattn.eval import runner
    from dattn.eval.client import Completion
    from dattn.eval.datasets import EvalRow
    from dattn.eval.prompts import BuiltPrompt
    from dattn.eval.runner import RunPaths

    layout = PromptLayout(n_prompt_tokens=100, question_start=80, segments=((16, 60),))
    prompt = BuiltPrompt("prompt", tuple(range(100)), layout, "", 1)
    monkeypatch.setattr(runner, "build_prompt", lambda *a, **k: prompt)
    calls, judge_calls = [], []

    class Client:
        def __init__(self, server):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def complete_many(self, prompts, **kwargs):
            calls.append(kwargs)
            return [
                [
                    Completion(text=f"<answer>{i}</answer>", n_generated_tokens=i + 1, extra={"index": i})
                    for i in (2, 0, 3, 1)
                ]
                for p in prompts
            ]

        def chat_many(self, messages, **kwargs):
            judge_calls.extend(messages)
            return [[Completion(text='{"correct": true}')] for m in messages]

    monkeypatch.setattr(runner, "ChatClient", Client)
    rows = [EvalRow(subset="s", row_index=0, split="test", context="c", question="q", answer="a", rubric="r")]
    paths = RunPaths(tmp_path / "samples")
    options = {
        "tokenizer": FakeTokenizer(),
        "config": samples.sample_config("qwen", "vanilla"),
        "server": None,
        "paths": paths,
        "response_index_start": 1,
    }
    assert runner.run_rollouts(rows, **options) == 4
    records = samples.readonly(paths.responses)
    assert {r["response_index"] for r in records} == {1, 2, 3, 4}
    assert all(r["response"] == f"<answer>{r['response_index'] - 1}</answer>" for r in records)
    assert all(r["n_generated_tokens"] == r["response_index"] for r in records)
    assert calls[0]["bodies"][0]["n"] == 4
    assert "seed" not in calls[0]["bodies"][0]
    kept = [r for r in records if r["response_index"] != 4]
    paths.responses.write_text("".join(json.dumps(r) + "\n" for r in kept))
    before = paths.responses.read_bytes()
    assert runner.run_rollouts(rows, **options) == 1
    assert paths.responses.read_bytes().startswith(before)
    assert runner.run_rollouts(rows, **options) == 0
    assert len(calls) == 2
    assert runner.run_judge(paths=paths, server=None) == 4
    assert {r["response_index"] for r in samples.readonly(paths.judgments)} == {1, 2, 3, 4}
    assert runner.run_judge(paths=paths, server=None) == 0
    assert len(judge_calls) == 4


def test_multi_choice_counts_are_per_child():
    from dattn.eval.client import _parse

    results = _parse(
        {
            "usage": {"completion_tokens": 10},
            "choices": [{"index": i, "text": "x", "token_ids": [0] * (i + 1)} for i in range(4)],
        },
        0,
    )
    assert [c.n_generated_tokens for c in results] == [1, 2, 3, 4]


def test_samples_cli_cannot_select_rep0(monkeypatch, tmp_path):
    monkeypatch.setattr(driver, "output_root", lambda path: path)
    monkeypatch.setattr(samples, "run", lambda args: pytest.fail("must not run"))
    with pytest.raises(SystemExit):
        driver.main(["samples", "--repeat", "rep0", "--root", str(tmp_path), "--gpus", "0", "1", "2", "3"])
