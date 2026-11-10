"""CPU incident collection and resume using the rep0 file schemas."""

import functools
import importlib
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from fakes import FakeClient, FakeTokenizer

from dattn.campaign import driver
from dattn.campaign.population import PILOT, SLUGS, digest, identity
from dattn.eval import runner
from dattn.eval.datasets import EvalRow
from dattn.eval.persistence import atomic_json, read_records
from dattn.eval.runner import RunPaths
from dattn.prompt.template_policy import resolve_template, template_metadata

from .test_runner_resume import RESPONSE, _renderer


def fake_campaign(tmp_path, monkeypatch):
    tok = FakeTokenizer(RESPONSE)
    rows = [
        EvalRow(subset=PILOT[0], row_index=i, split="test", context="c", question=str(i), answer="4", rubric="r")
        for i in range(402)
    ]
    population = [r.to_dict() for r in rows]
    monkeypatch.setattr(driver, "load_population", lambda *args: population)
    monkeypatch.setattr(driver, "tokenizer", lambda *args: tok)

    def provenance(model, arm, tok):
        cfg = driver.config(model, arm)
        return {
            "rollout": cfg.to_dict(),
            "rendering": template_metadata(
                resolve_template(tok, cfg.model_id, cfg.template_policy), cfg.template_policy
            ),
            "segmenter_code_sha256": "fake-segmenter",
            "tokenizer_sha256": "fake-tokenizer",
        }

    monkeypatch.setattr(driver, "provenance", provenance)
    monkeypatch.setattr(driver, "check_pidfds", lambda: None)
    monkeypatch.setattr(driver, "wait_all", lambda *args: None)
    monkeypatch.setattr(importlib.import_module("dattn.prompt.render"), "render", _renderer)
    seen = []
    monkeypatch.setattr(runner, "ChatClient", functools.partial(FakeClient, seen=seen, response=RESPONSE))
    args = SimpleNamespace(root=tmp_path, repeat="rep0", command="pass1", rows_root=tmp_path, dac_root=tmp_path)
    jobs = driver.prepare(args, ["da", "da_nm", "vanilla"])
    return tok, rows, jobs, seen


def command(root):
    return ["pass1", "--repeat", "rep0", "--root", str(root), "--gpus", "0", "1", "2", "3"]


def test_incident_records_all_workers_and_oom_after_submit_order_cancellation(tmp_path, monkeypatch):
    _, _, jobs, _ = fake_campaign(tmp_path, monkeypatch)
    barrier = threading.Barrier(4)
    root_failed = threading.Event()
    assigned = {}

    @contextmanager
    def server(args, gpu, port, model, log, stop):
        assigned[gpu] = log
        with log.open("a") as stream:
            stream.write("current attempt\n")
        barrier.wait(timeout=10)
        try:
            yield SimpleNamespace(process=SimpleNamespace(pid=1))
        finally:
            # Errors must be read after the server has finished its cleanup.
            with log.open("a") as stream:
                if gpu == "1":
                    stream.write("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 672.00 MiB\n")
                stream.write("shutdown chatter\n" * 100)

    def client(command, log, owner, stop, gpu):
        if gpu == "1":
            root_failed.set()
            raise RuntimeError("engine exit")
        assert root_failed.wait(10)
        raise RuntimeError("campaign incident or engine exit")

    for job in jobs:
        Path(job["shard"], "engine.log").write_text("ERROR stale attempt must not appear\n")
    monkeypatch.setattr(driver, "server", server)
    monkeypatch.setattr(driver, "client", client)
    monkeypatch.setattr(driver, "judge", lambda *args: pytest.fail("judged after generation failure"))
    assert driver.main(command(tmp_path)) == 1
    incident = json.loads((tmp_path / "rep0/incident.json").read_text())
    failures = incident["failures"]
    assert [f["gpu"] for f in failures] == ["0", "1", "2", "3"]
    for failure in failures:
        log = assigned[failure["gpu"]]
        assert failure["engine_log"] == str(log)
        assert failure["shard"] == str(log.parent)
        assert failure["cell"] == str(log.parents[2])
        assert "stale attempt" not in json.dumps(failure)
        assert failure["engine_log_tail"][-1] == "shutdown chatter"
        progress = json.loads(log.with_name("progress.json").read_text())
        assert progress["state"] == "incident" and progress["gpu"] == failure["gpu"]
    assert "OutOfMemoryError" in "\n".join(failures[1]["engine_errors"])


def test_pass1_resumes_incident_shards_keeps_64_rows_and_separates_judge(tmp_path, monkeypatch):
    tok, rows, jobs, seen = fake_campaign(tmp_path, monkeypatch)
    cell = tmp_path / "rep0" / SLUGS["qwen"] / "da"
    runner.run_rollouts(
        rows[:64], tokenizer=tok, config=driver.config("qwen", "da"), server=object(), paths=RunPaths(cell)
    )
    committed = (cell / "responses.jsonl").read_bytes()
    assert len(read_records(cell / "responses.jsonl")) == 64
    seen.clear()
    incident_cells = {(SLUGS["gemma"], arm) for arm in ("da", "da_nm", "vanilla")} | {(SLUGS["qwen"], "da")}
    for job in jobs:
        path = Path(job["cell"])
        if (path.parent.name, path.name) in incident_cells:
            driver.update_progress(path, Path(job["shard"]), "incident")
    atomic_json(tmp_path / "rep0/incident.json", {"command": "pass1", "error": "campaign incident or engine exit"})
    qwen_job = next(j for j in jobs if Path(j["cell"]) == cell)
    assert len(driver.pending(qwen_job)) == 338
    active = set()
    mutex = threading.Lock()
    resumed = set()

    @contextmanager
    def server(args, gpu, port, model, log, stop):
        with mutex:
            assert gpu not in active
            active.add(gpu)
        try:
            yield SimpleNamespace(process=SimpleNamespace(pid=1))
        finally:
            with mutex:
                active.remove(gpu)

    def client(cmd, log, owner, stop, gpu):
        path = Path(cmd[cmd.index("--campaign-cell") + 1])
        manifest = json.loads((path / "campaign.json").read_text())
        assert manifest["selection_sha256"] == digest([r.to_dict() for r in rows])
        before = len(seen)
        runner.run_rollouts(
            rows,
            tokenizer=tok,
            config=driver.config(manifest["model"], manifest["arm"]),
            server=object(),
            paths=RunPaths(path),
        )
        if path == cell:
            # Persisted bytes are a stronger guarantee than equal decoded records.
            assert (path / "responses.jsonl").read_bytes().startswith(committed)
        assert len(seen) > before
        resumed.add((path.parent.name, path.name))

    judged = []

    def judge(args, cells, stop):
        assert not active
        assert all(not driver.pending(job) for job in jobs)
        judged.append(len(cells))

    monkeypatch.setattr(driver, "server", server)
    monkeypatch.setattr(driver, "client", client)
    monkeypatch.setattr(driver, "judge", judge)
    monkeypatch.setattr(driver, "comparison", lambda *args, **kwargs: None)
    assert driver.main(command(tmp_path)) == 0
    assert incident_cells <= resumed
    assert judged == [6, 6]
    records = read_records(cell / "responses.jsonl")
    assert len(records) == len({identity(r) for r in records}) == 402
    # Across six cells there were 64 committed requests that must never be sent again.
    assert sum(map(len, seen)) == 6 * 402 - 64
    for job in jobs:
        assert json.loads(Path(job["shard"], "progress.json").read_text())["state"] == "complete"
    seen.clear()
    resumed.clear()
    assert driver.main(command(tmp_path)) == 0
    assert not seen and not resumed
    assert (cell / "responses.jsonl").read_bytes().startswith(committed)
