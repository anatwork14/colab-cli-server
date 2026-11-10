"""Small recorded fixtures, process-level persistence, and fail-closed campaign gates."""

from __future__ import annotations

import copy
import json
import multiprocessing
import os
import random
from pathlib import Path

import pytest

from dattn.campaign import compare, driver
from dattn.campaign.population import ARMS, HEADLINE_SUBSETS, MODELS, PILOT, identity, pilot_order, shards
from dattn.eval.persistence import append_records, lock, read_records

FIXTURE = Path(__file__).parents[2] / "fixtures/step3/recorded.json"


def fixture():
    return json.loads(FIXTURE.read_text())


def test_recorded_macro_matches_dac_and_exposes_parser():
    f = fixture()
    got = compare.summarize(f["responses"], f["candidate_judgments"], ("a", "b"), candidate=True)
    expect = f["expected_macro"]
    assert got["macro"]["dac_correct"] == pytest.approx(expect["strict_accuracy"])
    assert got["macro"]["correct"] == pytest.approx(expect["tolerant_accuracy"])
    assert got["macro"]["total_steps"] == expect["steps"]
    assert got["macro"]["attended_tokens"] == expect["attended"]
    assert got["a"]["parser_disagreements"] == [["a", 1, 0]]
    assert got["b"]["n"] == 3  # format failure stays in denominator
    # Macro of subset means, not the flat response mean (68 steps).
    assert got["macro"]["total_steps"] != 68


def test_sharding_and_append_order_produce_byte_identical_aggregate_numbers(tmp_path):
    f = fixture()
    expected = None
    for seed in range(12):
        rows = copy.deepcopy(f["responses"])
        random.Random(seed).shuffle(rows)
        path = tmp_path / str(seed) / "responses.jsonl"
        for start in range(0, len(rows), seed % 4 + 1):
            append_records(path, rows[start : start + seed % 4 + 1])
        got = compare.summarize(
            read_records(path), list(reversed(f["candidate_judgments"])), ("a", "b"), candidate=True
        )
        encoded = json.dumps(got, sort_keys=True).encode()
        if expected is None:
            expected = encoded
        assert encoded == expected


def test_pilot_order_and_arbitrary_shards_preserve_selected_set_and_within_task_order():
    rows = [
        {"subset": t, "row_index": i, "context": "x" * (100 + i), "raw_context_tokens": 100 + i}
        for t in HEADLINE_SUBSETS
        for i in (9, 3, 8, 2)
    ]
    ordered = pilot_order(rows)
    assert [r["subset"] for r in ordered[::4]][:4] == list(PILOT)
    for hours in (1.2, 1.5, 1.75, 2.5, 5):
        groups = shards(rows, hours)
        flattened = [r for _, group, _ in groups for r in group]
        assert len(flattened) == len(rows)
        assert {identity(r) for r in flattened} == {identity(r) for r in rows}
        for t in HEADLINE_SUBSETS:
            assert [r["row_index"] for r in flattened if r["subset"] == t] == [9, 3, 8, 2]


def _append_process(path, start):
    for i in range(start, start + 16):
        append_records(path, [{"subset": "s", "row_index": i, "response": "x" * 120000}])


def test_separate_process_writers_have_no_interleaving_or_duplicate_keys(tmp_path):
    path = tmp_path / "responses.jsonl"
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=_append_process, args=(path, start)) for start in (0, 8, 16, 24)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(15)
        assert process.exitcode == 0
    records = read_records(path)
    assert len(records) == len({identity(r) for r in records}) == 40
    assert all(r["response"] == "x" * 120000 for r in records)


def _die_during_append(path):
    with lock(path.with_suffix(".lock")), path.open("ab", buffering=0) as stream:
        stream.write(b'{"subset":"torn"')
        os.fsync(stream.fileno())
        os._exit(7)


def test_death_releases_lock_and_only_torn_tail_is_discarded(tmp_path):
    path = tmp_path / "responses.jsonl"
    first = {"subset": "s", "row_index": 0}
    append_records(path, [first])
    process = multiprocessing.get_context("spawn").Process(target=_die_during_append, args=(path,))
    process.start()
    process.join(10)
    assert process.exitcode == 7
    append_records(path, [{"subset": "s", "row_index": 1}])
    assert [r["row_index"] for r in read_records(path)] == [0, 1]
    assert read_records(path)[0] == first


def test_malformed_complete_line_and_duplicate_success_are_not_hidden(tmp_path):
    path = tmp_path / "responses.jsonl"
    path.write_text('{"bad":\n')
    with pytest.raises(ValueError):
        append_records(path, [])
    row = {"subset": "s", "row_index": 0}
    path.write_text((json.dumps(row) + "\n") * 2)
    with pytest.raises(ValueError, match="duplicate"):
        append_records(path, [])


def _rate_summary(rates, ns):
    out = {
        t: {"correct": p, "dac_correct": p, "n": n, "keys": [[t, i, 0] for i in range(n)], "parser_disagreements": []}
        for t, p, n in zip(("a", "b"), rates, ns, strict=True)
    }
    out["macro"] = {"correct": sum(rates) / 2, "dac_correct": sum(rates) / 2, "n": sum(ns)}
    return out


def test_pilot_uses_single_run_observed_se_and_records_all_thresholds():
    ref = _rate_summary((0.5, 0.5), (18, 128))
    ours = _rate_summary((0.8, 0.2), (18, 128))
    got = compare.compare_cell(ref, [ours], ("a", "b"), macro_pp=8, task_pp=15, task_se=3, per_task_stop=True)
    assert got["a"]["se_pp"] == pytest.approx(11.785113)
    assert got["a"]["threshold_pp"] == pytest.approx(35.355339)
    assert got["a"]["stop"] is False
    assert got["b"]["se_pp"] == pytest.approx(4.419417)
    assert got["b"]["threshold_pp"] == 15
    assert got["b"]["stop"] is True
    assert got["macro"]["stop"] is False
    assert got["b"]["delta_se"] == pytest.approx(-30 / 4.419417, rel=1e-6)
    high = _rate_summary((0.9, 0.9), (18, 128))
    assert compare.compare_cell(high, [ours], ("a", "b"))["a"]["se_pp"] < 11


def test_full_task_flags_do_not_fail_macro_and_sd_is_sample_sd():
    ref = _rate_summary((0.5, 0.5), (128, 128))
    ours = _rate_summary((0.6, 0.4), (128, 128))
    got = compare.compare_cell(ref, [ours], ("a", "b"))
    assert got["a"]["flag"] and got["b"]["flag"]
    assert not any(x["stop"] for x in got.values())
    assert got["a"]["run_sd_pp"] is None
    got = compare.compare_cell(ref, [ours, ref], ("a", "b"))
    assert got["a"]["dac_correct_pct"] == 55
    assert got["a"]["run_sd_pp"] == pytest.approx(7.0710678)
    worse = _rate_summary((0.6, 0.5), (128, 128))
    assert compare.compare_cell(ref, [worse], ("a", "b"))["macro"]["stop"]


@pytest.mark.parametrize("kind", ["missing", "transport", "duplicate", "foreign", "parser"])
def test_incomplete_or_invalid_judgments_refuse_comparison(kind):
    f = fixture()
    judgments = f["candidate_judgments"]
    if kind == "missing":
        judgments.pop()
    elif kind == "transport":
        judgments[0]["transport_error"] = True
    elif kind == "duplicate":
        judgments.append(judgments[0])
    elif kind == "foreign":
        judgments.append({"subset": "z", "row_index": 4})
    else:
        judgments[0].pop("dac_correct")
    with pytest.raises(ValueError):
        compare.summarize(f["responses"], judgments, ("a", "b"), candidate=True)


@pytest.mark.parametrize("value", [None, 0, 1, "false", "true"])
def test_candidate_format_flag_must_be_boolean(value):
    f = fixture()
    f["responses"][0]["format_correct"] = value
    with pytest.raises(ValueError, match="boolean format_correct"):
        compare.summarize(f["responses"], f["candidate_judgments"], ("a", "b"), candidate=True)
    f["responses"][0].pop("format_correct")
    with pytest.raises(ValueError, match="boolean format_correct"):
        compare.summarize(f["responses"], f["candidate_judgments"], ("a", "b"), candidate=True)


@pytest.mark.parametrize("field", ["correct", "dac_correct"])
@pytest.mark.parametrize("value", [0, 1, "true", "false", [], {}])
def test_candidate_parser_fields_refuse_invalid_values_and_absence(field, value):
    f = fixture()
    f["candidate_judgments"][0][field] = value
    with pytest.raises(ValueError, match=f"invalid {field}"):
        compare.summarize(f["responses"], f["candidate_judgments"], ("a", "b"), candidate=True)
    f["candidate_judgments"][0].pop(field)
    with pytest.raises(ValueError, match=f"invalid {field}"):
        compare.summarize(f["responses"], f["candidate_judgments"], ("a", "b"), candidate=True)


def test_terminal_null_is_wrong_but_format_failed_judgment_is_refused():
    f = fixture()
    f["candidate_judgments"][0].update(correct=None, dac_correct=None)
    got = compare.summarize(f["responses"], f["candidate_judgments"], ("a", "b"), candidate=True)
    assert got["a"]["dac_correct"] == 0
    assert got["a"]["correct"] == 0.5
    f["candidate_judgments"].append({"subset": "b", "row_index": 1, "correct": True, "dac_correct": True})
    with pytest.raises(ValueError, match="format-failed"):
        compare.summarize(f["responses"], f["candidate_judgments"], ("a", "b"), candidate=True)


def valid_verdict(root):
    item = {"stop": False, "delta_pp": 0, "se_pp": 1, "threshold_pp": 5, "n": 128}
    return {
        "schema": compare.SCHEMA,
        "stage": "full",
        "status": "pass",
        "complete": True,
        "root": str(root.resolve()),
        "repeats": ["rep0"],
        "tasks": list(HEADLINE_SUBSETS),
        "thresholds": {"macro_pp": 2, "task_pp": 5, "task_se": 0, "per_task_stop": False},
        "stops": [],
        "cells": {
            f"{m}/{a}": {**dict.fromkeys(HEADLINE_SUBSETS, item), "macro": {**item, "n": 1611, "threshold_pp": 2}}
            for m in MODELS
            for a in ARMS
        },
    }


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.update(status="fail"),
        lambda d: d.update(complete=False),
        lambda d: d.update(stage="pilot"),
        lambda d: d.update(repeats=["rep1"]),
        lambda d: d.update(cells={}),
        lambda d: d.update(thresholds={}),
        lambda d: d.update(stops=[{}]),
        lambda d: d.update(schema="unknown"),
    ],
)
def test_gate_b_rejects_noncanonical_or_malformed_pass(tmp_path, mutation):
    doc = valid_verdict(tmp_path)
    mutation(doc)
    with pytest.raises(ValueError):
        compare.validate_verdict(doc, repeat="rep0", root=tmp_path)


def test_gate_b_missing_malformed_and_explicit_override(tmp_path):
    path = tmp_path / "verdict.json"
    with pytest.raises(ValueError, match="REFUSED"):
        compare.require_pass(path, repeat="rep0", root=tmp_path)
    path.write_text("{")
    with pytest.raises(ValueError, match="REFUSED"):
        compare.require_pass(path, repeat="rep0", root=tmp_path)
    assert compare.require_pass(path, repeat="rep0", root=tmp_path, override=True)["override"]
    assert compare.validate_verdict(valid_verdict(tmp_path), repeat="rep0", root=tmp_path)["status"] == "pass"


def test_refuses_foreign_process_before_gpu_reuse(monkeypatch):
    monkeypatch.setattr(driver, "gpu_processes", lambda: ({"2": "GPU-a"}, {"GPU-a": {999}}))
    monkeypatch.setattr(driver.select, "select", lambda *args: ([], [], []))
    with pytest.raises(RuntimeError, match="foreign"):
        driver.assert_free("2")
    driver.assert_free("2", {999: 10})
    monkeypatch.setattr(driver.select, "select", lambda *args: ([10], [], []))
    with pytest.raises(RuntimeError, match="foreign"):
        driver.assert_free("2", {999: 10})
    with pytest.raises(ValueError):
        driver.assert_free("3")


@pytest.mark.parametrize(
    "live_parent,observed_parent,accepted", [(True, 100, True), (True, 999, False), (False, 100, False)]
)
def test_child_pidfd_claim_rechecks_ancestry_after_open(monkeypatch, live_parent, observed_parent, accepted):
    from types import SimpleNamespace

    closed = []
    reads = iter([100, observed_parent])
    monkeypatch.setattr(Path, "glob", lambda *args: [Path("/proc/200/stat")])
    monkeypatch.setattr(driver.Owned, "_parent", staticmethod(lambda path: next(reads)))
    monkeypatch.setattr(driver, "pidfd_open", lambda pid: pid + 1000)
    monkeypatch.setattr(driver.os, "close", closed.append)

    def select(fds, *args):
        # Parent can exit after the pre-open check, so reject it at the second check.
        return ([1100] if len(fds) == 2 and not live_parent else [], [], [])

    monkeypatch.setattr(driver.select, "select", select)
    owner = driver.Owned(SimpleNamespace(pid=100))
    assert (200 in owner.fds) is accepted
    assert closed == ([] if accepted else [1200])


def test_pidfd_cleanup_works_in_actual_python_environment():
    import signal
    import subprocess
    import sys

    driver.check_pidfds()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], env=driver.cpu_env())
    owner = driver.Owned(process)
    owner.close()
    assert process.returncode in (-signal.SIGTERM, -signal.SIGKILL)
    for fd in owner.fds.values():
        with pytest.raises(OSError):
            os.fstat(fd)


def test_unsupported_pidfds_refuse_before_gpu_or_engine_use(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "prepare", lambda *args: [{"cell": str(tmp_path)}])
    monkeypatch.setattr(driver, "gpu_processes", lambda: pytest.fail("GPU query before pidfd preflight"))
    monkeypatch.setattr(driver.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("spawned before preflight"))

    def unsupported():
        raise OSError("pidfds unavailable")

    monkeypatch.setattr(driver, "check_pidfds", unsupported)
    assert driver.main(["pass1", "--root", str(tmp_path), "--gpus", "0", "1", "2", "3"]) == 1
    assert "pidfds unavailable" in (tmp_path / "rep0/incident.json").read_text()


def test_campaign_children_discard_inherited_profile_and_recording_overrides(monkeypatch):
    for key in ("DATTN_BACKEND_PROFILES", "DATTN_BOOT_ASSERT", "DATTN_RECORD", "DATTN_RECORD_DIR"):
        monkeypatch.setenv(key, "unsafe inherited setting")
    monkeypatch.setenv("VLLM_PLUGINS", "foreign_plugin")
    env = driver.cpu_env()
    assert not any(k.startswith("DATTN_") for k in env)
    assert env["VLLM_PLUGINS"] == ""
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["HF_HUB_OFFLINE"] == env["HF_DATASETS_OFFLINE"] == "1"


def test_progress_failure_stops_generation_workers(tmp_path, monkeypatch):
    import threading
    from types import SimpleNamespace

    stop = threading.Event()
    job = {"cell": str(tmp_path), "shard": str(tmp_path), "name": "pilot", "minutes": 30}
    monkeypatch.setattr(driver, "pending", lambda job: {("s", 0, 0)})
    monkeypatch.setattr(driver, "server", lambda *args: pytest.fail("booted after progress failure"))

    def fail(*args, **kwargs):
        raise OSError("progress disk failure")

    monkeypatch.setattr(driver, "update_progress", fail)
    with pytest.raises(driver.CampaignIncident, match="progress disk failure") as caught:
        driver.generate(SimpleNamespace(gpus=["0"]), [job], stop)
    assert caught.value.failures[0]["progress_error"] == "progress disk failure"
    assert stop.is_set()


def test_campaign_root_cannot_enter_repo_even_through_symlink(tmp_path):
    repo = Path(driver.__file__).resolve().parents[3]
    with pytest.raises(ValueError, match="outside"):
        driver.output_root(repo / "runs/step3")
    (tmp_path / "link").symlink_to(repo, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        driver.output_root(tmp_path / "link/outputs")


def test_partial_judging_preserves_pilot_bytes_and_only_adds_present_remaining_rows(tmp_path, monkeypatch):
    from dattn.eval import runner
    from dattn.eval.client import Completion
    from dattn.eval.runner import RunPaths
    from dattn.eval.schema import ResponseRecord

    seen = []

    class Client:
        def __init__(self, server):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def chat_many(self, messages, **kwargs):
            seen.append(len(messages))
            return [[Completion(text='{"correct": true}')] for _ in messages]

    monkeypatch.setattr(runner, "ChatClient", Client)
    paths = RunPaths(tmp_path)

    def row(i):
        return ResponseRecord(subset="s", row_index=i, format_correct=True).to_dict()

    append_records(paths.responses, [row(0), row(1)])
    runner.run_judge(paths=paths, server=object())
    pilot = paths.judgments.read_bytes()
    append_records(paths.responses, [row(2)])
    runner.run_judge(paths=paths, server=object())
    assert seen == [2, 1]
    assert paths.judgments.read_bytes().startswith(pilot)
    assert {identity(r) for r in read_records(paths.judgments)} == {("s", i, 0) for i in range(3)}


@pytest.mark.parametrize("null_verdict", [False, True])
def test_full_report_is_byte_identical_across_shards_and_repeats_have_sd(tmp_path, monkeypatch, null_verdict):
    from dattn.campaign.population import SLUGS
    from dattn.eval.persistence import atomic_json
    from dattn.eval.schema import write_jsonl

    f = fixture()
    if null_verdict:
        for name in ("candidate_judgments", "reference_judgments"):
            f[name][0]["correct"] = None
            f[name][0]["dac_correct"] = None
    monkeypatch.setattr(compare, "HEADLINE_SUBSETS", ("a", "b"))
    monkeypatch.setattr(compare, "reference", lambda *args: (f["responses"], {"total": 5, "headline": 5, "dropped": 0}))
    for model in MODELS:
        for arm, refarm in ARMS.items():
            reference_path = tmp_path / "dac" / SLUGS[model] / refarm
            write_jsonl(reference_path / "judgments.jsonl", f["reference_judgments"])
            for rep in ("rep0", "rep1"):
                cell = tmp_path / rep / SLUGS[model] / arm
                write_jsonl(cell / "responses.jsonl", f["responses"])
                write_jsonl(cell / "judgments.jsonl", f["candidate_judgments"])
    kwargs = {"stage": "full", "macro_pp": 2, "task_pp": 5, "task_se": 0, "per_task_stop": False}
    before, table = compare.report(tmp_path, ["rep0"], tmp_path / "dac", ("a", "b"), **kwargs)
    assert before["status"] == "pass"
    assert before["n_unparsable"]["gemma/da"] == {
        "reference": int(null_verdict),
        "by_repeat": {"rep0": {"n_unparsable": int(null_verdict), "n_dac_unparsable": int(null_verdict)}},
    }
    assert "reference n_unparsable | n_unparsable | n_dac_unparsable" in table
    assert f"| gemma/da | rep0 | {int(null_verdict)} | {int(null_verdict)} | {int(null_verdict)} |" in table
    assert "32-token block rounding does not apply" in table
    assert "under 0.07 percent, ours up" in table
    pilot, pilot_text = compare.report(tmp_path, ["rep0"], tmp_path / "dac", ("a", "b"), **{**kwargs, "stage": "pilot"})
    assert pilot["thresholds"]["per_task_stop"] is False
    assert "Pilot task stops are disabled by explicit operator override." in pilot_text
    assert "Single run: SD is unavailable" in table
    assert "strict Δ SE" in table and "correct %" in table
    for path in tmp_path.glob("rep0/*/*/responses.jsonl"):
        records = read_records(path)
        path.unlink()
        for row in reversed(records):
            append_records(path, [row])
    after, reordered = compare.report(tmp_path, ["rep0"], tmp_path / "dac", ("a", "b"), **kwargs)
    atomic_json(tmp_path / "before.json", before)
    atomic_json(tmp_path / "after.json", after)
    assert (tmp_path / "before.json").read_bytes() == (tmp_path / "after.json").read_bytes()
    assert table == reordered
    repeated, text = compare.report(tmp_path, ["rep0", "rep1"], tmp_path / "dac", ("a", "b"), **kwargs)
    assert "sample run-to-run SD" in text
    assert repeated["cells"]["gemma/da"]["a"]["run_sd_pp"] == 0

    for model in MODELS:
        cell = tmp_path / "rep0" / SLUGS[model] / "da_eqseg"
        write_jsonl(cell / "responses.jsonl", f["responses"])
        write_jsonl(cell / "judgments.jsonl", f["candidate_judgments"])
    ablation, text = compare.report(tmp_path, ["rep0"], tmp_path / "dac", ("a", "b"), ablation=True, **kwargs)
    assert ablation["n_unparsable"]["gemma/da_eqseg"]["by_repeat"]["rep0"]["n_unparsable"] == int(null_verdict)
    assert "ablation rep0" in text
    cell = tmp_path / "rep0" / SLUGS["gemma"] / "da_eqseg"
    for field in ("question", "answer", "rubric"):
        changed = copy.deepcopy(f["responses"])
        changed[0][field] = "different selected row"
        write_jsonl(cell / "responses.jsonl", changed)
        with pytest.raises(ValueError, match="content mismatch"):
            compare.report(tmp_path, ["rep0"], tmp_path / "dac", ("a", "b"), ablation=True, **kwargs)


def test_pass2_command_refuses_before_population_or_gpu_use(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "prepare", lambda *args: pytest.fail("prepared before Gate B"))
    monkeypatch.setattr(driver, "gpu_processes", lambda: pytest.fail("touched GPU before Gate B"))
    assert driver.main(["pass2", "--root", str(tmp_path), "--gpus", "4", "5", "6", "7"]) == 1


def test_metadata_mismatch_refused_even_with_zero_completed_rows(tmp_path):
    driver.ensure_json(tmp_path / "campaign.json", {"segmenter": "semantic", "selection": "abc"})
    with pytest.raises(ValueError, match="metadata mismatch"):
        driver.ensure_json(tmp_path / "campaign.json", {"segmenter": "equal_length", "selection": "abc"})
    with pytest.raises(ValueError, match="metadata mismatch"):
        driver.ensure_json(tmp_path / "campaign.json", {"segmenter": "semantic", "selection": "changed"})


def test_wait_for_all_four_does_not_accept_three_free(monkeypatch):
    snapshots = iter(
        [
            ({str(i): str(i) for i in range(4)}, {str(i): ({42} if i == 3 else set()) for i in range(4)}),
            ({str(i): str(i) for i in range(4)}, {str(i): set() for i in range(4)}),
        ]
    )
    waits = []
    monkeypatch.setattr(driver, "gpu_processes", lambda: next(snapshots))
    monkeypatch.setattr(driver.time, "sleep", waits.append)
    driver.wait_all(["0", "1", "2", "3"])
    assert waits == [15]


def test_gate_b_recomputes_recorded_inputs_and_refuses_stale(tmp_path, monkeypatch):
    doc = valid_verdict(tmp_path)
    path = tmp_path / "full.verdict.json"
    path.write_text(json.dumps(doc))
    monkeypatch.setattr(compare, "report", lambda *args, **kwargs: ({**doc, "changed": True}, ""))
    with pytest.raises(ValueError, match="no longer matches"):
        compare.require_pass(path, repeat="rep0", root=tmp_path)
    monkeypatch.setattr(compare, "report", lambda *args, **kwargs: (doc, ""))
    assert compare.require_pass(path, repeat="rep0", root=tmp_path) == doc


def _overlapping_rollout_process(path, slot):
    from dattn.eval.datasets import EvalRow

    from .test_runner_resume import _run

    rows = [
        EvalRow(subset="s", row_index=i, split="test", context="c", question=str(i), answer="4", rubric="r")
        for i in range(4)
    ]
    with pytest.MonkeyPatch.context() as monkeypatch:
        _, seen = _run(path, monkeypatch, rows)
    (path / f"seen-{slot}.json").write_text(json.dumps(seen))


def test_overlapping_process_rollouts_generate_each_key_only_once(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    processes = [ctx.Process(target=_overlapping_rollout_process, args=(tmp_path, i)) for i in range(2)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(20)
        assert process.exitcode == 0
    seen = [i for path in tmp_path.glob("seen-*.json") for batch in json.loads(path.read_text()) for i in batch]
    assert sorted(seen) == list(range(4))
    assert len(read_records(tmp_path / "cell/responses.jsonl")) == 4


@pytest.mark.parametrize("returncode,override,raises", [(1, True, False), (1, False, True), (2, True, True)])
def test_gate_override_allows_reported_macro_failure_but_not_incomplete_analysis(
    tmp_path, monkeypatch, returncode, override, raises
):
    from types import SimpleNamespace

    args = SimpleNamespace(
        root=tmp_path, repeat="rep0", dac_root=tmp_path, pilot_task_stop=True, override_gate_b=override
    )
    monkeypatch.setattr(compare, "main", lambda command: returncode)
    if raises:
        with pytest.raises(RuntimeError):
            driver.comparison(args, "full", ablation=True)
    else:
        driver.comparison(args, "full", ablation=True)


@pytest.mark.parametrize("override", [False, True])
def test_pilot_task_stop_override_is_recorded_before_preparation(tmp_path, monkeypatch, override):
    artifact = tmp_path / "rep0/pilot_task_stop_override.json"

    def prepare(*args):
        assert artifact.exists() is override
        if override:
            assert json.loads(artifact.read_text()) == {"explicit_override": True, "pilot_task_stop": False}
        return []

    monkeypatch.setattr(driver, "prepare", prepare)
    monkeypatch.setattr(driver, "gpu_processes", lambda: pytest.fail("prepare must not touch GPUs"))
    args = ["prepare", "--root", str(tmp_path)]
    if override:
        args += ["--no-pilot-task-stop"]
    assert driver.main(args) == 0


def _full_population_rollout_with_low_fd_limit(path):
    import resource

    from dattn.eval.datasets import EvalRow

    from .test_runner_resume import _run

    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, hard))
    rows = [
        EvalRow(subset="s", row_index=i, split="test", context="c", question=str(i), answer="4", rubric="r")
        for i in range(1611)
    ]
    with pytest.MonkeyPatch.context() as monkeypatch:
        paths, seen = _run(path, monkeypatch, rows)
        assert [i for batch in seen for i in batch] == list(range(1611))
        assert len(read_records(paths.responses)) == 1611
        _, resumed = _run(path, monkeypatch, rows)
        assert resumed == []


def test_full_population_rollouts_fit_a_low_fd_limit(tmp_path):
    process = multiprocessing.get_context("spawn").Process(
        target=_full_population_rollout_with_low_fd_limit, args=(tmp_path,)
    )
    process.start()
    process.join(60)
    assert process.exitcode == 0
