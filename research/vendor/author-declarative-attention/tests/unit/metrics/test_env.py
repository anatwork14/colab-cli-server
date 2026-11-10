"""Environment observation does not sample during timed steps or resolve GPU identities."""

import json

from dattn.metrics import env


def test_two_snapshots_and_one_dump(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    def output(command, **kwargs):
        calls.append(command)
        return "full device dump" if "-q" in command else "2, 1900, 4000, 60, 0x4\n"

    monkeypatch.setattr(env.subprocess, "check_output", output)
    env.dump_cell(tmp_path / "smi.txt")
    env.snapshot(tmp_path / "env.jsonl", "before")
    env.snapshot(tmp_path / "env.jsonl", "after")
    rows = [json.loads(s) for s in (tmp_path / "env.jsonl").read_text().splitlines()]
    assert [r["edge"] for r in rows] == ["before", "after"]
    assert rows[0]["gpus"][0] == dict(zip(env.FIELDS, ["2", "1900", "4000", "60", "0x4"], strict=True))
    assert len(calls) == 3 and all(c[-2:] == ["-i", "2"] for c in calls)


def test_disabled_cuda_never_launches_smi(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(env.subprocess, "check_output", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    assert env.snapshot(tmp_path / "env.jsonl", "before")["gpus"] == []
