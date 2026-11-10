"""Independent processes must agree before any duration is paired."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dattn.bench.driver import identity_hash
from dattn.bench.pairing import config_faults, pair_runs
from dattn.bench.protocol import Protocol
from dattn.contracts import StepRecord, StepRows
from dattn.metrics.record import write_table


def proto(**kw):
    return Protocol(
        **({"model_id": "m", "concurrency": 8, "repetitions": 2, "leading_trim": 0, "n_resamples": 10} | kw)
    )


def make_series(n, *, arm="da", vanilla=False):
    return [
        StepRecord(
            step_id=i,
            engine_step=i,
            t_sched_ns=1_000_000 + i * 10_000_000,
            num_reqs=8,
            num_reqs_padded=8,
            dattn_block_size=32,
            cudagraph_mode="FULL",
            rows=StepRows(seq_len=[128] * 8, att_blk=[] if vanilla else [64] * 8, att_tok=[128 if vanilla else 64] * 8),
        )
        for i in range(n)
    ]


def meta(arm: str) -> dict:
    return {
        "arm": arm,
        "dattn_builder_installed": arm == "da",
        "dattn_lp_installed": arm == "da",
        "resolved_config": {
            "engine_args": {
                "model": "m",
                "seed": 1,
                **(
                    {
                        "logits_processors": ["dattn"],
                        "attention_config": {"backend_per_kind": {"full_attention": "CUSTOM"}},
                    }
                    if arm == "da"
                    else {}
                ),
            },
            "attention_backend": "TRITON_ATTN",
            "block_sizes": [32],
            "cudagraph_mode": "FULL",
            "capture_sizes": [8],
            "max_num_batched_tokens": 16384,
            "num_gpu_blocks": 1000,
        },
        "backend_cls": "TritonAttentionBackend",
        "use_v2_model_runner": False,
        "async_scheduling": False,
        "enable_prefix_caching": False,
        "audit": {"checked": 30, "mismatches": 0, "groups_dropped": 0},
        "gpu_index": "0",
    }


def captures(root: Path, *, aa=False, reverse=False, repetitions=2, batches=1) -> tuple[Path, Path]:
    protocol = proto(repetitions=repetitions, batches_per_task=batches)
    for label, arm in (("left", "vanilla" if aa else "da"), ("right", "vanilla")):
        for batch in range(batches):
            for rep in range(repetitions):
                first = ((label == "left") == ((batch if repetitions == 1 else rep) % 2 == 0)) != reverse
                start = 100 * (batch * repetitions + rep) + (0 if first else 40)
                path = root / label
                if batches > 1:
                    path /= f"b{batch}"
                path /= f"r{rep}"
                path.mkdir(parents=True, exist_ok=True)
                m = meta(arm) | {"process_start_unix": start}
                records = make_series(100 if arm == "da" else 150, arm=arm, vanilla=arm == "vanilla")
                identity = {"task": "t", "indices": [2 * batch, 2 * batch + 1], "seed": batch + 1}
                doc = {
                    "arm": arm,
                    "task": "t",
                    "batch_index": batch,
                    "rep_index": rep,
                    "protocol": protocol.to_dict(),
                    "meta": m,
                    "windows": {"w": {"start": 0, "stop": len(records)}},
                    "t_pass_start": start + 10,
                    "t_pass_end": start + 20,
                    "identity": identity,
                    "identity_hash": identity_hash(identity),
                    "faults": [],
                }
                (path / "arm.json").write_text(json.dumps(doc))
                write_table(path / "steps_w.jsonl", records)
    return root / "left", root / "right"


def edit(path: Path, fn) -> None:
    doc = json.loads(path.read_text())
    fn(doc)
    path.write_text(json.dumps(doc))


@pytest.mark.parametrize("aa", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_same_pairing_handles_either_process_order_and_aa(tmp_path, monkeypatch, aa, reverse):
    left, right = captures(tmp_path, aa=aa, reverse=reverse)
    cells = []
    monkeypatch.setattr("dattn.bench.pairing.analyse_cell", lambda cell, **kw: cells.append(cell))
    pair_runs(left, right, out_dir=tmp_path / "paired")
    cell = cells[0]
    assert len(cell.sets["da"]) == len(cell.sets["vanilla"]) == 2
    rect = cell.rect()
    assert len(rect) == (149 if aa else 99) * 2
    assert rect.da_ms.max() < 40  # never a duration between processes
    assert cell.meta["arm_kinds"]["da"] == ("vanilla" if aa else "da")
    order = ["left_first", "right_first"] if aa else ["da_first", "vanilla_first"]
    assert cell.meta["pass_order"] == (order[::-1] if reverse else order)


@pytest.mark.parametrize(
    "key",
    ["attention_backend", "block_sizes", "cudagraph_mode", "capture_sizes", "max_num_batched_tokens", "num_gpu_blocks"],
)
def test_every_resolved_geometry_difference_is_refused(key):
    a, b = meta("da"), meta("vanilla")
    b["resolved_config"][key] = "changed"
    assert config_faults(a, b)
    del b["resolved_config"][key]
    assert any("missing resolved" in f for f in config_faults(a, b))


def test_only_da_binding_differences_are_allowed():
    a, b = meta("da"), meta("vanilla")
    assert config_faults(a, b) == []
    b["resolved_config"]["engine_args"]["seed"] = 9
    assert config_faults(a, b)


@pytest.mark.parametrize("arm", ["da", "vanilla"])
@pytest.mark.parametrize("key", ["dattn_builder_installed", "dattn_lp_installed"])
def test_wrong_or_absent_live_binding_is_refused(arm, key):
    a, b = meta(arm), meta("vanilla")
    a[key] = not a[key]
    assert config_faults(a, b)
    del a[key]
    assert config_faults(a, b)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda d: d["faults"].append("preemption"), "capture faults"),
        (lambda d: d.update(identity_hash="wrong"), "corrupt"),
        (lambda d: d["meta"].update(process_start_unix=10), "overlapped"),
        (lambda d: d["meta"]["audit"].update(checked=0), "audit"),
        (lambda d: d["meta"]["audit"].update(mismatches=1), "audit"),
        (lambda d: d["meta"].update(dattn_lp_installed=False), "dattn_lp"),
    ],
)
def test_bad_capture_is_refused_before_analysis(tmp_path, change, match):
    left, right = captures(tmp_path)
    target = right if match == "overlapped" else left
    edit(target / "r0" / "arm.json", change)
    with pytest.raises(ValueError, match=match):
        pair_runs(left, right, out_dir=tmp_path / "paired")


def test_missing_rep_and_repeated_process_order_are_refused(tmp_path):
    left, right = captures(tmp_path)
    (right / "r1" / "arm.json").unlink()
    with pytest.raises(ValueError, match="unmatched"):
        pair_runs(left, right, out_dir=tmp_path / "paired")
    captures(tmp_path)
    edit(left / "r1" / "arm.json", lambda d: d["meta"].update(process_start_unix=90))
    edit(left / "r1" / "arm.json", lambda d: d.update(t_pass_end=99))
    with pytest.raises(ValueError, match="alternate"):
        pair_runs(left, right, out_dir=tmp_path / "paired")


def test_identity_drift_is_refused_even_with_a_valid_hash(tmp_path):
    left, right = captures(tmp_path)

    def drift(d):
        d["identity"]["seed"] = 7
        d["identity_hash"] = identity_hash(d["identity"])

    edit(right / "r0" / "arm.json", drift)
    with pytest.raises(ValueError, match="batch identity mismatch"):
        pair_runs(left, right, out_dir=tmp_path / "paired")


@pytest.mark.parametrize("repetitions", [0, -1])
def test_protocol_requires_positive_repetitions(repetitions):
    with pytest.raises(ValueError, match="repetitions"):
        proto(repetitions=repetitions)


def test_pair_writes_a_real_analysis_summary(tmp_path):
    left, right = captures(tmp_path)
    summaries = pair_runs(left, right, out_dir=tmp_path / "paired")
    assert summaries[0]["n_kept"] == 198
    assert (tmp_path / "paired" / "m" / "t" / "b0" / "summary.json").is_file()


def test_five_batches_one_rep_records_order_and_imbalance(tmp_path):
    left, right = captures(tmp_path, repetitions=1, batches=5)
    summaries = pair_runs(left, right, out_dir=tmp_path / "paired")
    assert len(summaries) == 5
    for batch, summary in enumerate(summaries):
        assert summary["n_kept"] == 99
        assert summary["meta"]["batch_index"] == batch
        assert summary["meta"]["pass_order"] == ["da_first" if batch % 2 == 0 else "vanilla_first"]
        assert summary["meta"]["task_pass_order"] == {
            "da_first": 3,
            "vanilla_first": 2,
            "imbalance": 1,
            "planned_batches": 5,
        }
        saved = json.loads((tmp_path / "paired" / "m" / "t" / f"b{batch}" / "summary.json").read_text())
        assert saved["meta"] == summary["meta"]
    # An odd batch running DA first violates the cross-batch schedule.
    edit(left / "b1" / "r0" / "arm.json", lambda d: d["meta"].update(process_start_unix=70))
    edit(left / "b1" / "r0" / "arm.json", lambda d: d.update(t_pass_end=90))
    with pytest.raises(ValueError, match="expected DA first"):
        pair_runs(left, right, out_dir=tmp_path / "wrong")


@pytest.mark.parametrize("repetitions", [1, 3])
def test_only_aa_requires_even_multiple_repetitions(tmp_path, repetitions):
    left, right = captures(tmp_path / "ab", repetitions=repetitions)
    assert pair_runs(left, right, out_dir=tmp_path / "paired")[0]["n_kept"] == 99 * repetitions
    left, right = captures(tmp_path / "aa", aa=True, repetitions=repetitions)
    with pytest.raises(ValueError, match="A/A requires even"):
        pair_runs(left, right, out_dir=tmp_path / "refused")


@pytest.mark.parametrize("task_index", [0, 1])
def test_eight_reps_share_one_boot_and_task_batch_order(tmp_path, task_index):
    left, right = captures(tmp_path, repetitions=8)
    for label, root in enumerate((left, right)):
        first = (label == 0) == (task_index % 2 == 0)
        start = 0 if first else 1000
        for rep in range(8):

            def update(doc, rep=rep, start=start, label=label):
                doc["meta"].update(
                    task_index=task_index,
                    process_start_unix=start,
                    process_end_unix=start + 900,
                    boot_id={
                        "pid": label,
                        "started": start,
                        "config_hash": identity_hash(doc["meta"]["resolved_config"]),
                    },
                )
                doc.update(t_pass_start=start + 10 + rep * 100, t_pass_end=start + 20 + rep * 100)

            edit(root / f"r{rep}/arm.json", update)
    report = pair_runs(left, right, out_dir=tmp_path / "paired")[0]
    assert report["meta"]["pass_order"] == ["vanilla_first" if task_index else "da_first"] * 8
    assert report["per_rep_n_kept"] == {str(r): 99 for r in range(8)}
    assert len(report["vanilla_rep_pair_drift"]) == 28
    assert all(row["n_kept"] == 99 and row["relative_change"] == 0 for row in report["vanilla_rep_pair_drift"])
    edit(left / "r7/arm.json", lambda d: d["meta"]["boot_id"].update(pid=99))
    with pytest.raises(ValueError, match="boot identity"):
        pair_runs(left, right, out_dir=tmp_path / "bad")
