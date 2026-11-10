"""The step recorder, driven end to end by the fake engine (WP6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeEvent, FakeTorch, decode_step

from dattn.metrics.record import read_jsonl
from dattn.metrics.recorder import AUDIT_EVERY, DEFAULT_RING, StepFacts, StepRecorder


def rec_for(fake_torch: type[FakeTorch], **kw: object) -> StepRecorder:
    kw.setdefault("window_id", "w0")
    kw.setdefault("max_num_seqs", 8)
    kw.setdefault("ring", 8)
    return StepRecorder(torch_mod=fake_torch, **kw)  # type: ignore[arg-type]


def one_step(r: StepRecorder, *, n: int = 4, att: int | None = None, seq: int = 1024) -> None:
    """Exactly the call order the worker makes: LP, compaction, forward, sample, close."""
    att = seq if att is None else att
    r.begin_step(t_sched_ns=1, facts=StepFacts.from_scheduler_output(decode_step(n), num_reqs=n))
    r.note_lp(90.0, [(f"r{i}", 0, seq, att, 5, 5) for i in range(n)])
    r.note_compaction(5, [att] * n, [att] * n, 12.0)
    r.forward_start()
    pass
    r.forward_end()
    r.sample_start()
    pass
    r.sample_end()
    r.end_step(t_ready_ns=30_000_001)


# -- the protocol -------------------------------------------------------------------------


def test_the_declared_constants_are_the_design_numbers() -> None:
    assert AUDIT_EVERY == 500
    assert DEFAULT_RING >= 512


# -- one step ------------------------------------------------------------------------------


def test_one_step_produces_exactly_one_record_with_every_block_joined(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch, dattn_block_size=32)
    one_step(r, n=3, seq=2048, att=1024)
    (out,) = r.drain()
    assert out.step_id == 0
    assert out.kind == "decode"
    assert out.num_reqs == 3
    assert out.num_gen_tokens == 3
    assert out.lp_host_us == 90.0
    assert out.compact_host_us == 12.0
    assert out.n_compacted_groups == 1
    assert out.dattn_block_size == 32
    assert out.rows.req == ["r0", "r1", "r2"]
    assert out.rows.seq_len == [2048] * 3
    assert out.rows.att_blk == [1024] * 3, "att_blk is the post-compaction length, from note_compaction"
    assert out.rows.att_tok == [1024] * 3
    assert out.gpu_forward_ms > 0 and out.gpu_sample_ms > 0


def test_the_step_id_is_recorder_local_and_monotonic(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch)
    for _ in range(5):
        one_step(r)
    assert [x.step_id for x in r.drain()] == [0, 1, 2, 3, 4]


# -- no sync, no allocation on the step path ------------------------------------------------


def test_nothing_on_the_step_path_synchronises(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch)
    for _ in range(8):
        one_step(r)
    assert fake_torch.cuda.syncs == 0, "a per-step synchronize would serialise CPU against GPU"
    assert fake_torch.D2H == 0
    r.drain()
    assert fake_torch.cuda.syncs == 1, "exactly one synchronize per drain, never per step"


def test_the_events_are_preallocated_in_a_ring_and_never_created_per_step(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch, ring=8)
    created = FakeEvent.created
    assert created == 3 * 2 * 8, "one start/stop pair per slot for compact, forward and sample"
    for _ in range(8):
        one_step(r)
    assert FakeEvent.created == created, "creating a cuda.Event per step is an allocation and is forbidden"


def test_the_row_ring_copy_is_non_blocking(fake_torch: type[FakeTorch]) -> None:
    """`FakeTensor.copy_` asserts `non_blocking=True`, so this test fails if it regresses."""
    r = rec_for(fake_torch)
    one_step(r)
    assert fake_torch.D2H == 0


# -- the ring, and the early drain ----------------------------------------------------------


def test_the_ring_drains_early_rather_than_wrapping(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch, ring=4)
    for _ in range(4):
        one_step(r)
    assert r.pending == 4
    one_step(r)  # the fifth step would wrap
    assert r.n_early_drains == 1
    assert r.pending == 1
    tail = r.drain()
    assert [x.step_id for x in tail] == [0, 1, 2, 3, 4]
    assert [x.drained_early for x in tail] == [True, True, True, True, False]


def test_an_early_drain_marks_the_records_it_returns(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch, ring=2)
    for _ in range(2):
        one_step(r)
    early = r.drain(early=True)
    assert all(x.drained_early for x in early)
    assert r.audit()["early_drains"] == 1


def test_no_record_is_ever_dropped_silently(fake_torch: type[FakeTorch]) -> None:
    """Three early drains happen inside `_open`, and every record still comes back."""
    r = rec_for(fake_torch, ring=4)
    for _ in range(13):
        one_step(r)
    assert r.n_early_drains == 3
    out = r.drain()
    assert [x.step_id for x in out] == list(range(13))
    assert sum(x.drained_early for x in out) == 12


# -- the audit ---------------------------------------------------------------------------------


def test_the_audit_compares_the_device_ring_against_the_host_mirror(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch, ring=8, audit_every=1)
    for _ in range(4):
        one_step(r, att=512, seq=1024)
    r.drain()
    a = r.audit()
    assert a["checked"] == 4
    assert a["mismatches"] == 0
    assert a["first_mismatch"] is None


def test_a_mirror_that_lies_about_the_device_is_caught(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch, ring=8, audit_every=1)
    r.begin_step(facts=StepFacts(num_reqs=2, num_gen_tokens=2))
    r.note_lp(1.0, [("r0", 0, 1024, 512, 1, 1), ("r1", 0, 1024, 512, 1, 1)])
    r.note_compaction(5, [512, 512], [512, 999], 1.0)  # the mirror disagrees with the device
    r.end_step()
    r.drain()
    a = r.audit()
    assert a["mismatches"] == 1
    assert a["first_mismatch"]["mirror"] == [512, 999]  # type: ignore[index]
    assert a["first_mismatch"]["device"] == [512, 512]  # type: ignore[index]


def test_the_audit_cadence_is_every_500_steps_by_default(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch, ring=1024)
    for _ in range(600):
        one_step(r, n=2)
    r.drain()
    # step 0, step 500, and the last slot of the drain
    assert r.audit()["checked"] == 3
    assert r.audit()["mismatches"] == 0


def test_the_row_schema_violation_of_a_slot_index_in_the_seq_len_field_is_reported(
    fake_torch: type[FakeTorch],
) -> None:
    """`att_tok > seq_len` is impossible: it means row 3 of `note_lp` is not `seq_len`."""
    r = rec_for(fake_torch, audit_every=1)
    r.begin_step(facts=StepFacts(num_reqs=2))
    r.note_lp(1.0, [("r0", 0, 0, 4096, 1, 1), ("r1", 0, 1, 4096, 1, 1)])
    r.end_step()
    a = r.audit()
    assert a["row_schema_violations"] == 1
    assert "note_lp row 3 must be seq_len" in a["first_row_violation"]["hint"]  # type: ignore[index]


# -- the scheduler-stat index shift ---------------------------------------------------------------


def test_scheduler_stats_are_index_shifted_not_timestamp_joined(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch)
    one_step(r)  # step 0, nothing observed yet
    r.note_scheduler_stats(0.80, 0)  # observed after step 0
    one_step(r)  # step 1 carries it
    r.note_scheduler_stats(0.81, 2)
    one_step(r)
    out = r.drain()
    assert [x.kv_usage for x in out] == pytest.approx([0.0, 0.80, 0.81])
    assert [x.num_preempted for x in out] == [0, 0, 2]


# -- async scheduling and cudagraph safety ---------------------------------------------------------


def test_under_async_scheduling_the_host_bracket_is_marked_invalid(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch, async_scheduling=True)
    one_step(r)
    (out,) = r.drain()
    assert out.host_bracket_valid is False
    assert out.fsm_lag == 1, "fsm_lag is what the profile asserts, not what the recorder guesses"
    assert out.t_sched_ns and out.t_ready_ns, "the host stamps are still stored, only marked"


def test_a_cudagraph_replayed_step_records_the_resolved_mode(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch)
    r.begin_step(facts=StepFacts(num_reqs=4, num_reqs_padded=8, cudagraph_replayed=True, cudagraph_mode="FULL"))
    r.forward_start()
    pass
    r.forward_end()
    r.end_step()
    (out,) = r.drain()
    assert out.cudagraph_replayed and out.cudagraph_mode == "FULL"
    assert out.num_reqs_padded == 8, "the bucket actually replayed, not the request count"


# -- host-only mode ------------------------------------------------------------------------------


def test_the_recorder_runs_with_no_torch_at_all() -> None:
    r = StepRecorder(window_id="w", max_num_seqs=4, ring=4)
    assert not r.device_backed
    one_step(r)
    (out,) = r.drain()
    assert out.gpu_forward_ms == 0.0
    assert out.rows.att_blk == [1024] * 4, "the host mirror is still the record's truth"
    assert r.audit()["checked"] == 0


# -- flush ---------------------------------------------------------------------------------------


def test_flush_writes_the_drained_records_off_the_step_path(tmp_path: Path, fake_torch: type[FakeTorch]) -> None:
    p = tmp_path / "steps_w0.jsonl"
    r = rec_for(fake_torch, out_path=p)
    for _ in range(3):
        one_step(r)
    assert r.flush() == 3
    assert [x.step_id for x in read_jsonl(p)] == [0, 1, 2]
    assert r.close() == 0


def test_flush_without_a_writer_keeps_the_ring_for_the_caller_to_drain(fake_torch: type[FakeTorch]) -> None:
    """The bench driver's window: arm, run, disarm (which flushes), then drain in process.

    With no writer there is nowhere to flush TO, so flushing must not empty the ring. It
    used to, and every bench cell came back with `n_records: 0` although the engine had
    generated (found on hardware at integration, 2026-09-17).
    """
    r = rec_for(fake_torch)  # no out_path: the in-process bench case
    for _ in range(3):
        one_step(r)
    assert r.flush() == 0, "nothing to serialise without a writer"
    assert [x.step_id for x in r.drain()] == [0, 1, 2], "flush must not have eaten the window"
    assert r.drain() == [], "and drain is still the thing that empties it"


def test_drain_returns_detached_copies_so_the_ring_can_be_reused(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch, ring=2)
    one_step(r, n=2, seq=100, att=50)
    (first,) = r.drain()
    one_step(r, n=2, seq=200, att=200)
    (second,) = r.drain()
    assert first.rows.seq_len == [100, 100], "the first record must not be rewritten by the second step"
    assert second.rows.seq_len == [200, 200]


def test_a_recorder_refuses_an_impossible_ring_or_batch() -> None:
    with pytest.raises(ValueError, match="max_num_seqs"):
        StepRecorder(window_id="w", max_num_seqs=0)
    with pytest.raises(ValueError, match="ring"):
        StepRecorder(window_id="w", max_num_seqs=1, ring=0)


# --------------------------------------------------------------------------------------
# The paper's bracket: the compaction and the logits-processor host cost, both arms
# --------------------------------------------------------------------------------------


def da_step(r: StepRecorder, *, n: int = 4, lp_us: float = 90.0, compact_us: float = 12.0) -> None:
    """A DA step, in the worker's call order: LP, compaction, forward, sample, close."""
    r.begin_step(t_sched_ns=1, facts=StepFacts.from_scheduler_output(decode_step(n), num_reqs=n))
    r.note_lp(lp_us, [(f"r{i}", 0, 1024, 1024, 5, 5) for i in range(n)])
    r.compact_start()
    r.note_compaction(5, [1024] * n, [1024] * n, compact_us)
    r.compact_end()
    r.forward_start()
    pass
    r.forward_end()
    r.sample_start()
    pass
    r.sample_end()
    r.end_step(t_ready_ns=30_000_001)


def vanilla_step(r: StepRecorder, *, n: int = 4, lp_us: float = 3.0) -> None:
    """A stock step. The logits processor still runs (`processor.py:129` is
    unconditional and handles the anonymous vanilla row), and nothing compacts."""
    r.begin_step(t_sched_ns=1, facts=StepFacts.from_scheduler_output(decode_step(n), num_reqs=n))
    r.note_lp(lp_us, [(f"r{i}", 0, 1024, 1024, 5, 5) for i in range(n)])
    r.forward_start()
    pass
    r.forward_end()
    r.sample_start()
    pass
    r.sample_end()
    r.end_step(t_ready_ns=30_000_001)


def test_the_compaction_has_its_own_event_pair(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch, ring=4)
    da_step(r)
    (out,) = r.drain()
    assert out.gpu_compact_ms > 0
    assert out.gpu_forward_ms > 0 and out.gpu_sample_ms > 0

    # And the three brackets are independent: a step that compacts and never forwards
    # reports the compaction alone, which it could not do if the pair were shared.
    r2 = rec_for(fake_torch, ring=4)
    r2.begin_step(t_sched_ns=1)
    r2.compact_start()
    r2.note_compaction(5, [1024], [1024], 12.0)
    r2.compact_end()
    r2.end_step(t_ready_ns=2)
    (only_compact,) = r2.drain()
    assert only_compact.gpu_compact_ms > 0
    assert only_compact.gpu_forward_ms == 0.0 and only_compact.gpu_sample_ms == 0.0


def test_the_compaction_bracket_is_not_folded_into_the_forward(fake_torch: type[FakeTorch]) -> None:
    """The host bracket already contains the compaction, so adding it to `gpu_forward_ms`
    would double count the same work (owner decision #11)."""
    r = rec_for(fake_torch, ring=4)
    with_compaction = da_step(r) or r.drain()[0]
    r2 = rec_for(fake_torch, ring=4)
    without = vanilla_step(r2) or r2.drain()[0]
    assert without.gpu_compact_ms == 0.0
    assert with_compaction.gpu_forward_ms == without.gpu_forward_ms


def test_both_host_costs_are_on_every_record_for_both_arms(fake_torch: type[FakeTorch]) -> None:
    """A vanilla step must carry a real zero, never a missing field.

    `main.tex:843` defines the step time as prepare_inputs plus forward plus sample for
    BOTH arms, so the derivation adds `compact_host_us` and `lp_host_us` on both. If the
    stock arm omitted them the two arms would be measured under two definitions and the
    saving would be whatever the difference in definitions happened to be.
    """
    r = rec_for(fake_torch, ring=8)
    for _ in range(3):
        da_step(r, lp_us=90.0, compact_us=12.0)
    for _ in range(3):
        vanilla_step(r, lp_us=3.0)
    recs = r.drain()
    assert len(recs) == 6
    for rec in recs:
        d = rec.to_dict()
        assert "compact_host_us" in d and "lp_host_us" in d and "gpu_compact_ms" in d
        assert isinstance(rec.compact_host_us, float) and isinstance(rec.lp_host_us, float)
    assert [rec.compact_host_us for rec in recs] == [12.0, 12.0, 12.0, 0.0, 0.0, 0.0]
    assert [rec.lp_host_us for rec in recs] == [90.0, 90.0, 90.0, 3.0, 3.0, 3.0]
    assert all(rec.n_compacted_groups == 0 for rec in recs[3:])


def test_each_host_cost_lands_on_the_step_it_was_measured_in(fake_torch: type[FakeTorch]) -> None:
    """No index shift. The LP runs in `refresh_metadata` and the compaction in
    `_prepare_inputs`, both of step k, both before step k's forward, so both belong to
    the record `_open()` opened for step k."""
    r = rec_for(fake_torch, ring=16)
    for i in range(5):
        da_step(r, lp_us=float(i + 1), compact_us=float(10 * (i + 1)))
    recs = r.drain()
    assert [rec.step_id for rec in recs] == [0, 1, 2, 3, 4]
    assert [rec.lp_host_us for rec in recs] == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert [rec.compact_host_us for rec in recs] == [10.0, 20.0, 30.0, 40.0, 50.0]


def test_a_reused_ring_slot_starts_both_host_costs_at_zero(fake_torch: type[FakeTorch]) -> None:
    """The ring is preallocated, so a stale `compact_host_us` from four steps ago would
    read as a vanilla step that compacted."""
    r = rec_for(fake_torch, ring=2)
    da_step(r, compact_us=99.0, lp_us=99.0)
    da_step(r, compact_us=99.0, lp_us=99.0)
    r.drain()
    vanilla_step(r, lp_us=1.0)
    (out,) = r.drain()
    assert out.compact_host_us == 0.0
    assert out.lp_host_us == 1.0
    assert out.gpu_compact_ms == 0.0


def test_several_compacted_groups_sum_their_host_cost_into_one_step(fake_torch: type[FakeTorch]) -> None:
    """A hybrid model compacts more than one KV-cache group per step and the step paid
    for all of them."""
    r = rec_for(fake_torch, ring=4)
    r.begin_step(t_sched_ns=1)
    r.compact_start()
    r.note_compaction(0, [1024] * 4, [1024] * 4, 7.0)
    r.note_compaction(1, [1024] * 4, [1024] * 4, 5.0)
    r.compact_end()
    r.end_step(t_ready_ns=2)
    (out,) = r.drain()
    assert out.compact_host_us == 12.0
    assert out.n_compacted_groups == 2


def test_the_compaction_pair_costs_no_synchronize_on_the_step_path(fake_torch: type[FakeTorch]) -> None:
    r = rec_for(fake_torch, ring=64)
    for _ in range(20):
        da_step(r)
    assert fake_torch.cuda.syncs == 0
    assert fake_torch.D2H == 0


def test_the_engine_brackets_the_compaction_body_through_the_published_slot() -> None:
    """A call-site guard, because the device numbers themselves need a GPU.

    `DattnState.compact` must open the pair before the kernel that does the rewrite and
    close it after the rewrite is complete, and it must resolve both by name off `_REC`
    rather than through `StepRecorderProtocol`, so the engine layer keeps importing
    nothing from `dattn.metrics` and a recorder that predates the pair still works.
    """
    import inspect

    from dattn.engine.state import DattnState

    src = inspect.getsource(DattnState.compact)
    assert '"compact_start"' in src and '"compact_end"' in src
    assert "from dattn.metrics" not in src
    start = src.index('"compact_start"')
    kernel = src.index("impl(")
    end = src.index('"compact_end"')
    note = src.index("rec.note_compaction(")
    assert start < kernel < end < note, "the pair does not bracket the rewrite it is timing"


# -- P7: every compacted group is mirrored and audited, not only group 0 ------------------


def test_every_compacted_group_is_mirrored_under_its_own_group_id(fake_torch: type[FakeTorch]) -> None:
    """Group 0 used to be the only one kept, on an assumption the audit exists to check.

    `note_compaction` accepted `group_id` and never used it, and returned early on the
    second group with the comment "every compacted group writes the identical seq_lens
    vector". That is precisely the claim the device-versus-mirror audit is supposed to
    prove, so assuming it made the audit prove nothing about any group but the first.
    """
    r = rec_for(fake_torch)
    r.begin_step(t_sched_ns=1, facts=StepFacts.from_scheduler_output(decode_step(2), num_reqs=2))
    r.note_lp(90.0, [(f"r{i}", 0, 1024, 512, 5, 5) for i in range(2)])
    r.note_compaction(0, [512, 512], [512, 512], 4.0)
    r.note_compaction(7, [512, 512], [512, 512], 4.0)  # a NON-contiguous group id
    r.end_step(t_ready_ns=30_000_001)
    r.drain()

    a = r.audit()
    assert a["groups_mirrored"] == 2, "both groups must have a device row of their own"
    assert a["groups_dropped"] == 0
    assert int(a["checked"]) >= 2, "each group is a check, not each step"
    assert a["mismatches"] == 0


def test_a_second_group_that_disagrees_with_the_device_is_caught(fake_torch: type[FakeTorch]) -> None:
    """The regression the old code could not see: group 0 agrees, group 1 does not."""
    r = rec_for(fake_torch)
    r.begin_step(t_sched_ns=1, facts=StepFacts.from_scheduler_output(decode_step(2), num_reqs=2))
    r.note_lp(90.0, [(f"r{i}", 0, 1024, 512, 5, 5) for i in range(2)])
    r.note_compaction(0, [512, 512], [512, 512], 4.0)
    r.note_compaction(1, [512, 512], [512, 999], 4.0)  # mirror disagrees with the device
    r.end_step(t_ready_ns=30_000_001)
    r.drain()

    a = r.audit()
    assert a["mismatches"] == 1
    first = a["first_mismatch"]
    assert isinstance(first, dict)
    assert first["group_id"] == 1, "the audit must name WHICH group disagreed"


def test_a_group_the_device_ring_cannot_hold_is_reported_not_dropped(fake_torch: type[FakeTorch]) -> None:
    """Silently not mirroring a group would be the vacuous audit all over again."""
    r = rec_for(fake_torch, max_compacted_groups=2)
    r.begin_step(t_sched_ns=1, facts=StepFacts.from_scheduler_output(decode_step(2), num_reqs=2))
    r.note_lp(90.0, [(f"r{i}", 0, 1024, 512, 5, 5) for i in range(2)])
    for gid in range(4):
        r.note_compaction(gid, [512, 512], [512, 512], 1.0)
    r.end_step(t_ready_ns=30_000_001)
    r.drain()

    a = r.audit()
    assert a["groups_mirrored"] == 2
    assert a["groups_dropped"] == 2, "the two groups with no device row are counted, not forgotten"


def test_att_blk_still_carries_the_first_group(fake_torch: type[FakeTorch]) -> None:
    """The record holds one attended-length vector per step; the audit holds all of them."""
    r = rec_for(fake_torch)
    r.begin_step(t_sched_ns=1, facts=StepFacts.from_scheduler_output(decode_step(2), num_reqs=2))
    r.note_lp(90.0, [(f"r{i}", 0, 1024, 512, 5, 5) for i in range(2)])
    r.note_compaction(0, [512, 512], [512, 512], 4.0)
    r.note_compaction(1, [256, 256], [256, 256], 4.0)
    r.end_step(t_ready_ns=30_000_001)
    (out,) = r.drain()
    assert out.rows.att_blk == [512, 512]
    assert out.n_compacted_groups == 2
