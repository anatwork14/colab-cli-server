"""`StepRecorder`: CUDA events, ring drain, single writer.

One record per engine iteration, written by ONE writer inside the worker process,
carrying timing, scheduler facts and per-row state together. No second stream and no
join (DESIGN.md section 5.1).

What the hot path costs when the recorder is armed
--------------------------------------------------
* two `cuda.Event.record()` calls around `_model_forward`, two around `_sample` and two
  around the compaction, from a preallocated ring. Creating an event per step is an
  allocation and is forbidden (DESIGN.md section 5.3),
* one D2D `copy_` of the post-compaction `seq_lens` into a preallocated device ring, per
  compacted KV-cache group, `non_blocking=True`, no sync,
* an O(C) host list build for the per-row block (`note_lp`), measured at 0.4 to 1.0 us
  per request,
* nothing else. `elapsed_time()` and `synchronize()` are called once per drain, never
  per step, so the recorder adds no host-device sync to the step path. That is what
  makes it safe under cudagraph replay and under async scheduling: the events record
  into the stream around the graph launch, and the per-row ring is keyed by the
  recorder-local `step_id` inside the worker rather than by any engine-side bracket.

When the recorder is NOT armed it does not exist on the hot path at all: `install()` is
the only thing that rebinds anything, and `uninstall()` restores the original bound
method by identity. See `dattn.metrics.hooks`.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

from dattn.contracts import StepRecord, StepRows, Tensor
from dattn.metrics.record import JsonlWriter, classify_step

#: Sized so a measurement window never wraps. If it would, the recorder drains early at
#: a step boundary and records `drained_early` rather than dropping records silently.
DEFAULT_RING: Final[int] = 4096

#: The per-row audit compares the device ring against the host mirror this often.
AUDIT_EVERY: Final[int] = 500

#: Device rows reserved per ring slot for the per-group `seq_lens` mirror. Gemma 4 and
#: Qwen 3.6 both compact well under this; a window that exceeds it reports the excess as
#: `audit()["groups_dropped"]` rather than mirroring a group and pretending it checked it.
MAX_COMPACTED_GROUPS: Final[int] = 16

__all__ = ["AUDIT_EVERY", "DEFAULT_RING", "StepFacts", "StepRecorder"]


@dataclass(slots=True)
class StepFacts:
    """The scheduler facts that join for free, from the `SchedulerOutput` the worker has.

    Never from a second stream and never from a timestamp (DESIGN.md section 5.3).
    """

    engine_step: int = 0
    kind: str = "decode"
    num_reqs: int = 0
    num_reqs_padded: int = 0
    #: Whether THIS step replayed a captured graph, and the mode it actually ran under:
    #: `FULL`, `PIECEWISE` or `NONE`. Per step, not the configured value. A step that
    #: replayed a FULL graph and a step that fell back to eager cost different amounts and
    #: honour the compacted `seq_lens` differently, so the mode has to travel with the
    #: record (research/NOTES.md section 2.0).
    cudagraph_replayed: bool = False
    cudagraph_mode: str = ""
    num_new_reqs: int = 0
    num_ctx_tokens: int = 0
    num_gen_tokens: int = 0
    n_full_attn_groups: int = 0

    @classmethod
    def from_scheduler_output(cls, so: Any, *, num_reqs: int, **kw: Any) -> StepFacts:
        """Build from a `SchedulerOutput`, using vLLM's own phase predicate."""
        n_new = len(so.scheduled_new_reqs)
        cached = so.scheduled_cached_reqs
        ctx_ids = [rid for rid in cached.req_ids if cached.is_context_phase(rid)]
        num_sched = getattr(so, "num_scheduled_tokens", None) or {}
        n_ctx_tokens = sum(int(num_sched.get(rid, 0)) for rid in ctx_ids)
        for new in so.scheduled_new_reqs:
            rid = getattr(new, "req_id", None)
            n_ctx_tokens += int(num_sched.get(rid, 0))
        return cls(
            kind=classify_step(so, num_reqs),
            num_reqs=int(num_reqs),
            num_new_reqs=n_new,
            num_ctx_tokens=n_ctx_tokens,
            num_gen_tokens=int(so.total_num_scheduled_tokens) - n_ctx_tokens,
            **kw,
        )


class StepRecorder:
    """Records worker clocks, rows and device/mirror audits.

    `torch_mod` is injected rather than imported so the unit suite can drive every path
    (event ring, device ring, early drain, audit mismatch) with a fake engine on a CPU
    box. Pass `torch_mod=None` for a host-only recorder, `torch_mod=torch` inside the
    worker. `install()` does the latter.
    """

    def __init__(
        self,
        *,
        window_id: str,
        max_num_seqs: int,
        ring: int = DEFAULT_RING,
        torch_mod: Any = None,
        device: Any = None,
        async_scheduling: bool = False,
        fsm_lag: int = 1,
        runner_generation: str = "V1",
        dattn_block_size: int = 0,
        audit_every: int = AUDIT_EVERY,
        max_compacted_groups: int = MAX_COMPACTED_GROUPS,
        out_path: Path | str | None = None,
    ) -> None:
        if max_num_seqs < 1:
            raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
        if ring < 1:
            raise ValueError(f"ring must be positive, got {ring}")
        self.window_id = str(window_id)
        self.max_num_seqs = int(max_num_seqs)
        self.ring = int(ring)
        self.audit_every = int(audit_every)
        self.max_compacted_groups = max(1, int(max_compacted_groups))
        self.fsm_lag = int(fsm_lag)
        self.runner_generation = str(runner_generation)
        self.dattn_block_size = int(dattn_block_size)
        self.host_bracket_valid = not bool(async_scheduling)

        self._torch = torch_mod
        self._device = device
        self.step_id = 0
        self.n_early_drains = 0
        self.audit_checked = 0
        self.audit_mismatches = 0
        self.first_mismatch: dict[str, object] | None = None
        self.row_schema_violations = 0
        self.first_row_violation: dict[str, object] | None = None
        #: Compacted groups the device ring had no room for. Never silently zero: the
        #: audit reports it and pairing refuses it, because a group that was not mirrored
        #: is a group whose attended counts nothing checked.
        self.audit_groups_dropped = 0
        #: `group_id -> device index`, assigned on first sight and stable for the window,
        #: so a non-contiguous or out-of-order group id still lands in its own row.
        self._group_index: dict[int, int] = {}

        # Index-shifted engine-core facts: written into step k from the value observed
        # after step k-1 (DESIGN.md section 5.3). A deterministic shift, not a join.
        self._kv_usage = 0.0
        self._num_preempted = 0

        # Preallocated record ring. Nothing on the step path allocates a record.
        self._recs: list[StepRecord] = [StepRecord(rows=StepRows()) for _ in range(self.ring)]
        self._slots: list[int] = []  # pending slots, in order
        self._cur: int = -1  # the open slot, -1 when no step is open
        #: Per slot, `group_id -> the host mirror for THAT group`. Every compacted group
        #: is mirrored and audited, not only the first.
        self._mirror: list[dict[int, list[int]]] = [{} for _ in range(self.ring)]
        self._have_fwd = [False] * self.ring
        self._have_smp = [False] * self.ring
        self._have_cmp = [False] * self.ring

        self._ev_fwd: list[tuple[Any, Any]] = []
        self._ev_smp: list[tuple[Any, Any]] = []
        self._ev_cmp: list[tuple[Any, Any]] = []
        self._ring_blk: Any = None
        if torch_mod is not None:
            mk = lambda: torch_mod.cuda.Event(enable_timing=True)  # noqa: E731
            self._ev_fwd = [(mk(), mk()) for _ in range(self.ring)]
            self._ev_smp = [(mk(), mk()) for _ in range(self.ring)]
            self._ev_cmp = [(mk(), mk()) for _ in range(self.ring)]
            self._ring_blk = torch_mod.zeros(
                (self.ring, self.max_compacted_groups, self.max_num_seqs), dtype=torch_mod.int32, device=device
            )

        #: Records an early drain produced that no caller has taken yet. Never dropped.
        self._overflow: list[StepRecord] = []
        self._writer = JsonlWriter(out_path) if out_path is not None else None
        self._written = 0
        #: Set by `hooks.install`: whether `api.set_step_recorder` accepted publication.
        self.publish_ok = False

    # -- properties --------------------------------------------------------------------

    @property
    def device_backed(self) -> bool:
        return self._torch is not None

    @property
    def pending(self) -> int:
        return len(self._slots)

    # -- the step lifecycle ------------------------------------------------------------

    def _open(self) -> int:
        """Open a slot for the current step. Drains early rather than wrapping."""
        if self._cur >= 0:
            return self._cur
        if len(self._slots) >= self.ring:
            # The window would wrap. Drain at this step boundary, keep every record, and
            # mark it, rather than overwriting a slot nobody has read (section 5.3).
            early = self.drain(early=True)
            if self._writer is None:
                self._overflow.extend(early)
            else:
                # A SERVED window holds a writer and can run for 10^5 steps, so parking
                # the early drain in `_overflow` grows without bound in the worker and
                # still leaves the file empty until end of run. Handing it to the writer
                # here adds NO device work: the `synchronize()` and the audit copy already
                # happened inside the `drain()` above, which the wrap forced whether or
                # not anybody reads the result. This is not the every-512-steps flush that
                # `d000e38` removed, and the difference is the whole point: that one
                # CAUSED a drain the ring did not need, once per 512 steps, inside a timed
                # step. This one causes nothing and rides a drain that was already due.
                # Every record it writes carries `drained_early`, so analysis can exclude
                # the one step in `ring` that paid for the wrap.
                self._writer.extend(early)
                self._writer.flush()
                self._written += len(early)
        slot = (self.step_id) % self.ring
        rec = self._recs[slot]
        rec.step_id = self.step_id
        rec.engine_step = 0
        rec.t_sched_ns = 0
        rec.t_ready_ns = 0
        rec.gpu_forward_ms = 0.0
        rec.gpu_sample_ms = 0.0
        rec.gpu_compact_ms = 0.0
        # Zeroed, never left stale and never left absent. A vanilla step runs no
        # compaction, so it must record a real 0.0 here: the analysis adds these two into
        # the paper's bracket for BOTH arms (main.tex:843), and a missing field would
        # silently drop the vanilla arm out of the same definition.
        rec.lp_host_us = 0.0
        rec.compact_host_us = 0.0
        rec.kind = "decode"
        rec.num_reqs = 0
        rec.num_reqs_padded = 0
        rec.cudagraph_replayed = False
        rec.cudagraph_mode = ""
        rec.runner_generation = self.runner_generation
        rec.dattn_block_size = self.dattn_block_size
        rec.num_new_reqs = 0
        rec.num_ctx_tokens = 0
        rec.num_gen_tokens = 0
        rec.num_preempted = self._num_preempted
        rec.kv_usage = self._kv_usage
        rec.n_full_attn_groups = 0
        rec.n_compacted_groups = 0
        rec.fsm_lag = self.fsm_lag
        rec.host_bracket_valid = self.host_bracket_valid
        rec.drained_early = False
        rows = rec.rows
        rows.req.clear()
        rows.mode.clear()
        rows.seq_len.clear()
        rows.att_blk.clear()
        rows.att_tok.clear()
        rows.out_pos.clear()
        rows.since_adm.clear()
        self._mirror[slot].clear()
        self._have_fwd[slot] = False
        self._have_smp[slot] = False
        self._have_cmp[slot] = False
        self._cur = slot
        self._slots.append(slot)
        return slot

    def begin_step(self, *, t_sched_ns: int | None = None, facts: StepFacts | None = None) -> None:
        """Open the step and stamp host clock 1 (`perf_counter_ns` in `EngineCore.step`)."""
        slot = self._open()
        rec = self._recs[slot]
        rec.t_sched_ns = int(t_sched_ns) if t_sched_ns is not None else time.perf_counter_ns()
        if facts is not None:
            self.note_facts(facts)

    def note_facts(self, facts: StepFacts) -> None:
        """The scheduler facts, joined into the same record with no cross-process hop."""
        rec = self._recs[self._open()]
        rec.engine_step = facts.engine_step
        rec.kind = facts.kind
        rec.num_reqs = facts.num_reqs
        rec.num_reqs_padded = facts.num_reqs_padded or facts.num_reqs
        rec.cudagraph_replayed = facts.cudagraph_replayed
        rec.cudagraph_mode = facts.cudagraph_mode
        rec.num_new_reqs = facts.num_new_reqs
        rec.num_ctx_tokens = facts.num_ctx_tokens
        rec.num_gen_tokens = facts.num_gen_tokens
        rec.n_full_attn_groups = facts.n_full_attn_groups

    def set_block_size(self, block_size: int) -> None:
        """The DA kernel block size, which only the ENGINE knows and only after `build()`.

        32 on Gemma and 16 on Qwen, and it is the unit every `att_blk` is rounded to, so a
        record that carries 0 cannot be interpreted. `install()` cannot pass it: the
        builder that resolves it does not exist yet when the recorder is created.
        """
        block_size = int(block_size)
        if block_size <= 0 or block_size == self.dattn_block_size:
            return
        self.dattn_block_size = block_size
        if self._cur >= 0:
            self._recs[self._cur].dattn_block_size = block_size

    def note_scheduler_stats(self, kv_usage: float, num_preempted: int) -> None:
        """`SchedulerStats` observed after step k-1, written into step k. Index shift."""
        self._kv_usage = float(kv_usage)
        self._num_preempted = int(num_preempted)
        if self._cur >= 0:
            self._recs[self._cur].kv_usage = self._kv_usage
            self._recs[self._cur].num_preempted = self._num_preempted

    # -- the protocol surface --------------------------------

    def note_compaction(self, group_id: int, out_seq_lens: Tensor, host_mirror: Sequence[int], host_us: float) -> None:
        """Called once per compacted KV-cache group, immediately after the kernel.

        `out_seq_lens` is the device post-compaction `seq_lens[:R]`, the kernel truth.
        `host_mirror` is the pure host function of `(intervals, seq_len, block_size)`.
        The device copy is D2D and non-blocking; the mirror is what the record carries;
        `audit()` proves they agree.

        EVERY compacted group is mirrored, in its own device row, keyed by `group_id`.
        This used to keep group 0 only, on the stated assumption that "every compacted
        group writes the identical seq_lens vector". That assumption is exactly what the
        audit exists to CHECK, so assuming it made the audit prove nothing about any group
        but the first. `group_id` was accepted and never used, so a
        second group could not even be told apart from the first.

        `rec.rows.att_blk` still carries the FIRST group's mirror, because the record holds
        one attended-length vector per step and every group is supposed to agree. If they
        ever do not, the audit now says so instead of the record quietly being one of them.
        """
        slot = self._open()
        rec = self._recs[slot]
        rec.compact_host_us += float(host_us)
        rec.n_compacted_groups += 1
        mirror = [int(v) for v in host_mirror]
        if rec.n_compacted_groups == 1:
            rec.rows.att_blk.clear()
            rec.rows.att_blk.extend(mirror)
        gidx = self._device_index(int(group_id))
        if gidx < 0:
            self.audit_groups_dropped += 1
            return
        self._mirror[slot][int(group_id)] = mirror
        if self._ring_blk is not None and out_seq_lens is not None:
            n = len(mirror)
            if n:
                self._ring_blk[slot, gidx, :n].copy_(out_seq_lens[:n], non_blocking=True)

    def _device_index(self, group_id: int) -> int:
        """A stable device row for `group_id`, or -1 when the ring has no room left."""
        gidx = self._group_index.get(group_id)
        if gidx is not None:
            return gidx
        if len(self._group_index) >= self.max_compacted_groups:
            return -1
        gidx = len(self._group_index)
        self._group_index[group_id] = gidx
        return gidx

    def note_lp(self, host_us: float, rows: Sequence[tuple[str, int, int, int, int, int]]) -> None:
        """Called once per step from the logits processor, on the host, with no device access.

        Each row is `(req_id, mode, seq_len, att_tok, out_pos, since_adm)`, scheduled
        rows only, in `InputBatch` slot order. That order is the contract: `seq_len` is
        the vanilla KV length the row would read (`n_prompt_tokens + len(output_ids)`),
        not the batch slot index, because it is the denominator of every ratio the
        analysis bins on. A cheap sanity check runs on the audit cadence and is reported
        by `audit()["row_schema_violations"]` rather than raising on the step path.
        """
        rec = self._recs[self._open()]
        rec.lp_host_us += float(host_us)
        r = rec.rows
        r.req.clear()
        r.mode.clear()
        r.seq_len.clear()
        r.att_tok.clear()
        r.out_pos.clear()
        r.since_adm.clear()
        for req, mode, seq_len, att_tok, out_pos, since_adm in rows:
            r.req.append(req)
            r.mode.append(int(mode))
            r.seq_len.append(int(seq_len))
            r.att_tok.append(int(att_tok))
            r.out_pos.append(int(out_pos))
            r.since_adm.append(int(since_adm))
        if self.audit_every and self.step_id % self.audit_every == 0:
            self._check_row_schema(rec)

    def _check_row_schema(self, rec: StepRecord) -> None:
        """`att_tok <= seq_len` must hold for every row: a mask cannot admit more than exists."""
        r = rec.rows
        for i, (sl, at) in enumerate(zip(r.seq_len, r.att_tok, strict=True)):
            if at > sl:
                self.row_schema_violations += 1
                if self.first_row_violation is None:
                    self.first_row_violation = {
                        "step_id": rec.step_id,
                        "row": i,
                        "seq_len": sl,
                        "att_tok": at,
                        "hint": "note_lp row 3 must be seq_len, not the batch slot index",
                    }
                return

    # -- the GPU bracket ---------------------------------------------------------------

    def forward_start(self) -> None:
        slot = self._open()
        if self._ev_fwd:
            self._ev_fwd[slot][0].record()

    def forward_end(self) -> None:
        slot = self._cur if self._cur >= 0 else self._open()
        if self._ev_fwd:
            self._ev_fwd[slot][1].record()
            self._have_fwd[slot] = True

    def compact_start(self) -> None:
        """Open the compaction's device bracket, from `engine.state.compact`.

        The compaction runs in `_prepare_inputs`, BEFORE the forward, so this is the
        first event of the step and it is what opens the slot on a DA step. The host cost
        of the same region arrives separately through `note_compaction`; this pair is the
        device half, and the two together decompose the host bracket
        `(t_ready_ns - t_sched_ns)` that the paper defines as the DA step time. Neither is
        added to `gpu_forward_ms`: the host bracket already contains both and adding them
        would double count (owner decision #11).

        A vanilla step never calls this and keeps `gpu_compact_ms == 0.0`, which is a
        measured zero rather than a missing field.
        """
        slot = self._open()
        if self._ev_cmp:
            self._ev_cmp[slot][0].record()

    def compact_end(self) -> None:
        slot = self._cur if self._cur >= 0 else self._open()
        if self._ev_cmp:
            self._ev_cmp[slot][1].record()
            self._have_cmp[slot] = True

    def sample_start(self) -> None:
        slot = self._open()
        if self._ev_smp:
            self._ev_smp[slot][0].record()

    def sample_end(self) -> None:
        slot = self._cur if self._cur >= 0 else self._open()
        if self._ev_smp:
            self._ev_smp[slot][1].record()
            self._have_smp[slot] = True

    def end_step(self, *, t_ready_ns: int | None = None, facts: StepFacts | None = None) -> None:
        """Close the step, stamp host clock 3, and advance the recorder-local key."""
        if self._cur < 0:
            self._open()
        if facts is not None:
            self.note_facts(facts)
        rec = self._recs[self._cur]
        rec.t_ready_ns = int(t_ready_ns) if t_ready_ns is not None else time.perf_counter_ns()
        self._cur = -1
        self.step_id += 1

    # -- drain, flush, audit -----------------------------------------------------------

    def drain(self, *, early: bool = False, audit: bool = True) -> list[StepRecord]:
        """One `synchronize()` per drain, never per step. Returns detached copies."""
        if self._cur >= 0:
            self.end_step()
        slots = self._slots
        if not slots:
            out, self._overflow = self._overflow, []
            return out
        if self._torch is not None:
            self._torch.cuda.synchronize()
            for slot in slots:
                rec = self._recs[slot]
                if self._have_fwd[slot]:
                    a, b = self._ev_fwd[slot]
                    rec.gpu_forward_ms = float(a.elapsed_time(b))
                if self._have_smp[slot]:
                    a, b = self._ev_smp[slot]
                    rec.gpu_sample_ms = float(a.elapsed_time(b))
                if self._have_cmp[slot]:
                    a, b = self._ev_cmp[slot]
                    rec.gpu_compact_ms = float(a.elapsed_time(b))
            if audit:
                self._audit_slots(slots)
        if early:
            self.n_early_drains += 1
        out: list[StepRecord] = []
        for slot in slots:
            rec = self._recs[slot]
            if early:
                rec.drained_early = True
            out.append(replace(rec, rows=StepRows(**rec.rows.to_dict())))  # type: ignore[arg-type]
        slots.clear()
        if self._overflow:
            out = self._overflow + out
            self._overflow = []
        return out

    def _audit_slots(self, slots: Sequence[int]) -> None:
        """Device ring versus host mirror. Every `audit_every` steps and at window end."""
        blk = self._ring_blk
        if blk is None:
            return
        host = blk.to("cpu")
        for slot in slots:
            rec = self._recs[slot]
            if rec.step_id % self.audit_every and slot != slots[-1]:
                continue
            mirrors = self._mirror[slot]
            if not mirrors:
                continue
            for group_id, mirror in sorted(mirrors.items()):
                if not mirror:
                    continue
                gidx = self._group_index.get(group_id)
                if gidx is None:  # pragma: no cover - set together with the mirror
                    continue
                row = host[slot, gidx]
                self.audit_checked += 1
                got = [int(v) for v in row[: len(mirror)]]
                if got != mirror:
                    self.audit_mismatches += 1
                    if self.first_mismatch is None:
                        self.first_mismatch = {
                            "step_id": rec.step_id,
                            "group_id": group_id,
                            "device": got,
                            "mirror": list(mirror),
                        }

    def audit(self) -> dict[str, object]:
        """`{"checked": n, "mismatches": m, "first_mismatch": ...}`. `m` must be 0."""
        return {
            "window_id": self.window_id,
            "checked": self.audit_checked,
            "mismatches": self.audit_mismatches,
            "first_mismatch": self.first_mismatch,
            "groups_mirrored": len(self._group_index),
            "groups_dropped": self.audit_groups_dropped,
            "row_schema_violations": self.row_schema_violations,
            "first_row_violation": self.first_row_violation,
            "early_drains": self.n_early_drains,
            "steps": self.step_id,
        }

    def attach_writer(self, path: Path | str) -> None:
        if self._writer is not None:
            self._writer.close()
        self._writer = JsonlWriter(path)

    def flush(self) -> int:
        """Drain and serialise. Off the step path, buffered, never inside a step.

        With no writer there is nothing to serialise TO, and draining anyway destroys the
        window: the bench driver arms, runs an arm, disarms and *then* drains in process,
        so an eager drain here silently emptied every cell it ran (found on hardware at
        integration, 2026-09-17, as `n_records: 0` on a cell whose engine had generated).
        `flush` means "push the buffer to the writer"; with no writer that is a no-op, and
        `drain()` stays the only thing that empties the ring.
        """
        if self._writer is None:
            return 0
        recs = self.drain()
        self._writer.extend(recs)
        self._writer.flush()
        self._written += len(recs)
        return len(recs)

    def close(self) -> int:
        n = self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        return n
