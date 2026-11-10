"""`DattnState`: per-request rows, per-builder scratch, the ring buffer.

Two indexes over the same records:

* `by_key`, keyed by the request's identity, which survives preemption. A preempted
  request leaves the persistent batch and comes back later, usually in a different
  slot, with the same live `output_token_ids` list. Keeping the record lets the FSM
  resume instead of replaying every emitted token.
* `rows`, keyed by `InputBatch` slot, which is also the attention-metadata row index.
  This is the index the mask buffers and the compaction use.

`_REC` is the single module-level coupling to `dattn.metrics`: one load and one branch
per step, `None` when measurement is disabled, written by `dattn.metrics.hooks` and never
imported from there.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from dattn.contracts import (
    INF,
    MAX_INTERVALS,
    DattnInvariantError,
    DattnRequest,
    Mask,
    Mode,
)
from dattn.engine import compact as _compact
from dattn.engine import kernel as _kernel
from dattn.engine.scratch import DattnScratch, alloc_scratch

__all__ = ["DATTN_STATE", "DattnState", "RequestState", "set_recorder"]

_REC: Any | None = None

#: The mask that admits everything. A dense row is a fixed point of the compaction, so
#: it is also the safe value for an unknown or padded slot.
_DENSE: Mask = Mask.of((0, INF))


def set_recorder(rec: Any | None) -> None:
    """Install or clear the step recorder. Called only by `dattn.metrics.hooks`."""
    global _REC
    _REC = rec


def get_recorder() -> Any | None:
    """Read the recorder slot. `dattn.metrics.hooks` writes it, nothing else."""
    return _REC


@dataclass(slots=True)
class RequestState:
    """One request's DA state. Lives across preemption, keyed by `key`."""

    key: str
    req_id: str
    #: `None` for a request with no `extra_args["dattn"]` payload. Such a request is
    #: served densely (the vanilla arm), but it still owns its row: the row must exist or
    #: a stale mask from the previous occupant would apply to it, and the recorder's per-row
    #: arrays are indexed by `InputBatch` slot and would go out of alignment.
    request: DattnRequest | None
    output_ids: list[int]
    #: `len(prompt_token_ids)`. With `len(output_ids)` it gives the vanilla KV length,
    #: which is `StepRows.seq_len` and the denominator of every ratio metrics bins on.
    n_prompt: int = 0
    fsm: Any = None
    n_fed: int = 0
    mask: Mask = _DENSE
    mode: int = int(Mode.GLOBAL)
    row: int = -1
    n_updates: int = 0
    #: Steps in which this row was a genuine DECODE row: `query_len == 1` and not
    #: prefilling. `DattnState._gate_for` counts them, because the attention metadata is
    #: the only place that knows. It is what the no-token watchdog in
    #: `DattnLogitsProcessor._advance` counts, and the reason it counts THIS rather than
    #: `n_updates`: a chunked prefill occupies the row for many steps that are not decode
    #: steps and must not be read as a decode that failed to produce a token.
    n_decode_steps: int = 0
    n_preemptions: int = 0
    admitted_at: int = 0

    @property
    def enabled(self) -> bool:
        return self.request is not None and self.request.enabled

    def out_pos(self) -> int:
        return len(self.output_ids)

    def seq_len(self) -> int:
        """The vanilla KV length this row would read. `StepRows.seq_len`."""
        return self.n_prompt + len(self.output_ids)


@dataclass(slots=True)
class _Gate:
    """The per-step activity gate, shared by every DA builder in the step."""

    step_id: int
    rows: int
    active_gpu: torch.Tensor
    active_np: np.ndarray
    seq_lens_np: np.ndarray


@dataclass(slots=True)
class _Counters:
    steps: int = 0
    compactions: int = 0
    passthroughs: int = 0
    rows_compacted: int = 0
    preemptions: int = 0
    audits: int = 0
    audit_mismatches: int = 0
    first_mismatch: dict[str, Any] | None = field(default=None)


class DattnState:
    """The per-request mask store."""

    def __init__(self) -> None:
        self.max_num_seqs: int = 0
        self.device: Any = None
        self.step_id: int = 0
        self.rows: list[RequestState | None] = []
        self.by_key: dict[str, RequestState] = {}
        self.detached: dict[str, RequestState] = {}
        self.counters = _Counters()
        self._intervals_host: torch.Tensor | None = None
        self._counts_host: torch.Tensor | None = None
        self._intervals_gpu: torch.Tensor | None = None
        self._counts_gpu: torch.Tensor | None = None
        self._active_host: torch.Tensor | None = None
        self._active_gpu: torch.Tensor | None = None
        self._row_masked: np.ndarray | None = None
        self._gate: _Gate | None = None
        self._dirty: bool = False
        self._scratch: dict[int, DattnScratch] = {}
        #: Rows of each scratch that hold a MATCHED device/host pair, keyed by builder id.
        #: A cudagraph capture resets it to zero: the base builder may write the device
        #: `seq_lens` during capture (Triton does `seq_lens.fill_(1)`) without the host
        #: mirror moving, so auditing those rows would compare two different things.
        self._valid_rows: dict[int, int] = {}
        self._audit_every: int = 500
        #: Set by `compact`, read by `dattn.engine.api.step_geometry` for metrics.
        self.last_num_reqs_padded: int = 0
        self.last_n_compacted_groups: int = 0
        #: Set by `assertions.validate_engine`, so the RPC surface can find the runner
        #: when `collective_rpc` hands it a bare callable rather than a worker method.
        self.runner: Any = None

    # -- configuration ------------------------------------------------------------------

    @property
    def configured(self) -> bool:
        return self._intervals_host is not None

    def configure(self, max_num_seqs: int, device: Any, *, pin_memory: bool = True) -> None:
        """Allocate the mask transport once. Idempotent for the same size and device."""
        if self.configured:
            if max_num_seqs > self.max_num_seqs:
                raise DattnInvariantError(
                    f"DattnState was configured for {self.max_num_seqs} rows and is now asked for {max_num_seqs}"
                )
            return
        pin = bool(pin_memory) and torch.cuda.is_available()
        self.max_num_seqs = int(max_num_seqs)
        self.device = device
        shape = (self.max_num_seqs, MAX_INTERVALS, 2)
        self._intervals_host = torch.zeros(shape, dtype=torch.int32, pin_memory=pin)
        self._counts_host = torch.zeros((self.max_num_seqs,), dtype=torch.int32, pin_memory=pin)
        self._active_host = torch.zeros((self.max_num_seqs,), dtype=torch.bool, pin_memory=pin)
        self._intervals_gpu = torch.zeros(shape, dtype=torch.int32, device=device)
        self._counts_gpu = torch.zeros((self.max_num_seqs,), dtype=torch.int32, device=device)
        self._active_gpu = torch.zeros((self.max_num_seqs,), dtype=torch.bool, device=device)
        self._row_masked = np.zeros((self.max_num_seqs,), dtype=bool)
        self.rows = [None] * self.max_num_seqs
        for r in range(self.max_num_seqs):
            self._write_row(r, _DENSE)
        self._dirty = True

    def reset(self) -> None:
        """Drop every request record. The buffers and their pointers are untouched."""
        self.rows = [None] * self.max_num_seqs
        self.by_key.clear()
        self.detached.clear()
        if self.configured:
            for r in range(self.max_num_seqs):
                self._write_row(r, _DENSE)
            self._dirty = True

    def _require(self) -> None:
        if not self.configured:
            raise DattnInvariantError(
                "DattnState is not configured. DattnLogitsProcessor.__init__ calls "
                "configure(); if you see this, the logits processor was never installed, "
                "which means --logits-processors dattn.engine.processor:DattnLogitsProcessor "
                "is missing from the engine args (DESIGN.md section 2.7)."
            )

    # -- the request lifecycle ----------------------------------------------------------

    def begin_step(self) -> None:
        self.step_id += 1
        self.counters.steps += 1
        self._gate = None
        self.last_n_compacted_groups = 0

    def n_live_rows(self) -> int:
        """`1 + the highest occupied slot`. vLLM condenses the batch, so this is the count.

        The recorder's per-row arrays and the compaction's host mirror are both sliced to this,
        so `rows.req[i]` and `rows.att_blk[i]` describe the same request.
        """
        for i in range(len(self.rows) - 1, -1, -1):
            if self.rows[i] is not None:
                return i + 1
        return 0

    def admit(
        self,
        row: int,
        key: str,
        req_id: str,
        request: DattnRequest | None,
        output_ids: list[int],
        n_prompt: int = 0,
    ) -> RequestState:
        """Add or re-admit a request at `row`. Re-admission after preemption resumes.

        vLLM does not hand the logits processor a request id, so `key` is whatever
        identity the caller can produce. `DattnLogitsProcessor` uses the `req_id` the
        client put in `extra_args["dattn"]` when there is one and the identity of the
        live output-token list otherwise, which is stable for the life of the request.

        **What a resume restores, and what it does not** (DESIGN.md section 3.6). The
        re-admission branch below restores the FSM: the mode, the mask, the attempt and
        success counters and `n_preemptions` all survive, because the `RequestState` was
        only detached, not dropped. The ATTENTION HISTORY does not survive. On a
        recompute preemption vLLM replays the prompt plus the tokens generated so far as
        a prefill, and `_gate` (see `is_prefilling` below) makes every prefilling row
        inactive, so the replayed tokens are recomputed against a DENSE mask even though
        they were originally produced under a focus. The resumed KV cache therefore holds
        dense-history keys where the uninterrupted run held focused-history keys.

        This is bounded, not benign. It is bounded because the bench INVALIDATES any cell
        whose window saw a preemption (`bench/analysis.py`, `num_preempted` differenced
        per window), so no reported number is computed from a resumed request, and
        because dac behaves identically, so a dattn-vs-dac parity run cannot diverge here.
        It would stop being bounded the moment the bench stopped invalidating preempted
        cells, or if an eval run under memory pressure were reported without that filter:
        the affected rows would carry a DIFFERENT generation than the same request run to
        completion, with no error and no marker on the row. Fixing it properly means
        either refusing recompute preemption for DA requests or replaying the mask
        schedule through the recomputed prefill. Neither is implemented.
        """
        self._require()
        self.remove_row(row)
        st = self.detached.pop(key, None)
        if st is not None and st.output_ids is output_ids:
            st.n_preemptions += 1
            self.counters.preemptions += 1
        else:
            st = RequestState(key=key, req_id=req_id, request=request, output_ids=output_ids, n_prompt=n_prompt)
            if request is not None and request.forced_mask is not None:
                st.mask = request.forced_mask
        st.row = row
        st.n_updates = 0
        st.n_decode_steps = 0
        st.admitted_at = self.step_id
        self.rows[row] = st
        self.by_key[key] = st
        self.set_mask(row, st.mask)
        return st

    def remove_row(self, row: int, *, detach: bool = True) -> RequestState | None:
        """Take the request out of `row`. `detach` keeps the record for a re-admission."""
        self._require()
        st = self.rows[row] if 0 <= row < len(self.rows) else None
        self.rows[row] = None
        self.clear_row(row)
        if st is not None:
            self.by_key.pop(st.key, None)
            st.row = -1
            if detach:
                self.detached[st.key] = st
            if len(self.detached) > 4 * max(self.max_num_seqs, 1):
                self.detached.clear()
        return st

    def finish(self, key: str) -> None:
        """Forget a request for good. Safe to call for an unknown key."""
        self.detached.pop(key, None)
        st = self.by_key.pop(key, None)
        if st is not None and 0 <= st.row < len(self.rows) and self.rows[st.row] is st:
            self.rows[st.row] = None
            self.clear_row(st.row)

    def move_row(self, src: int, dst: int) -> None:
        """One-way move `src -> dst`. `src` becomes dense and empty."""
        self._require()
        st = self.rows[src]
        self.rows[dst] = st
        self.rows[src] = None
        if st is not None:
            st.row = dst
        self._copy_row(src, dst)
        self.clear_row(src)

    def swap_rows(self, a: int, b: int) -> None:
        self._require()
        self.rows[a], self.rows[b] = self.rows[b], self.rows[a]
        for r in (a, b):
            if self.rows[r] is not None:
                self.rows[r].row = r  # type: ignore[union-attr]
        assert self._intervals_host is not None and self._counts_host is not None
        iv_a = self._intervals_host[a].clone()
        cnt_a = int(self._counts_host[a].item())
        mask_a = bool(self._row_masked[a])  # type: ignore[index]
        self._copy_row(b, a)
        self._intervals_host[b].copy_(iv_a)
        self._counts_host[b] = cnt_a
        self._row_masked[b] = mask_a  # type: ignore[index]
        self._dirty = True

    # -- the mask buffers ---------------------------------------------------------------

    def set_mask(self, row: int, mask: Mask) -> None:
        self._require()
        if len(mask.intervals) > MAX_INTERVALS:
            raise DattnInvariantError(
                f"row {row}: a mask with {len(mask.intervals)} intervals cannot be transported; "
                f"MAX_INTERVALS is {MAX_INTERVALS}. PromptLayout is supposed to make this unreachable."
            )
        self._write_row(row, mask)
        self._dirty = True

    def clear_row(self, row: int) -> None:
        """Make `row` dense, which is the identity of the compaction."""
        self._require()
        self._write_row(row, _DENSE)
        self._dirty = True

    def _write_row(self, row: int, mask: Mask) -> None:
        assert self._intervals_host is not None and self._counts_host is not None
        assert self._row_masked is not None
        iv = self._intervals_host[row]
        iv.zero_()
        n = 0
        for s, e in mask.intervals[:MAX_INTERVALS]:
            iv[n, 0] = int(s)
            iv[n, 1] = int(min(e, INF))
            n += 1
        if n == 0:  # an empty mask would keep zero blocks; hold the row dense instead
            iv[0, 0] = 0
            iv[0, 1] = INF
            n = 1
        self._counts_host[row] = n
        self._row_masked[row] = not mask.dense and bool(mask.intervals)

    def _copy_row(self, src: int, dst: int) -> None:
        assert self._intervals_host is not None and self._counts_host is not None
        assert self._row_masked is not None
        self._intervals_host[dst].copy_(self._intervals_host[src])
        self._counts_host[dst] = self._counts_host[src]
        self._row_masked[dst] = self._row_masked[src]
        self._dirty = True

    def flush(self) -> None:
        """Two non-blocking H2D copies from pinned host."""
        self._require()
        if not self._dirty:
            return
        assert self._intervals_gpu is not None and self._counts_gpu is not None
        assert self._intervals_host is not None and self._counts_host is not None
        self._intervals_gpu.copy_(self._intervals_host, non_blocking=True)
        self._counts_gpu.copy_(self._counts_host, non_blocking=True)
        self._dirty = False

    # -- compaction ---------------------------------------------------------------------

    def scratch_for(self, builder: Any, n_cols: int, bs: int, device: Any) -> DattnScratch:
        """The per-builder derived buffers. Allocated once, never resized (A20)."""
        sc = getattr(builder, "scratch", None)
        if sc is None:
            rows = max(self.max_num_seqs, 1)
            sc = alloc_scratch(rows, n_cols, bs, device)
            builder.scratch = sc
            self._scratch[id(builder)] = sc
        if sc.n_cols != n_cols or sc.block_size != bs:
            raise DattnInvariantError(
                f"scratch was allocated for n_cols={sc.n_cols}, block_size={sc.block_size} and is now "
                f"asked for n_cols={n_cols}, block_size={bs}. Scratch is never resized: a captured "
                "graph holds its pointers (DESIGN.md section 2.4)."
            )
        return sc

    def _gate_for(self, cm: Any, rows: int) -> _Gate:
        """`active` for this step: masked, decode-only, not prefilling, not padded."""
        g = self._gate
        if g is not None and g.step_id == self.step_id and g.rows == rows:
            return g
        assert self._active_host is not None and self._active_gpu is not None
        assert self._row_masked is not None

        qsl = cm.query_start_loc_cpu
        query_lens = (qsl[1 : rows + 1] - qsl[:rows]).numpy()
        # A prefilling row is never compacted. That is what makes chunked prefill safe,
        # and it is also why a RESUMED request recomputes its generated history against a
        # dense mask rather than the mask that produced it. See `_add` for the bound on
        # that and for what would break if the bench stopped invalidating preempted cells.
        decode_np = query_lens == 1
        prefilling = getattr(cm, "is_prefilling", None)
        if prefilling is not None:
            pf = prefilling[:rows]
            decode_np = decode_np & ~(pf.cpu().numpy() if pf.is_cuda else pf.numpy())
        # `decode_np` is "this row is decoding", INDEPENDENT of whether it is masked.
        # `active` is the compaction's gate and is the masked rows among them. The
        # no-token watchdog needs the former: it is asking whether decode steps are
        # producing tokens, and a DA row is dense until its first focus tag.
        self._note_decode_steps(decode_np, rows)
        active_np = self._row_masked[:rows] & decode_np

        seq_cpu = getattr(cm, "_seq_lens_cpu", None)
        if seq_cpu is None:
            raise DattnInvariantError(
                "CommonAttentionMetadata carries no host seq_lens mirror. dattn derives "
                "max_seq_len and the audit mirror from it without a device sync, so the "
                "async-spec-decode path (which sets it to None) is refused at boot (A10)."
            )
        seq_lens_np = seq_cpu[:rows].numpy().astype(np.int64)

        self._active_host[:rows].copy_(torch.from_numpy(active_np))
        self._active_gpu[:rows].copy_(self._active_host[:rows], non_blocking=True)
        g = _Gate(
            step_id=self.step_id,
            rows=rows,
            active_gpu=self._active_gpu[:rows],
            active_np=active_np,
            seq_lens_np=seq_lens_np,
        )
        self._gate = g
        return g

    def _note_decode_steps(self, decode_np: Any, rows: int) -> None:
        """Credit one decode step to every row that really decoded this step.

        Called once per step from `_gate_for`, which caches on `step_id`.
        """
        for r in range(rows):
            st = self.rows[r]
            if st is not None and bool(decode_np[r]):
                st.n_decode_steps += 1

    def compact(self, cm: Any, builder: Any) -> Any:
        """`cm: CommonAttentionMetadata -> CommonAttentionMetadata`, re-pointed by `replace`."""
        self._require()
        t0 = time.perf_counter()
        bs = int(builder.block_size)
        rows = int(cm.num_reqs)
        bt = cm.block_table_tensor
        width = int(bt.shape[1])
        scratch = self.scratch_for(builder, width, bs, bt.device)
        want = getattr(builder, "_dattn_capture_ptrs", None)
        if want is not None and scratch.pointers() != want:
            raise DattnInvariantError(
                f"A20: the scratch a captured graph holds has moved: {want} -> {scratch.pointers()}"
            )

        if rows > scratch.n_rows:
            raise DattnInvariantError(
                f"batch has {rows} rows but the scratch holds {scratch.n_rows}. "
                "A20: max_num_seqs must not exceed the allocated row count."
            )
        if rows == 0:
            return cm

        # The DEVICE half of the compaction's bracket. The host half is `t0` above, and
        # both are decompositions of `(t_ready_ns - t_sched_ns)`, which main.tex:843
        # defines as the DA step time. Resolved through `getattr` rather than added to
        # `Any`, so a recorder that predates this pair keeps working and
        # the engine layer still imports nothing from `dattn.metrics`.
        rec0 = _REC
        cmp_start = None if rec0 is None else getattr(rec0, "compact_start", None)
        if cmp_start is not None:
            cmp_start()

        n_cols = max(1, min(width, -(-int(cm.max_seq_len) // bs)))
        gate = self._gate_for(cm, rows)
        self.flush()

        out_bt = scratch.block_table[:rows, :n_cols]
        out_sl = scratch.seq_lens[:rows]
        out_nk = scratch.n_kept[:rows]
        assert self._intervals_gpu is not None and self._counts_gpu is not None
        impl = _kernel.compact_rows
        impl(
            self._intervals_gpu[:rows],
            self._counts_gpu[:rows],
            cm.seq_lens[:rows],
            gate.active_gpu,
            bt[:rows],
            bs,
            out_block_table=out_bt,
            out_seq_lens=out_sl,
            out_n_kept=out_nk,
        )

        # The host mirror. No device read, so no sync: max_seq_len, the compacted CPU
        # mirror and the audit all come from here (DESIGN.md sections 4.3 and 5.3).
        assert self._intervals_host is not None and self._counts_host is not None
        out_len_np, n_kept_np, act_np = _compact.host_lengths(
            self._intervals_host[:rows].numpy(),
            self._counts_host[:rows].numpy(),
            gate.seq_lens_np,
            gate.active_np,
            bs,
            n_cols,
        )
        if _compact.checks_enabled():
            _compact.assert_row_invariants(
                self._intervals_host[:rows].numpy(),
                self._counts_host[:rows].numpy(),
                gate.seq_lens_np,
                act_np,
                out_len_np,
                n_kept_np,
                bs,
            )
        scratch.seq_lens_cpu[:rows].copy_(torch.from_numpy(out_len_np))
        self._valid_rows[id(builder)] = rows

        max_seq_len = min(int(out_len_np.max()), int(cm.max_seq_len))
        needs_cpu = bool(getattr(builder, "dattn_needs_seq_lens_cpu", False))
        cm2 = cm.replace(
            block_table_tensor=scratch.block_table[:rows],
            seq_lens=out_sl,
            max_seq_len=max_seq_len,
            _seq_lens_cpu=scratch.seq_lens_cpu[:rows] if needs_cpu else None,
            _num_computed_tokens_cpu=None,
            _num_computed_tokens_cache=None,
            _token_to_req_indices_cache=None,
        )

        self.counters.compactions += 1
        self.last_num_reqs_padded = rows
        self.last_n_compacted_groups += 1
        n_active = int(act_np.sum())
        self.counters.rows_compacted += n_active
        rec = _REC
        if rec is not None:
            cmp_end = getattr(rec, "compact_end", None)
            if cmp_end is not None:
                cmp_end()
            # Slice to the live rows so `rows.att_blk[i]` and `rows.req[i]` describe the
            # same request: `rows` here is num_reqs_padded and the padded tail is not a
            # request at all.
            #
            # A HOLE IS NOT A ROW. `n_live_rows()` is `1 + the highest occupied slot`, so
            # a `None` below that slot -- a request removed this step, before the batch
            # condenses -- is inside the slice, while `processor._advance` SKIPS it when it
            # builds `rows.req`. The two vectors then differ in length and, worse, pair
            # `att_blk[i]` with a different request's `req[i]` for every row after the hole.
            # The mask ratio is `att_blk / seq_len`, so a
            # misaligned pair is a wrong number on the paper's own axis, with no symptom.
            # THE PADDING IS UPWARD, AND THAT IS ASSERTED, NOT ASSUMED. `processor` builds
            # its per-row list out of `n_live_rows()` with nothing bounding it by this
            # batch's width, while the clamp below takes `min(n_live_rows(), rows)`. The
            # two agree exactly while `rows` (num_reqs_padded) is at least the live row
            # count, which is what a cudagraph bucket does; if it ever were not, the two
            # sides would describe different requests and the mask ratio would be wrong
            # with no symptom.
            n_live = self.n_live_rows()
            if rows < n_live:
                raise DattnInvariantError(
                    f"the batch is {rows} rows wide but {n_live} rows are live: the recorder pairs "
                    "`rows.req[i]` with `rows.att_blk[i]` by index and they would describe different "
                    "requests"
                )
            live = min(n_live, rows)
            keep = [i for i in range(live) if self.rows[i] is not None]
            if len(keep) == live:
                dev_lens, mirror = out_sl[:live], out_len_np[:live].tolist()
            else:
                # The rare path: gather the same slots the recorder's row list kept. One small
                # D2D gather, only on a step that really has a hole.
                idx = torch.as_tensor(keep, dtype=torch.long, device=out_sl.device)
                dev_lens, mirror = out_sl.index_select(0, idx), [int(out_len_np[i]) for i in keep]
            rec.note_compaction(
                int(getattr(builder, "dattn_group_id", -1)),
                dev_lens,
                mirror,
                (time.perf_counter() - t0) * 1e6,
            )
        return cm2

    def stage_for_capture(self, cm: Any, builder: Any) -> Any:
        """Stage identity metadata into the scratch pointers captured by the DA graph."""
        self._require()
        bs = int(builder.block_size)
        rows = int(cm.num_reqs)
        bt = cm.block_table_tensor
        width = int(bt.shape[1])
        scratch = self.scratch_for(builder, width, bs, bt.device)
        if rows == 0:
            return cm
        if rows > scratch.n_rows:
            raise DattnInvariantError(f"capture batch has {rows} rows but the scratch holds {scratch.n_rows} (A20).")
        scratch.block_table[:rows].copy_(bt[:rows])
        scratch.seq_lens[:rows].copy_(cm.seq_lens[:rows])
        scratch.n_kept[:rows].zero_()
        self._valid_rows[id(builder)] = 0
        needs_cpu = bool(getattr(builder, "dattn_needs_seq_lens_cpu", False))
        seq_cpu = getattr(cm, "_seq_lens_cpu", None)
        if needs_cpu and seq_cpu is not None:
            scratch.seq_lens_cpu[:rows].copy_(seq_cpu[:rows])
        return cm.replace(
            block_table_tensor=scratch.block_table[:rows],
            seq_lens=scratch.seq_lens[:rows],
            _seq_lens_cpu=scratch.seq_lens_cpu[:rows] if (needs_cpu and seq_cpu is not None) else None,
            _num_computed_tokens_cpu=None,
            _num_computed_tokens_cache=None,
            _token_to_req_indices_cache=None,
        )

    # -- introspection ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """What `dattn_probe` and the GPU tests read. Host only, no device access."""
        return {
            "step_id": self.step_id,
            "max_num_seqs": self.max_num_seqs,
            "configured": self.configured,
            "n_rows_live": sum(1 for r in self.rows if r is not None),
            "n_detached": len(self.detached),
            "counters": {
                "steps": self.counters.steps,
                "compactions": self.counters.compactions,
                "passthroughs": self.counters.passthroughs,
                "rows_compacted": self.counters.rows_compacted,
                "preemptions": self.counters.preemptions,
                "audits": self.counters.audits,
                "audit_mismatches": self.counters.audit_mismatches,
                "first_mismatch": self.counters.first_mismatch,
            },
            "scratch": {
                str(k): {"n_cols": v.n_cols, "block_size": v.block_size, "ptrs": v.pointers()}
                for k, v in self._scratch.items()
            },
        }

    def audit(self, builder: Any | None = None) -> dict[str, Any]:
        """Compare the device `seq_lens` against the host mirror. One sync, on purpose.

        This is what turns "the recorded per-row attended count is truthful" from an
        assumption into a tested invariant (DESIGN.md section 5.3). Only rows the last
        `compact()` wrote are compared: a cudagraph capture writes the device side
        without the host mirror, and padded rows beyond `R` are stale by construction.
        """
        results: dict[str, Any] = {"checked": 0, "mismatches": 0, "first_mismatch": None}
        if builder is not None:
            items = [(id(builder), getattr(builder, "scratch", None))]
        else:
            items = list(self._scratch.items())
        for key, sc in items:
            if sc is None:
                continue
            n = int(min(self._valid_rows.get(key, 0), sc.seq_lens.shape[0], sc.seq_lens_cpu.shape[0]))
            if n <= 0:
                continue
            dev = sc.seq_lens[:n].cpu()
            host = sc.seq_lens_cpu[:n]
            results["checked"] += n
            bad = torch.nonzero(dev != host).flatten().tolist()
            if bad:
                results["mismatches"] += len(bad)
                if results["first_mismatch"] is None:
                    r = bad[0]
                    results["first_mismatch"] = {
                        "row": r,
                        "device": int(dev[r]),
                        "host": int(host[r]),
                    }
        self.counters.audits += 1
        self.counters.audit_mismatches += int(results["mismatches"])
        if results["first_mismatch"] and self.counters.first_mismatch is None:
            self.counters.first_mismatch = results["first_mismatch"]
        return results


#: The process-wide instance the builder and the logits processor reach through.
DATTN_STATE: DattnState = DattnState()
