"""`DattnState`: the mask buffers, the request lifecycle and `compact` (WP5)."""

from __future__ import annotations

import pytest
import torch
from fakes import FakeBuilder, make_cm

from dattn.contracts import INF, MAX_INTERVALS, NULL_BLOCK_ID, DattnInvariantError, DattnRequest, Mask
from dattn.engine import compact as compact_mod
from dattn.engine.state import DattnState


def make_state(rows: int = 8) -> DattnState:
    st = DattnState()
    st.configure(rows, torch.device("cpu"), pin_memory=False)
    return st


def req(layout, mask=None, enabled=True) -> DattnRequest:
    return DattnRequest(layout=layout, enabled=enabled, forced_mask=mask)


# ---------------------------------------------------------------------------------------
# buffers
# ---------------------------------------------------------------------------------------


def test_every_row_starts_dense() -> None:
    st = make_state(4)
    assert st._counts_host.tolist() == [1, 1, 1, 1]
    assert st._intervals_host[0, 0].tolist() == [0, INF]
    assert not st._row_masked.any()


def test_da_compaction_refuses_scratch_that_moved_after_capture() -> None:
    st = make_state(4)
    builder = FakeBuilder()
    cm = make_cm([64], n_cols=8)
    st.stage_for_capture(cm, builder)
    builder._dattn_capture_ptrs = (1, 2, 3)
    with pytest.raises(DattnInvariantError, match="scratch a captured graph holds has moved"):
        st.compact(cm, builder)


def test_set_mask_writes_the_host_mirror_and_marks_the_row() -> None:
    st = make_state(4)
    st.set_mask(2, Mask.of((0, 16), (400, INF)))
    assert int(st._counts_host[2]) == 2
    assert st._intervals_host[2, :2].tolist() == [[0, 16], [400, INF]]
    assert bool(st._row_masked[2])
    st.clear_row(2)
    assert not bool(st._row_masked[2])
    assert int(st._counts_host[2]) == 1


def test_a_dense_mask_is_not_marked_as_masked() -> None:
    st = make_state(4)
    st.set_mask(1, Mask.of((0, INF)))
    assert not bool(st._row_masked[1])


def test_an_empty_mask_is_held_dense_rather_than_keeping_zero_blocks() -> None:
    st = make_state(4)
    st.set_mask(1, Mask.of())
    assert int(st._counts_host[1]) == 1
    assert st._intervals_host[1, 0].tolist() == [0, INF]
    assert not bool(st._row_masked[1])


def test_a_mask_with_too_many_intervals_is_refused_loudly() -> None:
    st = make_state(4)
    too_many = Mask(tuple((10 * i, 10 * i + 1) for i in range(MAX_INTERVALS + 1)))
    with pytest.raises(DattnInvariantError, match="MAX_INTERVALS"):
        st.set_mask(0, too_many)


def test_flush_is_a_no_op_when_nothing_changed() -> None:
    st = make_state(4)
    st.flush()
    assert st._dirty is False
    st.set_mask(0, Mask.of((0, 16), (100, INF)))
    assert st._dirty is True
    st.flush()
    assert torch.equal(st._intervals_gpu, st._intervals_host)
    assert st._dirty is False


# ---------------------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------------------


def test_admit_remove_and_finish(layout) -> None:
    st = make_state(4)
    out: list[int] = []
    rs = st.admit(1, "r1", "r1", req(layout), out)
    assert st.rows[1] is rs and st.by_key["r1"] is rs and rs.row == 1
    st.remove_row(1)
    assert st.rows[1] is None and "r1" not in st.by_key and "r1" in st.detached
    st.finish("r1")
    assert "r1" not in st.detached


def test_preemption_resumes_the_same_record(layout) -> None:
    st = make_state(4)
    out: list[int] = [1, 2, 3]
    rs = st.admit(0, "r1", "r1", req(layout), out)
    rs.n_fed = 3
    rs.mask = Mask.of((0, 16), (480, INF))
    st.remove_row(0)
    again = st.admit(2, "r1", "r1", req(layout), out)
    assert again is rs
    assert again.n_fed == 3, "a resumed request must not replay tokens the FSM already saw"
    assert again.n_preemptions == 1
    assert st.counters.preemptions == 1
    assert st._intervals_host[2, :2].tolist() == [[0, 16], [480, INF]]


def test_a_different_output_list_forces_a_fresh_record(layout) -> None:
    st = make_state(4)
    rs = st.admit(0, "r1", "r1", req(layout), [1, 2])
    rs.n_fed = 2
    st.remove_row(0)
    again = st.admit(0, "r1", "r1", req(layout), [9])
    assert again is not rs and again.n_fed == 0


def test_admitting_over_a_live_row_evicts_it(layout) -> None:
    st = make_state(4)
    st.admit(0, "a", "a", req(layout), [])
    st.set_mask(0, Mask.of((0, 16), (480, INF)))
    st.admit(0, "b", "b", req(layout), [])
    assert st.rows[0].key == "b"
    assert not bool(st._row_masked[0]), "a stale mask must never survive into a new request"


def test_move_row_carries_the_mask_and_leaves_the_source_dense(layout) -> None:
    st = make_state(4)
    rs = st.admit(0, "a", "a", req(layout), [])
    st.set_mask(0, Mask.of((0, 16), (480, INF)))
    st.move_row(0, 3)
    assert st.rows[3] is rs and rs.row == 3 and st.rows[0] is None
    assert st._intervals_host[3, :2].tolist() == [[0, 16], [480, INF]]
    assert not bool(st._row_masked[0])
    assert int(st._counts_host[0]) == 1


def test_swap_rows_exchanges_both_records_and_both_masks(layout) -> None:
    st = make_state(4)
    a = st.admit(0, "a", "a", req(layout), [])
    b = st.admit(1, "b", "b", req(layout), [])
    st.set_mask(0, Mask.of((0, 16), (480, INF)))
    st.set_mask(1, Mask.of((0, 8), (500, INF)))
    st.swap_rows(0, 1)
    assert st.rows[0] is b and st.rows[1] is a
    assert a.row == 1 and b.row == 0
    assert st._intervals_host[0, :2].tolist() == [[0, 8], [500, INF]]
    assert st._intervals_host[1, :2].tolist() == [[0, 16], [480, INF]]


def test_a_forced_mask_is_written_at_admission(layout) -> None:
    st = make_state(4)
    st.admit(0, "a", "a", req(layout, Mask.of((0, 16), (480, INF))), [])
    assert st._intervals_host[0, :2].tolist() == [[0, 16], [480, INF]]
    assert bool(st._row_masked[0])


def test_only_a_real_decode_row_is_credited_a_decode_step(layout) -> None:
    """`n_decode_steps` is what the no-token watchdog counts, so a prefilling row and a
    chunked row must not be credited. Row 0 decodes, row 1 is mid-prefill."""
    st = make_state(4)
    a = st.admit(0, "a", "a", req(layout), [])
    b = st.admit(1, "b", "b", req(layout), [])
    st.begin_step()
    st.compact(make_cm([200, 300], n_cols=32, query_lens=[1, 128]), FakeBuilder(block_size=16))
    assert (a.n_decode_steps, b.n_decode_steps) == (1, 0)
    # A row with query_len 1 that vLLM still calls prefilling is not decoding either.
    st.begin_step()
    cm = make_cm([201, 428], n_cols=32, query_lens=[1, 1], prefilling=[False, True])
    st.compact(cm, FakeBuilder(block_size=16))
    assert (a.n_decode_steps, b.n_decode_steps) == (2, 0)


def test_crediting_a_decode_step_does_not_widen_the_compaction_gate(layout) -> None:
    """A dense row that is decoding is credited, and still not active."""
    st = make_state(4)
    a = st.admit(0, "a", "a", req(layout), [])
    st.begin_step()
    st.compact(make_cm([200], n_cols=32, query_lens=[1]), FakeBuilder(block_size=16))
    assert a.n_decode_steps == 1
    assert st._gate is not None and not bool(st._gate.active_np[0])


def test_an_unconfigured_state_names_the_missing_flag() -> None:
    st = DattnState()
    with pytest.raises(DattnInvariantError, match="--logits-processors"):
        st.set_mask(0, Mask.of((0, INF)))


# ---------------------------------------------------------------------------------------
# compact
# ---------------------------------------------------------------------------------------


def test_compact_never_touches_the_tensors_vllm_owns(layout) -> None:
    st = make_state(4)
    cm = make_cm([200, 300], n_cols=32)
    before_bt = cm.block_table_tensor.clone()
    before_sl = cm.seq_lens.clone()
    st.admit(0, "a", "a", req(layout), [])
    st.set_mask(0, Mask.of((0, 16), (160, INF)))
    st.begin_step()
    cm2 = st.compact(cm, FakeBuilder(block_size=16))
    assert torch.equal(cm.block_table_tensor, before_bt)
    assert torch.equal(cm.seq_lens, before_sl)
    assert cm2.seq_lens is not cm.seq_lens
    assert cm2.block_table_tensor.data_ptr() != cm.block_table_tensor.data_ptr()
    assert cm2 is not cm


def test_compact_shrinks_the_masked_row_and_passes_the_others_through(layout) -> None:
    st = make_state(4)
    cm = make_cm([200, 300], n_cols=32)
    st.admit(0, "a", "a", req(layout), [])
    st.set_mask(0, Mask.of((0, 16), (160, INF)))
    st.begin_step()
    cm2 = st.compact(cm, FakeBuilder(block_size=16))
    blocks, out_len = compact_mod.oracle([[0, 16], [160, INF]], 2, 200, 16)
    assert cm2.seq_lens.tolist() == [out_len, 300]
    row = cm2.block_table_tensor[0].tolist()
    assert row[: len(blocks)] == [int(cm.block_table_tensor[0, b]) for b in blocks]
    assert set(row[len(blocks) : 13]) == {NULL_BLOCK_ID}
    assert cm2.block_table_tensor[1].tolist()[:19] == cm.block_table_tensor[1].tolist()[:19]


def test_max_seq_len_comes_from_the_host_mirror_and_is_never_raised(layout) -> None:
    st = make_state(4)
    cm = make_cm([200, 300], n_cols=32)
    st.admit(0, "a", "a", req(layout), [])
    st.set_mask(0, Mask.of((0, 16), (160, INF)))
    st.begin_step()
    cm2 = st.compact(cm, FakeBuilder(block_size=16))
    assert cm2.max_seq_len == max(cm2.seq_lens.tolist())
    assert cm2.max_seq_len <= cm.max_seq_len


def test_the_deprecated_cpu_caches_are_cleared(layout) -> None:
    st = make_state(4)
    cm = make_cm([200], n_cols=32)
    st.begin_step()
    cm2 = st.compact(cm, FakeBuilder(block_size=16))
    assert cm2._num_computed_tokens_cpu is None
    assert cm2._num_computed_tokens_cache is None
    assert cm2._token_to_req_indices_cache is None
    assert cm2._seq_lens_cpu is None, "Triton and trtllm read GPU tensors; a mirror would only sync"
    assert cm2.seq_lens_cpu_upper_bound is cm.seq_lens_cpu_upper_bound


def test_the_flashinfer_native_profile_gets_the_compacted_cpu_mirror(layout) -> None:
    st = make_state(4)
    cm = make_cm([200], n_cols=32)
    st.admit(0, "a", "a", req(layout), [])
    st.set_mask(0, Mask.of((0, 16), (160, INF)))
    st.begin_step()
    cm2 = st.compact(cm, FakeBuilder(block_size=16, dattn_needs_seq_lens_cpu=True))
    assert cm2._seq_lens_cpu is not None
    assert cm2._seq_lens_cpu.tolist() == cm2.seq_lens.tolist()
    assert cm2._seq_lens_cpu.tolist() != cm._seq_lens_cpu.tolist()


def test_prefill_rows_are_never_compacted(layout) -> None:
    st = make_state(4)
    cm = make_cm([200, 200], n_cols=32, query_lens=[1, 1], prefilling=[True, False])
    for row, key in ((0, "a"), (1, "b")):
        st.admit(row, key, key, req(layout), [])
        st.set_mask(row, Mask.of((0, 16), (160, INF)))
    st.begin_step()
    cm2 = st.compact(cm, FakeBuilder(block_size=16))
    assert cm2.seq_lens[0] == 200, "the final chunk of a chunked prefill can schedule one token"
    assert cm2.seq_lens[1] < 200


def test_multi_token_query_rows_are_never_compacted(layout) -> None:
    st = make_state(4)
    cm = make_cm([200, 200], n_cols=32, query_lens=[4, 1])
    for row, key in ((0, "a"), (1, "b")):
        st.admit(row, key, key, req(layout), [])
        st.set_mask(row, Mask.of((0, 16), (160, INF)))
    st.begin_step()
    cm2 = st.compact(cm, FakeBuilder(block_size=16))
    assert cm2.seq_lens[0] == 200
    assert cm2.seq_lens[1] < 200


def test_the_scratch_is_allocated_once_and_keeps_its_pointers(layout) -> None:
    st = make_state(4)
    builder = FakeBuilder(block_size=16)
    st.begin_step()
    st.compact(make_cm([200, 300], n_cols=32), builder)
    ptrs = builder.scratch.pointers()
    for seq in ([100], [200, 300, 400]):
        st.begin_step()
        st.compact(make_cm(seq, n_cols=32), builder)
        assert builder.scratch.pointers() == ptrs


def test_a_changed_block_table_width_is_refused_rather_than_reallocated(layout) -> None:
    st = make_state(4)
    builder = FakeBuilder(block_size=16)
    st.begin_step()
    st.compact(make_cm([200], n_cols=32), builder)
    st.begin_step()
    with pytest.raises(DattnInvariantError, match="never resized"):
        st.compact(make_cm([200], n_cols=64), builder)


def test_a_batch_wider_than_the_scratch_is_refused(layout) -> None:
    st = make_state(2)
    st.begin_step()
    with pytest.raises(DattnInvariantError, match="A20"):
        st.compact(make_cm([10, 10, 10], n_cols=32), FakeBuilder(block_size=16))


def test_stage_for_capture_binds_the_captured_graph_to_the_scratch(layout) -> None:
    st = make_state(4)
    builder = FakeBuilder(block_size=16)
    cm = make_cm([200, 300], n_cols=32)
    cm.max_seq_len = 1 << 20  # the runner sets max_model_len during capture
    cm2 = st.stage_for_capture(cm, builder)
    assert cm2.block_table_tensor.data_ptr() == builder.scratch.block_table.data_ptr()
    assert cm2.seq_lens.data_ptr() == builder.scratch.seq_lens.data_ptr()
    assert torch.equal(cm2.block_table_tensor, cm.block_table_tensor)
    assert torch.equal(cm2.seq_lens, cm.seq_lens)
    assert cm2.max_seq_len == cm.max_seq_len, "never write max_seq_len during capture"


def test_the_gate_is_computed_once_per_step_and_shared_by_every_builder(layout) -> None:
    st = make_state(4)
    cm = make_cm([200], n_cols=32)
    st.begin_step()
    st.compact(cm, FakeBuilder(block_size=16))
    gate = st._gate
    st.compact(cm, FakeBuilder(block_size=16))
    assert st._gate is gate
    st.begin_step()
    st.compact(cm, FakeBuilder(block_size=16))
    assert st._gate is not gate


def test_audit_compares_the_device_seq_lens_against_the_host_mirror(layout) -> None:
    st = make_state(4)
    builder = FakeBuilder(block_size=16)
    st.admit(0, "a", "a", req(layout), [])
    st.set_mask(0, Mask.of((0, 16), (160, INF)))
    st.begin_step()
    st.compact(make_cm([200, 300], n_cols=32), builder)
    assert st.audit(builder)["mismatches"] == 0
    builder.scratch.seq_lens[0] += 1
    assert st.audit(builder)["mismatches"] == 1


def test_the_check_env_catches_a_hand_written_bad_mask(layout, monkeypatch) -> None:
    monkeypatch.setenv("DATTN_CHECK", "1")
    st = make_state(4)
    st.admit(0, "a", "a", req(layout), [])
    st.set_mask(0, Mask.of((0, 16), (5000, INF)))  # the W 2.4 shape, on a short row
    st.begin_step()
    with pytest.raises(DattnInvariantError, match="at or beyond seq_len"):
        st.compact(make_cm([20], n_cols=32), FakeBuilder(block_size=16))


# ---------------------------------------------------------------------------------------
# Lane C row alignment across a hole (review item 37 residual)
# ---------------------------------------------------------------------------------------


def test_a_none_slot_does_not_misalign_the_compaction_mirror(layout) -> None:
    """`rows.att_blk[i]` and `rows.req[i]` must describe the SAME request.

    `n_live_rows()` is `1 + the highest occupied slot`, so a request removed this step
    leaves a `None` INSIDE that range, while the logits processor's row list skips it.
    Slicing the mirror to `n_live_rows()` therefore handed Lane C one more entry than
    there were rows and paired every row after the hole with the next request's attended
    length -- a wrong mask ratio, on the paper's own axis, with no symptom.
    """

    class Rec:
        def __init__(self) -> None:
            self.mirrors: list[list[int]] = []
            self.dev_lens: list[list[int]] = []

        def note_lp(self, host_us, rows) -> None: ...

        def note_compaction(self, group_id, out_seq_lens, host_mirror, host_us) -> None:
            self.mirrors.append([int(v) for v in host_mirror])
            self.dev_lens.append([int(v) for v in out_seq_lens.tolist()])

    from dattn.engine import state as state_mod

    st = make_state(4)
    st.admit(0, "a", "a", req(layout), [])
    st.admit(1, "b", "b", req(layout), [])
    st.admit(2, "c", "c", req(layout), [])
    st.remove_row(1)  # the hole: slot 1 is None, slot 2 is still live
    assert st.n_live_rows() == 3, "the hole is INSIDE the live range, which is the whole point"

    rec = Rec()
    state_mod.set_recorder(rec)
    try:
        st.begin_step()
        st.compact(make_cm([600, 700, 800], n_cols=32, query_lens=[1, 1, 1]), FakeBuilder(block_size=16))
    finally:
        state_mod.set_recorder(None)

    assert rec.mirrors, "the compaction did not reach the recorder at all"
    live_rows = [i for i in range(st.n_live_rows()) if st.rows[i] is not None]
    assert len(rec.mirrors[0]) == len(live_rows) == 2
    assert len(rec.dev_lens[0]) == len(live_rows), "the device vector must be the same rows as the mirror"


def test_a_batch_narrower_than_the_live_rows_is_refused(layout) -> None:
    """Lane C pairs `rows.req[i]` with `rows.att_blk[i]` BY INDEX (review item 9 residual).

    `processor` builds its row list out of `n_live_rows()` with nothing bounding it by the
    batch width, and `compact` clamps its mirror to `min(n_live_rows(), num_reqs_padded)`.
    The two agree exactly while the cudagraph padding is UPWARD. If a batch were ever
    narrower than the live rows they would describe different requests, and the mask ratio
    -- the paper's own x-axis -- would be wrong with no symptom, so it is asserted.
    """

    class Rec:
        def note_lp(self, host_us, rows) -> None: ...

        def note_compaction(self, group_id, out_seq_lens, host_mirror, host_us) -> None: ...

    from dattn.engine import state as state_mod

    st = make_state(4)
    st.admit(0, "a", "a", req(layout), [])
    st.admit(1, "b", "b", req(layout), [])
    state_mod.set_recorder(Rec())
    try:
        st.begin_step()
        with pytest.raises(DattnInvariantError, match="rows are live"):
            st.compact(make_cm([600], n_cols=32, query_lens=[1]), FakeBuilder(block_size=16))
    finally:
        state_mod.set_recorder(None)
