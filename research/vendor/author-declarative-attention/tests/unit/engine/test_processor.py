"""`DattnLogitsProcessor`: batch updates, the one-step lag and the `-1` guard (WP5).

Lane A's `DeclarativeFSM` is stubbed (`_fakes.StubFSM`), so this directory tests the
vLLM-facing half on its own. The contract tests pin the vLLM half.
"""

from __future__ import annotations

import pytest
import torch
from fakes import FakeBatchUpdate, FakeSamplingParams, StubFSM
from fakes import FakeTokenizer as RecordingTokenizer

from dattn.contracts import INF, DattnInvariantError, DattnRequest, Mask, Mode
from dattn.engine import processor as proc
from dattn.engine.state import DattnState


class Swap:
    name = "SWAP"


class Uni:
    name = "UNIDIRECTIONAL"


@pytest.fixture
def lp(monkeypatch, layout):
    monkeypatch.setattr(proc, "_make_fsm", StubFSM)
    p = proc.DattnLogitsProcessor()
    p.state = DattnState()
    p.state.configure(8, torch.device("cpu"), pin_memory=False)
    p._decode = lambda tid: f"<{tid}>"  # type: ignore[assignment]
    p._text = lambda tid: f"<{tid}>"  # type: ignore[method-assign]
    return p


def params(layout, *, mask=None, enabled=True, req_id="") -> FakeSamplingParams:
    payload = DattnRequest(layout=layout, enabled=enabled, forced_mask=mask).to_extra_args()
    if req_id:
        payload["req_id"] = req_id
    return FakeSamplingParams(extra_args={"dattn": payload})


def added(index, layout, out, *, mask=None, enabled=True, req_id=""):
    return (index, params(layout, mask=mask, enabled=enabled, req_id=req_id), [0] * layout.n_prompt_tokens, out)


# ---------------------------------------------------------------------------------------


def test_the_processor_never_touches_logits_but_is_not_argmax_invariant() -> None:
    """False on purpose: an argmax-invariant processor's `apply` never runs under greedy.

    `Sampler` applies the argmax-invariant list on the random branch only and the
    non-invariant list on the greedy branch only, so at `temperature=0.0` -- every number
    this project reports -- an argmax-invariant `apply` is dead code. Measured in
    `research/gpu_unknowns.md` 0.2. `update_state`, which is where dattn's whole effect
    lives, runs for every processor either way.
    """
    p = proc.DattnLogitsProcessor()
    sentinel = object()
    assert p.is_argmax_invariant() is False
    assert p.apply(sentinel) is sentinel


def test_the_measured_fsm_lag_is_one_in_both_scheduling_modes() -> None:
    """gpu_unknowns 1: the design's "2 under async" prediction is refuted on MRV1."""
    from types import SimpleNamespace

    import torch

    for async_scheduling in (False, True):
        cfg = SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=4, async_scheduling=async_scheduling))
        p = proc.DattnLogitsProcessor(cfg, torch.device("cpu"), False)
        assert p.fsm_lag == 1
        p.state.__init__()  # type: ignore[misc]


def test_it_really_subclasses_vllms_abc() -> None:
    """`_load_logitsprocs_by_fqcns` refuses anything that is not a subclass."""
    vllm_lp = pytest.importorskip("vllm.v1.sample.logits_processor").LogitsProcessor
    assert issubclass(proc.DattnLogitsProcessor, vllm_lp)


def test_add_then_step_writes_the_mask_on_the_pivot_token(lp, layout) -> None:
    out: list[int] = []
    lp.update_state(FakeBatchUpdate(1, added=[added(0, layout, out)]))
    assert not bool(lp.state._row_masked[0]), "a request with no output tokens yet is dense"

    out.append(5)
    lp.update_state(None)
    assert not bool(lp.state._row_masked[0])

    out.append(StubFSM.PIVOT)
    lp.update_state(None)
    assert bool(lp.state._row_masked[0])
    assert lp.state._intervals_host[0, :2].tolist() == [[16, 16], [480, INF]] or lp.state._intervals_host[
        0, :2
    ].tolist() == [[0, 16], [480, INF]]
    assert lp.state.rows[0].mode == int(Mode.FOCUS)


def test_the_mask_written_now_is_read_by_this_steps_build(lp, layout) -> None:
    """There is no lag in the transport: update_state runs before _build_attention_metadata."""
    out = [StubFSM.PIVOT]
    lp.update_state(FakeBatchUpdate(1, added=[added(0, layout, out)]))
    assert bool(lp.state._row_masked[0])
    assert lp.state._dirty is False, "flush() ran at the end of update_state"
    assert torch.equal(lp.state._intervals_gpu, lp.state._intervals_host)


def test_a_negative_token_id_stalls_and_resumes(lp, layout) -> None:
    out = [1, -1]
    lp.update_state(FakeBatchUpdate(1, added=[added(0, layout, out)]))
    req = lp.state.rows[0]
    assert req.n_fed == 1, "the FSM stops at the first placeholder"
    assert lp.n_neg_stalls == 1

    out[1] = StubFSM.PIVOT
    lp.update_state(None)
    assert req.n_fed == 2
    assert bool(lp.state._row_masked[0])


def test_a_chunked_prefill_is_not_counted_as_a_failed_decode(lp, layout) -> None:
    """The 250K-token shape: tens of prefill refreshes, no token, THEN decode.

    `update_state` runs on every engine step whether the batch changed or not, and a
    request joins `InputBatch` at its first scheduled CHUNK rather than at its first
    sampled token, so a long prompt reaches `_advance` with an empty output list once
    per prefill iteration. At `max_num_batched_tokens = 8192` a 250K-token prompt is
    about 31 of them. The watchdog used to read the fourth as the FSM freezing and
    raise `DattnInvariantError`, which kills EngineCore.
    """
    out: list[int] = []
    lp.update_state(FakeBatchUpdate(1, added=[added(0, layout, out)]))
    for _ in range(40):  # 40 prefill iterations, none of them a decode step
        lp.update_state(None)
    req = lp.state.rows[0]
    assert req.n_updates == 41
    assert req.n_decode_steps == 0, "a prefill iteration is not a decode step"
    assert req.n_fed == 0
    out.append(StubFSM.PIVOT)  # prefill done, the first token lands
    lp.update_state(None)
    assert req.n_fed == 1
    assert bool(lp.state._row_masked[0])


def test_a_row_that_never_sees_output_ids_crashes_rather_than_freezing(lp, layout) -> None:
    """The watchdog still fires, on DECODE steps. A decode step samples a token."""
    lp.update_state(FakeBatchUpdate(1, added=[added(0, layout, [])]))
    req = lp.state.rows[0]
    for _ in range(proc._NO_TOKEN_GRACE):
        req.n_decode_steps += 1  # what DattnState._gate_for does on a real step
        lp.update_state(None)
    req.n_decode_steps += 1
    with pytest.raises(DattnInvariantError, match="genuine decode"):
        lp.update_state(None)


def test_removed_added_moved_is_the_order(lp, layout) -> None:
    """A slot may be removed and re-added in the same update, and then moved."""
    a, b = [], []
    lp.update_state(FakeBatchUpdate(2, added=[added(0, layout, a, req_id="a"), added(1, layout, b, req_id="b")]))
    lp.update_state(FakeBatchUpdate(1, removed=[0], moved=[(1, 0, Uni())]))
    assert lp.state.rows[0].req_id == "b"
    assert lp.state.rows[1] is None


def test_a_swap_exchanges_both_rows(lp, layout) -> None:
    a, b = [StubFSM.PIVOT], []
    lp.update_state(FakeBatchUpdate(2, added=[added(0, layout, a, req_id="a"), added(1, layout, b, req_id="b")]))
    assert bool(lp.state._row_masked[0]) and not bool(lp.state._row_masked[1])
    lp.update_state(FakeBatchUpdate(2, moved=[(0, 1, Swap())]))
    assert lp.state.rows[0].req_id == "b" and lp.state.rows[1].req_id == "a"
    assert bool(lp.state._row_masked[1]) and not bool(lp.state._row_masked[0])


def test_a_request_with_no_dattn_payload_gets_a_dense_row_it_still_owns(lp, layout) -> None:
    """The vanilla arm runs on the same engine, so its rows must exist and be dense.

    A row left empty would keep the previous occupant's mask, and it would knock Lane C's
    per-row arrays out of slot alignment with the compaction's host mirror.
    """
    lp.update_state(FakeBatchUpdate(1, added=[added(0, layout, [StubFSM.PIVOT])]))
    assert bool(lp.state._row_masked[0])
    out: list[int] = []
    lp.update_state(FakeBatchUpdate(1, added=[(0, FakeSamplingParams(), [0] * 64, out)]))
    req = lp.state.rows[0]
    assert req is not None and req.request is None and req.fsm is None
    assert not bool(lp.state._row_masked[0])
    out.extend([1, 2, 3])
    assert req.seq_len() == 67
    lp.update_state(None)
    assert not bool(lp.state._row_masked[0])


def test_a_forced_mask_never_calls_the_fsm(lp, layout) -> None:
    mask = Mask.of((0, 16), (480, INF))
    out = [StubFSM.PIVOT, StubFSM.PIVOT]
    lp.update_state(FakeBatchUpdate(1, added=[added(0, layout, out, mask=mask)]))
    req = lp.state.rows[0]
    assert req.fsm is None
    assert req.mask == mask
    lp.update_state(None)
    assert lp.state._intervals_host[0, :2].tolist() == [[0, 16], [480, INF]]


def test_a_disabled_request_stays_dense(lp, layout) -> None:
    out = [StubFSM.PIVOT]
    lp.update_state(FakeBatchUpdate(1, added=[added(0, layout, out, enabled=False)]))
    assert lp.state.rows[0].fsm is None
    assert not bool(lp.state._row_masked[0])


def test_a_layout_that_does_not_match_the_prompt_is_refused(lp, layout) -> None:
    entry = (0, params(layout), [0] * (layout.n_prompt_tokens + 1), [])
    with pytest.raises(DattnInvariantError, match="n_prompt_tokens"):
        lp.update_state(FakeBatchUpdate(1, added=[entry]))


def test_preemption_resumes_the_fsm_rather_than_replaying(lp, layout) -> None:
    out = [1, 2, StubFSM.PIVOT]
    lp.update_state(FakeBatchUpdate(1, added=[added(0, layout, out, req_id="r")]))
    fsm = lp.state.rows[0].fsm
    assert fsm.seen == [1, 2, StubFSM.PIVOT]
    lp.update_state(FakeBatchUpdate(0, removed=[0]))
    out.append(7)
    lp.update_state(FakeBatchUpdate(1, added=[added(3, layout, out, req_id="r")]))
    assert lp.state.rows[3].fsm is fsm
    assert fsm.seen == [1, 2, StubFSM.PIVOT, 7], "a resumed request must not re-feed old tokens"
    assert bool(lp.state._row_masked[3])


def test_the_lp_rows_carry_seq_len_in_slot_two(lp, layout) -> None:
    """Lane C interface request C2: the third element is the vanilla KV length.

    `att_tok` is `Mask.area(seq_len)`, so a row whose mask is dense reports its whole
    context and a focused row reports strictly less. The recorder's own schema check is
    `att_tok <= seq_len`, which the batch slot index would have violated at once.
    """

    class Rec:
        def __init__(self) -> None:
            self.rows = None
            self.compactions = 0

        def note_lp(self, host_us, rows):
            self.rows = list(rows)

        def note_compaction(self, *a, **k):
            self.compactions += 1

    from dattn.engine import state as state_mod

    rec = Rec()
    state_mod.set_recorder(rec)
    try:
        out = [StubFSM.PIVOT]
        lp.update_state(FakeBatchUpdate(1, added=[added(0, layout, out, req_id="r")]))
    finally:
        state_mod.set_recorder(None)

    assert rec.rows is not None and len(rec.rows) == 1
    req_id, mode, seq_len, att_tok, out_pos, since_adm = rec.rows[0]
    assert req_id == "r"
    assert seq_len == layout.n_prompt_tokens + 1
    assert att_tok < seq_len, "a focused mask must attend strictly less than the whole row"
    assert out_pos == 1
    assert mode == int(Mode.FOCUS)
    assert since_adm == 0


# ---------------------------------------------------------------------------------------
# The token decoder (review major 16)
# ---------------------------------------------------------------------------------------


def test_the_token_decoder_strips_special_tokens() -> None:
    """dac's convention (`src/dac/dac_state.py:715`, `:540-547`): a special token is "".

    Keeping them let the stop token, which lands in `output_token_ids` on essentially every
    request, split a `<focus ...>` tag across the scan window.
    """
    tok = RecordingTokenizer()
    assert proc.SKIP_SPECIAL_TOKENS is True
    assert proc._TokenText(tok)(7) == ""
    assert tok.calls == [((7,), True)]


def test_the_decoder_memoises_per_token_id() -> None:
    tok = RecordingTokenizer()
    decode = proc._TokenText(tok)
    assert decode(7) == decode(7) == ""
    assert len(tok.calls) == 1
    decode(8)
    assert [ids for ids, _ in tok.calls] == [(7,), (8,)]


def test_a_decoder_with_no_tokenizer_is_inert() -> None:
    assert proc._TokenText(None)(7) == ""
