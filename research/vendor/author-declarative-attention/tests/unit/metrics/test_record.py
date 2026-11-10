"""The record schema, the classifier and the JSONL codec (WP6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import FakeCached, FakeNewReq, FakeSchedulerOutput, decode_step

from dattn.contracts import StepRecord, StepRows
from dattn.metrics import record


def make(step_id: int = 0, n_rows: int = 3) -> StepRecord:
    return StepRecord(
        step_id=step_id,
        engine_step=step_id + 7,
        t_sched_ns=1_000 * step_id,
        t_ready_ns=1_000 * step_id + 30_000_000,
        gpu_forward_ms=29.5,
        gpu_sample_ms=0.4,
        lp_host_us=97.3,
        compact_host_us=12.5,
        kind="decode",
        num_reqs=n_rows,
        num_reqs_padded=n_rows,
        cudagraph_replayed=True,
        cudagraph_mode="FULL",
        runner_generation="V1",
        dattn_block_size=32,
        num_gen_tokens=n_rows,
        kv_usage=0.7981,
        n_full_attn_groups=1,
        n_compacted_groups=1,
        rows=StepRows(
            req=[f"r{i}" for i in range(n_rows)],
            mode=[i % 3 for i in range(n_rows)],
            seq_len=[16384 + i for i in range(n_rows)],
            att_blk=[4096 + i for i in range(n_rows)],
            att_tok=[4090 + i for i in range(n_rows)],
            out_pos=[100 + i for i in range(n_rows)],
            since_adm=[100 + i for i in range(n_rows)],
        ),
    )


# -- the classifier ---------------------------------------------------------------------


def test_a_pure_decode_step_is_decode() -> None:
    assert record.classify_step(decode_step(4), 4) == "decode"


def test_a_new_request_makes_the_step_mixed() -> None:
    so = decode_step(3)
    so.scheduled_new_reqs = (FakeNewReq("r9"),)
    so.total_num_scheduled_tokens = 3 + 8192
    assert record.classify_step(so, 4) == "mixed"


def test_a_chunked_prefill_final_chunk_of_exactly_one_token_is_not_decode() -> None:
    """The acceptance case of WP6: token count says decode, the phase predicate says not."""
    so = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCached(req_ids=("r0",), context_phase=frozenset({"r0"})),
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"r0": 1},
    )
    assert record.classify_step(so, 1) == "prefill"


def test_a_mid_stream_chunked_prefill_next_to_decodes_is_mixed() -> None:
    so = FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCached(req_ids=("r0", "r1"), context_phase=frozenset({"r1"})),
        total_num_scheduled_tokens=1 + 4096,
        num_scheduled_tokens={"r0": 1, "r1": 4096},
    )
    assert record.classify_step(so, 2) == "mixed"


def test_the_classifier_refuses_to_guess_at_a_foreign_object() -> None:
    with pytest.raises(NotImplementedError, match="classify_step needs"):
        record.classify_step(None, 1)


# -- JSONL ------------------------------------------------------------------------------


def test_jsonl_round_trips_every_field(tmp_path: Path) -> None:
    recs = [make(i) for i in range(5)]
    p = tmp_path / "steps_c0.jsonl"
    assert record.write_jsonl(p, recs) == 5
    assert list(record.read_jsonl(p)) == recs


def test_the_writer_buffers_and_only_the_flush_touches_disk(tmp_path: Path) -> None:
    p = tmp_path / "steps_c1.jsonl"
    w = record.JsonlWriter(p, flush_lines=4)
    for i in range(3):
        w.write(make(i))
    assert p.read_text() == "", "three lines under a buffer of four must still be in memory"
    w.write(make(3))
    assert len(p.read_text().splitlines()) == 4
    w.close()


def test_the_default_buffer_is_the_512_lines_the_design_states() -> None:
    assert record.FLUSH_LINES == 512


def test_appending_never_truncates(tmp_path: Path) -> None:
    p = tmp_path / "steps_c2.jsonl"
    record.write_jsonl(p, [make(0)])
    record.write_jsonl(p, [make(1)])
    assert [r.step_id for r in record.read_jsonl(p)] == [0, 1]


# -- table adapters ---------------------------------------------------------------------


def test_write_table_dispatches_on_the_suffix(tmp_path: Path) -> None:
    recs = [make(0)]
    assert record.write_table(tmp_path / "a.jsonl", recs) == 1
    assert record.read_table(tmp_path / "a.jsonl") == recs
    with pytest.raises(ValueError, match="unknown step-table suffix"):
        record.write_table(tmp_path / "a.csv", recs)


# -- the schema itself --------------------------------------------------------------------


def test_the_dtype_tables_cover_the_contract_exactly() -> None:
    assert tuple(record.DTYPES) == tuple(f for f in StepRecord.__slots__ if f != "rows")
    assert tuple(record.ROW_DTYPES) == StepRows.__slots__


def test_the_sidecars_are_named_but_never_merged() -> None:
    assert record.ENV_FILE.format(cell_id="c") == "env_c.jsonl"
    assert record.META_FILE.format(cell_id="c") == "meta_c.json"
    assert record.STEP_FILE.format(cell_id="c") == "steps_c.jsonl"
