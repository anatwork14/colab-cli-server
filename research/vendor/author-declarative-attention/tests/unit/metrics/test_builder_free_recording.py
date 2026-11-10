"""Native input-batch bookkeeping records vanilla without installing the DA LP."""

from types import SimpleNamespace as NS

import pytest
from fakes import FakeRunner, FakeTorch, decode_step

from dattn.metrics import hooks


@pytest.mark.parametrize("da", [False, True])
def test_shared_clock_and_dense_rows_survive_without_a_da_processor(monkeypatch, da):
    calls = []
    recorder = None

    class Batch:
        num_reqs = 2
        req_ids = ["a", "b"]
        num_tokens_no_spec = [101, 202]
        num_prompt_tokens = [100, 200]
        logitsprocs = NS(all=[type("DattnLogitsProcessor", (), {})()] if da else [])

        def refresh_metadata(self):
            calls.append("refresh")
            if da:
                recorder.note_lp(3.0, [("a", 1, 101, 64, 1, 0), ("b", 0, 202, 202, 2, 0)])

    runner = FakeRunner(decode_step(2))
    runner.input_batch = Batch()
    spec = type("FullAttentionSpec", (), {})()
    runner.kv_cache_config = NS(kv_cache_groups=[NS(kv_cache_spec=spec)])
    runner.attn_groups = [[NS(get_metadata_builder=lambda: NS(kv_cache_spec=NS(block_size=32)))]]
    monkeypatch.setattr(hooks, "_read_forward_context", lambda graph: None)
    recorder = hooks.install(runner, window_id="w", torch_mod=FakeTorch, ring=8)
    for _ in range(2):
        runner.execute_model(decode_step(2))
        runner.input_batch.refresh_metadata()
        runner._model_forward()
        runner._sample()
        runner.input_batch.num_tokens_no_spec = [102, 203]
    records = recorder.drain()
    assert [r.engine_step for r in records] == [1, 2]
    assert 0 < records[0].t_sched_ns < records[0].t_ready_ns < records[1].t_sched_ns
    assert [r.dattn_block_size for r in records] == [32, 32]
    assert records[0].rows.seq_len == [101, 202]
    if da:
        assert records[0].rows.att_tok == [64, 202]
        assert records[0].lp_host_us == 3.0
    else:
        assert records[1].rows.seq_len == records[1].rows.att_tok == [102, 203]
        assert records[1].rows.out_pos == [2, 3]
        assert records[1].rows.since_adm == [1, 1]
        assert all(r.rows.att_blk == [] and r.lp_host_us == 0 and r.n_compacted_groups == 0 for r in records)
    hooks.uninstall(runner)
    assert "refresh_metadata" not in vars(runner.input_batch)
    assert len(calls) == 2
