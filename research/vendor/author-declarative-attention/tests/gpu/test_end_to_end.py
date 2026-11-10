"""A short public prompt-to-generation-to-recorder-and-replay check."""

import pytest
from conftest import GPU_MEM_UTIL, MODELS, get_engine

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("model_id", MODELS)
def test_prompt_generation_record_and_replay(require_cuda, monkeypatch, model_id, tmp_path):
    from vllm import SamplingParams, TokensPrompt

    from dattn.contracts import DattnRequest
    from dattn.engine import launch
    from dattn.engine.profile import load_profile
    from dattn.eval.replay import replay_record
    from dattn.metrics.record import read_table, write_table
    from dattn.prompt.render import render

    profile = load_profile(model_id)
    launch.apply_env(profile)
    monkeypatch.setenv("DATTN_ARM", "da")
    monkeypatch.setenv("DATTN_RECORD", "1")
    llm = get_engine(
        f"{model_id}/end-to-end",
        **launch.engine_kwargs(profile, max_model_len=4096, max_num_seqs=1, gpu_memory_utilization=GPU_MEM_UTIL),
    )
    tokenizer = llm.get_tokenizer()
    prompt = render(
        tokenizer,
        context="The archive access code is cobalt. " * 20,
        question="What is the archive access code?",
        fmt="chunk_multi_turn_tool",
    )
    armed = llm.collective_rpc("dattn_record", args=(True,), kwargs={"window_id": "end-to-end"})[0]
    assert armed["armed"]
    output = llm.generate(
        [TokensPrompt(prompt_token_ids=list(prompt.ids))],
        SamplingParams(
            temperature=0,
            max_tokens=32,
            min_tokens=8,
            ignore_eos=True,
            extra_args={"dattn": DattnRequest(layout=prompt.layout).to_extra_args()},
        ),
    )[0].outputs[0]
    llm.collective_rpc("dattn_record", args=(False,))
    records = llm.collective_rpc("dattn_drain")[0]
    audit = llm.collective_rpc("dattn_audit")[0]["recorder"]
    assert output.token_ids and records
    assert audit["checked"] > 0 and audit["mismatches"] == 0
    assert all(r.t_sched_ns > 0 for r in records)
    trace = replay_record(output.text, prompt.layout, tokenizer)
    assert trace["dac_metrics"]["total_steps"] == len(trace["mode_trace"])
    path = tmp_path / "steps.jsonl"
    write_table(path, records)
    assert [r.to_dict() for r in read_table(path)] == [r.to_dict() for r in records]
