"""One boot, one warm pass and eight sampled decodes of a persisted draw."""

import json
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
from fakes import FakeEngine as LLM
from fakes import FakeScheduler as Scheduler

from dattn.bench import driver
from dattn.bench.protocol import Protocol
from dattn.contracts import PromptLayout
from dattn.engine.profile import load_profile


def setup(monkeypatch, tmp_path):
    import vllm

    monkeypatch.setattr(vllm, "LLM", LLM)
    monkeypatch.setattr(driver, "load_tasks", lambda *a, **kw: {"t": [NS(sample_id="a"), NS(sample_id="b")]})

    def prompts(*a, **kw):
        p = {
            fmt: [{"prompt_token_ids": [1] * 64}, {"prompt_token_ids": [2] * 96}] for fmt in driver.ARM_FORMAT.values()
        }
        layouts = [PromptLayout(n_prompt_tokens=128, question_start=100, segments=((16, 90),))] * 2
        return p, layouts, {fmt: [64, 96] for fmt in p}

    monkeypatch.setattr(driver, "build_prompts", prompts)
    monkeypatch.setenv("DATTN_POOL_RECORD", "")
    LLM.calls = []
    monkeypatch.setattr(LLM, "instances", 0)
    monkeypatch.setattr(LLM, "warm_leftover", 0.0)
    profile = load_profile("google/gemma-4-31B-it", use="walltime")
    profile = replace(
        profile, engine_args=dict(profile.engine_args, enable_prefix_caching=True, kv_cache_memory_bytes=9250)
    )
    (tmp_path / "pins").mkdir(exist_ok=True)
    (tmp_path / "pins/pool.json").write_text("{}")
    return Protocol(model_id=profile.model_id), profile


def test_eight_reps_one_boot_same_draw_cap_and_seed(monkeypatch, tmp_path):
    p, profile = setup(monkeypatch, tmp_path)
    da = driver.run_protocol(p, tmp_path / "da", profile=profile, pins=tmp_path / "pins")
    vanilla = driver.run_protocol(p, tmp_path / "vanilla", profile=profile, pins=tmp_path / "pins", arm="vanilla")
    assert LLM.instances == 2 and len(LLM.calls) == 18
    assert len(da["cells"]) == len(vanilla["cells"]) == 8
    assert len({d["meta"]["boot_id"]["started"] for d in da["cells"]}) == 1
    for rep, (a, b) in enumerate(zip(da["cells"], vanilla["cells"], strict=True)):
        assert a["identity"] == b["identity"]
        assert a["meta"]["n_steps"] == 19 and b["meta"]["n_steps"] == 511
        assert a["meta"]["pre_rep_hygiene"]["ok"] and b["meta"]["pre_rep_hygiene"]["ok"]
        assert a["meta"]["requests"][0]["seed"] == b["meta"]["requests"][0]["seed"]
        da_params, vanilla_params = LLM.calls[rep + 1][2], LLM.calls[rep + 10][2]
        assert da_params.max_tokens == vanilla_params.max_tokens == 512
        assert not da_params.ignore_eos and da_params.min_tokens == 0
        assert vanilla_params.ignore_eos and vanilla_params.min_tokens == 512
    assert len({d["meta"]["requests"][0]["seed"] for d in da["cells"]}) == 8


@pytest.mark.parametrize("arm", ["vanilla", "da"])
@pytest.mark.parametrize("alignment", [16, 128])
def test_hybrid_primes_precede_warm_pass_only_for_aligned_prompts(monkeypatch, tmp_path, arm, alignment):
    p, profile = setup(monkeypatch, tmp_path)
    original_rpc, original_generate = LLM.collective_rpc, LLM.generate
    calls = []

    def rpc(engine, name, *args, **kwargs):
        result = original_rpc(engine, name, *args, **kwargs)
        if name == "dattn_probe":
            result[0]["kv_pool"]["groups"].append({"kind": "mamba", "block_size": 16})
            engine.sched.kv_cache_manager.coordinator._cache_hit_alignment_tokens = alignment
        return result

    def generate(engine, prompts, params):
        calls.append((prompts, params, engine.recording))
        return original_generate(engine, prompts, params)

    monkeypatch.setattr(LLM, "collective_rpc", rpc)
    monkeypatch.setattr(LLM, "generate", generate)
    result = driver.run_protocol(p, tmp_path, profile=profile, arm=arm)
    warm = json.loads((tmp_path / f"t/b0/{arm}/warmup.json").read_text())["warmup"]
    primes = warm["primed_requests"]
    if alignment == 128:
        assert primes == []
        # Exactly the original warm generation followed by eight timed calls.
        assert len(calls) == 9
    else:
        indices = result["cells"][0]["identity"]["indices"]
        assert primes == [
            {"request_index": j, "prompt_len": [64, 96][i], "prime_len": [49, 81][i]} for j, i in enumerate(indices)
        ]
        assert len(calls) == 11
        for j, i in enumerate(indices):
            prompts, params, recording = calls[j]
            assert prompts == [{"prompt_token_ids": [i + 1] * primes[j]["prime_len"]}]
            assert params[0].max_tokens == params[0].min_tokens == 1 and params[0].ignore_eos
            assert params[0].extra_args is None and not recording
            assert params[0].seed == driver.request_seed(result["cells"][0]["identity"]["seed"], j, -2)
    warm_call, *timed_calls = calls[len(primes) :]
    assert warm_call[1][0].max_tokens == 64 and not warm_call[2]
    assert len(timed_calls) == 8
    for rep, (prompts, params, recording) in enumerate(timed_calls):
        assert prompts == warm_call[0] and recording
        assert params[0].max_tokens == 512
        assert params[0].seed == result["cells"][rep]["meta"]["requests"][0]["seed"]
        assert not result["cells"][rep]["meta"]["cache_miss"]


@pytest.mark.parametrize(
    "caching, kind, alignment", [(False, "mamba", 16), (True, "sliding_window", 16), (True, "mamba", None)]
)
def test_priming_skips_uncached_non_mamba_and_unknown_alignment(monkeypatch, caching, kind, alignment):
    engine = LLM()
    adapter = driver.VllmEngineAdapter(engine, prompts=[{"prompt_token_ids": [1] * 64}], engine_kwargs={})
    batch = driver.Batch("t", 0, (0,), (64,), 0.8, 0)
    monkeypatch.setattr(adapter, "hygiene", lambda: {"cache_alignment": alignment})
    monkeypatch.setattr(engine, "generate", lambda *args: pytest.fail("unexpected engine call"))
    meta = {"enable_prefix_caching": caching, "kv_pool": {"groups": [{"kind": kind}]}}
    assert driver.prime_aligned_prompts(adapter, batch, "vanilla", meta) == []


def test_priming_refuses_dirty_pre_warm_state(monkeypatch):
    engine = LLM()
    adapter = driver.VllmEngineAdapter(engine, prompts=[{"prompt_token_ids": [1] * 64}], engine_kwargs={})
    batch = driver.Batch("t", 0, (0,), (64,), 0.8, 0)
    engine.sched.kv_cache_manager.usage = 0.1
    monkeypatch.setattr(engine, "generate", lambda *args: pytest.fail("prime before hygiene"))
    meta = {"enable_prefix_caching": True, "kv_pool": {"groups": [{"kind": "mamba"}]}}
    with pytest.raises(ValueError, match="pre-warm hygiene"):
        driver.prime_aligned_prompts(adapter, batch, "vanilla", meta)


def test_warmup_must_leave_scheduler_idle_and_pool_empty(monkeypatch, tmp_path):
    p, profile = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(LLM, "warm_leftover", 0.1)
    with pytest.raises(ValueError, match="warmup"):
        driver.run_protocol(p, tmp_path, profile=profile)
    assert (tmp_path / "t/b0/da/warmup.json").exists()


def test_hygiene_is_checked_again_before_every_rep(monkeypatch, tmp_path):
    p, profile = setup(monkeypatch, tmp_path)
    original = driver.hygiene_witness
    calls = []

    def witness(*args):
        result = original(*args)
        calls.append(result)
        if len(calls) == 4:
            result.update(running=1, ok=False)
        return result

    monkeypatch.setattr(driver, "hygiene_witness", witness)
    with pytest.raises(ValueError, match="pre-rep hygiene"):
        driver.run_protocol(p, tmp_path, profile=profile)
    failed = json.loads((tmp_path / "t/b0/da/r2/arm.json").read_text())
    assert failed["faults"] and failed["meta"]["pre_rep_hygiene"]["running"] == 1
    assert len(LLM.calls) == 3


def test_cache_witness_is_informational_but_request_hits_are_a_gate(monkeypatch, tmp_path):
    p, profile = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(LLM, "cached_tokens", 0)
    with pytest.raises(ValueError, match="cache hit gate"):
        driver.run_protocol(p, tmp_path, profile=profile)
    failed = json.loads((tmp_path / "t/b0/da/r0/arm.json").read_text())
    assert failed["meta"]["requests"][0]["num_cached_tokens"] == 0
    assert "prefill_steps_before_rectangle" in failed["meta"] and failed["faults"]


def test_preemption_past_rectangle_still_refuses(monkeypatch, tmp_path):
    p, profile = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(LLM, "close_window_at", 10)
    # Warm pass cannot preempt either. Fail a measured step through the real wrapper.
    original = driver.run_cell

    def measured(*args, **kwargs):
        monkeypatch.setattr(Scheduler, "preempt_at", 15)
        return original(*args, **kwargs)

    monkeypatch.setattr(driver, "run_cell", measured)
    with pytest.raises(AssertionError, match="preempted batch"):
        driver.run_protocol(p, tmp_path, profile=profile)


def test_seed_replays_by_occurrence_and_rep():
    assert driver.request_seed(3, 1, 2) == driver.request_seed(3, 1, 2)
    assert len({driver.request_seed(3, 1, r) for r in range(8)}) == 8
    assert driver.request_seed(3, 1, 0) != driver.request_seed(3, 2, 0)


def test_walltime_rejects_another_cap_before_boot(monkeypatch, tmp_path):
    p, profile = setup(monkeypatch, tmp_path)
    with pytest.raises(ValueError, match="decode cap"):
        driver.run_protocol(p.with_(max_tokens=2048), tmp_path, profile=profile)
    assert LLM.instances == 0


def test_run_arm_wrapper_passes_schedule_arguments_through():
    """The observer installed by run_arm must forward EngineCore's throttle argument."""

    class ThrottlingEngine(LLM):
        def generate(self, prompts, params):
            self.sched.schedule(True)  # goes through the installed wrapper
            return super().generate(prompts, params)

    engine = ThrottlingEngine(attention_backend="FLASHINFER")
    adapter = driver.VllmEngineAdapter(
        engine,
        prompts=["p0", "p1"],
        engine_kwargs={"attention_backend": "FLASHINFER"},
        model_id="google/gemma-4-31B-it",
    )
    batch = driver.Batch(task="t", set_index=0, indices=(0, 1), prompt_lens=(4, 4), occupancy=0.5, seed=0)
    adapter.run_arm("vanilla", batch, max_tokens=3, ignore_eos=True, min_tokens=3, seed=0)
    assert engine.sched.received[0] is True
    assert engine.sched.received[1:] == [False, False, False]
    assert "schedule" not in vars(engine.sched)  # the wrapper is removed afterwards


def test_aggregate_cache_count_does_not_override_per_request_hits(monkeypatch, tmp_path):
    p, profile = setup(monkeypatch, tmp_path)
    original = driver.VllmEngineAdapter.hygiene

    def hygiene(adapter):
        return original(adapter) | {"cached_blocks": 0, "cached_hashes": 0}

    monkeypatch.setattr(driver.VllmEngineAdapter, "hygiene", hygiene)
    result = driver.run_protocol(p, tmp_path, profile=profile)
    assert not result["cells"][0]["meta"]["pre_rep_hygiene"]["cache_coverage_witness"]
    assert all(r["cache_hit_ok"] for r in result["cells"][0]["meta"]["requests"])


def test_empty_rectangle_keeps_cache_evidence_and_refusal(monkeypatch, tmp_path):
    p, profile = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(LLM, "close_window_at", 0)
    with pytest.raises(ValueError, match="empty closed rectangle"):
        driver.run_protocol(p, tmp_path, profile=profile)
    failed = json.loads((tmp_path / "t/b0/da/r0/arm.json").read_text())
    assert failed["faults"] == ["empty closed rectangle"]
    assert len(failed["meta"]["requests"]) == 2
    assert failed["meta"]["n_steps"] == 0
    assert "prefill_steps_before_rectangle" in failed["meta"]


def test_s1_refuses_a_bad_prediction_and_persists_both_numbers(monkeypatch, tmp_path):
    p, profile = setup(monkeypatch, tmp_path)
    monkeypatch.setattr(LLM, "usage_by_step", {0: 0.9})
    with pytest.raises(ValueError, match="S1 sizing validation failed"):
        driver.run_protocol(p, tmp_path, profile=profile)
    validation = json.loads((tmp_path / "pins/sizing_validation.json").read_text())
    assert validation["gated"] and not validation["passed"]
    assert validation["snapshot"]["running"] == 2 and validation["measured"] == 0.9
    assert len(LLM.calls) == 1


def test_later_batches_record_snapshot_without_gating(monkeypatch, tmp_path):
    p, profile = setup(monkeypatch, tmp_path)
    driver.run_protocol(p, tmp_path, profile=profile)
    monkeypatch.setattr(LLM, "usage_by_step", {0: 0.9})
    result = driver.run_protocol(p, tmp_path, profile=profile, batch_index=1)
    check = result["cells"][0]["meta"]["sizing_validation"]
    assert not check["gated"] and not check["passed"]
    assert check["validated_at"]["batch_index"] == 0


def test_warm_snapshot_is_first_maximum_even_without_full_residency(monkeypatch):
    original = Scheduler.schedule

    def schedule(scheduler, *args):
        result = original(scheduler, *args)
        scheduler.running = scheduler.running[: [1, 2, 2, 1][scheduler.step]]
        for request in scheduler.running:
            request.num_computed_tokens = 1  # Prefill residency counts too.
        return result

    monkeypatch.setattr(Scheduler, "schedule", schedule)
    engine = LLM(attention_backend="FLASHINFER")
    adapter = driver.VllmEngineAdapter(
        engine, prompts=["abc"] * 3, model_id="google/gemma-4-31B-it", engine_kwargs={"attention_backend": "FLASHINFER"}
    )
    batch = driver.Batch(task="t", set_index=0, indices=(0, 1, 2), prompt_lens=(3, 3, 3), occupancy=0.5, seed=0)
    adapter.run_arm("vanilla", batch, max_tokens=4, min_tokens=4, ignore_eos=True, seed=0, observe_resident=True)
    snapshot = adapter.resident_snapshot
    assert snapshot["step_index"] == 1 and snapshot["n_running"] == 2 and snapshot["C"] == 3
    assert [r["resident_tokens"] for r in snapshot["requests"]] == [1, 1]
    assert snapshot["snapshot_kind"] == "max_running"


@pytest.mark.parametrize(
    "steps, expected_step, kind",
    [
        # Even one last prefill token is not decode. Prefer the next step,
        # after reclamation, and retain the first all-decode observation.
        ([([9, 10], [1, 1]), ([10, 11], [1, 1]), ([11, 12], [1, 1])], 1, "all_decode"),
        # The larger mixed batch supersedes a smaller all-decode snapshot.
        ([([10], [1]), ([11, 8], [1, 1]), ([12], [1])], 1, "max_running"),
        # Every running request must get its reclamation/allocation call.
        ([([10, 10], [1, 0]), ([11, 10], [1, 1])], 1, "all_decode"),
        # A smaller decode batch cannot replace the maximum-running fallback.
        ([([8, 8, 8], [2, 2, 2]), ([10, 10], [1, 1])], 0, "max_running"),
    ],
)
def test_warm_snapshot_prefers_first_all_decode_at_maximum(monkeypatch, steps, expected_step, kind):
    original = Scheduler.schedule

    def schedule(scheduler, *args):
        result = original(scheduler, *args)
        counts = steps[scheduler.step][1]
        result.num_scheduled_tokens = {str(i): n for i, n in enumerate(counts) if n}
        result.total_num_scheduled_tokens = sum(counts)
        # Match vLLM's optimistic post-schedule counters.
        for request, count in zip(scheduler.running, counts, strict=True):
            request.num_computed_tokens += count
            request.num_in_flight_tokens += count
        scheduler.kv_cache_manager.usage = 0.5 + scheduler.step / 100
        return result

    def generate(engine, *args):
        for i, (computed, _) in enumerate(steps):
            engine.sched.step = i
            engine.sched.running = [
                NS(request_id=str(j), num_computed_tokens=n, num_prompt_tokens=10, num_in_flight_tokens=0)
                for j, n in enumerate(computed)
            ]
            engine.sched.schedule()
        return []

    monkeypatch.setattr(Scheduler, "schedule", schedule)
    monkeypatch.setattr(LLM, "generate", generate)
    engine = LLM(attention_backend="FLASHINFER")
    adapter = driver.VllmEngineAdapter(
        engine, prompts=["0123456789"] * 3, model_id="google/gemma-4-31B-it", engine_kwargs={}
    )
    batch = driver.Batch(task="t", set_index=0, indices=(0, 1, 2), prompt_lens=(10, 10, 10), occupancy=0.5, seed=0)
    adapter.run_arm("vanilla", batch, max_tokens=64, min_tokens=64, ignore_eos=True, seed=0, observe_resident=True)
    snapshot = adapter.resident_snapshot
    assert snapshot["step_index"] == expected_step and snapshot["snapshot_kind"] == kind
    assert snapshot["kv_usage"] == 0.5 + expected_step / 100
    assert snapshot["n_running"] == max(len(computed) for computed, _ in steps)
    assert snapshot["C"] == 3


@pytest.mark.parametrize(
    "alignment, prompt_len, expected", [(16, 64, 48), (16, 65, 64), (784, 1568, 784), (784, 1569, 1568), (784, 700, 0)]
)
def test_resolved_alignment_and_last_token_witness(monkeypatch, tmp_path, alignment, prompt_len, expected):
    p, profile = setup(monkeypatch, tmp_path)
    original = driver.VllmEngineAdapter.hygiene

    def hygiene(adapter):
        adapter.scheduler.kv_cache_manager.coordinator._cache_hit_alignment_tokens = alignment
        return original(adapter)

    monkeypatch.setattr(driver.VllmEngineAdapter, "hygiene", hygiene)
    # Preserve the small sizing fixture, changing prompt lengths only for the
    # measured request evidence after the batch has been pinned.
    original_cell = driver.run_cell

    def run_cell(protocol, adapter, **kwargs):
        for prompts in adapter.prompts.values():
            for prompt in prompts:
                prompt["prompt_token_ids"] = [1] * prompt_len
        return original_cell(protocol, adapter, **kwargs)

    monkeypatch.setattr(driver, "run_cell", run_cell)
    monkeypatch.setattr(LLM, "cached_tokens", expected)
    result = driver.run_protocol(p, tmp_path, profile=profile)
    for cell in result["cells"]:
        meta = cell["meta"]
        assert meta["cache_alignment"] == alignment
        assert meta["cache_alignment_basis"] == "resolved_hybrid_lookup"
        assert meta["recompute_tokens"] == 2 * (prompt_len - expected)
        assert meta["prefill_steps_lower_bound"] == 1
        assert all(r["expected_aligned_hit"] == expected and r["deficit"] == 0 for r in meta["requests"])


def test_one_missing_cached_token_refuses_and_names_request(monkeypatch, tmp_path):
    p, profile = setup(monkeypatch, tmp_path)
    original = LLM.generate

    def generate(engine, *args):
        outputs = original(engine, *args)
        outputs[0].num_cached_tokens -= 1
        return outputs

    monkeypatch.setattr(LLM, "generate", generate)
    with pytest.raises(ValueError, match=r"request, deficit, cached.*'0', 1"):
        driver.run_protocol(p, tmp_path, profile=profile)
    doc = json.loads((tmp_path / "t/b0/da/r0/arm.json").read_text())
    assert doc["meta"]["requests"][0]["deficit"] == 1 and doc["faults"]


def test_more_than_two_prefill_steps_is_only_a_diagnostic(monkeypatch, tmp_path):
    p, profile = setup(monkeypatch, tmp_path)
    original = LLM.generate

    def generate(engine, *args):
        outputs = original(engine, *args)
        for record in engine.records[:3]:
            record.num_ctx_tokens = 32
        return outputs

    monkeypatch.setattr(LLM, "generate", generate)
    result = driver.run_protocol(p, tmp_path, profile=profile)
    meta = result["cells"][0]["meta"]
    assert meta["prefill_steps_before_rectangle"] == 3 and not meta["cache_miss"]
    assert meta["recompute_tokens"] == 32 and meta["prefill_steps_lower_bound"] == 1
