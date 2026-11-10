"""The explicit configurations retain backend selection and the Qwen planner distinction."""

from dataclasses import replace

import pytest

from dattn.contracts import DattnError
from dattn.engine.launch import apply_env, builder_free_kwargs, engine_kwargs
from dattn.engine.profile import load_profile


@pytest.mark.parametrize("model", ["google/gemma-4-31B-it", "Qwen/Qwen3.6-27B"])
@pytest.mark.parametrize("use", ["parity", "walltime"])
def test_explicit_profiles(model, use):
    p = load_profile(model, use=use)
    if use == "walltime":
        assert p.engine_args["kv_cache_memory_bytes"] == "TO BE PINNED FROM PROBE"
        with pytest.raises(DattnError, match="TO BE PINNED"):
            engine_kwargs(p)
        p = replace(p, engine_args=dict(p.engine_args, kv_cache_memory_bytes=100 << 30))
        assert p.engine_args["gpu_memory_utilization"] == 0.92
        assert p.bench["max_tokens"] == 512 and p.bench["repetitions"] == 8
    da = engine_kwargs(p)
    vanilla = builder_free_kwargs(da)
    assert da["attention_backend"] == vanilla["attention_backend"] == p.base_backend
    assert da["max_num_batched_tokens"] == 16384
    assert da["enable_prefix_caching"] is (use == "walltime")
    assert bool(p.needs_compacted_seq_lens_cpu) == ("Qwen" in model and use == "walltime")
    assert "logits_processors" not in vanilla


@pytest.mark.parametrize("plugins", ["", "unrelated_plugin"])
def test_plugin_allowlist_refuses_missing_dattn(plugins):
    profile = load_profile("google/gemma-4-31B-it")
    with pytest.raises(DattnError, match="VLLM_PLUGINS"):
        apply_env(profile, {"VLLM_PLUGINS": plugins})
