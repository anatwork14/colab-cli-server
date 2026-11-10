"""The vanilla boot checks live bindings and the backend selected by vLLM."""

from types import SimpleNamespace as NS

import pytest

from dattn.engine.assertions import _a02_a07, _a14, _Ctx, resolved_config


@pytest.mark.parametrize(
    ("backend", "builder", "fails"),
    [("TRITON_ATTN", False, False), ("FLASHINFER", False, True), ("TRITON_ATTN", True, True)],
)
def test_vanilla_binding_and_resolved_backend(backend, builder, fails):
    cfg = NS(attention_config=NS(backend="TRITON_ATTN"))
    ctx = _Ctx(cfg, None, NS(base_backend="TRITON_ATTN"), arm="vanilla")
    groups = [{"kv_cache_group_id": 0, "kind": "full_attention", "base_backend": backend, "is_dattn": builder}]
    _a02_a07(ctx, groups)
    assert bool(ctx.report.failures) is fails
    assert ctx.report.resolved["dattn_builder_installed"] is builder


def test_vanilla_refuses_a_live_lp_even_when_the_config_omits_it():
    lp = type("DattnLogitsProcessor", (), {})()
    runner = NS(input_batch=NS(logitsprocs=NS(all=[lp])))
    ctx = _Ctx(NS(model_config=NS(logits_processors=[])), runner, None, arm="vanilla")
    ctx.current = "A14"
    _a14(ctx)
    assert ctx.report.failures == ["A14: vanilla must not install DattnLogitsProcessor"]
    assert ctx.report.resolved["dattn_lp_installed"] is True


def test_snapshot_uses_resolved_budget_and_pool_not_requested_values():
    cfg = NS(scheduler_config=NS(max_num_batched_tokens=8192), cache_config=NS(num_gpu_blocks=42))
    profile = NS(engine_args={"max_num_batched_tokens": 16384})
    actual = resolved_config(cfg, None, profile, [{"kernel_block_size": 32}])
    assert actual["engine_args"]["max_num_batched_tokens"] == 8192
    assert actual["max_num_batched_tokens"] == 8192
    assert actual["num_gpu_blocks"] == 42
    assert actual["block_sizes"] == [32]


@pytest.mark.parametrize("supports_update", [False, True])
def test_da_boot_a03_refuses_block_table_update_fast_path(supports_update):
    ctx = _Ctx(NS(), None, None, arm="da")
    groups = [
        {
            "kv_cache_group_id": 3,
            "kind": "full_attention",
            "is_dattn": True,
            "builder_cls": "DattnBuilder",
            "kernel_block_size": 32,
            "sliding_window": None,
            "supports_update_block_table": supports_update,
        }
    ]
    _a02_a07(ctx, groups)
    assert ctx.report.failures == (
        ["A03: group 3/DattnBuilder supports_update_block_table=True can bypass DA scratch staging"]
        if supports_update
        else []
    )
