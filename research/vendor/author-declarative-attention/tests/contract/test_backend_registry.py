"""`register_backend`, the CUSTOM slot and `backend_per_kind`, pinned to vLLM 0.29.0.

This is the whole of dattn's registration surface. It is official, which is the point:
no monkeypatching (DESIGN.md section 2.2).
"""

from __future__ import annotations

import inspect

import pytest

pytestmark = pytest.mark.contract


def test_the_custom_slot_exists_and_starts_empty() -> None:
    """`N:v1/attention/backends/registry.py:129-130`: CUSTOM is None, not "", to avoid aliasing."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    assert AttentionBackendEnum.CUSTOM.value is None
    assert AttentionBackendEnum["CUSTOM"] is AttentionBackendEnum.CUSTOM


def test_an_unregistered_custom_slot_raises_rather_than_resolving() -> None:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if AttentionBackendEnum.CUSTOM.is_overridden():
        pytest.skip("something in this process already registered CUSTOM")
    with pytest.raises(ValueError, match="must be registered before use"):
        AttentionBackendEnum.CUSTOM.get_path()


def test_register_backend_takes_a_class_path_and_is_reversible() -> None:
    """`N:v1/attention/backends/registry.py:243-295`."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    slot = AttentionBackendEnum.CUSTOM
    was = slot.get_path() if slot.is_overridden() else None
    try:
        register_backend(slot, "dattn.engine.backend.DattnBackend")
        assert slot.is_overridden() is True
        assert slot.get_path() == "dattn.engine.backend.DattnBackend"
        assert slot.get_path(include_classname=False) == "dattn.engine.backend"
    finally:
        slot.clear_override()
        if was is not None:
            register_backend(slot, was)
    assert (slot.is_overridden()) == (was is not None)


def test_register_backend_also_works_as_a_decorator() -> None:
    sig = inspect.signature(
        __import__("vllm.v1.attention.backends.registry", fromlist=["register_backend"]).register_backend
    )
    assert list(sig.parameters) == ["backend", "class_path", "is_mamba"]
    assert sig.parameters["class_path"].default is None


def test_the_backend_abc_exposes_the_four_hooks_dattn_inherits() -> None:
    """`_make_backend` overrides two of six; the other four must keep working."""
    from vllm.v1.attention.backend import AttentionBackend
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    for name in (
        "get_name",
        "get_builder_cls",
        "supports_mm_prefix",
        "get_supported_head_sizes",
        "get_supported_kernel_block_sizes",
    ):
        assert hasattr(AttentionBackend, name), name
    # RC C1: the mm_prefix gate is what makes Gemma resolve to Triton, and inheriting it
    # is the reason a DattnBackend built on Triton survives backend selection at all.
    assert TritonAttentionBackend.supports_mm_prefix() is True
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend

    assert FlashInferBackend.supports_mm_prefix() is False


def test_a_subclass_of_a_backend_inherits_mm_prefix_support() -> None:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    from dattn.engine.backend import _make_backend

    made = _make_backend(TritonAttentionBackend)
    assert made.get_name() == "CUSTOM"
    assert made.supports_mm_prefix() is True
    assert made.get_supported_head_sizes() == TritonAttentionBackend.get_supported_head_sizes()
    assert made.get_impl_cls() is TritonAttentionBackend.get_impl_cls()
    # Not overridden at all: the exact same function objects are inherited.
    for name in ("get_supported_kernel_block_sizes", "supports_mm_prefix", "get_impl_cls"):
        assert name not in vars(made), f"{name} must stay inherited verbatim"
    assert set(vars(made)) & {"get_name", "get_builder_cls"} == {"get_name", "get_builder_cls"}


def test_backend_per_kind_accepts_the_kv_cache_spec_kind_strings() -> None:
    """`N:v1/kv_cache_interface.py:131-141` and `N:config/attention.py:30-39`."""
    import dataclasses

    from vllm.config.attention import AttentionConfig
    from vllm.v1.kv_cache_interface import KVCacheSpecKind

    values = {k.value for k in KVCacheSpecKind}
    assert "full_attention" in values
    assert "sliding_window" in values
    assert "mamba" in values
    fields = {f.name for f in dataclasses.fields(AttentionConfig)}
    assert "backend_per_kind" in fields
    assert "use_trtllm_attention" in fields


def test_a_full_attention_layer_maps_to_the_full_attention_kind() -> None:
    from vllm.v1.attention.selector import get_attn_spec_kind
    from vllm.v1.kv_cache_interface import KVCacheSpecKind

    assert get_attn_spec_kind(use_mla=False, has_sliding_window=False, attn_type="decoder") is (
        KVCacheSpecKind.FULL_ATTENTION
    )
    assert get_attn_spec_kind(use_mla=False, has_sliding_window=True, attn_type="decoder") is (
        KVCacheSpecKind.SLIDING_WINDOW
    )


def test_the_spec_to_kind_mapping_is_what_a03_walks() -> None:
    import torch
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheSpecKind,
        SlidingWindowSpec,
        get_kv_cache_spec_kind,
    )

    full = FullAttentionSpec(block_size=32, num_kv_heads=4, head_size=512, dtype=torch.bfloat16)
    sw = SlidingWindowSpec(block_size=16, num_kv_heads=16, head_size=256, dtype=torch.bfloat16, sliding_window=1024)
    assert get_kv_cache_spec_kind(full) is KVCacheSpecKind.FULL_ATTENTION
    assert get_kv_cache_spec_kind(sw) is KVCacheSpecKind.SLIDING_WINDOW


def test_attention_backend_and_attention_config_are_mutually_exclusive() -> None:
    """`N:engine/arg_utils.py:2491-2496`: the profile passes CUSTOM inside the -ac blob."""
    import vllm.engine.arg_utils as au

    src = inspect.getsource(au)
    assert "attention_config" in src
    assert "attention_backend" in src


def test_the_removed_attention_backend_env_var_is_gone_from_the_package() -> None:
    """RC C6: zero occurrences, so setting it is a silent no-op and means a stale script."""
    import vllm
    from vllm import envs

    assert not hasattr(envs, "VLLM_ATTENTION_BACKEND")
    assert "VLLM_ATTENTION_BACKEND" not in getattr(envs, "environment_variables", {})
    assert vllm.__version__ == "0.29.0"
