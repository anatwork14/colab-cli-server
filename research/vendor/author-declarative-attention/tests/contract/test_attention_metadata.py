"""`CommonAttentionMetadata` and the metadata-builder ABC, pinned to vLLM 0.29.0.

Everything `DattnState.compact` reads or re-points lives here. A field rename upstream
is a silent wrong-answer bug for dattn, so each one is named explicitly.
"""

from __future__ import annotations

import dataclasses
import inspect
import warnings

import pytest
import torch

pytestmark = pytest.mark.contract


def _cm(**kw):
    from vllm.v1.attention.backend import CommonAttentionMetadata

    rows = kw.pop("rows", 2)
    n_cols = kw.pop("n_cols", 8)
    qsl = torch.arange(rows + 1, dtype=torch.int32)
    base = {
        "query_start_loc": qsl,
        "query_start_loc_cpu": qsl,
        "seq_lens": torch.full((rows,), 100, dtype=torch.int32),
        "num_reqs": rows,
        "num_actual_tokens": rows,
        "max_query_len": 1,
        "max_seq_len": 100,
        "block_table_tensor": torch.arange(rows * n_cols, dtype=torch.int32).reshape(rows, n_cols),
        "slot_mapping": torch.zeros((rows,), dtype=torch.int64),
    }
    base.update(kw)
    return CommonAttentionMetadata(**base)


def test_common_attention_metadata_carries_every_field_dattn_reads() -> None:
    """`N:v1/attention/backend.py:367-460`."""
    from vllm.v1.attention.backend import CommonAttentionMetadata

    names = {f.name: f for f in dataclasses.fields(CommonAttentionMetadata)}
    for required in (
        "seq_lens",
        "block_table_tensor",
        "slot_mapping",
        "max_seq_len",
        "num_reqs",
        "query_start_loc",
        "query_start_loc_cpu",
        "is_prefilling",
        "seq_lens_cpu_upper_bound",
        "_seq_lens_cpu",
        "_num_computed_tokens_cpu",
        "_num_computed_tokens_cache",
        "_token_to_req_indices_cache",
    ):
        assert required in names, f"CommonAttentionMetadata lost {required!r}"
    assert names["max_seq_len"].type in (int, "int")
    assert names["num_reqs"].type in (int, "int")


def test_replace_returns_a_new_object_and_leaves_the_original_alone() -> None:
    """`N:v1/attention/backend.py:460-461`: dattn re-points a copy, never the original."""
    cm = _cm()
    scratch = torch.zeros_like(cm.seq_lens)
    cm2 = cm.replace(seq_lens=scratch, max_seq_len=7, _seq_lens_cpu=None)
    assert cm2 is not cm
    assert cm2.seq_lens is scratch
    assert cm.seq_lens is not scratch
    assert cm.max_seq_len == 100 and cm2.max_seq_len == 7
    assert cm2.block_table_tensor is cm.block_table_tensor


def test_the_deprecated_cpu_mirror_falls_back_to_a_d2h_copy() -> None:
    """`N:v1/attention/backend.py:468-473`. This is exactly the sync dattn must avoid."""
    cm = _cm(_seq_lens_cpu=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mirror = cm.seq_lens_cpu
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert mirror.device.type == "cpu"
    assert cm._seq_lens_cpu is mirror, "the property memoises, so the sync happens once"


def test_num_computed_tokens_cpu_would_sync_too() -> None:
    cm = _cm(_seq_lens_cpu=None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = cm.num_computed_tokens_cpu
    assert got.device.type == "cpu"


def test_the_builder_build_signature_is_the_one_dattn_overrides() -> None:
    """`N:v1/attention/backend.py:658`: build(self, common_prefix_len, common_attn_metadata, fast_build=False)."""
    from vllm.v1.attention.backend import AttentionMetadataBuilder

    sig = inspect.signature(AttentionMetadataBuilder.build)
    assert list(sig.parameters) == ["self", "common_prefix_len", "common_attn_metadata", "fast_build"]
    assert sig.parameters["fast_build"].default is False

    init = inspect.signature(AttentionMetadataBuilder.__init__)
    assert list(init.parameters) == ["self", "kv_cache_spec", "layer_names", "vllm_config", "device"]

    capture = inspect.signature(AttentionMetadataBuilder.build_for_cudagraph_capture)
    assert list(capture.parameters) == ["self", "common_attn_metadata"]


def test_supports_update_block_table_is_a_plain_class_attribute() -> None:
    """`N:v1/attention/backend.py:594`. A06 reads it off the class, not an instance."""
    from vllm.v1.attention.backend import AttentionMetadataBuilder

    assert AttentionMetadataBuilder.supports_update_block_table is False
    assert "supports_update_block_table" in vars(AttentionMetadataBuilder)
    sig = inspect.signature(AttentionMetadataBuilder.update_block_table)
    assert list(sig.parameters) == ["self", "metadata", "blk_table", "slot_mapping"]


def test_the_flash_attention_builder_is_the_one_that_sets_it_true() -> None:
    """RC C9: FA is True, Triton and FlashInfer are False. A06 widens to every builder."""
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
    from vllm.v1.attention.backends.flashinfer import FlashInferMetadataBuilder
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

    assert FlashAttentionMetadataBuilder.supports_update_block_table is True
    assert TritonAttentionMetadataBuilder.supports_update_block_table is False
    assert FlashInferMetadataBuilder.supports_update_block_table is False


def test_use_cascade_attention_is_a_builder_method_defaulting_to_false() -> None:
    """A08. `N:v1/attention/backend.py:734`."""
    from vllm.v1.attention.backend import AttentionMetadataBuilder

    sig = inspect.signature(AttentionMetadataBuilder.use_cascade_attention)
    assert list(sig.parameters)[:3] == ["self", "common_prefix_len", "query_lens"]


def test_disable_cascade_attn_defaults_to_true() -> None:
    from vllm.config.model import ModelConfig

    fields = {f.name: f for f in dataclasses.fields(ModelConfig)}
    assert fields["disable_cascade_attn"].default is True


def test_triton_stores_seq_lens_and_block_table_by_reference() -> None:
    """The whole design rests on this: re-pointing `cm` re-points the kernel's inputs.

    `TritonAttentionMetadataBuilder.build` reads only five instance attributes outside
    the cascade branch, so the real `build` runs here against a stand-in `self` with no
    engine, no model and no GPU.
    """
    from types import SimpleNamespace

    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder

    fake_self = SimpleNamespace(
        device=torch.device("cpu"),
        seq_threshold_3D=8,
        num_par_softmax_segments=16,
        softmax_segm_output=torch.zeros(1),
        softmax_segm_max=torch.zeros(1),
        softmax_segm_expsum=torch.zeros(1),
        rswa_window=None,
    )
    cm = _cm()
    md = TritonAttentionMetadataBuilder.build(fake_self, 0, cm)
    assert md.seq_lens is cm.seq_lens
    assert md.block_table is cm.block_table_tensor
    assert md.slot_mapping is cm.slot_mapping
    assert md.max_seq_len == cm.max_seq_len
    assert md.use_cascade is False


def test_triton_reads_the_cpu_mirror_only_under_cascade() -> None:
    """So Gemma needs no mirror, and `_seq_lens_cpu = None` is safe there."""
    from vllm.v1.attention.backends import triton_attn

    src = inspect.getsource(triton_attn.TritonAttentionMetadataBuilder.build)
    head, _, cascade = src.partition("if use_cascade:")
    assert "seq_lens.cpu()" not in head, "Triton grew a CPU read outside the cascade branch"
    assert "seq_lens.cpu()" in cascade


def test_triton_capture_fills_seq_lens_with_one() -> None:
    """It writes through `attn_metadata.seq_lens`, which is dattn's scratch during capture."""
    from vllm.v1.attention.backends import triton_attn

    src = inspect.getsource(triton_attn.TritonAttentionMetadataBuilder.build_for_cudagraph_capture)
    assert "seq_lens.fill_(1)" in src
