"""`_make_backend` and `make_builder_cls`: what is replaced and what is inherited (WP5).

The point of the design is that **only** the metadata builder is replaced, and inside it
only the derivation of the block table and `seq_lens`. These tests use a fake base so
they cost nothing; `tests/gpu/test_boot_assertions.py` proves it on the real backends.
"""

from __future__ import annotations

import pytest
import torch
from fakes import (  # noqa: F401  (make_cm is used through build())
    BaseBackend,
    BaseBuilder,
    FakeBuilder,
    FakeSchedulerConfig,
    FakeSpec,
    FakeVllmConfig,
    make_cm,
)

from dattn.contracts import INF, DattnBootError, DattnInvariantError, Mask
from dattn.engine import backend as backend_mod
from dattn.engine.builder import make_builder_cls, needs_compacted_seq_lens_cpu
from dattn.engine.state import DATTN_STATE


@pytest.fixture
def builder_cls():
    return make_builder_cls(BaseBuilder)


@pytest.fixture(autouse=True)
def _fresh_state():
    DATTN_STATE.__init__()  # type: ignore[misc]
    yield
    DATTN_STATE.__init__()  # type: ignore[misc]


def make_builder(builder_cls, block_size: int = 16):
    return builder_cls(FakeSpec(block_size), ["l0"], FakeVllmConfig(), torch.device("cpu"))


# ---------------------------------------------------------------------------------------
# the backend
# ---------------------------------------------------------------------------------------


def test_the_backend_overrides_only_the_name_and_the_builder() -> None:
    cls = backend_mod._make_backend(BaseBackend)
    assert cls.get_name() == "CUSTOM"
    assert cls.get_builder_cls() is not BaseBuilder
    assert issubclass(cls, BaseBackend)
    # Load-bearing for Gemma: TRITON_ATTN is the only non-Flex backend whose
    # supports_mm_prefix() is True, and a subclass that broke it would fail selection.
    assert cls.supports_mm_prefix() is True
    assert cls.get_supported_kernel_block_sizes() == [16]
    assert cls.supported_dtypes == BaseBackend.supported_dtypes
    assert cls.dattn_base_backend is BaseBackend


def test_reading_dattn_backend_before_the_plugin_ran_says_why() -> None:
    saved = backend_mod.__dict__.pop("DattnBackend", None)
    try:
        with pytest.raises(DattnBootError, match="DATTN_MODEL"):
            backend_mod.DattnBackend  # noqa: B018
    finally:
        if saved is not None:
            backend_mod.__dict__["DattnBackend"] = saved


def test_install_backend_publishes_the_module_level_name() -> None:
    try:
        cls = backend_mod.install_backend(BaseBackend)
        assert backend_mod.DattnBackend is cls
        assert backend_mod.installed_base() is BaseBackend
    finally:
        backend_mod.__dict__.pop("DattnBackend", None)
        backend_mod._INSTALLED.update({"backend": None, "base": None, "profile": None})


# ---------------------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------------------


def test_the_builder_inherits_everything_it_does_not_override(builder_cls) -> None:
    b = make_builder(builder_cls)
    assert b.something_inherited() == "inherited"
    assert isinstance(b, BaseBuilder)


def test_supports_update_block_table_is_forced_false(builder_cls) -> None:
    """A06: the MRV1 bypass would apply one group's table under a shared shrunk seq_lens."""
    assert BaseBuilder.supports_update_block_table is True
    assert builder_cls.supports_update_block_table is False
    b = make_builder(builder_cls)
    with pytest.raises(DattnInvariantError, match="A06"):
        b.update_block_table(None, None, None)


def test_the_block_size_is_the_kernel_block_size(builder_cls) -> None:
    assert make_builder(builder_cls, 32).block_size == 32
    assert make_builder(builder_cls, 16).block_size == 16


def test_cascade_is_refused_in_both_directions(builder_cls) -> None:
    b = make_builder(builder_cls)
    assert b.use_cascade_attention(0, None, 1, 1, False, False, False, 1, 1) is False
    with pytest.raises(DattnInvariantError, match="A08"):
        b.build(7, make_cm([10], n_cols=4))


def test_build_hands_the_base_a_compacted_metadata(builder_cls, layout) -> None:
    from dattn.contracts import DattnRequest

    b = make_builder(builder_cls)
    DATTN_STATE.begin_step()
    DATTN_STATE.admit(0, "a", "a", DattnRequest(layout=layout), [])
    DATTN_STATE.set_mask(0, Mask.of((0, 16), (160, INF)))
    cm = make_cm([200], n_cols=32)
    out = b.build(0, cm)
    passed = out[2]
    assert passed is not cm
    assert passed.seq_lens.data_ptr() == b.scratch.seq_lens.data_ptr()
    assert int(passed.seq_lens[0]) < 200


def test_capture_is_a_pass_through_that_still_binds_the_scratch(builder_cls) -> None:
    b = make_builder(builder_cls)
    cm = make_cm([200, 300], n_cols=32)
    cm.max_seq_len = 1 << 20
    out = b.build_for_cudagraph_capture(cm)
    passed = out[2]
    assert passed.block_table_tensor.data_ptr() == b.scratch.block_table.data_ptr()
    assert passed.seq_lens.data_ptr() == b.scratch.seq_lens.data_ptr()
    assert torch.equal(passed.seq_lens, cm.seq_lens), "capture must not compact"
    assert b._dattn_capturing is False
    assert b._dattn_capture_ptrs == b.scratch.pointers()


def test_the_profile_pins_the_block_size() -> None:
    from dattn.engine.profile import load_profile

    gemma = load_profile("google/gemma-4-31B-it")
    cls = make_builder_cls(BaseBuilder, gemma)
    make_builder(cls, 32)
    with pytest.raises(DattnInvariantError, match="A04"):
        make_builder(cls, 16)


def test_needs_compacted_seq_lens_cpu_follows_the_profile() -> None:
    from dattn.engine.profile import load_profile

    assert needs_compacted_seq_lens_cpu(load_profile("Qwen/Qwen3.6-27B", "sm100", "walltime"), BaseBuilder) is True
    assert needs_compacted_seq_lens_cpu(load_profile("Qwen/Qwen3.6-27B", "sm100", "parity"), BaseBuilder) is False
    assert needs_compacted_seq_lens_cpu(load_profile("google/gemma-4-31B-it"), BaseBuilder) is False


def test_without_a_profile_flashinfer_still_gets_the_mirror() -> None:
    class FlashInferMetadataBuilder(BaseBuilder):
        pass

    FlashInferMetadataBuilder.__module__ = "vllm.v1.attention.backends.flashinfer"
    assert needs_compacted_seq_lens_cpu(None, FlashInferMetadataBuilder) is True
    assert needs_compacted_seq_lens_cpu(None, BaseBuilder) is False
