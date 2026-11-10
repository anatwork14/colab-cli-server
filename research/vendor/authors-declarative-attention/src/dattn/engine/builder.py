"""The `DattnMetadataBuilder` mixin.

`make_builder_cls(base)` returns a subclass of the pinned base builder that replaces
**only** the metadata builder, and inside it only the derivation of the block table and
`seq_lens`. Everything else -- the kernel, the wrapper, the plan step, the cudagraph
machinery -- is the base backend's, untouched.

`self.block_size` is `kv_cache_spec.block_size`, the KERNEL block size: 32 on Gemma,
16 on Qwen. Never `cache_config.block_size`, which is 16 and 784 respectively
(DESIGN.md section 2.3).
"""

from __future__ import annotations

from typing import Any

from dattn.contracts import BackendProfile, DattnInvariantError
from dattn.engine.state import DATTN_STATE

__all__ = ["make_builder_cls", "needs_compacted_seq_lens_cpu"]


def needs_compacted_seq_lens_cpu(profile: BackendProfile | None, base: type) -> bool:
    """Section 2.5: the one genuinely backend-specific line in the design.

    FlashInfer's native `plan()` derives `num_blocks` and `paged_kv_last_page_len` from
    the CPU mirror. Handing it `None` there makes it plan against the UNCOMPACTED
    lengths and quietly undoes the compaction. Every other path in scope reads the GPU
    tensors, and for those the mirror is set to `None` so nothing can trigger a D2H.

    With no profile the fallback is "FlashInfer always gets the compacted mirror", which
    is the safe direction: an unread mirror costs nothing, an unset one is a silent
    correctness bug.
    """
    if profile is not None:
        return bool(profile.needs_compacted_seq_lens_cpu)
    return "flashinfer" in f"{base.__module__}.{base.__qualname__}".lower()


def make_builder_cls(base: type, profile: BackendProfile | None = None) -> type:
    """Return `DattnMetadataBuilder` derived from the pinned base builder class."""

    class DattnMetadataBuilder(base):  # type: ignore[misc, valid-type]
        # A06. MRV1 caches built metadata on `(kv_cache_spec, type(builder))` and, for a
        # second group with the same key and a True flag, calls `update_block_table`
        # instead of `build()`. That hook swaps only the table and the slot mapping, so
        # there is no place to put a per-group `seq_lens`, and `seq_lens` is one object
        # shared by every group (N:v1/worker/gpu_model_runner.py:2565-2572).
        supports_update_block_table = False

        dattn_profile: BackendProfile | None = profile
        dattn_base_builder_cls: type = base

        def __init__(self, kv_cache_spec: Any, layer_names: Any, vllm_config: Any, device: Any, **kw: Any) -> None:
            super().__init__(kv_cache_spec, layer_names, vllm_config, device, **kw)
            self.block_size = int(kv_cache_spec.block_size)
            self.scratch = None
            self.dattn_group_id = -1
            self.dattn_needs_seq_lens_cpu = needs_compacted_seq_lens_cpu(profile, base)
            self._dattn_capturing = False
            self._dattn_capture_ptrs: tuple[int, int, int] | None = None
            if profile is not None and profile.dattn_block_size != self.block_size:
                raise DattnInvariantError(
                    f"A04: the profile says the DA block size is {profile.dattn_block_size} but this "
                    f"builder's kv_cache_spec.block_size is {self.block_size}. That number comes from "
                    "the hybrid page arithmetic, not from --block-size, so a mismatch means the "
                    "profile was recorded on a different configuration."
                )
            sched = getattr(vllm_config, "scheduler_config", None)
            if sched is not None:
                DATTN_STATE.configure(int(sched.max_num_seqs), device)

        def build(
            self,
            common_prefix_len: int = 0,
            common_attn_metadata: Any = None,
            fast_build: bool = False,
            **kw: Any,
        ) -> Any:
            if common_prefix_len:
                raise DattnInvariantError(
                    f"A08: cascade prefix must be 0, got {common_prefix_len}. Cascade attention "
                    "positionally slices the block table, which a permuted table breaks. "
                    "Pass --disable-cascade-attn."
                )
            if self._dattn_capturing:
                return super().build(0, common_attn_metadata, fast_build=fast_build, **kw)
            cm2 = DATTN_STATE.compact(common_attn_metadata, self)
            return super().build(0, cm2, fast_build=fast_build, **kw)

        def build_for_cudagraph_capture(self, common_attn_metadata: Any) -> Any:
            """Stage the scratch so capture and DA compaction use the same pointers."""
            cm2 = DATTN_STATE.stage_for_capture(common_attn_metadata, self)
            self._dattn_capturing = True
            try:
                md = super().build_for_cudagraph_capture(cm2)
            finally:
                self._dattn_capturing = False
            if self.scratch is not None:
                ptrs = self.scratch.pointers()
                if self._dattn_capture_ptrs is not None and ptrs != self._dattn_capture_ptrs:
                    raise DattnInvariantError(
                        f"A20: scratch pointers moved across cudagraph captures: "
                        f"{self._dattn_capture_ptrs} -> {ptrs}. A captured graph holds them."
                    )
                self._dattn_capture_ptrs = ptrs
            return md

        def update_block_table(self, *a: Any, **k: Any) -> Any:
            raise DattnInvariantError("A06: unreachable, supports_update_block_table is False on every dattn builder")

        def use_cascade_attention(self, *a: Any, **k: Any) -> bool:
            return False

    DattnMetadataBuilder.__name__ = f"Dattn{base.__name__}"
    DattnMetadataBuilder.__qualname__ = DattnMetadataBuilder.__name__
    return DattnMetadataBuilder
