"""The MRV1 model-runner internals and the hybrid block table, pinned to vLLM 0.29.0.

Three facts the whole design rests on:

1. The per-group loop shallow-copies one `CommonAttentionMetadata` and reassigns only
   the encoder seq lens, the block table and the slot mapping, so `seq_lens` is ONE
   object shared by every KV-cache group. Mutating it in place would corrupt Gemma's
   five sliding groups and Qwen's three Mamba groups (RC C11, RC C14).
2. There is a `build()` bypass keyed on `(kv_cache_spec, type(builder))` that fires only
   when `supports_update_block_table` is True, which is why A06 is load-bearing.
3. `BlockTable` hands the builder a table already expanded into kernel blocks,
   `manager_id * 49 + j` on Qwen, and Gemma's page unification doubles the global
   group's block size to 32 (RC C10, RC C11).
"""

from __future__ import annotations

import inspect

import pytest
import torch

pytestmark = pytest.mark.contract


def test_the_per_group_loop_shares_one_seq_lens_object() -> None:
    """`N:v1/worker/gpu_model_runner.py:2593-2611`."""
    from vllm.v1.worker import gpu_model_runner

    src = inspect.getsource(gpu_model_runner.GPUModelRunner._build_attention_metadata)
    assert "cm = copy(cm_base)" in src
    loop = src.split("for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):")[1]
    body = loop.split("if self.speculative_config")[0]
    reassigned = {
        line.split("=")[0].strip().removeprefix("cm.").split(",")[0].strip()
        for line in body.splitlines()
        if line.strip().startswith("cm.") and "=" in line
    }
    assert "seq_lens" not in reassigned, "the loop started reassigning seq_lens; re-read RC C11"
    assert {"block_table_tensor", "slot_mapping"} <= reassigned


def test_the_update_block_table_bypass_exists_and_is_keyed_on_spec_and_builder() -> None:
    """`N:v1/worker/gpu_model_runner.py:2514-2516,2525,2565-2572`. A06 depends on this."""
    from vllm.v1.worker import gpu_model_runner

    src = inspect.getsource(gpu_model_runner.GPUModelRunner._build_attention_metadata)
    assert "cache_key = (kv_cache_spec, type(builder))" in src
    assert "cache_key in cached_attn_metadata" in src
    assert "and builder.supports_update_block_table" in src
    assert "builder.update_block_table(" in src


def test_the_capture_path_calls_build_for_cudagraph_capture_with_max_model_len() -> None:
    """`N:v1/worker/gpu_model_runner.py:2350-2356`: never write max_seq_len during capture."""
    from vllm.v1.worker import gpu_model_runner

    src = inspect.getsource(gpu_model_runner.GPUModelRunner._build_attention_metadata)
    assert "if for_cudagraph_capture:" in src
    assert "max_seq_len = self.max_model_len" in src
    assert "builder.build_for_cudagraph_capture(" in src


def test_padded_rows_are_filled_with_the_null_block_id() -> None:
    """Rule 4 of DESIGN.md section 3: padded rows are dense, hence inactive."""
    from vllm.v1.worker import gpu_model_runner

    src = inspect.getsource(gpu_model_runner.GPUModelRunner._build_attention_metadata)
    assert "fill_(NULL_BLOCK_ID)" in src
    from vllm.v1.worker.gpu_model_runner import NULL_BLOCK_ID

    from dattn.contracts import NULL_BLOCK_ID as OURS

    assert NULL_BLOCK_ID == OURS == 0


def test_metadata_is_built_outside_the_forward_context() -> None:
    """So nothing `build()` does is ever captured into a graph."""
    from vllm.v1.worker import gpu_model_runner

    src = inspect.getsource(gpu_model_runner.GPUModelRunner.execute_model)
    build_at = src.index("_build_attention_metadata")
    ctx_at = src.index("set_forward_context")
    assert build_at < ctx_at


def test_model_forward_is_a_documented_override_point() -> None:
    """O5: the single permitted rebinding in dattn is Lane C's CUDA-event wrapper."""
    from vllm.v1.worker import gpu_model_runner

    fn = gpu_model_runner.GPUModelRunner._model_forward
    assert callable(fn)
    assert "overridden by subclasses" in (fn.__doc__ or "")
    params = list(inspect.signature(fn).parameters)
    assert params[0] == "self"


def test_the_hybrid_block_table_expands_manager_blocks_into_kernel_blocks() -> None:
    """RC C11, EXEC: `BlockTable(block_size=784, kernel_block_size=16)` gives 49 ids each."""
    from vllm.v1.worker.block_table import BlockTable

    bt = BlockTable(
        block_size=784,
        max_num_reqs=4,
        max_num_blocks_per_req=42,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=16,
        cp_kv_cache_interleave_size=1,
    )
    assert bt.use_hybrid_blocks is True
    assert bt.blocks_per_kv_block == 49
    assert bt.block_size == 16
    assert tuple(bt.block_table.gpu.shape) == (4, 42 * 49)

    bt.add_row([5, 9], 0)
    row = bt.block_table.np[0]
    assert row[:3].tolist() == [5 * 49, 5 * 49 + 1, 5 * 49 + 2]
    assert row[48] == 5 * 49 + 48
    assert row[49] == 9 * 49
    assert int(bt.num_blocks_per_row[0]) == 2 * 49


def test_a_non_hybrid_block_table_is_left_alone() -> None:
    from vllm.v1.worker.block_table import BlockTable

    bt = BlockTable(
        block_size=32,
        max_num_reqs=2,
        max_num_blocks_per_req=8,
        max_num_batched_tokens=32,
        pin_memory=False,
        device=torch.device("cpu"),
        kernel_block_size=32,
        cp_kv_cache_interleave_size=1,
    )
    assert bt.use_hybrid_blocks is False
    assert bt.block_size == 32
    assert tuple(bt.block_table.gpu.shape) == (2, 8)


def test_gemma_page_unification_doubles_the_global_groups_block_size() -> None:
    """RC C10: the DA block size on Gemma is 32, and `--block-size 16` cannot change it."""
    from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    full = FullAttentionSpec(block_size=16, num_kv_heads=4, head_size=512, dtype=torch.bfloat16)
    sliding = SlidingWindowSpec(
        block_size=16, num_kv_heads=16, head_size=256, dtype=torch.bfloat16, sliding_window=1024
    )
    out = unify_kv_cache_spec_page_size({"full": full, "sliding": sliding})
    assert out["full"].block_size == 32
    assert out["sliding"].block_size == 16
    assert out["full"].page_size_bytes == out["sliding"].page_size_bytes


def test_select_common_block_size_splits_784_down_to_16_for_flashinfer() -> None:
    """RC C11: 784 = 16 * 49 and FlashInfer advertises [16, 32, 64], so 16 wins."""
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    from vllm.v1.worker.utils import select_common_block_size

    assert select_common_block_size(784, [FlashInferBackend]) == 16


def test_flashinfer_reads_the_cpu_mirror_only_when_not_everything_is_trtllm() -> None:
    """Section 2.5, `N:v1/attention/backends/flashinfer.py:1397-1411`.

    The compacted mirror is load-bearing on the native-plan arm and is deliberately
    `None` on the trtllm arm, where reading it would only add a D2H sync.
    """
    from vllm.v1.attention.backends import flashinfer

    src = inspect.getsource(flashinfer.FlashInferMetadataBuilder.build)
    assert "needs_seq_lens_cpu = self.use_dcp or use_cascade or not all_uses_trtllm" in src
    assert "if needs_seq_lens_cpu:" in src
    before, _, after = src.partition("needs_seq_lens_cpu = ")
    assert "common_attn_metadata.seq_lens_cpu" not in before
    assert "common_attn_metadata.seq_lens_cpu" in after
    assert "num_blocks_np = (seq_lens_np + (page_size - 1)) // page_size" in after
    # The compaction lives or dies on this line: the plan's page count comes from the
    # CPU mirror, so an uncompacted mirror plans the uncompacted row.
    assert "paged_kv_last_page_len_np = seq_lens_np % page_size" in inspect.getsource(
        flashinfer.FlashInferMetadataBuilder._compute_flashinfer_kv_metadata
    )


def test_update_states_runs_before_prepare_inputs_and_carries_refresh_metadata() -> None:
    """The ONE ordering the whole design rests on (review RANKED item 20).

    `engine/processor.py` and `DESIGN.md` section 3 assert that `update_state` runs inside
    `InputBatch.refresh_metadata()`, which `_update_states` calls BEFORE `_prepare_inputs`
    and therefore before `_build_attention_metadata`: "the mask written here is read by
    THIS step's `build()`". Nothing pinned it. `test_metadata_is_built_outside_the_forward
    _context` pins a different pair. If upstream moves `refresh_metadata` after the build,
    every mask lands one step late -- no crash, no log line, and every parity and rho
    number shifts silently.

    Three source-order facts, on the pinned 0.29.0:

    1. `execute_model` calls `_update_states` before `_prepare_inputs`,
    2. `refresh_metadata` is called from `_update_states`, and
    3. `_build_attention_metadata` is called from `_prepare_inputs`.
    """
    from vllm.v1.worker import gpu_model_runner

    runner = gpu_model_runner.GPUModelRunner
    src = inspect.getsource(runner.execute_model)
    update_at = src.find("_update_states")
    prepare_at = src.find("_prepare_inputs")
    assert update_at >= 0, "GPUModelRunner.execute_model no longer calls _update_states"
    assert prepare_at >= 0, "GPUModelRunner.execute_model no longer calls _prepare_inputs"
    assert update_at < prepare_at, (
        "_prepare_inputs now runs before _update_states: every dattn mask would be built from the "
        "PREVIOUS step's state. See DESIGN.md section 3"
    )
    assert "refresh_metadata" in inspect.getsource(runner._update_states), (
        "refresh_metadata left _update_states: update_state no longer runs before the attention "
        "metadata is built, so the mask lands one step late"
    )
    assert "_build_attention_metadata" in inspect.getsource(runner._prepare_inputs), (
        "_build_attention_metadata left _prepare_inputs: the ordering this test pins is no longer "
        "the ordering that matters, re-read the runner"
    )
