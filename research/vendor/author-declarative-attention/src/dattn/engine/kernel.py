"""Fused Triton compaction with int64 indices and stable output scratch.

The independent set-union oracle in compact.py specifies the kept blocks and lengths.
GPU equivalence tests cover partial tails, inactive rows and multiple tiles.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from dattn.contracts import MAX_INTERVALS, NULL_BLOCK_ID, DattnInvariantError, Tensor

__all__ = ["TILE", "compact_rows"]

#: Columns per program along the second grid axis.
TILE = 512


@triton.jit
def _compact_rows_kernel(
    intervals_ptr,
    counts_ptr,
    seq_lens_ptr,
    active_ptr,
    src_ptr,
    src_stride,
    dst_ptr,
    dst_stride,
    out_len_ptr,
    out_n_kept_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    K: tl.constexpr,
    TILE_C: tl.constexpr,
    NULL_ID: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)

    seq = tl.load(seq_lens_ptr + r).to(tl.int64)
    zero = seq * 0  # an int64 zero, without materialising a python constant
    n_cols64 = n_cols + zero
    nblk = (seq + BLOCK_SIZE - 1) // BLOCK_SIZE
    nblk = tl.minimum(nblk, n_cols64)
    count = tl.load(counts_ptr + r).to(tl.int64)

    col = tile * TILE_C + tl.arange(0, TILE_C).to(tl.int64)
    src = col
    run = zero
    end = zero
    tail_reach = zero

    for k in tl.static_range(K):
        valid = k < count
        s = tl.load(intervals_ptr + (r * K + k) * 2).to(tl.int64)
        e = tl.load(intervals_ptr + (r * K + k) * 2 + 1).to(tl.int64)
        lo = s // BLOCK_SIZE
        hi = (e + BLOCK_SIZE - 1) // BLOCK_SIZE
        hi = tl.minimum(hi, nblk)
        hi = tl.where(valid, hi, zero)
        lo = tl.minimum(lo, hi)
        lo = tl.maximum(lo, run)
        length = tl.maximum(hi - lo, zero)
        piece = (col >= end) & (col < end + length)
        src = tl.where(piece, lo + (col - end), src)
        # The tail question is asked of the last POSITIVE-LENGTH piece, not of the
        # cummax: an interval starting at or beyond seq_len raises the cummax to
        # nblk while contributing nothing (DESIGN.md section 4.4).
        tail_reach = tl.where(length > 0, tl.maximum(tail_reach, hi), tail_reach)
        end = end + length
        run = tl.maximum(run, hi)

    n_kept = end
    active = (tl.load(active_ptr + r) != 0) & (n_kept > 0)

    cmask = col < n_cols64
    keep = active & (col < n_kept)
    src = tl.where(keep, src, col)
    src = tl.minimum(tl.maximum(src, zero), tl.maximum(n_cols64 - 1, zero))
    vals = tl.load(src_ptr + r * src_stride + src, mask=cmask, other=0)
    vals = tl.where(active & (col >= n_kept), NULL_ID, vals)
    tl.store(dst_ptr + r * dst_stride + col, vals, mask=cmask)

    if tile == 0:
        tail_kept = (tail_reach == nblk) & (n_kept > 0)
        tail_valid = seq - tl.maximum(nblk - 1, zero) * BLOCK_SIZE
        compact_len = tl.where(tail_kept, (n_kept - 1) * BLOCK_SIZE + tail_valid, n_kept * BLOCK_SIZE)
        tl.store(out_len_ptr + r, tl.where(active, compact_len, seq).to(tl.int32))
        tl.store(out_n_kept_ptr + r, tl.where(active, n_kept, nblk).to(tl.int32))


def compact_rows(
    intervals: Tensor,
    counts: Tensor,
    seq_lens: Tensor,
    active: Tensor,
    block_table: Tensor,
    bs: int,
    *,
    out_block_table: Tensor,
    out_seq_lens: Tensor,
    out_n_kept: Tensor,
) -> None:
    """Gather retained blocks and write the three preallocated output tensors."""
    if bs < 1:
        raise ValueError(f"block size must be positive: {bs}")
    rows = int(out_block_table.shape[0])
    n_cols = int(out_block_table.shape[1])
    if rows == 0 or n_cols == 0:
        return
    if block_table.shape[1] < n_cols:
        raise DattnInvariantError(
            f"source block table is narrower than the destination: {block_table.shape[1]} < {n_cols}"
        )
    if intervals.shape[1] != MAX_INTERVALS:
        raise DattnInvariantError(f"intervals must be (R, {MAX_INTERVALS}, 2), got {tuple(intervals.shape)}")
    if not intervals.is_contiguous():
        raise DattnInvariantError("intervals must be contiguous")
    if block_table.stride(1) != 1 or out_block_table.stride(1) != 1:
        raise DattnInvariantError("block tables must be row-major (stride(1) == 1)")
    if not (out_seq_lens.is_contiguous() and out_n_kept.is_contiguous()):
        raise DattnInvariantError("out_seq_lens and out_n_kept must be contiguous")

    grid = (rows, (n_cols + TILE - 1) // TILE)
    _compact_rows_kernel[grid](
        intervals,
        counts,
        seq_lens,
        active.to(torch.bool),
        block_table,
        block_table.stride(0),
        out_block_table,
        out_block_table.stride(0),
        out_seq_lens,
        out_n_kept,
        n_cols,
        BLOCK_SIZE=bs,
        K=MAX_INTERVALS,
        TILE_C=TILE,
        NULL_ID=NULL_BLOCK_ID,
        num_warps=4,
    )
