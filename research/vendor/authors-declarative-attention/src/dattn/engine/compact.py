"""Independent set-union oracle and the host mirror used by the Triton gather.

Partial-tail length depends on the last positive-length interval, not the cummax of
all interval endpoints. The host mirror supplies max_seq_len without a device read.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

from dattn.contracts import MAX_INTERVALS, DattnInvariantError

__all__ = [
    "checks_enabled",
    "host_lengths",
    "oracle",
    "oracle_row",
]

#: Sentinel written into `oracle_row` columns that carry `NULL_BLOCK_ID` rather than a
#: gathered source column. Not a block id: it is "this column has no source".
NO_SOURCE: int = -1


def checks_enabled() -> bool:
    """`DATTN_CHECK=1` turns on the per-step assertions of DESIGN.md section 6."""
    return os.environ.get("DATTN_CHECK", "") not in ("", "0", "false", "False")


# --------------------------------------------------------------------------------------
# 1. The oracle. Pure Python. THE SPECIFICATION.
# --------------------------------------------------------------------------------------


def oracle(intervals: Sequence[Sequence[int]], count: int, seq_len: int, bs: int) -> tuple[list[int], int]:
    """Return `(kept_blocks, out_len)` for one row. THE SPECIFICATION.

    Set union over `range(s // bs, min(ceil(e / bs), nblk))` per interval, sorted, with
    `out_len` from section 4.4. `tail_kept` is "is block `nblk - 1` in the kept set",
    which is the same question as "does the last positive-length piece reach `nblk`" and
    is correct unconditionally.
    """
    if bs < 1:
        raise ValueError(f"block size must be positive: {bs}")
    nblk = -(-seq_len // bs)
    kept: set[int] = set()
    for k in range(min(count, len(intervals))):
        s, e = int(intervals[k][0]), int(intervals[k][1])
        kept.update(range(s // bs, min(-(-e // bs), nblk)))
    blocks = sorted(kept)
    n = len(blocks)
    if n == 0:
        return [], 0
    tail_valid = seq_len - (nblk - 1) * bs
    out_len = (n - 1) * bs + tail_valid if blocks[-1] == nblk - 1 else n * bs
    return blocks, out_len


def oracle_row(intervals: Sequence[Sequence[int]], count: int, seq_len: int, bs: int, n_cols: int) -> list[int]:
    """The full permuted destination row, so tests compare the permutation itself.

    Entry `c` is the **source column** destination column `c` gathers from, or
    `NO_SOURCE` when the column holds `NULL_BLOCK_ID` because the row shrank.
    """
    blocks, _ = oracle(intervals, count, seq_len, bs)
    blocks = [b for b in blocks if b < n_cols]
    return blocks[:n_cols] + [NO_SOURCE] * (n_cols - len(blocks))


# --------------------------------------------------------------------------------------
# 2. The pure-torch plan. CPU or CUDA. The CI reference and the no-Triton fallback.
# --------------------------------------------------------------------------------------


def host_lengths(
    intervals_np: Any,
    counts_np: Any,
    seq_lens_np: Any,
    active_np: Any,
    bs: int,
    n_cols: int,
) -> tuple[Any, Any, Any]:
    """`(out_len, n_kept, active)` on the host, from the pinned mirrors. No sync.

    DESIGN.md section 4.3 takes `max_seq_len` from this mirror and never from the
    device, section 2.5 takes the compacted `seq_lens_cpu` from it on the FlashInfer
    native-plan profile, and section 5.3 audits the device ring against it.

    The returned `active` is the caller's gate ANDed with `n_kept > 0`, which is the
    gate the kernel actually applies.
    """
    import numpy as np

    if bs < 1:
        raise ValueError(f"block size must be positive: {bs}")
    seq = np.asarray(seq_lens_np, dtype=np.int64)
    rows = seq.shape[0]
    iv = np.asarray(intervals_np, dtype=np.int64)[:rows]
    cnt = np.asarray(counts_np, dtype=np.int64)[:rows]
    act_in = np.asarray(active_np).astype(bool)[:rows]

    nblk = np.minimum((seq + bs - 1) // bs, n_cols)
    valid = np.arange(iv.shape[1])[None, :] < cnt[:, None]
    lo = iv[..., 0] // bs
    hi = np.minimum((iv[..., 1] + bs - 1) // bs, nblk[:, None])
    hi = np.where(valid, hi, 0)
    lo = np.minimum(lo, hi)
    run = np.maximum.accumulate(hi, axis=1)
    prev_end = np.concatenate([np.zeros_like(run[:, :1]), run[:, :-1]], axis=1)
    lo = np.maximum(lo, prev_end)
    length = np.maximum(hi - lo, 0)
    n_kept = length.sum(axis=1)

    act = act_in & (n_kept > 0)
    tail_reach = np.where(length > 0, hi, 0).max(axis=1)
    tail_kept = (tail_reach == nblk) & (n_kept > 0)
    tail_valid = seq - np.maximum(nblk - 1, 0) * bs
    compact_len = np.where(tail_kept, (n_kept - 1) * bs + tail_valid, n_kept * bs)
    return (
        np.where(act, compact_len, seq).astype(np.int32),
        np.where(act, n_kept, nblk).astype(np.int32),
        act,
    )


def assert_row_invariants(
    intervals_np: Any,
    counts_np: Any,
    seq_lens_np: Any,
    active_np: Any,
    out_len_np: Any,
    n_kept_np: Any,
    bs: int,
) -> None:
    """The per-step assertions of DESIGN.md section 6, on the host mirror only.

    Cheap enough to leave behind `DATTN_CHECK=1`: `MAX_INTERVALS` is 5 and `R` is at
    most a few hundred, so this is a handful of vectorised numpy comparisons.
    """
    import numpy as np

    seq = np.asarray(seq_lens_np, dtype=np.int64)
    rows = seq.shape[0]
    iv = np.asarray(intervals_np, dtype=np.int64)[:rows]
    cnt = np.asarray(counts_np, dtype=np.int64)[:rows]
    act = np.asarray(active_np).astype(bool)[:rows]
    out_len = np.asarray(out_len_np, dtype=np.int64)[:rows]
    n_kept = np.asarray(n_kept_np, dtype=np.int64)[:rows]

    if not act.any():
        return
    bad = np.flatnonzero(act & (n_kept < 1))
    if bad.size:
        raise DattnInvariantError(f"active rows {bad.tolist()} kept zero blocks: n_kept={n_kept[bad].tolist()}")
    # seq_lens is uint32 on the trtllm path, so an underflow would WRAP rather than fail.
    bad = np.flatnonzero(act & ((out_len <= 0) | (out_len > seq)))
    if bad.size:
        raise DattnInvariantError(
            f"active rows {bad.tolist()} violate 0 < out_len <= seq_len: "
            f"out_len={out_len[bad].tolist()}, seq_len={seq[bad].tolist()}"
        )
    # Every interval must start before seq_len, or the tail-block question is ill posed.
    starts = np.where(np.arange(iv.shape[1])[None, :] < cnt[:, None], iv[..., 0], 0)
    bad = np.flatnonzero(act & (starts.max(axis=1) >= seq))
    if bad.size:
        raise DattnInvariantError(
            f"active rows {bad.tolist()} have an interval starting at or beyond seq_len: "
            f"starts={starts[bad].tolist()}, seq_len={seq[bad].tolist()}"
        )
    # The tail block carries the K/V just written for the current token.
    nblk = (seq + bs - 1) // bs
    tail_valid = seq - np.maximum(nblk - 1, 0) * bs
    expect = (n_kept - 1) * bs + tail_valid
    bad = np.flatnonzero(act & (out_len != expect))
    if bad.size:
        raise DattnInvariantError(
            f"active rows {bad.tolist()} dropped the tail block: out_len={out_len[bad].tolist()}, "
            f"expected {expect[bad].tolist()} = (n_kept-1)*{bs} + tail_valid"
        )


assert MAX_INTERVALS >= 2, "the sink and the open tail alone need two intervals"
