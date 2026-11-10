"""Host lengths and invariant checks against the independent set-union oracle."""

import random

import pytest

from dattn.contracts import INF, MAX_INTERVALS, Mask
from dattn.engine import compact

BLOCK_SIZES = [1, 2, 4, 16, 32, 64, 128]


def random_mask(rng, seq_len):
    """A realistic DA mask: sink, up to three focus spans, an open tail."""
    ivals = [(0, rng.choice([8, 16, 32]))]
    tail = rng.randrange(max(1, seq_len // 2), max(2, seq_len))
    for _ in range(rng.randrange(0, MAX_INTERVALS - 2 + 1)):
        s = rng.randrange(0, max(1, tail))
        ivals.append((s, s + rng.randrange(1, max(2, seq_len // 4))))
    ivals.append((tail, INF))
    return Mask.of(*ivals)


def test_the_oracle_is_a_plain_set_union() -> None:
    blocks, out_len = compact.oracle([[0, 16], [64, 96]], 2, 100, 16)
    assert blocks == [0, 4, 5]
    assert out_len == 3 * 16


def test_the_oracle_counts_only_the_valid_prefix_of_intervals() -> None:
    rows = [[0, 16], [64, 96], [900, 950]]
    assert compact.oracle(rows, 2, 1000, 16)[0] == compact.oracle(rows[:2], 2, 1000, 16)[0]


@pytest.mark.parametrize("bs", [16, 32])
def test_host_lengths_agrees_with_the_oracle(bs: int) -> None:
    import numpy as np

    rng = random.Random(11 + bs)
    n_cols = 300
    rows = 24
    seq_lens = [rng.randrange(1, n_cols * bs) for _ in range(rows)]
    masks = [random_mask(rng, s) for s in seq_lens]
    active = [rng.random() < 0.7 for _ in range(rows)]
    iv = np.zeros((rows, MAX_INTERVALS, 2), dtype=np.int64)
    counts = np.array([len(m.intervals) for m in masks])
    for r, mask in enumerate(masks):
        iv[r, : counts[r]] = mask.intervals
    host_len, host_kept, host_act = compact.host_lengths(
        iv, counts, np.asarray(seq_lens), np.asarray(active), bs, n_cols
    )
    for r in range(rows):
        blocks, out_len = compact.oracle(iv[r].tolist(), int(counts[r]), seq_lens[r], bs)
        assert host_kept[r] == (len(blocks) if active[r] and blocks else (seq_lens[r] + bs - 1) // bs)
        assert bool(host_act[r]) == (active[r] and bool(blocks))
        assert int(host_len[r]) == (out_len if host_act[r] else seq_lens[r])


def test_assert_row_invariants_catches_a_dropped_tail() -> None:
    import numpy as np

    from dattn.contracts import DattnInvariantError

    iv = np.zeros((1, MAX_INTERVALS, 2), dtype=np.int64)
    iv[0, 0] = (0, 16)
    with pytest.raises(DattnInvariantError, match="dropped the tail block"):
        compact.assert_row_invariants(
            iv,
            np.array([1]),
            np.array([100]),
            np.array([True]),
            np.array([16]),
            np.array([1]),
            16,
        )


def test_assert_row_invariants_catches_an_interval_past_seq_len() -> None:
    import numpy as np

    from dattn.contracts import DattnInvariantError

    iv = np.zeros((1, MAX_INTERVALS, 2), dtype=np.int64)
    iv[0, 0] = (0, 16)
    iv[0, 1] = (5000, INF)
    with pytest.raises(DattnInvariantError, match="at or beyond seq_len"):
        compact.assert_row_invariants(
            iv,
            np.array([2]),
            np.array([20]),
            np.array([True]),
            np.array([16]),
            np.array([1]),
            16,
        )
