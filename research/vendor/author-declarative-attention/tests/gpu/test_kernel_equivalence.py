"""Fused Triton output against the independent set-union oracle on the assigned GPU."""

from __future__ import annotations

import random

import pytest

from dattn.contracts import INF, MAX_INTERVALS, NULL_BLOCK_ID, Mask
from dattn.engine import compact, kernel

pytestmark = pytest.mark.gpu

BLOCK_SIZES = [1, 16, 32, 64, 128]


@pytest.fixture(scope="module")
def cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible; assign an idle GPU with CUDA_VISIBLE_DEVICES")
    return torch


def _masks_to_tensors(torch, masks, device):
    rows = len(masks)
    iv = torch.zeros((rows, MAX_INTERVALS, 2), dtype=torch.int32)
    counts = torch.zeros((rows,), dtype=torch.int32)
    for r, m in enumerate(masks):
        ivals = list(m.intervals) or [(0, INF)]
        for k, (s, e) in enumerate(ivals[:MAX_INTERVALS]):
            iv[r, k, 0] = s
            iv[r, k, 1] = min(e, INF)
        counts[r] = min(len(ivals), MAX_INTERVALS)
    return iv.to(device), counts.to(device)


def _random_mask(rng, seq_len):
    ivals = [(0, rng.choice([8, 16, 32]))]
    tail = rng.randrange(max(1, seq_len // 2), max(2, seq_len))
    for _ in range(rng.randrange(0, MAX_INTERVALS - 1)):
        s = rng.randrange(0, max(1, tail))
        ivals.append((s, s + rng.randrange(1, max(2, seq_len // 4))))
    ivals.append((tail, INF))
    return Mask.of(*ivals)


def _run(torch, impl, iv, counts, seq_lens, active, bt, bs, n_cols):
    rows = bt.shape[0]
    out_bt = torch.full((rows, n_cols), -7, dtype=torch.int32, device=bt.device)
    out_sl = torch.zeros((rows,), dtype=torch.int32, device=bt.device)
    out_nk = torch.zeros((rows,), dtype=torch.int32, device=bt.device)
    impl(
        iv,
        counts,
        seq_lens,
        active,
        bt,
        bs,
        out_block_table=out_bt,
        out_seq_lens=out_sl,
        out_n_kept=out_nk,
    )
    return out_bt, out_sl, out_nk


@pytest.mark.parametrize("bs", BLOCK_SIZES)
def test_the_triton_kernel_agrees_with_the_oracle(cuda, bs: int) -> None:
    torch = cuda
    dev = torch.device("cuda")
    rng = random.Random(1234 + bs)
    peak_before = torch.cuda.max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    n_active_seen = 0

    for _ in range(300 // len(BLOCK_SIZES)):
        rows = rng.randrange(1, 18)
        n_cols = rng.choice([8, 64, 600, 1300])  # 1300 forces the multi-tile grid
        seq_lens = [rng.randrange(1, min(4000, n_cols * bs) + 1) for _ in range(rows)]
        masks = [_random_mask(rng, s) for s in seq_lens]
        active = [rng.random() < 0.8 for _ in range(rows)]
        bt = torch.randint(1, 5000, (rows, n_cols), dtype=torch.int32, device=dev)
        iv, counts = _masks_to_tensors(torch, masks, dev)
        sl = torch.tensor(seq_lens, dtype=torch.int32, device=dev)
        act = torch.tensor(active, dtype=torch.bool, device=dev)

        k_bt, k_sl, k_nk = _run(torch, kernel.compact_rows, iv, counts, sl, act, bt, bs, n_cols)
        iv_cpu = iv.cpu().tolist()
        counts_cpu = counts.cpu().tolist()
        k_bt_cpu = k_bt.cpu().tolist()
        bt_cpu = bt.cpu().tolist()
        for r in range(rows):
            blocks, out_len = compact.oracle(iv_cpu[r], counts_cpu[r], seq_lens[r], bs)
            is_active = active[r] and bool(blocks)
            n_active_seen += int(is_active)
            if not is_active:
                assert k_bt_cpu[r] == bt_cpu[r][:n_cols]
                assert int(k_sl[r]) == seq_lens[r]
                continue
            want = [bt_cpu[r][b] for b in blocks] + [NULL_BLOCK_ID] * (n_cols - len(blocks))
            assert k_bt_cpu[r] == want, f"row {r}, bs={bs}, mask={masks[r].intervals}"
            assert int(k_sl[r]) == out_len
            assert int(k_nk[r]) == len(blocks)

    peak = torch.cuda.max_memory_allocated() - peak_before
    assert n_active_seen > 0, "the generator produced no active row"
    assert peak < 64 << 20, f"the kernel path allocated {peak} bytes"


def test_the_tail_kept_regression_on_device(cuda) -> None:
    """`W 2.4`, the defect both reference kernels hide, checked in the real kernel."""
    torch = cuda
    dev = torch.device("cuda")
    for bs, seq_len in ((16, 20), (16, 17), (64, 70), (32, 100)):
        bt = torch.arange(1, 65, dtype=torch.int32, device=dev).reshape(1, 64)
        iv, counts = _masks_to_tensors(torch, [Mask.of((0, bs), (5000, INF))], dev)
        sl = torch.tensor([seq_len], dtype=torch.int32, device=dev)
        act = torch.tensor([True], device=dev)
        out_bt, out_sl, out_nk = _run(torch, kernel.compact_rows, iv, counts, sl, act, bt, bs, 64)
        assert int(out_sl[0]) == bs, f"bs={bs}, seq_len={seq_len}"
        assert int(out_nk[0]) == 1
        assert int(out_bt[0, 0]) == 1


def test_the_partial_last_page_survives(cuda) -> None:
    torch = cuda
    dev = torch.device("cuda")
    bs, seq_len = 16, 100  # nblk = 7, tail_valid = 4
    bt = torch.arange(1, 65, dtype=torch.int32, device=dev).reshape(1, 64)
    iv, counts = _masks_to_tensors(torch, [Mask.of((0, 16), (90, INF))], dev)
    sl = torch.tensor([seq_len], dtype=torch.int32, device=dev)
    act = torch.tensor([True], device=dev)
    out_bt, out_sl, out_nk = _run(torch, kernel.compact_rows, iv, counts, sl, act, bt, bs, 64)
    assert int(out_nk[0]) == 3
    assert int(out_sl[0]) == 2 * 16 + 4
    assert out_bt[0, :3].cpu().tolist() == [1, 6, 7]


def test_columns_beyond_the_output_width_are_untouched(cuda) -> None:
    torch = cuda
    dev = torch.device("cuda")
    bt = torch.arange(1, 65, dtype=torch.int32, device=dev).reshape(1, 64)
    out_bt = torch.full((1, 64), -7, dtype=torch.int32, device=dev)
    iv, counts = _masks_to_tensors(torch, [Mask.of((0, 16), (200, INF))], dev)
    kernel.compact_rows(
        iv,
        counts,
        torch.tensor([300], dtype=torch.int32, device=dev),
        torch.tensor([True], device=dev),
        bt,
        16,
        out_block_table=out_bt[:, :20],
        out_seq_lens=torch.zeros(1, dtype=torch.int32, device=dev),
        out_n_kept=torch.zeros(1, dtype=torch.int32, device=dev),
    )
    assert out_bt[0, 20:].cpu().tolist() == [-7] * 44
