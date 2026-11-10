"""Fixtures for the Lane B engine tests.

The stand-ins live in `tests/fakes.py` so the test modules can import them directly. A CPU
`CommonAttentionMetadata` and a stub FSM keep the whole directory under a second on a
box with no GPU; `tests/contract/` is where the real vLLM types are pinned.
"""

from __future__ import annotations

import pytest

from dattn.contracts import PromptLayout


@pytest.fixture
def layout() -> PromptLayout:
    return PromptLayout(n_prompt_tokens=512, question_start=480, segments=((16, 200), (200, 400)), sink=16)


@pytest.fixture
def fresh_state():
    from dattn.engine.state import DATTN_STATE

    DATTN_STATE.__init__()  # type: ignore[misc]
    yield DATTN_STATE
    DATTN_STATE.__init__()  # type: ignore[misc]


@pytest.fixture(autouse=True)
def cpu_compaction(monkeypatch):
    """Exercise state bookkeeping with the independent oracle in CPU tests."""
    import torch

    from dattn.engine import compact, kernel

    def gather(intervals, counts, seq_lens, active, block_table, bs, *, out_block_table, out_seq_lens, out_n_kept):
        width = out_block_table.shape[1]
        for r in range(len(seq_lens)):
            length = int(seq_lens[r])
            blocks, kept = compact.oracle(intervals[r].tolist(), int(counts[r]), length, bs)
            blocks = [b for b in blocks if b < width]
            if not bool(active[r]) or not blocks:
                out_block_table[r].copy_(block_table[r, :width])
                out_seq_lens[r] = length
                out_n_kept[r] = min(width, (length + bs - 1) // bs)
            else:
                out_block_table[r].zero_()
                out_block_table[r, : len(blocks)] = block_table[r, torch.tensor(blocks)]
                out_seq_lens[r] = kept
                out_n_kept[r] = len(blocks)

    monkeypatch.setattr(kernel, "compact_rows", gather)
