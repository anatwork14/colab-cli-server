"""`DattnScratch`: the derived persistent buffers.

Allocated once, on the first `build()` or the first capture, from the runner-supplied
width (`cm.block_table_tensor.shape[1]`) rather than from anything cached at `__init__`.
Never resized. Always sliced `[:R]` from index 0, so `data_ptr` is identical for every
`R`, which is what makes it safe for a captured graph to hold the pointer
(DESIGN.md section 2.4).

DA never writes to a tensor vLLM owns. Everything derived lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from dattn.contracts import Tensor

__all__ = ["DattnScratch", "alloc_scratch"]


@dataclass(slots=True)
class DattnScratch:
    block_table: Tensor  # (max_num_seqs, n_cols) int32, device
    seq_lens: Tensor  # (max_num_seqs,)        int32, device
    n_kept: Tensor  # (max_num_seqs,)        int32, device
    seq_lens_cpu: Tensor  # (max_num_seqs,)        int32, pinned host mirror
    n_cols: int
    block_size: int

    def pointers(self) -> tuple[int, int, int]:
        """The three device `data_ptr`s, for the post-capture stability assertion (A20)."""
        return (
            int(self.block_table.data_ptr()),
            int(self.seq_lens.data_ptr()),
            int(self.n_kept.data_ptr()),
        )

    @property
    def n_rows(self) -> int:
        return int(self.block_table.shape[0])


def alloc_scratch(max_num_seqs: int, n_cols: int, bs: int, device: Any) -> DattnScratch:
    """Allocate the four buffers once. Never call this twice for one builder."""
    if max_num_seqs < 1 or n_cols < 1 or bs < 1:
        raise ValueError(f"alloc_scratch({max_num_seqs=}, {n_cols=}, {bs=}) needs three positive sizes")
    pin = torch.cuda.is_available()
    return DattnScratch(
        block_table=torch.zeros((max_num_seqs, n_cols), dtype=torch.int32, device=device),
        seq_lens=torch.zeros((max_num_seqs,), dtype=torch.int32, device=device),
        n_kept=torch.zeros((max_num_seqs,), dtype=torch.int32, device=device),
        seq_lens_cpu=torch.zeros((max_num_seqs,), dtype=torch.int32, device="cpu", pin_memory=pin),
        n_cols=int(n_cols),
        block_size=int(bs),
    )
