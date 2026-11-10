"""Two arm boots per batch, with eight sampled timing repetitions inside each boot.

Walltime uses a 512-token decode cap, not an accuracy setting. DA may stop at EOS,
vanilla uses min_tokens=max_tokens=512 and ignore_eos. Each repetition is trimmed
to its own DA rectangle. The 64-step warm pass establishes the intended prefix-cache
state. Per-request aligned cache hits witness retention in every timed rep.
Residual prefill steps are recorded, never gated by a fixed step count. Sizing
normalizes logical per-sequence KV read load, not physical occupancy. Process
order alternates with task_index + batch_index. Repetitions estimate a within-batch
mean by pooling kept steps over repetitions. Whole batches remain the statistical
bootstrap unit, not the individual repetitions.
"""

from dataclasses import asdict, dataclass, replace

MACRO_TASKS = 15


@dataclass(frozen=True)
class Protocol:
    model_id: str
    use: str = "walltime"
    concurrency: int = 0
    repetitions: int = 8
    batches_per_task: int = 5
    subset_size: int = 128
    max_tokens: int = 512
    seed: int = 0
    leading_trim: int = 8
    bin_width: float = 0.1
    n_resamples: int = 2000
    ci: float = 0.95

    def __post_init__(self):
        if self.repetitions < 1:
            raise ValueError("repetitions must be positive")
        if min(self.max_tokens, self.batches_per_task) < 1:
            raise ValueError("budgets must be positive")

    @property
    def n_bins(self):
        return round(1 / self.bin_width)

    def with_(self, **kwargs):
        return replace(self, **kwargs)

    def to_dict(self):
        return asdict(self)
