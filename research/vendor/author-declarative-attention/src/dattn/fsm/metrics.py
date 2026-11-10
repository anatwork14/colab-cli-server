"""The FSM ledger and dac's metric key set.

Reproduce dac's two documented asymmetries (`focus_ratio` NaN but `local_ratio` 0.0 when
`total_steps == 0`, and `focus_successes` incrementing only on `</focus>`). Do NOT
repeat the reference's omission: count every position, per `main.tex:629`.

`Ledger` carries every per-request counter, not just the per-step ones, because
`full_metrics(ledger)` is the shared signature and it has nothing else to read from.
"""

import math
from typing import Final

from dattn.contracts import Mask, Mode
from dattn.fsm.tags import FOCUS_FAIL_REASONS

#: The 20 shared keys, plus `focus_fail_{reason}` for the five index-path reasons.
METRIC_KEYS: Final[tuple[str, ...]] = (
    "total_steps",
    "attended_tokens",
    "total_attendable",
    "attention_ratio",
    "global_steps",
    "focus_steps",
    "focus_ratio",
    "focus_attempts",
    "focus_successes",
    "focus_success_rate",
    "mean_focus_area",
    "mean_focus_span_area",
    "mean_segments_per_focus",
    "n_segments",
    "n_unique_segments_focused",
    "local_steps",
    "local_ratio",
    "mean_local_area",
    "segment_format_error",
    "segment_format_error_reason",
    *(f"focus_fail_{r}" for r in FOCUS_FAIL_REASONS),
)

__all__ = ["METRIC_KEYS", "Ledger", "full_metrics"]


def _mean(values: list[int]) -> float:
    """dac's mean: NaN on an empty history, never 0.0."""
    return sum(values) / len(values) if values else math.nan


class Ledger:
    """Accumulates per-step counts and per-tag history for one request."""

    __slots__ = (
        "attended",
        "fail_counts",
        "fail_log",
        "focus_areas",
        "focus_attempts",
        "focus_ids",
        "focus_span_areas",
        "focus_successes",
        "format_error",
        "local_areas",
        "n_segments",
        "steps_by_mode",
        "total_attendable",
        "total_steps",
    )

    def __init__(self, n_segments: int) -> None:
        self.n_segments = int(n_segments)
        self.total_steps = 0
        self.total_attendable = 0
        self.attended: list[int] = []
        self.steps_by_mode: dict[Mode, int] = {}
        self.focus_attempts = 0
        self.focus_successes = 0
        self.focus_areas: list[int] = []
        self.focus_span_areas: list[int] = []
        self.focus_ids: list[tuple[int, ...]] = []
        self.local_areas: list[int] = []
        self.fail_counts: dict[str, int] = {}
        self.fail_log: list[str] = []
        self.format_error: str = ""

    def record(self, step: int, mode: Mode, mask: Mask, seq_len: int) -> None:
        """One decode step, recorded with the mask that was in force for that forward.

        `step` is the 1-based step index and is carried for the caller's own bookkeeping.
        `seq_len` is `now`, the number of positions the forward could have attended, so
        `total_attendable` counts every position including pads (docs/PROVENANCE.md P5).
        """
        del step  # kept in the signature: it is the shared contract and the replay index.
        self.total_steps += 1
        self.total_attendable += seq_len
        self.attended.append(mask.area(seq_len))
        self.steps_by_mode[mode] = self.steps_by_mode.get(mode, 0) + 1

    # -- tag history -------------------------------------------------------------------

    def note_focus_attempt(self) -> None:
        self.focus_attempts += 1

    def note_focus_open(self, ids: tuple[int, ...], area: int, span_area: int) -> None:
        self.focus_ids.append(ids)
        self.focus_areas.append(area)
        self.focus_span_areas.append(span_area)

    def note_focus_close(self) -> None:
        """dac counts a success on `</focus>`, not on the open (`DAC_SUCCESS_ON_CLOSE`)."""
        self.focus_successes += 1

    def note_focus_fail(self, reason: str) -> None:
        self.fail_counts[reason] = self.fail_counts.get(reason, 0) + 1
        self.fail_log.append(reason)

    def note_local_open(self, area: int) -> None:
        self.local_areas.append(area)


def full_metrics(ledger: Ledger) -> dict[str, float | int | str]:
    """Return exactly `METRIC_KEYS`, so parity against dac is a dict comparison."""
    steps = ledger.total_steps
    attended = sum(ledger.attended)
    focus_steps = ledger.steps_by_mode.get(Mode.FOCUS, 0)
    local_steps = ledger.steps_by_mode.get(Mode.LOCAL, 0)
    out: dict[str, float | int | str] = {
        "total_steps": steps,
        "attended_tokens": attended,
        "total_attendable": ledger.total_attendable,
        "attention_ratio": attended / ledger.total_attendable if ledger.total_attendable else math.nan,
        "global_steps": ledger.steps_by_mode.get(Mode.GLOBAL, 0),
        "focus_steps": focus_steps,
        # dac's asymmetry: focus_ratio is NaN at zero steps, local_ratio is 0.0.
        "focus_ratio": focus_steps / steps if steps else math.nan,
        "focus_attempts": ledger.focus_attempts,
        "focus_successes": ledger.focus_successes,
        "focus_success_rate": (ledger.focus_successes / ledger.focus_attempts if ledger.focus_attempts else math.nan),
        "mean_focus_area": _mean(ledger.focus_areas),
        "mean_focus_span_area": _mean(ledger.focus_span_areas),
        "mean_segments_per_focus": _mean([len(ids) for ids in ledger.focus_ids]),
        "n_segments": ledger.n_segments,
        "n_unique_segments_focused": len({i for ids in ledger.focus_ids for i in ids}),
        "local_steps": local_steps,
        "local_ratio": local_steps / steps if steps else 0.0,
        "mean_local_area": _mean(ledger.local_areas),
        "segment_format_error": 1 if ledger.format_error else 0,
        "segment_format_error_reason": ledger.format_error,
    }
    for reason in FOCUS_FAIL_REASONS:
        out[f"focus_fail_{reason}"] = ledger.fail_counts.get(reason, 0)
    return out
