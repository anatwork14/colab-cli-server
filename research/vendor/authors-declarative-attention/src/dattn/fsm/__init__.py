"""The declarative FSM. Pure Python and stdlib only: no torch, no vllm, no transformers.

The specification is the author's FSM semantics notes, which established by
execution that the two prior implementations agree on 56 adversarial cases x 5 tokenizers
x 3 layouts plus 3800 fuzz cases. That behaviour, not either codebase, is the spec.
"""

from dattn.fsm.layout import MAX_SEGMENTS, validate_layout
from dattn.fsm.machine import DeclarativeFSM
from dattn.fsm.mask import (
    DENSE,
    LOCAL_WINDOW_TOKENS,
    base_mask,
    compose_mask,
    focus_mask,
    local_window,
    round_to_blocks,
)
from dattn.fsm.metrics import METRIC_KEYS, Ledger, full_metrics
from dattn.fsm.tags import FOCUS_FAIL_REASONS, parse_focus_ids, scan_tail

__all__ = [
    "DENSE",
    "FOCUS_FAIL_REASONS",
    "LOCAL_WINDOW_TOKENS",
    "MAX_SEGMENTS",
    "METRIC_KEYS",
    "DeclarativeFSM",
    "Ledger",
    "base_mask",
    "compose_mask",
    "focus_mask",
    "full_metrics",
    "local_window",
    "parse_focus_ids",
    "round_to_blocks",
    "scan_tail",
    "validate_layout",
]
