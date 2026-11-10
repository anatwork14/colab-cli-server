"""Step-level measurement.

Consumes `dattn.engine.api` only. It must not import `dattn.engine.builder` or
`dattn.engine.state`: the coupling is the one module-level `_REC` slot that
`dattn.engine.state` reads and `dattn.metrics.hooks` writes.
"""

from dattn.contracts import StepRecord, StepRows
from dattn.metrics.hooks import install, installed, uninstall
from dattn.metrics.record import (
    JsonlWriter,
    classify_step,
    read_jsonl,
    read_table,
    write_jsonl,
    write_table,
)
from dattn.metrics.recorder import AUDIT_EVERY, DEFAULT_RING, StepFacts, StepRecorder

__all__ = [
    "AUDIT_EVERY",
    "DEFAULT_RING",
    "JsonlWriter",
    "StepFacts",
    "StepRecord",
    "StepRecorder",
    "StepRows",
    "classify_step",
    "install",
    "installed",
    "read_jsonl",
    "read_table",
    "uninstall",
    "write_jsonl",
    "write_table",
]
