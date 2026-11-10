"""`StepRecord` schema, dtypes and the JSONL codec.

`StepRecord` itself is in `dattn.contracts`, because eval reads the records too.
This module owns the dtype table, the JSONL codec and the `kind` classifier.

`steps_<cell>.jsonl` stores one JSON line per engine iteration, losslessly round-tripping
through `StepRecord`. Writes are buffered and flushed off the step path (DESIGN.md
section 5.4).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Final

from dattn.contracts import StepRecord, StepRows

#: field -> dtype, exactly DESIGN.md section 5.2. Checked by a unit test against StepRecord.
DTYPES: Final[dict[str, str]] = {
    "step_id": "int64",
    "engine_step": "int64",
    "t_sched_ns": "int64",
    "t_ready_ns": "int64",
    "gpu_forward_ms": "float32",
    "gpu_sample_ms": "float32",
    "lp_host_us": "float32",
    "compact_host_us": "float32",
    "gpu_compact_ms": "float32",
    "kind": "str",
    "num_reqs": "int32",
    "num_reqs_padded": "int32",
    "cudagraph_replayed": "bool",
    "cudagraph_mode": "str",
    "runner_generation": "str",
    "dattn_block_size": "int32",
    "num_new_reqs": "int32",
    "num_ctx_tokens": "int32",
    "num_gen_tokens": "int32",
    "num_preempted": "int32",
    "kv_usage": "float32",
    "n_full_attn_groups": "int32",
    "n_compacted_groups": "int32",
    "fsm_lag": "int8",
    "host_bracket_valid": "bool",
    "drained_early": "bool",
}

#: `rows.*` are parallel fixed-width arrays indexed by batch slot.
ROW_DTYPES: Final[dict[str, str]] = {
    "req": "str",
    "mode": "int8",
    "seq_len": "int32",
    "att_blk": "int32",
    "att_tok": "int32",
    "out_pos": "int32",
    "since_adm": "int32",
}

#: `steps_<cell_id>.jsonl`, plus the two sidecars that are never merged into it.
STEP_FILE = "steps_{cell_id}.jsonl"
ENV_FILE = "env_{cell_id}.jsonl"
META_FILE = "meta_{cell_id}.json"

#: Serialisation is buffered this many lines and flushed off the step path (section 5.4).
FLUSH_LINES: Final[int] = 512

#: `kind` is one of these three and nothing else.
KINDS: Final[tuple[str, ...]] = ("decode", "mixed", "prefill")

__all__ = [
    "DTYPES",
    "ENV_FILE",
    "FLUSH_LINES",
    "KINDS",
    "META_FILE",
    "ROW_DTYPES",
    "STEP_FILE",
    "JsonlWriter",
    "StepRecord",
    "StepRows",
    "classify_step",
    "read_jsonl",
    "read_table",
    "write_jsonl",
    "write_table",
]


# --------------------------------------------------------------------------------------
# The classifier (DESIGN.md section 5.2)
# --------------------------------------------------------------------------------------


def classify_step(scheduler_output: Any, num_reqs: int) -> str:
    """`decode` / `mixed` / `prefill`, from the `SchedulerOutput` the worker already has.

    Never from `IterationStats.num_prompt_tokens`, which files intermediate
    chunked-prefill steps as decode . The predicate is vLLM's own
    `CachedRequestData.is_context_phase` (`N:v1/core/sched/output.py:183-185`), which is
    `num_output_tokens == 0` and therefore covers the one-token final prefill chunk.

    `scheduler_output` must expose `scheduled_new_reqs`, `scheduled_cached_reqs`
    (with `req_ids` and `is_context_phase`) and `total_num_scheduled_tokens`. Anything
    else raises, because a classifier that guesses is the defect this replaces.
    """
    so = scheduler_output
    for attr in ("scheduled_new_reqs", "scheduled_cached_reqs", "total_num_scheduled_tokens"):
        if not hasattr(so, attr):
            raise NotImplementedError(
                f"classify_step needs a SchedulerOutput-shaped object with .{attr}, got {type(so).__name__}"
            )
    cached = so.scheduled_cached_reqs
    n_new = len(so.scheduled_new_reqs)
    ctx_cached = sum(1 for rid in cached.req_ids if cached.is_context_phase(rid))
    n_ctx = n_new + ctx_cached
    n_gen = int(num_reqs) - n_ctx
    if n_ctx == 0 and int(so.total_num_scheduled_tokens) == int(num_reqs):
        return "decode"
    if n_gen <= 0:
        return "prefill"
    return "mixed"


# --------------------------------------------------------------------------------------
# JSONL
# --------------------------------------------------------------------------------------


class JsonlWriter:
    """Append-only buffered JSONL writer. `flush()` is the only thing that touches disk."""

    __slots__ = ("_buf", "_fh", "_n", "flush_lines", "path")

    def __init__(self, path: Path | str, *, flush_lines: int = FLUSH_LINES) -> None:
        self.path = Path(path)
        self.flush_lines = int(flush_lines)
        self._buf: list[str] = []
        self._n = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, record: StepRecord) -> None:
        self._buf.append(record.to_json())
        self._n += 1
        if len(self._buf) >= self.flush_lines:
            self.flush()

    def extend(self, records: Iterable[StepRecord]) -> None:
        for rec in records:
            self.write(rec)

    @property
    def count(self) -> int:
        return self._n

    def flush(self) -> None:
        if self._buf:
            self._fh.write("\n".join(self._buf) + "\n")
            self._buf.clear()
        self._fh.flush()

    def close(self) -> int:
        self.flush()
        self._fh.close()
        return self._n

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def write_jsonl(path: Path | str, records: Iterable[StepRecord]) -> int:
    """Append records, buffered `FLUSH_LINES` lines, flushed off the step path."""
    with JsonlWriter(path) as w:
        w.extend(records)
        return w.count


def read_jsonl(path: Path | str) -> Iterator[StepRecord]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield StepRecord.from_json(line)


def write_table(path: Path | str, records: Sequence[StepRecord]) -> int:
    """Write a JSONL step table, rejecting unsupported suffixes."""
    p = Path(path)
    if p.suffix == ".jsonl":
        return write_jsonl(p, records)
    raise ValueError(f"unknown step-table suffix {p.suffix!r}: expected .jsonl")


def read_table(path: Path | str) -> list[StepRecord]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return list(read_jsonl(p))
    raise ValueError(f"unknown step-table suffix {p.suffix!r}: expected .jsonl")
