"""Real task rows for the walltime bench, drawn the way `dattn eval` draws them.

Batches draw without replacement from each task's selected question set. The headline
macro gives each of the 15 tasks equal weight. Real contexts and questions matter:
a repeated filler string can leave DA fully dense and fail to exercise the mask.

Selection is delegated to `dattn.eval.datasets.select_rows`, never reimplemented, so the
bench and the accuracy pipeline see the same rows for the same `(model_id, seed, n)`. That
file is the single definition of "the 128-question subset" and a second copy of its
filter order (rubric, then context tokens, then shuffle, then prefix) would be a silent
divergence waiting to happen.

`dattn.eval` imports neither torch nor vllm, so importing this module does not change the
bench's import surface.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from dattn.eval.datasets import (
    ALL_SUBSETS,
    DEFAULT_MIN_CONTEXT_TOKENS,
    EVAL_SAMPLING_SEED,
    HEADLINE_SUBSETS,
    EvalRow,
    resolve_subset_key,
)

__all__ = [
    "ALL_SUBSETS",
    "DEFAULT_SUBSET_SIZE",
    "HEADLINE_SUBSETS",
    "SYNTHETIC_TASK",
    "TaskSample",
    "load_task",
    "load_tasks",
    "resolve_subsets",
    "samples_from_rows",
    "synthetic_task",
]

#: `Protocol.subset_size`, spelled here too so a caller that only has this module can
#: draw the paper's subset without importing the protocol.
DEFAULT_SUBSET_SIZE: Final[int] = 128

#: The label a filler cell is recorded under. Deliberately not the name of a real task:
#: a walltime cell drawn from filler text is a kernel measurement, not a task
#: measurement, and the reader must be able to see that in `meta_<cell>.json`.
SYNTHETIC_TASK: Final[str] = "synthetic_filler"


@dataclass(frozen=True, slots=True)
class TaskSample:
    """One drawn row: the real context AND the real question that goes with it.

    `sample_id` is `dattn.eval.datasets.row_id`, so a bench batch can be traced back to
    the exact accuracy-pipeline row it was drawn from.
    """

    subset: str
    sample_id: str
    context: str
    question: str

    def to_dict(self) -> dict[str, str]:
        return {
            "subset": self.subset,
            "sample_id": self.sample_id,
            "context": self.context,
            "question": self.question,
        }


def resolve_subsets(names: Sequence[str] | None) -> tuple[str, ...]:
    """`None` or `("headline",)` means the paper's 15, otherwise resolve each name.

    A bare subset name is accepted, exactly as `dattn eval` accepts it, so an operator
    does not have to type `longbench_synthetic_v4_1_` in front of everything.
    """
    if not names or tuple(names) == ("headline",):
        return tuple(HEADLINE_SUBSETS)
    return tuple(resolve_subset_key(str(n)) for n in names)


def samples_from_rows(rows: Iterable[EvalRow]) -> list[TaskSample]:
    """`EvalRow` to `TaskSample`, dropping the fields the bench must not look at.

    The bench never sees `answer` or `rubric`. It measures step time, and carrying the
    reference answer next to a timing batch is how a timing harness quietly turns into an
    accuracy harness with no judge.
    """
    return [
        TaskSample(subset=r.subset, sample_id=r.id, context=r.context, question=r.question)
        for r in rows
        if r.context and r.question
    ]


def _rows_from_file(path: Path) -> list[EvalRow]:
    """A pre-materialised selection: JSON list or JSONL of `EvalRow.to_dict()` rows.

    This exists so a box with `HF_DATASETS_OFFLINE=1` and no dataset cache can still run
    the real task arm: dump the selection once where the cache is, copy the file, run the
    bench against it. The rows are the eval pipeline's rows either way.
    """
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        raw: list[Any] = list(json.loads(text))
    else:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not raw:
        raise ValueError(f"{path}: no rows")
    if isinstance(raw[0], str):
        raise ValueError(
            f"{path}: this is a list of bare context strings, which has no question and cannot "
            "provoke a focus tag. Pass a selection dump of EvalRow dicts, or omit the file and "
            "let --subsets draw the real rows"
        )
    return [EvalRow.from_dict(d) for d in raw]


def load_tasks(
    subsets: Sequence[str] | None = None,
    *,
    model_id: str,
    tokenizer: Any = None,
    n: int = DEFAULT_SUBSET_SIZE,
    seed: int = EVAL_SAMPLING_SEED,
    min_context_tokens: int = DEFAULT_MIN_CONTEXT_TOKENS,
    max_context_tokens: int | None = None,
    rows_file: Path | str | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, list[TaskSample]]:
    """`{subset key: rows}` for every requested subset, in the requested order.

    One `select_rows` call for the whole suite, because it loads and filters each subset
    once and the bench then draws 10 batches per subset out of the result.
    """
    if rows_file is not None:
        rows = _rows_from_file(Path(rows_file))
        # A dump names its own tasks. Honour `subsets` as a filter when it was given
        # explicitly, and otherwise take the file's subsets in the order they appear, so
        # a pre-materialised selection does not have to be the headline 15.
        present = tuple(dict.fromkeys(r.subset for r in rows))
        keys = present if not subsets else tuple(s if s in present else resolve_subset_key(str(s)) for s in subsets)
    else:
        keys = resolve_subsets(subsets)
        from dattn.eval.datasets import select_rows

        rows = select_rows(
            model_id=model_id,
            tokenizer=tokenizer,
            subsets=keys,
            n_per_subset=n,
            seed=seed,
            min_context_tokens=min_context_tokens,
            max_context_tokens=max_context_tokens,
            progress=progress,
        )
    out: dict[str, list[TaskSample]] = {k: [] for k in keys}
    for sample in samples_from_rows(rows):
        out.setdefault(sample.subset, []).append(sample)
    empty = sorted(k for k in keys if not out.get(k))
    if empty:
        raise ValueError(
            f"no rows selected for {empty}: the bench cannot draw a batch for a task with no rows. "
            f"Requested {len(keys)} subsets at n={n}, seed={seed}, "
            f"min_context_tokens={min_context_tokens}"
        )
    return {k: out[k] for k in keys}


def load_task(
    subset: str,
    *,
    model_id: str,
    tokenizer: Any = None,
    n: int = DEFAULT_SUBSET_SIZE,
    seed: int = EVAL_SAMPLING_SEED,
    **kw: Any,
) -> list[TaskSample]:
    """One subset's rows. A thin front for `load_tasks` for tests and one-off cells."""
    by_task = load_tasks([subset], model_id=model_id, tokenizer=tokenizer, n=n, seed=seed, **kw)
    # Not `by_task[resolve_subset_key(subset)]`: a selection dump names its own tasks and
    # they need not be registry keys at all.
    return by_task[subset] if subset in by_task else next(iter(by_task.values()))


_FILLER: Final[str] = (
    "The municipal archive records that the harbour survey of that season was carried out on foot, "
    "that the tide tables were copied by hand into the ledger, and that the assistant surveyor noted "
    "the weather each morning before setting out. "
)


def synthetic_task(n: int, *, approx_tokens: int, seed: int = 0) -> list[TaskSample]:
    """`n` filler samples, for a smoke cell only.

    Filler measures the kernel against a KV length. It does NOT measure the mask, because
    the model never emits a focus tag on it, so a cell built from these samples runs the
    DA arm entirely at kept fraction 1. `run_protocol` refuses to
    use it for a production cell.
    """
    import random

    rng = random.Random(seed)
    words = _FILLER.split()
    repeats = max(1, approx_tokens // max(1, len(words)))
    out: list[TaskSample] = []
    for i in range(n):
        body = [f"Record {i}.{j}: " + " ".join(rng.sample(words, len(words))) for j in range(repeats)]
        out.append(
            TaskSample(
                subset=SYNTHETIC_TASK,
                sample_id=f"{SYNTHETIC_TASK}:{i}",
                context="\n\n".join(body),
                question="Summarise the archive in one sentence.",
            )
        )
    return out
