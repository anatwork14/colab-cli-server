"""Row selection, rollouts, judging, aggregation, replay and parity.

Consumes `dattn.fsm`, `dattn.prompt` and `dattn.engine.api`. Never `vllm` directly: the
eval client talks to a dattn-served engine over its OpenAI-compatible HTTP surface, which is
what keeps this package importable, and 90% of it testable, on a CPU box.

Optional third-party dependencies, each needed by exactly one entry point and imported
inside it: `datasets` (`select_rows`) and `transformers` (via `dattn.prompt`). The HTTP client,
parity self-check and eval aggregator use the standard library.
"""

from dattn.eval.aggregate import Aggregate, Cell, aggregate, load_cell
from dattn.eval.datasets import (
    HEADLINE_SUBSETS,
    EvalRow,
    SelectionSpec,
    load_subset,
    select_rows,
)
from dattn.eval.extract import Extraction, extract_answer
from dattn.eval.judge import Judgment, Verdict, parse_verdict, render_judge_prompt
from dattn.eval.parity import ParityReport, compare_trees
from dattn.eval.replay import replay
from dattn.eval.sampling import RolloutConfig
from dattn.eval.schema import ResponseRecord, RunMeta

__all__ = [
    "HEADLINE_SUBSETS",
    "Aggregate",
    "Cell",
    "EvalRow",
    "Extraction",
    "Judgment",
    "ParityReport",
    "ResponseRecord",
    "RolloutConfig",
    "RunMeta",
    "SelectionSpec",
    "Verdict",
    "aggregate",
    "compare_trees",
    "extract_answer",
    "load_cell",
    "load_subset",
    "parse_verdict",
    "render_judge_prompt",
    "replay",
    "select_rows",
]
