"""Mask-versus-dense self-check of row identity and rendered prompt hashes."""

from dataclasses import asdict, dataclass
from pathlib import Path

from dattn.eval.schema import ResponseRecord, iter_jsonl


def load_tree(directory):
    rows = {}
    for doc in iter_jsonl(Path(directory) / "responses.jsonl"):
        row = ResponseRecord.from_dict(doc)
        if row.error or row.transport_error:
            continue
        if row.key in rows:
            raise ValueError(f"duplicate response key {row.key} in {directory}")
        rows[row.key] = row
    return rows


@dataclass
class ParityReport:
    n_rows: int
    n_left_only: int
    n_right_only: int
    faults: list[str]
    mode: str = "mask"

    @property
    def passed(self):
        return bool(self.n_rows) and not (self.n_left_only or self.n_right_only or self.faults)

    def to_dict(self):
        return {**asdict(self), "passed": self.passed}

    def render(self):
        return (
            f"mask identity/prompt self-check: {self.n_rows} aligned, "
            f"{self.n_left_only} left-only, {self.n_right_only} right-only\n"
            + "\n".join(self.faults)
            + ("\nPASS" if self.passed else "\nFAIL")
        )


def compare_trees(left, right, *, mode="mask"):
    """No claims about sampled text, metrics or correctness follow from this check."""
    if mode != "mask":
        raise ValueError("only mask mode is supported")
    a, b = load_tree(left), load_tree(right)
    shared = sorted(a.keys() & b.keys())
    faults = []
    for key in shared:
        x, y = a[key], b[key]
        if (x.question, x.rubric) != (y.question, y.rubric):
            faults.append(f"{key}: row identity differs")
        if not x.prompt_sha256 or not y.prompt_sha256 or x.prompt_sha256 != y.prompt_sha256:
            faults.append(f"{key}: prompt hash missing or different")
    return ParityReport(len(shared), len(a.keys() - b.keys()), len(b.keys() - a.keys()), faults)
