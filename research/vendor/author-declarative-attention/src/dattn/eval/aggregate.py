"""Macro subset accuracy and pooled micro accuracy over the same answer rows."""

from dataclasses import asdict, dataclass, field

from dattn.eval.judge import Judgment
from dattn.eval.persistence import failed
from dattn.eval.schema import ResponseRecord, iter_jsonl


@dataclass(frozen=True)
class Outcome:
    id: str
    subset: str
    correct: bool | None
    format_correct: bool
    judged: bool

    @property
    def unparsable(self):
        return self.judged and self.correct is None


@dataclass
class Cell:
    model: str
    arm: str
    outcomes: list[Outcome] = field(default_factory=list)
    n_errored: int = 0

    def subsets(self):
        return sorted({o.subset for o in self.outcomes})


def load_cell(responses_path, judgments_path=None, *, model="", arm="", subsets=None):
    """Join unique rollout and judgment keys. Failed transports are not model answers.

    Judge resume atomically removes transport failures before retrying. This is not
    a last-record-wins log, so duplicate judgments remain an error.
    """
    verdicts = {}
    if judgments_path is not None and judgments_path.exists():
        for doc in iter_jsonl(judgments_path):
            j = Judgment.from_dict(doc)
            if j.key in verdicts:
                raise ValueError(f"duplicate judgment for {j.key} in {judgments_path}")
            verdicts[j.key] = j
    seen, outcomes, errors = set(), [], 0
    for doc in iter_jsonl(responses_path):
        r = ResponseRecord.from_dict(doc)
        if subsets is not None and r.subset not in subsets:
            continue
        if failed(doc):
            errors += 1
            continue
        if r.key in seen:
            raise ValueError(f"duplicate response for {r.key} in {responses_path}")
        seen.add(r.key)
        j = verdicts.get(r.key)
        outcomes.append(Outcome(r.id, r.subset, None if j is None else j.correct, r.format_correct, j is not None))
    if subsets is None and (orphans := verdicts.keys() - seen):
        raise ValueError(f"judgments have no response: {sorted(orphans)[:3]}")
    return Cell(model, arm, outcomes, errors)


@dataclass
class Aggregate:
    model: str
    arm: str
    n: int
    n_subsets: int
    macro: float
    micro: float
    format_rate: float
    n_correct: int
    n_format_fail: int
    n_judged: int
    n_unparsable: int
    n_errored: int
    per_subset: dict

    def to_dict(self):
        return asdict(self)


def _rate(n, d):
    return n / d if d else float("nan")


def aggregate(cell, *, subsets=None):
    """Unjudged, unparsable and format-failed answers remain in the denominator."""
    rows = [o for o in cell.outcomes if subsets is None or o.subset in subsets]
    buckets = {}
    for row in rows:
        buckets.setdefault(row.subset, []).append(row)
    per_subset = {}
    for name, bucket in sorted(buckets.items()):
        correct = sum(o.correct is True for o in bucket)
        per_subset[name] = {
            "n": len(bucket),
            "n_correct": correct,
            "accuracy": correct / len(bucket),
            "n_format_fail": sum(not o.format_correct for o in bucket),
            "n_unparsable": sum(o.unparsable for o in bucket),
        }
    n = len(rows)
    n_correct = sum(o.correct is True for o in rows)
    n_format_fail = sum(not o.format_correct for o in rows)
    return Aggregate(
        model=cell.model,
        arm=cell.arm,
        n=n,
        n_subsets=len(buckets),
        macro=_rate(sum(v["accuracy"] for v in per_subset.values()), len(buckets)),
        micro=_rate(n_correct, n),
        format_rate=_rate(n - n_format_fail, n),
        n_correct=n_correct,
        n_format_fail=n_format_fail,
        n_judged=sum(o.judged for o in rows),
        n_unparsable=sum(o.unparsable for o in rows),
        n_errored=cell.n_errored,
        per_subset=per_subset,
    )


def format_table(rows):
    header = f"{'model':<26} {'arm':<24} {'n':>6} {'macro':>7} {'micro':>7} {'fmt_fail':>8} {'unpar':>6} {'errors':>6}"
    lines = [header, "-" * len(header)]
    for a in rows:
        lines.append(
            f"{a.model:<26} {a.arm:<24} {a.n:>6} {100 * a.macro:>7.2f} {100 * a.micro:>7.2f} "
            f"{a.n_format_fail:>8} {a.n_unparsable:>6} {a.n_errored:>6}"
        )
    return "\n".join(lines)
