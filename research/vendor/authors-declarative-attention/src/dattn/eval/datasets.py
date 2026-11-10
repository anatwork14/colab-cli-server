"""Dataset registry, the context filter and dac-exact row selection.

Row selection is a **reproduction of a measurement instrument**, not a design choice.
The reference eval_datasets.py:230-243 does, per subset:

    filter(rubric non-blank)
      -> filter(min_tokens < model-tokenizer context tokens <= max_tokens)
      -> shuffle(seed=42)
      -> select(range(n))

The order matters. Filtering **before** shuffling means the surviving pool size decides
the permutation, so a one-row change in the pool re-permutes the whole prefix. dattn keeps
dac's order exactly so the 1611 rows per model are the same 1611 rows, and
`dattn parity` can align by id.

FSM fixtures were regenerated from dac's recorded responses, never copied.
See `tests/fixtures/fsm/README.md` and `docs/PROVENANCE.md` for provenance.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

__all__ = [
    "ALL_SUBSETS",
    "DEFAULT_MIN_CONTEXT_TOKENS",
    "EVAL_SAMPLING_SEED",
    "HEADLINE_SUBSETS",
    "KNOWN_STALE_SUBSETS",
    "MODEL_MAX_LEN",
    "REGISTRY",
    "RESERVED_NEW_TOKENS",
    "RESERVED_OVERHEAD",
    "EvalRow",
    "SelectionSpec",
    "SubsetSelection",
    "count_context_tokens",
    "derive_max_context_tokens",
    "load_subset",
    "resolve_subset_key",
    "resolve_subsets",
    "row_id",
    "select_rows",
    "write_rows",
]

# --------------------------------------------------------------------------------------
# Registry. Mirrors dac's two wired registries, subset -> split, character for character
# (`eval_datasets.py:40-99`). Both reference implementations agree on this map.
# --------------------------------------------------------------------------------------

_V4_1: Final[dict[str, str]] = {
    "longbench_v2_code_repo": "test",
    "longbench_v2_dialogue_history": "test",
    "longbench_v2_in_context_learning": "test",
    "longbench_v2_multidoc_qa": "validation",
    "longbench_v2_singledoc_qa": "validation",
    "longbench_v2_structured_data": "test",
    "loogle_longdep_qa": "validation",
    "loogle_shortdep_cloze": "test",
    "loogle_shortdep_qa": "validation",
    "loogle_summarization": "validation",
    "ruler_cwe": "test",
    "ruler_fwe": "test",
    "ruler_niah_multikey_1": "test",
    "ruler_niah_multikey_2": "test",
    "ruler_niah_multikey_3": "test",
    "ruler_niah_multiquery": "test",
    "ruler_niah_multivalue": "test",
    "ruler_niah_single_1": "test",
    "ruler_niah_single_2": "test",
    "ruler_niah_single_3": "test",
    "ruler_qa_1": "test",
    "ruler_qa_2": "test",
    "ruler_vt": "test",
    "zeroscrolls_book_sum_sort": "test",
    "zeroscrolls_quality": "test",
    "zeroscrolls_space_digest": "test",
    "zeroscrolls_squality": "test",
    "zeroscrolls_summ_screen_fd": "test",
}

_V3_1: Final[dict[str, str]] = {
    "2wikimqa": "validation",
    "gov_report": "validation",
    "hotpotqa": "test",
    "hotpotqa_e": "test",
    "multi_news": "validation",
    "multi_news_e": "validation",
    "multifieldqa_en": "validation",
    "musique": "test",
    "narrativeqa": "test",
    "qasper": "test",
    "qmsum": "validation",
    "samsum": "validation",
    "triviaqa": "validation",
    "triviaqa_e": "validation",
}

#: `{dataset key: {"repo_id": ..., "subsets": {subset: split}}}`.
REGISTRY: Final[dict[str, dict[str, Any]]] = {
    "longbench_synthetic_v4_1": {"repo_id": "dac-research/longbench_synthetic_v4_1", "subsets": _V4_1},
    "longbench_synthetic_v3_1": {"repo_id": "dac-research/longbench_synthetic_v3_1", "subsets": _V3_1},
}

#: Every fully-qualified subset key, in dac's iteration order.
ALL_SUBSETS: Final[tuple[str, ...]] = tuple(
    f"{name}_{subset}" for name, group in REGISTRY.items() for subset in group["subsets"]
)

#: The 15 headline subsets, in the order of dac's `full_eval.sh:21` `SUBSETS`. These are
#: the subsets behind every number in the paper's main table. Two of them
#: (`longbench_v2_code_repo` n=18, `zeroscrolls_quality` n=18) are smaller than the cap,
#: and `longbench_v2_dialogue_history` has 39 rows, so a 128-cap run is 1611 rows, not
#: 15 * 128.
HEADLINE_SUBSETS: Final[tuple[str, ...]] = (
    "longbench_synthetic_v4_1_ruler_niah_single_1",
    "longbench_synthetic_v4_1_ruler_niah_single_2",
    "longbench_synthetic_v4_1_ruler_niah_single_3",
    "longbench_synthetic_v4_1_ruler_niah_multikey_2",
    "longbench_synthetic_v4_1_ruler_niah_multikey_3",
    "longbench_synthetic_v3_1_qmsum",
    "longbench_synthetic_v4_1_longbench_v2_multidoc_qa",
    "longbench_synthetic_v4_1_loogle_summarization",
    "longbench_synthetic_v4_1_longbench_v2_code_repo",
    "longbench_synthetic_v4_1_zeroscrolls_quality",
    "longbench_synthetic_v4_1_longbench_v2_singledoc_qa",
    "longbench_synthetic_v4_1_longbench_v2_dialogue_history",
    "longbench_synthetic_v4_1_loogle_longdep_qa",
    "longbench_synthetic_v4_1_loogle_shortdep_cloze",
    "longbench_synthetic_v4_1_zeroscrolls_space_digest",
)

#: The 128-cap headline row counts per subset, as they are on disk in
#: the reference 128-cap outputs. Unit tests use these as a structural check
#: that does not need the tokenizer.
HEADLINE_128CAP_COUNTS: Final[dict[str, int]] = {
    "longbench_synthetic_v3_1_qmsum": 128,
    "longbench_synthetic_v4_1_longbench_v2_code_repo": 18,
    "longbench_synthetic_v4_1_longbench_v2_dialogue_history": 39,
    "longbench_synthetic_v4_1_longbench_v2_multidoc_qa": 128,
    "longbench_synthetic_v4_1_longbench_v2_singledoc_qa": 128,
    "longbench_synthetic_v4_1_loogle_longdep_qa": 128,
    "longbench_synthetic_v4_1_loogle_shortdep_cloze": 128,
    "longbench_synthetic_v4_1_loogle_summarization": 128,
    "longbench_synthetic_v4_1_ruler_niah_multikey_2": 128,
    "longbench_synthetic_v4_1_ruler_niah_multikey_3": 128,
    "longbench_synthetic_v4_1_ruler_niah_single_1": 128,
    "longbench_synthetic_v4_1_ruler_niah_single_2": 128,
    "longbench_synthetic_v4_1_ruler_niah_single_3": 128,
    "longbench_synthetic_v4_1_zeroscrolls_quality": 18,
    "longbench_synthetic_v4_1_zeroscrolls_space_digest": 128,
}

#: Subsets that turn up in a recorded dac tree and are NOT part of the paper's 15. The
#: DA-nm backup arm (`dac_chunk_multi_turn_tool_v1_no_mask`, both models) carries a
#: leftover `loogle_shortdep_qa` at 128 rows, so that arm is 1739 rows over 16 subsets
#: while its own `summary.json` reports `n_samples_attempted: 1611`. Its published
#: accuracy is therefore not the paper's number. The origin of the extra subset has not
#: been identified and it does not match the test-run subset list. Anything that quotes a
#: paper number MUST filter to `HEADLINE_SUBSETS` first, which is why `dattn analyse` and
#: `dattn parity` now do so by default (`--all-subsets` opts out).
KNOWN_STALE_SUBSETS: Final[dict[str, str]] = {
    "longbench_synthetic_v4_1_loogle_shortdep_qa": (
        "leftover in outputs_128cap_backup_260703/*/dac_chunk_multi_turn_tool_v1_no_mask, "
        "128 rows, making that arm 1739 rows over 16 subsets against a summary.json that "
        "claims n_samples_attempted 1611"
    ),
}


@dataclass(frozen=True, slots=True)
class SubsetSelection:
    """What a readout tool decided to count, and what it is dropping to get there."""

    #: The subsets to keep, or `None` for "everything present".
    subsets: tuple[str, ...] | None
    #: Subsets on disk that the restriction drops.
    dropped: tuple[str, ...]
    #: Subsets on disk that `KNOWN_STALE_SUBSETS` names, dropped or not.
    stale: tuple[str, ...]
    #: True when the result is the paper's population.
    restricted: bool
    #: Lines a caller must print. Never empty when rows are dropped.
    warnings: tuple[str, ...]


def resolve_subsets(
    present: Iterable[str],
    *,
    all_subsets: bool = False,
    explicit: Sequence[str] | None = None,
) -> SubsetSelection:
    """Pick the subsets a readout counts, and say out loud what that leaves behind.

    The default is the paper's 15. A cell that does not carry all 15 is NOT silently
    filtered down to whatever overlap exists, because the macro of a partial subset list
    is not the paper's number and quoting it as one is the failure mode this guards. In
    that case every subset present is kept and the caller is told the macro is not
    comparable.
    """
    have = set(present)
    stale = tuple(sorted(have & set(KNOWN_STALE_SUBSETS)))
    notes: list[str] = []

    if explicit is not None:
        keep: tuple[str, ...] | None = tuple(explicit)
        dropped = tuple(sorted(have - set(keep)))
        restricted = True
    elif all_subsets:
        keep, dropped, restricted = None, (), False
    elif set(HEADLINE_SUBSETS) <= have:
        keep = HEADLINE_SUBSETS
        dropped = tuple(sorted(have - set(HEADLINE_SUBSETS)))
        restricted = True
    else:
        keep, dropped, restricted = None, (), False
        missing = sorted(set(HEADLINE_SUBSETS) - have)
        shown = ", ".join(missing[:3]) + (f", and {len(missing) - 3} more" if len(missing) > 3 else "")
        notes.append(
            f"WARNING: {len(missing)} of the {len(HEADLINE_SUBSETS)} headline subsets are absent, so "
            f"nothing was dropped and this is NOT the paper's population (missing: {shown})"
        )

    if dropped:
        notes.append(
            f"restricted to the {len(keep or ())} headline subsets, "
            f"dropped {len(dropped)} subset(s) outside them: {', '.join(dropped)}"
        )
    for name in stale:
        notes.append(f"WARNING: stale subset {name!r} present: {KNOWN_STALE_SUBSETS[name]}")
    return SubsetSelection(subsets=keep, dropped=dropped, stale=stale, restricted=restricted, warnings=tuple(notes))


EVAL_SAMPLING_SEED: Final[int] = 42
DEFAULT_MIN_CONTEXT_TOKENS: Final[int] = 4096
RESERVED_NEW_TOKENS: Final[int] = 8192
RESERVED_OVERHEAD: Final[int] = 4096

#: dac's `model_specs.MODEL_MAX_LEN`, the only input to the upper filter bound.
MODEL_MAX_LEN: Final[dict[str, int]] = {
    "Qwen/Qwen3-4B": 40960,
    "Qwen/Qwen3-32B": 40960,
    "Qwen/Qwen3.5-4B": 262144,
    "Qwen/Qwen3.5-9B": 262144,
    "Qwen/Qwen3.5-27B": 262144,
    "Qwen/Qwen3.5-35B-A3B": 262144,
    "Qwen/Qwen3.5-397B-A17B": 262144,
    "Qwen/Qwen3.6-27B": 262144,
    "Qwen/Qwen3.6-27B-FP8": 262144,
    "Qwen/Qwen3.6-35B-A3B": 262144,
    "google/gemma-4-E2B": 131072,
    "google/gemma-4-E4B": 131072,
    "google/gemma-4-E4B-it": 131072,
    "google/gemma-4-12B-it": 262144,
    "google/gemma-4-31B": 262144,
    "google/gemma-4-31B-it": 262144,
    "google/gemma-4-26B-A4B-it": 262144,
    "meituan-longcat/LongCat-Flash-Lite": 327680,
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": 262144,
    "openai/gpt-oss-20b": 131072,
}


def derive_max_context_tokens(model_id: str) -> int:
    """`max_model_len - 8192 - 4096`, or 0 (no upper bound) for an unknown model.

    262144 - 8192 - 4096 = 249856 for both headline models, which is the bound every
    recorded headline run used.
    """
    max_len = MODEL_MAX_LEN.get(model_id)
    return 0 if max_len is None else max_len - RESERVED_NEW_TOKENS - RESERVED_OVERHEAD


def resolve_subset_key(name: str) -> str:
    """Accept either a bare subset name or a fully-qualified `{dataset}_{subset}` key."""
    if name in ALL_SUBSETS:
        return name
    matches = [f"{ds}_{sub}" for ds, group in REGISTRY.items() for sub in group["subsets"] if sub == name]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise KeyError(f"unknown subset {name!r}")
    raise KeyError(f"ambiguous subset {name!r}, matches {matches}")


def row_id(subset: str, row_index: int) -> str:
    """The alignment key for evaluation. dac's identity is `(subset, row_index)`.

    `row_index` is a **post-shuffle prefix index, not a row identity**: it is
    only meaningful together with (seed, filter bounds, tokenizer). `SelectionSpec` is
    persisted next to every output tree so an id can never be silently reinterpreted.
    """
    return f"{subset}#{row_index}"


@dataclass(frozen=True, slots=True)
class SelectionSpec:
    """Everything that decides which rows a run sees. Persisted with the outputs."""

    model_id: str
    subsets: tuple[str, ...]
    n_per_subset: int
    seed: int
    min_context_tokens: int
    max_context_tokens: int
    datasets_version: str = ""
    tokenizer_revision: str = ""

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "subsets": list(self.subsets)}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> SelectionSpec:
        return cls(
            model_id=str(d["model_id"]),
            subsets=tuple(str(s) for s in d.get("subsets", ())),  # type: ignore[union-attr]
            n_per_subset=int(d["n_per_subset"]),  # type: ignore[arg-type]
            seed=int(d.get("seed", EVAL_SAMPLING_SEED)),  # type: ignore[arg-type]
            min_context_tokens=int(d.get("min_context_tokens", 0)),  # type: ignore[arg-type]
            max_context_tokens=int(d.get("max_context_tokens", 0)),  # type: ignore[arg-type]
            datasets_version=str(d.get("datasets_version", "")),
            tokenizer_revision=str(d.get("tokenizer_revision", "")),
        )


@dataclass(frozen=True, slots=True)
class EvalRow:
    """One selected dataset row, in dac's `_samples_from_hf` shape plus an id."""

    subset: str
    row_index: int
    split: str
    context: str
    question: str
    answer: str
    rubric: str
    raw_context_tokens: int | None = None

    @property
    def id(self) -> str:
        return row_id(self.subset, self.row_index)

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, **asdict(self)}

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> EvalRow:
        raw = d.get("raw_context_tokens")
        return cls(
            subset=str(d["subset"]),
            row_index=int(d["row_index"]),  # type: ignore[arg-type]
            split=str(d.get("split", "")),
            context=str(d.get("context", d.get("raw_context", ""))),
            question=str(d.get("question", "")),
            answer=str(d.get("answer", "")),
            rubric=str(d.get("rubric", "") or ""),
            raw_context_tokens=None if raw is None else int(raw),  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------------------
# Filtering. Both predicates are dac's, spelled out so a reader can check them against
# `eval_datasets.py:134-182` without leaving this file.
# --------------------------------------------------------------------------------------


def has_rubric(row: Mapping[str, Any]) -> bool:
    """dac `filter_by_rubric`: a present, non-None, non-whitespace `rubric`."""
    r = row.get("rubric")
    return bool(r) and bool(str(r).strip())


def count_context_tokens(tokenizer: Any, context: str) -> int:
    """dac counts with `add_special_tokens=False` on the SERVED model's tokenizer.

    Not the dataset's published `context_tokens` column. The published column is a
    different tokenizer's count and is 15-20% low for Gemma, which is exactly what makes
    a column-based filter select a different row set (`E 2c`).
    """
    return len(tokenizer(str(context), add_special_tokens=False)["input_ids"])


def within_bounds(n_tokens: int, *, min_tokens: int, max_tokens: int) -> bool:
    """dac's `_within`: keep iff `min_tokens < n <= max_tokens`, each bound optional.

    Note the asymmetry, which is dac's: the lower bound is strict, the upper is not.
    """
    if max_tokens > 0 and n_tokens > max_tokens:
        return False
    return not (min_tokens > 0 and n_tokens <= min_tokens)


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------


def select_rows(
    *,
    model_id: str,
    tokenizer: Any,
    subsets: Sequence[str] = HEADLINE_SUBSETS,
    n_per_subset: int = 128,
    seed: int = EVAL_SAMPLING_SEED,
    min_context_tokens: int = DEFAULT_MIN_CONTEXT_TOKENS,
    max_context_tokens: int | None = None,
    num_proc: int | None = None,
    progress: Callable[[str], None] | None = None,
) -> list[EvalRow]:
    """Reproduce dac's row selection exactly, subset by subset.

    `max_context_tokens=None` derives the bound from `model_id`, which is what dac does
    (`run_eval.py:1209-1219`). Pass `0` to disable the upper bound.

    Requires the `datasets` package, which is deliberately NOT a hard dependency of
    `dattn`: only this one function needs it, and 90% of the suite must run on a CPU box
    with no dataset cache. Install it for an eval run
    (`uv run --with datasets dattn eval ...`).
    """
    try:
        from datasets import Dataset, load_dataset
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "dattn.eval.datasets.select_rows needs the `datasets` package. "
            "Run it as `uv run --with datasets dattn eval ...`."
        ) from exc

    if max_context_tokens is None:
        max_context_tokens = derive_max_context_tokens(model_id)

    keys = [resolve_subset_key(s) for s in subsets]
    out: list[EvalRow] = []
    for key in keys:
        ds_name, subset = _split_key(key)
        split = REGISTRY[ds_name]["subsets"][subset]
        if progress:
            progress(f"loading {key} ({split})")
        ds: Dataset = load_dataset(REGISTRY[ds_name]["repo_id"], subset, split=split)

        # ---- dac's exact order: rubric -> tokens -> shuffle -> prefix select --------
        if "rubric" in ds.column_names:
            ds = ds.filter(has_rubric)
        if max_context_tokens > 0 or min_context_tokens > 0:
            if tokenizer is None:
                raise ValueError("a context-token filter needs the served model's tokenizer")
            ds = ds.filter(
                lambda r: within_bounds(
                    count_context_tokens(tokenizer, r["context"]),
                    min_tokens=min_context_tokens,
                    max_tokens=max_context_tokens,
                ),
                num_proc=num_proc,
            )
        pool_size = len(ds)
        ds = ds.shuffle(seed=seed)
        if n_per_subset >= 0:
            ds = ds.select(range(min(len(ds), n_per_subset)))
        if progress:
            progress(f"  {key}: pool={pool_size} selected={len(ds)}")

        for i in range(len(ds)):
            row = ds[i]
            out.append(
                EvalRow(
                    subset=key,
                    row_index=i,
                    split=split,
                    context=str(row["context"]),
                    question=str(row["question"]),
                    answer=str(row.get("reference_answer") or row.get("answer") or ""),
                    rubric=str(row.get("rubric", "") or ""),
                    raw_context_tokens=row.get("context_tokens"),
                )
            )
    return out


def _split_key(key: str) -> tuple[str, str]:
    for ds_name in REGISTRY:
        prefix = ds_name + "_"
        if key.startswith(prefix):
            return ds_name, key[len(prefix) :]
    raise KeyError(f"unknown subset key {key!r}")


# --------------------------------------------------------------------------------------
# On-disk row files. `load_subset` keeps the shared signature.
# --------------------------------------------------------------------------------------


def write_rows(path: Path, rows: Sequence[EvalRow], *, spec: SelectionSpec | None = None) -> None:
    """Write a JSONL row file, with the `SelectionSpec` as an optional header line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        if spec is not None:
            fh.write(json.dumps({"_selection_spec": spec.to_dict()}, separators=(",", ":")) + "\n")
        for row in rows:
            fh.write(json.dumps(row.to_dict(), separators=(",", ":")) + "\n")


def load_subset(path: Path, *, min_context_tokens: int = 0, limit: int | None = None) -> Iterator[Mapping]:
    """Stream a JSONL row file written by `write_rows`, skipping the spec header.

    `min_context_tokens` filters on the row's recorded `raw_context_tokens` and is a
    convenience for slicing an already-selected file. It is NOT dac's filter: dac's runs
    on the served tokenizer's count and must run before the shuffle, which is what
    `select_rows` does.
    """
    n = 0
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_selection_spec" in row:
                continue
            if min_context_tokens:
                tok = row.get("raw_context_tokens")
                if tok is not None and int(tok) <= min_context_tokens:
                    continue
            yield row
            n += 1
            if limit is not None and n >= limit:
                return
