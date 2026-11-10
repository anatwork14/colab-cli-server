"""Row selection: the registry, the filter bounds, and the recorded dac ids.

The expensive half of row selection (load 15 subsets, tokenize ~15k long contexts with
the served model's tokenizer, shuffle, prefix-select) needs the HF cache, the `datasets`
package and about ten minutes, so it lives behind `$DATTN_DAC_OUTPUTS` +
`$DATTN_SELECTION_ROWS`. The cheap half, which is everything that can go wrong without a
GPU or a download, is pinned here unconditionally:

- the registry and the 15-subset whitelist, against the ids actually on disk in dac's
  `outputs_128cap_backup_260703`;
- the filter bounds, which are the arithmetic that decides the pool and therefore the
  whole permutation;
- the asymmetric `min < n <= max` predicate, which is dac's and is easy to "fix" by
  accident.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dattn.eval.datasets import (
    ALL_SUBSETS,
    DEFAULT_MIN_CONTEXT_TOKENS,
    HEADLINE_128CAP_COUNTS,
    HEADLINE_SUBSETS,
    KNOWN_STALE_SUBSETS,
    REGISTRY,
    EvalRow,
    SelectionSpec,
    derive_max_context_tokens,
    has_rubric,
    load_subset,
    resolve_subset_key,
    resolve_subsets,
    row_id,
    within_bounds,
    write_rows,
)

DAC_ARMS = ("dac_chunk_multi_turn_tool_v1", "vanilla_v3")


def test_the_registry_has_dacs_28_plus_14_subsets() -> None:
    assert len(REGISTRY["longbench_synthetic_v4_1"]["subsets"]) == 28
    assert len(REGISTRY["longbench_synthetic_v3_1"]["subsets"]) == 14
    assert len(ALL_SUBSETS) == 42
    # The two registries share no subset name, which is what lets a bare name resolve.
    v4 = set(REGISTRY["longbench_synthetic_v4_1"]["subsets"])
    v3 = set(REGISTRY["longbench_synthetic_v3_1"]["subsets"])
    assert not v4 & v3


def test_the_headline_list_is_15_subsets_and_they_all_exist() -> None:
    assert len(HEADLINE_SUBSETS) == 15
    assert len(set(HEADLINE_SUBSETS)) == 15
    for key in HEADLINE_SUBSETS:
        assert key in ALL_SUBSETS
    assert set(HEADLINE_SUBSETS) == set(HEADLINE_128CAP_COUNTS)


def test_the_128cap_row_counts_sum_to_1611() -> None:
    """Not 15 * 128: two subsets have 18 rows and one has 39, and they are cap-invariant."""
    assert sum(HEADLINE_128CAP_COUNTS.values()) == 1611


def test_resolve_accepts_bare_and_qualified_names() -> None:
    assert resolve_subset_key("qmsum") == "longbench_synthetic_v3_1_qmsum"
    assert resolve_subset_key("longbench_synthetic_v3_1_qmsum") == "longbench_synthetic_v3_1_qmsum"
    with pytest.raises(KeyError):
        resolve_subset_key("no_such_subset")


def test_the_upper_filter_bound_is_249856_for_both_headline_models() -> None:
    """262144 - 8192 reserved new tokens - 4096 overhead. The bound every run used."""
    assert derive_max_context_tokens("google/gemma-4-31B-it") == 249856
    assert derive_max_context_tokens("Qwen/Qwen3.6-27B") == 249856
    assert derive_max_context_tokens("google/gemma-4-E4B-it") == 118784
    assert derive_max_context_tokens("not/a-model") == 0


def test_the_context_filter_is_strict_below_and_inclusive_above() -> None:
    """dac's asymmetry: `min < n <= max`. Changing either end re-permutes everything."""
    lo, hi = DEFAULT_MIN_CONTEXT_TOKENS, 249856
    assert not within_bounds(lo, min_tokens=lo, max_tokens=hi)
    assert within_bounds(lo + 1, min_tokens=lo, max_tokens=hi)
    assert within_bounds(hi, min_tokens=lo, max_tokens=hi)
    assert not within_bounds(hi + 1, min_tokens=lo, max_tokens=hi)
    # A bound of 0 disables that end.
    assert within_bounds(1, min_tokens=0, max_tokens=0)


def test_the_rubric_filter_drops_blank_and_missing() -> None:
    assert has_rubric({"rubric": "x"})
    assert not has_rubric({"rubric": "   "})
    assert not has_rubric({"rubric": None})
    assert not has_rubric({})


def test_row_id_is_the_alignment_key() -> None:
    assert row_id("s", 7) == "s#7"
    assert EvalRow(subset="s", row_index=7, split="test", context="c", question="q", answer="a", rubric="r").id == "s#7"


def test_selection_spec_round_trips() -> None:
    spec = SelectionSpec(
        model_id="Qwen/Qwen3.6-27B",
        subsets=HEADLINE_SUBSETS,
        n_per_subset=128,
        seed=42,
        min_context_tokens=4096,
        max_context_tokens=249856,
    )
    assert SelectionSpec.from_dict(json.loads(json.dumps(spec.to_dict()))) == spec


def test_write_and_load_subset_round_trip(tmp_path: Path) -> None:
    rows = [
        EvalRow(subset="s", row_index=i, split="test", context="c", question=f"q{i}", answer="a", rubric="r")
        for i in range(3)
    ]
    spec = SelectionSpec("m", ("s",), 3, 42, 4096, 249856)
    path = tmp_path / "rows.jsonl"
    write_rows(path, rows, spec=spec)
    got = list(load_subset(path))
    assert [r["question"] for r in got] == ["q0", "q1", "q2"]  # the spec header is skipped
    assert [r["id"] for r in got] == ["s#0", "s#1", "s#2"]
    assert len(list(load_subset(path, limit=2))) == 2


# --------------------------------------------------------------------------------------
# Against the recorded dac ids
# --------------------------------------------------------------------------------------


def _dac_ids(root: Path, model_key: str, arm: str) -> set[tuple[str, int]]:
    ids: set[tuple[str, int]] = set()
    with (root / model_key / arm / "responses.jsonl").open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                ids.add((r["subset"], int(r["row_index"])))
    return ids


@pytest.mark.parametrize("model_key", ["gemma_4_31b_it", "qwen3.6_27b"])
def test_the_headline_subsets_and_counts_match_the_recorded_dac_trees(dac_outputs: Path, model_key: str) -> None:
    """The 15 subsets and the 1611 rows are read off disk, not asserted from memory."""
    root = dac_outputs / "outputs_128cap_backup_260703"
    if not root.is_dir():
        pytest.skip(f"no 128-cap backup under {dac_outputs}")
    for arm in DAC_ARMS:
        ids = _dac_ids(root, model_key, arm)
        counts: dict[str, int] = {}
        for subset, _ in ids:
            counts[subset] = counts.get(subset, 0) + 1
        assert counts == HEADLINE_128CAP_COUNTS
        assert len(ids) == 1611
        assert {(s, i) for s, i in ids if i >= 128} == set()


def test_selected_rows_reproduce_the_recorded_dac_ids(dac_outputs: Path) -> None:
    """The full check: `select_rows` output vs dac's ids, 100% overlap.

    Selection needs `datasets`, the HF cache and about five minutes per model, so this
    runs against a file `$DATTN_SELECTION_ROWS` that `dattn eval --select-only` wrote.
    `docs/LANE_D.md` records the command and the result.
    """
    env = os.environ.get("DATTN_SELECTION_ROWS")
    if not env:
        pytest.skip("set DATTN_SELECTION_ROWS=<rows.jsonl written by `dattn eval --select-only`>")
    model_key = os.environ.get("DATTN_SELECTION_MODEL_KEY", "qwen3.6_27b")
    mine = {(r["subset"], int(r["row_index"])): r for r in load_subset(Path(env))}
    theirs = _dac_ids(dac_outputs / "outputs_128cap_backup_260703", model_key, DAC_ARMS[0])
    assert set(mine) == theirs, f"{len(set(mine) ^ theirs)} ids differ"


# --------------------------------------------------------------------------------------
# Major 24: the stale 16th subset and the headline-15 default
# --------------------------------------------------------------------------------------


def test_the_stale_dac_subset_is_named_and_is_not_in_the_headline_list() -> None:
    """The one thing that makes the anomaly greppable."""
    assert "longbench_synthetic_v4_1_loogle_shortdep_qa" in KNOWN_STALE_SUBSETS
    assert set(KNOWN_STALE_SUBSETS) & set(HEADLINE_SUBSETS) == set()
    assert "1739" in KNOWN_STALE_SUBSETS["longbench_synthetic_v4_1_loogle_shortdep_qa"]


def test_resolve_subsets_restricts_to_the_fifteen_and_says_what_it_dropped() -> None:
    present = [*HEADLINE_SUBSETS, "longbench_synthetic_v4_1_loogle_shortdep_qa"]
    sel = resolve_subsets(present)
    assert sel.subsets == HEADLINE_SUBSETS
    assert sel.restricted
    assert sel.dropped == ("longbench_synthetic_v4_1_loogle_shortdep_qa",)
    assert sel.stale == ("longbench_synthetic_v4_1_loogle_shortdep_qa",)
    assert any("dropped 1 subset" in w for w in sel.warnings)
    assert any("stale subset" in w for w in sel.warnings)


def test_resolve_subsets_never_drops_silently() -> None:
    """Every dropped row must be accounted for in a printable line."""
    sel = resolve_subsets([*HEADLINE_SUBSETS, "extra_a", "extra_b"])
    assert sel.dropped == ("extra_a", "extra_b")
    assert sel.warnings and all(n in " ".join(sel.warnings) for n in sel.dropped)


def test_all_subsets_keeps_everything_and_still_names_a_stale_one() -> None:
    present = [*HEADLINE_SUBSETS, "longbench_synthetic_v4_1_loogle_shortdep_qa"]
    sel = resolve_subsets(present, all_subsets=True)
    assert sel.subsets is None
    assert not sel.restricted
    assert sel.dropped == ()
    assert any("stale subset" in w for w in sel.warnings)


def test_a_partial_cell_is_not_trimmed_to_an_overlap_it_is_warned_about() -> None:
    """Filtering a 3-subset cell down to 'the 15 it has' would invent a paper number."""
    sel = resolve_subsets(HEADLINE_SUBSETS[:3])
    assert sel.subsets is None
    assert not sel.restricted
    assert sel.dropped == ()
    assert any("NOT the paper's population" in w for w in sel.warnings)


def test_an_explicit_subset_list_overrides_both() -> None:
    sel = resolve_subsets([*HEADLINE_SUBSETS, "extra"], explicit=["a", "b"])
    assert sel.subsets == ("a", "b")
    assert sel.restricted
    assert "extra" in " ".join(sel.warnings)


@pytest.mark.parametrize("model_key", ["gemma_4_31b_it", "qwen3.6_27b"])
def test_the_no_mask_arm_really_is_1739_rows_over_16_subsets(dac_outputs: Path, model_key: str) -> None:
    """Pins the anomaly to disk so the note in `KNOWN_STALE_SUBSETS` cannot rot."""
    root = dac_outputs / "outputs_128cap_backup_260703"
    if not root.is_dir():
        pytest.skip(f"no 128-cap backup under {dac_outputs}")
    ids = _dac_ids(root, model_key, "dac_chunk_multi_turn_tool_v1_no_mask")
    counts: dict[str, int] = {}
    for subset, _ in ids:
        counts[subset] = counts.get(subset, 0) + 1
    assert len(ids) == 1739
    assert len(counts) == 16
    assert counts["longbench_synthetic_v4_1_loogle_shortdep_qa"] == 128
    assert {k: v for k, v in counts.items() if k in HEADLINE_SUBSETS} == HEADLINE_128CAP_COUNTS

    summary = json.loads((root / model_key / "dac_chunk_multi_turn_tool_v1_no_mask" / "summary.json").read_text())
    assert summary["n_samples_attempted"] == 1611  # against 1739 rows on disk
    sel = resolve_subsets(counts)
    assert sel.dropped == ("longbench_synthetic_v4_1_loogle_shortdep_qa",)
