import json
import statistics

import pytest
from fakes import write_cell

from dattn.campaign import sample_compare as sc


@pytest.fixture
def comparison_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(sc, "MODELS", {"m": "model"})
    monkeypatch.setattr(sc, "ARMS", {"a": "refarm"})
    monkeypatch.setattr(sc, "SLUGS", {"m": "slug"})
    monkeypatch.setattr(sc, "HEADLINE_SUBSETS", ("short", "long"))
    rows = [
        {"subset": t, "row_index": i, "question": "q", "answer": "a", "rubric": "r", "format_correct": True}
        for t in ("short", "long")
        for i in range(2)
    ]
    js = [
        {
            "subset": r["subset"],
            "row_index": r["row_index"],
            "correct": r["row_index"] == 0,
            "dac_correct": r["row_index"] == 0,
        }
        for r in rows
    ]
    root, dac = tmp_path / "root", tmp_path / "dac"
    write_cell(root / "rep0/slug/a", rows, js)
    write_cell(dac / "slug/refarm", rows, js)
    monkeypatch.setattr(sc, "reference", lambda *a: (rows, {"headline": 4, "total": 4, "dropped": 0}))
    rr, jj = [], []
    for sample, correct_count in enumerate((0, 1, 2, 1), 1):
        rr += [{**r, "response_index": sample} for r in rows]
        jj += [
            {
                **j,
                "response_index": sample,
                "correct": j["row_index"] < correct_count,
                "dac_correct": j["row_index"] < correct_count,
            }
            for j in js
        ]
    write_cell(root / "samples/slug/a", rr, jj)
    return root, dac


def report(tree):
    root, dac = tree
    return sc.report(
        root, ["rep0"], dac, ("short", "long"), stage="full", macro_pp=2, task_pp=5, task_se=0, per_task_stop=False
    )


def test_five_sample_statistics_and_no_rep0_writes(comparison_tree):
    root, _ = comparison_tree
    before = {str(p): p.read_bytes() for p in (root / "rep0").rglob("*") if p.is_file()}
    doc, text = report(comparison_tree)
    assert doc["schema"] == "step3.verdict.v1"
    assert doc["status"] == "pass"
    for item in doc["cells"]["m/a"].values():
        assert item["dac_correct_pct"] == 50
        assert item["run_sd_pp"] == pytest.approx(statistics.stdev([50, 0, 50, 100, 50]))
        assert item["delta_pp"] == item["delta_se"] == 0
    assert doc["samples"]["n"] == 5
    assert "2/2, 2/2, 2/2, 2/2, 2/2" in text
    assert before == {str(p): p.read_bytes() for p in (root / "rep0").rglob("*") if p.is_file()}


def test_partial_tree_reports_each_sample_and_macro_coverage(comparison_tree):
    root, _ = comparison_tree
    cell = root / "samples/slug/a"
    for filename in ("responses.jsonl", "judgments.jsonl"):
        docs = sc.readonly(cell / filename)
        (cell / filename).write_text(
            "".join(
                json.dumps(d) + "\n"
                for d in docs
                if not (d["subset"] == "long" and d["response_index"] == 4 and d["row_index"] == 1)
            )
        )
    doc, text = report(comparison_tree)
    assert doc["complete"] is False and doc["status"] == "incomplete"
    assert doc["samples"]["macro_tasks"] == ["short"]
    assert doc["samples"]["coverage"]["m/a"]["long"]["by_sample"][4]["scored"] == 1
    assert doc["cells"]["m/a"]["long"]["dac_correct_pct"] is None
    assert "1/2" in text


def test_missing_samples_has_no_false_five_sample_estimate(comparison_tree):
    root, _ = comparison_tree
    for path in (root / "samples/slug/a").iterdir():
        path.unlink()
    doc, _ = report(comparison_tree)
    assert not doc["complete"]
    assert doc["samples"]["macro_tasks"] == []
    assert doc["cells"]["m/a"]["macro"]["dac_correct_pct"] is None


def test_torn_samples_tail_is_uncommitted_and_never_repaired(comparison_tree):
    root, _ = comparison_tree
    path = root / "samples/slug/a/responses.jsonl"
    with path.open("ab") as handle:
        handle.write(b'{"subset":')
    before = path.read_bytes()
    doc, _ = report(comparison_tree)
    assert doc["status"] == "pass"
    assert path.read_bytes() == before


def test_partial_sample_with_malformed_judgment_refuses(comparison_tree):
    root, _ = comparison_tree
    path = root / "samples/slug/a/judgments.jsonl"
    docs = sc.readonly(path)
    del docs[0]["dac_correct"]
    path.write_text("".join(json.dumps(d) + "\n" for d in docs[:-1]))
    with pytest.raises(ValueError, match="invalid judgment"):
        report(comparison_tree)
