"""Thin-bin boundaries count kept steps across contributing tasks and batches."""

import json

import pytest

from dattn.bench.aggregate import macro_report
from dattn.bench.plots import saving_plot


@pytest.mark.parametrize("n_kept,n_tasks,thin", [(19, 3, True), (20, 3, False), (20, 2, True), (19, 2, True)])
def test_thin_bin_boundary_and_hatching(tmp_path, monkeypatch, n_kept, n_tasks, thin):
    import matplotlib.pyplot as plt

    for task in range(n_tasks):
        n = n_kept // n_tasks + (task < n_kept % n_tasks)
        path = tmp_path / str(task)
        path.mkdir()
        path.joinpath("summary.json").write_text(
            json.dumps(
                {
                    "protocol": {"model_id": "m", "n_resamples": 10, "ci": 0.95, "seed": 0},
                    "task": str(task),
                    "meta": {"batch_index": 0},
                    "bins": {
                        "mask_ratio": [
                            {"task": str(task), "batch": 0, "bin_index": 0, "n": n, "da_sum": n, "vanilla_sum": 2 * n}
                        ]
                    },
                }
            )
        )
    row = macro_report([tmp_path], allow_partial=True)["macro"][0]
    assert (row["n_kept"], row["n_tasks"], row["thin"]) == (n_kept, n_tasks, thin)
    assert row["saving"] == 0.5 and not row["ci_defined"]
    fig, ax = plt.subplots()
    monkeypatch.setattr(plt, "subplots", lambda: (fig, ax))
    saving_plot([row], tmp_path / "saving.svg")
    assert [patch.get_hatch() for patch in ax.patches] == (["///"] if thin else [])
