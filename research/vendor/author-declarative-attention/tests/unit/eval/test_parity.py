"""Public self-check requires identical keys, rows and nonempty prompt hashes."""

import pytest

from dattn.eval.parity import compare_trees, load_tree

from .conftest import response_row, write_cell


def test_mask_self_check_allows_different_sampled_answers(tmp_path):
    a = write_cell(tmp_path / "a", [response_row("s", 0, prompt_sha256="hash")])
    b = write_cell(tmp_path / "b", [response_row("s", 0, prompt_sha256="hash", response="different")])
    report = compare_trees(a, b)
    assert report.passed and report.n_rows == 1
    assert "PASS" in report.render()


@pytest.mark.parametrize(
    "change", [{"prompt_sha256": ""}, {"prompt_sha256": "other"}, {"question": "other"}, {"row_index": 1}]
)
def test_missing_or_different_identity_fails(tmp_path, change):
    a = write_cell(tmp_path / "a", [response_row("s", 0, prompt_sha256="hash")])
    b = write_cell(tmp_path / "b", [response_row("s", 0, prompt_sha256="hash") | change])
    assert not compare_trees(a, b).passed


def test_empty_or_unaligned_trees_cannot_pass(tmp_path):
    a = write_cell(tmp_path / "a", [])
    b = write_cell(tmp_path / "b", [response_row("s", 0, prompt_sha256="hash")])
    assert not compare_trees(a, a).passed
    assert compare_trees(a, b).n_right_only == 1
    assert not compare_trees(a, b).passed


def test_duplicate_key_and_private_modes_are_refused(tmp_path):
    a = write_cell(tmp_path / "a", [response_row("s", 0)] * 2)
    with pytest.raises(ValueError, match="duplicate"):
        load_tree(a)
    with pytest.raises(ValueError, match="only mask"):
        compare_trees(a, a, mode="replicate")
