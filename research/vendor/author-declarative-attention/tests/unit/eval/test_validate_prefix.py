import copy
import json
from types import SimpleNamespace

import pytest

from dattn.bench import prefix_validation as worker
from dattn.campaign import driver
from dattn.campaign import validate_prefix as vp


def test_selection_uses_rendered_pilot_extremes_and_shared_document(monkeypatch):
    monkeypatch.setattr(vp, "PILOT", ("pilot",))
    rows = [
        {"subset": "pilot", "row_index": i, "question": str(i), "context": "doc" if i < 2 else str(i)} for i in range(8)
    ]
    records = [{**r, "rendered_prompt_tokens": (8 - r["row_index"]) * 100} for r in rows]
    plan = vp.select_rows(rows, records)
    assert [r["row_index"] for r in plan["rows"]] == [7, 6, 5, 2, 1, 0]
    assert [r["row_index"] for r in plan["pair"]] == [0, 1]


def test_selection_refuses_missing_pair(monkeypatch):
    monkeypatch.setattr(vp, "PILOT", ("pilot",))
    rows = [
        {"subset": "pilot", "row_index": i, "question": str(i), "context": str(i), "rendered_prompt_tokens": i + 1}
        for i in range(6)
    ]
    with pytest.raises(ValueError, match="sharing"):
        vp.select_rows(rows, rows)


@pytest.fixture
def output():
    return {
        "token_ids": [10, 20, 30],
        "mode_trace": [0, 1, 1],
        "attended_tokens_exact": 123,
        "live_trace": [[1, 1, 16, {"0": 32}], [2, 1, 17, {"0": 32}]],
        "text": "answer",
        "finish_reason": "length",
    }


@pytest.mark.parametrize("tokens,position", [([10, 21, 30], 1), ([10, 20], 2), ([10, 20, 30, 40], 3)])
def test_first_token_difference_and_json_failure(output, tokens, position):
    cached = {**output, "token_ids": tokens}
    difference = vp.compare_outputs(output, cached)
    doc = vp.verdict("gemma", [{"difference": difference}])
    assert doc["status"] == "FAIL"
    assert doc["first_differing_token_position"] == position
    assert json.loads(json.dumps(doc))["schema"] == vp.SCHEMA


def test_equal_tokens_do_not_hide_live_mask_mismatch(output):
    cached = copy.deepcopy(output)
    cached["live_trace"][0][2] = 17
    difference = vp.compare_outputs(output, cached)
    assert difference["field"] == "live_trace"
    assert difference["first_differing_token_position"] is None
    assert difference["first_differing_trace_position"] == 1
    assert vp.verdict("qwen", [{"difference": vp.compare_outputs(output, output)}])["status"] == "PASS"
    assert vp.verdict("qwen", [])["status"] == "FAIL"


def test_capture_separates_children_and_records_real_compaction():
    rec = worker.Capture()
    rec.note_lp(0, [("a", 1, 100, 16, 1, 2), ("b", 0, 100, 100, 1, 2)])
    rec.note_compaction(0, None, [32, 100], 0)
    assert rec.traces["a"][1] == [1, 1, 16, {"0": 32}]
    assert rec.traces["b"][1] == [1, 0, 100, {"0": 100}]


def test_real_vllm_greedy_parent_fans_out_to_four_valid_children():
    from vllm import SamplingParams
    from vllm.v1.engine.parallel_sampling import ParentRequest

    params = worker.greedy_params(SamplingParams, {"dattn": {"req_id": "parent"}}, 4)
    parent = ParentRequest(SimpleNamespace(external_req_id="parent", request_id="parent", params=params))
    children = [parent.get_child_info(i) for i in range(4)]
    assert len({key for key, _ in children}) == 4
    assert all(p.n == 1 and p.temperature == 0 for _, p in children)
    assert all(p.extra_args == params.extra_args for _, p in children)


def test_foreign_gpu_refusal_writes_fail_without_starting_worker(tmp_path, monkeypatch):
    def refuse(gpu):
        assert gpu == "3"
        raise RuntimeError("GPU 3 REFUSED, foreign PIDs [123]")

    monkeypatch.setattr(driver, "assert_free", refuse)
    monkeypatch.setattr(vp.subprocess, "Popen", lambda *a, **k: pytest.fail("must not launch"))
    out = tmp_path / "result.json"
    assert vp.run(SimpleNamespace(gpu="3", model="gemma", root=tmp_path, out=out)) == 1
    doc = json.loads(out.read_text())
    assert doc["status"] == "FAIL" and "foreign" in doc["error"]


def test_complete_artifacts_check_every_arm_child_and_document_pair(output):
    a = {"subset": "pilot", "row_index": 0}
    b = {"subset": "pilot", "row_index": 1}
    plan = {"rows": [a, b], "pair": [a, b]}
    keys = [repr(vp.identity(r)) for r in (a, b)]
    plain = {"arms": {arm: {key: {"cold": [copy.deepcopy(output)]} for key in keys} for arm in vp.ARMS}}
    prefix = {
        "arms": {
            arm: {
                key: {
                    "cold": [copy.deepcopy(output)],
                    "warm": [copy.deepcopy(output)],
                    "children": [copy.deepcopy(output) for _ in range(4)],
                }
                for key in keys
            }
            for arm in vp.ARMS
        },
        "pair": {arm: copy.deepcopy(output) for arm in vp.ARMS},
    }
    checks = vp.check_artifacts(plain, prefix, plan)
    assert len(checks) == 39
    assert vp.verdict("gemma", checks)["status"] == "PASS"
    prefix["arms"]["da"][keys[0]]["children"][2]["token_ids"][1] = 999
    doc = vp.verdict("gemma", vp.check_artifacts(plain, prefix, plan))
    assert doc["status"] == "FAIL"
    assert doc["first_differing_token_position"] == 1
    assert doc["failure"]["child"] == 2
    del prefix["arms"]["da_nm"][keys[1]]
    with pytest.raises(ValueError, match="coverage"):
        vp.check_artifacts(plain, prefix, plan)
