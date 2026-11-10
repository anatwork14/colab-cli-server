"""The judge sends chat messages with the model's own template and thinking enabled.

The HTTP recorder checks the request without a server.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeHTTP as Recorder

from dattn.eval import client as client_mod
from dattn.eval.judge import JudgeConfig, render_judge_prompt
from dattn.eval.runner import RunPaths, run_judge
from dattn.eval.schema import ResponseRecord, write_jsonl

VERDICT = 'reasoning about it</think>\n```json\n{"correct": true}\n```'


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder({"choices": [{"message": {"content": VERDICT}, "finish_reason": "stop"}]})
    monkeypatch.setattr(client_mod, "post_json", rec)
    return rec


def _cell(tmp_path: Path) -> RunPaths:
    paths = RunPaths(tmp_path / "cell")
    paths.root.mkdir(parents=True)
    write_jsonl(
        paths.responses,
        [
            ResponseRecord(
                subset="s",
                row_index=i,
                question=f"Q{i}",
                rubric=f"R{i}",
                parsed_answer=f"A{i}",
                format_correct=True,
            ).to_dict()
            for i in range(2)
        ],
    )
    return paths


def _server() -> Any:
    from dattn.eval.client import ServerConfig

    return ServerConfig(base_url="http://judge:8000/")


def test_run_judge_posts_to_the_chat_endpoint_not_completions(tmp_path: Path, recorder: Recorder) -> None:
    """The blocker itself: one raw `/v1/completions` call applies no chat template at all."""
    paths = _cell(tmp_path)
    n = run_judge(paths=paths, server=_server(), config=JudgeConfig())

    assert n == 2
    assert recorder.calls, "the judge made no request"
    urls = {url for url, _ in recorder.calls}
    assert urls == {"http://judge:8000/v1/chat/completions"}
    assert not any(url.endswith("/v1/completions") for url, _ in recorder.calls)


def test_run_judge_sends_one_user_turn_and_enables_thinking(tmp_path: Path, recorder: Recorder) -> None:
    """The wire shape `JudgeConfig.request_body` promises, reaching the wire for real."""
    paths = _cell(tmp_path)
    run_judge(paths=paths, server=_server(), config=JudgeConfig())

    _, payload = recorder.calls[0]
    assert "prompt" not in payload
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["max_tokens"] == 32768
    assert payload["temperature"] == 1.0
    assert payload["top_k"] == 20
    assert payload["presence_penalty"] == 1.5
    assert [m["role"] for m in payload["messages"]] == ["user"]
    assert payload["messages"][0]["content"] == render_judge_prompt("Q0", "R0", "A0")


def test_the_verdict_is_read_off_the_chat_message(tmp_path: Path, recorder: Recorder) -> None:
    """A real dac judgment carries `</think>` and no opener, and still parses."""
    paths = _cell(tmp_path)
    run_judge(paths=paths, server=_server(), config=JudgeConfig())

    rows = [json.loads(line) for line in paths.judgments.read_text().splitlines()]
    assert len(rows) == 2
    assert all(r["correct"] is True for r in rows)
    assert all(r["evaluation_text"] == VERDICT for r in rows)
    assert all(r["error"] == "" for r in rows)


def test_judge_new_count_excludes_deduplicated_records(tmp_path: Path, recorder: Recorder) -> None:
    paths = _cell(tmp_path)
    # Legacy repeated response rows can produce duplicate judgments within and
    # across client chunks. Only persisted unique verdicts contribute to n_new.
    paths.responses.write_bytes(paths.responses.read_bytes() * 2)
    assert run_judge(paths=paths, server=_server(), config=JudgeConfig(chunk_size=3)) == 2
    assert len(paths.judgments.read_text().splitlines()) == 2
    assert run_judge(paths=paths, server=_server(), config=JudgeConfig()) == 0
