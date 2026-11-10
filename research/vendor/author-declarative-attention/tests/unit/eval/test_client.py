"""The two endpoints the eval lane posts to, and what each one puts on the wire.

No GPU and no server: `post_json` is replaced by a recorder, so these tests pin the
request SHAPE, which is the thing Blocker 2 got wrong. The judge request is the paper's
instrument, so every field in it is asserted rather than sampled.
"""

from __future__ import annotations

from typing import Any

import pytest
from fakes import FakeHTTP as Recorder

from dattn.eval import client as client_mod
from dattn.eval.client import ChatClient, ServerConfig, _parse
from dattn.eval.judge import JudgeConfig, parse_verdict


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    rec = Recorder()
    monkeypatch.setattr(client_mod, "post_json", rec)
    return rec


def test_chat_posts_messages_to_the_chat_endpoint_with_thinking_on(recorder: Recorder) -> None:
    """The exact judge wire shape, dac's recipe field by field."""
    cfg = JudgeConfig()
    cli = ChatClient(ServerConfig(base_url="http://judge:8000/"))
    cli.chat([{"role": "user", "content": "P"}], body=cfg.request_body(), model=cfg.model_id)

    url, payload = recorder.calls[0]
    assert url == "http://judge:8000/v1/chat/completions"
    assert payload["model"] == "Qwen/Qwen3.5-4B"
    assert payload["messages"] == [{"role": "user", "content": "P"}]
    assert "prompt" not in payload
    assert "extra_body" not in payload  # flattened by `chat`
    assert payload["chat_template_kwargs"] == {"enable_thinking": True}
    assert payload["max_tokens"] == 32768
    assert payload["n"] == 1
    assert payload["temperature"] == 1.0
    assert payload["top_p"] == 0.95
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["presence_penalty"] == 1.5
    assert payload["repetition_penalty"] == 1.0
    # dac sends no stop, no seed, no tools and no system turn.
    assert not {"stop", "seed", "tools"} & set(payload)
    assert all(m["role"] == "user" for m in payload["messages"])


def test_the_judge_body_carries_no_extra_fields(recorder: Recorder) -> None:
    """A guard against a field creeping into the instrument unnoticed."""
    cli = ChatClient(ServerConfig(base_url="http://judge:8000"))
    cli.chat([{"role": "user", "content": "P"}], body=JudgeConfig().request_body(), model="m")
    assert set(recorder.calls[0][1]) == {
        "model",
        "messages",
        "chat_template_kwargs",
        "max_tokens",
        "n",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
    }


def test_a_seeded_judge_config_adds_seed_and_nothing_else() -> None:
    body = JudgeConfig(seed=7).request_body()
    assert body["seed"] == 7
    assert JudgeConfig().request_body().get("seed") is None


def test_chat_many_preserves_order_and_sends_one_request_per_turn(recorder: Recorder) -> None:
    cli = ChatClient(ServerConfig(base_url="http://judge:8000", concurrency=4))
    out = cli.chat_many([[{"role": "user", "content": f"P{i}"}] for i in range(5)], model="m")
    assert len(out) == 5
    assert len(recorder.calls) == 5
    assert sorted(c[1]["messages"][0]["content"] for c in recorder.calls) == [f"P{i}" for i in range(5)]


def test_the_rollout_path_still_posts_the_completions_endpoint(recorder: Recorder) -> None:
    """Blocker 2 is about the judge only. `complete` must not move."""
    cli = ChatClient(ServerConfig(base_url="http://engine:8000"))
    cli.complete([1, 2, 3], body={"max_tokens": 8192, "extra_body": {"top_k": 20}})
    url, payload = recorder.calls[0]
    assert url == "http://engine:8000/v1/completions"
    assert payload["prompt"] == [1, 2, 3]
    assert "messages" not in payload
    assert payload["top_k"] == 20


def test_a_chat_transport_failure_becomes_a_recorded_row(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_: Any, **__: Any) -> dict[str, Any]:
        raise OSError("connection refused")

    monkeypatch.setattr(client_mod, "post_json", boom)
    cli = ChatClient(ServerConfig(base_url="http://judge:8000", max_retries=0))
    got = cli.chat([{"role": "user", "content": "P"}])
    assert len(got) == 1
    assert "connection refused" in got[0].error
    assert got[0].text == ""


# -- parsing the response ----------------------------------------------------------------


def test_the_chat_verdict_is_read_from_message_content() -> None:
    got = _parse({"choices": [{"message": {"content": '...\n</think>\n\n{"correct": true}'}}]}, 0.0)
    assert parse_verdict(got[0].text).correct is True
    assert parse_verdict(got[0].text).stage == "json"


def test_no_reasoning_content_leaves_the_text_untouched() -> None:
    got = _parse({"choices": [{"message": {"content": "plain"}}]}, 0.0)
    assert got[0].text == "plain"
    assert got[0].extra == {}


def test_an_empty_choices_list_is_an_error_row() -> None:
    got = _parse({"choices": []}, 1.5)
    assert got[0].error
    assert got[0].latency_s == 1.5


@pytest.mark.parametrize("endpoint", ["chat", "complete"])
def test_retry_is_bounded_to_one_and_transport_failure_stays_flagged(monkeypatch, endpoint):
    calls = []

    def fail(*args, **kw):
        calls.append(args)
        raise OSError("unreachable")

    monkeypatch.setattr(client_mod, "post_json", fail)
    monkeypatch.setattr(client_mod.time, "sleep", lambda seconds: None)
    client = ChatClient(ServerConfig())
    result = client.complete("p") if endpoint == "complete" else client.chat([])
    assert len(calls) == 2
    assert result[0].transport_error and not result[0].is_answer
    with pytest.raises(ValueError, match="zero or one"):
        ServerConfig(max_retries=2)


def test_empty_model_answer_is_not_retried(monkeypatch):
    calls = []

    def empty(*args, **kw):
        calls.append(args)
        return {"choices": [{"text": "", "finish_reason": "stop"}]}

    monkeypatch.setattr(client_mod, "post_json", empty)
    result = ChatClient(ServerConfig()).complete("p")
    assert len(calls) == 1 and result[0].is_answer
