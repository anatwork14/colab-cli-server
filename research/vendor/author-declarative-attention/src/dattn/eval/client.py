"""Token-completion and chat HTTP endpoints with one bounded retry and a thread pool."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ChatClient", "Completion", "ServerConfig", "post_json"]


@dataclass(frozen=True, slots=True)
class ServerConfig:
    base_url: str = "http://localhost:8000"
    model: str = ""
    api_key: str = "EMPTY"
    timeout_s: float = 3600.0
    max_retries: int = 1
    concurrency: int = 32

    def __post_init__(self):
        if self.max_retries not in (0, 1):
            raise ValueError("max_retries must be zero or one")

    @property
    def url(self) -> str:
        return self.base_url.rstrip("/")


@dataclass(slots=True)
class Completion:
    """One choice. `finish_reason` is the engine's, not a hard-coded "stop"."""

    text: str = ""
    finish_reason: str = ""
    stop_reason: str | int | None = None
    n_prompt_tokens: int | None = None
    n_generated_tokens: int | None = None
    error: str = ""
    transport_error: bool = False
    latency_s: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_answer(self) -> bool:
        """True when the model produced text, however empty or malformed that text is."""
        return not self.transport_error


def _headers(cfg: ServerConfig) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    return headers


def post_json(url: str, payload: Mapping[str, Any], *, headers: Mapping[str, str], timeout_s: float) -> dict[str, Any]:
    """One JSON POST. Raises on any transport or HTTP error, which callers record."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:1000]
        raise urllib.error.HTTPError(exc.url, exc.code, f"{exc.reason}: {body}", exc.headers, None) from None


class ChatClient:
    """A thin, synchronous, retrying client. One instance per run."""

    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self._headers = _headers(cfg)

    def close(self) -> None:
        """Nothing to close: `urllib` opens a connection per request."""

    def __enter__(self) -> ChatClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def complete(
        self,
        prompt: str | Sequence[int],
        *,
        body: Mapping[str, Any] | None = None,
        xargs: Mapping[str, Any] | None = None,
        model: str | None = None,
    ) -> list[Completion]:
        """POST `/v1/completions`. `prompt` may be text or token ids."""
        payload: dict[str, Any] = {
            "model": model or self.cfg.model,
            "prompt": prompt if isinstance(prompt, str) else list(prompt),
        }
        extra = dict(body or {})
        payload.update(extra.pop("extra_body", {}) or {})
        payload.update(extra)
        if xargs:
            payload["vllm_xargs"] = dict(xargs)
        return self._post("/v1/completions", payload)

    def chat(
        self, messages: Sequence[Mapping[str, str]], *, body: Mapping[str, Any] | None = None, model: str | None = None
    ) -> list[Completion]:
        """POST `/v1/chat/completions`. The SERVER applies the model's own chat template."""
        payload: dict[str, Any] = {"model": model or self.cfg.model, "messages": [dict(m) for m in messages]}
        extra = dict(body or {})
        payload.update(extra.pop("extra_body", {}) or {})
        payload.update(extra)
        return self._post("/v1/chat/completions", payload)

    def _post(self, endpoint, payload):
        started = time.time()
        result = Completion()
        for attempt in range(self.cfg.max_retries + 1):
            try:
                choices = _parse(
                    post_json(self.cfg.url + endpoint, payload, headers=self._headers, timeout_s=self.cfg.timeout_s),
                    time.time() - started,
                )
                if all(c.is_answer for c in choices):
                    return choices
                result = choices[0]
            except Exception as exc:
                result = Completion(error=f"{type(exc).__name__}: {exc}", transport_error=True)
            if attempt < self.cfg.max_retries:
                time.sleep(1.0)
        result.latency_s = time.time() - started
        return [result]

    def complete_many(
        self,
        prompts: Sequence[str | Sequence[int]],
        *,
        bodies: Sequence[Mapping[str, Any]] | None = None,
        xargs: Sequence[Mapping[str, Any] | None] | None = None,
        model: str | None = None,
    ) -> list[list[Completion]]:
        """Fire `prompts` concurrently, preserving order. Failures come back as rows."""
        n = len(prompts)
        bodies = bodies or [{}] * n
        xargs = xargs or [None] * n

        def one(i: int) -> list[Completion]:
            return self.complete(prompts[i], body=bodies[i], xargs=xargs[i], model=model)

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.concurrency)) as pool:
            return list(pool.map(one, range(n)))

    def chat_many(
        self,
        messages_list: Sequence[Sequence[Mapping[str, str]]],
        *,
        bodies: Sequence[Mapping[str, Any]] | None = None,
        model: str | None = None,
    ) -> list[list[Completion]]:
        """Fire chat requests concurrently, preserving order. Failures come back as rows."""
        n = len(messages_list)
        bodies = bodies or [{}] * n

        def one(i: int) -> list[Completion]:
            return self.chat(messages_list[i], body=bodies[i], model=model)

        with ThreadPoolExecutor(max_workers=max(1, self.cfg.concurrency)) as pool:
            return list(pool.map(one, range(n)))


def _parse(payload: Mapping[str, Any], latency_s: float) -> list[Completion]:
    """Read choices from either endpoint. Judge servers use no reasoning parser."""
    usage = payload.get("usage") or {}
    out: list[Completion] = []
    for choice in payload.get("choices", []):
        message = choice.get("message") or {}
        text = choice.get("text") or message.get("content") or ""
        out.append(
            Completion(
                text=text,
                finish_reason=str(choice.get("finish_reason") or ""),
                stop_reason=choice.get("stop_reason"),
                n_prompt_tokens=usage.get("prompt_tokens"),
                n_generated_tokens=(
                    len(choice["token_ids"])
                    if choice.get("token_ids") is not None
                    else usage.get("completion_tokens")
                    if len(payload.get("choices", [])) == 1
                    else None
                ),
                extra={"index": choice["index"]} if "index" in choice else {},
                latency_s=latency_s,
            )
        )
    return out or [Completion(error="server returned no choices", transport_error=True, latency_s=latency_s)]
