"""Resumable rollout and judge drivers with selection-safe persistence."""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

from dattn.eval.client import ChatClient, ServerConfig
from dattn.eval.datasets import EvalRow, SelectionSpec
from dattn.eval.extract import extract_answer
from dattn.eval.judge import JudgeConfig, Judgment, parse_verdict
from dattn.eval.persistence import append_records, atomic_bytes, atomic_json, failed, lock, read_records
from dattn.eval.prompts import build_prompt, dattn_xargs
from dattn.eval.replay import dense_trace, replay_record
from dattn.eval.sampling import RolloutConfig
from dattn.eval.schema import ResponseRecord, RunMeta
from dattn.prompt.template_policy import resolve_template, template_metadata

__all__ = ["RunPaths", "run_judge", "run_rollouts"]
Progress = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class RunPaths:
    """One output cell, in dac's directory shape plus `meta.json`."""

    root: Path

    @property
    def responses(self) -> Path:
        return self.root / "responses.jsonl"

    @property
    def judgments(self) -> Path:
        return self.root / "judgments.jsonl"

    @property
    def meta(self) -> Path:
        return self.root / "meta.json"

    @property
    def rows(self) -> Path:
        return self.root / "rows.jsonl"

    @classmethod
    def for_cell(cls, out_dir: Path, model_id: str, arm: str) -> RunPaths:
        """dac's naming: model basename lowercased, dashes to underscores."""
        return cls(out_dir / model_id.split("/")[-1].lower().replace("-", "_") / arm)


def _done_keys(path: Path, *, retry_transport_failures=False):
    if not path.exists():
        return set()
    return {
        (d["subset"], d["row_index"], d.get("response_index", 0))
        for d in read_records(path)
        if not (retry_transport_failures and failed(d))
    }


def _prepare_judge_resume(path: Path) -> set[tuple[str, int, int]]:
    """Atomically drop failed transports before retrying, retaining unique verdicts.

    Judgments are unique-key records, not a last-record-wins log. Aggregation keeps
    rejecting duplicate judgments. Only transport failures are removed, never model
    refusals or unparsable verdicts. Legacy error-only records denote failed transports.
    A crash before replacement leaves the old file intact, after replacement leaves
    failed keys pending. The judge lock serializes judge passes per cell, including partial passes.
    """
    if not path.exists():
        return set()
    docs = read_records(path)
    kept, done = [], set()
    for doc in docs:
        judgment = Judgment.from_dict(doc)
        if judgment.transport_error:
            continue
        if judgment.key in done:
            raise ValueError(f"duplicate judgment for {judgment.key} in {path}")
        done.add(judgment.key)
        kept.append(doc)
    if len(kept) != len(docs):
        with lock(path.with_suffix(".lock")):
            atomic_bytes(path, b"".join((json.dumps(doc) + "\n").encode() for doc in kept))
    return done


def _now() -> str:
    return _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds")


def _prior_meta(path: Path) -> dict[str, Any]:
    """The `meta.json` already in the cell, or `{}`. Must be read before it is rewritten."""
    if not path.exists():
        return {}
    got = json.loads(path.read_text())
    return got if isinstance(got, dict) else {}


def _check_selection(paths: RunPaths, prior: Mapping[str, Any], spec: SelectionSpec | None):
    was = prior.get("selection") or {}
    now = spec.to_dict() if spec else was
    if was and was != now and paths.responses.exists():
        raise ValueError(f"{paths.root}: different SelectionSpec: recorded {was!r}, requested {now!r}")
    return now


def _write_meta(paths: RunPaths, meta: RunMeta) -> None:
    meta.created = meta.created or _now()
    atomic_json(paths.meta, meta.to_dict())


def _n_steps(completion: Any, prompt: Any, tokenizer: Any) -> int:
    """Decode steps behind a non-DA rollout, for `dense_trace`."""
    del prompt
    n = getattr(completion, "n_generated_tokens", None)
    if n is not None:
        return int(n)
    return len(tokenizer(completion.text, add_special_tokens=False)["input_ids"])


def _locked_rollouts(function):
    @wraps(function)
    def wrapped(rows, *, paths, **kwargs):
        # Hold row claims through generation and durable append. Disjoint shards
        # proceed concurrently. Overlapping invocations cannot regenerate a done row.
        # Bound open claim descriptors even for full-population evals and large
        # requested chunks. Recheck done keys after each batch acquires its claims.
        batch_size = min(kwargs.get("chunk_size", 64), 64)
        if batch_size < 1:
            raise ValueError("chunk_size must be positive")
        if not rows:
            return function(rows, paths=paths, **kwargs)
        n_new = 0
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            with ExitStack() as stack:
                for row_key in sorted({(r.subset, r.row_index) for r in batch}):
                    name = hashlib.sha256(repr(row_key).encode()).hexdigest()
                    stack.enter_context(lock(paths.root / ".claims" / name))
                n_new += function(batch, paths=paths, **kwargs)
        return n_new

    return wrapped


@_locked_rollouts
def run_rollouts(
    rows: Sequence[EvalRow],
    *,
    tokenizer: Any,
    config: RolloutConfig,
    server: ServerConfig,
    paths: RunPaths,
    spec: SelectionSpec | None = None,
    last_answer: bool = False,
    chunk_size: int = 64,
    progress: Progress | None = None,
    meta_extra: dict[str, Any] | None = None,
    record_replay_metrics: bool = True,
    dattn_block_size: int = 0,
    response_index_start: int = 0,
) -> int:
    """Render, generate and persist. Returns the number of new rows written."""
    config = config.resolved()
    if config.thinking:
        raise ValueError("rollout rendering pins enable_thinking=False for dac parity")
    template = resolve_template(tokenizer, config.model_id, config.template_policy)
    rendering = template_metadata(template, config.template_policy)
    paths.root.mkdir(parents=True, exist_ok=True)
    with lock(paths.root / ".meta.lock"):
        prior = _prior_meta(paths.meta)
        selection = _check_selection(paths, prior, spec)
        done = _done_keys(paths.responses, retry_transport_failures=True)
        if done and prior.get("rendering") != rendering:
            raise ValueError("different or unrecorded template policy/hash, use a new output cell")
        if done and prior.get("rollout", {}).get("segmenter", "semantic") != config.segmenter:
            raise ValueError("different segmenter, use a new output cell")
        if done and prior.get("rollout", {}).get("vanilla_system") != config.vanilla_system:
            raise ValueError("different vanilla system turn, use a new output cell")
        extra = dict(meta_extra or {})
        extra.setdefault("created", str(prior.get("created") or ""))
        extra.setdefault("judge", prior.get("judge") or {})
        _write_meta(
            paths,
            RunMeta(
                model_id=config.model_id,
                arm=config.arm,
                selection=selection,
                rollout=config.to_dict(),
                rendering=rendering,
                **extra,
            ),
        )
    indices = range(response_index_start, response_index_start + config.n)
    pending = [r for r in rows if any((r.subset, r.row_index, i) not in done for i in indices)]
    if progress:
        progress(f"rollout: {len(rows) - len(pending)}/{len(rows)} already done, {len(pending)} pending")
    body = config.request_body()
    if response_index_start:
        body["return_token_ids"] = True
    n_new = 0
    with ChatClient(server) as client:
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start : start + chunk_size]
            built = [
                build_prompt(
                    tokenizer,
                    context=r.context,
                    question=r.question,
                    fmt=config.prompt_format,
                    tool_schema=config.tool_schema,
                    chat_template=template,
                    segmenter=config.segmenter,
                    vanilla_system=config.vanilla_system,
                )
                for r in chunk
            ]
            results = client.complete_many(
                [b.ids for b in built],
                bodies=[body] * len(chunk),
                xargs=[dattn_xargs(b.layout, enabled=config.dattn_enabled) for b in built],
                model=config.model_id,
            )
            records: list[dict[str, Any]] = []
            n_failed = 0
            for row, prompt, completions in zip(chunk, built, results, strict=True):
                if response_index_start and all(c.is_answer for c in completions):
                    choices = {c.extra.get("index") for c in completions}
                    if choices != set(range(config.n)) or len(completions) != config.n:
                        raise ValueError("samples response has missing or duplicate child indices")
                for ordinal, c in enumerate(completions):
                    i = (
                        int(c.extra["index"]) if response_index_start and c.is_answer else ordinal
                    ) + response_index_start
                    if (row.subset, row.row_index, i) in done:
                        continue
                    if not c.is_answer:
                        n_failed += 1
                        records.append(
                            ResponseRecord(
                                subset=row.subset,
                                row_index=row.row_index,
                                response_index=i,
                                error=c.error,
                                transport_error=True,
                            ).to_dict()
                        )
                        continue
                    ex = extract_answer(c.text, last=last_answer)
                    trace: dict[str, Any] = {}
                    if record_replay_metrics:
                        trace = (
                            replay_record(c.text, prompt.layout, tokenizer, block_size=dattn_block_size)
                            if config.dattn_enabled
                            else dense_trace(_n_steps(c, prompt, tokenizer), prompt.layout)
                        )
                    records.append(
                        ResponseRecord(
                            subset=row.subset,
                            row_index=row.row_index,
                            response_index=i,
                            question=row.question,
                            answer=row.answer,
                            rubric=row.rubric,
                            raw_context_tokens=row.raw_context_tokens,
                            rendered_prompt_tokens=c.n_prompt_tokens or prompt.n_prompt_tokens,
                            n_bookmarks=prompt.n_segments,
                            n_semantic_segments=prompt.n_semantic_segments,
                            segmenter=config.segmenter,
                            prompt_sha256=prompt.sha256,
                            prompt_format=config.prompt_format,
                            format_error=prompt.format_error,
                            response=c.text,
                            parsed_answer=ex.parsed_answer,
                            format_correct=ex.format_correct,
                            finish_reason=c.finish_reason,
                            stop_reason=c.stop_reason,
                            n_generated_tokens=c.n_generated_tokens,
                            answer_policy=ex.policy,
                            n_answer_blocks=ex.n_blocks,
                            first_answer=ex.first_text,
                            last_answer=ex.last_text,
                            error=c.error,
                            **trace,
                        ).to_dict()
                    )
            n_new += append_records(paths.responses, records)
            if progress:
                failed = f", {n_failed} failed in transport and left pending" if n_failed else ""
                progress(f"  persisted {start + len(chunk)}/{len(pending)} ({n_new} rows{failed})")
    return n_new


def _locked_judge(function):
    @wraps(function)
    def wrapped(*, paths, **kwargs):
        with lock(paths.root / ".judge.lock"):
            return function(paths=paths, **kwargs)

    return wrapped


@_locked_judge
def run_judge(
    *,
    paths: RunPaths,
    server: ServerConfig,
    config: JudgeConfig | None = None,
    judge_all: bool = False,
    progress: Progress | None = None,
) -> int:
    """Grade every format-correct rollout that has no verdict yet."""
    config = config or JudgeConfig()
    rows = [ResponseRecord.from_dict(d) for d in read_records(paths.responses) if not failed(d)]
    done = _prepare_judge_resume(paths.judgments)
    pending = [r for r in rows if (judge_all or r.format_correct) and (r.key not in done)]
    if progress:
        progress(f"judge: {len(rows)} rollouts, {len(done)} already judged, {len(pending)} pending")
    body: dict[str, Any] = config.request_body()
    n_new = 0
    with ChatClient(server) as client:
        for start in range(0, len(pending), config.chunk_size):
            chunk = pending[start : start + config.chunk_size]
            messages = [config.messages(r.question, r.rubric, r.parsed_answer or "") for r in chunk]
            results = client.chat_many(messages, bodies=[body] * len(chunk), model=config.model_id)
            out: list[dict[str, Any]] = []
            for r, completions in zip(chunk, results, strict=True):
                c = completions[0]
                v = parse_verdict(c.text)
                out.append(
                    Judgment(
                        subset=r.subset,
                        row_index=r.row_index,
                        response_index=r.response_index,
                        correct=v.correct,
                        evaluation_text=c.text,
                        stage=v.stage,
                        dac_correct=v.dac_correct,
                        think_truncated=v.think_truncated,
                        n_candidates=v.n_candidates,
                        finish_reason=c.finish_reason,
                        error=c.error,
                        transport_error=c.transport_error,
                    ).to_dict()
                )
            n_new += append_records(paths.judgments, out)
            if progress:
                progress(f"  judged {start + len(chunk)}/{len(pending)}")
    with lock(paths.root / ".meta.lock"):
        if not paths.meta.exists():
            return n_new
        meta = json.loads(paths.meta.read_text())
        meta["judge"] = config.to_dict()
        atomic_json(paths.meta, meta)
    return n_new
