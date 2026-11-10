"""Offline FSM replay over a recorded response.

Replay is how a recorded rollout is turned back into a mode trace, a mask trace and the
metric dict, on a CPU, with no engine. It is the oracle for the parity harness and it is
what `dattn eval` uses to fill `dac_metrics` for a row when the engine's own step records
are not being collected.

One thing to get right, which dac gets wrong: **replay must tokenize with the same
tokenizer state the server generated under.** dac's replay builds its tokenizer with a
plain `AutoTokenizer.from_pretrained` (`run_eval.py:249`) and never installs the bundled
chat template the server installs (`serve.py:1120`), so on Qwen the replayed prompt is
**315 tokens LONGER** than the served one (11561 vs 11246 on a 5-segment fixture) and
every segment span shifts by that amount, while `run_eval.py:274-275` claims byte-identity
The extra tokens land outside the sink, the local window and every span, so the
FOCUS-step mask area is unchanged while the denominator grows, and the per-row
`attention_ratio` is biased **low, in DA's favour**. The attended-token totals the paper
plots move THE OTHER WAY, because GLOBAL steps attend the extra tokens as well, inflating
Qwen's DA totals by 0.45 per cent and DA-nm's by 0.73 per cent. Gemma
differs by 2 tokens. dattn passes the rollout's exact rendered layout directly into
replay_record. Offline replay_row reads the SAME RolloutConfig and template hash from
meta.json, with no independent template-policy option, and verifies recorded prompt
hashes and token counts when present. dac_bundled is the campaign default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from dattn.contracts import Mask, Mode, PromptLayout
from dattn.fsm.machine import DeclarativeFSM

__all__ = ["SKIP_SPECIAL_TOKENS", "dense_trace", "replay", "replay_ids", "replay_record", "replay_row"]

#: Must equal `dattn.engine.processor.SKIP_SPECIAL_TOKENS`, which is dac's convention
#: (`src/dac/dac_state.py:715`). It is restated here rather than imported so the CPU eval
#: path does not pull in the engine, and `test_replay.py` pins the two equal. The
#: convention is load-bearing: once a replayed metric IS the recorded metric, a mismatch
#: between the two would surface as a parity failure with no local cause.
SKIP_SPECIAL_TOKENS: Final[bool] = True


def replay_ids(
    token_ids: list[int], token_texts: list[str], layout: PromptLayout
) -> tuple[list[Mode], list[Mask], dict[str, float | int | str]]:
    """Replay from already-tokenized text. The tokenizer-free core, for unit tests."""
    if len(token_ids) != len(token_texts):
        raise ValueError(f"{len(token_ids)} ids but {len(token_texts)} texts")
    fsm = DeclarativeFSM(layout)
    modes: list[Mode] = []
    masks: list[Mask] = []
    for tid, text in zip(token_ids, token_texts, strict=True):
        # The mode and mask in force for THIS forward, captured before `step` applies any
        # transition. That ordering is dac's and it is what the published numbers use.
        modes.append(fsm.mode)
        masks.append(fsm.mask)
        fsm.step(tid, text)
    return modes, masks, fsm.metrics()


def replay(
    response: str, layout: PromptLayout, tokenizer: Any
) -> tuple[list[Mode], list[Mask], dict[str, float | int | str]]:
    """Return the per-step mode trace, the per-step mask trace and the full metrics.

    `response` is the generated text only, never the prompt. `add_special_tokens=False`,
    because the server sampled continuation tokens and never re-encoded a BOS.
    """
    ids = list(tokenizer(response, add_special_tokens=False)["input_ids"])
    texts = [tokenizer.decode([i], skip_special_tokens=SKIP_SPECIAL_TOKENS) for i in ids]
    return replay_ids(ids, texts, layout)


def dense_trace(n_generated_tokens: int, layout: PromptLayout) -> dict[str, Any]:
    """The trace of a NON-DA arm, computed arithmetically, with no FSM and no replay.

    The FSM must not run on a non-DA arm because no declarative mask was active during
    generation. Baseline attended-token totals must instead represent dense attention,
    so comparisons against the same model's DA arm use the correct denominator.

    A non-DA arm attends everything, by definition, so its baseline needs no replay: at
    step `i` (0-based) the row attends the whole sequence, whose length is
    `n_prompt_tokens + 1 + i`, the same `now` convention `DeclarativeFSM.step` uses
    (`fsm/machine.py:78`). Summing that closed form gives

        total_attendable = T * (P + 1) + T * (T - 1) / 2

    and `attended_tokens == total_attendable`, `attention_ratio == 1.0`, every step GLOBAL.
    `tests/unit/eval/test_replay.py` pins this against a real `replay_ids` over a response
    with no focus tag, so the arithmetic and the FSM are checked to agree rather than
    asserted to.

    The five dac keys the index method cannot produce stay absent, and the focus counters
    are the honest zeros of an arm that never attempted a focus. `attended_tokens_blocks`
    stays `None` here as everywhere outside a recorder-armed run (PROVENANCE P8).
    """
    steps = max(0, int(n_generated_tokens))
    prompt = int(layout.n_prompt_tokens)
    attendable = steps * (prompt + 1) + steps * (steps - 1) // 2
    metrics: dict[str, Any] = {
        "total_steps": steps,
        "attended_tokens": attendable,
        "total_attendable": attendable,
        "attention_ratio": 1.0 if attendable else 0.0,
        "focus_steps": 0,
        "focus_ratio": 0.0,
        "local_steps": 0,
        "local_ratio": 0.0,
        "global_steps": steps,
    }
    return {
        "dac_metrics": metrics,
        "mode_trace": [int(Mode.GLOBAL)] * steps,
        "attended_tokens_exact": attendable,
        "attended_tokens_blocks": None,
        "total_attendable": attendable,
    }


def replay_record(
    response: str,
    layout: PromptLayout,
    tokenizer: Any,
    *,
    block_size: int = 0,
) -> dict[str, Any]:
    """The five `ResponseRecord` trace fields a recorded rollout carries, from one replay.

    Returns exactly `dac_metrics`, `mode_trace`, `attended_tokens_exact`,
    `attended_tokens_blocks` and `total_attendable`, so the caller spreads it into
    `ResponseRecord(**trace)` and the runner never reimplements the step-to-position
    arithmetic. `dac_metrics` is dattn's 25 keys, which is dac's 29 minus the five
    search-method keys the index method cannot produce, plus `global_steps`.

    **`total_steps` is within 1 of `n_generated_tokens`, and that is correct.** Replay
    reads the response TEXT, not the sampled ids. The server detokenizes with
    `skip_special_tokens=True`, so the final EOS never reaches replay while it did reach
    the engine's FSM, which stepped over it. dac's replay worker has exactly this property
    and the paper's `total_steps`, `attended_tokens` and non-termination numbers ARE dac's
    replayed numbers, so reproducing it is the faithful choice. Do not "fix" it.

    `block_size` defaults to 0, which leaves `attended_tokens_blocks` as `None`, and the
    eval path must leave it there. `docs/PROVENANCE.md` P8 defines the block-rounded count
    as a device-read, post-compaction number (`dattn.metrics.record`); a CPU replay
    estimate is a different number with the same name, so only a recorder-armed run may
    fill that field. The parameter exists so the other branch is one argument away if P8 is
    ever amended to name both sources.
    """
    modes, masks, metrics = replay(response, layout, tokenizer)
    blocks: int | None = None
    if block_size > 0:
        # `now` reproduces `DeclarativeFSM.step`'s `self.n_seen + 1` (`fsm/machine.py:78`).
        blocks = sum(m.round_to_blocks(block_size).area(layout.n_prompt_tokens + 1 + i) for i, m in enumerate(masks))
    return {
        "dac_metrics": dict(metrics),
        "mode_trace": [int(m) for m in modes],
        "attended_tokens_exact": int(metrics["attended_tokens"]),
        "attended_tokens_blocks": blocks,
        "total_attendable": int(metrics["total_attendable"]),
    }


def replay_row(
    row: Mapping[str, Any],
    *,
    tokenizer: Any = None,
    run_meta: Mapping[str, Any] | None = None,
) -> dict[str, float | int | str]:
    """Re-render one recorded row's prompt, replay its response, return the metrics.

    `row` must carry `context` as well as the recorded `question` and `response`. dac's
    `responses.jsonl` does NOT store the context (it stores `raw_context_tokens` only), so
    the caller joins the rows against `dattn.eval.datasets.select_rows` first. `dattn
    replay` consumes the same run metadata. The caller must do this join.
    """
    from dattn.eval.prompts import build_prompt
    from dattn.eval.sampling import RolloutConfig
    from dattn.prompt.template_policy import resolve_template, template_metadata

    if tokenizer is None:
        raise ValueError("replay_row needs the served model's tokenizer")
    if "context" not in row:
        raise KeyError(
            "replay_row needs `context`. dac's responses.jsonl does not carry it: join "
            "the rows against dattn.eval.datasets.select_rows first."
        )
    if run_meta is None:
        raise ValueError("replay_row requires the rollout meta.json, not an independent rendering policy")
    config = RolloutConfig(**run_meta["rollout"])
    if row.get("segmenter", config.segmenter) != config.segmenter:
        raise ValueError("replay segmenter disagrees with recorded rollout")
    if config.thinking:
        raise ValueError("rollout rendering pins enable_thinking=False for dac parity")
    template = resolve_template(tokenizer, config.model_id, config.template_policy)
    if run_meta.get("rendering") != template_metadata(template, config.template_policy):
        raise ValueError("replay template policy/hash disagrees with rollout metadata")
    built = build_prompt(
        tokenizer,
        context=str(row["context"]),
        question=str(row.get("question", "")),
        fmt=config.prompt_format,
        tool_schema=config.tool_schema,
        chat_template=template,
        segmenter=config.segmenter,
        vanilla_system=config.vanilla_system,
    )
    if row.get("prompt_sha256") and row["prompt_sha256"] != built.sha256:
        raise ValueError("replayed prompt hash disagrees with recorded rollout")
    if row.get("rendered_prompt_tokens") is not None and row["rendered_prompt_tokens"] != built.n_prompt_tokens:
        raise ValueError("replayed prompt token count disagrees with recorded rollout")
    response = str(row.get("response", ""))
    if not config.dattn_enabled:
        steps = row.get("n_generated_tokens")
        if steps is None:
            steps = len(tokenizer(response, add_special_tokens=False)["input_ids"])
        return dense_trace(int(steps), built.layout)["dac_metrics"]
    _, _, metrics = replay(response, built.layout, tokenizer)
    return metrics
