"""`DattnLogitsProcessor`.

Registered by fully qualified class name through `ModelConfig.logits_processors`
(`--logits-processors dattn.engine.processor:DattnLogitsProcessor`), NEVER by the
`vllm.logits_processors` entry point (DESIGN.md section 2.7).

This class keeps EVERY vLLM-facing assumption (batch-index semantics, token-id source,
lifecycle) inside itself and calls only `DeclarativeFSM` and `DattnState`, so the MRV2
rewrite (upstream #56497) changes one file.

It never touches logits: `apply()` returns its argument unchanged, because the whole
point of the processor is the side effect on `DattnState`, which the attention metadata
builder reads later in the same step.

`is_argmax_invariant()` nevertheless returns **False**. `Sampler` applies the
argmax-invariant list only on the random-sampling branch and the non-invariant list only
on the greedy branch (`N:v1/sample/sampler.py:283,401`). For a greedy request with
`temperature=0.0`, an argmax-invariant processor's `apply` is dead code.
That is harmless today, because dattn's `apply` is the identity and
`update_state` runs for every processor regardless. Walltime uses the model's recommended sampled recipe
(`N:v1/worker/gpu_input_batch.py:853-854`). It would stop being harmless the moment
anyone put anything in `apply`, and "an FSM that rewrites the attention mask does not
change the argmax" is true by accident, not by construction. Measured on hardware in
`research/NOTES.md` section 0.2 and asserted at boot (A14).
"""

from __future__ import annotations

import json
import time
from typing import Any, Final

from dattn.contracts import (
    DattnInvariantError,
    DattnRequest,
    Mode,
    Tensor,
)
from dattn.engine import state as _state
from dattn.engine.state import DATTN_STATE, RequestState

__all__ = ["DattnLogitsProcessor", "EXTRA_ARGS_KEY"]

#: Where the per-request payload lives inside `SamplingParams.extra_args`.
EXTRA_ARGS_KEY = "dattn"

#: dac `src/dac/dac_state.py:715` and `:540-547`, both of which decode with
#: `skip_special_tokens=True`. A special token therefore decodes to the empty string, so it
#: never enters the `SCAN_CHARS` tail and can never carry the `>` that triggers a scan.
#: Keeping them (the pre-fix behaviour) let `<|im_end|>`, `<end_of_turn>`, `<think>` and
#: `</think>` split a `<focus ...>` tag across the window and turn a valid focus into
#: `focus_fail_syntax_error`: the sequence `['<focus', ' magic_chunks', '="1"', X, '>']`
#: gives FOCUS with `focus_attempts == 1` when X decodes empty, and GLOBAL with
#: `focus_attempts == 2` and `focus_fail_syntax_error == 2` when X is `<|im_end|>`.
#: `dattn.eval.replay.SKIP_SPECIAL_TOKENS` must stay equal to this.
SKIP_SPECIAL_TOKENS: Final[bool] = True

_NO_TOKEN_GRACE = 3


def _base_class() -> type:
    """vLLM's `LogitsProcessor` ABC when vllm is importable, else `object`.

    `_load_logitsprocs_by_fqcns` does `issubclass(obj, LogitsProcessor)`, so the real
    base is not optional in production. It is optional in a unit test on a box with no
    vllm, which is the only reason this indirection exists.
    """
    try:
        from vllm.v1.sample.logits_processor import LogitsProcessor
    except Exception:  # pragma: no cover - vllm is a hard dependency in production
        return object
    return LogitsProcessor


_Base = _base_class()


class _TokenText:
    """Token id -> text, memoised. The FSM scans at most `SCAN_CHARS` of the tail."""

    def __init__(self, tokenizer: Any) -> None:
        self._tok = tokenizer
        self._cache: dict[int, str] = {}

    def __call__(self, token_id: int) -> str:
        text = self._cache.get(token_id)
        if text is None:
            text = "" if self._tok is None else self._tok.decode([token_id], skip_special_tokens=SKIP_SPECIAL_TOKENS)
            self._cache[token_id] = text
        return text


class DattnLogitsProcessor(_Base):  # type: ignore[misc, valid-type]
    """Steps the FSM once per decode step and writes the mask rows for this step.

    Ordering (DESIGN.md section 3): `update_state` runs inside
    `InputBatch.refresh_metadata()`, which the model runner calls from `_update_states`,
    strictly before `_prepare_inputs` and `_build_attention_metadata`. So the mask
    written here is read by THIS step's `build()`: there is no lag in the transport.
    The only lag is in the token ids, and it is **1 in both scheduling modes**, measured
    on MRV1 over four runs and both models (`research/NOTES.md` section 1). The
    design predicted 2 under `--async-scheduling`, on the reasoning that step `k` appends
    a `-1` placeholder resolved only inside step `k+1`'s `_sample`; `_advance` stops at
    the first negative id either way, so a lag of 2 would be safe and merely slower.
    """

    def __init__(
        self,
        vllm_config: Any = None,
        device: Any = None,
        is_pin_memory: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.device = device
        self.state = DATTN_STATE
        self.fsm_lag = 1
        self.n_steps = 0
        self.n_neg_stalls = 0
        self._decode: _TokenText | None = None
        self._anon = 0
        if vllm_config is not None:
            sched = vllm_config.scheduler_config
            # Measured, not assumed: the lag is 1 in BOTH scheduling modes on MRV1
            # (research/NOTES.md section 1, four runs, both models). The design's
            # "2 under async" was a prediction and is refuted. `_advance` stops at the
            # first negative id anyway, so a lag of 2 would be safe, just slower.
            self.fsm_lag = 1
            self.state.configure(int(sched.max_num_seqs), device, pin_memory=bool(is_pin_memory))

    # -- the vLLM LogitsProcessor ABC ---------------------------------------------------

    def is_argmax_invariant(self) -> bool:
        # False on purpose. See the module docstring and research/NOTES.md 0.2.
        return False

    def apply(self, logits: Tensor) -> Tensor:
        return logits

    def update_state(self, batch_update: Any | None) -> None:
        """removed -> added -> moved, the documented order, then advance and flush."""
        t0 = time.perf_counter()
        st = self.state
        if not st.configured:
            return
        st.begin_step()
        self.n_steps += 1
        rec = _state.get_recorder()
        if batch_update is not None:
            for index in batch_update.removed:
                st.remove_row(int(index))
            for entry in batch_update.added:
                self._add(entry)
            for src, dst, direction in batch_update.moved:
                if _is_swap(direction):
                    st.swap_rows(int(src), int(dst))
                else:
                    st.move_row(int(src), int(dst))

        rows: list[tuple[str, int, int, int, int, int]] = []
        n_live = st.n_live_rows()
        for row in range(n_live):
            req = st.rows[row]
            if req is None:
                continue
            self._advance(row, req)
            if rec is not None:
                seq_len = req.seq_len()
                rows.append(
                    (
                        req.req_id,
                        int(req.mode),
                        # Slot 2 is the VANILLA KV length, not the batch row index: it is
                        # the denominator of every ratio the analysis bins on, and the
                        # rows are already in slot order so the index carries nothing the
                        # position does not.
                        seq_len,
                        req.mask.area(seq_len),
                        req.out_pos(),
                        st.step_id - req.admitted_at,
                    )
                )
        st.flush()

        if rec is not None:
            rec.note_lp((time.perf_counter() - t0) * 1e6, rows)

    # -- internals ----------------------------------------------------------------------

    def _add(self, entry: tuple) -> None:
        """`AddedRequest` is `(index, params, prompt_tok_ids, output_tok_ids)` on 0.29.0."""
        index, params, prompt_tok_ids, output_tok_ids = entry
        n_prompt = len(prompt_tok_ids) if prompt_tok_ids is not None else 0
        payload = _payload(params)
        if payload is None:
            # A request with no dattn payload is served densely. It still owns its row:
            # an empty slot would retain the previous occupant's mask and break the
            # alignment of the recorder's per-row arrays.
            key = f"anon@{id(output_tok_ids):x}"
            self.state.admit(int(index), key, key, None, output_tok_ids, n_prompt)
            return
        req = DattnRequest.from_extra_args(payload)
        key = str(payload.get("req_id") or f"anon@{id(output_tok_ids):x}")
        req_id = str(payload.get("req_id") or key)
        if getattr(getattr(self.vllm_config, "cache_config", None), "enable_prefix_caching", False):
            # Child SamplingParams share the parent payload. The live output list
            # is unique per child and stable across batch moves and readmission.
            key = f"{key}@{id(output_tok_ids):x}"
            existing = self.state.by_key.get(key) or self.state.detached.get(key)
            if existing is None:
                # List addresses can be reused after finished records are evicted.
                # Recorder labels must remain unique for the engine lifetime.
                self._anon += 1
                req_id = f"{key}#{self._anon}"
            else:
                req_id = existing.req_id
        st = self.state.admit(int(index), key, req_id, req, output_tok_ids, n_prompt)
        if st.n_prompt == 0:
            st.n_prompt = req.layout.n_prompt_tokens
        if st.fsm is None and req.forced_mask is None and req.enabled:
            st.fsm = _make_fsm(req.layout)
        if prompt_tok_ids is not None and req.layout.n_prompt_tokens != len(prompt_tok_ids):
            raise DattnInvariantError(
                f"request {req_id}: the layout says n_prompt_tokens="
                f"{req.layout.n_prompt_tokens} but vLLM scheduled {len(prompt_tok_ids)} "
                "prompt tokens. The layout is computed on the client and must describe "
                "the rendered prompt exactly (DESIGN.md section 1.1, F D11)."
            )

    def _advance(self, row: int, req: RequestState) -> None:
        req.n_updates += 1
        out = req.output_ids
        n = len(out)

        if req.fsm is None:
            # No payload, a forced mask, or DA disabled: the row never changes after
            # admission. There is nothing to feed and nothing to guard.
            return

        if n == 0:
            # DECODE steps, not steps. A chunked prefill occupies the row for as many
            # steps as the prompt needs chunks, and every one of them reaches here with
            # an empty output list, because `update_state` runs on every engine step
            # whether the batch changed or not (`N:v1/worker/gpu_input_batch.py:838-854`)
            # and a request joins `InputBatch` at its first scheduled CHUNK, not at its
            # first sampled token (`N:v1/worker/gpu_model_runner.py:1522`). Counting
            # those as failed decodes raised on the fourth chunk, which at
            # `max_num_batched_tokens = 8192` is any prompt past about 25K tokens: the
            # whole long-context workload. `n_decode_steps` is credited by
            # `DattnState._gate_for` from `query_len == 1` and `is_prefilling`, which is
            # the only place in the step that knows.
            if req.n_decode_steps > _NO_TOKEN_GRACE and req.n_preemptions == 0:
                raise DattnInvariantError(
                    f"request {req.req_id} has taken {req.n_decode_steps} genuine decode "
                    f"steps (query_len == 1, not prefilling) across {req.n_updates} steps "
                    "in the batch and its output token id list is still empty. A decode "
                    "step samples a token, so the list dattn was handed is not the list "
                    "vLLM appends to, and the FSM will never advance. Check that "
                    "DattnLogitsProcessor was installed by fully qualified class name "
                    "(--logits-processors dattn.engine.processor:DattnLogitsProcessor, "
                    "DESIGN.md section 2.7) and that the boot assertions ran (A14). The "
                    "FSM would freeze silently, so dattn crashes instead."
                )
            return

        changed = False
        while req.n_fed < n:
            tid = out[req.n_fed]
            if tid < 0:
                # The async-scheduling placeholder. Stop here and retry next step; this
                # is what makes the design async-safe rather than async-dependent.
                self.n_neg_stalls += 1
                break
            new = req.fsm.step(int(tid), self._text(int(tid)))
            req.n_fed += 1
            if new is not None:
                req.mask = new
                changed = True
        req.mode = int(getattr(req.fsm, "mode", Mode.GLOBAL))
        if changed:
            self.state.set_mask(row, req.mask)

    def _text(self, token_id: int) -> str:
        if self._decode is None:
            self._decode = _TokenText(_tokenizer(self.vllm_config))
        return self._decode(token_id)


def _is_swap(direction: Any) -> bool:
    return getattr(direction, "name", str(direction)).upper().endswith("SWAP")


def _payload(params: Any) -> dict | None:
    """The `dattn` payload, from either transport.

    Offline (`LLM.generate`) it arrives as the dict `DattnRequest.to_extra_args()` returns.
    Over HTTP it arrives as a JSON **string**: vLLM 0.29.0 types `vllm_xargs` as
    `dict[str, str | int | float | list[...]]`
    (`N:entrypoints/openai/chat_completion/protocol.py:492`), so a nested dict cannot be
    sent and `dattn.eval.prompts.dattn_xargs` encodes it. Accept both forms so offline
    and HTTP requests use the same admission path.
    """
    extra = getattr(params, "extra_args", None)
    if not extra:
        return None
    payload = extra.get(EXTRA_ARGS_KEY)
    if payload is None:
        return None
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except ValueError as exc:
            raise DattnInvariantError(
                f'extra_args["{EXTRA_ARGS_KEY}"] is a string but not JSON: {exc}. Over HTTP the '
                "payload travels as the JSON string dattn.eval.prompts.dattn_xargs produces."
            ) from None
    if not isinstance(payload, dict):
        raise DattnInvariantError(
            f'extra_args["{EXTRA_ARGS_KEY}"] must be the dict DattnRequest.to_extra_args() '
            f"returns, or that dict as a JSON string, got {type(payload).__name__}"
        )
    return payload


def _make_fsm(layout: Any) -> Any:
    from dattn.fsm.machine import DeclarativeFSM

    return DeclarativeFSM(layout)


#: Where 0.29.0 keeps the tokenizer factory. It is `vllm.tokenizers.registry`, NOT
#: `vllm.transformers_utils.tokenizer`, which is where it lived up to 0.2x and which no
#: longer exists. Pinned by `tests/contract/test_tokenizer_factory.py`, because the FSM's
#: tail scan needs the decoded text of every sampled token and an ImportError here takes
#: the whole EngineCore down on the first decode step of the first real request.
TOKENIZER_FACTORY: Final[tuple[str, str]] = ("vllm.tokenizers.registry", "cached_tokenizer_from_config")


def _tokenizer(vllm_config: Any) -> Any:
    """The served model's tokenizer, from the resolved config, cached by vLLM itself."""
    if vllm_config is None:
        return None
    import importlib

    module, attr = TOKENIZER_FACTORY
    from_config = getattr(importlib.import_module(module), attr)
    tok = from_config(vllm_config.model_config)
    if tok is None:
        raise DattnInvariantError(
            "the engine was booted with skip_tokenizer_init, so there is no tokenizer to decode "
            "sampled tokens with. The FSM scans the decoded tail of the response for its tags "
            "(DESIGN.md 1.1), so it cannot run on a tokenizer-less engine."
        )
    return tok
