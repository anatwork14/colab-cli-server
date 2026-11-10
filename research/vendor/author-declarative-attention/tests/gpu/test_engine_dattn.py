"""End-to-end DA on a real engine: boot, needle, parity and the cudagraph question.

Four things are proved here and nowhere else:

1. **The boot assertion list passes on the real backends** (DESIGN.md section 6), which is
   the only proof that `backend_per_kind = {"full_attention": "CUSTOM"}` actually bound
   and that every other builder is untouched.
2. **The needle test.** dac's own guide records that "the mask silently did nothing for
   weeks". A planted fact inside a focused segment must come back, and the same fact with
   the segment excluded must not. No walltime or accuracy number may be published from a
   run that did not pass this test in the same process (DESIGN.md section 7, GPU test 2).
3. **The forced-GLOBAL arm reproduces vanilla greedy output token for token.** GLOBAL is
   the dense mask, a dense row is a fixed point of the compaction, so the whole of Lane B
   -- plugin, backend, builder, logits processor, state store, scratch -- must be a
   no-op. Any divergence is a bug in the identity path, and the identity path is what
   every masked step also runs through. **It is a statement about the MASK, and about
   nothing else.** Its "DA-disabled" comparison arm carries `DattnRequest(enabled=False)`,
   which is the REQUEST-level flag: that request's group is still compacted on every step,
   so this checks request-level masking.
3b. **Captured DA scratch is live.** Poison beyond the staged columns must leave
   logprobs unchanged. Corrupting the staged lengths must change them.
4. **Whether a replayed FULL cudagraph honours the shrunken `seq_lens`.**
   `research/NOTES.md` calls this "the single most important remaining question":
   its `_dummy_run` harness was flat between a true context of 1024 and 32768 under a
   replayed graph. This test asks the same question on the real `execute_model` path,
   which is where dattn actually lives and where the capture is staged through dattn's
   own scratch so the graph binds to our pointers.

Engines are booted one at a time and torn down (see `conftest.shutdown_engine`). Budget
about three minutes per boot.
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
from conftest import (
    GPU_MEM_UTIL,
    MAX_MODEL_LEN,
    MAX_NUM_SEQS,
    MODELS,
    NEEDLE,
    build_needle_prompt,
    get_engine,
    shutdown_engine,
)

from dattn.contracts import DattnRequest, Mask
from dattn.engine import launch
from dattn.engine.profile import load_profile

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

OVERRIDES = {
    "max_model_len": MAX_MODEL_LEN,
    "max_num_seqs": MAX_NUM_SEQS,
    "gpu_memory_utilization": GPU_MEM_UTIL,
    "enforce_eager": False,
}


def _profile(model_id: str):
    return load_profile(model_id, "sm100", os.environ.get("DATTN_TEST_PROFILE", "parity"))


def _boot(model_id: str, *, vanilla: bool = False, **extra):
    profile = _profile(model_id)
    launch.apply_env(profile)
    os.environ["DATTN_ARM"] = "vanilla" if vanilla else "da"
    if profile.use == "walltime":
        kwargs = launch.engine_kwargs(profile, **extra)
        kwargs["compilation_config"] = dict(
            kwargs.get("compilation_config", {}),
            cudagraph_capture_sizes=list(range(1, kwargs["max_num_seqs"] + 1)),
        )
    else:
        kwargs = launch.engine_kwargs(profile, **OVERRIDES, **extra)
    if vanilla:
        kwargs = launch.builder_free_kwargs(kwargs)
    key = f"{model_id}|{'vanilla' if vanilla else 'dattn'}|{sorted(extra.items())}"
    return get_engine(key, **kwargs), profile


def _sampling(mask: Mask | None, prompt, *, max_tokens: int, req_id: str, enabled: bool = True):
    from vllm import SamplingParams

    extra = None
    if enabled or mask is not None:
        payload = DattnRequest(layout=prompt.layout, enabled=enabled, forced_mask=mask).to_extra_args()
        payload["req_id"] = req_id
        extra = {"dattn": payload}
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, extra_args=extra)


def _generate(llm, prompt, params_list):
    from vllm import TokensPrompt

    prompts = [TokensPrompt(prompt_token_ids=list(prompt.ids)) for _ in params_list]
    return llm.generate(prompts, params_list)


# ---------------------------------------------------------------------------------------
# 1. the boot assertion list
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", MODELS)
def test_the_boot_assertions_pass_on_the_real_backend(require_cuda, model_id: str) -> None:
    from dattn.engine.api import dattn_probe
    from dattn.engine.assertions import validate_engine
    from dattn.engine.plugin import current_profile

    llm, profile = _boot(model_id)
    runner = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    assert current_profile() is not None, "the general plugin was inert; DATTN_MODEL was not exported"

    report = validate_engine(runner.vllm_config, runner, profile, raise_on_failure=False)
    hard = [f for f in report["failures"] if not f.startswith(("A17",))]
    assert not hard, "\n".join(hard)

    resolved = report["resolved"]
    groups = resolved["kv_cache_groups"]
    dattn_gids = [g["kv_cache_group_id"] for g in groups if g["is_dattn"]]
    assert dattn_gids == [g["kv_cache_group_id"] for g in groups if g["kind"] == "full_attention"]
    assert resolved["dattn_block_sizes"] == [profile.dattn_block_size]
    assert all(not g["supports_update_block_table"] for g in groups)
    assert resolved["runner_class"] == "vllm.v1.worker.gpu_model_runner.GPUModelRunner"

    probe = dattn_probe(None)
    for key in (
        "backend_cls",
        "use_v2_model_runner",
        "async_scheduling",
        "enable_prefix_caching",
    ):
        assert key in probe, f"meta.json key {key} is missing (Lane C interface request C5)"
    assert probe["audit"]["mismatches"] == 0


# ---------------------------------------------------------------------------------------
# 2. the needle test
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", MODELS)
def test_mask_hides_excluded_context(require_cuda, model_id: str) -> None:
    llm, profile = _boot(model_id)
    prompt = build_needle_prompt(llm.get_tokenizer())
    assert prompt.layout.n_prompt_tokens < MAX_MODEL_LEN

    outs = _generate(
        llm,
        prompt,
        [
            _sampling(prompt.focused(), prompt, max_tokens=24, req_id="focused"),
            _sampling(prompt.excluded(), prompt, max_tokens=24, req_id="excluded"),
            _sampling(prompt.dense(), prompt, max_tokens=24, req_id="dense"),
        ],
    )
    focused, excluded, dense = (o.outputs[0].text for o in outs)
    print(f"\n[{model_id}] focused  : {focused!r}")
    print(f"[{model_id}] excluded : {excluded!r}")
    print(f"[{model_id}] dense    : {dense!r}")

    assert NEEDLE in dense, f"the model cannot answer even with the whole context: {dense!r}"
    assert NEEDLE in focused, (
        f"the needle chunk was kept but the model did not see it: {focused!r}. "
        "Either the compaction dropped a kept block or the mask was not applied."
    )
    assert NEEDLE not in excluded, (
        f"the needle chunk was EXCLUDED and the model still produced it: {excluded!r}. "
        "That is the failure dac's guide records as 'the mask silently did nothing for weeks'."
    )

    state = llm.llm_engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
    del state
    from dattn.engine.state import DATTN_STATE

    assert DATTN_STATE.counters.rows_compacted > 0, "no row was ever compacted"
    assert DATTN_STATE.audit()["mismatches"] == 0


# ---------------------------------------------------------------------------------------
# 3. forced GLOBAL == vanilla, token for token
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", MODELS)
def test_forced_global_reproduces_vanilla_greedy_output(require_cuda, model_id: str) -> None:
    from vllm import SamplingParams, TokensPrompt

    max_tokens = 48
    # Pin the KV pool in bytes, identically for both arms. Sized from
    # `gpu_memory_utilization` it is NOT identical: the vanilla engine boots on an empty
    # device and the DA engine boots after it, on whatever the teardown gave back, so the
    # two pools differ by a GiB and the block count with them. That is a configuration
    # difference between the arms, and a greedy decode that has run past its EOS into
    # degenerate text sits on near-ties where any change of reduction order flips a token.
    pinned = {} if _profile(model_id).use == "walltime" else {"kv_cache_memory_bytes": 32 << 30}
    # Vanilla first, then DA: one engine alive at a time.
    llm, _ = _boot(model_id, vanilla=True, **pinned)
    prompt = build_needle_prompt(llm.get_tokenizer())
    # The vanilla arm stops at its EOS. Running it past the stop with `ignore_eos` is what
    # the same-engine comparison below does, but it must not be done ACROSS engines: the
    # post-EOS continuation is degenerate text where the top two logits are a hair apart,
    # and two separately booted engines of the SAME vanilla configuration were measured
    # disagreeing there (one boot emitted `-lantern-881`, the next `\n\nde\nre`, both
    # vanilla). vLLM makes no cross-boot bitwise promise with batch invariance off, so a
    # comparison there measures the baseline against itself.
    plain = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    vanilla = llm.generate([TokensPrompt(prompt_token_ids=list(prompt.ids))], [plain])
    vanilla_ids = list(vanilla[0].outputs[0].token_ids)
    vanilla_text = vanilla[0].outputs[0].text
    # Drop this frame's handle before tearing down, or the weights cannot be collected
    # and the DA boot below runs out of memory.
    llm = vanilla = None
    free = shutdown_engine()
    print(f"\n[{model_id}] {free:.1f} GiB free after tearing the vanilla engine down")

    llm, _ = _boot(model_id, **pinned)

    def one(params, *, past_eos: bool) -> tuple[list[int], str]:
        # One request per generate call, so every arm runs at batch size 1. The batch
        # composition is not cosmetic: a decode kernel's split-K reduction order depends
        # on it, and two floating-point orders disagree on a near-tied argmax. That is a
        # property of the baseline against itself, not of DA, and comparing arms at
        # different batch sizes would measure it instead of the identity path.
        params.ignore_eos = past_eos
        out = llm.generate([TokensPrompt(prompt_token_ids=list(prompt.ids))], [params], use_tqdm=False)
        return list(out[0].outputs[0].token_ids), out[0].outputs[0].text

    deep = {"past_eos": True}
    forced_ids, forced_text = one(
        _sampling(prompt.dense(), prompt, max_tokens=max_tokens, req_id="forced-global"), **deep
    )
    disabled_ids, disabled_text = one(
        _sampling(None, prompt, max_tokens=max_tokens, req_id="disabled", enabled=False), **deep
    )
    cross_ids, cross_text = one(
        _sampling(None, prompt, max_tokens=max_tokens, req_id="cross", enabled=False), past_eos=False
    )

    def diff(a: list[int], b: list[int]) -> int | None:
        return next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)

    assert NEEDLE in vanilla_text, f"the vanilla arm did not answer: {vanilla_text!r}"
    assert len(forced_ids) == len(disabled_ids) == max_tokens, "the deep arms must run the full horizon"
    # The claim in two halves, because they fail for different reasons.
    #
    assert forced_ids == disabled_ids, (
        f"forced GLOBAL diverged from a payload-disabled request in the SAME engine at token "
        f"{diff(forced_ids, disabled_ids)}.\n"
        f"disabled: {disabled_text!r}\n"
        f"forced  : {forced_text!r}\n"
        "GLOBAL is the dense mask and a dense row is a fixed point of the compaction, so "
        "the whole of Lane B -- plugin, backend, builder, logits processor, state store, "
        "scratch -- must be a no-op here."
    )
    # 2. Across the two ENGINES, the payload-disabled request must equal vanilla. This one
    #    is about the engine configuration, not about the mask, and it is the half that
    #    a float-order difference between two separately booted engines can break: with
    #    `ignore_eos` the decode runs past its EOS into degenerate text where the top two
    #    logits are a hair apart. `first_diff` is reported so a failure can be read rather
    #    than guessed at.
    d = diff(cross_ids, vanilla_ids)
    assert cross_ids == vanilla_ids, (
        f"a payload-disabled request diverged from the vanilla engine at token {d} of "
        f"{len(vanilla_ids)} (same profile, same pinned KV pool, batch 1 on both, both "
        "stopping at their own EOS).\n"
        f"vanilla : {vanilla_text!r}\n"
        f"disabled: {cross_text!r}"
    )
    assert len(vanilla_ids) < max_tokens, (
        f"the vanilla arm never hit its EOS in {max_tokens} tokens, so the cross-engine "
        "comparison ran into the degenerate tail after all"
    )


# ---------------------------------------------------------------------------------------
# 3b. Captured DA scratch is live and bounded
# ---------------------------------------------------------------------------------------

#: A question whose greedy answer runs well past 64 tokens on both models, so the arms can
#: be compared over 64 tokens. All three scratch-poison conditions use the same DA engine.
LONG_ANSWER_Q = (
    "Summarise the archive notes for a colleague. Write at least eight full sentences, "
    "describing what the notes record and mentioning any codes or passphrases you find."
)

#: 64 greedy decode steps. Long enough that the sequence crosses several kernel blocks
#: (32 on Gemma, 16 on Qwen), which is what makes a frozen block table and a frozen
#: `seq_lens` produce different text rather than the same text a little cheaper.
CERT_STEPS = 64

#: How many logprobs per step the two arms are compared on. Token identity alone is a
#: coarse instrument: a baseline can attend to the wrong keys and still argmax the same id
#: for a while. The distribution cannot (review audit of 64d8b80).
CERT_TOPK = 5


def _topk(out: Any, k: int = CERT_TOPK) -> list[list[tuple[int, float]]]:
    """Per step, the `k` highest logprobs as `(token_id, logprob)`, highest first."""
    per_step: list[list[tuple[int, float]]] = []
    for d in out[0].outputs[0].logprobs or []:
        items = sorted(d.items(), key=lambda kv: -float(kv[1].logprob))[:k]
        per_step.append([(int(t), float(lp.logprob)) for t, lp in items])
    return per_step


@pytest.mark.parametrize("model_id", MODELS)
def test_da_capture_uses_live_scratch_and_ignores_unstaged_tail(require_cuda, monkeypatch, model_id: str) -> None:
    """A replay must read staged DA metadata, with no reads beyond its live columns."""
    import torch
    from vllm import TokensPrompt

    from dattn.engine.state import DATTN_STATE
    from dattn.metrics.hooks import ENV_RECORD

    monkeypatch.setenv(ENV_RECORD, "1")
    pinned = {} if _profile(model_id).use == "walltime" else {"kv_cache_memory_bytes": 32 << 30}
    llm, _ = _boot(model_id, **pinned)
    prompt = build_needle_prompt(llm.get_tokenizer(), question=LONG_ANSWER_Q)
    params = _sampling(prompt.dense(), prompt, max_tokens=CERT_STEPS, req_id="scratch")
    params.min_tokens = CERT_STEPS
    params.ignore_eos = True
    params.logprobs = CERT_TOPK
    original = DATTN_STATE.compact
    mode = "clean"
    poisoned = 0

    def compact(cm, builder):
        nonlocal poisoned
        staged = original(cm, builder)
        if int(cm.max_query_len) == 1:
            sc = DATTN_STATE._scratch[id(builder)]
            cols = min(sc.n_cols, -(-int(cm.max_seq_len) // sc.block_size))
            with torch.inference_mode():
                if mode == "tail" and cols < sc.n_cols:
                    # Block 1 is valid and differs from the zero-filled unused tail.
                    sc.block_table[:, cols:].fill_(1)
                    poisoned += 1
                elif mode == "garbage":
                    sc.seq_lens.fill_(1)
                    sc.seq_lens_cpu.fill_(1)
                    staged = staged.replace(max_seq_len=1)
                    poisoned += 1
        return staged

    monkeypatch.setattr(DATTN_STATE, "compact", compact)

    def window(label):
        # Keep all three prefills identical even when the profile enables caching.
        assert llm.reset_prefix_cache(), f"{label}: prefix cache did not reset"
        llm.collective_rpc("dattn_record", args=(True,), kwargs={"window_id": label})
        out = llm.generate([TokensPrompt(prompt_token_ids=list(prompt.ids))], [params], use_tqdm=False)
        assert all(r.num_cached_tokens == 0 for r in out), f"{label}: prefix cache changed the prefill"
        llm.collective_rpc("dattn_record", args=(False,))
        drained = llm.collective_rpc("dattn_drain")
        recs = drained[0] if drained and isinstance(drained[0], list) else drained
        full = [r for r in recs if r.kind == "decode" and r.cudagraph_mode == "FULL" and r.cudagraph_replayed]
        assert len(full) >= CERT_STEPS - 2, f"{label}: only {len(full)} FULL replays"
        return _topk(out)

    clean = window("clean")
    mode = "tail"
    tail = window("tail")
    assert poisoned > 0, "no unstaged scratch tail was poisoned"
    assert clean == tail, "the DA graph read beyond the staged columns"
    mode, poisoned = "garbage", 0
    garbage = window("garbage")
    assert poisoned > 0
    assert garbage != clean, "the DA graph ignored corrupted staged lengths"


# 4. does a replayed FULL cudagraph honour the shrink?
# ---------------------------------------------------------------------------------------


def _run_arm(llm, prompt, keep: float, *, batch: int, steps: int) -> float:
    from vllm import TokensPrompt

    mask = prompt.nearly_dense() if keep >= 0.999 else prompt.fraction(keep)
    params = [_sampling(mask, prompt, max_tokens=steps, req_id=f"t{keep}-{i}") for i in range(batch)]
    for p in params:
        p.ignore_eos = True
    prompts = [TokensPrompt(prompt_token_ids=list(prompt.ids)) for _ in params]
    t0 = time.perf_counter()
    llm.generate(prompts, params, use_tqdm=False)
    return time.perf_counter() - t0


def _decode_ms(llm, prompt, keep: float, *, batch: int, reps: int, short: int = 16, long: int = 144) -> float:
    """Per-decode-step milliseconds, with the prefill differenced out.

    A single `generate` of `N` tokens is dominated by the prefill at these context
    lengths: eight 8.7k-token prefills cost about seven seconds, which at `N = 64` is
    110 ms of the "per step" figure and swamps a 25 ms decode. Timing two lengths and
    taking the slope cancels the prefill, the sampler warmup and the harness overhead
    exactly, because both runs pay them once.
    """
    per_len: dict[int, float] = {}
    for n in (short, long):
        runs = sorted(_run_arm(llm, prompt, keep, batch=batch, steps=n) for _ in range(reps))
        per_len[n] = runs[len(runs) // 2]
    return (per_len[long] - per_len[short]) / (long - short) * 1e3


@pytest.mark.parametrize("model_id", MODELS)
def test_a_replayed_cudagraph_honours_the_shrunken_seq_lens(require_cuda, model_id: str) -> None:
    """U3's follow-up, on the real `execute_model` path (`gpu_unknowns.md` section 2.0).

    The capture is staged through dattn's own scratch, so the captured graph binds to the
    buffers `compact()` writes every step. If the shrink is honoured, the f=0.25 arm is
    faster; if it is not, the DA arms must be pinned to PIECEWISE and this test says so in
    its failure message.
    """
    llm, profile = _boot(model_id)
    prompt = build_needle_prompt(llm.get_tokenizer(), n_chunks=78, chunk_repeats=4)
    if prompt.layout.n_prompt_tokens > MAX_MODEL_LEN - 256:
        pytest.skip(f"prompt is {prompt.layout.n_prompt_tokens} tokens, over max_model_len")

    batch, reps = MAX_NUM_SEQS, 2
    # Warm the Triton JIT and the graph replay path before timing anything: Gemma's first
    # decode step logs a kernel_unified_attention JIT compile and is not representative.
    _run_arm(llm, prompt, 1.0, batch=batch, steps=8)

    full = _decode_ms(llm, prompt, 1.0, batch=batch, reps=reps)
    quarter = _decode_ms(llm, prompt, 0.25, batch=batch, reps=reps)
    ratio = quarter / full
    mode = llm.llm_engine.vllm_config.compilation_config.cudagraph_mode
    print(
        f"\n[{model_id}] cudagraph_mode={mode} prompt={prompt.layout.n_prompt_tokens} "
        f"B={batch}: f=1.0 {full:.3f} ms/step, f=0.25 {quarter:.3f} ms/step, ratio {ratio:.3f}"
    )

    assert ratio < 0.97, (
        f"a shrunken seq_lens bought nothing under {mode}: f=0.25 is {quarter:.3f} ms/step "
        f"against {full:.3f} at f=1.0 (ratio {ratio:.3f}). Either the replayed graph ignores "
        "the compacted seq_lens, in which case every DA arm must be pinned to "
        "--cuda-graph-mode PIECEWISE and docs/LANE_B.md must record it, or the step is "
        "weight-bound at this batch size and context (gpu_unknowns 2.2: B=8 is a dead zone "
        "on Qwen)."
    )


# ---------------------------------------------------------------------------------------
# 5. the plugin, which used to abort every boot
# ---------------------------------------------------------------------------------------
