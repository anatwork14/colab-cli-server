"""A DA request whose prompt needs many prefill chunks must not raise (review item 1).

`DattnLogitsProcessor._advance` runs on every engine step, and a request joins
`InputBatch` at its first scheduled CHUNK rather than at its first sampled token
(`N:v1/worker/gpu_model_runner.py:1522`, `N:v1/worker/gpu_input_batch.py:838-854`). The
no-token watchdog used to count those prefill iterations as decode steps that had failed
to produce a token and raise `DattnInvariantError` on the fourth, which kills EngineCore.
At `max_num_batched_tokens = 8192` (`bench/protocol.py`) that is any prompt past roughly
25K tokens, which is the whole long-context workload. The end-to-end test missed it
because its prompts were 6.4K to 9.9K tokens, i.e. one or two chunks.

The fix counts `RequestState.n_decode_steps`, credited by `DattnState._gate_for` from
`query_len == 1` and `is_prefilling`, so a prefill iteration is never counted. This test
is the proof on hardware: it boots the DA arm with a small token budget and a prompt well
past four chunks of it.

Run separately on an explicitly assigned GPU.

**It is parametrised over BOTH models on purpose.** It used to be `MODELS[:1]`, which is
Gemma, and the test id said `model_id0`, so invoking it the obvious way gave you Gemma
while the name suggested a choice had been made. Getting any Qwen coverage needed a pytest
plugin that reordered `MODELS`. The premise this test has to satisfy is TOKENIZER
dependent, so one model passing says nothing about the other: see `_long_needle_prompt`.

**The premise is asserted before the engine is touched.** The prompt must be longer than
`GRACE_CHUNKS * MAX_NUM_BATCHED_TOKENS`, or the old watchdog would never have fired and a
pass means nothing, and it must fit `MAX_MODEL_LEN` with room for the sampled tokens, or
vLLM's renderer rejects it before a single engine step. The first version of this test
checked only the lower bound and was UNSATISFIABLE on Qwen: the filler ladder steps from
25,597 tokens straight to 50,941, and `MAX_MODEL_LEN` was 40,960, so the only rung past
four chunks was also past the context limit. It failed 90 seconds into a boot with a
renderer message about output tokens, which names neither real cause.
"""

from __future__ import annotations

import pytest
from conftest import (
    GPU_MEM_UTIL,
    MAX_NUM_SEQS,
    MODELS,
    NEEDLE,
    build_needle_prompt,
    get_engine,
    shutdown_engine,
)

from dattn.contracts import DattnRequest
from dattn.engine import launch
from dattn.engine.profile import load_profile

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

#: The bench's own value (`bench/protocol.py`), so this reproduces the configuration that
#: crashes rather than a configuration chosen to crash.
MAX_NUM_BATCHED_TOKENS = 8192
#: `_NO_TOKEN_GRACE` is 3, so the old watchdog raised on the FOURTH prefill iteration. A
#: prompt of four chunks or fewer would never have reached it and a pass would be vacuous.
GRACE_CHUNKS = 4
#: 65,536 rather than 40,960. The filler ladder's only rung past four chunks is 50,941
#: tokens on Qwen, which 40,960 could not hold, so the test could not be satisfied on that
#: model at all. Raising the window makes the EXISTING rung legal at seven chunks and
#: changes nothing about the ladder or about what is being tested.
MAX_MODEL_LEN = 65_536
MAX_OUTPUT_TOKENS = 32

#: The band a usable prompt must land in. Both ends are real constraints, and the first
#: version of this test checked only the lower one.
MIN_PROMPT_TOKENS = GRACE_CHUNKS * MAX_NUM_BATCHED_TOKENS + 1  # 32_769
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_OUTPUT_TOKENS  # 65_504

#: Filler sizes, smallest first. The tokenizer decides where each one lands, which is why
#: a rung that fits one model can miss the band entirely on another.
CHUNK_REPEATS = (3, 6, 12, 24, 48, 96)


def _long_needle_prompt(tokenizer):
    """The first filler rung inside [MIN_PROMPT_TOKENS, MAX_PROMPT_TOKENS].

    Selecting on the lower bound alone is what made this test unsatisfiable on Qwen: the
    ladder steps 25,597 -> 50,941, and the old 40,960 window rejected the only rung that
    was past four chunks. A premise that cannot be met must say so HERE, naming every
    rung and both bounds, rather than 90 seconds later inside vLLM's renderer.
    """
    sizes = []
    for chunk_repeats in CHUNK_REPEATS:
        prompt = build_needle_prompt(tokenizer, n_chunks=24, chunk_repeats=chunk_repeats, needle_chunk=12)
        n = len(prompt.ids)
        sizes.append((chunk_repeats, n))
        if MIN_PROMPT_TOKENS <= n <= MAX_PROMPT_TOKENS:
            return prompt
    pytest.fail(
        f"no filler rung lands in [{MIN_PROMPT_TOKENS}, {MAX_PROMPT_TOKENS}] tokens, which is "
        f"'past {GRACE_CHUNKS} chunks of {MAX_NUM_BATCHED_TOKENS}' and 'inside max_model_len "
        f"{MAX_MODEL_LEN} less {MAX_OUTPUT_TOKENS} sampled tokens'. Rungs (chunk_repeats, tokens): "
        f"{sizes}. This is a property of THIS tokenizer, not of the engine"
    )
    return None


@pytest.mark.parametrize("model_id", MODELS, ids=MODELS)
def test_a_many_chunk_prefill_does_not_trip_the_no_token_watchdog(require_cuda, model_id: str) -> None:
    from vllm import SamplingParams, TokensPrompt

    profile = load_profile(model_id, "sm100", "parity")
    launch.apply_env(profile)
    kwargs = launch.engine_kwargs(
        profile,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=GPU_MEM_UTIL,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        enforce_eager=False,
    )
    llm = get_engine(f"{model_id}|long-prefill", **kwargs)
    prompt = _long_needle_prompt(llm.get_tokenizer())
    n_prompt = len(prompt.ids)
    n_chunks = -(-n_prompt // MAX_NUM_BATCHED_TOKENS)
    # Both halves of the premise, before the engine is asked to do anything, each naming
    # the number it is measured against.
    assert n_chunks > GRACE_CHUNKS, (
        f"{n_prompt} tokens is {n_chunks} chunks of {MAX_NUM_BATCHED_TOKENS}, which is not past the "
        f"grace of {GRACE_CHUNKS}: the old watchdog would never have fired, so a pass proves nothing"
    )
    assert n_prompt + MAX_OUTPUT_TOKENS <= MAX_MODEL_LEN, (
        f"{n_prompt} prompt tokens plus {MAX_OUTPUT_TOKENS} sampled exceeds max_model_len "
        f"{MAX_MODEL_LEN}: vLLM's renderer rejects this before a single engine step, and the "
        f"failure it reports is about output tokens, which is neither cause"
    )

    # The DA arm: mask live, no forced mask, so the FSM is built and the watchdog arms.
    payload = DattnRequest(layout=prompt.layout, enabled=True).to_extra_args()
    payload["req_id"] = "long-prefill"
    params = SamplingParams(temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS, extra_args={"dattn": payload})

    out = llm.generate([TokensPrompt(prompt_token_ids=list(prompt.ids))], [params], use_tqdm=False)

    completion = out[0].outputs[0]
    # NOT `len(token_ids) == MAX_OUTPUT_TOKENS`. That was a claim about the MODEL, not about
    # the watchdog: the prompt ends "Reply with the passphrase and nothing else", and both
    # models comply in about ten tokens and then emit EOS. Measured on this box, Gemma
    # answers in 9 and Qwen in 11, so a CORRECT answer failed this test on both, and the
    # failure it reported named the output horizon, which is neither cause. What this test
    # is entitled to assert is that the request finished for a reason that is not a crash,
    # and that the answer came out of the long context it was made to prefill.
    assert completion.finish_reason in ("stop", "length"), (
        f"finish_reason {completion.finish_reason!r}, so the request did not complete normally. "
        f"The watchdog raising DattnInvariantError and killing EngineCore part-way through "
        f"decode is exactly what this test exists to catch."
    )
    assert NEEDLE in completion.text, (
        f"the needle {NEEDLE!r} is missing from {completion.text!r}: the engine survived the "
        f"{n_chunks}-chunk prefill but did not attend the chunk that carried the passphrase"
    )

    from dattn.engine.state import DATTN_STATE

    assert DATTN_STATE.counters.steps > n_chunks, "the engine must have run past the prefill"
    shutdown_engine()
