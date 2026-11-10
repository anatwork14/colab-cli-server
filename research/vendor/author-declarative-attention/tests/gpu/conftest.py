"""GPU fixtures: one live engine at a time, plus a token-exact needle prompt builder.

Assign an idle GPU explicitly. Engines run one at a time.
"""

from __future__ import annotations

import contextlib
import gc
import os
from dataclasses import dataclass

import pytest

from dattn.contracts import INF, Mask, PromptLayout

#: Small on purpose. These are correctness tests, not capacity tests, and a short
#: max_model_len keeps the KV pool and the boot time down.
MAX_MODEL_LEN = 16384
MAX_NUM_SEQS = 8
#: Well under the 0.85 ceiling. Two of these tests boot a second engine after tearing the
#: first down, and an in-process teardown never returns every byte, so the headroom is not
#: a courtesy: it is what makes the second boot deterministic.
GPU_MEM_UTIL = 0.55

MODELS = ["google/gemma-4-31B-it", "Qwen/Qwen3.6-27B"]

_LIVE: dict[str, object] = {"key": None, "llm": None}


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    # The per-step assertions of DESIGN.md section 6 are cheap; leave them on for every
    # GPU test, which is where they can actually catch a device-side mistake.
    os.environ.setdefault("DATTN_CHECK", "1")


def shutdown_engine() -> float:
    """Tear the live engine down and report the GiB free afterwards.

    `InprocClient.shutdown` forwards to `EngineCore.shutdown`, which shuts the executor
    and the worker down, but the weights only go when the last Python reference does, so
    the caller must drop its own handle first. Every caller here does; the returned free
    figure is what the next boot has to fit in.
    """
    llm = _LIVE.get("llm")
    import torch

    from dattn.engine.state import DATTN_STATE

    if llm is not None:
        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        for target in (core, getattr(core, "engine_core", None)):
            if target is not None and hasattr(target, "shutdown"):
                # Teardown must never mask a test failure.
                with contextlib.suppress(Exception):
                    target.shutdown()
        if engine is not None:
            for name in ("engine_core", "output_processor", "processor"):
                with contextlib.suppress(Exception):
                    setattr(engine, name, None)
        with contextlib.suppress(Exception):
            llm.llm_engine = None
    _LIVE["llm"] = None
    _LIVE["key"] = None
    del llm
    DATTN_STATE.__init__()  # type: ignore[misc]
    # Dropping the last Python reference is necessary but NOT sufficient: torch.compile
    # keeps the compiled module (and therefore every parameter) alive in the Dynamo code
    # cache, so a plain gc + empty_cache leaves the weights resident -- about 50 GiB on
    # either model. The first teardown in a session still leaves room for the next boot,
    # the second does not, and the failure surfaces as vLLM's "Free memory on device ...
    # is less than desired GPU memory utilization" three tests later. `_dynamo.reset()`
    # is what actually releases them; `cleanup_dist_env_and_memory` tears down the process
    # group the next `LLM(...)` would otherwise re-enter.
    with contextlib.suppress(Exception):
        torch._dynamo.reset()
    with contextlib.suppress(Exception):
        from vllm.distributed import cleanup_dist_env_and_memory

        cleanup_dist_env_and_memory()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    return free / (1 << 30)


def get_engine(key: str, **kwargs):
    """Boot (or reuse) one engine. Any other live engine is shut down first.

    Skips rather than crashes when the device cannot hold another engine. vLLM 0.29 does
    not give the model weights back on an in-process teardown: `LLM.shutdown` releases the
    KV pool, but about 48 GiB of parameters stay resident behind the compiled artefacts,
    and neither `gc.collect`, `empty_cache`, `torch._dynamo.reset` nor
    `cleanup_dist_env_and_memory` recovers them (measured on both models). Two boots fit,
    a third does not, so a whole-file run of this module reaches the fourth boot with
    81 GiB free and vLLM raises `Free memory on device ... is less than desired GPU memory
    utilization`. The procedure is one test per process (`tests/gpu/run_gpu_suite.sh`);
    this makes the alternative an honest skip instead of a confusing failure.
    """
    import torch
    from vllm import LLM

    if _LIVE["key"] == key:
        return _LIVE["llm"]
    shutdown_engine()
    free, total = torch.cuda.mem_get_info()
    want = float(kwargs.get("gpu_memory_utilization", GPU_MEM_UTIL)) * total
    if free < want:
        pytest.skip(
            f"only {free / 2**30:.1f} GiB free, this boot wants {want / 2**30:.1f}: an earlier engine in "
            "this process did not give its weights back (vLLM 0.29 in-process teardown). Run this test in "
            "its own process, e.g. tests/gpu/run_gpu_suite.sh"
        )
    llm = LLM(**kwargs)
    _LIVE["key"] = key
    _LIVE["llm"] = llm
    return llm


@pytest.fixture(scope="session", autouse=True)
def _teardown_engines():
    yield
    shutdown_engine()


@pytest.fixture(scope="session")
def require_cuda():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible; assign an idle GPU with CUDA_VISIBLE_DEVICES")
    free, total = torch.cuda.mem_get_info()
    if free < 90 * (1 << 30):
        pytest.skip(f"only {free / 2**30:.0f} GiB free on this device; a 31B engine needs about 90")
    return torch


# ---------------------------------------------------------------------------------------
# the needle prompt
# ---------------------------------------------------------------------------------------

NEEDLE = "quartz-lantern-8813"
DECOY = "copper-bramble-2247"

_FILLER = (
    "The municipal archive records that the harbour survey of that season was carried out "
    "on foot, that the tide tables were copied by hand into the ledger, and that the "
    "assistant surveyor noted the weather each morning before setting out. "
)


@dataclass(frozen=True)
class NeedlePrompt:
    """A token-exact prompt whose chunk spans are known by construction.

    Built by tokenizing the pieces separately and concatenating the ids, rather than by
    tokenizing the whole string and hunting for offsets. That makes the spans exact,
    which is what the mask needs, and it is the same reason the design's own span
    recovery renders prefixes instead of matching regexes (DESIGN.md section 1.2).
    """

    ids: list[int]
    layout: PromptLayout
    spans: list[tuple[int, int]]
    needle_chunk: int
    decoy_chunk: int

    def focused(self) -> Mask:
        s, e = self.spans[self.needle_chunk]
        return Mask.of((0, self.layout.sink), (s, e), (self.layout.question_start, INF))

    def excluded(self) -> Mask:
        s, e = self.spans[self.decoy_chunk]
        return Mask.of((0, self.layout.sink), (s, e), (self.layout.question_start, INF))

    def dense(self) -> Mask:
        return Mask.of((0, INF))

    def nearly_dense(self) -> Mask:
        """Keeps everything but a 32-token hole, so the row is ACTIVE but not dense.

        A dense row is a fixed point of the compaction and is gated out, so it cannot be
        the f=1.0 arm of a timing comparison: it would measure the gate, not the kernel.
        """
        q = self.layout.question_start
        return Mask.of((0, max(q - 64, 1)), (max(q - 32, 1), INF))

    def fraction(self, keep: float) -> Mask:
        """Keep the sink plus the last `keep` of the context, plus the open tail."""
        q = self.layout.question_start
        start = max(self.layout.sink, int(q * (1.0 - keep)))
        return Mask.of((0, self.layout.sink), (start, INF))


def build_needle_prompt(
    tokenizer,
    *,
    n_chunks: int = 24,
    chunk_repeats: int = 3,
    needle_chunk: int = 12,
    decoy_chunk: int = 5,
    question: str = "What is the secret passphrase? Reply with the passphrase and nothing else.",
) -> NeedlePrompt:
    placeholder = "@@DATTN_CONTEXT@@"
    messages = [{"role": "user", "content": f"Here are some archive notes.\n\n{placeholder}\n\n{question}"}]
    # Qwen3.6 is a thinking model: with the reasoning block on, 24 tokens of output are a
    # generic preamble that is identical whether the needle was kept or excluded, so the
    # test would measure nothing. The toggle is a template variable, so it is only passed
    # to a template that declares it; Gemma's has no such switch and needs none.
    extra = {"enable_thinking": False} if "enable_thinking" in (tokenizer.chat_template or "") else {}
    rendered = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **extra)
    head_text, _, tail_text = rendered.partition(placeholder)

    def enc(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    head = enc(head_text)
    ids = list(head)
    spans: list[tuple[int, int]] = []
    for i in range(n_chunks):
        body = _FILLER * chunk_repeats
        if i == needle_chunk:
            body += f"The secret passphrase is {NEEDLE}. Remember it. "
        elif i == decoy_chunk:
            body += f"The old mooring code was {DECOY}. It is obsolete. "
        chunk = enc(f"\n\nNote {i + 1}.\n{body}")
        spans.append((len(ids), len(ids) + len(chunk)))
        ids.extend(chunk)
    question_start = len(ids)
    ids.extend(enc(tail_text))

    layout = PromptLayout(
        n_prompt_tokens=len(ids),
        question_start=question_start,
        segments=tuple(spans),
        sink=16,
    )
    return NeedlePrompt(
        ids=ids,
        layout=layout,
        spans=spans,
        needle_chunk=needle_chunk,
        decoy_chunk=decoy_chunk,
    )
