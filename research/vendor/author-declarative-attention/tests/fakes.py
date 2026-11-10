"""Shared CPU doubles for engine, recorder and HTTP evaluation tests."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import torch

from dattn.contracts import INF, Mask, Mode, PromptLayout, StepRecord, StepRows
from dattn.eval.client import Completion


@dataclass
class FakeCM:
    """The fields of `CommonAttentionMetadata` that `DattnState.compact` reads.

    `tests/contract/test_common_attention_metadata.py` pins that the real dataclass has
    exactly these names and that `replace()` exists, so this stand-in cannot drift
    silently.
    """

    seq_lens: torch.Tensor
    block_table_tensor: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    num_reqs: int
    max_seq_len: int
    slot_mapping: torch.Tensor | None = None
    is_prefilling: torch.Tensor | None = None
    _seq_lens_cpu: torch.Tensor | None = None
    _num_computed_tokens_cpu: torch.Tensor | None = None
    _num_computed_tokens_cache: torch.Tensor | None = None
    _token_to_req_indices_cache: torch.Tensor | None = None
    seq_lens_cpu_upper_bound: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None

    def replace(self, **kw: Any) -> FakeCM:
        return replace(self, **kw)


def make_cm(seq_lens: list[int], n_cols: int, *, query_lens: list[int] | None = None, prefilling=None) -> FakeCM:
    rows = len(seq_lens)
    qlens = query_lens if query_lens is not None else [1] * rows
    qsl = torch.zeros((rows + 1,), dtype=torch.int32)
    for i, q in enumerate(qlens):
        qsl[i + 1] = qsl[i] + q
    sl = torch.tensor(seq_lens, dtype=torch.int32)
    bt = torch.arange(1, rows * n_cols + 1, dtype=torch.int32).reshape(rows, n_cols)
    pf = None if prefilling is None else torch.tensor(prefilling, dtype=torch.bool)
    return FakeCM(
        seq_lens=sl,
        block_table_tensor=bt,
        query_start_loc_cpu=qsl,
        query_start_loc=qsl,
        num_reqs=rows,
        max_seq_len=max(seq_lens) if seq_lens else 0,
        slot_mapping=torch.zeros((sum(qlens),), dtype=torch.int64),
        is_prefilling=pf,
        _seq_lens_cpu=sl.clone(),
    )


@dataclass
class FakeBuilder:
    """Only what `DattnState` touches: a block size and a scratch slot."""

    block_size: int = 16
    scratch: Any = None
    dattn_group_id: int = 0
    dattn_needs_seq_lens_cpu: bool = False


class StubFSM:
    """Lane A's `DeclarativeFSM` reduced to what the logits processor calls.

    Emits a new mask on the token id `PIVOT`, and raises on a negative id exactly as
    the real one must (DESIGN.md section 2.7).
    """

    PIVOT = 99

    def __init__(self, layout: PromptLayout) -> None:
        self.layout = layout
        self.mode = Mode.GLOBAL
        self.mask = Mask.of((0, INF))
        self.seen: list[int] = []

    def step(self, token_id: int, token_text: str) -> Mask | None:
        from dattn.contracts import DattnInvariantError

        if token_id < 0:
            raise DattnInvariantError("negative token id")
        self.seen.append(token_id)
        if token_id == self.PIVOT:
            self.mode = Mode.FOCUS
            self.mask = Mask.of((0, self.layout.sink), (self.layout.question_start, INF))
            return self.mask
        return None


@dataclass
class FakeSamplingParams:
    extra_args: dict[str, Any] | None = field(default=None)


class FakeEvent:
    """`torch.cuda.Event(enable_timing=True)`. Counts records, reports a scripted time."""

    created = 0

    def __init__(self, enable_timing: bool = False) -> None:
        FakeEvent.created += 1
        self.enable_timing = enable_timing
        self.records = 0
        self.stamp = 0.0

    def record(self) -> None:
        self.records += 1
        self.stamp = float(FakeTorch.CLOCK)
        FakeTorch.CLOCK += 1.0

    def elapsed_time(self, other: FakeEvent) -> float:
        return other.stamp - self.stamp


class FakeTensor:
    """The slice of an int32 ring that `copy_` writes into.

    The ring is `(ring, groups, max_num_seqs)` since the recorder started mirroring EVERY
    compacted group rather than only group 0. It is stored flat, `ring * groups` rows, and
    `[slot, gidx]` resolves to `slot * groups + gidx`, so the 2D indexing the older tests
    use still works when `groups == 1`.
    """

    def __init__(
        self,
        data: list[list[int]],
        groups: int = 1,
        row: int | None = None,
        lo: int = 0,
        hi: int | None = None,
    ) -> None:
        self._data = data
        self._groups = groups
        self._row = row
        self._lo = lo
        self._hi = hi

    def _sub(self, row: int | None = None, lo: int = 0, hi: int | None = None) -> FakeTensor:
        return FakeTensor(self._data, self._groups, row, lo, hi)

    def __getitem__(self, key: Any) -> FakeTensor:
        if isinstance(key, tuple) and len(key) == 3:
            slot, gidx, sl = key
            return self._sub(int(slot) * self._groups + int(gidx), sl.start or 0, sl.stop)
        if isinstance(key, tuple):
            first, second = key
            if isinstance(second, slice):
                return self._sub(int(first), second.start or 0, second.stop)
            return self._sub(int(first) * self._groups + int(second))
        if isinstance(key, slice):
            return FakeTensor(self._data, self._groups, self._row, key.start or 0, key.stop)
        return self._sub(int(key))

    def copy_(self, src: Any, non_blocking: bool = False) -> None:
        assert non_blocking, "the per-row ring copy must be non-blocking: it is a D2D copy"
        vals = list(src)
        row = self._data[self._row] if self._row is not None else self._data[0]
        for i, v in enumerate(vals):
            row[self._lo + i] = int(v)

    def to(self, device: str) -> FakeTensor:
        assert device == "cpu"
        FakeTorch.D2H += 1
        return FakeTensor([list(r) for r in self._data], self._groups)

    def __iter__(self) -> Any:
        row = self._data[self._row] if self._row is not None else self._data[0]
        return iter(row[self._lo : self._hi])

    def __len__(self) -> int:
        row = self._data[self._row] if self._row is not None else self._data[0]
        return len(row[self._lo : self._hi])


class _Cuda:
    syncs = 0

    @staticmethod
    def Event(enable_timing: bool = False) -> FakeEvent:  # noqa: N802 -- mirrors torch
        return FakeEvent(enable_timing)

    @staticmethod
    def synchronize() -> None:
        _Cuda.syncs += 1


class FakeTorch:
    """The three torch entry points the recorder uses, and nothing else."""

    CLOCK = 0.0
    D2H = 0
    int32 = "int32"
    cuda = _Cuda

    @staticmethod
    def zeros(shape: tuple[int, ...], dtype: str = "int32", device: Any = None) -> FakeTensor:
        if len(shape) == 3:
            rows, groups, cols = shape
        else:
            (rows, cols), groups = shape, 1
        return FakeTensor([[0] * cols for _ in range(rows * groups)], groups)

    @classmethod
    def reset(cls) -> None:
        cls.CLOCK = 0.0
        cls.D2H = 0
        _Cuda.syncs = 0
        FakeEvent.created = 0


@dataclass
class FakeCached:
    req_ids: tuple[str, ...] = ()
    context_phase: frozenset[str] = frozenset()

    def is_context_phase(self, req_id: str) -> bool:
        return req_id in self.context_phase


@dataclass
class FakeNewReq:
    req_id: str


@dataclass
class FakeSchedulerOutput:
    scheduled_new_reqs: tuple[FakeNewReq, ...] = ()
    scheduled_cached_reqs: FakeCached = field(default_factory=FakeCached)
    total_num_scheduled_tokens: int = 0
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    #: 0.29.0 carries the ids preempted in THIS step; the worker is the only place they
    #: are visible without a cross-process hop (`N:v1/core/sched/output.py:257-259`).
    preempted_req_ids: set[str] | None = None


@dataclass
class FakeExecuteModelState:
    """`GPUModelRunner.execute_model_state`, the cache `execute_model` leaves for `_sample`."""

    scheduler_output: FakeSchedulerOutput


@dataclass
class FakeInputBatch:
    num_reqs: int = 0


def decode_step(n: int = 4) -> FakeSchedulerOutput:
    ids = tuple(f"r{i}" for i in range(n))
    return FakeSchedulerOutput(
        scheduled_cached_reqs=FakeCached(req_ids=ids),
        total_num_scheduled_tokens=n,
        num_scheduled_tokens=dict.fromkeys(ids, 1),
    )


class FakeRunner:
    """Exposes exactly the surface `hooks.install` reads: the 0.29.0 override point."""

    max_num_seqs = 8
    async_scheduling = False
    use_v2_model_runner = False
    device = "cuda:0"

    def __init__(self, scheduler_output: FakeSchedulerOutput | None = None) -> None:
        self.forwards = 0
        self.samples = 0
        self.executes = 0
        #: `None` until `execute_model` has run, exactly as on the real runner. With one
        #: set, the `_sample` wrapper can build the `StepFacts` of interface request C4.
        self.execute_model_state = FakeExecuteModelState(scheduler_output) if scheduler_output else None
        self.input_batch = FakeInputBatch(
            num_reqs=len(scheduler_output.scheduled_cached_reqs.req_ids) if scheduler_output else 0
        )

    def execute_model(self, scheduler_output: Any = None, *args: Any, **kwargs: Any) -> None:
        """0.29.0 caches the SchedulerOutput for `sample_tokens` and `sample_tokens` clears
        it BEFORE calling `_sample`, so the fake clears it here to reproduce exactly that:
        the object is gone by the time the sampler runs, and only the argument remains."""
        self.executes += 1
        self.execute_model_state = None
        return None

    def _model_forward(self, *args: Any, **kwargs: Any) -> str:
        self.forwards += 1
        return "logits"

    def _sample(self, *args: Any, **kwargs: Any) -> str:
        self.samples += 1
        return "tokens"


class FakeSpec:
    def __init__(self, block_size: int = 16) -> None:
        self.block_size = block_size

    def __hash__(self) -> int:
        return hash(self.block_size)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FakeSpec) and other.block_size == self.block_size


class FakeSchedulerConfig:
    max_num_seqs = 8


class FakeVllmConfig:
    scheduler_config = FakeSchedulerConfig()


class BaseBuilder:
    supports_update_block_table = True  # deliberately wrong, so the override is visible

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device, **kw):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device
        self.seen: list = []

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False, **kw):
        self.seen.append(common_attn_metadata)
        return ("built", common_prefix_len, common_attn_metadata)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(0, common_attn_metadata)

    def update_block_table(self, metadata, blk_table, slot_mapping):
        return "base-update"

    def use_cascade_attention(self, *a, **k):
        return True

    def something_inherited(self) -> str:
        return "inherited"


class BaseBackend:
    supported_dtypes = [torch.bfloat16]

    @staticmethod
    def get_name() -> str:
        return "TRITON_ATTN"

    @staticmethod
    def get_builder_cls() -> type:
        return BaseBuilder

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        return True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return [16]


class FakeBatchUpdate:
    def __init__(self, batch_size=0, removed=(), added=(), moved=()):
        self.batch_size = batch_size
        self.removed = list(removed)
        self.added = list(added)
        self.moved = list(moved)


class _FakeProfile:
    def __init__(self, engine_args: dict[str, object]) -> None:
        self.engine_args = engine_args
        self.model_id = "m"


class FakeScheduler:
    preempt_at = None

    def __init__(self):
        self.running, self.waiting = [], []
        self.kv_cache_manager = NS(
            usage=0.0,
            coordinator=NS(scheduler_block_size=16),
            block_pool=NS(
                blocks=[NS(block_hash="cached", is_null=False) for _ in range(100)],
                cached_block_hash_to_block={i: i for i in range(100)},
            ),
        )
        self.block_size = 16
        self.c = 0
        self.received: list[bool] = []
        self.step = -1

    def schedule(self, throttle_prefills: bool = False):
        # Same signature as vllm.v1.core.sched.scheduler.Scheduler.schedule in 0.29.0.
        self.received.append(throttle_prefills)
        return NS(
            total_num_scheduled_tokens=self.c,
            num_scheduled_tokens=dict.fromkeys((r.request_id for r in self.running), 1),
            preempted_req_ids={"r"} if self.step == self.preempt_at else set(),
        )


class FakeEngine:
    calls = []
    usage_by_step = {}
    close_window_at = None
    warm_leftover = 0.0
    cached_tokens = None
    instances = 0

    def __init__(self, **kwargs):
        type(self).instances += 1
        self.kwargs = kwargs
        self.da = bool(kwargs.get("logits_processors"))
        self.sched = FakeScheduler()
        self.llm_engine = NS(engine_core=NS(engine_core=NS(scheduler=self.sched)))
        self.recording = False
        self.records = []

    def get_tokenizer(self):
        return NS(chat_template="test")

    def collective_rpc(self, name, args=(), kwargs=None):
        if name == "dattn_probe":
            return [
                {
                    "dattn_builder_installed": self.da,
                    "dattn_lp_installed": self.da,
                    "enable_prefix_caching": self.kwargs.get("enable_prefix_caching", False),
                    "kv_pool": {
                        "pool_bytes": 9250,
                        "physical_bytes_per_block": 100,
                        "groups": [{"kind": "full_attention", "block_size": 16, "page_bytes": 100}],
                    },
                    "resolved_config": {"capture_sizes": self.kwargs["compilation_config"]["cudagraph_capture_sizes"]},
                }
            ]
        if name == "dattn_record":
            self.recording = args[0]
            if self.recording:
                self.records = []
            return [{"armed": self.recording}]
        if name == "dattn_drain":
            return [self.records]
        if name == "dattn_audit":
            return [{"checked": 10, "mismatches": 0, "groups_dropped": 0}]
        raise AssertionError(name)

    def generate(self, prompts, params):
        self.calls.append((self.da, len(prompts), params[0]))
        c = len(prompts)
        self.sched.c = c
        n = params[0].max_tokens if params[0].min_tokens == params[0].max_tokens else 20
        lens = [len(p["prompt_token_ids"]) if isinstance(p, dict) else len(p) for p in prompts]
        for i in range(n):
            self.sched.running = [
                NS(
                    request_id=str(j),
                    num_computed_tokens=lens[j] + i,
                    num_prompt_tokens=lens[j],
                    num_in_flight_tokens=0,
                )
                for j in range(c)
            ]
            self.sched.step = i
            self.sched.kv_cache_manager.usage = self.usage_by_step.get(
                i, sum((n + i + 15) // 16 for n in lens) * 100 / 9250
            )
            out = self.sched.schedule(False)  # one positional argument, as EngineCore.step does
            if self.recording:
                self.records.append(
                    StepRecord(
                        step_id=i,
                        engine_step=i,
                        t_sched_ns=1_000_000 + i * 10_000_000,
                        num_reqs=c - int(self.close_window_at is not None and i >= self.close_window_at),
                        num_reqs_padded=c,
                        dattn_block_size=32,
                        cudagraph_mode="FULL",
                        num_gen_tokens=out.total_num_scheduled_tokens,
                        rows=StepRows(seq_len=[128] * c, att_tok=[128] * c),
                    )
                )
        self.sched.running = []
        self.sched.kv_cache_manager.usage = self.warm_leftover
        return [
            NS(
                request_id=str(j),
                num_cached_tokens=max(0, lens[j] - 16) if self.cached_tokens is None else self.cached_tokens,
                outputs=[NS(token_ids=[1] * n, finish_reason="length" if n == params[j].max_tokens else "stop")],
            )
            for j in range(c)
        ]


class FakeTokenizer:
    def __init__(self, response=None, *, chat_template="test"):
        self.response = response
        self.calls = []
        self.chat_template = chat_template

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(1, len(text) + 1))}

    def decode(self, ids, skip_special_tokens=False):
        self.calls.append((tuple(ids), skip_special_tokens))
        if self.response is None:
            return "" if skip_special_tokens else "<SPECIAL>"
        return "".join(self.response[i - 1] for i in ids)


class FakeClient:
    def __init__(self, server, *, response="", seen=None, errors=frozenset(), empty=frozenset()):
        self.response, self.seen, self.errors, self.empty = response, seen, errors, empty
        self.n_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def complete_many(self, prompts, **kw):
        self.n_calls += 1
        keys = [int(p[0]) for p in prompts] if self.seen is not None else list(range(len(prompts)))
        if self.seen is not None:
            self.seen.append(keys)
        return [
            [
                Completion(error="connection reset", transport_error=True)
                if i in self.errors
                else Completion(
                    text="" if i in self.empty else self.response,
                    finish_reason="stop",
                    n_generated_tokens=0 if i in self.empty else len(self.response),
                )
            ]
            for i in keys
        ]


class FakeHTTP:
    def __init__(self, response=None):
        self.calls = []
        self.response = response or {
            "choices": [{"message": {"content": '{"correct": true}'}, "finish_reason": "stop"}]
        }

    def __call__(self, url, payload, **kwargs):
        self.calls.append((url, dict(payload)))
        return self.response


def write_cell(
    root: Path,
    rows: Sequence[dict],
    judgments: Sequence[dict] | None = None,
) -> Path:
    """Write a minimal `responses.jsonl` (+ `judgments.jsonl`) cell directory."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if judgments is not None:
        (root / "judgments.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judgments))
    return root


def response_row(
    subset: str,
    row_index: int,
    *,
    question: str = "q",
    rubric: str = "r",
    response: str = "<answer>x</answer>",
    prompt_tokens: int = 100,
    n_segments: int = 3,
    format_correct: bool = True,
    parsed_answer: str = "x",
    metrics: dict | None = None,
    **extra: object,
) -> dict:
    row = {
        "subset": subset,
        "row_index": row_index,
        "response_index": 0,
        "question": question,
        "rubric": rubric,
        "answer": "gold",
        "response": response,
        "parsed_answer": parsed_answer,
        "format_correct": format_correct,
        "rendered_prompt_tokens": prompt_tokens,
        "n_bookmarks": n_segments,
        "dac_metrics": metrics if metrics is not None else {"total_steps": 10, "attended_tokens": 5},
    }
    row.update(extra)
    return row


def judgment_row(subset: str, row_index: int, correct: bool | None, **extra: object) -> dict:
    row = {"subset": subset, "row_index": row_index, "response_index": 0, "correct": correct}
    row.update(extra)
    return row
