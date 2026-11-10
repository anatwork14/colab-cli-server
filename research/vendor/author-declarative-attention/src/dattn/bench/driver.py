"""One arm and one batch per boot, with a warm pass and sampled repetitions."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dattn.bench.protocol import Protocol
from dattn.bench.sizing import CAP, pin_batches, sizing_validation
from dattn.bench.tasks import TaskSample, load_tasks
from dattn.contracts import StepRecord
from dattn.engine.launch import builder_free_kwargs
from dattn.metrics import env
from dattn.metrics.record import write_table

ARMS = ("da", "vanilla")
DA_FORMAT = "chunk_multi_turn_tool"
VANILLA_FORMAT = "vanilla"
ARM_FORMAT = {"da": DA_FORMAT, "vanilla": VANILLA_FORMAT}
BUILDER_BINDINGS = {"CUSTOM", "DATTN"}
PLAN_MISMATCH = "batch identity mismatch"
WARMUP_STEPS = 64


def is_plain_arm(arm):
    return arm == "vanilla"


def _attention_config_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        import json as json_lib

        try:
            loaded = json_lib.loads(value)
        except ValueError:
            return {"_unparsed": value}
        return dict(loaded) if isinstance(loaded, Mapping) else {"_unparsed": value}
    return {}


def builder_free_violations(engine_kwargs: Mapping[str, Any]) -> list[str]:
    bad: list[str] = []
    if engine_kwargs.get("logits_processors"):
        bad.append("logits_processors")
    if _attention_config_dict(engine_kwargs.get("attention_config")).get("backend_per_kind"):
        bad.append("attention_config.backend_per_kind")
    bad += [k for k in engine_kwargs if str(k).startswith("dattn")]
    return sorted(set(bad))


def sampling_recipe(model_id: str, *, thinking: bool = False) -> dict[str, float | int]:
    from dattn.eval.sampling import lookup_sampling

    return lookup_sampling(model_id, thinking=thinking)


def request_seed(base_seed: int, occurrence: int, rep_index: int) -> int:
    return (int(base_seed) * 1_000_003 + int(occurrence) * 9_176 + int(rep_index) * 104_729 + 1) % (2**31 - 1)


@dataclass(frozen=True, slots=True)
class Batch:
    task: str
    set_index: int
    indices: tuple[int, ...]
    prompt_lens: tuple[int, ...]
    occupancy: float
    seed: int

    @property
    def size(self) -> int:
        return len(self.indices)


def prompt_digest(prompt: Any) -> str:
    import hashlib

    ids = None
    if isinstance(prompt, Mapping):
        ids = prompt.get("prompt_token_ids")
    if ids is None:
        ids = getattr(prompt, "prompt_token_ids", None)
    payload = repr(list(ids)) if ids is not None else repr(prompt)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def batch_identity(
    batch: Batch,
    *,
    protocol: Protocol,
    sample_ids: Sequence[str] = (),
    prompts_by_fmt: Mapping[str, Sequence[Any]] | None = None,
) -> dict[str, Any]:
    fmts = dict(prompts_by_fmt or {})
    return {
        "task": batch.task,
        "set_index": int(batch.set_index),
        "indices": [int(i) for i in batch.indices],
        "prompt_lens": [int(n) for n in batch.prompt_lens],
        "occupancy": float(batch.occupancy),
        "seed": int(batch.seed),
        "sample_ids": [str(x) for x in sample_ids],
        "max_tokens": int(protocol.max_tokens),
        "concurrency": int(protocol.concurrency),
        "prompt_digests": {
            fmt: [prompt_digest(prompts[i]) for i in batch.indices] for fmt, prompts in sorted(fmts.items())
        },
    }


def identity_hash(identity: Mapping[str, Any]) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def identity_faults(recorded: Mapping[str, Any], observed: Mapping[str, Any], *, cell_id: str) -> list[str]:
    h1, h2 = identity_hash(recorded), identity_hash(observed)
    if h1 == h2:
        return []
    out = [
        f"{cell_id}: {PLAN_MISMATCH}, pass 1 hashed {h1[:16]} and pass 2 hashed {h2[:16]}. "
        "The two passes did NOT run the same batch, so nothing in this cell is paired"
    ]
    for key in sorted(set(recorded) | set(observed)):
        a, b = recorded.get(key), observed.get(key)
        if a != b:
            out.append(f"{cell_id}: {PLAN_MISMATCH} in {key!r}: pass 1 {a!r} against pass 2 {b!r}")
    return out


@dataclass(frozen=True, slots=True)
class Window:
    start: int
    stop: int
    reason: str

    @property
    def n_steps(self) -> int:
        return max(0, self.stop - self.start)


class VllmEngineAdapter:
    def __init__(
        self,
        llm: Any,
        *,
        prompts: Mapping[str, Sequence[Any]] | Sequence[Any],
        layouts: Sequence[Any] | None = None,
        engine_kwargs: Mapping[str, Any] | None = None,
        model_id: str = "",
        thinking: bool = False,
    ) -> None:
        self.model_id = model_id
        self.thinking = bool(thinking)
        self.llm = llm
        if isinstance(prompts, Mapping):
            self.prompts: dict[str, list[Any]] = {k: list(v) for k, v in prompts.items()}
        else:  # one render for everything, which is a smoke path and not the protocol
            self.prompts = {fmt: list(prompts) for fmt in set(ARM_FORMAT.values())}
        self.layouts = list(layouts) if layouts is not None else None
        self.engine_kwargs: dict[str, Any] = dict(engine_kwargs or {})
        self.is_builder_free: bool | None = (
            None if engine_kwargs is None else not builder_free_violations(self.engine_kwargs)
        )

    def _prompts_for(self, arm: str) -> list[Any]:
        fmt = ARM_FORMAT.get(arm, DA_FORMAT)
        try:
            return self.prompts[fmt]
        except KeyError as exc:
            raise KeyError(
                f"arm {arm!r} needs the {fmt!r} render and only {sorted(self.prompts)} were built. "
                "build_prompts must be called with every format the arm set asks for"
            ) from exc

    @property
    def scheduler(self):
        return self.llm.llm_engine.engine_core.engine_core.scheduler

    def hygiene(self):
        scheduler = self.scheduler
        pool = scheduler.kv_cache_manager.block_pool
        coordinator = scheduler.kv_cache_manager.coordinator
        hybrid_alignment = getattr(coordinator, "_cache_hit_alignment_tokens", None)
        return {
            "kv_usage": float(scheduler.kv_cache_manager.usage),
            "running": len(scheduler.running),
            "waiting": len(scheduler.waiting),
            "cached_blocks": sum(b.block_hash is not None for b in pool.blocks if not b.is_null),
            "cached_hashes": len(pool.cached_block_hash_to_block),
            "manager_block_size": scheduler.block_size,
            "cache_alignment": hybrid_alignment or coordinator.scheduler_block_size,
            "cache_alignment_basis": "resolved_hybrid_lookup" if hybrid_alignment else "resolved_scheduler",
        }

    def _rpc(self, name: str, *args: Any, **kwargs: Any) -> Any:
        out = self.llm.collective_rpc(name, args=args, kwargs=kwargs)
        return out[0] if isinstance(out, list) and out else out

    def probe(self) -> dict[str, Any]:
        return dict(self._rpc("dattn_probe"))

    def arm(self, window_id: str) -> dict[str, Any]:
        return dict(self._rpc("dattn_record", True, window_id=window_id) or {})

    def disarm(self) -> dict[str, Any]:
        return dict(self._rpc("dattn_record", False, window_id="") or {})

    def drain(self) -> list[StepRecord]:
        return list(self._rpc("dattn_drain"))

    def audit(self) -> dict[str, Any]:
        try:
            return dict(self._rpc("dattn_audit") or {})
        except Exception as exc:  # noqa: BLE001 - never lose a measured cell to a probe
            return {"error": f"{type(exc).__name__}: {exc}"}

    def run_arm(
        self,
        arm: str,
        batch: Batch,
        *,
        max_tokens: int,
        ignore_eos: bool,
        min_tokens: int,
        seed: int,
        rep_index: int = 0,
        observe_resident: bool = False,
    ) -> Any:
        from vllm import SamplingParams

        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")
        plain = is_plain_arm(arm)
        if plain and self.is_builder_free is not True:
            raise ValueError("vanilla requires a builder-free engine with recorded engine kwargs")
        if not plain and self.layouts is None:
            raise ValueError(
                f"arm {arm!r} needs per-request layouts and this adapter has none, so every request would "
                "go out with no `dattn` payload: no FSM, no mask, no compaction. Build the adapter with "
                "layouts=... (run_protocol does) or run only plain arms on it"
            )
        recipe = sampling_recipe(self.model_id, thinking=self.thinking) if self.model_id else {}
        params = []
        for occurrence, i in enumerate(batch.indices):
            extra: dict[str, Any] = {}
            if not plain and self.layouts is not None:
                from dattn.contracts import DattnRequest

                layout = self.layouts[i]
                req = DattnRequest(layout=layout)
                extra = {"dattn": req.to_extra_args()}
            params.append(
                SamplingParams(
                    **recipe,
                    max_tokens=max_tokens,
                    min_tokens=min_tokens,
                    ignore_eos=ignore_eos,
                    seed=request_seed(seed, occurrence, rep_index),
                    extra_args=extra or None,
                )
            )
        self.request_params = params
        prompts = self._prompts_for(arm)
        scheduler = self.scheduler
        original = scheduler.schedule
        self.stats = []
        self.preempted = 0
        self.resident_snapshot = None
        step_index = -1

        def schedule(*args: Any, **kwargs: Any):
            nonlocal step_index
            step_index += 1
            if observe_resident:
                # schedule() advances counters before returning. Do not mistake
                # the final prefill token for a decode step.
                decode_ready = {
                    r.request_id
                    for r in scheduler.running
                    if r.num_computed_tokens >= r.num_prompt_tokens and not r.num_in_flight_tokens
                }
            output = original(*args, **kwargs)
            self.preempted += len(output.preempted_req_ids or ())
            all_decode = (
                observe_resident
                and bool(scheduler.running)
                and (
                    all(r.request_id in decode_ready for r in scheduler.running)
                    and output.num_scheduled_tokens == dict.fromkeys((r.request_id for r in scheduler.running), 1)
                )
            )
            if observe_resident and (
                self.resident_snapshot is None
                or len(scheduler.running) > self.resident_snapshot["n_running"]
                or (
                    len(scheduler.running) == self.resident_snapshot["n_running"]
                    and all_decode
                    and self.resident_snapshot["snapshot_kind"] != "all_decode"
                )
            ):
                self.resident_snapshot = {
                    "snapshot_kind": "all_decode" if all_decode else "max_running",
                    "kv_usage": float(scheduler.kv_cache_manager.usage),
                    "step_index": step_index,
                    "n_running": len(scheduler.running),
                    "C": batch.size,
                    "running": len(scheduler.running),
                    "waiting": len(scheduler.waiting),
                    "requests": [
                        {
                            "request_id": r.request_id,
                            "resident_tokens": r.num_computed_tokens,
                            "prompt_tokens": r.num_prompt_tokens,
                        }
                        for r in scheduler.running
                    ],
                }
            if output.total_num_scheduled_tokens:
                usage = float(scheduler.kv_cache_manager.usage)
                self.stats.append((usage, self.preempted))
            return output

        scheduler.schedule = schedule
        try:
            return self.llm.generate([prompts[i] for i in batch.indices], params)
        finally:
            del scheduler.schedule


def build_prompts(
    tokenizer: Any,
    samples: Sequence[TaskSample],
    *,
    formats: Sequence[str] = (DA_FORMAT, VANILLA_FORMAT),
) -> tuple[dict[str, list[Any]], list[Any], dict[str, list[int]]]:
    from vllm import TokensPrompt

    from dattn.prompt.render import render

    wanted = list(dict.fromkeys(formats))
    if DA_FORMAT not in wanted:
        raise ValueError(f"the DA render {DA_FORMAT!r} is always needed, it carries the layouts, got {wanted}")
    prompts: dict[str, list[Any]] = {f: [] for f in wanted}
    lens: dict[str, list[int]] = {f: [] for f in wanted}
    layouts: list[Any] = []
    for sample in samples:
        for fmt in wanted:
            r = render(tokenizer, context=sample.context, question=sample.question, fmt=fmt)
            prompts[fmt].append(TokensPrompt(prompt_token_ids=list(r.ids)))
            lens[fmt].append(len(r.ids))
            if fmt == DA_FORMAT:
                layouts.append(r.layout)
    return prompts, layouts, lens


def detect_window(records, *, concurrency, protocol):
    start = next(
        (i for i, r in enumerate(records) if r.kind == "decode" and r.num_reqs == concurrency and not r.num_ctx_tokens),
        len(records),
    )
    # The first decode JIT step is outside the window, as in phase 2.
    start = min(start + 1, len(records))
    stop = next(
        (
            i
            for i in range(start, len(records))
            if records[i].num_reqs != concurrency or records[i].kind != "decode" or records[i].num_ctx_tokens
        ),
        len(records),
    )
    return Window(start, stop, "first EOS or end of records")


def hygiene_witness(adapter, batch, arm):
    witness = adapter.hygiene()
    lens = [len(adapter._prompts_for(arm)[i]["prompt_token_ids"]) for i in batch.indices]
    witness["full_prompt_blocks"] = sum(n // witness["manager_block_size"] for n in lens)
    witness["cache_coverage_witness"] = witness["cached_blocks"] >= witness["full_prompt_blocks"]
    witness["ok"] = all(witness[k] == 0 for k in ("kv_usage", "running", "waiting"))
    return witness


def prime_aligned_prompts(adapter, batch, arm, meta):
    """Materialize hybrid replay checkpoints before the ordinary warm pass."""
    if not meta["enable_prefix_caching"] or not any(g["kind"] == "mamba" for g in meta["kv_pool"]["groups"]):
        return []
    witness = adapter.hygiene()
    alignment = witness.get("cache_alignment")
    if not alignment or alignment <= 0:
        return []
    prompts = adapter._prompts_for(arm)
    primed = []
    for occurrence, i in enumerate(batch.indices):
        tokens = prompts[i]["prompt_token_ids"]
        if len(tokens) % alignment:
            continue
        if any(witness[k] != 0 for k in ("kv_usage", "running", "waiting")):
            raise ValueError(f"pre-warm hygiene failed: {witness}")
        from vllm import SamplingParams

        prime_len = (len(tokens) - 1) // alignment * alignment + 1
        # No DA payload means dense attention, including on the DA engine.
        # The original layout would be invalid for this truncated token prefix.
        adapter.llm.generate(
            [{"prompt_token_ids": tokens[:prime_len]}],
            [
                SamplingParams(
                    max_tokens=1, min_tokens=1, ignore_eos=True, seed=request_seed(batch.seed, occurrence, -2)
                )
            ],
        )
        primed.append({"request_index": occurrence, "prompt_len": len(tokens), "prime_len": prime_len})
    return primed


def run_cell(protocol, adapter, *, batch, out, arm, rep_index, boot_meta):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    wid = f"{batch.task}.b{batch.set_index}.{arm}.{rep_index}"
    meta = dict(boot_meta, arm=arm, rep_index=rep_index, batch_index=batch.set_index)
    doc = {
        "protocol": protocol.to_dict(),
        "task": batch.task,
        "batch_index": batch.set_index,
        "rep_index": rep_index,
        "arm": arm,
        "meta": meta,
        "faults": [],
        "windows": {},
    }
    try:
        witness = hygiene_witness(adapter, batch, arm)
        meta["pre_rep_hygiene"] = witness
        if not witness["ok"]:
            raise ValueError(f"pre-rep hygiene failed: {witness}")
        env.snapshot(out / "env.jsonl", "before")
        armed = adapter.arm(wid)
        if not armed.get("armed"):
            raise ValueError("worker recorder did not arm")
        doc["t_pass_start"] = time.time()
        try:
            outputs = adapter.run_arm(
                arm,
                batch,
                max_tokens=protocol.max_tokens,
                ignore_eos=arm == "vanilla",
                min_tokens=protocol.max_tokens if arm == "vanilla" else 0,
                seed=batch.seed,
                rep_index=rep_index,
            )
        finally:
            adapter.disarm()
            doc["t_pass_end"] = time.time()
            env.snapshot(out / "env.jsonl", "after")
        records = adapter.drain()
        if len(adapter.stats) != len(records):
            raise ValueError("scheduler and worker step counts disagree")
        for r, (usage, preempted) in zip(records, adapter.stats, strict=True):
            r.kv_usage, r.num_preempted = usage, preempted
        window = detect_window(records, concurrency=batch.size, protocol=protocol)
        write_table(out / f"steps_{wid}.jsonl", records)
        doc["windows"][wid] = {"start": window.start, "stop": window.stop}
        if len(outputs) != batch.size:
            raise ValueError("missing request outputs")
        requests = []
        for occurrence, (i, output) in enumerate(zip(batch.indices, outputs, strict=True)):
            prompt_len = len(adapter._prompts_for(arm)[i]["prompt_token_ids"])
            cached = output.num_cached_tokens
            # vLLM v1/core/kv_cache_manager.py:252-258 recomputes the last token.
            alignment = witness["cache_alignment"]
            expected = max(0, prompt_len - 1) // alignment * alignment
            deficit = None if cached is None else max(0, expected - cached)
            requests.append(
                {
                    "occurrence": occurrence,
                    "row_index": i,
                    "request_id": output.request_id,
                    "prompt_len": prompt_len,
                    "num_cached_tokens": cached,
                    "expected_aligned_hit": expected,
                    "deficit": deficit,
                    "seed": request_seed(batch.seed, occurrence, rep_index),
                    "output_tokens": len(output.outputs[0].token_ids),
                    "output_token_ids": list(output.outputs[0].token_ids),
                    "dattn_payload": (adapter.request_params[occurrence].extra_args or {}).get("dattn"),
                    "finish_reason": output.outputs[0].finish_reason,
                    "cache_hit_ok": deficit == 0 and 0 <= cached < prompt_len,
                }
            )
        occupancy = [r.kv_usage for r in records[window.start : window.stop]]
        meta.update(
            requests=requests,
            preempted_cell=adapter.preempted,
            prefill_steps_before_rectangle=sum(r.num_ctx_tokens > 0 for r in records[: window.start]),
            n_steps=window.n_steps,
            audit=adapter.audit(),
            realised_occupancy=sum(occupancy) / len(occupancy) if occupancy else None,
            measured_step_occupancy=occupancy,
            peak_kv_usage=max((r.kv_usage for r in records), default=0),
            prompt_tokens=sum(r["prompt_len"] for r in requests),
            cache_alignment=witness["cache_alignment"],
            cache_alignment_basis=witness["cache_alignment_basis"],
            recompute_tokens=sum(r["prompt_len"] - r["num_cached_tokens"] for r in requests)
            if all(r["num_cached_tokens"] is not None for r in requests)
            else None,
        )
        meta["prefill_steps_lower_bound"] = (
            math.ceil(meta["recompute_tokens"] / adapter.engine_kwargs["max_num_batched_tokens"])
            if meta["recompute_tokens"] is not None
            else None
        )
        meta["cache_miss"] = any(not r["cache_hit_ok"] for r in requests)
        assert not adapter.preempted and not any(r.num_preempted for r in records), "preempted batch"
        if meta["enable_prefix_caching"] and meta["cache_miss"]:
            failures = [
                (r["request_id"], r["deficit"], r["num_cached_tokens"]) for r in requests if not r["cache_hit_ok"]
            ]
            raise ValueError(f"prompt aligned cache hit gate failed (request, deficit, cached): {failures}")
        if not window.n_steps:
            raise ValueError("empty closed rectangle")
    except (ValueError, AssertionError) as exc:
        doc["faults"].append(str(exc))
        raise
    finally:
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        (out / "arm.json").write_text(json.dumps(doc, indent=2))
    return doc


def run_protocol(
    protocol, out_dir, *, profile=None, subsets=None, rows_file=None, arm="da", task_index=0, batch_index=0, pins=None
):
    from dataclasses import asdict

    from vllm import LLM

    from dattn.engine import launch
    from dattn.engine.assertions import check_prompt_templates
    from dattn.engine.profile import load_profile

    if arm not in ARMS or not 0 <= batch_index < protocol.batches_per_task:
        raise ValueError("invalid arm or batch index")
    if protocol.use == "walltime" and protocol.max_tokens != CAP:
        raise ValueError("walltime decode cap must be 512")
    profile = profile or load_profile(protocol.model_id, use=protocol.use)
    launch.apply_env(profile)
    os.environ.update(DATTN_ARM=arm, DATTN_RECORD="1")
    kwargs = launch.engine_kwargs(profile)
    max_seqs = int(kwargs["max_num_seqs"])
    kwargs["compilation_config"] = dict(
        kwargs.get("compilation_config", {}), cudagraph_capture_sizes=list(range(1, max_seqs + 1))
    )
    if arm == "vanilla":
        kwargs = builder_free_kwargs(kwargs)
    pins = Path(pins) if pins else Path(out_dir) / "pins"
    pool_record = pins / "pool.json"
    if not pool_record.is_file():
        raise ValueError(f"missing pinned boot pool record: {pool_record}")
    os.environ["DATTN_POOL_RECORD"] = str(pool_record.resolve())
    started = time.time()
    llm = LLM(**kwargs)
    tokenizer = llm.get_tokenizer()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pins = Path(pins) if pins else out / "pins"
    tasks = load_tasks(
        subsets, model_id=protocol.model_id, tokenizer=tokenizer, n=protocol.subset_size, rows_file=rows_file
    )
    task_list = list(tasks)
    if not 0 <= task_index < len(task_list):
        raise ValueError("task_index is outside the protocol task list")
    task = task_list[task_index]
    samples = tasks[task]
    prompts, layouts, lengths = build_prompts(tokenizer, samples)
    adapter = VllmEngineAdapter(llm, prompts=prompts, layouts=layouts, engine_kwargs=kwargs, model_id=protocol.model_id)
    meta = adapter.probe()
    if any(meta.get(k) is not (arm == "da") for k in ("dattn_builder_installed", "dattn_lp_installed")):
        raise ValueError("wrong live arm bindings")
    pool = meta["kv_pool"]
    identity = {
        "model": protocol.model_id,
        "task": task,
        "task_index": task_index,
        "task_list": task_list,
        "protocol": protocol.to_dict(),
        "profile": asdict(profile),
        "engine_kwargs": builder_free_kwargs(kwargs),
        "da_lengths": lengths[DA_FORMAT],
        "prompt_sha": identity_hash(prompts),
    }
    pin = pin_batches(
        pins / (task + ".json"),
        task=task,
        lengths=lengths[DA_FORMAT],
        protocol=protocol,
        groups=pool["groups"],
        physical_bytes_per_block=pool["physical_bytes_per_block"],
        pool_bytes=pool["pool_bytes"],
        max_num_seqs=max_seqs,
        identity=identity,
    )
    draw = pin["batches"][batch_index]
    c = draw["concurrency"]
    if not 0 < c <= max_seqs:
        raise ValueError(f"draw has invalid C={c}: {draw}")
    if c not in meta["resolved_config"]["capture_sizes"]:
        raise ValueError("resolved capture sizes omit pinned C")
    batch = Batch(
        task,
        batch_index,
        tuple(draw["indices"]),
        tuple(lengths[DA_FORMAT][i] for i in draw["indices"]),
        draw["modelled_occupancy"],
        draw["seed"],
    )
    task_proto = protocol.with_(concurrency=c)
    cell_dir = out / task / f"b{batch_index}/{arm}"
    if any(cell_dir.rglob("arm.json")):
        raise ValueError(f"capture already exists: {cell_dir}. Use a fresh output root")
    cell_dir.mkdir(parents=True, exist_ok=True)
    env.dump_cell(cell_dir / "nvidia-smi.txt")
    meta.update(
        process_start_unix=started,
        task_index=task_index,
        task_list=task_list,
        boot_id={"pid": os.getpid(), "started": started, "config_hash": identity_hash(meta["resolved_config"])},
        draw=draw,
        sampling_recipe=sampling_recipe(protocol.model_id),
        **check_prompt_templates(tokenizer, profile)["resolved"],
    )
    adapter.disarm()
    primed = prime_aligned_prompts(adapter, batch, arm, meta)
    adapter.run_arm(
        arm,
        batch,
        max_tokens=WARMUP_STEPS,
        ignore_eos=True,
        min_tokens=WARMUP_STEPS,
        seed=batch.seed,
        rep_index=-1,
        observe_resident=True,
    )
    witness = hygiene_witness(adapter, batch, arm)
    meta["warmup"] = {
        "ran": True,
        "steps": WARMUP_STEPS,
        "primed_requests": primed,
        "hygiene": witness,
        "resident_snapshot": adapter.resident_snapshot,
    }
    (cell_dir / "warmup.json").write_text(json.dumps(meta, indent=2))
    if not witness["ok"] or adapter.preempted:
        raise ValueError(f"warmup hygiene or preemption failed: {meta['warmup']}")
    validation = sizing_validation(adapter.resident_snapshot, pool, enable_prefix_caching=meta["enable_prefix_caching"])
    validation.update(task=task, task_index=task_index, batch_index=batch_index, arm=arm, boot_id=meta["boot_id"])
    validation_path = pins / "sizing_validation.json"
    validation["gated"] = not validation_path.exists() and task_index == batch_index == 0
    if validation_path.exists():
        prior = json.loads(validation_path.read_text())
        if prior["pool"] != pool or not prior["passed"]:
            raise ValueError("sizing validation pool changed or the first validation failed")
        validation["validated_at"] = {k: prior[k] for k in ("task", "batch_index", "arm", "boot_id")}
    elif not validation["gated"]:
        raise ValueError("run the first batch of the first task before later cells")
    meta["sizing_validation"] = validation
    (cell_dir / "warmup.json").write_text(json.dumps(meta, indent=2))
    print("S1 " + json.dumps(validation))
    if validation["gated"]:
        with validation_path.open("x") as f:
            json.dump(validation, f, indent=2)
        if not validation["passed"]:
            raise ValueError(
                f"S1 sizing validation failed: predicted={validation['predicted']}, measured={validation['measured']}"
            )

    cells = []
    for rep_index in range(protocol.repetitions):
        doc = run_cell(
            task_proto,
            adapter,
            batch=batch,
            out=cell_dir / f"r{rep_index}",
            arm=arm,
            rep_index=rep_index,
            boot_meta=meta,
        )
        doc["identity"] = batch_identity(
            batch, protocol=task_proto, sample_ids=[samples[i].sample_id for i in batch.indices], prompts_by_fmt=prompts
        )
        doc["identity"].update(task_index=task_index, pin_hash=identity_hash(pin))
        doc["identity_hash"] = identity_hash(doc["identity"])
        cells.append(doc)
    ended = time.time()
    for doc in cells:
        doc["meta"]["process_end_unix"] = ended
        path = cell_dir / f"r{doc['rep_index']}"
        (path / "meta.json").write_text(json.dumps(doc["meta"], indent=2))
        (path / "arm.json").write_text(json.dumps(doc, indent=2))
    result = {"n_cells": len(cells), "cells": cells, "faults": []}
    (out / "run.json").write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                "task": task,
                "C": c,
                "modelled_occupancy": draw["modelled_occupancy"],
                "shortfall": draw["shortfall"],
                "n_steps": [d["meta"]["n_steps"] for d in cells],
            }
        )
    )
    return result
