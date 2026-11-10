"""`install()` / `uninstall()`, the zero-overhead toggle.

Disabled is ZERO overhead, not cheap overhead. The mechanism is instance-level rebinding
done once: `install(runner)` rebinds `runner._model_forward` to a wrapper and installs
the event ring, `uninstall` restores the original bound method. `_model_forward` is an
explicitly documented override point in 0.29.0. This is the ONLY rebinding in dattn
(docs/DESIGN.md section 6).

The disabled path, exactly
--------------------------
After `uninstall()` (or before any `install()`), `runner._model_forward` is the original
bound method **by identity**: there is no wrapper frame, no closure, no try/finally and
no event record. The rest of dattn consults one module-level slot:

    from dattn.engine import state
    if state._REC is not None:              # <- one LOAD_FAST-class load and one branch
        state._REC.note_lp(...)             #    per step, never per row

`_REC` is `None` when disabled, so the whole measurement path is one `is not None` test
per call site per step. `GATE_SOURCE` below is that idiom as a string, and
`tests/unit/metrics/test_hooks.py` compiles it, executes it 20 000 times under
`tracemalloc` and asserts zero bytes allocated and exactly one traced line per call.

Publishing `_REC`
-----------------
`dattn.metrics` may import `dattn.engine.api` and nothing else from engine
(`tests/unit/test_import_discipline.py`), while `dattn.engine` may not import
`dattn.metrics` at all. So the slot is published through one function on the api
surface, `dattn.engine.api.set_step_recorder(rec)`, which forwards to
`dattn.engine.state.set_recorder`. If that API is unavailable, `install()` records
the failed publication in `recorder.publish_ok`.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any, Final

import dattn.engine.api as engine_api
from dattn.contracts import Mode
from dattn.metrics.recorder import DEFAULT_RING, StepFacts, StepRecorder

#: The documented disabled-path idiom, as source, so a test can measure it.
GATE_SOURCE: Final[str] = "if _REC is not None:\n    _REC.note_lp(0.0, ())\n"

#: Set on the runner instance so `uninstall` is exact and `install` is idempotent-safe.
ATTR: Final[str] = "_dattn_metrics"

#: Set to a truthy value to arm the recorder from inside a `vllm serve` worker. Read by
#: `install_from_env`, which the engine's boot hook resolves by name (see
#: `dattn.engine.boot`), never by an import from engine into metrics.
ENV_RECORD: Final[str] = "DATTN_RECORD"
#: Where the worker writes `steps_<window>.jsonl`. Empty means "hold in the ring only".
ENV_RECORD_DIR: Final[str] = "DATTN_RECORD_DIR"

__all__ = [
    "ATTR",
    "ENV_RECORD",
    "ENV_RECORD_DIR",
    "GATE_SOURCE",
    "flush_now",
    "install",
    "install_from_env",
    "installed",
    "publish",
    "uninstall",
    "write_audit",
]


def publish(rec: StepRecorder | None) -> bool:
    """Publish the engine recorder slot. Return False if the setter is unavailable."""
    setter = getattr(engine_api, "set_step_recorder", None)
    if setter is None:
        return False
    setter(rec)
    return True


def installed(runner: Any) -> StepRecorder | None:
    """The recorder currently bound to `runner`, or None."""
    state = getattr(runner, ATTR, None)
    return None if state is None else state["recorder"]


def _resolve(runner: Any, name: str, default: Any) -> Any:
    """Read a resolved fact off the runner, never off a CLI string."""
    if hasattr(runner, name):
        return getattr(runner, name)
    cfg = getattr(runner, "vllm_config", None)
    for holder in ("scheduler_config", "model_config", "cache_config", "parallel_config"):
        sub = getattr(cfg, holder, None)
        if sub is not None and hasattr(sub, name):
            return getattr(sub, name)
    return default


def install(
    runner: Any,
    *,
    window_id: str,
    ring: int = DEFAULT_RING,
    max_num_seqs: int | None = None,
    torch_mod: Any = None,
    device: Any = None,
    dattn_block_size: int = 0,
    fsm_lag: int = 1,
    out_path: str | None = None,
) -> StepRecorder:
    """Rebind `runner._model_forward`, allocate the event ring, publish `_REC`.

    Raises `NotImplementedError` if `runner` does not expose the 0.29.0 hook surface
    engine owns: the recorder has no path for an object without `_model_forward`, and a
    silent no-op here is exactly the "the mask did nothing for weeks" defect class.
    """
    if not callable(getattr(runner, "_model_forward", None)):
        raise NotImplementedError(
            "metrics.install needs a vLLM 0.29.0 GPUModelRunner exposing _model_forward "
            f"(the documented override point, N:v1/worker/gpu_model_runner.py:3917); got {type(runner).__name__}"
        )
    if getattr(runner, ATTR, None) is not None:
        raise RuntimeError(f"dattn metrics already installed on {type(runner).__name__}: uninstall first")

    if torch_mod is None:  # pragma: no cover -- exercised only inside a real worker
        try:
            import torch

            torch_mod = torch
        except ImportError:
            torch_mod = None
    if max_num_seqs is None:
        max_num_seqs = int(_resolve(runner, "max_num_seqs", 0) or 0)
    if max_num_seqs < 1:
        raise NotImplementedError("metrics.install could not resolve max_num_seqs from the runner; pass it explicitly")

    if not dattn_block_size:
        groups = getattr(getattr(runner, "kv_cache_config", None), "kv_cache_groups", [])
        for group, attention in zip(groups, getattr(runner, "attn_groups", []), strict=False):
            if type(group.kv_cache_spec).__name__ == "FullAttentionSpec" and attention:
                dattn_block_size = int(attention[0].get_metadata_builder().kv_cache_spec.block_size)
                break

    rec = StepRecorder(
        window_id=window_id,
        max_num_seqs=max_num_seqs,
        ring=ring,
        torch_mod=torch_mod,
        device=device if device is not None else _resolve(runner, "device", None),
        async_scheduling=bool(_resolve(runner, "async_scheduling", False)),
        # The measured lag is 1 in BOTH scheduling modes on MRV1: the "2 under async"
        # comment of DESIGN.md section 3 is refuted by the on-hardware probe
        # (research/NOTES.md section 1). It is recorded, never inferred from the
        # mode, and the boot assertion checks the measured value on every arm.
        fsm_lag=fsm_lag,
        runner_generation="V2" if _resolve(runner, "use_v2_model_runner", False) else "V1",
        dattn_block_size=dattn_block_size,
        out_path=out_path,
    )

    orig_forward = runner._model_forward
    orig_sample = getattr(runner, "_sample", None)
    orig_execute = getattr(runner, "execute_model", None)
    # `_sample` IS THE STEP TERMINATOR. It is the only caller of `rec.end_step`, so a
    # runner without it produces a recorder that opens every step and closes none: no
    # `t_ready_ns`, no step row, and the failure looks like an empty window rather than a
    # missing hook. `install` used to wrap it only
    # `if callable(orig_sample)` and say nothing otherwise, which made a renamed upstream
    # method a silent loss of the whole measurement.
    if not callable(orig_sample):
        raise NotImplementedError(
            "metrics.install needs a runner exposing a callable _sample "
            f"(N:v1/worker/gpu_model_runner.py, the only caller of end_step); got "
            f"{type(runner).__name__} with _sample = {orig_sample!r}. Without it every step would "
            "be opened and never closed"
        )
    n_full_attn_groups = _count_full_attention_groups(runner)
    #: Filled inside `_model_forward`, which is the ONLY scope where the resolved
    #: cudagraph mode of THIS step is readable, and consumed in `_sample`, which is the
    #: only scope that has the `SchedulerOutput`.
    graph: dict[str, Any] = {"mode": "", "replayed": False, "num_tokens": 0, "so": None, "engine_step": 0}
    tally: dict[str, int] = {"preempted": 0}
    #: A one-shot callable that `_arm_end_of_run_drain` parks here and the FIRST forward
    #: runs and clears. It cannot run at install time: see `_install_signal_drain`.
    arm: dict[str, Any] = {"fn": None}

    batch = getattr(runner, "input_batch", None)
    orig_refresh = getattr(batch, "refresh_metadata", None)
    has_da_lp = any(
        type(lp).__name__ == "DattnLogitsProcessor" for lp in getattr(getattr(batch, "logitsprocs", None), "all", ())
    )
    admitted: dict[str, int] = {}

    def _refresh(*args: Any, **kwargs: Any) -> Any:
        result = orig_refresh(*args, **kwargs)
        if not has_da_lp:
            rows = []
            live = set(batch.req_ids)
            for req in admitted.keys() - live:
                del admitted[req]
            for i, req in enumerate(batch.req_ids):
                length = int(batch.num_tokens_no_spec[i])
                out_pos = length - int(batch.num_prompt_tokens[i])
                since = graph["engine_step"] - admitted.setdefault(req, graph["engine_step"])
                rows.append((req, int(Mode.GLOBAL), length, length, out_pos, since))
            # Dense att_blk remains absent, as on the frozen builder-free recorder.
            rec.note_lp(0.0, rows)
        return result

    def _execute(scheduler_output: Any = None, *args: Any, **kwargs: Any) -> Any:
        # Both arms start the host period at the same worker entry boundary.
        graph["engine_step"] += 1
        rec.begin_step(t_sched_ns=time.perf_counter_ns())
        # The `SchedulerOutput` is an ARGUMENT of `execute_model` and nothing else keeps
        # it: `execute_model` caches it in `runner.execute_model_state` for `sample_tokens`,
        # and `sample_tokens` clears that slot BEFORE it calls `_sample`
        # (`N:v1/worker/gpu_model_runner.py:4653-4655`), so by the time the sampler runs it
        # is gone. Reading it here is the only way the facts can be right, and a record
        # whose `kind` is a default rather than a classification is worse than no record.
        graph["so"] = scheduler_output
        return orig_execute(scheduler_output, *args, **kwargs)

    def _forward(*args: Any, **kwargs: Any) -> Any:
        pending = arm["fn"]
        if pending is not None:
            # One dict load and one branch per step, and the body runs exactly once per
            # worker. It is deferred to here rather than done in `install` because the
            # thing it arms is a signal handler and vLLM installs its own AFTER the
            # engine constructor that called us returns. See `_install_signal_drain`.
            arm["fn"] = None
            pending()
        rec.forward_start()
        try:
            return orig_forward(*args, **kwargs)
        finally:
            rec.forward_end()
            _read_forward_context(graph)

    def _sample(*args: Any, **kwargs: Any) -> Any:
        _note_step_facts(rec, runner, graph, tally, n_full_attn_groups)
        rec.sample_start()
        try:
            return orig_sample(*args, **kwargs)
        finally:
            rec.sample_end()
            rec.end_step(t_ready_ns=time.perf_counter_ns())
            # NOTHING ELSE GOES HERE. This closure is the step path. A flush drains the
            # ring, and `drain()` costs one `cuda.synchronize()` plus a whole-ring D2H
            # copy of the audit mirror, so a periodic flush from here is an instrument
            # perturbing the thing it measures. Draining happens at end of run
            # (`uninstall`, or the `atexit` hook `install_from_env` registers) or
            # on an explicit out-of-band `flush_now(runner)`. See `test_hooks.py`
            # ::test_no_synchronize_happens_between_arm_and_disarm.

    own = vars(runner) if hasattr(runner, "__dict__") else {}
    saved = {
        "recorder": rec,
        "_model_forward": (orig_forward if "_model_forward" in own else None),
        "_sample": (orig_sample if "_sample" in own else None),
        "execute_model": (orig_execute if "execute_model" in own else None),
        "wrapped_sample": callable(orig_sample),
        "wrapped_execute": callable(orig_execute),
        "arm": arm,
        "refresh": (batch, orig_refresh if "refresh_metadata" in vars(batch) else None)
        if callable(orig_refresh)
        else None,
    }
    if callable(orig_refresh):
        batch.refresh_metadata = _refresh
    runner._model_forward = _forward
    runner._sample = _sample
    if callable(orig_execute):
        runner.execute_model = _execute
    setattr(runner, ATTR, saved)
    rec.publish_ok = publish(rec)
    return rec


def uninstall(runner: Any) -> StepRecorder | None:
    """Restore the original bound method and clear `_REC`.

    After this the attribute resolves exactly as it did before `install()`: when the
    original lived on the class, the instance attribute is deleted rather than
    overwritten, so the step path is byte-identical to stock vLLM. Returns the recorder
    that was removed.
    """
    state = getattr(runner, ATTR, None)
    if state is None:
        publish(None)
        return None
    _disarm_signal_drain(state.get("arm", {}).pop("restore", ()))
    if state.get("refresh") is not None:
        batch, original = state["refresh"]
        _restore(batch, "refresh_metadata", original)
    _restore(runner, "_model_forward", state["_model_forward"])
    if state["wrapped_sample"]:
        _restore(runner, "_sample", state["_sample"])
    if state.get("wrapped_execute"):
        _restore(runner, "execute_model", state["execute_model"])
    setattr(runner, ATTR, None)
    publish(None)
    rec: StepRecorder = state["recorder"]
    rec.close()
    return rec


def install_from_env(runner: Any, **over: Any) -> StepRecorder | None:
    """Arm the recorder from the environment, or do nothing. The `vllm serve` door.

    `dattn.engine.boot` resolves this function **by name** out of `$DATTN_RECORD_HOOK`
    (defaulting to `dattn.metrics.hooks:install_from_env`) and calls it once per worker,
    after KV-cache configuration and cudagraph capture and before the first real forward.
    Resolving by name rather than by import is not a trick: it is what keeps
    `dattn.engine` free of any reference to `dattn.metrics`, which
    `tests/unit/test_import_discipline.py` enforces, and it is the same late binding vLLM
    itself uses for the logits processor.

    Returns the recorder, or `None` when `$DATTN_RECORD` is unset, which is the normal
    case and costs one `os.environ` lookup per worker boot.
    """
    if not _truthy(os.environ.get(ENV_RECORD)):
        return None
    if installed(runner) is not None:
        return installed(runner)
    out_dir = os.environ.get(ENV_RECORD_DIR, "").strip()
    window = os.environ.get("DATTN_RECORD_WINDOW", "serve").strip() or "serve"
    kwargs: dict[str, Any] = {"window_id": window}
    if out_dir:
        from pathlib import Path

        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        kwargs["out_path"] = str(d / f"steps_{window}.jsonl")
    kwargs.update(over)
    rec = install(runner, **kwargs)
    if out_dir:
        _arm_end_of_run_drain(runner, rec, str(kwargs["out_path"]))
    return rec


def _arm_end_of_run_drain(runner: Any, rec: StepRecorder, out_path: str) -> Callable[[], None]:
    """Make end of run reach the file, from a worker nobody calls into. Never a step hook.

    A server is stopped with a signal rather than with a call into dattn, and the ring
    that holds the records lives in the EngineCore worker PROCESS, not in the API server
    that the operator signals. Normal API-server shutdown does not guarantee that the
    worker executes `atexit`. Without a worker-side drain, an armed recorder can leave
    its output file empty despite completing generation.

    So end of run is now reached three ways, and each is off the step path:

    * `atexit`, as before, for an interpreter that does shut down normally.
    * `SIGTERM` and `SIGINT` delivered to THIS process, chained in front of vLLM's own
      handlers by `_install_signal_drain`. `proc.terminate()` on the EngineCore process is
      literally how `vllm.v1.utils.shutdown` stops a worker
      (`N:v1/utils.py:618-623`), so this is the path a clean stop actually takes.
    * the ring wrap, in `StepRecorder._open`: when a writer is attached the early drain
      the wrap already forced now goes to the writer instead of into an unbounded
      in-memory `_overflow` list.

    None of the three restores the every-512-steps flush that `d000e38` removed. That one
    CAUSED a `cuda.synchronize()` and a whole-ring device-to-host copy inside a timed
    step; these three either run when no step will be timed again, or ride a drain the
    ring wrap had already forced. A caller that wants the file complete at an arbitrary
    moment mid-run still asks out of band with `flush_now(runner)`.

    `rec.close()` and `rec.flush()` are both idempotent, so registering this next to an
    explicit `uninstall()` is safe. Returns the atexit closure so a test can call it.
    """
    import atexit

    done = {"closed": False, "busy": False}

    def _close() -> None:
        if done["closed"]:
            return
        done["closed"] = True
        rec.close()
        write_audit(rec, out_path)

    def _drain() -> int:
        """Push the ring and KEEP the writer open, so a late step still has a home.

        The signal path uses this rather than `_close` on purpose: vLLM's `SIGTERM`
        handler only REQUESTS shutdown (`N:v1/engine/core.py:1349-1355`) and the busy loop
        then finishes the work it is holding, so steps can still arrive after us. Closing
        the writer here would drop exactly those. `_writer.flush()` hands the bytes to the
        OS, so they survive this process even if `atexit` never runs.
        """
        if done["closed"] or done["busy"]:
            return 0
        done["busy"] = True
        try:
            n = rec.flush()
            write_audit(rec, out_path)
            return n
        except Exception:  # noqa: BLE001 - a lost tail is bad, a crashed shutdown is worse
            return 0
        finally:
            done["busy"] = False

    atexit.register(_close)
    state = getattr(runner, ATTR, None)
    if state is not None:
        arm = state["arm"]

        def _do_arm() -> None:
            arm["restore"] = _install_signal_drain(_drain)

        arm["fn"] = _do_arm
    return _close


def _install_signal_drain(drain: Callable[[], int]) -> list[tuple[int, Any]]:
    """Chain `drain` in front of whatever already handles `SIGTERM` and `SIGINT` here.

    Armed from the first forward, not from `install_from_env`, and the ordering is the
    whole reason. `dattn.engine.boot` calls the record hook from inside the EngineCore
    constructor, and `EngineCoreProc.run_engine_core` installs vLLM's own handlers AFTER
    that constructor returns (`N:v1/engine/core.py:1328-1358`), so a handler armed at boot
    is overwritten before the first step and would never fire. By the first forward the
    busy loop is running and vLLM's handler is the one we capture and call through to.

    Chaining, not replacing: `previous` runs after the drain, so vLLM still sets its
    shutdown state and still exits its loop, and a `SIG_DFL` is restored and re-raised so
    the process still dies of the signal it was sent. `signal.signal` is main-thread only
    and raises `ValueError` elsewhere, in which case `atexit` stays the only end-of-run
    drain and nothing else changes.

    Returns `(signum, previous)` pairs for `_disarm_signal_drain`, so `uninstall` puts the
    process back the way it found it: this module's rule is that disabled leaves no trace,
    and a signal handler is as much a trace as a rebound method.
    """
    import signal

    armed: list[tuple[int, Any]] = []
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous = signal.getsignal(signum)
        except (OSError, ValueError):  # pragma: no cover - not reachable on posix
            continue
        if getattr(previous, "_dattn_drain", False):
            continue

        def _handler(sig: int, frame: Any, previous: Any = previous) -> Any:
            drain()
            if callable(previous):
                return previous(sig, frame)
            if previous is signal.SIG_DFL:
                signal.signal(sig, signal.SIG_DFL)
                os.kill(os.getpid(), sig)
            return None

        _handler._dattn_drain = True  # type: ignore[attr-defined]
        try:
            signal.signal(signum, _handler)
        except ValueError:
            continue
        armed.append((int(signum), previous))
    return armed


def _disarm_signal_drain(armed: Any) -> None:
    """Put back what `_install_signal_drain` replaced, and only where we are still it."""
    import signal

    for signum, previous in armed:
        try:
            if getattr(signal.getsignal(signum), "_dattn_drain", False):
                signal.signal(signum, previous)
        except (OSError, ValueError):  # pragma: no cover - off the main thread
            continue


def flush_now(runner: Any) -> int:
    """Drain the ring to the writer, off the step path, on an explicit request.

    This is the ONLY sanctioned mid-run drain. It costs a `cuda.synchronize()`, so a
    caller that invokes it while the engine is stepping is perturbing its own
    measurement; call it between windows, never from inside one. Returns the number of
    records written, and 0 when the recorder holds no writer (the bench path, which
    drains in process through `engine.drain()` instead).
    """
    rec = installed(runner)
    if rec is None:
        return 0
    return rec.flush()


def write_audit(rec: StepRecorder, out_path: str) -> str:
    """Write `audit_<window>.json` next to the step file, at every drain to the writer.

    The audit is the only thing that proves the per-row numbers in the records are the
    numbers the KERNEL read: `checked`/`mismatches` compare the device ring, copied D2D
    from the compacted `seq_lens` the builder handed the metadata, against the host mirror
    the compaction computed independently in numpy. A mismatch is a real defect and not a
    tautology, and `mismatches` must be 0.

    It is a sidecar rather than a field on the records because it is a property of the
    window, not of a step. It is written when the ring reaches the writer, which is at
    end of run (`uninstall`, or the `atexit` hook `install_from_env` registers) or on an
    explicit `flush_now`, and never from inside a step.
    """
    import json
    from pathlib import Path

    path = Path(out_path)
    target = path.with_name(path.name.replace("steps_", "audit_", 1)).with_suffix(".json")
    target.write_text(json.dumps(rec.audit(), indent=2, default=str) + "\n", encoding="utf-8")
    return str(target)


def _truthy(value: str | None) -> bool:
    return bool(value) and value.strip().lower() not in {"0", "false", "no", "off", ""}


def _count_full_attention_groups(runner: Any) -> int:
    """`n_full_attn_groups`, resolved once at install and never on the step path."""
    cfg = getattr(runner, "kv_cache_config", None)
    groups = getattr(cfg, "kv_cache_groups", None) or []
    n = 0
    for group in groups:
        spec = getattr(group, "kv_cache_spec", None)
        if type(spec).__name__ == "FullAttentionSpec":
            n += 1
    return n


def _read_forward_context(graph: dict[str, Any]) -> None:
    """The resolved cudagraph mode of THIS step. Only readable inside the forward.

    `_build_attention_metadata` runs before `set_forward_context` and the logits
    processor runs before that, so neither the engine layer nor `begin_step` can see
    this. `research/NOTES.md` section 2.0 is the reason it has to travel with the
    record: a replayed graph and an eager step cost different amounts. Pairing checks
    the modes the kept steps actually ran under.
    """
    try:
        from vllm.forward_context import get_forward_context

        ctx = get_forward_context()
    except Exception:  # noqa: BLE001 - outside a forward context there is nothing to read
        return
    mode = getattr(ctx, "cudagraph_runtime_mode", None)
    graph["mode"] = getattr(mode, "name", "") or str(mode or "")
    graph["replayed"] = graph["mode"] not in ("", "NONE")
    descriptor = getattr(ctx, "batch_descriptor", None)
    graph["num_tokens"] = int(getattr(descriptor, "num_tokens", 0) or 0)


def _note_step_facts(
    rec: StepRecorder, runner: Any, graph: dict[str, Any], tally: dict[str, int], n_full_attn_groups: int
) -> None:
    """Build the `StepFacts` and hand them to the recorder, once per step.

    `_sample` is the one scope that sees both halves: `runner.execute_model_state` still
    holds the `SchedulerOutput` that `execute_model` cached for it
    (`N:v1/worker/gpu_model_runner.py:486-500,4591`), and `graph` was filled a few
    microseconds earlier inside the forward. Engine geometry that the recorder cannot
    inspect directly arrives through `api.step_geometry()`.
    """
    so = graph["so"]
    if so is None:
        so = getattr(getattr(runner, "execute_model_state", None), "scheduler_output", None)
    if so is None:
        return
    try:
        geom = engine_api.step_geometry()
    except Exception:  # noqa: BLE001 - a step before the state store is configured
        geom = {}
    rec.set_block_size(int(geom.get("dattn_block_size", 0) or 0))
    preempted = getattr(so, "preempted_req_ids", None) or ()
    tally["preempted"] += len(preempted)
    num_reqs = int(getattr(getattr(runner, "input_batch", None), "num_reqs", 0) or 0)
    padded = graph["num_tokens"] if graph["replayed"] else 0
    facts = StepFacts.from_scheduler_output(
        so,
        num_reqs=num_reqs,
        engine_step=int(graph.get("engine_step") or geom.get("step_id", 0)),
        num_reqs_padded=int(padded or geom.get("num_reqs_padded", 0) or num_reqs),
        cudagraph_replayed=bool(graph["replayed"]),
        cudagraph_mode=str(graph["mode"]),
        n_full_attn_groups=n_full_attn_groups,
    )
    rec.note_facts(facts)
    # The driver observes scheduler allocations in process and joins by step ordinal.
    # The worker can count preemptions directly but does not own the KV pool gauge.
    rec.note_scheduler_stats(0.0, tally["preempted"])


def _restore(runner: Any, name: str, original: Any) -> None:
    """Put the attribute back exactly as it was: a class descriptor stays a descriptor."""
    if original is None:
        delattr(runner, name)
    else:
        setattr(runner, name, original)
