"""The worker-side RPC surface, reached through `LLM.collective_rpc`.

`collective_rpc` accepts either a method name on the worker or a callable that is handed
the worker, so every function here takes an optional `worker` and falls back to the
process-global state when it is not given. That is what lets the same functions serve the
GPU tests in process and a real TP run over the RPC channel.

This module is also the ONLY door between engine and metrics. `tests/unit/test_import_discipline.py`
lets `dattn.metrics` import `dattn.engine.api` and nothing else from the engine, and forbids
`dattn.engine` from importing `dattn.metrics` at all, so the recorder is published through
`set_step_recorder` and read back through `get_step_recorder`.
"""

from __future__ import annotations

from typing import Any, Final

from dattn.contracts import DattnBootError, StepRecord

#: The names `collective_rpc` may be handed.
RPC_NAMES: Final[tuple[str, ...]] = (
    "dattn_probe",
    "dattn_record",
    "dattn_drain",
    "dattn_audit",
    "dattn_assert",
    "dattn_boot",
)

__all__ = [
    "RPC_NAMES",
    "dattn_assert",
    "dattn_boot",
    "dattn_audit",
    "dattn_drain",
    "dattn_probe",
    "dattn_record",
    "get_step_recorder",
    "set_step_recorder",
    "step_geometry",
]


# --------------------------------------------------------------------------------------
# The engine / metrics door
# --------------------------------------------------------------------------------------


def set_step_recorder(rec: Any | None) -> None:
    """Install or clear the `_REC` slot. Called only by `dattn.metrics.hooks`."""
    from dattn.engine import state  # local: api is inside the engine layer

    state.set_recorder(rec)


def get_step_recorder() -> Any | None:
    """Read the `_REC` slot. Lets `hooks.install` confirm the publish round-tripped."""
    from dattn.engine import state

    return state.get_recorder()


def step_geometry() -> dict[str, Any]:
    """The per-step facts the ENGINE layer can see, for the recorder's `StepFacts`.

    `num_reqs_padded`, the DA block size and the compacted-group count are known here and
    nowhere else in the recorder's reach. `cudagraph_mode`, `cudagraph_replayed` and the
    `SchedulerOutput` are **not**: the engine layer never sees either. The recorder's
    `_model_forward` wrapper does, because it runs inside `set_forward_context`, where
    `vllm.forward_context.get_forward_context()` resolves `batch_descriptor` and
    `cudagraph_runtime_mode` for THIS step.
    """
    from dattn.engine.state import DATTN_STATE

    scratches = list(DATTN_STATE._scratch.values())
    return {
        "step_id": DATTN_STATE.step_id,
        "num_reqs_padded": DATTN_STATE.last_num_reqs_padded,
        "n_compacted_groups": DATTN_STATE.last_n_compacted_groups,
        "dattn_block_size": scratches[0].block_size if scratches else 0,
        "n_live_rows": DATTN_STATE.n_live_rows(),
    }


# --------------------------------------------------------------------------------------
# The RPC surface
# --------------------------------------------------------------------------------------


def _runner(worker: Any = None) -> Any:
    if worker is not None:
        runner = getattr(worker, "model_runner", None)
        if runner is not None:
            return runner
    from dattn.engine.state import DATTN_STATE

    return DATTN_STATE.runner


def _vllm_config(worker: Any, runner: Any) -> Any:
    for holder in (runner, worker):
        cfg = getattr(holder, "vllm_config", None)
        if cfg is not None:
            return cfg
    raise DattnBootError("no VllmConfig reachable from the worker or the model runner")


def dattn_probe(worker: Any = None) -> dict[str, Any]:
    """The resolved engine and recorder facts written into each arm's metadata."""
    import os

    import torch

    from dattn.engine import plugin
    from dattn.engine.assertions import validate_engine
    from dattn.engine.boot import boot_report
    from dattn.engine.state import DATTN_STATE

    runner = _runner(worker)
    cfg = _vllm_config(worker, runner)
    profile = plugin.current_profile()
    report = validate_engine(cfg, runner, profile, raise_on_failure=False, arm=os.environ.get("DATTN_ARM", "da"))
    resolved = report["resolved"]

    groups = resolved.get("kv_cache_groups") or []
    out: dict[str, Any] = {
        "resolved_config": resolved["resolved_config"],
        "kv_pool": runner.dattn_kv_pool,
        "dattn_builder_installed": resolved.get("dattn_builder_installed"),
        "dattn_lp_installed": resolved.get("dattn_lp_installed"),
        "vllm_version": resolved.get("vllm_version"),
        "backend_cls": next((g["backend_cls"] for g in groups if g["kind"] == "full_attention"), None),
        "base_backend": (profile.base_backend if profile is not None else None),
        "runner_generation": "V2" if resolved.get("use_v2_model_runner") else "V1",
        "use_v2_model_runner": resolved.get("use_v2_model_runner"),
        "async_scheduling": resolved.get("async_scheduling"),
        "enable_prefix_caching": resolved.get("enable_prefix_caching"),
        "fsm_lag": resolved.get("fsm_lag", 1),
        "kv_cache_dtype": resolved.get("kv_cache_dtype"),
        "cudagraph_mode": resolved.get("cudagraph_mode"),
        "cudagraph_capture_sizes": resolved.get("cudagraph_capture_sizes"),
        "max_num_seqs": resolved.get("max_num_seqs"),
        "max_model_len": resolved.get("max_model_len"),
        "dattn_block_size": (resolved.get("dattn_block_sizes") or [0])[0],
        "n_full_attn_groups": len(resolved.get("full_attention_group_ids") or []),
        "n_compacted_groups": sum(g["is_dattn"] for g in groups),
        "kv_cache_groups": groups,
        "num_gpu_blocks": getattr(getattr(runner, "cache_config", None), "num_gpu_blocks", None),
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "profile": resolved.get("profile"),
        "audit": DATTN_STATE.audit(),
        "dattn_state": DATTN_STATE.snapshot(),
        "boot_assertions": {
            "failures": report["failures"],
            "skipped": report.get("skipped", {}),
            # Why the recorder is or is not attached. `dattn_record` can only report that
            # no recorder was published; this comes from the BOOT report, so it says
            # whether the hook ran at all, which is the difference between "DATTN_RECORD
            # was unset", "the hook blew up" and "the boot point never fired".
            "recorder": (boot_report() or {}).get("recorder", "no boot report"),
        },
    }
    return out


def dattn_record(enabled: bool, *, window_id: str = "", worker: Any = None) -> dict[str, Any]:
    """Arm or disarm the step recorder for one measurement window.

    The engine layer must not import `dattn.metrics`, so the recorder itself is installed
    by the boot hook (`dattn.engine.boot`, which resolves metrics by name when
    `DATTN_RECORD` is set). This call moves it between windows:

    * **arm** relabels the recorder and drops whatever is pending, so a window never
      begins with the tail of the previous arm's steps. That is the ABBA protocol's one
      requirement of the recorder.
    * **disarm** flushes, so the window's records are on disk (or in the ring, if no
      writer was attached) before the caller drains.

    Returns `armed: False` with a reason when no recorder was published, which is what
    tells the bench driver that `DATTN_RECORD` was not set on the engine it booted.
    """
    rec = get_step_recorder()
    if rec is None:
        return {
            "armed": False,
            "window_id": window_id,
            "reason": "no recorder published: boot the engine with DATTN_RECORD=1 (dattn serve --record)",
        }
    dropped = 0
    if enabled:
        drain = getattr(rec, "drain", None)
        if drain is not None:
            dropped = len(drain())
    else:
        flush = getattr(rec, "flush", None)
        if flush is not None:
            flush()
    setattr(rec, "window_id", window_id)  # noqa: B010 - a plain attribute on the recorder's object
    return {"armed": bool(enabled), "window_id": window_id, "dropped": dropped}


def dattn_boot(worker: Any = None) -> dict[str, Any]:
    """Run the per-worker boot point by hand. Idempotent, and normally automatic.

    `dattn.engine.boot` installs itself from the general plugin and runs at the end of
    `Worker.compile_or_warm_up_model`, so nothing needs to call this. It exists for the
    offline path that wants the report back (`llm.collective_rpc("dattn_boot")`) and for
    a worker booted with the hook disabled.
    """
    from dattn.engine.boot import boot_report, boot_worker

    if worker is not None:
        return boot_worker(worker)
    return boot_report() or {}


def dattn_drain(worker: Any = None) -> list[StepRecord]:
    """Drain the ring once, with a single `synchronize()`. Empty when unarmed."""
    rec = get_step_recorder()
    if rec is None:
        return []
    return list(rec.drain())


def dattn_audit(worker: Any = None) -> dict[str, Any]:
    """`{"checked": n, "mismatches": m, "first_mismatch": ...}`. `m` must be 0.

    Two independent audits, merged: the engine's own device-vs-host-mirror check on the
    compacted `seq_lens`, and, when a recorder is armed, the recorder's ring audit.
    """
    from dattn.engine.state import DATTN_STATE

    out = dict(DATTN_STATE.audit())
    rec = get_step_recorder()
    ring = getattr(rec, "audit", None)
    if ring is not None:
        out["recorder"] = ring()
    return out


def dattn_assert(worker: Any = None) -> dict[str, Any]:
    """Re-run DESIGN.md section 6. Raises `DattnBootError` naming every failure."""
    import os

    from dattn.engine import plugin
    from dattn.engine.assertions import validate_engine

    runner = _runner(worker)
    return validate_engine(
        _vllm_config(worker, runner), runner, plugin.current_profile(), arm=os.environ.get("DATTN_ARM", "da")
    )
