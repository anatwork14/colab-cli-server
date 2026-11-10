"""The one per-worker boot point: assertions, then the optional recorder (integration).

vLLM 0.29.0 has no per-model-runner lifecycle hook. `vllm.general_plugins` load at
process init, long before a `GPUModelRunner` exists, and `worker_extension_cls` can only
*add* methods for `collective_rpc` to call, which the offline `LLM` path can drive but
the OpenAI server cannot: `vllm serve` exposes no RPC endpoint. So for
`dattn serve` there is exactly one place left to stand, and standing there is what this
module does:

    Worker.compile_or_warm_up_model()        N:v1/worker/gpu_worker.py:750
        ^ after KV-cache configuration, after cudagraph capture, before the first real
          forward. That is verbatim where DESIGN.md section 6 says the boot assertions
          belong, and it is after capture, so the recorder's wrapper is never inside a
          captured graph.

The hook is installed only when the general plugin resolves a profile. It is a no-op
when `DATTN_NO_BOOT_HOOK` is set. The wrapper delegates to the original method in every
case, including when dattn's own work raises.

The recorder is reached **by name**, never by import. `dattn.engine` may not import
`dattn.metrics` (`tests/unit/test_import_discipline.py`), so the hook resolves
`$DATTN_RECORD_HOOK`, default `dattn.metrics.hooks:install_from_env`, the same late
binding vLLM itself uses for the logits processor. The hook decides for itself whether to
arm, from `$DATTN_RECORD`.
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Final

__all__ = [
    "ENV_BOOT_ASSERT",
    "ENV_NO_BOOT_HOOK",
    "ENV_RECORD_HOOK",
    "RECORD_HOOK_DEFAULT",
    "boot_worker",
    "install_worker_boot_hook",
]

#: Set to a falsy value to boot even when a section 6 assertion fails. The default is to
#: refuse: a run that boots with the wrong backend bound produces numbers that look fine.
ENV_BOOT_ASSERT: Final[str] = "DATTN_BOOT_ASSERT"
#: Set to any truthy value to skip the rebinding entirely.
ENV_NO_BOOT_HOOK: Final[str] = "DATTN_NO_BOOT_HOOK"
#: `module:function`, called as `fn(runner)` once per worker.
ENV_RECORD_HOOK: Final[str] = "DATTN_RECORD_HOOK"
RECORD_HOOK_DEFAULT: Final[str] = "dattn.metrics.hooks:install_from_env"

_STATE: dict[str, Any] = {"installed": False, "reports": {}}


def _truthy(value: str | None, *, default: bool) -> bool:
    if value is None or not value.strip():
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def install_worker_boot_hook() -> bool:
    """Rebind `Worker.compile_or_warm_up_model` once. Returns whether it was installed."""
    if _STATE["installed"] or _truthy(os.environ.get(ENV_NO_BOOT_HOOK), default=False):
        return False
    try:
        from vllm.v1.worker.gpu_worker import Worker
    except Exception:  # noqa: BLE001 - a non-CUDA platform has no such worker; stay inert
        return False

    original = Worker.compile_or_warm_up_model

    def compile_or_warm_up_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        out = original(self, *args, **kwargs)
        boot_worker(self)
        return out

    compile_or_warm_up_model.__doc__ = original.__doc__
    compile_or_warm_up_model.__name__ = original.__name__
    compile_or_warm_up_model.__qualname__ = original.__qualname__
    Worker.compile_or_warm_up_model = compile_or_warm_up_model  # type: ignore[method-assign]
    _STATE["installed"] = True
    return True


def boot_worker(worker: Any) -> dict[str, Any]:
    """Run section 6 on this worker, then hand the runner to the recorder hook.

    Idempotent per process. Raises `DattnBootError` on a failed assertion unless
    `DATTN_BOOT_ASSERT` is falsy, because a silently mis-bound backend is the defect
    class this whole list exists to catch.
    """
    runner = getattr(worker, "model_runner", None)
    if runner is None:
        return {}
    cached = _STATE["reports"].get(id(runner))
    if cached is not None:
        return cached
    from dattn.engine import plugin
    from dattn.engine.assertions import check_pool_record, pool_geometry, validate_engine

    profile = plugin.current_profile()
    cfg = getattr(runner, "vllm_config", None) or worker.vllm_config
    budget = cfg.cache_config.kv_cache_memory_bytes or worker.available_kv_cache_memory_bytes
    runner.dattn_kv_pool = pool_geometry(runner.kv_cache_config, budget)
    check_pool_record(runner.dattn_kv_pool)
    report = validate_engine(
        cfg,
        runner,
        profile,
        arm=os.environ.get("DATTN_ARM", "da"),
        raise_on_failure=_truthy(os.environ.get(ENV_BOOT_ASSERT), default=True),
    )
    report["recorder"] = _call_record_hook(runner)
    _STATE["reports"][id(runner)] = report
    return report


def _call_record_hook(runner: Any) -> str:
    """Resolve `$DATTN_RECORD_HOOK` by name and call it. Never imports metrics."""
    spec = os.environ.get(ENV_RECORD_HOOK, RECORD_HOOK_DEFAULT).strip()
    if not spec:
        return "disabled: DATTN_RECORD_HOOK is empty"
    module_name, _, attr = spec.partition(":")
    try:
        fn = getattr(importlib.import_module(module_name), attr)
    except Exception as exc:  # noqa: BLE001 - a missing hook must not take the engine down
        return f"unavailable: {spec}: {type(exc).__name__}: {exc}"
    rec = fn(runner)
    return "armed" if rec is not None else "not armed"


def boot_report() -> dict[str, Any] | None:
    """The most recent report in this process, for `dattn_probe` to report.

    Keyed by runner, because a paired GPU run boots the DA arm and the vanilla arm in ONE
    process and the two have different reports.
    """
    reports = _STATE["reports"]
    return next(reversed(list(reports.values())), None) if reports else None
