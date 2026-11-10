"""The `vllm.general_plugins` entry point.

At plugin load (which runs in every worker, the EngineCore and the front end,
`N:plugins/__init__.py:18,77`):

    assert not AttentionBackendEnum.CUSTOM.is_overridden()   # one slot, last writer wins
    base = AttentionBackendEnum[profile.base_backend].get_class()
    DattnBackend = _make_backend(base)                       # module-level in dattn.engine.backend
    register_backend(AttentionBackendEnum.CUSTOM, "dattn.engine.backend.DattnBackend")

There is deliberately NO `vllm.logits_processors` entry point: the LP is registered by
fully qualified class name through `ModelConfig.logits_processors`, because only the CLI
route sets `logitsprocs_need_output_token_ids` and because the MRV2 demotion check counts
installed entry points (DESIGN.md section 2.7).

The plugin loads in **every** vLLM process on the box, including ones that have nothing
to do with dattn, so it is inert unless it is told which profile to build from. Two
environment variables do that, and one of them must be set:

    DATTN_MODEL=google/gemma-4-31B-it  [DATTN_CAPABILITY=sm100] [DATTN_USE=parity]
    DATTN_BASE_BACKEND=TRITON_ATTN     # the escape hatch, no profile, no assertions
"""

from __future__ import annotations

import os
from typing import Any, Final

from dattn.contracts import BackendProfile, DattnBootError

#: The FQCN that `--logits-processors` and `engine_args.logits_processors` must name.
LOGITS_PROCESSOR_FQCN: Final[str] = "dattn.engine.processor:DattnLogitsProcessor"

#: The class path handed to `register_backend` for `AttentionBackendEnum.CUSTOM`.
BACKEND_CLASS_PATH: Final[str] = "dattn.engine.backend.DattnBackend"

#: The environment that selects the profile the plugin builds from.
ENV_MODEL: Final[str] = "DATTN_MODEL"
ENV_CAPABILITY: Final[str] = "DATTN_CAPABILITY"
ENV_USE: Final[str] = "DATTN_USE"
ENV_BASE_BACKEND: Final[str] = "DATTN_BASE_BACKEND"

__all__ = [
    "BACKEND_CLASS_PATH",
    "ENV_BASE_BACKEND",
    "ENV_CAPABILITY",
    "ENV_MODEL",
    "ENV_USE",
    "LOGITS_PROCESSOR_FQCN",
    "current_profile",
    "install_plugin",
]

_STATE: dict[str, Any] = {"installed": False, "profile": None, "boot_hook": False}


def current_profile() -> BackendProfile | None:
    """The profile the plugin installed, or `None` if the plugin was inert."""
    return _STATE["profile"]


def _resolve_profile() -> tuple[str, BackendProfile | None]:
    model_id = os.environ.get(ENV_MODEL, "").strip()
    if model_id:
        from dattn.engine.profile import load_profile

        profile = load_profile(
            model_id,
            os.environ.get(ENV_CAPABILITY, "sm100").strip() or "sm100",
            os.environ.get(ENV_USE, "parity").strip() or "parity",
        )
        return profile.base_backend, profile
    base = os.environ.get(ENV_BASE_BACKEND, "").strip()
    if base:
        return base, None
    return "", None


def install_plugin(profile: BackendProfile | None = None) -> None:
    """The `vllm.general_plugins` hook. Registers `DattnBackend` in the CUSTOM slot."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    from dattn.engine.backend import install_backend

    if _STATE["installed"]:
        return

    if profile is not None:
        base_name: str = profile.base_backend
    else:
        base_name, profile = _resolve_profile()
    if not base_name:
        # Without an explicit DA profile, leave backend registration unchanged.
        return

    if AttentionBackendEnum.CUSTOM.is_overridden():
        raise DattnBootError(
            "AttentionBackendEnum.CUSTOM is already registered by someone else. There is one "
            "slot and the last writer wins, so dattn refuses to overwrite it silently "
            f"(current: {AttentionBackendEnum.CUSTOM.get_path()})."
        )
    if os.environ.get("VLLM_ATTENTION_BACKEND") is not None:
        raise DattnBootError(
            "A15: VLLM_ATTENTION_BACKEND is set, but it was removed from vLLM in 0.29.0 and is a "
            "silent no-op. Its presence means a stale launch script is running."
        )

    if profile is not None and profile.use == "parity_prefix":
        from dattn.engine.prefix import install_guards

        install_guards()

    base = AttentionBackendEnum[base_name].get_class()
    install_backend(base, profile)
    register_backend(AttentionBackendEnum.CUSTOM, BACKEND_CLASS_PATH)
    # The plugin runs in every process, including the worker, and it is the only dattn
    # code that runs there before a model runner exists. So this is where the per-worker
    # boot point is armed: after KV-cache configuration and cudagraph capture, and before
    # the first real forward, `dattn.engine.boot` runs DESIGN.md section 6 and hands the
    # runner to the step recorder. It is installed only on a process that resolved a
    # profile, i.e. only on a dattn engine.
    from dattn.engine.boot import install_worker_boot_hook

    _STATE["boot_hook"] = install_worker_boot_hook()
    _STATE["installed"] = True
    _STATE["profile"] = profile
