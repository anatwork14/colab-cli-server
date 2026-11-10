"""Turning a `BackendProfile` into an engine you can actually boot.

One place where the profile's `engine_args` and `env` become `vllm.LLM(**kwargs)` and
`os.environ`, so the GPU tests, `dattn probe` and the bench driver cannot drift apart.

Three things here are not obvious and are all load-bearing:

* **`DATTN_MODEL` must be exported before `vllm` builds its config.** The general plugin
  runs inside `EngineArgs.__post_init__` and has no access to the model id, so it reads
  the environment. `dattn_env` produces exactly what it needs.
* **`VLLM_PLUGINS` must be unset or contain `dattn`.** An empty string is a non-`None`
  allow-list that matches nothing, which silently skips the plugin and leaves
  `AttentionBackendEnum.CUSTOM` unregistered. Refuse this configuration rather than
  silently disable dattn (`research/NOTES.md` section 0.1).
* **`attention_config` is passed as a dict, not the TOML's JSON string.** `--attention-backend`
  and `--attention-config` are only mutually exclusive through `attention_config.backend`
  (`N:engine/arg_utils.py:2490-2496`), so a profile may keep both, and both do.
"""

from __future__ import annotations

import json
import os
from typing import Any

from dattn.contracts import BackendProfile, DattnBootError
from dattn.engine.plugin import ENV_CAPABILITY, ENV_MODEL, ENV_USE, LOGITS_PROCESSOR_FQCN

__all__ = ["apply_env", "dattn_env", "engine_kwargs", "builder_free_kwargs"]

#: A profile env value of this string means "assert this variable is NOT set".
_MUST_BE_UNSET = "__must_be_unset__"


def dattn_env(profile: BackendProfile) -> dict[str, str]:
    """The environment a worker needs. Includes the three the plugin reads."""
    env = {k: v for k, v in profile.env.items() if v != _MUST_BE_UNSET}
    env[ENV_MODEL] = profile.model_id
    env[ENV_CAPABILITY] = profile.capability
    env[ENV_USE] = profile.use
    return env


def apply_env(profile: BackendProfile, environ: dict[str, str] | None = None) -> dict[str, str]:
    """Export `dattn_env(profile)` and clear what the profile forbids. Returns what it set."""
    target = os.environ if environ is None else environ
    for name, value in profile.env.items():
        if value == _MUST_BE_UNSET:
            target.pop(name, None)
    env = dattn_env(profile)
    target.update(env)
    plugins = target.get("VLLM_PLUGINS")
    if plugins is not None and "dattn" not in plugins:
        raise DattnBootError(
            f"VLLM_PLUGINS={plugins!r} does not include 'dattn'. An allow-list that does not name "
            "dattn skips the general plugin, so AttentionBackendEnum.CUSTOM is never registered and "
            "the engine boots a silently vanilla configuration. Unset it or set it to 'dattn'."
        )
    return env


def _attention_config(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        return json.loads(raw)
    return dict(raw)


def engine_kwargs(profile: BackendProfile, **overrides: Any) -> dict[str, Any]:
    """`vllm.LLM(**engine_kwargs(profile))`. Overrides win, `attention_config` merges."""
    args = dict(profile.engine_args)
    ac = _attention_config(args.pop("attention_config", None))
    ac.update(_attention_config(overrides.pop("attention_config", None)))
    kwargs: dict[str, Any] = dict(args)
    if ac:
        kwargs["attention_config"] = ac
    kwargs.setdefault("logits_processors", [LOGITS_PROCESSOR_FQCN])
    kwargs.update(overrides)
    if kwargs.get("kv_cache_memory_bytes") == "TO BE PINNED FROM PROBE":
        raise DattnBootError(
            "walltime KV pool is TO BE PINNED FROM PROBE. Probe both arms at 0.92, "
            "then pin the smaller resolved pool rounded down to a whole GiB."
        )
    return kwargs


def builder_free_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Remove DA binding while retaining the backend and recording RPC bridge."""
    out = {k: v for k, v in kwargs.items() if k != "logits_processors" and not k.startswith("dattn")}
    ac = _attention_config(out.get("attention_config"))
    ac.pop("backend_per_kind", None)
    if ac:
        out["attention_config"] = ac
    else:
        out.pop("attention_config", None)
    return out
