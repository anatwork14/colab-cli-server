"""Live binding, window, cascade, prefix-cache and LP guards."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from dattn.contracts import BackendProfile, DattnError

PINNED_VLLM_VERSION = "0.29.0"  # Shared with the unchanged upstream contract tests.


@dataclass
class Report:
    resolved: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def to_dict(self):
        return {"resolved": self.resolved, "failures": self.failures}


class _Ctx:
    def __init__(self, vllm_config: Any, runner: Any, profile: BackendProfile | None, arm: str = "da") -> None:
        self.cfg = vllm_config
        self.runner = runner
        self.profile = profile
        self.arm = arm
        self.report = Report()
        self.current = ""

    def record(self, key: str, value: Any) -> Any:
        self.report.resolved[key] = value
        return value

    def want(self, ok: bool, message: str) -> None:
        if not ok:
            self.report.failures.append(f"{self.current}: {message}")

    def eq(self, label: str, got: Any, expected: Any) -> None:
        self.record(label, got)
        self.want(got == expected, f"{label} is {got!r}, expected {expected!r}")


def _groups(c: _Ctx) -> list[dict[str, Any]] | None:
    """Per KV-cache group: kind, backend name, builder class, kernel block size."""
    runner = c.runner
    if runner is None or not getattr(runner, "attn_groups", None):
        return None
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

    out: list[dict[str, Any]] = []
    kv_groups = runner.kv_cache_config.kv_cache_groups
    for gid, attn_groups in enumerate(runner.attn_groups):
        spec = kv_groups[gid].kv_cache_spec
        for ag in attn_groups:
            builder = ag.get_metadata_builder()
            if hasattr(builder, "dattn_group_id"):
                builder.dattn_group_id = gid
            out.append(
                {
                    "kv_cache_group_id": gid,
                    "kind": str(get_kv_cache_spec_kind(spec).value),
                    "backend": ag.backend.get_name(),
                    "base_backend": getattr(ag.backend, "dattn_base_backend", ag.backend).get_name(),
                    "backend_cls": f"{ag.backend.__module__}.{ag.backend.__qualname__}",
                    "builder_cls": type(builder).__name__,
                    "base_builder_cls": getattr(type(builder), "dattn_base_builder_cls", type(builder)).__name__,
                    "kernel_block_size": int(builder.kv_cache_spec.block_size),
                    "is_dattn": hasattr(builder, "dattn_needs_seq_lens_cpu"),
                    "supports_update_block_table": bool(builder.supports_update_block_table),
                    "sliding_window": getattr(spec, "sliding_window", None),
                    "layer_names": list(ag.layer_names),
                    "n_layers": len(ag.layer_names),
                    "_builder": builder,
                }
            )
    return out


def _a02_a07(c: _Ctx, groups: list[dict[str, Any]] | None) -> None:
    p = c.profile
    if not groups:
        c.current = "A03"
        c.want(False, "attention groups are unavailable")
        return

    public = [{k: v for k, v in g.items() if not k.startswith("_")} for g in groups]
    c.record("kv_cache_groups", public)
    dattn_gids = sorted({g["kv_cache_group_id"] for g in groups if g["is_dattn"]})
    full_gids = sorted({g["kv_cache_group_id"] for g in groups if g["kind"] == "full_attention"})
    c.record("dattn_group_ids", dattn_gids)
    c.record("full_attention_group_ids", full_gids)
    c.record("dattn_builder_installed", bool(dattn_gids))
    c.current = "A03"
    requested = getattr(getattr(c.cfg, "attention_config", None), "backend", None)
    requested = getattr(requested, "name", requested)
    if p is not None:
        c.eq("attention_backend", requested, p.base_backend)
        for g in groups:
            if g["kind"] == "full_attention":
                c.want(g["base_backend"] == requested, f"resolved backend {g['base_backend']} != requested {requested}")
    if c.arm == "vanilla":
        c.want(not dattn_gids, f"vanilla has DA builders on groups {dattn_gids}")
        c.record("dattn_block_sizes", [])
        return

    c.current = "A03"
    c.want(
        dattn_gids == full_gids,
        f"DA builders sit on groups {dattn_gids} but the full_attention groups are {full_gids}. "
        'The binding is --attention-config \'{"backend_per_kind": {"full_attention": "CUSTOM"}}\'.',
    )
    mismatched = [
        f"group {g['kv_cache_group_id']}/{g['builder_cls']} kind={g['kind']} is_dattn={g['is_dattn']}"
        for g in groups
        if g["is_dattn"] != (g["kind"] == "full_attention")
    ]
    c.want(
        not mismatched,
        "every full_attention builder must be a DA builder and no other builder may be one, "
        f"but: {' | '.join(mismatched)}",
    )

    for g in groups:
        if g["is_dattn"]:
            c.want(
                not g["supports_update_block_table"],
                f"group {g['kv_cache_group_id']}/{g['builder_cls']} supports_update_block_table=True "
                "can bypass DA scratch staging",
            )

    c.current = "A04"
    sizes = sorted({g["kernel_block_size"] for g in groups if g["is_dattn"]})
    c.record("dattn_block_sizes", sizes)
    c.want(len(sizes) <= 1, f"the DA groups disagree on the kernel block size: {sizes}")
    if p is not None and sizes:
        c.want(
            sizes[0] == p.dattn_block_size,
            f"the DA kernel block size is {sizes[0]}, the profile says {p.dattn_block_size}. "
            "It comes from the hybrid page arithmetic, not from --block-size.",
        )

    c.current = "A07"
    windowed = [g["kv_cache_group_id"] for g in groups if g["is_dattn"] and g["sliding_window"]]
    c.want(
        not windowed,
        f"DA is bound to windowed groups {windowed}. Every decode kernel in scope masks the window on "
        "the PACKED key index, so compacting a windowed row drops the sink and admits ancient pages.",
    )
    _a07_layer_windows(c, groups)


def _window_is_off(window: Any) -> bool:
    """True when `window` positively means "no sliding window"."""
    if window is None:
        return True
    if isinstance(window, int):
        return window < 0
    if isinstance(window, (tuple, list)) and window:
        try:
            return all(int(v) < 0 for v in window)
        except (TypeError, ValueError):
            return False
    return False


def _a07_layer_windows(c: _Ctx, groups: list[dict[str, Any]]) -> None:
    """A07's second conjunct: no COMPACTED layer runs with an active sliding window.

    The first conjunct reads the KV-cache spec, which is what decides cache layout. This
    one reads the attention implementation, which is what decides the kernel mask, and
    the two can disagree: a full-attention spec still gets a windowed `impl` if the model
    passed `sliding_window=` to `Attention` for that layer. The window is applied on the
    PACKED key index, so on a compacted row it counts the wrong keys.

    This check only ever fails on POSITIVE evidence of a window. A layer we cannot
    resolve, an impl with no `sliding_window` attribute and an unrecognised encoding are
    all recorded and skipped, never failed, because a boot assertion that refuses a good
    configuration is worse than one that misses a bad one. The resolved windows go into
    `meta.json` under `a07_layer_windows` so a missed case is still visible after the run.
    """
    ctx = getattr(getattr(c.cfg, "compilation_config", None), "static_forward_context", None)
    if not isinstance(ctx, dict):
        c.record("a07_layer_windows", None)
        c.record("a07_layers_unresolved", None)
        return
    observed: dict[str, Any] = {}
    unresolved: list[str] = []
    bad: list[str] = []
    for g in groups:
        if not g["is_dattn"]:
            continue
        for name in g.get("layer_names", ()):
            layer = ctx.get(name)
            impl = getattr(layer, "impl", None)
            if impl is None or not hasattr(impl, "sliding_window"):
                unresolved.append(name)
                continue
            window = impl.sliding_window
            observed[name] = list(window) if isinstance(window, (tuple, list)) else window
            if not _window_is_off(window):
                bad.append(f"{name}={window!r}")
    c.record("a07_layer_windows", observed)
    c.record("a07_layers_unresolved", unresolved)
    c.want(
        not bad,
        f"compacted layers carry an active sliding window: {', '.join(bad)}. The window is masked on "
        "the PACKED key index, so compacting the row drops the sink and admits ancient pages.",
    )


def _a08(c: _Ctx) -> None:
    c.eq("disable_cascade_attn", bool(c.cfg.model_config.disable_cascade_attn), True)


def _a11(c: _Ctx) -> None:
    expected = False if c.profile is None else c.profile.engine_args["enable_prefix_caching"]
    c.eq("enable_prefix_caching", bool(c.cfg.cache_config.enable_prefix_caching), expected)


def _a14(c: _Ctx) -> None:
    from importlib.metadata import entry_points

    from dattn.engine.plugin import LOGITS_PROCESSOR_FQCN

    lps = list(c.cfg.model_config.logits_processors or ())
    c.record("logits_processors", [str(x) for x in lps])
    declared = any("DattnLogitsProcessor" in str(x) for x in lps)
    batch = getattr(c.runner, "input_batch", None)
    live = getattr(getattr(batch, "logitsprocs", None), "all", None)
    installed_da = declared if live is None else any(type(lp).__name__ == "DattnLogitsProcessor" for lp in live)
    c.record("dattn_lp_installed", installed_da)
    if c.arm == "vanilla":
        c.want(not declared and not installed_da, "vanilla must not install DattnLogitsProcessor")
        return
    c.want(
        any(str(x) == LOGITS_PROCESSOR_FQCN or "DattnLogitsProcessor" in str(x) for x in lps),
        f"{LOGITS_PROCESSOR_FQCN} is not in model_config.logits_processors ({lps}). Only the "
        "--logits-processors route sets logitsprocs_need_output_token_ids, without which the FSM "
        "sees -1 forever and freezes with no log line (upstream #52461).",
    )
    installed = [ep.value for ep in entry_points(group="vllm.logits_processors")]
    c.record("vllm_logits_processor_entry_points", installed)
    c.want(
        not installed,
        f"a vllm.logits_processors entry point is installed ({installed}). The MRV2 demotion check "
        "counts installed entry points, so the runner would depend on what is in the venv (A14).",
    )
    if c.runner is None or getattr(c.runner, "input_batch", None) is None:
        return
    flag = bool(c.runner.input_batch.logitsprocs_need_output_token_ids)
    c.eq("logitsprocs_need_output_token_ids", flag, True)

    installed = [
        lp
        for lp in getattr(c.runner.input_batch, "logitsprocs", None).all  # type: ignore[union-attr]
        if type(lp).__name__ == "DattnLogitsProcessor"
    ]
    c.record("n_dattn_logits_processors", len(installed))
    c.want(len(installed) == 1, f"expected exactly one DattnLogitsProcessor in the batch, found {len(installed)}")
    for lp in installed:
        c.eq("is_argmax_invariant", bool(lp.is_argmax_invariant()), False)


def resolved_config(
    cfg: Any, runner: Any, profile: BackendProfile | None, groups: list[dict[str, Any]]
) -> dict[str, Any]:
    """Record resolved engine arguments and geometry for the cross-process pair check."""
    import json

    sections = [cfg] + [
        getattr(cfg, name, None)
        for name in (
            "model_config",
            "cache_config",
            "parallel_config",
            "scheduler_config",
            "compilation_config",
            "attention_config",
        )
    ]
    args = {}
    keys = set(() if profile is None else profile.engine_args) | {
        "model",
        "kv_cache_memory_bytes",
        "max_model_len",
        "max_num_seqs",
        "max_num_batched_tokens",
        "worker_extension_cls",
        "logits_processors",
        "attention_config",
        "seed",
    }
    for key in sorted(keys):
        attr = {"attention_backend": "backend", "kv_cache_dtype": "cache_dtype"}.get(key, key)
        for section in sections:
            if section is not None and hasattr(section, attr):
                value = getattr(section, attr)
                if is_dataclass(value) and not isinstance(value, type):
                    value = asdict(value)
                args[key] = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
                break
    cache = getattr(cfg, "cache_config", None)
    compilation = getattr(cfg, "compilation_config", None)
    scheduler = getattr(cfg, "scheduler_config", None)
    out = {
        "engine_args": args,
        "attention_backend": getattr(getattr(cfg, "attention_config", None), "backend", None),
        "block_sizes": [g["kernel_block_size"] for g in groups],
        "cudagraph_mode": getattr(compilation, "cudagraph_mode", None),
        "capture_sizes": getattr(compilation, "cudagraph_capture_sizes", None),
        "max_num_batched_tokens": getattr(scheduler, "max_num_batched_tokens", None),
        "num_gpu_blocks": getattr(getattr(runner, "cache_config", cache), "num_gpu_blocks", None),
    }
    if hasattr(runner, "dattn_kv_pool"):
        out["kv_pool"] = runner.dattn_kv_pool
    return json.loads(json.dumps(out, default=str))


def check_prompt_templates(tokenizer: Any, profile: BackendProfile | None) -> dict:
    """Template hashes are provenance, recorded without a version/hash gate."""
    from dattn.prompt.render import template_hashes

    hashes = template_hashes(tokenizer)
    return {
        "resolved": {"chat_template_sha256": hashes["chat_template"], "instruction_template_sha256": hashes},
        "failures": [],
    }


def validate_engine(
    vllm_config: Any,
    runner: Any = None,
    profile: BackendProfile | None = None,
    *,
    raise_on_failure: bool = True,
    arm: str = "da",
) -> dict:
    import triton

    assert callable(triton.jit), "Triton is required for DA compaction"
    from dattn.engine.state import DATTN_STATE

    DATTN_STATE.runner = runner
    c = _Ctx(vllm_config, runner, profile, arm)
    groups = _groups(c)
    _a02_a07(c, groups)
    for name, fn in (("A08", _a08), ("A11", _a11), ("A14", _a14)):
        c.current = name
        fn(c)
    snapshot = resolved_config(vllm_config, runner, profile, groups or [])
    c.record("resolved_config", snapshot)
    for key in ("cudagraph_mode", "num_gpu_blocks", "max_num_batched_tokens"):
        c.record(key, snapshot[key])
    c.record("cudagraph_capture_sizes", snapshot["capture_sizes"])
    for section, keys in (
        ("model_config", ("max_model_len",)),
        ("scheduler_config", ("max_num_seqs", "async_scheduling")),
        ("cache_config", ("cache_dtype",)),
    ):
        for key in keys:
            c.record(key, getattr(getattr(vllm_config, section, None), key, None))
    c.record("runner_class", f"{type(runner).__module__}.{type(runner).__qualname__}")
    c.record("use_v2_model_runner", bool(getattr(vllm_config, "use_v2_model_runner", False)))
    c.record("profile", None if profile is None else f"{profile.model_id}/{profile.use}")
    if c.report.failures and raise_on_failure:
        raise DattnError("\n".join(c.report.failures))
    return c.report.to_dict()


def pool_geometry(cache, budget):
    """Reconcile the shared backing allocation, not the sum of aliased capacities."""
    from vllm.v1.kv_cache_interface import get_kv_cache_spec_kind

    groups = []
    for gid, group in enumerate(cache.kv_cache_groups):
        spec = group.kv_cache_spec
        specs = getattr(spec, "kv_cache_specs", None)
        pages = [specs[name].page_size_bytes if specs else spec.page_size_bytes for name in group.layer_names]
        groups.append(
            {
                "group_id": gid,
                "kind": get_kv_cache_spec_kind(spec).value,
                "block_size": spec.block_size,
                "sliding_window": getattr(spec, "sliding_window", None),
                "num_gpu_blocks": cache.num_blocks,
                "layer_names": list(group.layer_names),
                "page_size_bytes": pages,
                "page_bytes": sum(pages),
                "logical_full_capacity_bytes": cache.num_blocks * sum(pages),
            }
        )
    physical = max(g["page_bytes"] for g in groups)
    allocation = cache.num_blocks * physical
    backing = {t.size for t in cache.kv_cache_tensors}
    assert backing == {allocation}, f"physical allocation {allocation} != vLLM backing sizes {backing}"
    remainder = int(budget) - allocation
    assert 0 <= remainder < physical, f"pool budget {budget}, allocation {allocation}, remainder {remainder}"
    return {
        "resolved_kv_cache_memory_bytes": int(budget),
        "pool_bytes": allocation,
        "physical_bytes_per_block": physical,
        "num_gpu_blocks": cache.num_blocks,
        "null_block_reservation": physical,
        "usable_pool_bytes": allocation - physical,
        "rounding_remainder": remainder,
        "backing_allocation_bytes": allocation,
        "groups": groups,
    }


def check_pool_record(pool):
    """A recorded pinned boot fixes geometry for both arm processes."""
    import json
    import os
    from pathlib import Path

    if path := os.environ.get("DATTN_POOL_RECORD"):
        expected = json.loads(Path(path).read_text())
        if expected != pool:
            raise DattnError(f"pinned KV pool geometry changed from {path}")
