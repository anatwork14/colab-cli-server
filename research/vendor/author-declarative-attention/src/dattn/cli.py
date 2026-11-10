"""Public commands for serving, evaluation, replay and paired benchmarks.

Heavy dependencies load inside handlers. Engine commands resolve the explicit
model and use from research/backend_profiles.toml. See docs/DESIGN.md section 4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

COMMANDS: Final[tuple[str, ...]] = (
    "serve",
    "probe",
    "render",
    "replay",
    "eval",
    "judge",
    "parity",
    "bench",
    "analyse",
)

#: Default directory for engine step recordings, relative to the working directory.
DEFAULT_RECORD_DIR: Final[str] = "runs/records"

#: The headline arms. `--arm` picks the prompt format and whether the mask is live, which
#: is the only difference between DA and DA-nm: dac's two template files are byte
#: identical and the `_no_mask` arm is the same prompt with the mask off.
ARMS: Final[dict[str, tuple[str, bool]]] = {
    "da": ("chunk_multi_turn_tool", True),
    "da_eqseg": ("chunk_multi_turn_tool", True),
    "da_nm": ("chunk_multi_turn_tool", False),
    "vanilla": ("vanilla", False),
    "xml": ("chunk_single_turn_xml", True),
    "xml_nm": ("chunk_single_turn_xml", False),
}

#: Segmentation is independent of prompt format. Every existing arm keeps its default.
ARM_SEGMENTERS: Final[dict[str, str]] = {arm: "equal_length" if arm == "da_eqseg" else "semantic" for arm in ARMS}

__all__ = ["ARMS", "ARM_SEGMENTERS", "COMMANDS", "build_parser", "main"]


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------


def _add_profile_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True, help="model id, e.g. google/gemma-4-31B-it")
    p.add_argument("--capability", default="sm100", help="GPU capability key in backend_profiles.toml")
    p.add_argument("--profile", default="parity", help="profile key: parity | walltime | ...")


def _add_server_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--base-url", default="http://localhost:8000", help="OpenAI-compatible server")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--timeout", type=float, default=3600.0)
    p.add_argument("--concurrency", type=int, default=32)


def build_parser() -> argparse.ArgumentParser:
    """The top-level parser. One sub-parser per entry of `COMMANDS`."""
    parser = argparse.ArgumentParser(prog="dattn", description="Declarative Attention on vLLM")
    parser.add_argument("--version", action="store_true", help="print the dattn version and exit")
    sub = parser.add_subparsers(dest="command", metavar="{" + " | ".join(COMMANDS) + "}")

    p_serve = sub.add_parser("serve", help="print or exec the vllm serve command for a profile")
    _add_profile_args(p_serve)
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--exec", action="store_true", help="exec the server instead of printing it")
    # REMAINDER, not "*": everything after `--extra` is a `vllm serve` flag, and `nargs="*"`
    # stops at the first token that starts with a dash, which is every flag there is.
    p_serve.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="everything after this is passed through to `vllm serve`; must come last",
    )
    p_serve.add_argument(
        "--record",
        action="store_true",
        help="arm the step recorder in the worker (DATTN_RECORD=1). Off means zero overhead, "
        "not cheap overhead: no wrapper is installed at all",
    )
    p_serve.add_argument(
        "--record-dir",
        type=Path,
        default=None,
        help=f"where the worker writes steps_<window>.jsonl (default {DEFAULT_RECORD_DIR} when --record is given)",
    )
    p_serve.add_argument("--window", default="serve", help="the window id the recorder labels its records with")
    p_serve.add_argument(
        "--no-check",
        action="store_true",
        help="skip the A17 chat-template check, which needs the tokenizer and runs before the boot",
    )

    p_probe = sub.add_parser("probe", help="resolve a profile and re-run the boot assertions")
    _add_profile_args(p_probe)
    p_probe.add_argument("--json", action="store_true")
    p_probe.add_argument("--no-boot", action="store_true", help="resolve the profile only, do not boot an engine")
    p_probe.add_argument("--max-model-len", type=int, default=None)
    p_probe.add_argument("--max-num-seqs", type=int, default=None)
    p_probe.add_argument("--gpu-memory-utilization", type=float, default=None)
    p_probe.add_argument("--arm", choices=("da", "vanilla"), default="da")
    p_probe.add_argument("--resolve-pool", action="store_true", help="probe utilization without a KV byte pin")
    p_probe.add_argument("--pool-out", type=Path, help="write the resolved pool record")
    p_probe.add_argument("--pool-record", type=Path, help="refuse a change from a recorded pinned boot")

    p_render = sub.add_parser("render", help="render one prompt and print its layout")
    p_render.add_argument("--model", required=True)
    p_render.add_argument("--context-file", type=Path, required=True)
    p_render.add_argument("--question", required=True)
    p_render.add_argument("--format", dest="fmt", default="chunk_multi_turn_tool")
    from dattn.prompt.segment import SEGMENTERS

    p_render.add_argument("--segmenter", choices=SEGMENTERS, default="semantic")
    p_render.add_argument("--no-tool-schema", action="store_true")
    p_render.add_argument("--show-text", action="store_true")

    p_replay = sub.add_parser("replay", help="replay a recorded response through the FSM")
    p_replay.add_argument("--model", required=True)
    p_replay.add_argument("--context-file", type=Path, required=True)
    p_replay.add_argument("--question", required=True)
    p_replay.add_argument("--response-file", type=Path, required=True)
    p_replay.add_argument("--run-meta", type=Path, required=True, help="the rollout meta.json controls rendering")

    p_eval = sub.add_parser("eval", help="select rows, render, generate, persist")
    _add_profile_args(p_eval)
    _add_server_args(p_eval)
    p_eval.add_argument("--campaign-cell", type=Path)
    p_eval.add_argument("--campaign-shard", type=Path)
    p_eval.add_argument("--arm", choices=sorted(ARMS), default="da")
    p_eval.add_argument("--template-policy", choices=("dac_bundled", "stock"), default="dac_bundled")
    p_eval.add_argument("--out", type=Path, default=Path("runs/eval"))
    p_eval.add_argument("--n-per-subset", type=int, default=128)
    p_eval.add_argument("--subsets", nargs="*", default=None, help="default: the 15 headline subsets")
    p_eval.add_argument("--seed", type=int, default=42)
    p_eval.add_argument("--max-tokens", type=int, default=8192)
    p_eval.add_argument("--chunk-size", type=int, default=64)
    p_eval.add_argument("--num-proc", type=int, default=None, help="processes for the context-token filter")
    # The two answer policies are mutually exclusive: `--first-answer` is a no-op alias for
    # what is now the default, so asking for both is a contradiction rather than a
    # precedence question, and argparse refuses it instead of one of them winning silently.
    g_answer = p_eval.add_mutually_exclusive_group()
    g_answer.add_argument(
        "--last-answer",
        action="store_true",
        help="grade the LAST <answer> block. The default is the FIRST, which is the paper's "
        "measurement (see docs/DESIGN.md section 10)",
    )
    g_answer.add_argument(
        "--first-answer",
        action="store_true",
        help=argparse.SUPPRESS,  # now the default; kept so a running script does not break
    )
    p_eval.add_argument(
        "--no-replay-metrics",
        action="store_true",
        help="skip the CPU replay that fills dac_metrics, mode_trace and the attended-token "
        "counts on every row. A rollout-only smoke run, not a publishable cell",
    )
    p_eval.add_argument("--select-only", action="store_true", help="write rows.jsonl and stop, no engine needed")

    p_judge = sub.add_parser("judge", help="grade an output directory with the pinned judge")
    _add_server_args(p_judge)
    p_judge.add_argument("--dir", type=Path, required=True, help="a cell directory holding responses.jsonl")
    p_judge.add_argument("--judge-model", default=None)
    p_judge.add_argument(
        "--judge-max-tokens",
        type=int,
        default=0,
        help="0 keeps the pinned 32768. Lower it only for a smoke run, and say so: the cap is "
        "part of the judge recipe and a truncated verdict is recorded as unparsable",
    )
    p_judge.add_argument("--judge-all", action="store_true", help="also judge rows that failed the format gate")
    p_judge.add_argument("--chunk-size", type=int, default=256)
    p_judge.add_argument("--seed", type=int, default=None)

    p_parity = sub.add_parser("parity", help="mask-versus-dense identity and prompt self-check")
    p_parity.add_argument("--left", type=Path, required=True)
    p_parity.add_argument("--right", type=Path, required=True)
    p_parity.add_argument("--mode", choices=("mask",), default="mask")
    p_parity.add_argument("--json-out", type=Path, default=None)

    p_bench = sub.add_parser("bench", help="capture one benchmark arm and batch with repetitions in one boot")
    _add_profile_args(p_bench)
    p_bench.set_defaults(profile="walltime")
    p_bench.add_argument("--out", type=Path, required=True)
    p_bench.add_argument("--arm", choices=("da", "vanilla"), required=True)
    p_bench.add_argument("--pins", type=Path, required=True, help="shared per-task draw JSON directory")
    p_bench.add_argument("--subsets", nargs="+", default=["headline"])
    p_bench.add_argument("--rows", type=Path, default=None)
    p_bench.add_argument("--task-index", type=int, default=0, help="position in the --subsets task list")
    p_bench.add_argument("--batch-index", type=int, default=0)

    p_analyse = sub.add_parser("analyse", help="aggregate eval or bench outputs")
    p_analyse.add_argument("--pair", action="store_true", help="pair two bench arm roots before aggregating")
    p_analyse.add_argument(
        "--pair-out", type=Path, default=Path("bench_paired"), help="directory for paired cell summaries"
    )
    p_analyse.add_argument("--dirs", type=Path, nargs="+", required=True)
    p_analyse.add_argument(
        "--all-subsets",
        action="store_true",
        help="count every subset on disk. The DEFAULT is the 15 headline subsets, which is the "
        "paper's population. A cell that does not carry all 15 is never quietly trimmed to an "
        "overlap: it is counted whole with a warning that its macro is not the paper's number",
    )
    p_analyse.add_argument(
        "--headline-subsets",
        action="store_true",
        help=argparse.SUPPRESS,  # now the default; kept for one release so scripts do not break
    )
    p_analyse.add_argument("--subsets", nargs="*", default=None, help="an explicit subset list, overriding both")
    # The bench half of `analyse`. `--headline-subsets` is not redeclared here:
    # the hidden alias above is the same flag and argparse refuses a duplicate.
    p_analyse.add_argument(
        "--bench",
        action="store_true",
        help="aggregate bench cells (--dirs are run roots) into the 15-task macro-average of "
        "main.tex:845, instead of aggregating eval judgments",
    )
    p_analyse.add_argument(
        "--allow-partial", action="store_true", help="label a diagnostic macro with fewer than 15 tasks"
    )
    p_analyse.add_argument("--plot", type=Path, default=None)
    p_analyse.add_argument("--json-out", type=Path, default=None)
    return parser


# --------------------------------------------------------------------------------------
# Handlers
# --------------------------------------------------------------------------------------


def _profile(args: argparse.Namespace) -> Any:
    from dattn.engine.profile import load_profile

    return load_profile(args.model, args.capability, args.profile)


def _tokenizer(model_id: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id)


def _server(args: argparse.Namespace) -> Any:
    from dattn.eval.client import ServerConfig

    return ServerConfig(
        base_url=args.base_url,
        model=getattr(args, "model", "") or "",
        api_key=args.api_key,
        timeout_s=args.timeout,
        concurrency=args.concurrency,
    )


#: vLLM 0.29.0 `EngineArgs` booleans declared `store_true` rather than
#: `argparse.BooleanOptionalAction`, so `--no-<flag>` is NOT a thing the parser accepts.
#: Measured by introspecting `EngineArgs.add_cli_args` on the installed 0.29.0: exactly
#: three of them, against 58 that do take `--no-`. Every one of the three defaults to
#: False, which is what makes OMITTING a false value identical to expressing it.
#:
#: This is the fallback list, for a box with no vllm importable. `_no_negated_form` reads
#: the real parser when it can, because that is ground truth and a hand-kept list rots.
NO_NEGATED_FORM: Final[frozenset[str]] = frozenset(
    {"disable-log-stats", "aggregate-engine-logging", "data-parallel-multi-port-external-lb"}
)

_NEGATION_CACHE: dict[str, frozenset[str]] = {}


def _no_negated_form() -> frozenset[str]:
    """Flag names (dashed, no leading `--`) that `vllm serve` takes only as store_true.

    Read off the installed vLLM's own parser, so it cannot disagree with the binary that
    is about to parse the line. Falls back to `NO_NEGATED_FORM` when vllm is not
    importable, which is the same set measured on 0.29.0.
    """
    if "v" in _NEGATION_CACHE:
        return _NEGATION_CACHE["v"]
    found = NO_NEGATED_FORM
    try:
        import argparse as _ap

        from vllm.engine.arg_utils import EngineArgs

        parser = EngineArgs.add_cli_args(_ap.ArgumentParser())
        names = set()
        for action in parser._actions:  # noqa: SLF001 - argparse exposes no public view
            if isinstance(action, _ap._StoreTrueAction | _ap._StoreFalseAction):  # noqa: SLF001
                names |= {o.removeprefix("--") for o in action.option_strings if o.startswith("--")}
        if names:
            found = frozenset(names)
    except Exception:  # noqa: BLE001 - no vllm, or a parser that moved: use the measured set
        found = NO_NEGATED_FORM
    _NEGATION_CACHE["v"] = found
    return found


def serve_argv(profile: Any, *, host: str, port: int, extra: Sequence[str] = ()) -> list[str]:
    """Render profile engine flags with vLLM's boolean argument conventions.

    False store_true flags are omitted because vLLM has no negated spelling.
    Other false booleans use --no-<name>. True booleans use the bare flag.
    """
    argv = ["vllm", "serve", str(profile.engine_args.get("model", profile.model_id))]
    store_true_only = _no_negated_form()
    for key, value in profile.engine_args.items():
        if key == "model":
            continue
        name = key.replace("_", "-")
        flag = "--" + name
        if isinstance(value, bool):
            if value:
                argv.append(flag)
            elif name not in store_true_only:
                argv.append(f"--no-{name}")
            # else: false, and the flag has no negated form. vLLM's default is already
            # False, so saying nothing says exactly this.
        elif isinstance(value, list):
            argv += [flag, *[str(v) for v in value]]
        else:
            argv += [flag, str(value)]
    return [*argv, "--host", host, "--port", str(port), *extra]


def serve_env(profile: Any, *, record: bool = False, record_dir: Path | None = None, window: str = "serve") -> dict:
    """The environment the server needs, including the three the general plugin reads.

    `profile.env` alone is NOT enough and getting this wrong is silent: the plugin runs
    inside `EngineArgs.__post_init__` with no access to the model id, so without
    `DATTN_MODEL` it stays inert, `AttentionBackendEnum.CUSTOM` is never registered and
    the server boots a configuration that is vanilla with a logits processor bolted on.
    `launch.dattn_env` is the one function that gets this right, so it is the one used.
    """
    from dattn.engine.launch import dattn_env
    from dattn.metrics.hooks import ENV_RECORD, ENV_RECORD_DIR

    env = dict(dattn_env(profile))
    if record:
        env[ENV_RECORD] = "1"
        env["DATTN_RECORD_WINDOW"] = window
        # A recorder with no writer keeps its records in the ring and drops the oldest on
        # wrap, so `--record` without a directory would arm an instrument nobody can read.
        # It defaults rather than warns: a server is stopped with a signal, and a warning
        # printed at boot is not what the operator is looking at three hours later.
        env[ENV_RECORD_DIR] = str(record_dir if record_dir is not None else DEFAULT_RECORD_DIR)
    return env


def _cmd_serve(args: argparse.Namespace) -> int:
    import os
    import shlex

    profile = _profile(args)
    if not args.no_check:
        rc = _check_templates(profile)
        if rc:
            return rc
    argv = serve_argv(profile, host=args.host, port=args.port, extra=args.extra)
    env = serve_env(profile, record=args.record, record_dir=args.record_dir, window=args.window)
    unset = [k for k, v in profile.env.items() if v == "__must_be_unset__"]

    if not args.exec:
        for name in sorted(unset):
            print(f"unset {name}")
        for key, value in sorted(env.items()):
            print(f"export {key}={shlex.quote(str(value))}")
        print(shlex.join(argv))
        return 0
    from dattn.engine.launch import apply_env

    apply_env(profile)
    os.environ.update({str(k): str(v) for k, v in env.items()})
    # Use the same installed environment even when PATH selects another vLLM.
    argv[0] = str(Path(sys.executable).with_name("vllm"))
    print("+ " + shlex.join(argv), file=sys.stderr)
    os.execvp(argv[0], argv)  # noqa: S606 - the whole point of --exec


def _check_templates(profile: Any) -> int:
    """Record resolved chat-template provenance before launching the engine."""
    from dattn.engine.assertions import check_prompt_templates

    try:
        tok = _tokenizer(profile.model_id)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal on its own
        print(f"dattn serve: could not load the tokenizer for the A17 check: {exc}", file=sys.stderr)
        return 0
    report = check_prompt_templates(tok, profile)
    print(json.dumps(report["resolved"]), file=sys.stderr)
    for line in report["failures"]:
        print(f"dattn serve: {line}", file=sys.stderr)
    return 1 if report["failures"] else 0


def _cmd_probe(args: argparse.Namespace) -> int:
    """Boot the actual profile and print the physical pool and aliased group table."""
    from dataclasses import asdict

    profile = _profile(args)
    out: dict[str, Any] = {"profile": asdict(profile), "arm": args.arm}
    if args.no_boot:
        print(json.dumps(out, indent=2, default=str) if args.json else _plain(out))
        return 0

    from dattn.engine import launch

    launch.apply_env(profile)
    os.environ["DATTN_ARM"] = args.arm
    os.environ.pop("DATTN_POOL_RECORD", None)
    if args.pool_record:
        os.environ["DATTN_POOL_RECORD"] = str(args.pool_record.resolve())
    rc, llm = 0, None
    try:
        from vllm import LLM, SamplingParams

        overrides = {
            k: getattr(args, k)
            for k in ("max_model_len", "max_num_seqs", "gpu_memory_utilization")
            if getattr(args, k) is not None
        }
        if args.resolve_pool:
            overrides["kv_cache_memory_bytes"] = None
        kwargs = launch.engine_kwargs(profile, **overrides)
        if profile.use == "walltime":
            kwargs["compilation_config"] = dict(
                kwargs.get("compilation_config", {}),
                cudagraph_capture_sizes=list(range(1, int(kwargs["max_num_seqs"]) + 1)),
            )
        if args.arm == "vanilla":
            kwargs = launch.builder_free_kwargs(kwargs)
        llm = LLM(**kwargs)
        llm.generate(["ok"], SamplingParams(max_tokens=1, temperature=0.0))
        out["assertions"] = llm.collective_rpc("dattn_probe")[0]
        out["kv_pool"] = out["assertions"]["kv_pool"]
        out["templates"] = _template_report(llm, profile)
        out["failures"] = out["assertions"]["boot_assertions"]["failures"] + out["templates"]["failures"]
        rc = int(bool(out["failures"]))
        if not rc and args.pool_out:
            args.pool_out.parent.mkdir(parents=True, exist_ok=True)
            with args.pool_out.open("x") as f:
                json.dump(out["kv_pool"], f, indent=2)
    except Exception as exc:  # noqa: BLE001 - a probe reports failures, it does not raise
        out["error"] = f"{type(exc).__name__}: {exc}"
        rc = 1
    finally:
        del llm
    print(json.dumps(out, indent=2, default=str) if args.json else _plain(out))
    return rc


def _template_report(llm: Any, profile: Any) -> dict[str, Any]:
    from dattn.engine.assertions import check_prompt_templates

    try:
        return check_prompt_templates(llm.get_tokenizer(), profile)
    except Exception as exc:  # noqa: BLE001
        return {"resolved": {}, "failures": [], "skipped": {"A17": f"{type(exc).__name__}: {exc}"}}


def _plain(d: dict[str, Any]) -> str:
    return "\n".join(f"{k}: {json.dumps(v, default=str, indent=2)}" for k, v in d.items())


def _cmd_render(args: argparse.Namespace) -> int:
    from dattn.eval.prompts import build_prompt

    built = build_prompt(
        _tokenizer(args.model),
        context=args.context_file.read_text(),
        question=args.question,
        fmt=args.fmt,
        tool_schema=not args.no_tool_schema,
        segmenter=args.segmenter,
    )
    print(
        json.dumps(
            {
                "sha256": built.sha256,
                "n_tokens": built.n_prompt_tokens,
                "segmenter": args.segmenter,
                "n_semantic_segments": built.n_semantic_segments,
                "n_segments": built.n_segments,
                **built.layout.to_dict(),
            },
            indent=2,
        )
    )
    if args.show_text:
        print(built.text)
    return 0


def _cmd_replay(args: argparse.Namespace) -> int:
    from dattn.eval.replay import replay_row

    meta = json.loads(args.run_meta.read_text())
    if meta["rollout"]["model_id"] != args.model:
        raise ValueError("--model disagrees with rollout metadata")
    metrics = replay_row(
        {
            "context": args.context_file.read_text(),
            "question": args.question,
            "response": args.response_file.read_text(),
        },
        tokenizer=_tokenizer(args.model),
        run_meta=meta,
    )
    print(json.dumps({"n_steps": metrics["total_steps"], **metrics}, indent=2, default=str))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    if args.campaign_cell or args.campaign_shard:
        if not (args.campaign_cell and args.campaign_shard):
            raise ValueError("campaign cell and shard must be supplied together")
        from dattn.campaign.driver import eval_shard

        return eval_shard(args)
    from dattn.eval.datasets import HEADLINE_SUBSETS, SelectionSpec, derive_max_context_tokens, select_rows, write_rows
    from dattn.eval.runner import RunPaths, _check_selection, _prior_meta, run_rollouts
    from dattn.eval.sampling import RolloutConfig

    fmt, mask_on = ARMS[args.arm]
    subsets = tuple(args.subsets) if args.subsets else HEADLINE_SUBSETS
    max_ctx = derive_max_context_tokens(args.model)
    tok = _tokenizer(args.model)
    rows = select_rows(
        model_id=args.model,
        tokenizer=tok,
        subsets=subsets,
        n_per_subset=args.n_per_subset,
        seed=args.seed,
        max_context_tokens=max_ctx,
        num_proc=args.num_proc,
        progress=lambda m: print(m, file=sys.stderr),
    )
    spec = SelectionSpec(
        model_id=args.model,
        subsets=subsets,
        n_per_subset=args.n_per_subset,
        seed=args.seed,
        min_context_tokens=4096,
        max_context_tokens=max_ctx,
    )
    paths = RunPaths.for_cell(args.out, args.model, args.arm)
    paths.root.mkdir(parents=True, exist_ok=True)
    _check_selection(paths, _prior_meta(paths.meta), spec)
    write_rows(paths.rows, rows, spec=spec)
    print(f"selected {len(rows)} rows over {len(subsets)} subsets -> {paths.rows}", file=sys.stderr)

    if args.select_only:
        return 0

    n = run_rollouts(
        rows,
        tokenizer=tok,
        config=RolloutConfig(
            model_id=args.model,
            arm=args.arm,
            prompt_format=fmt,
            dattn_enabled=mask_on,
            max_tokens=args.max_tokens,
            template_policy=args.template_policy,
            segmenter=ARM_SEGMENTERS[args.arm],
        ),
        server=_server(args),
        paths=paths,
        spec=spec,
        last_answer=args.last_answer,
        record_replay_metrics=not args.no_replay_metrics,
        chunk_size=args.chunk_size,
        progress=lambda m: print(m, file=sys.stderr),
    )
    print(f"wrote {n} new rollout rows -> {paths.responses}", file=sys.stderr)
    return 0


def _cmd_judge(args: argparse.Namespace) -> int:
    from dattn.eval.judge import JudgeConfig
    from dattn.eval.runner import RunPaths, run_judge

    over: dict[str, Any] = {"chunk_size": args.chunk_size, "seed": args.seed}
    if args.judge_model:
        over["model_id"] = args.judge_model
    if args.judge_max_tokens:
        over["max_tokens"] = args.judge_max_tokens
    cfg = JudgeConfig(**over)
    n = run_judge(
        paths=RunPaths(args.dir),
        server=_server(args),
        config=cfg,
        judge_all=args.judge_all,
        progress=lambda m: print(m, file=sys.stderr),
    )
    print(f"wrote {n} new judgments -> {RunPaths(args.dir).judgments}", file=sys.stderr)
    return 0


def _cmd_parity(args: argparse.Namespace) -> int:
    from dattn.eval.parity import compare_trees

    try:
        report = compare_trees(args.left, args.right, mode=args.mode)
    except ValueError as exc:
        print(f"dattn parity: {exc}", file=sys.stderr)
        return 2
    print(report.render())
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return 0 if report.passed else 1


def _cmd_bench(args: argparse.Namespace) -> int:
    from dattn.bench.driver import run_protocol
    from dattn.bench.protocol import Protocol

    profile = _profile(args)
    protocol = Protocol(model_id=profile.model_id, use=profile.use, **profile.bench)
    result = run_protocol(
        protocol,
        args.out,
        profile=profile,
        subsets=args.subsets,
        rows_file=args.rows,
        arm=args.arm,
        task_index=args.task_index,
        batch_index=args.batch_index,
        pins=args.pins,
    )
    print(json.dumps({k: v for k, v in result.items() if k != "cells"}, indent=2))
    return 0


def _cmd_analyse_bench(args: argparse.Namespace) -> int:
    """The bench macro-average: per-task bins first, then the average across tasks."""
    from dattn.bench.aggregate import macro_report

    try:
        if args.pair:
            from dattn.bench.pairing import pair_runs

            if len(args.dirs) != 2:
                raise ValueError("--pair requires exactly two arm roots")
            pair_runs(*args.dirs, out_dir=args.pair_out)
            args.dirs = [args.pair_out]
        report = macro_report(
            args.dirs,
            allow_partial=args.allow_partial,
            plot=args.plot,
        )
    except ValueError as exc:
        print(f"dattn analyse --bench: {exc}", file=sys.stderr)
        return 1
    body = {k: v for k, v in report.items() if k not in ("cells", "per_task_bins")}
    print(json.dumps(body, indent=2, default=str))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return 0


def _cmd_analyse(args: argparse.Namespace) -> int:
    if args.bench:
        return _cmd_analyse_bench(args)
    from dattn.eval.aggregate import aggregate, format_table, load_cell
    from dattn.eval.datasets import resolve_subsets
    from dattn.eval.runner import RunPaths
    from dattn.eval.schema import json_safe

    cells, selections = [], []
    for d in args.dirs:
        paths = RunPaths(d)
        cell = load_cell(paths.responses, paths.judgments, model=d.parent.name, arm=d.name)
        sel = resolve_subsets(cell.subsets(), all_subsets=args.all_subsets, explicit=args.subsets)
        for warning in sel.warnings:
            print(f"dattn analyse: {d}: {warning}", file=sys.stderr)
        cells.append(aggregate(cell, subsets=sel.subsets))
        selections.append({"dir": str(d), "restricted": sel.restricted, "dropped": list(sel.dropped)})
    print(format_table(cells))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(
                json_safe(
                    {
                        "cells": [c.to_dict() for c in cells],
                        "subsets": selections,
                    }
                ),
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
    return 0


_HANDLERS: Final[dict[str, Any]] = {
    "serve": _cmd_serve,
    "probe": _cmd_probe,
    "render": _cmd_render,
    "replay": _cmd_replay,
    "eval": _cmd_eval,
    "judge": _cmd_judge,
    "parity": _cmd_parity,
    "bench": _cmd_bench,
    "analyse": _cmd_analyse,
}


def main(argv: Sequence[str] | None = None) -> int:
    from dattn import __version__

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command is None:
        parser.print_help()
        return 2
    return int(_HANDLERS[args.command](args))


def _prefix_validation_worker(args):
    """Explicit CLI boundary between CPU campaign orchestration and a GPU worker."""
    from dattn.bench.prefix_validation import worker

    return worker(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
