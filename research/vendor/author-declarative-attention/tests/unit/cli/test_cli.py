"""The CLI surface: the sub-commands, profile selection, and the stub-tolerant paths.

`dattn --help` and every parser path must work with no vllm, no transformers and no
datasets import, so these tests are also the regression test for the lazy-import rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fakes import FakeTokenizer, _FakeProfile

from dattn import cli

from .conftest import judgment_row, response_row, write_cell


def test_the_command_list_covers_the_design_and_the_two_eval_phases() -> None:
    assert cli.COMMANDS == ("serve", "probe", "render", "replay", "eval", "judge", "parity", "bench", "analyse")
    parser = cli.build_parser()
    for name in cli.COMMANDS:
        assert parser.format_help().count(name) >= 1


def test_version_exits_clean(capsys: pytest.CaptureFixture[str]) -> None:
    from dattn import __version__

    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == __version__


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 2
    assert "probe" in capsys.readouterr().out


def test_the_arms_map_a_name_to_a_format_and_a_mask_switch() -> None:
    """DA and DA-nm differ ONLY in the mask: the two dac templates are byte identical."""
    assert cli.ARMS["da"] == ("chunk_multi_turn_tool", True)
    assert cli.ARMS["da_nm"] == ("chunk_multi_turn_tool", False)
    assert cli.ARMS["vanilla"] == ("vanilla", False)
    assert cli.ARMS["da"][0] == cli.ARMS["da_nm"][0]


def test_building_the_parser_imports_nothing_heavy() -> None:
    heavy = {"vllm", "torch", "transformers", "datasets"}
    before = heavy & set(sys.modules)
    cli.build_parser()
    assert (heavy & set(sys.modules)) == before


@pytest.mark.parametrize("cmd", ["serve", "probe", "eval", "bench"])
def test_engine_commands_require_a_model(cmd: str) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([cmd])


def test_parity_parses_its_two_directories() -> None:
    args = cli.build_parser().parse_args(["parity", "--left", "a", "--right", "b", "--mode", "mask"])
    assert args.command == "parity"
    assert args.left == Path("a")
    assert args.mode == "mask"


def test_eval_defaults_to_the_published_configuration() -> None:
    args = cli.build_parser().parse_args(["eval", "--model", "Qwen/Qwen3.6-27B"])
    assert args.n_per_subset == 128
    assert args.seed == 42
    assert args.max_tokens == 8192
    assert args.template_policy == "dac_bundled"
    assert args.arm == "da"
    assert args.profile == "parity"
    assert args.capability == "sm100"
    # The FIRST <answer> block is the default, which is the paper's measurement.
    assert args.last_answer is False
    assert args.first_answer is False  # a hidden no-op alias for what is now the default
    assert args.no_replay_metrics is False  # the trace is recorded on every row by default


def test_the_answer_policy_flags_reach_the_namespace_and_exclude_each_other() -> None:
    """`--last-answer` parsed and went nowhere before the integrator wired it (FIX_WPA H2)."""
    parser = cli.build_parser()
    assert parser.parse_args(["eval", "--model", "m", "--last-answer"]).last_answer is True
    assert parser.parse_args(["eval", "--model", "m", "--first-answer"]).first_answer is True
    # Asking for both is a contradiction, not a precedence question.
    with pytest.raises(SystemExit):
        parser.parse_args(["eval", "--model", "m", "--last-answer", "--first-answer"])


def test_eval_forwards_the_answer_policy_and_the_replay_switch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The two keywords `_cmd_eval` must hand `run_rollouts`, pinned at the call site.

    `--last-answer` is FIX_WPA handoff H2 and `--no-replay-metrics` is FIX_WPB's escape
    hatch, which wpC correctly refused to add from outside its cli.py ownership.
    """
    import dattn.eval.runner as runner_mod

    seen: dict[str, object] = {}

    def fake_run_rollouts(_rows: object, **kw: object) -> int:
        seen.update(kw)
        return 0

    monkeypatch.setattr(runner_mod, "run_rollouts", fake_run_rollouts)
    monkeypatch.setattr(cli, "_tokenizer", lambda _m: object())
    monkeypatch.setattr(cli, "_server", lambda _a: object())

    import dattn.eval.datasets as ds_mod

    monkeypatch.setattr(ds_mod, "derive_max_context_tokens", lambda _m: 100_000)
    monkeypatch.setattr(ds_mod, "select_rows", lambda **_kw: [])
    monkeypatch.setattr(ds_mod, "write_rows", lambda *_a, **_kw: None)

    rc = cli.main(
        [
            "eval",
            "--model",
            "m",
            "--last-answer",
            "--no-replay-metrics",
            "--template-policy",
            "stock",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    assert seen["config"].template_policy == "stock"
    assert seen["last_answer"] is True
    assert seen["record_replay_metrics"] is False
    # Never from a profile: only a recorder-armed run may fill `attended_tokens_blocks`.
    assert seen.get("dattn_block_size", 0) == 0


def test_serve_prints_the_command_for_a_real_profile(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli.main(["serve", "--model", "google/gemma-4-31B-it", "--port", "8123", "--no-check"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith("vllm serve google/gemma-4-31B-it")
    assert "--attention-backend TRITON_ATTN" in out
    assert "--port 8123" in out
    assert "dattn.engine.processor:DattnLogitsProcessor" in out


def test_serve_exports_the_three_variables_the_plugin_reads(capsys: pytest.CaptureFixture[str]) -> None:
    """Without DATTN_MODEL the general plugin is inert and the server boots vanilla.

    This is the failure mode that has no symptom: `--attention-config` still parses,
    `CUSTOM` is simply never registered, and the run produces numbers.
    """
    assert cli.main(["serve", "--model", "Qwen/Qwen3.6-27B", "--no-check"]) == 0
    out = capsys.readouterr().out
    assert "export DATTN_MODEL=Qwen/Qwen3.6-27B" in out
    assert "export DATTN_CAPABILITY=sm100" in out
    assert "export DATTN_USE=parity" in out
    assert "export VLLM_USE_V2_MODEL_RUNNER=0" in out
    assert "unset VLLM_ATTENTION_BACKEND" in out
    assert "DATTN_RECORD" not in out, "the recorder is off unless --record asks for it"


def test_serve_record_arms_the_worker_side_recorder(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    argv = ["serve", "--model", "Qwen/Qwen3.6-27B", "--no-check", "--record"]
    rc = cli.main([*argv, "--record-dir", str(tmp_path), "--window", "w1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "export DATTN_RECORD=1" in out
    assert "export DATTN_RECORD_WINDOW=w1" in out
    assert f"export DATTN_RECORD_DIR={tmp_path}" in out


def test_serve_records_the_current_chat_template(monkeypatch: pytest.MonkeyPatch) -> None:
    """Template provenance is recorded without a profile hash gate."""

    monkeypatch.setattr(
        cli, "_tokenizer", lambda model_id: FakeTokenizer(chat_template="this is not the pinned template")
    )
    assert cli.main(["serve", "--model", "Qwen/Qwen3.6-27B"]) == 0


def test_serve_refuses_a_model_with_no_profile() -> None:
    from dattn.contracts import DattnBootError

    with pytest.raises(DattnBootError):
        cli.main(["serve", "--model", "nobody/nothing"])


def test_parity_returns_1_when_it_fails(tmp_path: Path) -> None:
    left = write_cell(tmp_path / "l", [response_row("s", 0, question="a")])
    right = write_cell(tmp_path / "r", [response_row("s", 0, question="b")])
    assert cli.main(["parity", "--left", str(left), "--right", str(right)]) == 1


def test_analyse_prints_macro_and_micro_together(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cell = write_cell(
        tmp_path / "m" / "da",
        [response_row("a", i) for i in range(4)] + [response_row("b", 0)],
        [judgment_row("a", i, i < 3) for i in range(4)] + [judgment_row("b", 0, False)],
    )
    rc = cli.main(["analyse", "--dirs", str(cell)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "macro" in out and "micro" in out
    assert "fmt_fail" in out and "unpar" in out


def _sixteen_subset_cell(tmp_path: Path, arm: str = "da") -> Path:
    """The shape of dac's DA-nm backup arm: the 15 headline subsets plus the stale one."""
    from dattn.eval.datasets import HEADLINE_SUBSETS

    rows = [response_row(s, 0, metrics={"total_steps": 10, "attended_tokens": 7}) for s in HEADLINE_SUBSETS]
    rows.append(response_row("longbench_synthetic_v4_1_loogle_shortdep_qa", 0))
    js = [judgment_row(r["subset"], 0, True) for r in rows]
    return write_cell(tmp_path / "m" / arm, rows, js)


def test_analyse_defaults_to_the_fifteen_headline_subsets(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--headline-subsets` used to be opt-in, so a 16-subset arm silently got a 16 macro."""
    cell = _sixteen_subset_cell(tmp_path)
    assert cli.main(["analyse", "--dirs", str(cell)]) == 0
    cap = capsys.readouterr()
    assert "dropped 1 subset(s) outside them" in cap.err
    assert "loogle_shortdep_qa" in cap.err
    assert "WARNING: stale subset" in cap.err
    # 15 rows counted, not 16.
    assert [line for line in cap.out.splitlines() if "/da" in line or " da " in line]
    assert " 15 " in cap.out.split("\n")[2]


def test_analyse_with_all_subsets_keeps_the_sixteenth(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cell = _sixteen_subset_cell(tmp_path)
    assert cli.main(["analyse", "--dirs", str(cell), "--all-subsets"]) == 0
    cap = capsys.readouterr()
    assert "dropped" not in cap.err
    assert "WARNING: stale subset" in cap.err  # still named, just not dropped
    assert " 16 " in cap.out.split("\n")[2]


def test_analyse_on_a_cell_missing_headline_subsets_warns_and_does_not_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cell = write_cell(tmp_path / "m" / "da", [response_row("a", i) for i in range(4)])
    assert cli.main(["analyse", "--dirs", str(cell)]) == 0
    cap = capsys.readouterr()
    assert "NOT the paper's population" in cap.err
    assert " 4 " in cap.out.split("\n")[2]


# -- bench: the task matrix and the bench macro-average (WP7 items 7, 12, 13) -----------


# --------------------------------------------------------------------------------------
# P12: serve_argv and the boolean that has no negated form
# --------------------------------------------------------------------------------------


def test_a_false_boolean_with_no_negated_form_is_omitted_not_negated() -> None:
    """The regression that stopped every `dattn serve` for twenty minutes tonight.

    `--disable-log-stats` is `store_true` in vLLM 0.29.0, so `--no-disable-log-stats` is
    `error: unrecognized arguments` and the server never boots. `6596494` added
    `disable_log_stats = false` to `[common]` and `serve_argv` rendered exactly that. The
    in-process GPU tests missed it because they go through `LLM(**kwargs)`, where it is an
    ordinary Python keyword and never sees a command line.
    """
    argv = cli.serve_argv(_FakeProfile({"disable_log_stats": False}), host="h", port=1)
    assert "--no-disable-log-stats" not in argv
    assert "--disable-log-stats" not in argv, "vLLM's default is already False; saying nothing says it"


def test_a_false_boolean_that_does_have_a_negated_form_is_still_negated() -> None:
    """The other 58 booleans are `BooleanOptionalAction`, and omitting those WOULD change
    the engine: `enable_prefix_caching` defaults to on, and A11 refuses a boot with it."""
    argv = cli.serve_argv(_FakeProfile({"enable_prefix_caching": False}), host="h", port=1)
    assert "--no-enable-prefix-caching" in argv


def test_a_true_boolean_is_the_bare_flag_under_either_argparse_style() -> None:
    argv = cli.serve_argv(_FakeProfile({"disable_log_stats": True, "disable_cascade_attn": True}), host="h", port=1)
    assert "--disable-log-stats" in argv
    assert "--disable-cascade-attn" in argv
    assert not [a for a in argv if a.startswith("--no-")]


def test_non_boolean_engine_args_are_unchanged() -> None:
    argv = cli.serve_argv(
        _FakeProfile({"max_model_len": 40960, "logits_processors": ["a", "b"]}), host="0.0.0.0", port=8000
    )
    assert argv[:3] == ["vllm", "serve", "m"]
    assert argv[argv.index("--max-model-len") + 1] == "40960"
    assert argv[argv.index("--logits-processors") + 1 : argv.index("--logits-processors") + 3] == ["a", "b"]
    assert argv[-4:] == ["--host", "0.0.0.0", "--port", "8000"]


def test_the_store_true_set_is_read_from_vllm_and_falls_back_to_the_measured_one() -> None:
    """A hand-kept list rots. The real parser is ground truth and cannot disagree with the
    binary that is about to parse the line, so it is preferred and the constant is the
    fallback for a box with no vllm."""
    names = cli._no_negated_form()
    assert "disable-log-stats" in names
    assert "enable-prefix-caching" not in names, "that one DOES take --no-"
    # Every name on the no-negated-form list must be a real vLLM flag, or the list is
    # silently protecting a flag that does not exist. The `or ==` half of this was
    # redundant: set equality already implies the subset.
    assert names >= cli.NO_NEGATED_FORM, f"not real vLLM flags: {cli.NO_NEGATED_FORM - names}"


@pytest.mark.parametrize("model_id", ["google/gemma-4-31B-it", "Qwen/Qwen3.6-27B"])
@pytest.mark.parametrize("use", ["parity", "walltime"])
def test_every_shipping_profile_renders_flags_vllm_will_accept(model_id: str, use: str) -> None:
    """The end-to-end property, checked against vLLM's OWN parser rather than a list.

    Nothing in the tree covered `serve_argv` at all, which is why a flag that no vLLM
    accepts shipped and was found on hardware.
    """
    argparse_mod = pytest.importorskip("argparse")
    engine_args = pytest.importorskip("vllm.engine.arg_utils")
    from dattn.engine.profile import load_profile

    parser = engine_args.EngineArgs.add_cli_args(argparse_mod.ArgumentParser())
    parser.add_argument("--host")
    parser.add_argument("--port")
    from dataclasses import replace

    profile = load_profile(model_id, "sm100", use)
    if use == "walltime":
        profile = replace(profile, engine_args=dict(profile.engine_args, kv_cache_memory_bytes=100 << 30))
    argv = cli.serve_argv(profile, host="0.0.0.0", port=8000)
    parser.parse_args(argv[3:])  # drop "vllm serve <model>"


def test_bench_uses_walltime_profile_and_shared_draw_pins():
    args = cli.build_parser().parse_args(
        ["bench", "--model", "m", "--out", "o", "--pins", "p", "--arm", "da", "--task-index", "2"]
    )
    assert args.profile == "walltime" and args.task_index == 2 and args.pins == Path("p")
    assert not hasattr(args, "concurrency_map") and not hasattr(args, "env_period")


def test_mask_parity_cli_writes_report(tmp_path, capsys):
    root = write_cell(tmp_path / "cell", [response_row("s", 0, prompt_sha256="abc")])
    out = tmp_path / "report.json"
    assert cli.main(["parity", "--left", str(root), "--right", str(root), "--json-out", str(out)]) == 0
    assert json.loads(out.read_text())["passed"]
    assert "PASS" in capsys.readouterr().out


def test_probe_prints_and_records_the_actual_pool(monkeypatch, tmp_path, capsys):
    import vllm

    from dattn.engine import launch

    pool = {
        "resolved_kv_cache_memory_bytes": 2050,
        "pool_bytes": 2000,
        "physical_bytes_per_block": 200,
        "num_gpu_blocks": 10,
        "null_block_reservation": 200,
        "rounding_remainder": 50,
        "groups": [{"logical_full_capacity_bytes": 2000}, {"logical_full_capacity_bytes": 1500}],
    }
    seen = {}

    class ProbeEngine:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def generate(self, *args, **kwargs):
            return []

        def get_tokenizer(self):
            from fakes import FakeTokenizer

            return FakeTokenizer()

        def collective_rpc(self, *args, **kwargs):
            return [{"kv_pool": pool, "boot_assertions": {"failures": []}}]

    monkeypatch.setattr(vllm, "LLM", ProbeEngine)
    monkeypatch.setattr(launch, "apply_env", lambda p: {})
    monkeypatch.setenv("DATTN_POOL_RECORD", "")
    path = tmp_path / "pool.json"
    rc = cli.main(
        [
            "probe",
            "--model",
            "google/gemma-4-31B-it",
            "--profile",
            "walltime",
            "--arm",
            "vanilla",
            "--resolve-pool",
            "--pool-out",
            str(path),
            "--json",
        ]
    )
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["kv_pool"] == json.loads(path.read_text()) == pool
    assert seen["kv_cache_memory_bytes"] is None and "logits_processors" not in seen
    assert seen["compilation_config"]["cudagraph_capture_sizes"] == list(range(1, 65))
