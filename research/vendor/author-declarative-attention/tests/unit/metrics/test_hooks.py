"""The toggle: install, uninstall, and the proof that disabled costs nothing (WP6).

"Disabled is zero overhead, not cheap overhead" is a testable claim and it is tested
here in two halves, because the disabled path has exactly two parts:

1. the rebinding. After `uninstall()` the instance attribute is gone, so
   `runner._model_forward` resolves through the class descriptor exactly as it did
   before `install()`. There is no wrapper frame, no closure cell and no try/finally.
   `test_uninstall_leaves_no_trace` asserts that on the object, not in prose.
2. the gate. Every other dattn call site is `if _REC is not None: _REC.note_*(...)`.
   `test_the_disabled_gate_allocates_nothing` runs that idiom at two very different
   iteration counts under `tracemalloc` and asserts the two totals are equal, which is
   zero bytes per gate, and `test_the_disabled_gate_is_one_line_of_python_per_call`
   traces it and asserts exactly one executed line per call: the branch itself.
"""

from __future__ import annotations

import inspect
import json
import os
import signal
import subprocess
import sys
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pytest
from fakes import FakeRunner, FakeTorch, decode_step

import dattn.engine.api as engine_api
from dattn.metrics import hooks
from dattn.metrics.recorder import StepRecorder

# --------------------------------------------------------------------------------------
# The rebinding
# --------------------------------------------------------------------------------------


def test_install_wraps_exactly_the_documented_override_point(runner: FakeRunner, fake_torch: type[FakeTorch]) -> None:
    rec = hooks.install(runner, window_id="w0", torch_mod=fake_torch, ring=8)
    assert isinstance(rec, StepRecorder)
    assert hooks.installed(runner) is rec
    assert runner._model_forward is not FakeRunner._model_forward
    assert runner._model_forward() == "logits"
    assert runner._sample() == "tokens"
    assert runner.forwards == 1 and runner.samples == 1
    # The `_sample` wrapper closes the step itself: that is the Lane C half of interface
    # request C4, and without it nothing on the real path ever calls `end_step`, so every
    # step of a run would be merged into one open record.
    (out,) = rec.drain()
    assert out.gpu_forward_ms > 0 and out.gpu_sample_ms > 0
    assert out.t_ready_ns > 0


def test_the_sample_wrapper_closes_one_step_per_call(runner: FakeRunner, fake_torch: type[FakeTorch]) -> None:
    rec = hooks.install(runner, window_id="w0", torch_mod=fake_torch, ring=16)
    for _ in range(5):
        runner._model_forward()
        runner._sample()
    assert rec.step_id == 5, "one record per engine iteration, not one per window"
    assert len(rec.drain()) == 5


def test_the_step_facts_come_from_the_scheduler_output(fake_torch: type[FakeTorch]) -> None:
    """C4: the wrapper is the only scope that sees the `SchedulerOutput` and the graph."""
    so = decode_step(4)
    so.preempted_req_ids = {"r3"}
    runner = FakeRunner(scheduler_output=so)
    rec = hooks.install(runner, window_id="w0", torch_mod=fake_torch, ring=8)
    runner._model_forward()
    runner._sample()
    (out,) = rec.drain()
    assert out.kind == "decode"
    assert out.num_reqs == 4
    assert out.num_reqs_padded == 4
    assert out.num_gen_tokens == 4 and out.num_ctx_tokens == 0
    # Cumulative, so `driver.preemptions` can difference it (DESIGN.md section 5.2).
    assert out.num_preempted == 1
    # No forward context outside a real forward, so the mode is empty and `replayed` is
    # False. That is honest: gate S1 reports "unknown mode" rather than inventing one.
    assert out.cudagraph_mode == "" and out.cudagraph_replayed is False


def test_the_facts_survive_sample_tokens_clearing_the_cached_scheduler_output(
    fake_torch: type[FakeTorch],
) -> None:
    """The bug this guards: `execute_model_state` is None by the time `_sample` runs.

    `sample_tokens` unpacks the cache and sets the slot to None before it calls `_sample`
    (`N:v1/worker/gpu_model_runner.py:4653-4655`), so a wrapper that reads it from the
    runner gets nothing and every record keeps its default `kind`, `num_reqs` and
    `cudagraph_mode`. Reading the argument of `execute_model` is the fix, and this is the
    test that fails without it.
    """
    runner = FakeRunner(scheduler_output=decode_step(3))
    rec = hooks.install(runner, window_id="w0", torch_mod=fake_torch, ring=8)
    runner.execute_model(runner.execute_model_state.scheduler_output)
    assert runner.execute_model_state is None, "the fake must reproduce the clearing"
    runner._model_forward()
    runner._sample()
    (out,) = rec.drain()
    assert out.kind == "decode" and out.num_reqs == 3
    hooks.uninstall(runner)
    assert "execute_model" not in vars(runner)


def test_install_from_env_is_a_no_op_unless_the_toggle_is_set(
    runner: FakeRunner, fake_torch: type[FakeTorch], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(hooks.ENV_RECORD, raising=False)
    assert hooks.install_from_env(runner, torch_mod=fake_torch) is None
    assert hooks.installed(runner) is None
    monkeypatch.setenv(hooks.ENV_RECORD, "0")
    assert hooks.install_from_env(runner, torch_mod=fake_torch) is None
    monkeypatch.setenv(hooks.ENV_RECORD, "1")
    rec = hooks.install_from_env(runner, torch_mod=fake_torch, ring=4)
    assert isinstance(rec, StepRecorder)
    assert hooks.install_from_env(runner, torch_mod=fake_torch) is rec, "idempotent per worker"
    hooks.uninstall(runner)


def test_install_from_env_writes_where_the_toggle_says(
    runner: FakeRunner, fake_torch: type[FakeTorch], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(hooks.ENV_RECORD, "1")
    monkeypatch.setenv(hooks.ENV_RECORD_DIR, str(tmp_path / "recs"))
    monkeypatch.setenv("DATTN_RECORD_WINDOW", "cell0")
    rec = hooks.install_from_env(runner, torch_mod=fake_torch, ring=4)
    assert rec is not None
    runner._model_forward()
    runner._sample()
    hooks.uninstall(runner)
    out = tmp_path / "recs" / "steps_cell0.jsonl"
    assert out.is_file() and out.read_text().count("\n") == 1


def test_uninstall_leaves_no_trace(runner: FakeRunner, fake_torch: type[FakeTorch]) -> None:
    """The disabled path is byte-identical to stock vLLM: the attribute is gone."""
    assert "_model_forward" not in vars(runner)
    hooks.install(runner, window_id="w0", torch_mod=fake_torch, ring=4)
    assert "_model_forward" in vars(runner)
    hooks.uninstall(runner)
    assert "_model_forward" not in vars(runner), "an instance attribute would still cost a dict lookup"
    assert "_sample" not in vars(runner)
    assert runner._model_forward.__func__ is FakeRunner._model_forward
    assert hooks.installed(runner) is None


def test_calling_the_runner_after_uninstall_records_nothing(runner: FakeRunner, fake_torch: type[FakeTorch]) -> None:
    rec = hooks.install(runner, window_id="w0", torch_mod=fake_torch, ring=4)
    hooks.uninstall(runner)
    before = rec.step_id
    for _ in range(100):
        runner._model_forward()
    assert rec.step_id == before
    assert runner.forwards == 100


def test_install_refuses_an_object_that_is_not_the_0_29_0_runner() -> None:
    with pytest.raises(NotImplementedError, match="metrics.install needs"):
        hooks.install(object(), window_id="w0")


def test_install_refuses_a_second_install(runner: FakeRunner, fake_torch: type[FakeTorch]) -> None:
    hooks.install(runner, window_id="w0", torch_mod=fake_torch, ring=4)
    with pytest.raises(RuntimeError, match="already installed"):
        hooks.install(runner, window_id="w1", torch_mod=fake_torch, ring=4)
    hooks.uninstall(runner)


def test_install_refuses_a_runner_whose_batch_size_it_cannot_resolve(fake_torch: type[FakeTorch]) -> None:
    class Bare:
        def _model_forward(self) -> None: ...

    with pytest.raises(NotImplementedError, match="max_num_seqs"):
        hooks.install(Bare(), window_id="w0", torch_mod=fake_torch)


def test_uninstall_on_a_clean_runner_is_a_no_op(runner: FakeRunner) -> None:
    assert hooks.uninstall(runner) is None


def test_install_resolves_the_runner_facts_it_records(fake_torch: type[FakeTorch]) -> None:
    class AsyncRunner(FakeRunner):
        async_scheduling = True
        use_v2_model_runner = True

    r = AsyncRunner()
    rec = hooks.install(r, window_id="w0", torch_mod=fake_torch, ring=4)
    assert rec.host_bracket_valid is False
    assert rec.fsm_lag == 1, "the measured lag is 1 in both scheduling modes (gpu_unknowns section 1)"
    assert rec.runner_generation == "V2"
    hooks.uninstall(r)


def test_install_reads_max_num_seqs_off_the_vllm_config_when_the_runner_hides_it(
    fake_torch: type[FakeTorch],
) -> None:
    class Cfg:
        class scheduler_config:  # noqa: N801 -- mirrors vLLM's attribute name
            max_num_seqs = 48

    class Runner2:
        vllm_config = Cfg()

        def _model_forward(self) -> None: ...

        # `install` REQUIRES a callable `_sample`: it is the only caller of `end_step`,
        # and a runner without one records no step at all.
        def _sample(self) -> None: ...

    r = Runner2()
    rec = hooks.install(r, window_id="w0", torch_mod=fake_torch, ring=2)
    assert rec.max_num_seqs == 48
    hooks.uninstall(r)


def test_the_recorder_writes_its_window_to_disk_on_uninstall(
    runner: FakeRunner, fake_torch: type[FakeTorch], tmp_path: Path
) -> None:
    p = tmp_path / "steps_w0.jsonl"
    rec = hooks.install(runner, window_id="w0", torch_mod=fake_torch, ring=4, out_path=str(p))
    runner._model_forward()
    rec.end_step()
    hooks.uninstall(runner)
    assert p.read_text().count("\n") == 1


# --------------------------------------------------------------------------------------
# Publishing the `_REC` slot
# --------------------------------------------------------------------------------------


def test_publish_is_a_no_op_while_lane_b_has_no_setter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(engine_api, "set_step_recorder", raising=False)
    assert hooks.publish(None) is False


def test_publish_reaches_lane_b_the_moment_the_setter_exists(
    monkeypatch: pytest.MonkeyPatch, runner: FakeRunner, fake_torch: type[FakeTorch]
) -> None:
    seen: list[Any] = []
    monkeypatch.setattr(engine_api, "set_step_recorder", seen.append, raising=False)
    rec = hooks.install(runner, window_id="w0", torch_mod=fake_torch, ring=2)
    assert rec.publish_ok is True
    assert seen == [rec]
    hooks.uninstall(runner)
    assert seen == [rec, None], "uninstall must clear the slot, not leave a dangling recorder"


# --------------------------------------------------------------------------------------
# The disabled gate: zero allocation, one branch
# --------------------------------------------------------------------------------------

_GATE = compile(
    "for _ in range(N):\n" + "    " + hooks.GATE_SOURCE.replace("\n", "\n    ").rstrip() + "\n",
    "<gate>",
    "exec",
)


def test_the_gate_source_is_the_idiom_lane_b_actually_writes() -> None:
    import dattn.engine.state as state

    src = Path(state.__file__).read_text(encoding="utf-8")
    assert "_REC" in src, "the coupling is the one module-level slot, and Lane B owns it"
    assert hasattr(state, "set_recorder"), "hooks writes the slot, dattn.engine.state reads it"
    assert "_REC is not None" in hooks.GATE_SOURCE


def _gate_bytes(n: int) -> int:
    """Bytes allocated inside `<gate>` while running the disabled gate `n` times."""
    g: dict[str, Any] = {"_REC": None, "N": n, "range": range}
    exec(_GATE, g)  # noqa: S102 -- warm the code object and any interned constants
    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    exec(_GATE, g)  # noqa: S102 -- the measured run
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()
    return sum(s.size_diff for s in after.compare_to(before, "filename") if "<gate>" in str(s))


def test_the_disabled_gate_allocates_nothing() -> None:
    """The disabled path allocates a constant, independent of how many steps run.

    The loop's own `range` iterator is one object whatever `n` is, so measuring at two
    very different `n` and asserting the two totals are equal is the exact claim: zero
    bytes per gate. A wrapper object, a closure cell, a bound method or a formatted
    message on the disabled path would all make this grow with `n`.
    """
    small = _gate_bytes(1_000)
    large = _gate_bytes(200_000)
    assert large == small, f"the disabled path allocated {large - small} bytes over 199 000 extra gates"
    assert small <= 64, f"even the one-off loop setup should be tiny, got {small} bytes"


def test_the_disabled_gate_is_one_line_of_python_per_call() -> None:
    """Trace the gate: exactly one line executes per iteration, the `is not None` test."""
    lines: list[int] = []

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if event == "line" and frame.f_code.co_filename == "<gate>":
            lines.append(frame.f_lineno)
        return tracer

    g: dict[str, Any] = {"_REC": None, "N": 50, "range": range}
    sys.settrace(tracer)
    try:
        exec(_GATE, g)  # noqa: S102 -- the traced run
    finally:
        sys.settrace(None)
    # one `for` line plus one gate line per iteration, and never the body
    body = [n for n in lines if n == 3]
    assert body == [], "the call inside the gate must never execute when disabled"
    assert lines.count(2) == 50, f"expected exactly one gate line per iteration, got {lines.count(2)}"


def test_the_enabled_gate_does_call_through() -> None:
    """The mirror image: with a recorder in the slot, the body runs. Otherwise the
    zero-allocation test above would pass on a gate that is simply dead code."""
    calls: list[Any] = []

    class Rec:
        def note_lp(self, host_us: float, rows: Any) -> None:
            calls.append(host_us)

    g: dict[str, Any] = {"_REC": Rec(), "N": 10, "range": range}
    exec(_GATE, g)  # noqa: S102
    assert len(calls) == 10


# --------------------------------------------------------------------------------------
# The step path holds no drain
# --------------------------------------------------------------------------------------


def test_no_synchronize_happens_between_arm_and_disarm(
    fake_torch: type[FakeTorch], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A recorder with a writer must not drain mid-run, at any step count.

    `dattn serve --record` defaults `DATTN_RECORD_DIR`, so every served run has a writer,
    and the periodic flush that used to sit in the `_sample` closure fired every 512
    steps: one `cuda.synchronize()` and one whole-ring device-to-host copy inside a timed
    step. That is an instrument perturbing what it measures, and it silently contaminated
    any timing taken through the serve path. The step counts here straddle the old
    cadence deliberately.
    """
    monkeypatch.setenv(hooks.ENV_RECORD, "1")
    monkeypatch.setenv(hooks.ENV_RECORD_DIR, str(tmp_path))
    monkeypatch.setenv("DATTN_RECORD_WINDOW", "serve")
    runner = FakeRunner(decode_step(4))
    rec = hooks.install_from_env(runner, torch_mod=fake_torch, ring=2048)
    assert rec is not None
    for _ in range(1100):
        runner.execute_model(decode_step(4))
        runner._model_forward()
        runner._sample()
    assert rec.step_id == 1100
    assert fake_torch.cuda.syncs == 0, "a synchronize fired inside the armed window"
    assert fake_torch.D2H == 0, "the audit ring was copied to the host inside the window"
    assert rec.pending == 1100, "the ring was drained mid-run"
    assert (tmp_path / "steps_serve.jsonl").read_text() == "", "the writer was fed mid-run"

    hooks.uninstall(runner)
    assert fake_torch.cuda.syncs == 1, "disarm is the one drain, and it is off the step path"
    assert (tmp_path / "steps_serve.jsonl").read_text().count("\n") == 1100


def test_the_bench_path_stays_writer_less_and_drains_in_process(
    runner: FakeRunner, fake_torch: type[FakeTorch], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`DATTN_RECORD` with no `DATTN_RECORD_DIR` is the bench arm: no file, no drain."""
    monkeypatch.setenv(hooks.ENV_RECORD, "1")
    monkeypatch.delenv(hooks.ENV_RECORD_DIR, raising=False)
    rec = hooks.install_from_env(runner, torch_mod=fake_torch, ring=64)
    assert rec is not None
    for _ in range(40):
        runner._model_forward()
        runner._sample()
    assert fake_torch.cuda.syncs == 0
    assert hooks.flush_now(runner) == 0, "no writer, so nothing to serialise to"
    assert rec.pending == 40, "flush_now must not empty a writer-less ring either"
    assert len(rec.drain()) == 40
    hooks.uninstall(runner)


def test_flush_now_is_the_only_sanctioned_mid_run_drain(
    runner: FakeRunner, fake_torch: type[FakeTorch], tmp_path: Path
) -> None:
    p = tmp_path / "steps_w0.jsonl"
    rec = hooks.install(runner, window_id="w0", torch_mod=fake_torch, ring=64, out_path=str(p))
    for _ in range(3):
        runner._model_forward()
        runner._sample()
    assert fake_torch.cuda.syncs == 0
    assert hooks.flush_now(runner) == 3
    assert fake_torch.cuda.syncs == 1, "an explicit request pays for its own sync, off the step path"
    assert p.read_text().count("\n") == 3
    assert rec.pending == 0
    hooks.uninstall(runner)


def test_flush_now_on_an_unarmed_runner_is_a_no_op(runner: FakeRunner) -> None:
    assert hooks.flush_now(runner) == 0


def test_the_periodic_in_step_flush_cadence_is_gone_from_the_module() -> None:
    """Regression guard for the defect, not for the fix.

    A cadence knob is the whole mechanism: if `FLUSH_EVERY` or `$DATTN_RECORD_FLUSH` come
    back, so does a `synchronize()` inside a timed step.
    """
    assert not hasattr(hooks, "FLUSH_EVERY")
    assert not hasattr(hooks, "ENV_RECORD_FLUSH")
    assert "flush_every" not in inspect.signature(hooks.install).parameters
    body = inspect.getsource(hooks.install)
    step_path = body[body.index("def _sample(") :]
    assert ".flush()" not in step_path and "write_audit" not in step_path


# --------------------------------------------------------------------------------------
# End of run actually reaches the file (P13)
# --------------------------------------------------------------------------------------


def _serve_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_torch: type[FakeTorch], ring: int = 2048
) -> tuple[FakeRunner, StepRecorder]:
    """A worker armed the way `dattn serve --record` arms one."""
    monkeypatch.setenv(hooks.ENV_RECORD, "1")
    monkeypatch.setenv(hooks.ENV_RECORD_DIR, str(tmp_path))
    monkeypatch.setenv("DATTN_RECORD_WINDOW", "serve")
    runner = FakeRunner(decode_step(4))
    rec = hooks.install_from_env(runner, torch_mod=fake_torch, ring=ring)
    assert rec is not None
    return runner, rec


def test_the_signal_drain_is_armed_by_the_first_forward_and_not_before(
    fake_torch: type[FakeTorch], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ordering is the defect. vLLM installs ITS handlers after our boot hook has run.

    `dattn.engine.boot` calls `install_from_env` from inside the EngineCore constructor,
    and `EngineCoreProc.run_engine_core` calls `signal.signal` only after that constructor
    returns (N:v1/engine/core.py:1328-1358). A handler armed at install time is therefore
    overwritten before the first step and can never fire, which is why the arming is
    deferred to the first forward and asserted here rather than assumed.
    """
    runner, _ = _serve_runner(monkeypatch, tmp_path, fake_torch)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL, "install must not touch signals"

    seen: list[int] = []
    signal.signal(signal.SIGTERM, lambda sig, frame: seen.append(sig))  # vLLM, arriving later
    runner.execute_model(decode_step(4))
    runner._model_forward()
    runner._sample()

    handler = signal.getsignal(signal.SIGTERM)
    assert getattr(handler, "_dattn_drain", False), "the first forward did not arm the drain"
    hooks.uninstall(runner)


def test_a_sigterm_to_the_worker_writes_the_window_and_still_runs_vllms_handler(
    fake_torch: type[FakeTorch], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The regression itself: 0 bytes over ~10^5 steps because only `atexit` ever drained.

    SIGTERM went to the API server; the ring lives in the EngineCore worker process, which
    `vllm.v1.utils.shutdown` stops with `proc.terminate()`, and that worker's `atexit`
    did not run. So the signal has to be a drain point in the worker, and it has to chain
    rather than replace, or vLLM never learns it was asked to stop.
    """
    runner, rec = _serve_runner(monkeypatch, tmp_path, fake_torch)
    out = tmp_path / "steps_serve.jsonl"

    seen: list[int] = []
    signal.signal(signal.SIGTERM, lambda sig, frame: seen.append(int(sig)))
    for _ in range(30):
        runner.execute_model(decode_step(4))
        runner._model_forward()
        runner._sample()
    assert out.read_text() == "", "nothing may reach the file mid-run"

    os.kill(os.getpid(), signal.SIGTERM)
    assert seen == [int(signal.SIGTERM)], "vLLM's own handler must still run after ours"
    assert out.read_text().count("\n") == 30, "the window did not reach the file"
    assert (tmp_path / "audit_serve.json").exists(), "the audit sidecar rides every drain"
    assert rec.pending == 0
    hooks.uninstall(runner)


def test_the_signal_drain_keeps_the_writer_open_for_the_steps_that_follow_it(
    fake_torch: type[FakeTorch], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """vLLM's SIGTERM handler REQUESTS shutdown; the busy loop then finishes its work.

    Steps therefore still arrive after the signal, so the drain flushes and keeps the
    writer, and only `atexit`/`uninstall` closes it. Closing on the signal would drop
    exactly the requests the graceful shutdown stayed alive to finish.
    """
    runner, _ = _serve_runner(monkeypatch, tmp_path, fake_torch)
    out = tmp_path / "steps_serve.jsonl"
    signal.signal(signal.SIGTERM, lambda sig, frame: None)
    for _ in range(5):
        runner._model_forward()
        runner._sample()

    os.kill(os.getpid(), signal.SIGTERM)
    assert out.read_text().count("\n") == 5
    for _ in range(4):
        runner._model_forward()
        runner._sample()
    hooks.uninstall(runner)
    assert out.read_text().count("\n") == 9, "the four steps after the signal were dropped"


def test_arming_off_the_main_thread_is_a_no_op_rather_than_a_crash(tmp_path: Path) -> None:
    """`signal.signal` is main-thread only. Elsewhere `atexit` stays the only drain."""
    result: dict[str, Any] = {}

    def body() -> None:
        try:
            result["armed"] = hooks._install_signal_drain(lambda: 0)
        except BaseException as exc:  # noqa: BLE001 - the point is that nothing escapes
            result["raised"] = exc

    t = threading.Thread(target=body)
    t.start()
    t.join()
    assert "raised" not in result, f"arming off the main thread raised {result.get('raised')!r}"
    assert result["armed"] == []


def test_the_ring_wrap_feeds_the_writer_instead_of_growing_in_memory(
    fake_torch: type[FakeTorch], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The early drain the wrap already forced now has somewhere to go on a served run.

    This adds no `synchronize()`: the count below is one per wrap, which is exactly what
    the wrap cost before this change, when the records went into an unbounded in-memory
    `_overflow` list and the file stayed empty anyway.
    """
    runner, rec = _serve_runner(monkeypatch, tmp_path, fake_torch, ring=8)
    out = tmp_path / "steps_serve.jsonl"
    for _ in range(20):
        runner._model_forward()
        runner._sample()

    assert fake_torch.cuda.syncs == 2, "one synchronize per wrap, and not one more"
    assert out.read_text().count("\n") == 16, "the wrapped records did not reach the writer"
    assert rec._overflow == [], "a served ring must not accumulate records in the worker"
    assert all(json.loads(line)["drained_early"] for line in out.read_text().splitlines())

    hooks.uninstall(runner)
    assert out.read_text().count("\n") == 20, "the tail was lost"


def test_the_bench_path_still_keeps_its_wrapped_records_in_process(
    runner: FakeRunner, fake_torch: type[FakeTorch]
) -> None:
    """Writer-less means `_overflow`, unchanged: the bench driver drains in process.

    `drain()` returns overflow plus slots, so a bench cell longer than the ring still
    yields every record and nothing is written to disk behind the driver's back.
    """
    rec = hooks.install(runner, window_id="w0", torch_mod=fake_torch, ring=4)
    for _ in range(11):
        runner._model_forward()
        runner._sample()
    assert len(rec._overflow) == 8
    assert len(rec.drain()) == 11
    hooks.uninstall(runner)


CHILD = """
import os, signal, sys, time
sys.path[:0] = sys.argv[3].split(os.pathsep)
from fakes import FakeRunner, FakeTorch, decode_step
from dattn.metrics import hooks

os.environ["DATTN_RECORD"] = "1"
os.environ["DATTN_RECORD_DIR"] = sys.argv[1]
os.environ["DATTN_RECORD_WINDOW"] = "serve"
runner = FakeRunner(decode_step(4))
hooks.install_from_env(runner, torch_mod=FakeTorch, ring=4096)

# vLLM's own handler: it REQUESTS shutdown and returns. It does not exit the process,
# and nothing here ever reaches `atexit`, which is the situation that lost the window.
state = {"stop": False}
signal.signal(signal.SIGTERM, lambda s, f: state.__setitem__("stop", True))

for _ in range(%(steps)d):
    runner._model_forward()
    runner._sample()
open(sys.argv[2], "w").close()
while True:
    time.sleep(0.02)
"""


def _run_child(tmp_path: Path, steps: int) -> Path:
    """A worker that is SIGTERMed, keeps running, and is then SIGKILLed. No `atexit`."""
    script = tmp_path / "child.py"
    script.write_text(CHILD % {"steps": steps})
    out_dir = tmp_path / "records"
    ready = tmp_path / "ready"
    roots = os.pathsep.join([str(Path(__file__).parents[2]), str(Path(hooks.__file__).parents[3])])
    proc = subprocess.Popen([sys.executable, str(script), str(out_dir), str(ready), roots])
    try:
        for _ in range(2400):  # Allow cold vLLM/Torch imports before testing shutdown.
            if ready.exists():
                break
            time.sleep(0.05)
        assert ready.exists(), "the child never finished its steps"
        proc.send_signal(signal.SIGTERM)
        time.sleep(0.5)
        proc.kill()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=10)
    return out_dir / "steps_serve.jsonl"


def test_a_worker_that_never_reaches_atexit_still_writes_its_window(tmp_path: Path) -> None:
    """The defect end to end, in a real process, with `atexit` deliberately unreachable.

    This is the shape of what happened on GPU 3: the file existed, the recorder had armed,
    ~10^5 steps ran, the server was stopped, and the file was 0 bytes. Here the child is
    SIGTERMed, handles the signal the way vLLM does (set a flag, return, keep running) and
    is then SIGKILLed, so `atexit` never runs at all. Before the chained signal drain this
    file was empty; the assertion is that it is not.
    """
    out = _run_child(tmp_path, steps=50)
    assert out.exists(), "the recorder never armed in the child"
    assert out.read_text().count("\n") == 50, "the window was lost exactly as it was on GPU 3"
    assert (out.parent / "audit_serve.json").exists()


SERVER = '''
import multiprocessing, os, signal, sys, time

sys.path[:0] = sys.argv[3].split(os.pathsep)


def engine_core(out_dir, ready, steps):
    """The process that owns the ring. On GPU 2 tonight this was PID 1934947."""
    from fakes import FakeRunner, FakeTorch, decode_step

    from dattn.metrics import hooks

    os.environ["DATTN_RECORD"] = "1"
    os.environ["DATTN_RECORD_DIR"] = out_dir
    os.environ["DATTN_RECORD_WINDOW"] = "serve"
    runner = FakeRunner(decode_step(4))
    hooks.install_from_env(runner, torch_mod=FakeTorch, ring=4096)
    state = {"stop": False}
    # vLLM installs THIS after the constructor that ran our boot hook, and it only
    # requests shutdown: N:v1/engine/core.py:1349-1358.
    signal.signal(signal.SIGTERM, lambda s, f: state.__setitem__("stop", True))
    for _ in range(steps):
        runner._model_forward()
        runner._sample()
    open(ready, "w").close()
    while not state["stop"]:
        time.sleep(0.01)
    os._exit(0)  # atexit deliberately unreachable, so only the signal drain can save this


if __name__ == "__main__":
    out_dir, ready, roots, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    proc = multiprocessing.get_context("fork").Process(
        target=engine_core, args=(out_dir, ready, steps), name="VLLM::EngineCore"
    )
    proc.start()

    def _term(s, f):
        # Exactly what vllm.v1.utils.shutdown does to the EngineCore process,
        # N:v1/utils.py:618-632: SIGTERM to the child, then join.
        proc.terminate()
        proc.join(5)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _term)
    while True:
        time.sleep(0.02)
'''


def test_the_drain_survives_a_sigterm_aimed_at_the_api_server_not_the_worker(tmp_path: Path) -> None:
    """The process boundary the old `atexit` assumption failed across, modelled exactly.

    On GPU 2 the operator signalled parent 1934354 and the ring lived in child 1934947,
    `VLLM::EngineCore`. The parent here does what `vllm.v1.utils.shutdown` does, the child
    handles SIGTERM the way vLLM's EngineCore does, and the child leaves through
    `os._exit`, so `atexit` cannot possibly be what writes the file. Nothing in this test
    calls into dattn to ask for a drain, which is the whole point: a served worker is
    stopped by a signal, never by a call.
    """
    script = tmp_path / "server.py"
    script.write_text(SERVER)
    out_dir, ready = tmp_path / "records", tmp_path / "ready"
    roots = os.pathsep.join([str(Path(__file__).parents[2]), str(Path(hooks.__file__).parents[3])])
    proc = subprocess.Popen([sys.executable, str(script), str(out_dir), str(ready), roots, "40"])
    try:
        for _ in range(2400):  # Allow cold vLLM/Torch imports before testing shutdown.
            if ready.exists():
                break
            time.sleep(0.05)
        assert ready.exists(), "the engine core never finished its steps"
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=20) == 0, "the api server did not shut down cleanly"
    finally:
        if proc.poll() is None:  # pragma: no cover - only on a hung child
            proc.kill()
            proc.wait(timeout=10)
    out = out_dir / "steps_serve.jsonl"
    assert out.exists(), "the recorder never armed in the engine core"
    assert out.read_text().count("\n") == 40, "0 bytes, exactly as on both cards tonight"


def test_install_refuses_a_runner_with_no_sample_instead_of_degrading(
    runner: FakeRunner, fake_torch: type[FakeTorch]
) -> None:
    """`_sample` is the ONLY caller of `end_step` (review item 37 residual).

    `install` used to wrap it `if callable(orig_sample)` and say nothing otherwise, so a
    runner whose sampler is named something else installed cleanly and then opened every
    step and closed none: no `t_ready_ns`, no step rows, and a window that looks empty
    rather than uninstrumented. The whole walltime measurement would be missing with no
    error anywhere.
    """
    runner._sample = None  # type: ignore[assignment]
    with pytest.raises(NotImplementedError, match="_sample"):
        hooks.install(runner, window_id="w0", torch_mod=fake_torch, ring=8)
    assert hooks.installed(runner) is None, "a refused install leaves nothing behind"
