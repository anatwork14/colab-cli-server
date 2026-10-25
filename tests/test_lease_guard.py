"""Offline tests for the host-side lease deadline guard (internal `project lease-guard`).

No test runs a real `colab` binary. Real-process tests use a fake `colab` script in a temp dir that
records its argv to a file and keeps a directory of "alive" session ids. HOME is redirected to the
temp dir so the default process lock never touches the real home directory.
"""
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from colab_adapter.approval import GlobalCapExceededError
from colab_adapter.errors import AllocationError, LeaseExpiredError, UnmanagedSessionError
from colab_adapter.lease_guard import (
    pid_argv,
    OUTCOME_ENDED_EXTERNALLY,
    OUTCOME_NEEDS_RECONCILIATION,
    OUTCOME_SIGNALLED,
    OUTCOME_STOPPED_VERIFIED,
    run_guard,
)
from colab_adapter.project_manifest import ProjectSpec
from colab_adapter.project_sessions import ProjectSession, ProjectSessionManager
from colab_adapter.provider import ColabProcessLock, ManagedSessionRegistry
from colab_adapter.transports.official_cli import OfficialCliTransport
from tests.approval_helpers import fake_guard_spawner, grant_v2

REPO_ROOT = Path(__file__).resolve().parents[1]

from tests.colab_fakes import FAKE_COLAB_SOURCE  # shared fake `colab` (upstream-shaped output)


def _iso(offset_s):
    return (datetime.now(timezone.utc) + timedelta(seconds=offset_s)).isoformat()


def make_session(session_id, expires_in, state="READY"):
    now = datetime.now(timezone.utc).isoformat()
    return ProjectSession(
        session_id=session_id,
        project_id="proj-guard",
        project_name="guard",
        state=state,
        accelerator="T4",
        created_at=now,
        updated_at=now,
        lease_seconds=float(expires_in),
        expires_at=_iso(expires_in),
        remote_root="/content/guard",
        approved_fingerprint="fp-guard",
        approval_id="apr-guard",
    )


class GuardFixture(unittest.TestCase):
    """Temp dir, fake colab binary, redirected HOME, and a registry with managed sessions."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lease-guard-test-"))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.state_dir = self.tmp / "fake_state"
        (self.state_dir / "alive").mkdir(parents=True)
        self.calls_file = self.tmp / "calls.jsonl"
        self.fake_colab = self.tmp / "bin" / "colab"
        self.fake_colab.parent.mkdir()
        self.fake_colab.write_text(FAKE_COLAB_SOURCE.format(python=sys.executable), encoding="utf-8")
        self.fake_colab.chmod(0o700)
        env = {
            "HOME": str(self.home),
            "FAKE_COLAB_DIR": str(self.state_dir),
            "FAKE_COLAB_CALLS": str(self.calls_file),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.registry_path = self.tmp / "registry" / "managed_sessions.json"
        self.registry = ManagedSessionRegistry(self.registry_path)
        self.lock = ColabProcessLock(self.tmp / "colab.lock")
        self.spawned = []

    def put_session(self, session):
        with self.registry._lock():
            data = self.registry._read_locked()
            data[session.session_id] = session.to_dict()
            self.registry._atomic_write_locked(data)

    def mark_alive(self, session_id):
        (self.state_dir / "alive" / session_id).write_text("", encoding="utf-8")

    def is_alive(self, session_id):
        return (self.state_dir / "alive" / session_id).exists()

    def calls(self):
        if not self.calls_file.exists():
            return []
        return [json.loads(line) for line in self.calls_file.read_text(encoding="utf-8").splitlines() if line]

    def stop_calls(self, session_id=None):
        out = [c for c in self.calls() if "stop" in c]
        if session_id is not None:
            out = [c for c in out if c[c.index("-s") + 1] == session_id]
        return out

    def manager(self, **kwargs):
        kwargs.setdefault("auth_mode", "oauth2")
        return ProjectSessionManager(
            registry=self.registry, lock=self.lock, colab_binary=str(self.fake_colab), **kwargs
        )

    def recording_spawner(self):
        """Real detached spawn, remembered so tests can wait for or kill the guard."""

        def spawn(cmd, **kwargs):
            proc = subprocess.Popen(cmd, **kwargs)
            self.spawned.append(proc)
            return proc

        def _cleanup():
            for proc in self.spawned:
                if proc.poll() is None:
                    proc.kill()
                proc.wait()

        self.addCleanup(_cleanup)
        return spawn

    def record(self, session_id):
        return self.registry.get_all_managed().get(session_id, {})

    def guard_metadata(self, session_id):
        return self.record(session_id).get("metadata", {})


class TestRealProcessGuard(GuardFixture):
    def test_deadline_stops_only_target_and_verifies_removal(self):
        """(a) After the deadline: exactly one `stop -s <id>` with --auth=oauth2, sessions verification, TERMINATED."""
        target = "colab-v2-guard-target01"
        other = "colab-v2-guard-other0002"
        self.put_session(make_session(target, expires_in=3))
        self.put_session(make_session(other, expires_in=3600))
        self.mark_alive(target)
        self.mark_alive(other)

        mgr = self.manager(spawn_fn=self.recording_spawner())
        mgr.start_lease_guard(ProjectSession.from_dict(self.record(target)))
        guard = self.spawned[-1]
        self.assertEqual(guard.wait(timeout=60), 0)

        self.assertEqual(self.stop_calls(target), [["--auth=oauth2", "stop", "-s", target]])
        self.assertEqual(self.stop_calls(other), [], "the other managed session must never be stopped")
        self.assertFalse(self.is_alive(target))
        self.assertTrue(self.is_alive(other))
        calls = self.calls()
        stop_index = calls.index(["--auth=oauth2", "stop", "-s", target])
        self.assertTrue(any("sessions" in c for c in calls[stop_index + 1:]),
                        "removal must be verified with `sessions` after the stop")
        self.assertEqual(self.record(target)["state"], "TERMINATED")
        self.assertEqual(self.record(other)["state"], "READY")
        self.assertEqual(self.guard_metadata(target)["guard_outcome"], OUTCOME_STOPPED_VERIFIED)
        self.assertEqual(self.guard_metadata(target)["guard_pid"], guard.pid)

        log_dir = self.registry_path.parent / "lease_guards"
        log_file = log_dir / f"{target}.log"
        self.assertEqual(stat.S_IMODE(os.stat(log_dir).st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(os.stat(log_file).st_mode), 0o600)
        self.assertNotIn(other, log_file.read_text(encoding="utf-8"))

    def test_manual_stop_before_deadline_makes_guard_exit_without_stop(self):
        """(b) A session that ends before its deadline is left alone by the guard."""
        target = "colab-v2-guard-manual01"
        self.put_session(make_session(target, expires_in=60))
        self.mark_alive(target)
        mgr = self.manager(spawn_fn=self.recording_spawner())
        mgr.start_lease_guard(ProjectSession.from_dict(self.record(target)))
        guard = self.spawned[-1]
        time.sleep(0.5)
        self.registry.update_state(target, "TERMINATED")
        self.assertEqual(guard.wait(timeout=30), 0)
        self.assertEqual(self.stop_calls(), [])
        self.assertEqual(self.guard_metadata(target)["guard_outcome"], OUTCOME_ENDED_EXTERNALLY)

    def test_guard_survives_parent_process_exit(self):
        """(c) The guard is spawned from a short-lived parent and still performs the stop."""
        target = "colab-v2-guard-orphan01"
        self.put_session(make_session(target, expires_in=4))
        self.mark_alive(target)
        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
            "from pathlib import Path\n"
            "from colab_adapter.project_sessions import ProjectSession, ProjectSessionManager\n"
            "from colab_adapter.provider import ManagedSessionRegistry\n"
            f"reg = ManagedSessionRegistry(Path({str(self.registry_path)!r}))\n"
            f"mgr = ProjectSessionManager(registry=reg, colab_binary={str(self.fake_colab)!r}, auth_mode='oauth2')\n"
            f"mgr.start_lease_guard(ProjectSession.from_dict(reg.get_all_managed()[{target!r}]))\n"
        )
        parent = subprocess.run([sys.executable, "-c", script], stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, text=True)
        self.assertEqual(parent.returncode, 0, parent.stderr)
        guard_pid = self.guard_metadata(target)["guard_pid"]
        deadline = time.time() + 45
        while time.time() < deadline and not self.stop_calls(target):
            time.sleep(0.2)
        self.assertEqual(self.stop_calls(target), [["--auth=oauth2", "stop", "-s", target]])
        # The stop is followed by read-only verification (sessions, usage); wait for the guard to finish
        # before reading the final state.
        for _ in range(100):
            if pid_argv(guard_pid) is None:
                break
            time.sleep(0.1)
        self.assertIsNone(pid_argv(guard_pid), "orphaned guard must exit after the stop")
        self.assertEqual(self.record(target)["state"], "TERMINATED")

    def test_sigterm_exits_cleanly_without_stopping_session(self):
        """(k) SIGTERM to the guard exits it without stopping the session."""
        target = "colab-v2-guard-sigterm01"
        self.put_session(make_session(target, expires_in=600))
        self.mark_alive(target)
        mgr = self.manager(spawn_fn=self.recording_spawner())
        mgr.start_lease_guard(ProjectSession.from_dict(self.record(target)))
        guard = self.spawned[-1]
        time.sleep(0.5)
        guard.send_signal(signal.SIGTERM)
        self.assertEqual(guard.wait(timeout=30), 0)
        self.assertEqual(self.stop_calls(), [])
        self.assertTrue(self.is_alive(target))
        self.assertEqual(self.record(target)["state"], "READY")
        self.assertEqual(self.guard_metadata(target)["guard_outcome"], OUTCOME_SIGNALLED)

    def test_stop_terminates_the_guard_and_it_exits_without_stopping(self):
        """(g) A successful stop() terminates this session's guard; the guard adds no second stop."""
        target = "colab-v2-guard-killed01"
        self.put_session(make_session(target, expires_in=600))
        self.mark_alive(target)
        mgr = self.manager(spawn_fn=self.recording_spawner())
        mgr.start_lease_guard(ProjectSession.from_dict(self.record(target)))
        guard = self.spawned[-1]
        time.sleep(0.5)
        self.assertTrue(mgr.stop(target))
        self.assertEqual(guard.wait(timeout=30), 0)
        self.assertEqual(self.stop_calls(target), [["--auth=oauth2", "stop", "-s", target]])
        self.assertEqual(self.record(target)["state"], "TERMINATED")

    def test_ensure_respawns_dead_guard_once_not_twice(self):
        """(h) A killed guard is respawned exactly once; a second ensure sees it alive and does nothing."""
        target = "colab-v2-guard-respawn01"
        self.put_session(make_session(target, expires_in=600))
        self.mark_alive(target)
        spawn = self.recording_spawner()
        counter = {"n": 0}

        def counting(cmd, **kwargs):
            counter["n"] += 1
            return spawn(cmd, **kwargs)

        mgr = self.manager(spawn_fn=counting)
        first = mgr.ensure_lease_guard(target)
        self.assertEqual((first["respawned"], counter["n"]), (True, 1))
        self.spawned[-1].kill()
        self.spawned[-1].wait()

        second = mgr.ensure_lease_guard(target)
        third = mgr.ensure_lease_guard(target)
        self.assertTrue(second["respawned"])
        self.assertFalse(third["respawned"])
        self.assertEqual(counter["n"], 2, "exactly one respawn for one dead guard")
        self.assertEqual(third["guard_pid"], second["guard_pid"])
        mgr.stop(target)
        self.spawned[-1].wait(timeout=30)

    def test_guard_refuses_unmanaged_and_expired_sessions_for_ensure(self):
        """ensure_lease_guard never spawns for unmanaged ids or expired leases."""
        spawns = []
        mgr = self.manager(spawn_fn=lambda cmd, **kw: spawns.append(cmd))
        with self.assertRaises(UnmanagedSessionError):
            mgr.ensure_lease_guard("colab-v2-not-managed01")
        expired = "colab-v2-guard-expired01"
        session = make_session(expired, expires_in=-5)
        self.put_session(session)
        with self.assertRaises(LeaseExpiredError):
            mgr.ensure_lease_guard(expired)
        self.assertEqual(spawns, [])


class TestGuardRunInProcess(GuardFixture):
    def guard(self, session_id, **kwargs):
        kwargs.setdefault("poll_interval", 0.05)
        kwargs.setdefault("backoff", lambda attempt: 0.01)
        return run_guard(
            session_id=session_id,
            registry_path=self.registry_path,
            colab_binary=str(self.fake_colab),
            auth_mode="oauth2",
            deadline_ts=time.time(),
            **kwargs,
        )

    def test_stop_failures_are_bounded_then_flagged(self):
        """(d) Stop fails every time: exactly max_stop_attempts stops, NEEDS_RECONCILIATION, exit 1."""
        target = "colab-v2-guard-fail01"
        self.put_session(make_session(target, expires_in=600))
        self.mark_alive(target)
        with mock.patch.dict(os.environ, {"FAKE_COLAB_STOP_FAIL": "1"}):
            code = self.guard(target, max_stop_attempts=3)
        self.assertEqual(code, 1)
        self.assertEqual(len(self.stop_calls(target)), 3, "bounded retries, no unbounded loop")
        self.assertEqual(self.record(target)["state"], "NEEDS_RECONCILIATION")
        self.assertEqual(self.guard_metadata(target)["guard_outcome"], OUTCOME_NEEDS_RECONCILIATION)

    def test_listed_after_zero_exit_stop_is_retried_then_flagged(self):
        """(e) stop exits 0 but `sessions` still lists the id: retried, then flagged, nonzero exit."""
        target = "colab-v2-guard-lie01"
        self.put_session(make_session(target, expires_in=600))
        self.mark_alive(target)
        with mock.patch.dict(os.environ, {"FAKE_COLAB_STOP_LIE": "1"}):
            code = self.guard(target, max_stop_attempts=2)
        self.assertEqual(code, 1)
        self.assertEqual(len(self.stop_calls(target)), 2)
        sessions_calls = [c for c in self.calls() if "sessions" in c]
        self.assertEqual(len(sessions_calls), 2, "every stop is verified with `sessions`")
        self.assertTrue(self.is_alive(target))
        self.assertEqual(self.record(target)["state"], "NEEDS_RECONCILIATION")

    def test_log_never_contains_token_looking_strings(self):
        """(l) Tokens echoed by the upstream CLI never reach the guard log."""
        target = "colab-v2-guard-redact01"
        self.put_session(make_session(target, expires_in=600))
        self.mark_alive(target)
        with mock.patch.dict(os.environ, {"FAKE_COLAB_STOP_FAIL": "1"}):
            self.guard(target, max_stop_attempts=2)
        log_text = (self.registry_path.parent / "lease_guards" / f"{target}.log").read_text(encoding="utf-8")
        self.assertIn("stop attempt 1 failed", log_text)
        self.assertNotIn("ya29.", log_text)
        self.assertNotIn("FAKETOKEN", log_text)
        self.assertNotIn("0123456789abcdef", log_text)

    def test_guard_ignores_other_sessions_in_same_registry(self):
        """Only the guard's own id is ever stopped, even if other managed sessions are expired."""
        target = "colab-v2-guard-scope01"
        other = "colab-v2-guard-scope02"
        self.put_session(make_session(target, expires_in=600))
        self.put_session(make_session(other, expires_in=-60))
        self.mark_alive(target)
        self.mark_alive(other)
        self.assertEqual(self.guard(target), 0)
        self.assertEqual(self.stop_calls(other), [])
        self.assertTrue(self.is_alive(other))


class TestUpFailClosed(GuardFixture):
    def make_spec(self):
        project_root = self.tmp / "project"
        project_root.mkdir()
        (project_root / "train.py").write_text("print('train')\n", encoding="utf-8")
        return ProjectSpec(name="guardproj", root=project_root, entrypoint="train.py",
                           gpu="T4", timeout_seconds=600.0)

    def run_cmd(self, cmd, **kwargs):
        self.cli_cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    def test_guard_spawn_failure_stops_session_and_raises(self):
        """(f) spawn fails: stop is called, AllocationError raised, no READY session left behind."""
        self.cli_cmds = []
        spec = self.make_spec()

        def broken_spawn(cmd, **kwargs):
            raise OSError("spawn refused")

        mgr = ProjectSessionManager(registry=self.registry, lock=self.lock, run_cmd_fn=self.run_cmd,
                                    spawn_fn=broken_spawn, auth_mode="oauth2")
        approval_id = grant_v2(mgr, spec, 900.0)
        with self.assertRaises(AllocationError) as ctx:
            mgr.up(spec, lease_seconds=900.0, approval_id=approval_id)
        self.assertIn("Lease guard could not be started", str(ctx.exception))
        new_calls = [c for c in self.cli_cmds if "new" in c]
        stop_calls = [c for c in self.cli_cmds if "stop" in c]
        self.assertEqual(len(new_calls), 1)
        self.assertEqual(len(stop_calls), 1)
        sid = new_calls[0][new_calls[0].index("-s") + 1]
        self.assertEqual(stop_calls[0], ["colab", "--auth=oauth2", "stop", "-s", sid])
        self.assertEqual(self.record(sid)["state"], "TERMINATED")
        self.assertEqual(self.registry.get_active_managed(), [])

    def test_guard_that_exits_immediately_is_not_confirmed(self):
        """A guard that dies during startup is treated like a failed spawn (fail closed)."""
        self.cli_cmds = []
        spec = self.make_spec()

        def dying_spawn(cmd, **kwargs):
            return subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True)

        mgr = ProjectSessionManager(registry=self.registry, lock=self.lock, run_cmd_fn=self.run_cmd,
                                    spawn_fn=dying_spawn, auth_mode="oauth2")
        approval_id = grant_v2(mgr, spec, 900.0)
        with self.assertRaises(AllocationError):
            mgr.up(spec, lease_seconds=900.0, approval_id=approval_id)
        self.assertTrue(any("stop" in c for c in self.cli_cmds))
        self.assertEqual(self.registry.get_active_managed(), [])

    def test_successful_up_starts_one_guard_and_records_it(self):
        self.cli_cmds = []
        spec = self.make_spec()
        spawner = fake_guard_spawner(self)
        mgr = ProjectSessionManager(registry=self.registry, lock=self.lock, run_cmd_fn=self.run_cmd,
                                    spawn_fn=spawner, auth_mode="oauth2")
        approval_id = grant_v2(mgr, spec, 900.0)
        session = mgr.up(spec, lease_seconds=900.0, approval_id=approval_id)
        meta = self.guard_metadata(session.session_id)
        self.assertTrue(meta["guard_pid"] > 0)
        self.assertIn("guard_deadline", meta)
        self.assertEqual(self.record(session.session_id)["state"], "READY")
        self.assertEqual([c for c in self.cli_cmds if "new" in c][0][:3], ["colab", "--auth=oauth2", "new"])


class TestAuthModeAndExecTimeout(GuardFixture):
    def test_auth_flag_on_new_stop_and_transport(self):
        """(i) new/stop/exec commands carry the same --auth value as the transport."""
        cli_cmds = []

        def run(cmd, **kwargs):
            cli_cmds.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        project_root = self.tmp / "authproject"
        project_root.mkdir()
        spec = ProjectSpec(name="authproj", root=project_root, entrypoint="train.py", gpu="T4", timeout_seconds=60.0)
        mgr = ProjectSessionManager(registry=self.registry, lock=self.lock, run_cmd_fn=run,
                                    spawn_fn=fake_guard_spawner(self), auth_mode="adc")
        self.assertEqual(mgr.transport.auth_mode, "adc")
        (project_root / "train.py").write_text("print(1)\n", encoding="utf-8")
        approval_id = grant_v2(mgr, spec, 900.0)
        session = mgr.up(spec, lease_seconds=900.0, approval_id=approval_id)
        mgr.stop(session.session_id)
        new_cmd = [c for c in cli_cmds if "new" in c][0]
        stop_cmd = [c for c in cli_cmds if "stop" in c][0]
        self.assertEqual(new_cmd[:3], ["colab", "--auth=adc", "new"])
        self.assertEqual(stop_cmd, ["colab", "--auth=adc", "stop", "-s", session.session_id])

    def test_transport_exec_passes_timeout_minus_margin(self):
        """(j) colab exec gets --timeout (budget minus margin); the subprocess budget is the full budget."""
        self.put_session(make_session("colab-v2-guard-tmo01", expires_in=600))
        transport = OfficialCliTransport(registry=self.registry, cli_path=str(self.fake_colab), auth_mode="oauth2")
        captured = {}

        def fake_run(cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["timeout"] = kwargs.get("timeout")
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

        with mock.patch("colab_adapter.transports.official_cli.subprocess.run", side_effect=fake_run):
            transport.exec_code("colab-v2-guard-tmo01", "print(1)", timeout=600.0)
        cmd = captured["cmd"]
        self.assertEqual(cmd[:2], [str(self.fake_colab), "--auth=oauth2"])
        self.assertIn("--timeout", cmd)
        self.assertEqual(cmd[cmd.index("--timeout") + 1], "585")
        self.assertEqual(captured["timeout"], 600.0)

    def test_manager_exec_never_passes_more_than_remaining_lease(self):
        """(j) A 600 s exec request on a ~100 s lease is clamped to the remaining lease."""
        sid = "colab-v2-guard-tmo02"
        self.put_session(make_session(sid, expires_in=100))
        transport = mock.MagicMock()
        transport.exec_code.return_value = {"success": True, "output": "", "has_traceback": False}
        mgr = ProjectSessionManager(registry=self.registry, lock=self.lock, transport=transport,
                                    spawn_fn=fake_guard_spawner(self), auth_mode="oauth2")
        mgr.exec(sid, "print(1)", timeout=600.0)
        passed = transport.exec_code.call_args.kwargs["timeout"]
        self.assertLessEqual(passed, 100.0)
        self.assertGreater(passed, 90.0)

    def test_exec_refused_when_remaining_lease_is_too_short(self):
        sid = "colab-v2-guard-tmo03"
        self.put_session(make_session(sid, expires_in=0.4))
        transport = mock.MagicMock()
        mgr = ProjectSessionManager(registry=self.registry, lock=self.lock, transport=transport,
                                    spawn_fn=fake_guard_spawner(self), auth_mode="oauth2")
        with self.assertRaises(LeaseExpiredError):
            mgr.exec(sid, "print(1)", timeout=600.0)
        transport.exec_code.assert_not_called()


class TestGuardRetriesAndLockWait(GuardFixture):
    """F2/F4: the guard never gives up on its own and waits for a busy project lock."""

    def guard(self, session_id, **kwargs):
        kwargs.setdefault("poll_interval", 0.05)
        kwargs.setdefault("backoff", lambda attempt: 0.01)
        return run_guard(
            session_id=session_id,
            registry_path=self.registry_path,
            colab_binary=str(self.fake_colab),
            auth_mode="oauth2",
            deadline_ts=time.time(),
            **kwargs,
        )

    def test_unbounded_guard_retries_beyond_five_attempts_until_verified(self):
        target = "colab-v2-guard-retry01"
        self.put_session(make_session(target, expires_in=600))
        self.mark_alive(target)
        (self.state_dir / "stop_fail_remaining").write_text("7", encoding="utf-8")
        code = self.guard(target)  # default: no attempt cap
        self.assertEqual(code, 0)
        self.assertEqual(len(self.stop_calls(target)), 8, "7 failures then one verified success")
        self.assertFalse(self.is_alive(target))
        self.assertEqual(self.record(target)["state"], "TERMINATED")
        self.assertEqual(self.guard_metadata(target)["guard_outcome"], OUTCOME_STOPPED_VERIFIED)
        log_text = (self.registry_path.parent / "lease_guards" / f"{target}.log").read_text(encoding="utf-8")
        self.assertIn("stop attempt 1 failed", log_text)
        self.assertIn("stop attempt 7 failed", log_text)
        self.assertNotIn("ya29.", log_text)

    def test_first_failure_flags_needs_reconciliation_while_guard_keeps_going(self):
        target = "colab-v2-guard-flag01"
        self.put_session(make_session(target, expires_in=600))
        self.mark_alive(target)
        (self.state_dir / "stop_fail_remaining").write_text("1", encoding="utf-8")
        seen_after_mark = []
        from colab_adapter import lease_guard as lg
        real_mark = lg._mark_needs_reconciliation

        def spy(registry, sid, reason):
            real_mark(registry, sid, reason)
            seen_after_mark.append(self.record(sid)["state"])

        with mock.patch.object(lg, "_mark_needs_reconciliation", side_effect=spy):
            self.assertEqual(self.guard(target), 0)
        self.assertEqual(seen_after_mark[:1], ["NEEDS_RECONCILIATION"])
        self.assertEqual(self.record(target)["state"], "TERMINATED")

    def test_guard_waits_for_busy_project_lock_then_stops(self):
        """A busy lock is not a failed attempt: with a one-attempt cap the stop still succeeds once freed."""
        target = "colab-v2-guard-lock01"
        self.put_session(make_session(target, expires_in=600))
        self.mark_alive(target)
        self.assertTrue(self.lock.acquire(blocking=False))
        timer = threading.Timer(1.0, self.lock.release)
        timer.start()
        self.addCleanup(timer.join)
        code = self.guard(target, max_stop_attempts=1, lock=self.lock, lock_wait_s=0.2)
        self.assertEqual(code, 0)
        self.assertEqual(len(self.stop_calls(target)), 1)
        self.assertFalse(self.is_alive(target))
        self.assertEqual(self.record(target)["state"], "TERMINATED")

    def test_stop_not_found_is_needs_reconciliation_not_terminated(self):
        target = "colab-v2-guard-nf0001"
        self.put_session(make_session(target, expires_in=600))  # no VM with this name exists
        code = self.guard(target, max_stop_attempts=2)
        self.assertEqual(code, 1)
        record = self.record(target)
        self.assertEqual(record["state"], "NEEDS_RECONCILIATION")
        self.assertIn("stop reported session not found; runtime may be orphaned",
                      record["metadata"].get("last_transition_reason", ""))


class TestServerSideVerification(GuardFixture):
    """F3b: verification uses `usage` active assignments against the baseline recorded before `new`."""

    def guard(self, session_id, **kwargs):
        kwargs.setdefault("poll_interval", 0.05)
        kwargs.setdefault("backoff", lambda attempt: 0.01)
        return run_guard(
            session_id=session_id,
            registry_path=self.registry_path,
            colab_binary=str(self.fake_colab),
            auth_mode="oauth2",
            deadline_ts=time.time(),
            **kwargs,
        )

    def set_baseline(self, session_id, before):
        self.put_session(make_session(session_id, expires_in=600))
        meta = {} if before is None else {"assignments_before": before}
        with self.registry._lock():
            data = self.registry._read_locked()
            data[session_id]["metadata"] = meta
            self.registry._atomic_write_locked(data)

    def test_stop_verified_strong_when_assignments_back_to_baseline(self):
        target = "colab-v2-verify-ok01"
        self.set_baseline(target, 0)
        self.mark_alive(target)
        self.assertTrue(self.manager().stop(target, terminate_guard=False))
        self.assertEqual(self.record(target)["state"], "TERMINATED")
        self.assertEqual(self.record(target)["metadata"]["verification"], "strong")
        self.assertTrue(any("usage" in c for c in self.calls()))

    def test_extra_assignment_after_stop_is_not_verified_and_is_not_restopped(self):
        target = "colab-v2-verify-extra"
        other = "colab-v2-verify-other"
        self.set_baseline(target, 0)
        self.mark_alive(target)
        self.mark_alive(other)  # another live assignment: usage shows 1 > baseline 0
        from colab_adapter.project_sessions import StopUnverifiedError
        with self.assertRaises(StopUnverifiedError) as ctx:
            self.manager().stop(target, terminate_guard=False)
        self.assertIn("active assignments 1 > baseline 0", str(ctx.exception))
        self.assertEqual(self.record(target)["state"], "NEEDS_RECONCILIATION")
        code = self.guard(target, max_stop_attempts=3)
        self.assertEqual(code, 1)
        self.assertEqual(len(self.stop_calls(target)), 1, "verification-only retries; no repeated stop")
        self.assertEqual(self.record(target)["state"], "NEEDS_RECONCILIATION")

    def test_weak_verification_when_baseline_unknown(self):
        target = "colab-v2-verify-weak"
        self.set_baseline(target, None)
        self.mark_alive(target)
        self.assertTrue(self.manager().stop(target, terminate_guard=False))
        self.assertEqual(self.record(target)["state"], "TERMINATED")
        self.assertEqual(self.record(target)["metadata"]["verification"], "weak")

    def test_listed_after_stop_is_never_verified(self):
        target = "colab-v2-verify-lst1"
        self.set_baseline(target, 0)
        self.mark_alive(target)
        with mock.patch.dict(os.environ, {"FAKE_COLAB_STOP_LIE": "1"}):
            with self.assertRaises(Exception):
                self.manager().stop(target, terminate_guard=False)
        self.assertEqual(self.record(target)["state"], "NEEDS_RECONCILIATION")


class TestUpBranchHoles(GuardFixture):
    """F3c/F4: interrupts and busy locks in up() never free the cap or leave a stuck record."""

    def make_spec(self):
        project_root = self.tmp / "project"
        project_root.mkdir(exist_ok=True)
        (project_root / "train.py").write_text("print('train')\n", encoding="utf-8")
        return ProjectSpec(name="holes", root=project_root, entrypoint="train.py",
                           gpu="T4", timeout_seconds=600.0)

    def scripted_run(self, new_behaviour):
        self.cli_cmds = []
        usage_out = "Current balance: 5.00 compute units\nUsage rate: 0.00/hr\nActive assignments: 0"

        def run(cmd, **kwargs):
            self.cli_cmds.append(list(cmd))
            if "usage" in cmd:
                return subprocess.CompletedProcess(cmd, 0, usage_out, "")
            if "new" in cmd:
                return new_behaviour(cmd)
            if "stop" in cmd:
                return subprocess.CompletedProcess(cmd, 0, "[colab] Session 'x' not found.", "")
            return subprocess.CompletedProcess(cmd, 0, "ok", "")

        return run

    def test_up_records_assignments_before_colab_new(self):
        spec = self.make_spec()
        run = self.scripted_run(lambda cmd: subprocess.CompletedProcess(cmd, 0, "[colab] Session READY.", ""))
        mgr = ProjectSessionManager(registry=self.registry, lock=self.lock, run_cmd_fn=run,
                                    spawn_fn=fake_guard_spawner(self), auth_mode="oauth2")
        approval_id = grant_v2(mgr, spec, 900.0)
        session = mgr.up(spec, lease_seconds=900.0, approval_id=approval_id)
        self.assertEqual(self.record(session.session_id)["metadata"]["assignments_before"], 0)
        usage_idx = next(i for i, c in enumerate(self.cli_cmds) if "usage" in c)
        new_idx = next(i for i, c in enumerate(self.cli_cmds) if "new" in c)
        self.assertLess(usage_idx, new_idx, "baseline is read before colab new")

    def test_keyboard_interrupt_during_new_keeps_record_and_cap(self):
        spec = self.make_spec()

        def interrupted(cmd):
            raise KeyboardInterrupt()

        run = self.scripted_run(interrupted)
        mgr = ProjectSessionManager(registry=self.registry, lock=self.lock, run_cmd_fn=run,
                                    spawn_fn=fake_guard_spawner(self), auth_mode="oauth2")
        approval_id = grant_v2(mgr, spec, 900.0)
        with self.assertRaises(KeyboardInterrupt):
            mgr.up(spec, lease_seconds=900.0, approval_id=approval_id)
        sids = list(self.registry.get_all_managed())
        self.assertEqual(len(sids), 1)
        sid = sids[0]
        self.assertEqual(self.record(sid)["state"], "NEEDS_RECONCILIATION")
        self.assertIn(sid, self.registry.get_active_managed(), "an interrupted allocation keeps the cap")
        self.assertTrue(any("stop" in c for c in self.cli_cmds), "best-effort stop is attempted")
        second = grant_v2(mgr, spec, 900.0)
        with self.assertRaises(GlobalCapExceededError):
            mgr.up(spec, lease_seconds=900.0, approval_id=second)

    def test_new_timeout_keeps_record_needs_reconciliation_and_flags_orphan_risk(self):
        spec = self.make_spec()

        def timed_out(cmd):
            raise subprocess.TimeoutExpired(cmd, 180)

        run = self.scripted_run(timed_out)
        mgr = ProjectSessionManager(registry=self.registry, lock=self.lock, run_cmd_fn=run,
                                    spawn_fn=fake_guard_spawner(self), auth_mode="oauth2")
        approval_id = grant_v2(mgr, spec, 900.0)
        with self.assertRaises(AllocationError):
            mgr.up(spec, lease_seconds=900.0, approval_id=approval_id)
        sid = next(iter(self.registry.get_all_managed()))
        self.assertEqual(self.record(sid)["state"], "NEEDS_RECONCILIATION")
        self.assertTrue(self.record(sid)["metadata"]["orphan_possible"])
        self.assertIn(sid, self.registry.get_active_managed())
        self.assertFalse(any("stop" in c for c in self.cli_cmds), "timeout path leaves the stop to reconcile")

    def test_fail_closed_with_busy_lock_flags_needs_reconciliation_with_accurate_message(self):
        spec = self.make_spec()
        run = self.scripted_run(lambda cmd: subprocess.CompletedProcess(cmd, 0, "[colab] Session READY.", ""))

        def spawn_while_lock_is_held(cmd, **kwargs):
            # Another holder owns the project lock when the guard fails to start.
            self.assertTrue(self.lock.acquire(blocking=False))
            self.addCleanup(self.lock.release)
            raise OSError("spawn refused")

        mgr = ProjectSessionManager(registry=self.registry, lock=self.lock, run_cmd_fn=run,
                                    spawn_fn=spawn_while_lock_is_held, auth_mode="oauth2")
        approval_id = grant_v2(mgr, spec, 900.0)
        with mock.patch("colab_adapter.project_sessions.GUARD_STOP_LOCK_WAIT_S", 0.3):
            with self.assertRaises(AllocationError) as ctx:
                mgr.up(spec, lease_seconds=900.0, approval_id=approval_id)
        sid = next(iter(self.registry.get_all_managed()))
        self.assertEqual(self.record(sid)["state"], "NEEDS_RECONCILIATION")
        message = str(ctx.exception)
        self.assertIn("NEEDS_RECONCILIATION", message)
        self.assertIn("lock was busy", message)
        self.assertNotIn("stopped and verified", message)
        self.assertEqual([c for c in self.cli_cmds if "stop" in c], [], "busy lock: no stop was issued")


class TestStatusExitCodes(GuardFixture):
    def run_cli(self, argv):
        import contextlib
        import io
        from colab_adapter.cli import main

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(argv)
        return code, out.getvalue()

    def put_default(self, session):
        reg = ManagedSessionRegistry(Path(self.home) / ".colab_experiments" / "managed_sessions.json")
        with reg._lock():
            data = reg._read_locked()
            data[session.session_id] = session.to_dict()
            reg._atomic_write_locked(data)

    def test_status_exits_5_and_warns_for_needs_reconciliation(self):
        sid = "colab-v2-status-nr01"
        self.put_default(make_session(sid, expires_in=600, state="NEEDS_RECONCILIATION"))
        code, out = self.run_cli(["project", "status", "--session-id", sid])
        self.assertEqual(code, 5)
        self.assertIn("WARNING", out)
        self.assertIn("[?]", out)

    def test_status_exits_5_when_guard_is_dead_while_lease_is_active(self):
        sid = "colab-v2-status-gd001"
        self.put_default(make_session(sid, expires_in=600, state="READY"))
        code, out = self.run_cli(["project", "status", "--session-id", sid])
        self.assertEqual(code, 5)
        self.assertIn("guard", out.lower())

    def test_status_exits_0_for_terminated(self):
        sid = "colab-v2-status-ok001"
        self.put_default(make_session(sid, expires_in=600, state="TERMINATED"))
        code, _ = self.run_cli(["project", "status", "--session-id", sid])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
