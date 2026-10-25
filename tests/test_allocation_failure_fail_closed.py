"""PR #28 review P1: a failed `colab new` must never free the one-session cap without proof (offline).

The fake `colab` simulates upstream: `new` can assign a VM (an unnamed orphan, shown as `[?]` by `sessions`) and
still exit nonzero. No real Colab is contacted. A nonzero exit is not evidence that no VM was assigned.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from colab_adapter.approval import GlobalCapExceededError
from colab_adapter.errors import AllocationError
from colab_adapter.project_manifest import ProjectSpec
from colab_adapter.project_sessions import ProjectSessionManager, ProjectSessionState
from colab_adapter.provider import ColabProcessLock, ManagedSessionRegistry
from tests.approval_helpers import fake_guard_spawner, grant_v2
from tests.colab_fakes import fake_env, write_fake_colab

NR = ProjectSessionState.NEEDS_RECONCILIATION.value
FAILED = ProjectSessionState.FAILED.value


class FailClosedFixture(unittest.TestCase):
    NEW_MODE = "fail_before"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="alloc-fail-"))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.state = self.tmp / "state"
        (self.state / "alive").mkdir(parents=True)
        self.calls = self.tmp / "calls.jsonl"
        self.root = self.tmp / "project"
        self.root.mkdir()
        (self.root / "train.py").write_text("print('x')\n", encoding="utf-8")
        self.spec = ProjectSpec(name="failclosed", root=self.root, entrypoint="train.py", gpu="T4",
                                timeout_seconds=600.0)
        self.fake = write_fake_colab(self.tmp / "bin" / "colab")
        env = fake_env(self.state, self.calls, self.home)
        env["FAKE_COLAB_NEW_MODE"] = self.NEW_MODE
        patcher = mock.patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.registry = ManagedSessionRegistry(self.tmp / "registry" / "managed_sessions.json")
        self.manager = self.make_manager()

    def make_manager(self, **kwargs):
        mgr = ProjectSessionManager(
            registry=self.registry, lock=ColabProcessLock(self.tmp / "colab.lock"),
            colab_binary=kwargs.pop("colab_binary", str(self.fake)), auth_mode="oauth2",
            spawn_fn=fake_guard_spawner(self), **kwargs,
        )
        mgr.FAILURE_SETTLE_SECONDS = 0.0  # keep tests fast; the two-reading rule is still enforced
        return mgr

    def new_calls(self):
        import json
        if not self.calls.exists():
            return 0
        return sum(1 for line in self.calls.read_text(encoding="utf-8").splitlines()
                   if "new" in json.loads(line))

    def states(self):
        return [i["state"] for i in self.registry.get_all_managed().values()]

    def only_record(self):
        recs = list(self.registry.get_all_managed().values())
        self.assertEqual(len(recs), 1)
        return recs[0]

    def approval(self):
        return grant_v2(self.manager, self.spec, 900.0)

    def up_fails(self, approval_id):
        with self.assertRaises(AllocationError) as ctx:
            self.manager.up(self.spec, lease_seconds=900.0, approval_id=approval_id)
        return ctx.exception

    def assert_cap_blocks_a_fresh_approval(self):
        before = self.new_calls()
        fresh = self.approval()
        with self.assertRaises(GlobalCapExceededError):
            self.manager.up(self.spec, lease_seconds=900.0, approval_id=fresh)
        self.assertEqual(self.new_calls(), before, "a second `colab new` was issued")


class TestNonzeroExitAfterProviderAllocated(FailClosedFixture):
    NEW_MODE = "fail_after_orphan"

    def test_orphan_keeps_needs_reconciliation_cap_and_warns(self):
        err = self.up_fails(self.approval())
        rec = self.only_record()
        self.assertEqual(rec["state"], NR)
        self.assertNotEqual(rec["state"], FAILED)
        self.assertTrue(rec["metadata"]["orphan_possible"])
        self.assertIn("POSSIBLE ORPHAN", str(err))
        self.assertTrue(err.orphan_possible)
        self.assertEqual(err.details["session_id"], rec["session_id"])
        self.assertEqual(self.new_calls(), 1)
        self.assert_cap_blocks_a_fresh_approval()
        self.assertEqual(self.states(), [NR])

    def test_failed_reconciliation_leaves_record_and_cap_and_never_allocates(self):
        self.up_fails(self.approval())
        sid = next(iter(self.registry.get_all_managed()))
        report = self.manager.reconcile_leases(auto_stop=True)
        # upstream `stop` for a name it does not know exits 0 with "not found": must not count as stopped
        self.assertIn(sid, [f["session_id"] for f in report["failed"]])
        self.assertEqual(self.registry.get_all_managed()[sid]["state"], NR)
        from colab_adapter.project_sessions import StopNotFoundError
        with self.assertRaises(StopNotFoundError):
            self.manager.stop(sid)
        self.assertEqual(self.registry.get_all_managed()[sid]["state"], NR)
        self.assert_cap_blocks_a_fresh_approval()
        self.assertEqual(self.new_calls(), 1)


class TestNonzeroExitWithNamedSession(FailClosedFixture):
    NEW_MODE = "fail_after_named"

    def test_session_listed_after_failed_new_is_not_failed(self):
        self.up_fails(self.approval())
        self.assertEqual(self.only_record()["state"], NR)
        self.assert_cap_blocks_a_fresh_approval()


class TestUnusualOutput(FailClosedFixture):
    NEW_MODE = "ok_output_but_fail"

    def test_ready_text_with_nonzero_exit_and_orphan_is_uncertain(self):
        self.up_fails(self.approval())
        self.assertEqual(self.only_record()["state"], NR)
        self.assertEqual(self.new_calls(), 1)


class TestLaggingUsageIsNotProof(FailClosedFixture):
    NEW_MODE = "fail_after_orphan"

    def test_single_clean_reading_followed_by_orphan_is_not_failed(self):
        # baseline read consumes one lag, the first post-failure reading is hidden, the second reveals the orphan
        (self.state / "usage_lag_remaining").write_text("2", encoding="utf-8")
        # isolate the usage readings: the `[?]` line check would otherwise catch the orphan first
        self.manager._unnamed_sessions_now = lambda: 0
        self.up_fails(self.approval())
        rec = self.only_record()
        self.assertEqual(rec["state"], NR)
        self.assertIn("assignments", rec["metadata"]["allocation_failure_evidence"])


class TestProvenAbsenceMayFree(FailClosedFixture):
    NEW_MODE = "fail_before"

    def test_failure_before_assignment_with_two_clean_readings_is_failed(self):
        err = self.up_fails(self.approval())
        rec = self.only_record()
        self.assertEqual(rec["state"], FAILED)
        self.assertIn("no assignment remains", rec["metadata"]["allocation_failure_evidence"])
        self.assertNotIn("POSSIBLE ORPHAN", str(err))
        self.assertFalse(rec["metadata"].get("orphan_possible"))
        self.assertEqual(self.new_calls(), 1)
        # FAILED records do not count toward the cap, but a new allocation still needs a fresh owner approval
        from colab_adapter.approval import ensure_global_cap
        ensure_global_cap(self.registry)


class TestUnavailableUsageData(FailClosedFixture):
    NEW_MODE = "fail_before"

    def test_unknown_baseline_is_uncertain_even_if_nothing_was_assigned(self):
        with mock.patch.dict(os.environ, {"FAKE_COLAB_USAGE_FAIL": "1"}):
            self.up_fails(self.approval())
        rec = self.only_record()
        self.assertEqual(rec["state"], NR)
        self.assertIn("baseline", rec["metadata"]["allocation_failure_evidence"])
        self.assert_cap_blocks_a_fresh_approval()

    def test_unreadable_usage_after_failure_is_uncertain(self):
        with mock.patch.dict(os.environ, {"FAKE_COLAB_USAGE_FAIL_AFTER_NEW": "1"}):
            self.up_fails(self.approval())
        rec = self.only_record()
        self.assertEqual(rec["state"], NR)
        self.assertTrue(rec["metadata"]["orphan_possible"])
        self.assert_cap_blocks_a_fresh_approval()


class TestExceptionPaths(FailClosedFixture):
    NEW_MODE = "fail_after_orphan"

    def test_unexpected_exception_after_new_ran_is_uncertain(self):
        real = subprocess.run

        def run_then_boom(cmd, **kwargs):
            res = real(cmd, **kwargs)
            if "new" in cmd:
                raise RuntimeError("boom after new")
            return res

        mgr = self.make_manager(run_cmd_fn=run_then_boom)
        approval = grant_v2(mgr, self.spec, 900.0)
        with self.assertRaises(AllocationError):
            mgr.up(self.spec, lease_seconds=900.0, approval_id=approval)
        self.assertEqual(self.only_record()["state"], NR)
        self.assertEqual(self.new_calls(), 1)

    def test_missing_executable_is_certain_no_allocation(self):
        mgr = self.make_manager(colab_binary=str(self.tmp / "bin" / "does-not-exist"))
        approval = grant_v2(mgr, self.spec, 900.0)
        with self.assertRaises(FileNotFoundError):
            mgr.up(self.spec, lease_seconds=900.0, approval_id=approval)
        self.assertEqual(self.only_record()["state"], FAILED)
        self.assertEqual(self.new_calls(), 0)


class TestApprovalStaysSpent(FailClosedFixture):
    NEW_MODE = "fail_after_orphan"

    def test_same_approval_cannot_be_reused_after_uncertain_failure(self):
        approval = self.approval()
        self.up_fails(approval)
        from colab_adapter.provider import ApprovalRequiredError
        # the cap check runs first (NEEDS_RECONCILIATION counts), and the approval is spent either way
        with self.assertRaises((ApprovalRequiredError, GlobalCapExceededError)):
            self.manager.up(self.spec, lease_seconds=900.0, approval_id=approval)
        self.assertEqual(self.new_calls(), 1)
        with open(self.manager.approvals.directory / f"{approval}.json", encoding="utf-8") as fh:
            import json
            self.assertEqual(json.load(fh)["status"], "consumed")

class TestConcurrentUnmanagedRuntime(FailClosedFixture):
    """Opus M1: a runtime that ends while `new` runs must not make an orphan look like proof of absence."""
    NEW_MODE = "fail_after_orphan"

    def test_owner_runtime_ends_while_new_assigns_an_orphan_is_not_proof(self):
        from tests.colab_fakes import set_alive
        set_alive(self.state, "owner-runtime")  # baseline 1, named, not ours
        with mock.patch.dict(os.environ, {"FAKE_COLAB_REMOVE_ON_NEW": "owner-runtime"}):
            self.up_fails(self.approval())
        rec = self.only_record()
        # usage after == baseline (1) because owner's ended and the orphan appeared, but a `[?]` line exists
        self.assertEqual(rec["state"], NR)
        self.assertIn("unnamed", rec["metadata"]["allocation_failure_evidence"])
        self.assert_cap_blocks_a_fresh_approval()

    def test_count_below_baseline_after_failure_is_not_proof(self):
        self.NEW_MODE = "fail_before"
        from tests.colab_fakes import set_alive
        set_alive(self.state, "owner-runtime")
        with mock.patch.dict(os.environ, {"FAKE_COLAB_NEW_MODE": "fail_before",
                                          "FAKE_COLAB_REMOVE_ON_NEW": "owner-runtime"}):
            self.up_fails(self.approval())
        rec = self.only_record()
        self.assertEqual(rec["state"], NR)
        self.assertIn("differ from baseline", rec["metadata"]["allocation_failure_evidence"])

    def test_unreadable_sessions_before_allocation_means_no_proof_later(self):
        with mock.patch.dict(os.environ, {"FAKE_COLAB_NEW_MODE": "fail_before", "FAKE_COLAB_SESSIONS_FAIL": "1"}):
            self.up_fails(self.approval())
        self.assertEqual(self.only_record()["state"], NR)

    def test_pre_existing_unnamed_runtime_does_not_by_itself_block_proof(self):
        from tests.colab_fakes import set_alive
        set_alive(self.state, "orphan-from-the-colab-ui")  # shows as `[?]` both before and after
        with mock.patch.dict(os.environ, {"FAKE_COLAB_NEW_MODE": "fail_before"}):
            self.up_fails(self.approval())
        self.assertEqual(self.only_record()["state"], FAILED)


class TestInterruptsAndCommitFailure(FailClosedFixture):
    NEW_MODE = "fail_after_orphan"

    def test_keyboard_interrupt_during_evidence_check_keeps_needs_reconciliation(self):
        real = subprocess.run
        seen = {"new": False}

        def run(cmd, **kwargs):
            if "new" in cmd:
                seen["new"] = True
            elif seen["new"] and "usage" in cmd:
                raise KeyboardInterrupt()
            return real(cmd, **kwargs)

        mgr = self.make_manager(run_cmd_fn=run)
        approval = grant_v2(mgr, self.spec, 900.0)
        with self.assertRaises(AllocationError):
            mgr.up(self.spec, lease_seconds=900.0, approval_id=approval)
        rec = self.only_record()
        self.assertEqual(rec["state"], NR)
        self.assertEqual(self.new_calls(), 1)

    def test_keyboard_interrupt_during_new_keeps_needs_reconciliation_and_reraises(self):
        real = subprocess.run

        def run(cmd, **kwargs):
            if "new" in cmd:
                real(cmd, **kwargs)  # the provider assigned the VM
                raise KeyboardInterrupt()
            return real(cmd, **kwargs)

        mgr = self.make_manager(run_cmd_fn=run)
        approval = grant_v2(mgr, self.spec, 900.0)
        with self.assertRaises(KeyboardInterrupt):
            mgr.up(self.spec, lease_seconds=900.0, approval_id=approval)
        self.assertEqual(self.only_record()["state"], NR)
        self.assertTrue(self.only_record()["metadata"]["orphan_possible"])

    def test_ready_commit_failure_flags_needs_reconciliation(self):
        with mock.patch.dict(os.environ, {"FAKE_COLAB_NEW_MODE": "ok"}):
            mgr = self.make_manager()
            approval = grant_v2(mgr, self.spec, 900.0)
            real_commit = mgr._commit_session
            calls = {"n": 0}

            def flaky(session):
                calls["n"] += 1
                if session.state == ProjectSessionState.READY.value:
                    raise OSError("registry write failed")
                return real_commit(session)

            with mock.patch.object(mgr, "_commit_session", side_effect=flaky):
                with self.assertRaises(OSError):
                    mgr.up(self.spec, lease_seconds=900.0, approval_id=approval)
            rec = self.only_record()
            self.assertEqual(rec["state"], NR)
            self.assertTrue(rec["metadata"]["orphan_possible"])


class TestCliExitCodesAndWarning(FailClosedFixture):
    NEW_MODE = "fail_after_orphan"

    def run_cli(self, *argv):
        import contextlib
        import io
        from colab_adapter import cli
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def make_cli_env(self, new_mode):
        import json as _json
        proj = self.tmp / "cliproj"
        proj.mkdir()
        (proj / "train.py").write_text("print(1)\n", encoding="utf-8")
        (proj / "requirements.txt").write_text("", encoding="utf-8")
        manifest = proj / "colab-project.json"
        manifest.write_text(_json.dumps({
            "schema_version": 1, "name": "cliproj", "execution": {"entrypoint": "train.py"},
            "runtime": {"gpu": "T4"},
        }), encoding="utf-8")
        env = {"PATH": str(self.fake.parent) + os.pathsep + os.environ.get("PATH", ""),
               "FAKE_COLAB_NEW_MODE": new_mode}
        return manifest, env

    def _approved_id_for(self, manifest):
        from colab_adapter.project_manifest import load_manifest
        spec = load_manifest(manifest)
        mgr = ProjectSessionManager(spawn_fn=fake_guard_spawner(self))  # default registry under the patched HOME
        return grant_v2(mgr, spec, 900.0)

    def test_uncertain_allocation_exits_6_with_orphan_warning_json(self):
        import json as _json
        manifest, env = self.make_cli_env("fail_after_orphan")
        with mock.patch.dict(os.environ, env):
            approval = self._approved_id_for(manifest)
            code, _, err = self.run_cli("project", "up", "--manifest", str(manifest), "--lease-seconds", "900",
                                        "--approval-id", approval, "--json")
        self.assertEqual(code, 6)
        payload = _json.loads(err)
        self.assertTrue(payload["orphan_possible"])
        self.assertIn("POSSIBLE ORPHAN", payload["error"])
        self.assertTrue(payload["session_id"].startswith("colab-v2-"))

    def test_proven_failure_exits_1_without_orphan_flag(self):
        import json as _json
        manifest, env = self.make_cli_env("fail_before")
        with mock.patch.dict(os.environ, env), mock.patch(
                "colab_adapter.project_sessions.ProjectSessionManager.FAILURE_SETTLE_SECONDS", 0.0):
            approval = self._approved_id_for(manifest)
            code, _, err = self.run_cli("project", "up", "--manifest", str(manifest), "--lease-seconds", "900",
                                        "--approval-id", approval, "--json")
        self.assertEqual(code, 1)
        self.assertFalse(_json.loads(err)["orphan_possible"])

    def test_keyboard_interrupt_during_up_prints_orphan_warning_and_exits_6(self):
        manifest, env = self.make_cli_env("ok")
        with mock.patch.dict(os.environ, env), mock.patch(
                "colab_adapter.project_sessions.ProjectSessionManager.up", side_effect=KeyboardInterrupt()):
            code, _, err = self.run_cli("project", "up", "--manifest", str(manifest), "--lease-seconds", "900",
                                        "--approval-id", "apr-" + "0" * 32)
        self.assertEqual(code, 6)
        self.assertIn("POSSIBLE ORPHAN", err)


if __name__ == "__main__":
    unittest.main()
