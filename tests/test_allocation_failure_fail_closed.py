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

if __name__ == "__main__":
    unittest.main()
