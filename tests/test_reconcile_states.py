"""Offline tests for `reconcile_leases` covering NEEDS_RECONCILIATION, STOPPING and stale PROVISIONING (F2).

Only managed ids are ever touched. The fake `colab` keeps server-side assignments in a directory; no
real colab binary is run.
"""
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from colab_adapter.project_sessions import ProjectSession, ProjectSessionManager
from colab_adapter.provider import ColabProcessLock, ManagedSessionRegistry
from colab_adapter.transports.official_cli import OfficialCliTransport
from tests.approval_helpers import fake_guard_spawner
from tests.colab_fakes import fake_env, is_alive, set_alive, write_fake_colab

NR = "NEEDS_RECONCILIATION"


def _ago(seconds):
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def _ahead(seconds):
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


class ReconcileFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="reconcile-states-"))
        self.addCleanup(shutil.rmtree, str(self.tmp), True)
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.state_dir = self.tmp / "state"
        (self.state_dir / "alive").mkdir(parents=True)
        self.calls_file = self.tmp / "calls.jsonl"
        self.fake = write_fake_colab(self.tmp / "bin" / "colab")
        patcher = mock.patch.dict(os.environ, fake_env(self.state_dir, self.calls_file, self.home))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.registry = ManagedSessionRegistry(self.tmp / "registry" / "managed_sessions.json")
        self.lock = ColabProcessLock(self.tmp / "colab.lock")

    def put_raw(self, sid, state, updated_at=None, expires_at=None, metadata=None):
        now = datetime.now(timezone.utc).isoformat()
        rec = ProjectSession(
            session_id=sid, project_id="proj-rec", project_name="rec", state=state, accelerator="T4",
            created_at=now, updated_at=updated_at or now, lease_seconds=900.0,
            expires_at=expires_at or _ahead(900), remote_root="/content/rec",
            approved_fingerprint="fp", approval_id="apr", metadata=dict(metadata or {}),
        ).to_dict()
        with self.registry._lock():
            data = self.registry._read_locked()
            data[sid] = rec
            self.registry._atomic_write_locked(data)

    def record(self, sid):
        return self.registry.get_all_managed()[sid]

    def stop_calls(self):
        import json
        if not self.calls_file.exists():
            return []
        out = []
        for line in self.calls_file.read_text(encoding="utf-8").splitlines():
            args = json.loads(line)
            if "stop" in args:
                out.append(args[args.index("-s") + 1])
        return out

    def manager(self):
        transport = OfficialCliTransport(registry=self.registry, cli_path=str(self.fake), auth_mode="oauth2")
        return ProjectSessionManager(registry=self.registry, lock=self.lock, transport=transport,
                                     colab_binary=str(self.fake), auth_mode="oauth2",
                                     spawn_fn=fake_guard_spawner(self))


class TestReconcileCategories(ReconcileFixture):
    def test_needs_reconciliation_and_stopping_managed_sessions_are_stopped_and_verified(self):
        nr = "colab-v2-rec-nr0001"
        stopping = "colab-v2-rec-stp001"
        self.put_raw(nr, NR)
        self.put_raw(stopping, "STOPPING")
        set_alive(self.state_dir, nr)
        set_alive(self.state_dir, stopping)
        report = self.manager().reconcile_leases(auto_stop=True)
        self.assertEqual(sorted(report["stopped"]), sorted([nr, stopping]))
        self.assertEqual(report["failed"], [])
        self.assertEqual(self.record(nr)["state"], "TERMINATED")
        self.assertEqual(self.record(stopping)["state"], "TERMINATED")
        self.assertFalse(is_alive(self.state_dir, nr))
        self.assertFalse(is_alive(self.state_dir, stopping))

    def test_stale_provisioning_is_reconciled_but_fresh_provisioning_is_left_alone(self):
        stale = "colab-v2-rec-stale1"
        fresh = "colab-v2-rec-fresh1"
        self.put_raw(stale, "PROVISIONING", updated_at=_ago(900))
        self.put_raw(fresh, "PROVISIONING", updated_at=_ago(10))
        set_alive(self.state_dir, stale)
        set_alive(self.state_dir, fresh)
        report = self.manager().reconcile_leases(auto_stop=True)
        self.assertEqual(report["stopped"], [stale])
        self.assertIn(fresh, report["active"])
        self.assertNotIn(fresh, self.stop_calls())
        self.assertEqual(self.record(fresh)["state"], "PROVISIONING")
        self.assertEqual(self.record(stale)["state"], "TERMINATED")

    def test_unmanaged_session_is_never_touched(self):
        managed = "colab-v2-rec-mgd001"
        unmanaged = "colab-v2-foreign-9999"
        self.put_raw(managed, NR)
        set_alive(self.state_dir, managed)
        set_alive(self.state_dir, unmanaged)
        self.manager().reconcile_leases(auto_stop=True)
        self.assertNotIn(unmanaged, self.stop_calls())
        self.assertTrue(is_alive(self.state_dir, unmanaged))
        self.assertNotIn(unmanaged, self.registry.get_all_managed())

    def test_no_auto_stop_reports_needs_attention_and_stops_nothing(self):
        nr = "colab-v2-rec-noauto1"
        self.put_raw(nr, NR)
        set_alive(self.state_dir, nr)
        report = self.manager().reconcile_leases(auto_stop=False)
        self.assertEqual(self.stop_calls(), [])
        self.assertEqual(report["stopped"], [])
        self.assertIn(nr, [item["session_id"] for item in report["needs_attention"]])
        self.assertEqual(self.record(nr)["state"], NR)

    def test_not_found_stop_leaves_needs_reconciliation_and_warns_about_orphans(self):
        nr = "colab-v2-rec-orphan1"
        self.put_raw(nr, NR)  # no VM registered under this name
        report = self.manager().reconcile_leases(auto_stop=True)
        self.assertEqual(report["stopped"], [])
        self.assertEqual([f["session_id"] for f in report["failed"]], [nr])
        self.assertEqual(self.record(nr)["state"], NR)
        self.assertNotEqual(self.record(nr)["state"], "TERMINATED")
        joined = " ".join(report["warnings"])
        self.assertIn("orphan", joined.lower())
        self.assertIn("[?]", joined)
        self.assertIn("usage", joined)

    def test_expired_ready_without_auto_stop_is_needs_attention_not_active(self):
        ready = "colab-v2-rec-expired"
        self.put_raw(ready, "READY", expires_at=_ago(60))
        set_alive(self.state_dir, ready)
        report = self.manager().reconcile_leases(auto_stop=False)
        self.assertNotIn(ready, report["active"])
        self.assertIn(ready, [item["session_id"] for item in report["needs_attention"]])
        self.assertEqual(self.stop_calls(), [])

    def test_report_always_has_all_categories(self):
        report = self.manager().reconcile_leases(auto_stop=True)
        for key in ("stopped", "failed", "active", "needs_attention", "warnings"):
            self.assertIn(key, report)

    def test_terminal_and_planned_records_are_ignored(self):
        self.put_raw("colab-v2-rec-term01", "TERMINATED")
        self.put_raw("colab-v2-rec-fail01", "FAILED")
        report = self.manager().reconcile_leases(auto_stop=True)
        self.assertEqual(self.stop_calls(), [])
        self.assertEqual(report["stopped"], [])
        self.assertEqual(report["failed"], [])


if __name__ == "__main__":
    unittest.main()
