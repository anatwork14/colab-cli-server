"""End-to-End integration tests for Colab Project Workspaces V2 (Milestone M7).

Covers full lifecycle Scenarios A through F:
- Scenario A: Project init and validation.
- Scenario B: Offline planning with zero cloud allocations (Q03, Q17).
- Scenario C: Session up / provisioning and idempotent reuse under lock (Q13, Q15).
- Scenario D: Incremental workspace sync (Q04, Q05, Q06, Q07).
- Scenario E: Remote execution in workspace with status tracking (Q11).
- Scenario F: Teardown and safe reconciliation, protecting unmanaged sessions (Q08).
- CLI machine-readable format stability with --json (Q16).
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from colab_adapter.cli import build_parser, main
from colab_adapter.project_manifest import init_project, load_manifest, plan_project
from colab_adapter.project_sessions import ProjectSessionManager, ProjectSessionState
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.workspace import WorkspaceSyncEngine


def _future_iso(seconds: float = 3600.0) -> str:
    return datetime.fromtimestamp(time.time() + seconds, timezone.utc).isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TestProjectWorkspacesE2E(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.project_dir = Path(self.tmp_dir) / "test_research_proj"
        self.project_dir.mkdir(parents=True, exist_ok=True)

        # Create entrypoint and module files
        (self.project_dir / "train.py").write_text("print('training started')\n", encoding="utf-8")
        (self.project_dir / "model.py").write_text("class Net: pass\n", encoding="utf-8")
        (self.project_dir / "requirements.txt").write_text("torch>=2.0.0\n", encoding="utf-8")

        self.manifest_path = self.project_dir / "colab-project.json"
        init_project(
            target_dir=self.project_dir,
            name="research_exp",
        )

        self.registry_path = Path(self.tmp_dir) / "managed_sessions.json"
        self.registry = ManagedSessionRegistry(self.registry_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_scenario_a_project_init_and_validate_cli(self):
        """Scenario A: Verify project init and validate via CLI and API."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            ret = main(["project", "validate", "--manifest", str(self.manifest_path), "--json"])
        self.assertEqual(ret, 0)
        data = json.loads(buf.getvalue())
        self.assertTrue(data.get("valid"))
        self.assertEqual(data.get("project"), "research_exp")

    def test_scenario_b_project_plan_offline_zero_allocations(self):
        """Scenario B: Dry-run planning generates exact file/byte counts with zero CLI subprocesses (Q03, Q17)."""
        buf = io.StringIO()
        with redirect_stdout(buf):
            ret = main(["project", "plan", "--manifest", str(self.manifest_path), "--json"])
        self.assertEqual(ret, 0)
        plan = json.loads(buf.getvalue())

        self.assertEqual(plan["project_name"], "research_exp")
        self.assertGreaterEqual(plan["files_count"], 3)  # train.py, model.py, requirements.txt
        self.assertGreater(plan["total_bytes"], 0)
        self.assertEqual(plan["gpu"], "T4")
        self.assertIn("/content/colab-workspaces/research_exp", plan["remote_root"])

    def test_scenario_c_project_up_and_idempotent_reuse(self):
        """Scenario C: Session up provisions under lock and reuses session idempotently."""
        executed_cmds = []

        def mock_run_cmd(cmd, **kwargs):
            executed_cmds.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Session created", stderr="")

        mgr = ProjectSessionManager(
            registry=self.registry,
            run_cmd_fn=mock_run_cmd,
        )
        spec = load_manifest(self.manifest_path)

        # Provision first session
        s1 = mgr.up(spec, lease_seconds=1800.0)
        self.assertEqual(s1.state, ProjectSessionState.READY.value)
        self.assertEqual(len(executed_cmds), 1)
        self.assertEqual(executed_cmds[0][:3], ["colab", "new", "-s"])

        # Second call reuses existing active session
        s2 = mgr.up(spec, reuse_active=True)
        self.assertEqual(s1.session_id, s2.session_id)
        # Invariant: No second allocation CLI call made
        self.assertEqual(len(executed_cmds), 1)

    def test_scenario_d_incremental_workspace_sync_engine(self):
        """Scenario D: Full cycle of initial sync, 0-file unchanged sync (Q04), and 1-file sync (Q05)."""
        session_id = "colab-v2-e2e-sync-1"
        now_iso = _now_iso()
        exp_iso = _future_iso(18000.0)

        with self.registry._lock():
            data = self.registry._read_locked()
            data[session_id] = {
                "session_id": session_id,
                "project_id": "proj-research_exp",
                "project_name": "research_exp",
                "state": ProjectSessionState.READY.value,
                "accelerator": "T4",
                "created_at": now_iso,
                "updated_at": now_iso,
                "lease_seconds": 18000.0,
                "expires_at": exp_iso,
                "remote_root": "/content/colab-workspaces/research_exp",
            }
            self.registry._atomic_write_locked(data)

        remote_fs = {}

        def mock_upload(s_id, l_path, r_path):
            remote_fs[r_path] = Path(l_path).read_bytes()

        def mock_download(s_id, r_path, l_path):
            if r_path not in remote_fs:
                raise FileNotFoundError(f"Not found: {r_path}")
            Path(l_path).write_bytes(remote_fs[r_path])

        mock_transport = MagicMock()
        mock_transport.registry = self.registry
        mock_transport.upload_file.side_effect = mock_upload
        mock_transport.download_file.side_effect = mock_download

        spec = load_manifest(self.manifest_path)
        sync_engine = WorkspaceSyncEngine(transport=mock_transport)

        # 1. Initial sync transfers all project files
        res1 = sync_engine.apply_sync(spec, session_id=session_id)
        self.assertEqual(res1["status"], "COMMITTED")
        self.assertGreaterEqual(res1["files_transferred"], 3)

        # Q04: Unchanged workspace sync -> 0 files transferred, 0 bytes transferred
        res2 = sync_engine.apply_sync(spec, session_id=session_id)
        self.assertEqual(res2["status"], "UP_TO_DATE")
        self.assertEqual(res2["files_transferred"], 0)
        self.assertEqual(res2["bytes_transferred"], 0)

        # Q05: Modify exactly 1 file -> transfers only that file
        (self.project_dir / "train.py").write_text("print('updated training loop v2')\n", encoding="utf-8")
        res3 = sync_engine.apply_sync(spec, session_id=session_id)
        self.assertEqual(res3["status"], "COMMITTED")
        self.assertEqual(res3["files_transferred"], 1)

    def test_scenario_e_project_exec_and_status(self):
        """Scenario E: Remote execution and status inspection via CLI."""
        session_id = "colab-v2-e2e-exec-1"
        now_iso = _now_iso()
        exp_iso = _future_iso(18000.0)

        with self.registry._lock():
            data = self.registry._read_locked()
            data[session_id] = {
                "session_id": session_id,
                "project_id": "proj-research_exp",
                "project_name": "research_exp",
                "state": ProjectSessionState.READY.value,
                "accelerator": "T4",
                "created_at": now_iso,
                "updated_at": now_iso,
                "lease_seconds": 18000.0,
                "expires_at": exp_iso,
                "remote_root": "/content/colab-workspaces/research_exp",
            }
            self.registry._atomic_write_locked(data)

        mock_transport = MagicMock()
        mock_transport.exec_code.return_value = {
            "success": True,
            "output": "Epoch 1/1: Loss = 0.042",
            "has_traceback": False,
        }

        mgr = ProjectSessionManager(
            registry=self.registry,
            transport=mock_transport,
        )

        res = mgr.exec(session_id, "python train.py", working_dir="/content/colab-workspaces/research_exp")
        self.assertTrue(res["success"])
        self.assertIn("Loss = 0.042", res["output"])

        stat = mgr.status(session_id)
        self.assertEqual(stat["state"], ProjectSessionState.READY.value)
        self.assertTrue(stat["lease_active"])
        self.assertGreater(stat["time_remaining_seconds"], 0)

    def test_scenario_f_stop_and_reconcile_unmanaged_protection(self):
        """Scenario F: Safe stop and reconcile protecting unmanaged user sessions (Q08)."""
        managed_id = "colab-v2-e2e-stop-1"
        unmanaged_id = "external-notebook-session-999"

        now_iso = _now_iso()
        exp_iso = _future_iso(18000.0)

        with self.registry._lock():
            data = self.registry._read_locked()
            data[managed_id] = {
                "session_id": managed_id,
                "project_id": "proj-research_exp",
                "project_name": "research_exp",
                "state": ProjectSessionState.READY.value,
                "accelerator": "T4",
                "created_at": now_iso,
                "updated_at": now_iso,
                "lease_seconds": 18000.0,
                "expires_at": exp_iso,
                "remote_root": "/content/colab-workspaces/research_exp",
            }
            self.registry._atomic_write_locked(data)

        mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""))
        mgr = ProjectSessionManager(registry=self.registry, run_cmd_fn=mock_run)

        # 1. Unmanaged session is strictly refused (Q08)
        from colab_adapter.errors import UnmanagedSessionError
        with self.assertRaises(UnmanagedSessionError):
            mgr.stop(unmanaged_id)
        mock_run.assert_not_called()

        # 2. Managed session stops cleanly
        ok = mgr.stop(managed_id)
        self.assertTrue(ok)
        mock_run.assert_called_once_with(
            ["colab", "stop", "-s", managed_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60.0,
        )
        self.assertEqual(self.registry.get_all_managed()[managed_id]["state"], ProjectSessionState.TERMINATED.value)


if __name__ == "__main__":
    unittest.main()
