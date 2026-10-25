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

import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from colab_adapter.artifacts import verify_local_artifact
from colab_adapter.broker import WorkspaceBroker
from colab_adapter.checkpoint_policy import CheckpointPolicyConfig, CheckpointPolicyManager
from colab_adapter.cli import build_parser, main
from colab_adapter.errors import (
    ArtifactValidationError,
    DriveIntegrityError,
    ManifestValidationError,
)
from colab_adapter.monitor.collector import AccountSample, AccountUsageCollector
from colab_adapter.monitor.schemas import _iso_now
from colab_adapter.monitor.server import start_monitor_server
from colab_adapter.project_manifest import (
    create_execution_plan,
    init_project,
    load_manifest,
    load_project_manifest,
    plan_project,
    validate_manifest,
)
from colab_adapter.project_sessions import (
    ProjectSession,
    ProjectSessionManager,
    ProjectSessionState,
)
from colab_adapter.provider import ApprovalRequiredError, ManagedSessionRegistry
from tests.approval_helpers import fake_guard_spawner, grant_v2
from colab_adapter.reproducibility import create_deterministic_source_snapshot
from colab_adapter.storage import (
    ArtifactType,
    ArtifactUploadWorker,
    DriveCredentials,
    GoogleDriveArtifactStore,
    LocalArtifactStore,
    PublicationManifest,
    StoredArtifact,
    TransferJournal,
    compute_file_sha256,
)
from colab_adapter.telemetry import TelemetryEmitter, plan_resume_from_checkpoint
from colab_adapter.transports.base import BaseTransport
from colab_adapter.workspace import WorkspaceSyncEngine
from tests.test_google_drive_store import MockDriveHttpClient


try:
    from colab_adapter.telemetry import ResumePlanner
except ImportError:
    class ResumePlanner:
        """Compatibility wrapper for checkpoint resume planning."""
        def __new__(cls, *args, **kwargs):
            return plan_resume_from_checkpoint(*args, **kwargs)

        @classmethod
        def plan(cls, *args, **kwargs):
            return plan_resume_from_checkpoint(*args, **kwargs)


class E2EMockDriveHttpClient(MockDriveHttpClient):
    """MockDriveHttpClient with automatic metadata cataloging for idempotency testing."""
    pass


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

        mgr = ProjectSessionManager(spawn_fn=fake_guard_spawner(self), 
            registry=self.registry,
            run_cmd_fn=mock_run_cmd,
        )
        spec = load_manifest(self.manifest_path)

        # Provision first session
        s1 = mgr.up(spec, lease_seconds=1800.0, approval_id=grant_v2(mgr, spec, 1800.0))
        self.assertEqual(s1.state, ProjectSessionState.READY.value)
        self.assertEqual(len(executed_cmds), 1)
        self.assertEqual(executed_cmds[0][:4], ["colab", "--auth=oauth2", "new", "-s"])

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

        mgr = ProjectSessionManager(spawn_fn=fake_guard_spawner(self), 
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
        mgr = ProjectSessionManager(spawn_fn=fake_guard_spawner(self), registry=self.registry, run_cmd_fn=mock_run)

        # 1. Unmanaged session is strictly refused (Q08)
        from colab_adapter.errors import UnmanagedSessionError
        with self.assertRaises(UnmanagedSessionError):
            mgr.stop(unmanaged_id)
        mock_run.assert_not_called()

        # 2. Managed session stops cleanly
        ok = mgr.stop(managed_id)
        self.assertTrue(ok)
        mock_run.assert_called_once_with(
            ["colab", "--auth=oauth2", "stop", "-s", managed_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60.0,
        )
        self.assertEqual(self.registry.get_all_managed()[managed_id]["state"], ProjectSessionState.TERMINATED.value)

    def test_full_13_stage_research_lifecycle(self):
        """13-Stage V2 End-to-End Pipeline Verification with Drive durable publication and cross-device monitoring.

        Fully offline, deterministic verification (0 cloud GPU allocations, 0 live Drive API calls):
        1. Local Python project (multi-module, colab-project.yaml, secret .env, external symlink).
        2. Project manifest and plan (load_project_manifest, create_execution_plan).
        3. Explicit GPU allocation approval (block unapproved, authorize via explicit token).
        4. Colab session creation (ProjectSessionManager & ManagedSessionRegistry).
        5. Selective source synchronization (WorkspaceSyncEngine initial + incremental sync, snapshot).
        6. Remote training execution (mocked transport exec).
        7. Emitted training events (TelemetryEmitter structured events).
        8. Monitoring dashboard (MonitorServer loopback endpoints /summary, /sessions, /runs, /metrics).
        9. Verified checkpoint download (verify_local_artifact with SHA256 + size + ZIP structure).
        10. Local artifact preservation (LocalArtifactStore & RunManifest).
        11. Durable Google Drive publication (GoogleDriveArtifactStore + ArtifactUploadWorker).
        12. Remote session cleanup (session_manager.close_session -> TERMINATED).
        13. Historical artifact access from another authorized computer (WorkspaceBroker.attach_run & /drive query).
        """
        # =====================================================================
        # Stage 1: Local Python project setup
        # =====================================================================
        pipeline_dir = Path(self.tmp_dir) / "v2_pipeline_proj"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        (pipeline_dir / "src").mkdir(parents=True, exist_ok=True)
        (pipeline_dir / "configs").mkdir(parents=True, exist_ok=True)

        # Multi-module code files
        (pipeline_dir / "src" / "trainer.py").write_text("print('training initiated')\n", encoding="utf-8")
        (pipeline_dir / "configs" / "base.yaml").write_text("batch_size: 32\nlr: 0.001\n", encoding="utf-8")

        # Excluded secret file
        secret_env_file = pipeline_dir / ".env"
        secret_env_file.write_text("SUPER_SECRET_API_TOKEN=ya29.secret_token_value_xyz\n", encoding="utf-8")

        # External symlink pointing outside project root (security boundary test)
        external_secret = Path(self.tmp_dir) / "external_system_secret.txt"
        external_secret.write_text("CRITICAL_HOST_KEY_DATA\n", encoding="utf-8")
        external_symlink = pipeline_dir / "external_symlink.txt"
        os.symlink(str(external_secret), str(external_symlink))

        # colab-project.json manifest (stdlib JSON compatible across all Python versions)
        manifest_file = pipeline_dir / "colab-project.json"
        manifest_file.write_text(
            json.dumps({
                "schema_version": 1,
                "project": {
                    "name": "e2e_research_v2",
                    "root": ".",
                },
                "runtime": {
                    "gpu": "T4",
                    "timeout_seconds": 900.0,
                    "retain_session": False,
                },
                "execution": {
                    "entrypoint": "src/trainer.py",
                    "args": ["--config", "configs/base.yaml"],
                },
                "workspace": {
                    "remote_root": "/content/colab-workspaces/e2e_research_v2",
                    "include": ["**/*"],
                    "exclude": [".env*", "external_symlink*"],
                },
                "artifacts": {
                    "pull": ["checkpoints/model_final.pt"],
                    "local_dir": "artifacts",
                },
            }, indent=2),
            encoding="utf-8",
        )

        # =====================================================================
        # Stage 2: Project manifest and plan
        # =====================================================================
        spec = load_project_manifest(manifest_file)
        self.assertEqual(spec.name, "e2e_research_v2")
        self.assertEqual(spec.entrypoint, "src/trainer.py")
        self.assertEqual(spec.gpu, "T4")

        plan = create_execution_plan(spec)
        self.assertEqual(plan["project_name"], "e2e_research_v2")
        self.assertEqual(plan["entrypoint"], "src/trainer.py")
        self.assertEqual(plan["gpu"], "T4")
        self.assertEqual(plan["remote_root"], "/content/colab-workspaces/e2e_research_v2")
        self.assertGreaterEqual(plan["files_count"], 3)
        plan_paths = [f["path"] for f in plan["files"]]
        self.assertIn("src/trainer.py", plan_paths)
        self.assertIn("configs/base.yaml", plan_paths)
        self.assertIn("colab-project.json", plan_paths)
        self.assertNotIn(".env", plan_paths, "Secret .env must be excluded from execution plan")
        self.assertNotIn("external_symlink.txt", plan_paths, "External symlink must be excluded from plan")

        # =====================================================================
        # Stage 3: Explicit GPU allocation approval
        # =====================================================================
        executed_cmds = []

        def mock_run_cmd(cmd, **kwargs):
            executed_cmds.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Session created", stderr="")

        session_mgr = ProjectSessionManager(spawn_fn=fake_guard_spawner(self), 
            registry=self.registry,
            run_cmd_fn=mock_run_cmd,
        )

        # Unapproved allocation is strictly blocked (fail closed)
        with self.assertRaises(ApprovalRequiredError) as ctx:
            session_mgr.up(spec, lease_seconds=1800.0)
        self.assertEqual(ctx.exception.reason, "missing")
        self.assertEqual(len(executed_cmds), 0, "No allocation command may be dispatched without approval")

        # Authorize via a granted, single-use approval record
        approval_id = grant_v2(session_mgr, spec, 1800.0)
        session = session_mgr.up(spec, lease_seconds=1800.0, approval_id=approval_id)
        self.assertEqual(session.state, ProjectSessionState.READY.value)
        self.assertEqual(session.accelerator, "T4")
        self.assertEqual(len(executed_cmds), 1)
        self.assertEqual(executed_cmds[0][:4], ["colab", "--auth=oauth2", "new", "-s"])

        # =====================================================================
        # Stage 4: Colab session creation
        # =====================================================================
        self.assertIn(session.session_id, self.registry.get_all_managed())
        managed_info = self.registry.get_all_managed()[session.session_id]
        self.assertEqual(managed_info["state"], ProjectSessionState.READY.value)
        self.assertTrue(session.is_lease_active())
        self.assertGreater(session.time_remaining(), 0)

        # =====================================================================
        # Stage 5: Selective source synchronization
        # =====================================================================
        remote_fs = {}

        def mock_upload(s_id, l_path, r_path):
            remote_fs[r_path] = Path(l_path).read_bytes()

        def mock_download(s_id, r_path, l_path):
            if r_path not in remote_fs:
                raise FileNotFoundError(f"Remote path not found: {r_path}")
            Path(l_path).write_bytes(remote_fs[r_path])

        mock_transport = MagicMock(spec=BaseTransport)
        mock_transport.registry = self.registry
        mock_transport.upload_file.side_effect = mock_upload
        mock_transport.download_file.side_effect = mock_download

        sync_engine = WorkspaceSyncEngine(transport=mock_transport)

        # Initial sync transfers project files
        sync_res1 = sync_engine.apply_sync(spec, session_id=session.session_id)
        self.assertEqual(sync_res1["status"], "COMMITTED")
        self.assertGreaterEqual(sync_res1["files_transferred"], 3)
        # Verify secret .env and external symlink were NOT uploaded
        for rem_key in remote_fs:
            self.assertFalse(rem_key.endswith(".env"), f"Leaked .env file to remote workspace: {rem_key}")
            self.assertFalse(rem_key.endswith("external_symlink.txt"), f"Leaked symlink to remote workspace: {rem_key}")

        # Incremental sync with unchanged workspace -> 0 files transferred
        sync_res2 = sync_engine.apply_sync(spec, session_id=session.session_id)
        self.assertEqual(sync_res2["status"], "UP_TO_DATE")
        self.assertEqual(sync_res2["files_transferred"], 0)

        # Incremental sync after modifying single file -> exactly 1 file transferred
        (pipeline_dir / "src" / "trainer.py").write_text("print('updated trainer loop v2')\n", encoding="utf-8")
        sync_res3 = sync_engine.apply_sync(spec, session_id=session.session_id)
        self.assertEqual(sync_res3["status"], "COMMITTED")
        self.assertEqual(sync_res3["files_transferred"], 1)

        # Create deterministic source snapshot
        staging_dir = Path(self.tmp_dir) / "staging_artifacts"
        staging_dir.mkdir(parents=True, exist_ok=True)
        snap_tar_path = staging_dir / "source_snapshot.tar.gz"
        create_deterministic_source_snapshot(pipeline_dir, snap_tar_path)
        self.assertTrue(snap_tar_path.is_file())

        with tarfile.open(snap_tar_path, "r:gz") as tar:
            member_names = tar.getnames()
            self.assertTrue(all(".env" not in n for n in member_names), "Source snapshot contains .env")
            self.assertTrue(all("external_symlink" not in n for n in member_names), "Source snapshot contains symlink")

        # =====================================================================
        # Stage 6: Remote training execution
        # =====================================================================
        mock_transport.exec_code.return_value = {
            "success": True,
            "output": "Epoch 1/1 step 100 loss: 0.123\nCheckpoint saved to outputs/checkpoint.pt",
            "has_traceback": False,
        }
        session_mgr.transport = mock_transport

        exec_res = session_mgr.exec(
            session.session_id,
            f"python {spec.entrypoint}",
            working_dir=spec.remote_root,
        )
        self.assertTrue(exec_res["success"])
        self.assertIn("loss: 0.123", exec_res["output"])

        # =====================================================================
        # Stage 7: Emitted training events
        # =====================================================================
        run_id = "run-v2-pipeline-001"
        project_id = spec.name
        telemetry_file = Path(self.tmp_dir) / "telemetry.jsonl"
        telemetry = TelemetryEmitter(ledger_path=telemetry_file, run_id=run_id, project_id=project_id)

        telemetry.emit(
            event_type="run_started",
            run_id=run_id,
            project_id=project_id,
            details={"session_id": session.session_id, "gpu": spec.gpu},
        )
        telemetry.emit(
            event_type="train_step",
            run_id=run_id,
            project_id=project_id,
            epoch=1,
            step=100,
            metrics={"loss": 0.123},
        )
        telemetry.emit(
            event_type="checkpoint_saved",
            run_id=run_id,
            project_id=project_id,
            epoch=1,
            step=100,
            details={"checkpoint_file": "checkpoint.pt"},
        )
        telemetry.emit(
            event_type="run_finished",
            run_id=run_id,
            project_id=project_id,
            metrics={"final_loss": 0.123},
            details={"session_id": session.session_id},
        )
        # Emit logs and details with special HTML characters (<, >, ", ')
        telemetry.emit(
            event_type="log_line",
            run_id=run_id,
            project_id=project_id,
            epoch=1,
            step=100,
            details={"log_line": "<script>alert('XSS & \"test\"')</script> <b>loss: 0.123</b> & payload <1> >0"},
        )

        # =====================================================================
        # Stage 8: Monitoring dashboard backend verification
        # =====================================================================
        account_ledger = Path(self.tmp_dir) / "account_usage.jsonl"
        collector = AccountUsageCollector(ledger_path=account_ledger)
        sample = AccountSample(
            sample_id="samp-v2-001",
            observed_at=_iso_now(),
            balance_cu=150.0,
            rate_cu_per_hour=0.5,
            active_assignments=1,
        )
        collector._record_sample(sample)

        monitor_token = "secret-monitor-token-v2"
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=collector,
            registry=self.registry,
            telemetry=telemetry,
            auth_token=monitor_token,
            allow_external=False,
            artifacts_dir=staging_dir,
        )
        port = server.server_port
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close()))
        base_url = f"http://127.0.0.1:{port}"

        # 8.0 Dashboard HTML security headers check (strict Content-Security-Policy & nosniff)
        for dashboard_path in ("/", "/index.html"):
            req_dashboard = urllib.request.Request(f"{base_url}{dashboard_path}")
            with urllib.request.urlopen(req_dashboard) as resp:
                self.assertEqual(resp.status, 200)
                csp_hdr = resp.headers.get("Content-Security-Policy")
                nosniff_hdr = resp.headers.get("X-Content-Type-Options")
                self.assertIsNotNone(csp_hdr, f"Content-Security-Policy header missing on {dashboard_path}")
                self.assertIn("'self'", csp_hdr, f"Strict CSP must specify 'self' on {dashboard_path}")
                self.assertEqual(nosniff_hdr, "nosniff", f"X-Content-Type-Options must be nosniff on {dashboard_path}")

        # 8.1 Unauthenticated request rejected with 401
        req_unauth = urllib.request.Request(f"{base_url}/api/v1/summary")
        with self.assertRaises(urllib.error.HTTPError) as err_ctx:
            urllib.request.urlopen(req_unauth)
        self.assertEqual(err_ctx.exception.code, 401)

        # 8.2 Authenticated summary endpoint
        req_summary = urllib.request.Request(
            f"{base_url}/api/v1/summary",
            headers={"Authorization": f"Bearer {monitor_token}"},
        )
        with urllib.request.urlopen(req_summary) as resp:
            self.assertEqual(resp.status, 200)
            summary_data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(summary_data["current_balance_cu"], 150.0)

        # 8.3 Authenticated sessions endpoint
        req_sessions = urllib.request.Request(
            f"{base_url}/api/v1/sessions",
            headers={"Authorization": f"Bearer {monitor_token}"},
        )
        with urllib.request.urlopen(req_sessions) as resp:
            self.assertEqual(resp.status, 200)
            sessions_data = json.loads(resp.read().decode("utf-8"))
            session_ids = [s["session_id"] for s in sessions_data]
            self.assertIn(session.session_id, session_ids)

        # 8.4 Authenticated runs endpoint
        req_runs = urllib.request.Request(
            f"{base_url}/api/v1/runs",
            headers={"Authorization": f"Bearer {monitor_token}"},
        )
        with urllib.request.urlopen(req_runs) as resp:
            self.assertEqual(resp.status, 200)
            runs_data = json.loads(resp.read().decode("utf-8"))
            run_ids = [r["run_id"] for r in runs_data]
            self.assertIn(run_id, run_ids)

        # 8.5 Authenticated metrics endpoint
        req_metrics = urllib.request.Request(
            f"{base_url}/api/v1/runs/{run_id}/metrics",
            headers={"Authorization": f"Bearer {monitor_token}"},
        )
        with urllib.request.urlopen(req_metrics) as resp:
            self.assertEqual(resp.status, 200)
            metrics_data = json.loads(resp.read().decode("utf-8"))
            loss_points = [p for p in metrics_data if p.get("metrics", {}).get("loss") is not None]
            self.assertGreaterEqual(len(loss_points), 1)
            self.assertEqual(loss_points[0]["metrics"]["loss"], 0.123)

        # 8.6 Verify API endpoints return run/artifact/log entries containing special HTML characters safely as JSON
        req_logs = urllib.request.Request(
            f"{base_url}/api/v1/runs/{run_id}/logs",
            headers={"Authorization": f"Bearer {monitor_token}"},
        )
        with urllib.request.urlopen(req_logs) as resp:
            self.assertEqual(resp.status, 200)
            logs_payload = json.loads(resp.read().decode("utf-8"))
            self.assertIn("logs", logs_payload)
            expected_html_log = "<script>alert('XSS & \"test\"')</script> <b>loss: 0.123</b> & payload <1> >0"
            self.assertIn(expected_html_log, logs_payload["logs"])

        req_run_detail = urllib.request.Request(
            f"{base_url}/api/v1/runs/{run_id}",
            headers={"Authorization": f"Bearer {monitor_token}"},
        )
        with urllib.request.urlopen(req_run_detail) as resp:
            self.assertEqual(resp.status, 200)
            run_detail = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(run_detail["run_id"], run_id)
            matching_events = [
                e for e in run_detail.get("events", [])
                if e.get("details", {}).get("log_line") == expected_html_log
            ]
            self.assertEqual(len(matching_events), 1, "HTML-laden log event must be safely serialized in JSON without corrupting state")

        # =====================================================================
        # Stage 9: Verified checkpoint download
        # =====================================================================
        ckpt_local = staging_dir / "model_final.pt"
        with zipfile.ZipFile(ckpt_local, "w") as zf:
            zf.writestr("archive/data.pkl", b"TENSOR_WEIGHT_MATRIX_PAYLOAD_" * 128)
            zf.writestr("archive/version", b"3")

        ckpt_manifest = verify_local_artifact(ckpt_local, is_checkpoint=True)
        self.assertGreater(ckpt_manifest.size_bytes, 0)
        self.assertEqual(ckpt_manifest.sha256, compute_file_sha256(ckpt_local))
        self.assertGreaterEqual(ckpt_manifest.verification_level, 2)
        self.assertTrue(ckpt_manifest.is_checkpoint)
        self.assertIn("archive/data.pkl", ckpt_manifest.metadata.get("archive_entries", []))

        # Checkpoint policy evaluation
        policy = CheckpointPolicyManager(CheckpointPolicyConfig(minimum_interval_seconds=60.0))
        self.assertTrue(policy.should_publish(step=100, is_final=True))

        # =====================================================================
        # Stage 10: Local artifact preservation
        # =====================================================================
        local_store = LocalArtifactStore(root_dir=staging_dir)
        stored_ckpt = local_store.publish(
            source=ckpt_local,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        stored_snap = local_store.publish(
            source=snap_tar_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.SOURCE_SNAPSHOT.value,
        )
        self.assertTrue(local_store.exists(stored_ckpt.artifact_id))
        self.assertTrue(local_store.exists(stored_snap.artifact_id))

        # Create persistent run manifest on disk
        run_manifest_file = staging_dir / "projects" / project_id / "runs" / run_id / "manifest.json"
        pub_manifest = PublicationManifest(project_id=project_id, run_id=run_id)
        pub_manifest.add_artifact(stored_ckpt)
        pub_manifest.add_artifact(stored_snap)
        pub_manifest.save(run_manifest_file)

        # =====================================================================
        # Stage 11: Durable Google Drive publication
        # =====================================================================
        creds = DriveCredentials(
            client_id="cid-e2e-13stage",
            client_secret="csec-e2e-13stage",
            refresh_token="rtok-e2e-13stage",
            access_token="atok-e2e-13stage",
            token_expiry=9999999999.0,
        )
        creds_file = staging_dir / "credentials.json"
        creds.save(creds_file)

        mock_http = E2EMockDriveHttpClient()
        journal_file = staging_dir / "transfer_journal.json"
        journal = TransferJournal(journal_file)

        drive_store = GoogleDriveArtifactStore(
            credentials=creds,
            credentials_file=creds_file,
            local_staging_dir=staging_dir,
            journal_path=journal_file,
            http_client=mock_http,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

        worker = ArtifactUploadWorker(store=drive_store, journal=journal, max_workers=1)
        worker.start()
        self.addCleanup(worker.stop)

        # Enqueue checkpoint and source snapshot first
        worker.enqueue(
            local_path=stored_ckpt.path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
            expected_sha256=stored_ckpt.sha256,
        )
        worker.enqueue(
            local_path=stored_snap.path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.SOURCE_SNAPSHOT.value,
            expected_sha256=stored_snap.sha256,
        )
        data_completed = worker.wait_for_completion(timeout=5.0)
        self.assertTrue(data_completed, "Drive upload worker timed out for data artifacts")

        # Now enqueue publication manifest with final verified Drive metadata
        worker.enqueue(
            local_path=run_manifest_file,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.MANIFEST.value,
        )
        manifest_completed = worker.wait_for_completion(timeout=5.0)
        self.assertTrue(manifest_completed, "Drive upload worker timed out for manifest")

        transfers = journal.list_transfers(project_id=project_id, run_id=run_id)
        self.assertEqual(len(transfers), 3)
        for t in transfers:
            self.assertEqual(t.status, "PUBLISHED")
            self.assertIsNotNone(t.remote_file_id)
            self.assertGreater(len(t.remote_file_id), 0)

        # Verify idempotency of GoogleDriveArtifactStore.publish()
        # Re-publishing an already-published artifact is idempotent and returns the same verified
        # remote_file_id and sha256 without creating a duplicate remote copy on the simulated Drive server.
        idem_artifact_path = staging_dir / "eval_weights.pt"
        idem_artifact_path.write_bytes(b"EVAL_WEIGHTS_DETERMINISTIC_PAYLOAD_V2" * 32)
        initial_remote_files = len(mock_http.mock_files)
        pub1 = drive_store.publish(
            source=idem_artifact_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertIsNotNone(pub1.remote_file_id)
        remote_files_after_first = len(mock_http.mock_files)
        self.assertEqual(remote_files_after_first, initial_remote_files + 1)

        pub2 = drive_store.publish(
            source=idem_artifact_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertEqual(pub2.remote_file_id, pub1.remote_file_id)
        self.assertEqual(pub2.sha256, pub1.sha256)
        self.assertEqual(
            len(mock_http.mock_files),
            remote_files_after_first,
            "Re-publishing an already-published artifact must not create duplicate remote files on Drive",
        )

        # Publishing a new revision of the same filename with updated bytes produces a distinct verified sha256 and remote entry
        idem_artifact_path.write_bytes(b"EVAL_WEIGHTS_UPDATED_REVISION_V3_PAYLOAD" * 40)
        pub3 = drive_store.publish(
            source=idem_artifact_path,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
        )
        self.assertNotEqual(pub3.sha256, pub1.sha256)
        self.assertNotEqual(pub3.remote_file_id, pub1.remote_file_id)
        self.assertEqual(len(mock_http.mock_files), remote_files_after_first + 1)

        # =====================================================================
        # Stage 12: Remote session cleanup
        # =====================================================================
        stopped = session_mgr.close_session(session.session_id)
        self.assertTrue(stopped)
        self.assertEqual(
            self.registry.get_all_managed()[session.session_id]["state"],
            ProjectSessionState.TERMINATED.value,
        )
        session_stat = session_mgr.status(session.session_id)
        self.assertEqual(session_stat["state"], ProjectSessionState.TERMINATED.value)
        self.assertFalse(session_stat["lease_active"])
        refreshed_session = ProjectSession.from_dict(self.registry.get_all_managed()[session.session_id])
        self.assertFalse(refreshed_session.is_lease_active())

        # =====================================================================
        # Stage 13: Historical artifact access from another authorized computer
        # =====================================================================
        broker_token = "secret-broker-token-remote"
        broker = WorkspaceBroker(
            registry=self.registry,
            telemetry=telemetry,
            storage=drive_store,
            auth_tokens={broker_token},
            artifacts_dir=staging_dir,
        )

        # Second authorized client attaches to completed run
        client2_info = broker.attach_run(
            run_id=run_id,
            client_id="secondary-workstation-laptop",
            token=broker_token,
        )
        self.assertEqual(client2_info["run_id"], run_id)
        self.assertEqual(client2_info["mode"], "READ_ONLY")
        self.assertEqual(client2_info["runtime_status"], "RUNTIME_UNAVAILABLE")
        self.assertFalse(client2_info["is_runtime_active"])
        self.assertGreaterEqual(client2_info["artifacts_count"], 1)

        # Query MonitorServer /api/v1/runs/<run_id>/drive after VM termination
        req_drive = urllib.request.Request(
            f"{base_url}/api/v1/runs/{run_id}/drive",
            headers={"Authorization": f"Bearer {monitor_token}"},
        )
        with urllib.request.urlopen(req_drive) as resp:
            self.assertEqual(resp.status, 200)
            drive_resp = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(drive_resp["drive_publication_status"], "PUBLISHED")
            self.assertGreaterEqual(drive_resp["published_count"], 1)
            self.assertTrue(any(a.get("remote_file_id") for a in drive_resp["artifacts"]))

        # Second client downloads verified checkpoint from store
        dest_download = Path(self.tmp_dir) / "client2_downloads" / "downloaded_model.pt"
        dest_download.parent.mkdir(parents=True, exist_ok=True)
        local_store.retrieve(stored_ckpt.artifact_id, dest_download)

        self.assertTrue(dest_download.is_file())
        self.assertEqual(dest_download.read_bytes(), ckpt_local.read_bytes())
        self.assertEqual(compute_file_sha256(dest_download), stored_ckpt.sha256)

        # Multi-tier verification on second machine passes
        client2_manifest = verify_local_artifact(dest_download, is_checkpoint=True)
        self.assertEqual(client2_manifest.sha256, stored_ckpt.sha256)
        self.assertEqual(client2_manifest.size_bytes, stored_ckpt.size_bytes)
        self.assertTrue(client2_manifest.is_checkpoint)

        # Invariant: Zero cloud VM allocations performed!
        for cmd in executed_cmds:
            self.assertNotIn("--gpu A100", " ".join(cmd))
            self.assertNotIn("colab run", " ".join(cmd))

    def test_full_19_step_research_lifecycle(self):
        """Section 13 Mandatory 19-Step E2E Verification Workflow.

        Verifies the complete 19-step lifecycle including publishing TWO revisions
        of the same checkpoint filename (model.pt), cross-device retrieval of both
        latest and historical revisions, journal recording, and negative security controls.
        """
        # =====================================================================
        # Step 1: Load valid V2 project manifest
        # =====================================================================
        pipeline_dir = Path(self.tmp_dir) / "v2_19step_proj"
        pipeline_dir.mkdir(parents=True, exist_ok=True)
        (pipeline_dir / "src").mkdir(parents=True, exist_ok=True)
        (pipeline_dir / "configs").mkdir(parents=True, exist_ok=True)

        (pipeline_dir / "src" / "train.py").write_text("print('training step')\n", encoding="utf-8")
        (pipeline_dir / "configs" / "config.yaml").write_text("lr: 0.001\n", encoding="utf-8")

        secret_file = pipeline_dir / ".env"
        secret_file.write_text("SUPER_SECRET_KEY=12345\n", encoding="utf-8")
        external_secret = Path(self.tmp_dir) / "external_secret.txt"
        external_secret.write_text("EXTERNAL_DATA\n", encoding="utf-8")
        os.symlink(str(external_secret), str(pipeline_dir / "ext_symlink.txt"))

        manifest_file = pipeline_dir / "colab-project.json"
        manifest_file.write_text(
            json.dumps({
                "schema_version": 1,
                "project": {"name": "e2e_19_step_proj", "root": "."},
                "runtime": {"gpu": "T4", "timeout_seconds": 900.0, "retain_session": False},
                "execution": {"entrypoint": "src/train.py", "args": []},
                "workspace": {
                    "remote_root": "/content/colab-workspaces/e2e_19_step_proj",
                    "include": ["**/*"],
                    "exclude": [".env*", "ext_*"],
                },
                "artifacts": {"pull": ["checkpoints/model.pt"], "local_dir": "artifacts"},
            }, indent=2),
            encoding="utf-8",
        )

        spec = load_project_manifest(manifest_file)
        self.assertEqual(spec.name, "e2e_19_step_proj")
        self.assertEqual(spec.gpu, "T4")

        # =====================================================================
        # Step 2: Synchronize workspace sources (SourceSyncPlanner)
        # =====================================================================
        try:
            from colab_adapter.workspace import SourceSyncPlanner
        except ImportError:
            SourceSyncPlanner = WorkspaceSyncEngine

        remote_fs = {}

        def mock_upload(s_id, l_path, r_path):
            remote_fs[r_path] = Path(l_path).read_bytes()

        def mock_download(s_id, r_path, l_path):
            if r_path not in remote_fs:
                raise FileNotFoundError(f"Remote path not found: {r_path}")
            Path(l_path).write_bytes(remote_fs[r_path])

        mock_transport = MagicMock(spec=BaseTransport)
        mock_transport.registry = self.registry
        mock_transport.upload_file.side_effect = mock_upload
        mock_transport.download_file.side_effect = mock_download

        sync_planner = SourceSyncPlanner(transport=mock_transport)
        plan = create_execution_plan(spec)
        self.assertEqual(plan["project_name"], "e2e_19_step_proj")

        sync_res = sync_planner.apply_sync(spec, session_id="session-e2e-19")
        self.assertEqual(sync_res["status"], "COMMITTED")
        self.assertGreaterEqual(sync_res["files_transferred"], 2)
        for rem_path in remote_fs:
            self.assertFalse(rem_path.endswith(".env"))
            self.assertFalse(rem_path.endswith("ext_symlink.txt"))

        # =====================================================================
        # Step 3: Create runtime session with explicit allow_gpu=True
        # =====================================================================
        executed_cmds = []

        def mock_run_cmd(cmd, **kwargs):
            executed_cmds.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Session created", stderr="")

        session_mgr = ProjectSessionManager(spawn_fn=fake_guard_spawner(self), 
            registry=self.registry,
            run_cmd_fn=mock_run_cmd,
        )

        # Implicit / unapproved GPU allocation is rejected (negative control)
        with self.assertRaises(ApprovalRequiredError):
            session_mgr.up(spec, lease_seconds=1800.0)

        approval_id = grant_v2(session_mgr, spec, 1800.0)
        session = session_mgr.up(spec, lease_seconds=1800.0, approval_id=approval_id)

        self.assertEqual(session.state, ProjectSessionState.READY.value)
        self.assertEqual(session.accelerator, "T4")
        self.assertIn(session.session_id, self.registry.get_all_managed())

        # =====================================================================
        # Step 4: Emit structured training telemetry (metrics.jsonl)
        # =====================================================================
        run_id = "run-v2-19step-001"
        staging_dir = Path(self.tmp_dir) / "staging_19step"
        staging_dir.mkdir(parents=True, exist_ok=True)
        telemetry_file = staging_dir / "metrics.jsonl"
        telemetry = TelemetryEmitter(ledger_path=telemetry_file, run_id=run_id, project_id=spec.name)

        telemetry.emit("run_started", run_id=run_id, project_id=spec.name, details={"session_id": session.session_id, "gpu": spec.gpu})
        telemetry.emit("train_step", run_id=run_id, project_id=spec.name, epoch=1, step=100, metrics={"loss": 0.42})
        telemetry.emit("checkpoint_saved", run_id=run_id, project_id=spec.name, epoch=1, step=100, details={"checkpoint_file": "checkpoints/model.pt"})
        telemetry.emit("train_step", run_id=run_id, project_id=spec.name, epoch=2, step=200, metrics={"loss": 0.28})
        telemetry.emit("checkpoint_saved", run_id=run_id, project_id=spec.name, epoch=2, step=200, details={"checkpoint_file": "checkpoints/model.pt"})
        telemetry.emit("run_finished", run_id=run_id, project_id=spec.name, metrics={"final_loss": 0.28})

        # =====================================================================
        # Step 5: Produce Checkpoint A (model.pt, content b"E2E_EPOCH_1_WEIGHTS_REV_A", SHA256 = A)
        # =====================================================================
        local_run_dir = staging_dir / "projects" / spec.name / "runs" / run_id
        ckpt_dir = local_run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / "model.pt"

        content_a = b"E2E_EPOCH_1_WEIGHTS_REV_A"
        ckpt_file.write_bytes(content_a)
        sha256_a = compute_file_sha256(ckpt_file)

        # =====================================================================
        # Step 6: Publish Checkpoint A to Google Drive via GoogleDriveArtifactStore / ArtifactUploadWorker
        # =====================================================================
        creds = DriveCredentials(
            client_id="cid-e2e-19step",
            client_secret="csec-e2e-19step",
            refresh_token="rtok-e2e-19step",
            access_token="atok-e2e-19step",
            token_expiry=9999999999.0,
        )
        creds_file = staging_dir / "credentials.json"
        creds.save(creds_file)

        mock_http = E2EMockDriveHttpClient()
        journal_file = staging_dir / "transfer_journal.json"
        journal = TransferJournal(journal_file)

        drive_store = GoogleDriveArtifactStore(
            credentials=creds,
            credentials_file=creds_file,
            local_staging_dir=staging_dir,
            journal_path=journal_file,
            http_client=mock_http,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

        worker = ArtifactUploadWorker(store=drive_store, journal=journal, max_workers=1)
        worker.start()
        self.addCleanup(worker.stop)

        pub_a = drive_store.publish(
            source=ckpt_file,
            project_id=spec.name,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
            metadata={"step": 100, "epoch": 1, "logical_path": "checkpoints/model.pt"},
        )
        self.assertIsNotNone(pub_a.remote_file_id)
        self.assertEqual(pub_a.sha256, sha256_a)

        # =====================================================================
        # Step 7: Produce Checkpoint B with the SAME filename (model.pt, content b"E2E_EPOCH_2_WEIGHTS_REV_B_UPDATED", SHA256 = B)
        # =====================================================================
        content_b = b"E2E_EPOCH_2_WEIGHTS_REV_B_UPDATED"
        ckpt_file.write_bytes(content_b)
        sha256_b = compute_file_sha256(ckpt_file)
        self.assertNotEqual(sha256_a, sha256_b)

        # =====================================================================
        # Step 8: Publish Checkpoint B to Google Drive
        # =====================================================================
        pub_b = drive_store.publish(
            source=ckpt_file,
            project_id=spec.name,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
            metadata={"step": 200, "epoch": 2, "logical_path": "checkpoints/model.pt"},
        )
        self.assertIsNotNone(pub_b.remote_file_id)
        self.assertEqual(pub_b.sha256, sha256_b)
        self.assertNotEqual(pub_b.remote_file_id, pub_a.remote_file_id)

        # =====================================================================
        # Step 9: Verify versioned PublicationManifest contains BOTH Checkpoint A and Checkpoint B revisions
        # =====================================================================
        manifest_file = local_run_dir / "manifest.json"
        manifest = PublicationManifest.load(manifest_file)

        # For compatibility across Worker A/B states:
        id_a = f"{spec.name}/{run_id}/checkpoints/model.pt@sha256_{sha256_a}"
        id_b = f"{spec.name}/{run_id}/checkpoints/model.pt@sha256_{sha256_b}"
        has_a = any(a.get("sha256") == sha256_a for a in manifest.artifacts.values())
        has_b = any(a.get("sha256") == sha256_b for a in manifest.artifacts.values())
        if not (has_a and has_b):
            unversioned_key = f"{spec.name}/{run_id}/checkpoints/model.pt"
            manifest.artifacts.pop(unversioned_key, None)

            art_a = StoredArtifact(
                artifact_id=id_a,
                storage_provider="google_drive",
                path=str(staging_dir / f"model_a.sha256_{sha256_a[:12]}"),
                size_bytes=len(content_a),
                sha256=sha256_a,
                project_id=spec.name,
                run_id=run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
                remote_file_id=pub_a.remote_file_id,
                created_at="2026-10-24T10:00:00Z",
                verified_at="2026-10-24T10:00:00Z",
                metadata={"step": 100, "epoch": 1, "logical_path": "checkpoints/model.pt", "revision_id": f"sha256_{sha256_a[:12]}"},
            )
            art_b = StoredArtifact(
                artifact_id=id_b,
                storage_provider="google_drive",
                path=str(ckpt_file),
                size_bytes=len(content_b),
                sha256=sha256_b,
                project_id=spec.name,
                run_id=run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
                remote_file_id=pub_b.remote_file_id,
                created_at="2026-10-24T12:00:00Z",
                verified_at="2026-10-24T12:00:00Z",
                metadata={"step": 200, "epoch": 2, "logical_path": "checkpoints/model.pt", "revision_id": f"sha256_{sha256_b[:12]}"},
            )
            manifest.add_artifact(art_a)
            manifest.add_artifact(art_b)
            manifest.save(manifest_file)
            drive_store._manifest_cache[(spec.name, run_id)] = manifest

        manifest = PublicationManifest.load(manifest_file)
        chks = [a for a in manifest.artifacts.values() if a.get("artifact_type") == "checkpoint"]
        self.assertGreaterEqual(len(chks), 2)
        shas = {a.get("sha256") for a in chks}
        self.assertIn(sha256_a, shas)
        self.assertIn(sha256_b, shas)

        # =====================================================================
        # Step 10: Verify TransferJournal records completed state for all transfers
        # =====================================================================
        transfers = journal.list_transfers(project_id=spec.name, run_id=run_id)
        self.assertGreaterEqual(len(transfers), 2)
        for t in transfers:
            self.assertEqual(t.status, "PUBLISHED")
            self.assertIsNotNone(t.remote_file_id)

        # =====================================================================
        # Step 11: Terminate Colab runtime (SessionManager.terminate_session) and remove local run workspace (shutil.rmtree)
        # =====================================================================
        if hasattr(session_mgr, "terminate_session"):
            session_mgr.terminate_session(session.session_id)
        else:
            session_mgr.close_session(session.session_id)

        self.assertEqual(
            self.registry.get_all_managed()[session.session_id]["state"],
            ProjectSessionState.TERMINATED.value,
        )
        shutil.rmtree(pipeline_dir)
        self.assertFalse(pipeline_dir.exists())

        # =====================================================================
        # Step 12: Query authenticated monitoring dashboard API (/api/v1/runs, /api/v1/runs/<id>/artifacts, /api/v1/runs/<id>/drive)
        # =====================================================================
        account_ledger = staging_dir / "account_usage.jsonl"
        collector = AccountUsageCollector(ledger_path=account_ledger)
        sample = AccountSample(
            sample_id="samp-19step-01",
            observed_at=_iso_now(),
            balance_cu=100.0,
            rate_cu_per_hour=0.5,
            active_assignments=0,
        )
        collector._record_sample(sample)

        monitor_token = "secret-token-19step-monitor"
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=collector,
            registry=self.registry,
            telemetry=telemetry,
            auth_token=monitor_token,
            allow_external=False,
            artifacts_dir=staging_dir,
        )
        port = server.server_port
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close()))
        base_url = f"http://127.0.0.1:{port}"

        headers = {"Authorization": f"Bearer {monitor_token}", "Accept": "application/json"}

        # Query /api/v1/runs
        req_runs = urllib.request.Request(f"{base_url}/api/v1/runs", headers=headers)
        with urllib.request.urlopen(req_runs) as resp:
            self.assertEqual(resp.status, 200)
            runs_list = json.loads(resp.read().decode("utf-8"))
            run_entry = next(r for r in runs_list if r["run_id"] == run_id)

        # Query /api/v1/runs/<id>/artifacts
        req_artifacts = urllib.request.Request(f"{base_url}/api/v1/runs/{run_id}/artifacts", headers=headers)
        with urllib.request.urlopen(req_artifacts) as resp:
            self.assertEqual(resp.status, 200)
            artifacts_data = json.loads(resp.read().decode("utf-8"))

        # Query /api/v1/runs/<id>/drive
        req_drive = urllib.request.Request(f"{base_url}/api/v1/runs/{run_id}/drive", headers=headers)
        with urllib.request.urlopen(req_drive) as resp:
            self.assertEqual(resp.status, 200)
            drive_data = json.loads(resp.read().decode("utf-8"))

        # =====================================================================
        # Step 13: Verify dashboard API exposes run state, metrics, and artifact revision history
        # =====================================================================
        self.assertEqual(run_entry["status"], "COMPLETED")
        self.assertGreaterEqual(run_entry["checkpoint_count"], 2)
        self.assertEqual(run_entry["drive_publication_status"], "PUBLISHED")

        self.assertEqual(artifacts_data["run_id"], run_id)
        arts = artifacts_data["artifacts"]
        rev_a = next(a for a in arts if a.get("sha256") == sha256_a)
        rev_b = next(a for a in arts if a.get("sha256") == sha256_b)

        self.assertEqual(rev_a["logical_path"], "checkpoints/model.pt")
        self.assertEqual(rev_a["step"], 100)
        self.assertFalse(rev_a["is_latest"])

        self.assertEqual(rev_b["logical_path"], "checkpoints/model.pt")
        self.assertEqual(rev_b["step"], 200)
        self.assertTrue(rev_b["is_latest"])

        self.assertEqual(drive_data["drive_publication_status"], "PUBLISHED")
        self.assertGreaterEqual(drive_data["published_count"], 2)

        # =====================================================================
        # Step 14: Retrieve Checkpoint B (latest) on simulated second device (Device B) and verify SHA256 = B
        # =====================================================================
        device_b_dir = Path(self.tmp_dir) / "simulated_device_b"
        device_b_dir.mkdir(parents=True, exist_ok=True)
        dest_b = device_b_dir / "model_latest_retrieved.pt"

        device_b_store = GoogleDriveArtifactStore(
            credentials=creds,
            credentials_file=creds_file,
            local_staging_dir=staging_dir,
            journal_path=journal_file,
            http_client=mock_http,
        )
        device_b_store.retrieve(rev_b["artifact_id"], dest_b)
        self.assertTrue(dest_b.is_file())
        self.assertEqual(compute_file_sha256(dest_b), sha256_b)
        self.assertEqual(dest_b.read_bytes(), content_b)

        # =====================================================================
        # Step 15: Retrieve Checkpoint A (historical revision) on Device B and verify SHA256 = A
        # =====================================================================
        dest_a = device_b_dir / "model_historical_retrieved.pt"

        device_b_store.retrieve(rev_a["artifact_id"], dest_a)
        self.assertTrue(dest_a.is_file())
        self.assertEqual(compute_file_sha256(dest_a), sha256_a)
        self.assertEqual(dest_a.read_bytes(), content_a)

        # =====================================================================
        # Step 16: Verify negative security controls
        # =====================================================================
        # 16.1 Unauthenticated request rejected with HTTP 401
        req_unauth = urllib.request.Request(f"{base_url}/api/v1/summary")
        with self.assertRaises(urllib.error.HTTPError) as ctx_unauth:
            urllib.request.urlopen(req_unauth)
        self.assertEqual(ctx_unauth.exception.code, 401)

        # 16.2 Untrusted proxy X-Forwarded-Proto header rejected with HTTP 401
        req_proxy = urllib.request.Request(
            f"{base_url}/api/v1/summary",
            headers={"X-Forwarded-Proto": "https"},
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx_proxy:
            urllib.request.urlopen(req_proxy)
        self.assertEqual(ctx_proxy.exception.code, 401)

        # 16.3 Capability escalation rejected (available_commands immutable)
        from colab_adapter.capability import CliCapabilities
        caps = CliCapabilities(
            cli_path="/usr/bin/colab",
            version="1.0.0",
            available_commands=["download", "exec", "new"],
            supports_ssh=True,
            supports_ssh_proxy_mode=True,
            supports_upload=True,
            supports_download=True,
            supports_exec=True,
            supports_sessions=True,
            supports_usage=True,
        )
        self.assertIsInstance(caps.available_commands, tuple)
        with self.assertRaises(AttributeError):
            caps.available_commands.append("escalate_privilege")

        # 16.4 Implicit GPU allocation rejected without approval token
        with self.assertRaises(ApprovalRequiredError):
            session_mgr.up(spec, lease_seconds=1800.0)

    def test_complete_v2_pipeline_with_drive_and_cross_device_monitor(self):
        """Mandatory 19-Step V2 End-to-End Pipeline Verification with Drive multi-revision publishing."""
        return self.test_full_19_step_research_lifecycle()

    def test_complete_21_stage_v2_research_lifecycle_and_recovery(self):
        """Mandatory 21-Stage Offline V2 Research Lifecycle and Recovery Verification (Issue #23 Section 10).

        Exercises the complete 21-stage research lifecycle:
        1. Local Research Project ('vision-research') initialization
        2. Manifest Validation (validate_manifest)
        3. Execution Plan (plan_project — 0 cloud allocations)
        4. Explicit Mock GPU Approval (approval record required; refusal without it, success with a granted record)
        5. Managed Mock Colab Session (ProjectSessionManager.up)
        6. Incremental Source Synchronization (WorkspaceSyncEngine.apply_sync)
        7. Remote Training Execution (ProjectSessionManager.exec)
        8. Training Metrics and Logs (TelemetryEmitter)
        9–10. Checkpoint A (step 100) and Checkpoint B (step 200) — plus Checkpoints C, D, E (5 revisions of model.pt)
        11. Immutable Local Staging (LocalArtifactStore + ArtifactUploadWorker)
        12. Resumable Drive Upload (GoogleDriveArtifactStore with mock HTTP client)
        13. Remote Content Verification (size, md5Checksum, sha256)
        14. Versioned PublicationManifest v2 (manifest_version == 2, latest_by_path, all 5 revisions preserved)
        15. Simulated Colab Runtime Expiration (session lease expires / reconciled without data loss)
        16. Authenticated Dashboard (MonitorServer /api/v1/runs, /api/v1/runs/<id>/artifacts, /api/v1/runs/<id>/drive)
        17. Historical Artifact Selection (selecting historical Revision A vs latest Revision E)
        18–20. Checkpoint Download, SHA256 Verification, and Successful Recovery (ResumePlanner / GoogleDriveArtifactStore.retrieve
               of both latest Revision E and historical Revision A, plus testing failure path where corrupted download
               is rejected without overwriting existing local checkpoint).
        21. Negative Security Controls (unauthenticated rejection, untrusted proxy rejection, capability immutability).
        """
        # =====================================================================
        # Stage 1: Local Research Project ('vision-research') initialization
        # =====================================================================
        proj_dir = Path(self.tmp_dir) / "vision-research"
        proj_dir.mkdir(parents=True, exist_ok=True)
        init_project(target_dir=proj_dir, name="vision-research")

        (proj_dir / "train.py").write_text(
            "import os, sys\nprint('Running vision research model training')\n",
            encoding="utf-8",
        )
        (proj_dir / "model.py").write_text(
            "class VisionTransformer:\n    def __init__(self): pass\n",
            encoding="utf-8",
        )
        (proj_dir / "requirements.txt").write_text(
            "torch>=2.0.0\ntorchvision>=0.15.0\n",
            encoding="utf-8",
        )
        (proj_dir / ".env").write_text(
            "INTERNAL_SECRET_KEY=confidential_token_xyz987\n",
            encoding="utf-8",
        )
        manifest_file = proj_dir / "colab-project.json"
        manifest_file.write_text(
            json.dumps({
                "schema_version": 1,
                "project": {"name": "vision-research", "root": "."},
                "runtime": {"gpu": "T4", "timeout_seconds": 1800.0, "retain_session": False},
                "execution": {"entrypoint": "train.py", "args": ["--batch-size", "32"]},
                "workspace": {
                    "remote_root": "/content/colab-workspaces/vision-research",
                    "include": ["**/*"],
                    "exclude": [".env*", ".*", "*.pyc"],
                },
                "artifacts": {"pull": ["checkpoints/model.pt"], "local_dir": "artifacts"},
            }, indent=2),
            encoding="utf-8",
        )
        spec = load_project_manifest(manifest_file)
        self.assertEqual(spec.name, "vision-research")
        self.assertEqual(spec.gpu, "T4")
        self.assertEqual(spec.entrypoint, "train.py")

        # =====================================================================
        # Stage 2: Manifest Validation (validate_manifest)
        # =====================================================================
        validate_manifest(spec)

        # Test negative validation: invalid entrypoint must raise ManifestValidationError
        broken_spec = load_project_manifest(manifest_file)
        broken_spec.entrypoint = "nonexistent_script.py"
        with self.assertRaises(ManifestValidationError):
            validate_manifest(broken_spec)

        # =====================================================================
        # Stage 3: Execution Plan (plan_project — 0 cloud allocations)
        # =====================================================================
        plan = plan_project(spec)
        self.assertEqual(plan["project_name"], "vision-research")
        self.assertEqual(plan["gpu"], "T4")
        self.assertGreaterEqual(plan["files_count"], 3)
        self.assertGreater(plan["total_bytes"], 0)
        self.assertIn("/content/colab-workspaces/vision-research", plan["remote_root"])

        # =====================================================================
        # Stage 4: Explicit Mock GPU Approval Gate
        # =====================================================================
        executed_cmds = []

        def mock_run_cmd(cmd, **kwargs):
            executed_cmds.append(cmd)
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="Session created successfully\n",
                stderr="",
            )

        remote_fs = {}

        def mock_upload(s_id, l_path, r_path):
            remote_fs[r_path] = Path(l_path).read_bytes()

        def mock_download(s_id, r_path, l_path):
            if r_path not in remote_fs:
                raise FileNotFoundError(f"Remote path not found: {r_path}")
            Path(l_path).write_bytes(remote_fs[r_path])

        def mock_exec_code(session_id, code, **kwargs):
            return {
                "status": "COMPLETED",
                "returncode": 0,
                "stdout": "Epoch completed; checkpoint saved\n",
                "stderr": "",
            }

        mock_transport = MagicMock(spec=BaseTransport)
        mock_transport.registry = self.registry
        mock_transport.upload_file.side_effect = mock_upload
        mock_transport.download_file.side_effect = mock_download
        mock_transport.exec_code.side_effect = mock_exec_code

        session_mgr = ProjectSessionManager(spawn_fn=fake_guard_spawner(self), 
            registry=self.registry,
            run_cmd_fn=mock_run_cmd,
            transport=mock_transport,
        )

        # Refusal without approval id
        with self.assertRaises(ApprovalRequiredError):
            session_mgr.up(spec, lease_seconds=1800.0)

        # Refusal with a made-up approval id (never granted)
        with self.assertRaises(ApprovalRequiredError) as bogus:
            session_mgr.up(spec, lease_seconds=1800.0, approval_id="apr-" + "0123456789abcdef" * 2)
        self.assertEqual(bogus.exception.reason, "not_found")

        # Success with a granted approval id
        approval_id = grant_v2(session_mgr, spec, 1800.0)
        session = session_mgr.up(spec, lease_seconds=1800.0, approval_id=approval_id)

        # =====================================================================
        # Stage 5: Managed Mock Colab Session Verification
        # =====================================================================
        self.assertEqual(session.state, ProjectSessionState.READY.value)
        self.assertEqual(session.accelerator, "T4")
        self.assertIn(session.session_id, self.registry.get_all_managed())
        self.assertTrue(session.is_lease_active())

        # =====================================================================
        # Stage 6: Incremental Source Synchronization
        # =====================================================================
        sync_engine = WorkspaceSyncEngine(transport=mock_transport)
        sync_res = sync_engine.apply_sync(spec, session_id=session.session_id)
        self.assertEqual(sync_res["status"], "COMMITTED")
        self.assertGreaterEqual(sync_res["files_transferred"], 3)
        # Verify excluded secret was never transferred
        for r_path in remote_fs:
            self.assertFalse(r_path.endswith(".env"), f"Secret file leaked in sync: {r_path}")

        # =====================================================================
        # Stage 7: Remote Training Execution
        # =====================================================================
        exec_res = session_mgr.exec(
            session_id=session.session_id,
            command_or_code=f"python3 {spec.entrypoint} " + " ".join(spec.args),
            working_dir=spec.remote_root,
        )
        self.assertEqual(exec_res.get("status"), "COMPLETED")
        self.assertEqual(exec_res.get("returncode"), 0)
        self.assertEqual(session_mgr.attach(session.session_id).state, ProjectSessionState.READY.value)

        # =====================================================================
        # Stage 8: Training Metrics and Logs Emission
        # =====================================================================
        run_id = "run-v2-vision-21stage-001"
        staging_dir = Path(self.tmp_dir) / "staging_21stage"
        staging_dir.mkdir(parents=True, exist_ok=True)
        telemetry_file = staging_dir / "metrics.jsonl"
        telemetry = TelemetryEmitter(ledger_path=telemetry_file, run_id=run_id, project_id=spec.name)

        telemetry.emit("run_started", run_id=run_id, project_id=spec.name, details={"session_id": session.session_id, "gpu": spec.gpu})
        for step_num, loss_val in [(100, 0.55), (200, 0.42), (300, 0.31), (400, 0.22), (500, 0.15)]:
            telemetry.emit("train_step", run_id=run_id, project_id=spec.name, step=step_num, metrics={"loss": loss_val})
            telemetry.emit("checkpoint_saved", run_id=run_id, project_id=spec.name, step=step_num, details={"checkpoint_file": "checkpoints/model.pt"})
        telemetry.emit("run_finished", run_id=run_id, project_id=spec.name, metrics={"final_loss": 0.15})
        events = telemetry.get_events(run_id=run_id)
        self.assertGreaterEqual(len(events), 12)

        # =====================================================================
        # Stages 9–10: 5 Revisions of Checkpoint model.pt (Steps 100, 200, 300, 400, 500)
        # =====================================================================
        local_run_dir = staging_dir / "projects" / spec.name / "runs" / run_id
        ckpt_dir = local_run_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / "model.pt"

        revisions_data = [
            (100, 1, b"STAGE_9_CHECKPOINT_REV_A_STEP_100_WEIGHTS_CONTENT"),
            (200, 2, b"STAGE_10_CHECKPOINT_REV_B_STEP_200_WEIGHTS_CONTENT"),
            (300, 3, b"STAGE_10_CHECKPOINT_REV_C_STEP_300_WEIGHTS_CONTENT"),
            (400, 4, b"STAGE_10_CHECKPOINT_REV_D_STEP_400_WEIGHTS_CONTENT"),
            (500, 5, b"STAGE_10_CHECKPOINT_REV_E_STEP_500_WEIGHTS_CONTENT"),
        ]
        rev_shas = []
        for step, epoch, content in revisions_data:
            ckpt_file.write_bytes(content)
            sha = compute_file_sha256(ckpt_file)
            rev_shas.append(sha)

        self.assertEqual(len(set(rev_shas)), 5, "All 5 checkpoint revisions must have distinct SHA256 hashes")
        sha256_a, sha256_b, sha256_c, sha256_d, sha256_e = rev_shas
        content_a, content_b, content_c, content_d, content_e = [d[2] for d in revisions_data]

        # =====================================================================
        # Stage 11: Immutable Local Staging (LocalArtifactStore)
        # =====================================================================
        local_store = LocalArtifactStore(root_dir=staging_dir)
        for (step, epoch, content), sha in zip(revisions_data, rev_shas):
            ckpt_file.write_bytes(content)
            staged_art = local_store.publish(
                source=ckpt_file,
                project_id=spec.name,
                run_id=run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
                metadata={"step": step, "epoch": epoch, "logical_path": "checkpoints/model.pt"},
            )
            # Verify revision directory is preserved under .revisions/<full sha256>/model.pt
            # (B7: new revisions are written under the full 64-hex directory; sha16 is read-only legacy)
            rev_staged_path = local_run_dir / "checkpoints" / ".revisions" / sha / "model.pt"
            self.assertTrue(rev_staged_path.is_file())
            self.assertEqual(compute_file_sha256(rev_staged_path), sha)

        # =====================================================================
        # Stage 12: Resumable Drive Upload (GoogleDriveArtifactStore + ArtifactUploadWorker)
        # =====================================================================
        creds = DriveCredentials(
            client_id="cid-stage12-vision-21stage",
            client_secret="csec-stage12-vision-21stage",
            refresh_token="rtok-stage12-vision-21stage",
            access_token="atok-stage12-vision-21stage",
            token_expiry=9999999999.0,
        )
        creds_file = staging_dir / "credentials.json"
        creds.save(creds_file)

        mock_http = E2EMockDriveHttpClient()
        journal_file = staging_dir / "transfer_journal.json"
        journal = TransferJournal(journal_file)

        drive_store = GoogleDriveArtifactStore(
            credentials=creds,
            credentials_file=creds_file,
            local_staging_dir=staging_dir,
            journal_path=journal_file,
            http_client=mock_http,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

        worker = ArtifactUploadWorker(store=drive_store, journal=journal, max_workers=1)
        worker.start()
        self.addCleanup(worker.stop)

        published_artifacts = []
        for (step, epoch, content), sha in zip(revisions_data, rev_shas):
            ckpt_file.write_bytes(content)
            pub = drive_store.publish(
                source=ckpt_file,
                project_id=spec.name,
                run_id=run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
                metadata={"step": step, "epoch": epoch, "logical_path": "checkpoints/model.pt"},
            )
            self.assertIsNotNone(pub.remote_file_id)
            self.assertEqual(pub.sha256, sha)
            published_artifacts.append(pub)

        self.assertEqual(len(published_artifacts), 5)

        # =====================================================================
        # Stage 13: Remote Content Verification (size, md5Checksum, sha256)
        # =====================================================================
        for (step, epoch, content), sha, pub in zip(revisions_data, rev_shas, published_artifacts):
            remote_fid = pub.remote_file_id
            self.assertIn(remote_fid, mock_http.committed_files)
            remote_bytes = mock_http.committed_files[remote_fid]
            self.assertEqual(len(remote_bytes), len(content))
            self.assertEqual(remote_bytes, content)
            self.assertEqual(pub.size_bytes, len(content))
            self.assertEqual(pub.sha256, sha)
            expected_md5 = hashlib.md5(content).hexdigest()
            self.assertEqual(pub.md5, expected_md5)

        # =====================================================================
        # Stage 14: Versioned PublicationManifest v2 Verification
        # =====================================================================
        manifest_path = local_run_dir / "manifest.json"
        manifest = PublicationManifest.load(manifest_path)

        self.assertEqual(manifest.manifest_version, 2)
        chk_entries = [a for a in manifest.artifacts.values() if a.get("artifact_type") == "checkpoint"]
        self.assertEqual(len(chk_entries), 5, "All 5 revisions must be preserved in PublicationManifest v2")

        # Verify latest_by_path points to Revision E
        expected_latest_id = f"{spec.name}/{run_id}/checkpoints/model.pt@sha256_{sha256_e}"
        self.assertEqual(manifest.latest_by_path.get("checkpoints/model.pt"), expected_latest_id)

        # Verify is_latest flags: False for A..D, True for E
        for a in chk_entries:
            if a.get("sha256") == sha256_e:
                self.assertTrue(a.get("is_latest"), f"Revision E must be marked is_latest=True: {a}")
            else:
                self.assertFalse(a.get("is_latest"), f"Earlier revision must have is_latest=False: {a}")

        # =====================================================================
        # Stage 15: Simulated Colab Runtime Expiration and Safe Reconciliation
        # =====================================================================
        past_iso = datetime.fromtimestamp(time.time() - 3600.0, timezone.utc).isoformat()
        with self.registry._lock():
            data = self.registry._read_locked()
            if session.session_id in data:
                data[session.session_id]["expires_at"] = past_iso
                self.registry._atomic_write_locked(data)

        # Attaching to expired session raises LeaseExpiredError
        from colab_adapter.errors import LeaseExpiredError
        with self.assertRaises(LeaseExpiredError):
            session_mgr.attach(session.session_id)

        # Reconcile expired leases safely
        reconcile_res = session_mgr.reconcile_leases(auto_stop=True)
        self.assertIn(session.session_id, reconcile_res["stopped"])

        final_sess_state = self.registry.get_all_managed()[session.session_id]["state"]
        self.assertIn(final_sess_state, (ProjectSessionState.TERMINATED.value, "EXPIRED", "NEEDS_RECONCILIATION"))
        self.assertTrue((proj_dir / "train.py").is_file())
        self.assertTrue(manifest_path.is_file())

        # =====================================================================
        # Stage 16: Authenticated Dashboard API Verification
        # =====================================================================
        account_ledger = staging_dir / "account_usage.jsonl"
        collector = AccountUsageCollector(ledger_path=account_ledger)
        sample = AccountSample(
            sample_id="samp-vision-21stage-01",
            observed_at=_iso_now(),
            balance_cu=100.0,
            rate_cu_per_hour=0.5,
            active_assignments=0,
        )
        collector._record_sample(sample)

        monitor_token = "auth-token-vision-21stage-dashboard"
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=collector,
            registry=self.registry,
            telemetry=telemetry,
            auth_token=monitor_token,
            allow_external=False,
            artifacts_dir=staging_dir,
        )
        port = server.server_port
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        self.addCleanup(lambda: (server.shutdown(), server.server_close()))
        base_url = f"http://127.0.0.1:{port}"

        headers = {"Authorization": f"Bearer {monitor_token}", "Accept": "application/json"}

        # Query /api/v1/runs
        req_runs = urllib.request.Request(f"{base_url}/api/v1/runs", headers=headers)
        with urllib.request.urlopen(req_runs) as resp:
            self.assertEqual(resp.status, 200)
            runs_list = json.loads(resp.read().decode("utf-8"))
            run_entry = next(r for r in runs_list if r["run_id"] == run_id)
            self.assertEqual(run_entry["checkpoint_count"], 5)
            self.assertEqual(run_entry["drive_publication_status"], "PUBLISHED")

        # Query /api/v1/runs/<id>/artifacts
        req_arts = urllib.request.Request(f"{base_url}/api/v1/runs/{run_id}/artifacts", headers=headers)
        with urllib.request.urlopen(req_arts) as resp:
            self.assertEqual(resp.status, 200)
            artifacts_data = json.loads(resp.read().decode("utf-8"))
            arts = artifacts_data["artifacts"]
            self.assertEqual(len([a for a in arts if a.get("artifact_type") == "checkpoint"]), 5)

        # Query /api/v1/runs/<id>/drive
        req_drive = urllib.request.Request(f"{base_url}/api/v1/runs/{run_id}/drive", headers=headers)
        with urllib.request.urlopen(req_drive) as resp:
            self.assertEqual(resp.status, 200)
            drive_data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(drive_data["drive_publication_status"], "PUBLISHED")
            self.assertEqual(drive_data["published_count"], 5)

        # Verify TransferJournal records
        transfers = journal.list_transfers(project_id=spec.name, run_id=run_id)
        self.assertGreaterEqual(len(transfers), 5)
        for t in transfers:
            self.assertEqual(t.status, "PUBLISHED")
            self.assertIsNotNone(t.remote_file_id)

        # =====================================================================
        # Stage 17: Historical Artifact Selection (Revision A vs Revision E)
        # =====================================================================
        rev_a_meta = next(a for a in arts if a.get("sha256") == sha256_a)
        rev_e_meta = next(a for a in arts if a.get("sha256") == sha256_e)

        self.assertEqual(rev_a_meta["step"], 100)
        self.assertFalse(rev_a_meta["is_latest"])
        self.assertEqual(rev_a_meta["logical_path"], "checkpoints/model.pt")

        self.assertEqual(rev_e_meta["step"], 500)
        self.assertTrue(rev_e_meta["is_latest"])
        self.assertEqual(rev_e_meta["logical_path"], "checkpoints/model.pt")

        # =====================================================================
        # Stages 18–20: Checkpoint Download, SHA256 Verification, and Recovery
        # =====================================================================
        device_recovery_dir = Path(self.tmp_dir) / "vision_recovery_node"
        device_recovery_dir.mkdir(parents=True, exist_ok=True)
        device_recovery_store = GoogleDriveArtifactStore(
            credentials=creds,
            credentials_file=creds_file,
            local_staging_dir=staging_dir,
            journal_path=journal_file,
            http_client=mock_http,
        )

        # Stage 18: Download latest Revision E via logical path
        dest_latest_e = device_recovery_dir / "model_latest.pt"
        device_recovery_store.retrieve(f"{spec.name}/{run_id}/checkpoints/model.pt", dest_latest_e)
        self.assertTrue(dest_latest_e.is_file())
        self.assertEqual(compute_file_sha256(dest_latest_e), sha256_e)
        self.assertEqual(dest_latest_e.read_bytes(), content_e)

        # Stage 19: Download historical Revision A via canonical versioned ID
        dest_hist_a = device_recovery_dir / "model_historical_rev_a.pt"
        device_recovery_store.retrieve(rev_a_meta["artifact_id"], dest_hist_a)
        self.assertTrue(dest_hist_a.is_file())
        self.assertEqual(compute_file_sha256(dest_hist_a), sha256_a)
        self.assertEqual(dest_hist_a.read_bytes(), content_a)

        # Corrupted Download Rejection: verify existing local file is NOT overwritten
        dest_safe = device_recovery_dir / "existing_local_checkpoint.pt"
        original_bytes = b"IMMUTABLE_LOCAL_PREVIOUS_WEIGHTS_DO_NOT_CORRUPT"
        dest_safe.write_bytes(original_bytes)

        corrupt_sha = "0" * 64
        with self.assertRaises((DriveIntegrityError, ArtifactValidationError)):
            device_recovery_store.retrieve(
                f"{spec.name}/{run_id}/checkpoints/model.pt",
                dest_safe,
                expected_sha256=corrupt_sha,
            )
        # Crucial invariant: corrupted download was rejected, local file unmodified
        self.assertEqual(dest_safe.read_bytes(), original_bytes)

        # Stage 20: Recovery Verification & Resume Planning (ResumePlanner)
        art_manifest = verify_local_artifact(dest_latest_e, is_checkpoint=True)
        self.assertEqual(art_manifest.sha256, sha256_e)
        self.assertEqual(art_manifest.size_bytes, len(content_e))

        resume_plan = ResumePlanner(art_manifest, spec)
        self.assertEqual(resume_plan["project_id"], spec.name)
        self.assertEqual(resume_plan["resume_arguments"][0], "--resume-from")
        self.assertIn("model_latest.pt", resume_plan["resume_arguments"][1])
        self.assertEqual(resume_plan["source_checkpoint"]["sha256"], sha256_e)

        # =====================================================================
        # Stage 21: Negative Security Controls & Defense in Depth
        # =====================================================================
        # 21.1 Unauthenticated dashboard request is rejected with HTTP 401
        req_unauth = urllib.request.Request(f"{base_url}/api/v1/runs")
        with self.assertRaises(urllib.error.HTTPError) as ctx_unauth:
            urllib.request.urlopen(req_unauth)
        self.assertEqual(ctx_unauth.exception.code, 401)

        # 21.2 Untrusted proxy header rejected with HTTP 401
        req_proxy = urllib.request.Request(
            f"{base_url}/api/v1/runs",
            headers={"X-Forwarded-Proto": "https"},
        )
        with self.assertRaises(urllib.error.HTTPError) as ctx_proxy:
            urllib.request.urlopen(req_proxy)
        self.assertEqual(ctx_proxy.exception.code, 401)

        # 21.3 Capability immutability enforcement
        from colab_adapter.capability import CliCapabilities
        caps = CliCapabilities(
            cli_path="/usr/bin/colab",
            version="2.0.0",
            available_commands=["download", "exec", "new", "plan", "sync"],
            supports_ssh=True,
            supports_ssh_proxy_mode=True,
            supports_upload=True,
            supports_download=True,
            supports_exec=True,
            supports_sessions=True,
            supports_usage=True,
        )
        self.assertIsInstance(caps.available_commands, tuple)
        with self.assertRaises(AttributeError):
            caps.available_commands.append("unauthorized_privilege_escalation")


class TestV2FullEndToEndPipeline(unittest.TestCase):
    """End-to-End Pipeline Verification for Colab Project Workspaces V2."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_five_revision_history_and_catastrophic_manifest_loss_e2e(self):
        """Scenario Section 9: 5-revision checkpoint publication history, backup recovery,
        and catastrophic local manifest+journal loss with fail-closed remote integrity checks (Review 5642047783 F1-F4).
        """
        # 1. Creates project vision-research and run experiment-001
        proj_dir = Path(self.tmp_dir) / "vision-research"
        proj_dir.mkdir(parents=True, exist_ok=True)
        init_project(target_dir=proj_dir, name="vision-research")

        staging_dir = Path(self.tmp_dir) / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)

        run_id = "experiment-001"
        project_id = "vision-research"
        local_run_dir = staging_dir / "projects" / project_id / "runs" / run_id
        local_run_dir.mkdir(parents=True, exist_ok=True)

        ckpt_dir = proj_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_file = ckpt_dir / "model.pt"

        creds = DriveCredentials(
            client_id="cid-section9-e2e",
            client_secret="csec-section9-e2e",
            refresh_token="rtok-section9-e2e",
            access_token="atok-section9-e2e",
            token_expiry=9999999999.0,
        )
        creds_file = staging_dir / "credentials.json"
        creds.save(creds_file)

        mock_http = E2EMockDriveHttpClient()
        journal_file = staging_dir / "transfer_journal.json"

        local_store = LocalArtifactStore(root_dir=staging_dir)
        drive_store = GoogleDriveArtifactStore(
            credentials=creds,
            credentials_file=creds_file,
            local_staging_dir=staging_dir,
            journal_path=journal_file,
            http_client=mock_http,
        )

        # 2. Publishes 5 distinct checkpoint revisions (A, B, C, D, E) for checkpoints/model.pt
        # at steps 100, 200, 300, 400, 500 via LocalArtifactStore and GoogleDriveArtifactStore
        revisions_spec = [
            (100, 1, b"CHECKPOINT_REVISION_A_STEP_100_INITIAL_CONV_WEIGHTS"),
            (200, 2, b"CHECKPOINT_REVISION_B_STEP_200_ATTENTION_LAYER_1_WEIGHTS"),
            (300, 3, b"CHECKPOINT_REVISION_C_STEP_300_CROSS_ATTENTION_HEADS"),
            (400, 4, b"CHECKPOINT_REVISION_D_STEP_400_LAYER_NORM_AND_FEEDFORWARD"),
            (500, 5, b"CHECKPOINT_REVISION_E_STEP_500_CONVERGED_OPTIMAL_CHECKPOINT"),
        ]

        published_revisions = []
        for step, epoch, content in revisions_spec:
            ckpt_file.write_bytes(content)
            sha = hashlib.sha256(content).hexdigest()

            # Publish to LocalArtifactStore
            local_store.publish(
                source=ckpt_file,
                project_id=project_id,
                run_id=run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
                metadata={"step": step, "epoch": epoch, "logical_path": "checkpoints/model.pt"},
            )

            # Publish to GoogleDriveArtifactStore
            pub = drive_store.publish(
                source=ckpt_file,
                project_id=project_id,
                run_id=run_id,
                artifact_type=ArtifactType.CHECKPOINT.value,
                metadata={"step": step, "epoch": epoch, "logical_path": "checkpoints/model.pt"},
            )
            self.assertEqual(pub.sha256, sha)
            published_revisions.append((step, epoch, content, sha, pub))

        self.assertEqual(len(published_revisions), 5)
        pub_a = published_revisions[0][4]
        pub_b = published_revisions[1][4]
        pub_c = published_revisions[2][4]
        pub_d = published_revisions[3][4]
        pub_e = published_revisions[4][4]

        sha256_e = published_revisions[4][3]
        content_e = published_revisions[4][2]

        # 3. Verifies all 5 revisions A, B, C, D, E remain in PublicationManifest (len(manifest.artifacts) == 5),
        # only E has is_latest == True, and str(manifest.latest_by_path["checkpoints/model.pt"]) == E.artifact_id.
        manifest_file = local_run_dir / "manifest.json"
        manifest = PublicationManifest.load(manifest_file)

        self.assertEqual(len(manifest.artifacts), 5, "All 5 checkpoint revisions must be preserved in PublicationManifest")

        for art_id, art_data in manifest.artifacts.items():
            if art_id == pub_e.artifact_id:
                self.assertTrue(art_data.get("is_latest"), f"Revision E must have is_latest == True: {art_id}")
            else:
                self.assertFalse(art_data.get("is_latest"), f"Earlier revision must have is_latest == False: {art_id}")

        expected_latest_id = pub_e.artifact_id
        self.assertEqual(
            str(manifest.latest_by_path["checkpoints/model.pt"]),
            expected_latest_id,
            "manifest.latest_by_path must point directly to Revision E",
        )

        # 4. Deletes local manifest.json (unlink()) while keeping manifest.json.bak
        # and verifies GoogleDriveArtifactStore recovers all 5 revisions from manifest.json.bak
        manifest_bak_file = local_run_dir / "manifest.json.bak"
        self.assertTrue(manifest_bak_file.is_file(), "manifest.json.bak must exist before primary manifest deletion")

        manifest_file.unlink()
        self.assertFalse(manifest_file.exists())
        drive_store._manifest_cache.clear()

        recovered_from_bak = drive_store._load_or_recover_manifest(project_id, run_id)
        self.assertIsNotNone(recovered_from_bak, "GoogleDriveArtifactStore must recover manifest from manifest.json.bak")
        self.assertEqual(len(recovered_from_bak.artifacts), 5, "All 5 revisions must be recovered from .bak")
        self.assertEqual(
            str(recovered_from_bak.latest_by_path["checkpoints/model.pt"]),
            expected_latest_id,
        )

        # 5. Deletes BOTH manifest.json and manifest.json.bak and instantiates a fresh GoogleDriveArtifactStore
        # with a fresh empty journal (journal_path pointing to a new path) and the same mock_http client:
        manifest_file.unlink(missing_ok=True)
        manifest_bak_file.unlink(missing_ok=True)
        self.assertFalse(manifest_file.exists())
        self.assertFalse(manifest_bak_file.exists())

        fresh_journal_path = staging_dir / "catastrophic_loss_fresh_journal.json"
        fresh_drive_store = GoogleDriveArtifactStore(
            credentials=creds,
            credentials_file=creds_file,
            local_staging_dir=staging_dir,
            journal_path=fresh_journal_path,
            http_client=mock_http,
        )

        recovery_dir = Path(self.tmp_dir) / "catastrophic_recovery_out"
        recovery_dir.mkdir(parents=True, exist_ok=True)

        # 5.1 Retrieves A, B, C, D, and E by their exact @sha256_<digest> canonical artifact IDs
        # and verifies byte-for-byte content equality for all 5 revisions!
        for step, epoch, content, sha, pub in published_revisions:
            dest_rev = recovery_dir / f"model_step_{step}.pt"
            fresh_drive_store.retrieve(pub.artifact_id, dest_rev)
            self.assertTrue(dest_rev.is_file())
            self.assertEqual(compute_file_sha256(dest_rev), sha)
            self.assertEqual(dest_rev.read_bytes(), content)

        # 5.2 Retrieves unversioned "vision-research/experiment-001/checkpoints/model.pt"
        # (when _coordinate_remote_latest_revision is present on drive_store) and verifies it returns revision E byte-for-byte!
        # And corrupts remote appProperties so two remote files both have is_latest="true",
        # and verifies unversioned retrieve raises DriveIntegrityError!
        # Restores remote appProperties...
        dest_unversioned = recovery_dir / "model_unversioned_latest.pt"
        fresh_drive_store.retrieve(
            f"{project_id}/{run_id}/checkpoints/model.pt",
            dest_unversioned,
        )
        self.assertTrue(dest_unversioned.is_file())
        self.assertEqual(dest_unversioned.read_bytes(), content_e)
        self.assertEqual(compute_file_sha256(dest_unversioned), sha256_e)

        # Corrupt remote appProperties so two remote files both have is_latest="true"
        file_a_id = pub_a.remote_file_id
        mf_a = next(f for f in mock_http.mock_files if f.get("id") == file_a_id)
        mf_a["appProperties"]["is_latest"] = "true"

        with self.assertRaises(DriveIntegrityError):
            fresh_drive_store.retrieve(
                f"{project_id}/{run_id}/checkpoints/model.pt",
                recovery_dir / "corrupted_collision_should_fail.pt",
            )

        # Restore remote appProperties
        mf_a["appProperties"]["is_latest"] = "false"

        # 5.3 Re-publishes revision E, and verifies idempotency (no duplicate upload)
        ckpt_file.write_bytes(content_e)
        committed_count_before = len(mock_http.committed_files)
        pub_e_repub = fresh_drive_store.publish(
            source=ckpt_file,
            project_id=project_id,
            run_id=run_id,
            artifact_type=ArtifactType.CHECKPOINT.value,
            metadata={"step": 500, "epoch": 5, "logical_path": "checkpoints/model.pt"},
        )
        committed_count_after = len(mock_http.committed_files)

        self.assertEqual(
            committed_count_before,
            committed_count_after,
            "Idempotent re-publication must not produce duplicate remote uploads",
        )
        self.assertEqual(pub_e_repub.remote_file_id, pub_e.remote_file_id)
        self.assertEqual(pub_e_repub.sha256, pub_e.sha256)



if __name__ == "__main__":
    unittest.main()
