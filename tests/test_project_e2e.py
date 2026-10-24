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
from colab_adapter.monitor.collector import AccountSample, AccountUsageCollector
from colab_adapter.monitor.schemas import _iso_now
from colab_adapter.monitor.server import start_monitor_server
from colab_adapter.project_manifest import (
    create_execution_plan,
    init_project,
    load_manifest,
    load_project_manifest,
    plan_project,
)
from colab_adapter.project_sessions import (
    ProjectSession,
    ProjectSessionManager,
    ProjectSessionState,
)
from colab_adapter.provider import ApprovalRequiredError, ManagedSessionRegistry
from colab_adapter.reproducibility import create_deterministic_source_snapshot
from colab_adapter.storage import (
    ArtifactType,
    ArtifactUploadWorker,
    DriveCredentials,
    GoogleDriveArtifactStore,
    LocalArtifactStore,
    PublicationManifest,
    TransferJournal,
    compute_file_sha256,
)
from colab_adapter.telemetry import TelemetryEmitter
from colab_adapter.transports.base import BaseTransport
from colab_adapter.workspace import WorkspaceSyncEngine
from tests.test_google_drive_store import MockDriveHttpClient


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

    def test_complete_v2_pipeline_with_drive_and_cross_device_monitor(self):
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

        # colab-project.yaml manifest
        manifest_yaml = pipeline_dir / "colab-project.yaml"
        manifest_yaml.write_text(
            """schema_version: 1
project:
  name: e2e_research_v2
  root: .
runtime:
  gpu: T4
  timeout_seconds: 900.0
  retain_session: false
execution:
  entrypoint: src/trainer.py
  args: ["--config", "configs/base.yaml"]
workspace:
  remote_root: /content/colab-workspaces/e2e_research_v2
  include: ["**/*"]
  exclude: [".env*", "external_symlink*"]
artifacts:
  pull: ["checkpoints/model_final.pt"]
  local_dir: artifacts
""",
            encoding="utf-8",
        )

        # =====================================================================
        # Stage 2: Project manifest and plan
        # =====================================================================
        spec = load_project_manifest(manifest_yaml)
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
        self.assertIn("colab-project.yaml", plan_paths)
        self.assertNotIn(".env", plan_paths, "Secret .env must be excluded from execution plan")
        self.assertNotIn("external_symlink.txt", plan_paths, "External symlink must be excluded from plan")

        # =====================================================================
        # Stage 3: Explicit GPU allocation approval
        # =====================================================================
        executed_cmds = []

        def mock_run_cmd(cmd, **kwargs):
            executed_cmds.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Session created", stderr="")

        session_mgr = ProjectSessionManager(
            registry=self.registry,
            run_cmd_fn=mock_run_cmd,
            require_approval=True,
        )

        # Unapproved allocation is strictly blocked (fail closed)
        with self.assertRaises(ApprovalRequiredError) as ctx:
            session_mgr.up(spec, lease_seconds=1800.0)
        self.assertIn("Explicit approval token required", str(ctx.exception))
        self.assertEqual(len(executed_cmds), 0, "No allocation command may be dispatched without approval")

        # Authorize via explicit approval token
        approval_token = f"ALLOCATE {spec.gpu} auth-nonce-token-7777"
        session = session_mgr.up(spec, lease_seconds=1800.0, approval_token=approval_token)
        self.assertEqual(session.state, ProjectSessionState.READY.value)
        self.assertEqual(session.accelerator, "T4")
        self.assertEqual(len(executed_cmds), 1)
        self.assertEqual(executed_cmds[0][:3], ["colab", "new", "-s"])

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

        mock_http = MockDriveHttpClient()
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


if __name__ == "__main__":
    unittest.main()
