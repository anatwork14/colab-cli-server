"""Integration and contract test suite for Colab Monitor Dashboard UI (Milestone M8).

Verifies:
1. `DASHBOARD_HTML` and `render_dashboard_html()` structure and invariants:
   - Contains responsive viewport and `@media (max-width: 768px)` rules.
   - Uses `sessionStorage` and `Authorization: Bearer ` header.
   - Strictly does NOT contain `URLSearchParams` or `location.search` token extraction.
   - Contains all required sections:
     - Compute Units Balance
     - Consumption Rate
     - Active Assignments
     - Active Managed Sessions
     - Training Runs
     - Loss & Evaluation Metrics / Loss/Metrics
     - Execution Logs & Failures Viewer / Logs viewer
     - Checkpoints & Google Drive Artifacts
   - Honest telemetry contracts:
     - "No historical compute samples observed yet"
     - "GPU utilization: not measured"
   - Interactive buttons for "Export Compute CSV" and "Auth / Token".

2. Full-stack data integration with `MonitorServer`:
   - `AccountUsageCollector` sample: balance_cu=84.5, rate_cu_per_hour=1.9, active_assignments=1
   - `ManagedSessionRegistry` active session: session-t4-01, project vision_v2, GPU T4
   - `TelemetryEmitter` events: run_started, metric (loss=0.45, val_loss=0.51), checkpoint_saved, log_line, run_finished
   - `PublicationManifest` on disk: verified checkpoint on Google Drive (remote_file_id="drive-file-999") and source_snapshot
   - Query all endpoints (/api/v1/health, /api/v1/summary, /api/v1/compute/samples,
     /api/v1/sessions, /api/v1/runs, /api/v1/runs/<run_id>, /api/v1/runs/<run_id>/metrics,
     /api/v1/runs/<run_id>/logs, /api/v1/runs/<run_id>/drive, /api/v1/export/compute.csv)
     and assert payloads match the exact schemas consumed by `ui.py`.
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from colab_adapter.monitor.collector import AccountSample, AccountUsageCollector
from colab_adapter.monitor.schemas import _iso_now
from colab_adapter.monitor.server import MonitorServer, start_monitor_server
from colab_adapter.monitor.ui import DASHBOARD_HTML, render_dashboard_html
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.storage.base import StoredArtifact
from colab_adapter.storage.manifest import PublicationManifest
from colab_adapter.telemetry import TelemetryEmitter


class TestDashboardUIIntegration(unittest.TestCase):
    """Test suite for Colab Monitor Dashboard UI integration and security invariants."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="colab_test_ui_")
        self.base_path = Path(self.tmp_dir)

        self.ledger_path = self.base_path / "account_usage.jsonl"
        self.telemetry_path = self.base_path / "telemetry_events.jsonl"
        self.registry_path = self.base_path / "managed_sessions.json"
        self.artifacts_dir = self.base_path / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.auth_token = "secret-bearer-test-token-xyz-789"
        self.run_id = "run-vision-001"
        self.project_id = "vision_v2"
        self.session_id = "session-t4-01"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # =========================================================================
    # Part 1: DASHBOARD_HTML and render_dashboard_html() validity & contracts
    # =========================================================================

    def test_dashboard_html_and_render_function(self):
        """Verify render_dashboard_html returns valid HTML and supports title override."""
        default_html = render_dashboard_html()
        self.assertIsInstance(default_html, str)
        self.assertIn("<!DOCTYPE html>", default_html)
        self.assertIn("<title>Colab Monitor — V2 Remote GPU Workspaces</title>", default_html)

        custom_html = render_dashboard_html(title="Custom Title - My Research Lab")
        self.assertIn("<title>Custom Title - My Research Lab</title>", custom_html)
        self.assertNotIn("<title>Colab Monitor — V2 Remote GPU Workspaces</title>", custom_html)

    def test_responsive_layout_and_media_queries(self):
        """Verify responsive viewport meta tag and @media (max-width: 768px) styling."""
        html = DASHBOARD_HTML
        self.assertIn('<meta name="viewport" content="width=device-width, initial-scale=1.0">', html)
        self.assertIn("@media (max-width: 768px)", html)

    def test_security_auth_storage_and_zero_url_token_extraction(self):
        """Verify tokens stored via sessionStorage only and URL query extraction is rejected."""
        html = DASHBOARD_HTML

        # Must use sessionStorage and Bearer authorization header
        self.assertIn("sessionStorage.getItem('colab_monitor_token')", html)
        self.assertIn("'Authorization': 'Bearer ' + token", html)

        # Must NOT extract tokens from location.search or URLSearchParams
        self.assertNotIn("URLSearchParams", html)
        self.assertNotIn("location.search", html)
        self.assertNotIn("window.location.search", html)

        # Interactive Auth button and prompt
        self.assertIn('id="btn-auth"', html)
        self.assertIn("Auth / Token", html)
        self.assertIn("promptAuthToken", html)

        # CSV export must use fetch with Bearer headers and Blob download, never navigating with token
        self.assertIn('id="btn-export-csv"', html)
        self.assertIn("Export Compute CSV", html)
        self.assertIn("exportComputeCsv", html)
        self.assertIn("window.URL.createObjectURL(blob)", html)

    def test_required_dashboard_sections_presence(self):
        """Verify all 8 required monitoring sections and KPI cards exist in DASHBOARD_HTML."""
        html = DASHBOARD_HTML

        # 1. Compute Units Balance
        self.assertIn("Compute Units Balance", html)
        self.assertIn('id="val-cu"', html)

        # 2. Consumption Rate
        self.assertIn("Consumption Rate", html)
        self.assertIn('id="val-rate"', html)

        # 3. Active Assignments
        self.assertIn("Active Assignments", html)
        self.assertIn('id="val-assignments"', html)

        # 4. Active Managed Sessions
        self.assertIn("Active Managed Sessions", html)
        self.assertIn('id="sessions-table"', html)

        # 5. Training Runs
        self.assertIn("Training Runs", html)
        self.assertIn('id="runs-table"', html)

        # 6. Loss & Evaluation Metrics / Loss/Metrics
        self.assertIn("Loss & Evaluation Metrics", html)
        self.assertIn("Loss/Metrics", html)
        self.assertIn('id="run-metrics-chart-container"', html)

        # 7. Execution Logs & Failures Viewer / Logs viewer
        self.assertIn("Execution Logs & Failures Viewer", html)
        self.assertIn('id="log-console"', html)

        # 8. Checkpoints & Google Drive Artifacts
        self.assertIn("Checkpoints & Google Drive Artifacts", html)
        self.assertIn('id="artifacts-table"', html)

    def test_honest_telemetry_guarantees_in_ui(self):
        """Verify UI displays explicit honest indicators when telemetry is absent."""
        html = DASHBOARD_HTML

        # Compute balance history: explicitly shows message when empty
        self.assertIn("No historical compute samples observed yet", html)

        # GPU utilization: explicitly indicates unmeasured state rather than fabricating 0% or fake VRAM
        self.assertIn("GPU utilization: not measured", html)

    # =========================================================================
    # Part 2: Full-Stack Data Integration with MonitorServer
    # =========================================================================

    def test_full_stack_data_integration_and_schema_conformance(self):
        """Start MonitorServer with full mock environment and verify every endpoint against UI schemas."""
        # 1. Populate AccountUsageCollector with sample (balance_cu=84.5, rate_cu_per_hour=1.9, active_assignments=1)
        collector = AccountUsageCollector(ledger_path=self.ledger_path)
        sample = AccountSample(
            sample_id="sample-seed-01",
            observed_at=_iso_now(),
            balance_cu=84.5,
            rate_cu_per_hour=1.9,
            active_assignments=1,
            net_balance_change=-0.1,
            change_type="CONSUMPTION",
            status="OK",
        )
        collector._record_sample(sample)
        collector._last_sample = sample

        # 2. Populate ManagedSessionRegistry with active session (session-t4-01, project vision_v2, GPU T4)
        registry = ManagedSessionRegistry(self.registry_path)
        registry.register(
            session_id=self.session_id,
            job_id="job-t4-01",
            gpu="T4",
            project_name=self.project_id,
        )
        registry.update_state(self.session_id, "READY")

        # 3. Populate TelemetryEmitter events
        emitter = TelemetryEmitter(
            ledger_path=self.telemetry_path,
            run_id=self.run_id,
            project_id=self.project_id,
        )
        emitter.emit("run_started")
        emitter.emit("metric", epoch=1, step=10, metrics={"loss": 0.45, "val_loss": 0.51})
        emitter.emit("checkpoint_saved", epoch=1, step=10)
        emitter.emit("log_line", details={"log_line": "Epoch 1 step 10 finished: loss=0.45, val_loss=0.51"})
        emitter.emit("run_finished")

        # 4. Populate PublicationManifest on disk with verified checkpoint (google_drive) and source_snapshot
        run_artifacts_dir = self.artifacts_dir / "projects" / self.project_id / "runs" / self.run_id
        run_artifacts_dir.mkdir(parents=True, exist_ok=True)

        manifest = PublicationManifest(
            project_id=self.project_id,
            run_id=self.run_id,
            git_commit="3cfc4cfeba1cde428012ad965e562acb8bfcd26c",
            git_branch="work/worker-d-ui",
            git_dirty=False,
            environment={"python": "3.8", "cuda": "12.2"},
        )
        chk_artifact = StoredArtifact(
            artifact_id=f"{self.project_id}/{self.run_id}/checkpoints/model_ep1.pt",
            storage_provider="google_drive",
            path=str(run_artifacts_dir / "checkpoints" / "model_ep1.pt"),
            size_bytes=2048,
            sha256="13582fd7670af844790c3316424576dd29b6024e0ef421dcb54dc1efec5f0a8d",
            project_id=self.project_id,
            run_id=self.run_id,
            artifact_type="checkpoint",
            remote_file_id="drive-file-999",
            verified_at="2026-10-24T16:05:00Z",
        )
        src_artifact = StoredArtifact(
            artifact_id=f"{self.project_id}/{self.run_id}/source_snapshot.tar.gz",
            storage_provider="local",
            path=str(run_artifacts_dir / "source_snapshot.tar.gz"),
            size_bytes=4096,
            sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            project_id=self.project_id,
            run_id=self.run_id,
            artifact_type="source_snapshot",
            verified_at="2026-10-24T16:00:00Z",
        )
        manifest.add_artifact(chk_artifact)
        manifest.add_artifact(src_artifact)
        manifest.save(run_artifacts_dir / "manifest.json")

        # Start MonitorServer on dynamic loopback port
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=collector,
            registry=registry,
            telemetry=emitter,
            auth_token=self.auth_token,
            allow_external=False,
            artifacts_dir=self.artifacts_dir,
        )
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"

        try:
            headers = {"Authorization": f"Bearer {self.auth_token}", "Accept": "application/json"}

            def http_get_json(path: str):
                req = urllib.request.Request(f"{base_url}{path}", headers=headers)
                with urllib.request.urlopen(req) as resp:
                    self.assertEqual(resp.status, 200)
                    return json.loads(resp.read().decode("utf-8"))

            # --- 1. GET /api/v1/health ---
            health = http_get_json("/api/v1/health")
            self.assertIn(health["status"], ("HEALTHY", "DEGRADED", "ERROR"))
            self.assertEqual(health["status"], "HEALTHY")
            self.assertFalse(health["is_stale"])
            self.assertEqual(health["sample_count"], 1)
            self.assertEqual(health["collector_backoff_level"], 0)
            self.assertIsNotNone(health["last_collected_at"])
            self.assertIsInstance(health["uptime_seconds"], (int, float))

            # --- 2. GET /api/v1/summary ---
            summary = http_get_json("/api/v1/summary")
            self.assertEqual(summary["current_balance_cu"], 84.5)
            self.assertEqual(summary["current_rate_cu_per_hour"], 1.9)
            self.assertEqual(summary["active_assignments"], 1)
            self.assertEqual(summary["managed_running_runs"], 1)  # session-t4-01 in READY state
            self.assertGreaterEqual(summary["total_runs"], 1)
            self.assertFalse(summary["is_stale"])

            # --- 3. GET /api/v1/compute/samples?limit=240 ---
            samples = http_get_json("/api/v1/compute/samples?limit=240")
            self.assertIsInstance(samples, list)
            self.assertGreaterEqual(len(samples), 1)
            sample0 = samples[0]
            self.assertEqual(sample0["sample_id"], "sample-seed-01")
            self.assertEqual(sample0["balance_cu"], 84.5)
            self.assertEqual(sample0["rate_cu_per_hour"], 1.9)
            self.assertEqual(sample0["active_assignments"], 1)
            self.assertIn("observed_at", sample0)

            # --- 4. GET /api/v1/sessions ---
            sessions = http_get_json("/api/v1/sessions")
            self.assertIsInstance(sessions, list)
            self.assertGreaterEqual(len(sessions), 1)
            sess0 = next(s for s in sessions if s["session_id"] == self.session_id)
            self.assertEqual(sess0["project_name"], self.project_id)
            self.assertEqual(sess0["accelerator"], "T4")
            self.assertEqual(sess0["state"], "READY")
            self.assertTrue(sess0["managed"])

            # --- 5. GET /api/v1/runs ---
            runs = http_get_json("/api/v1/runs")
            self.assertIsInstance(runs, list)
            run0 = next(r for r in runs if r["run_id"] == self.run_id)
            self.assertEqual(run0["project_id"], self.project_id)
            self.assertEqual(run0["status"], "COMPLETED")
            self.assertGreaterEqual(run0["checkpoint_count"], 1)
            self.assertEqual(run0["drive_publication_status"], "PUBLISHED")
            self.assertTrue(run0["source_snapshot_available"])
            self.assertEqual(run0["latest_metrics"].get("loss"), 0.45)
            self.assertEqual(run0["latest_metrics"].get("val_loss"), 0.51)

            # --- 6. GET /api/v1/runs/<run_id> ---
            run_detail = http_get_json(f"/api/v1/runs/{self.run_id}")
            self.assertEqual(run_detail["run_id"], self.run_id)
            self.assertEqual(run_detail["artifacts_count"], 2)
            self.assertEqual(len(run_detail["artifacts"]), 2)
            self.assertGreaterEqual(len(run_detail["events"]), 5)

            # --- 7. GET /api/v1/runs/<run_id>/metrics ---
            metrics_list = http_get_json(f"/api/v1/runs/{self.run_id}/metrics")
            self.assertIsInstance(metrics_list, list)
            self.assertGreaterEqual(len(metrics_list), 1)
            m0 = metrics_list[0]
            self.assertEqual(m0["epoch"], 1)
            self.assertEqual(m0["step"], 10)
            self.assertEqual(m0["metrics"]["loss"], 0.45)
            self.assertEqual(m0["metrics"]["val_loss"], 0.51)
            # Invariant: GPU metrics were not emitted, so neither gpu_util_pct nor gpu_mem_mb is present
            self.assertNotIn("gpu_util_pct", m0["metrics"])
            self.assertNotIn("gpu_mem_mb", m0["metrics"])

            # --- 8. GET /api/v1/runs/<run_id>/logs ---
            logs_res = http_get_json(f"/api/v1/runs/{self.run_id}/logs")
            self.assertEqual(logs_res["run_id"], self.run_id)
            self.assertIn("logs", logs_res)
            self.assertGreaterEqual(len(logs_res["logs"]), 1)
            self.assertIn("Epoch 1 step 10 finished: loss=0.45, val_loss=0.51", logs_res["logs"][0])

            # --- 9. GET /api/v1/runs/<run_id>/drive ---
            drive_res = http_get_json(f"/api/v1/runs/{self.run_id}/drive")
            self.assertEqual(drive_res["run_id"], self.run_id)
            self.assertEqual(drive_res["drive_publication_status"], "PUBLISHED")
            self.assertEqual(drive_res["published_count"], 1)
            self.assertEqual(drive_res["artifacts"][0]["remote_file_id"], "drive-file-999")

            # --- 10. GET /api/v1/runs/<run_id>/artifacts ---
            artifacts_res = http_get_json(f"/api/v1/runs/{self.run_id}/artifacts")
            self.assertEqual(artifacts_res["run_id"], self.run_id)
            self.assertEqual(len(artifacts_res["artifacts"]), 2)
            art_map = {a["artifact_type"]: a for a in artifacts_res["artifacts"]}
            self.assertIn("checkpoint", art_map)
            self.assertEqual(art_map["checkpoint"]["storage_provider"], "google_drive")
            self.assertEqual(art_map["checkpoint"]["size_bytes"], 2048)
            self.assertEqual(art_map["checkpoint"]["sha256"], "13582fd7670af844790c3316424576dd29b6024e0ef421dcb54dc1efec5f0a8d")
            self.assertIn("source_snapshot", art_map)
            self.assertEqual(art_map["source_snapshot"]["storage_provider"], "local")
            self.assertEqual(art_map["source_snapshot"]["size_bytes"], 4096)

            # --- 11. GET /api/v1/export/compute.csv ---
            req_csv = urllib.request.Request(f"{base_url}/api/v1/export/compute.csv", headers=headers)
            with urllib.request.urlopen(req_csv) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get("Content-Type"), "text/csv; charset=utf-8")
                csv_body = resp.read().decode("utf-8")
                self.assertIn("sample_id,observed_at,collected_at,source,balance_cu", csv_body)
                self.assertIn("sample-seed-01", csv_body)
                self.assertIn("84.5", csv_body)
                self.assertIn("1.9", csv_body)

        finally:
            server.shutdown()

    # =========================================================================
    # Part 3: Security & Rejection Invariants
    # =========================================================================

    def test_security_rejections_for_unauthorized_and_query_token(self):
        """Verify strict 401 on missing auth, rejection of ?token= parameter, and 405 on non-GET."""
        collector = AccountUsageCollector(ledger_path=self.ledger_path)
        registry = ManagedSessionRegistry(self.registry_path)
        emitter = TelemetryEmitter(ledger_path=self.telemetry_path, run_id="r1")

        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=collector,
            registry=registry,
            telemetry=emitter,
            auth_token=self.auth_token,
            allow_external=False,
        )
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"

        try:
            # 1. Unauthenticated request -> 401
            req_unauth = urllib.request.Request(f"{base_url}/api/v1/summary")
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req_unauth)
            self.assertEqual(ctx.exception.code, 401)

            # 2. Query parameter ?token= is strictly rejected -> 401
            req_query = urllib.request.Request(f"{base_url}/api/v1/summary?token={self.auth_token}")
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req_query)
            self.assertEqual(ctx.exception.code, 401)

            # 3. CSV export without token -> 401
            req_csv_unauth = urllib.request.Request(f"{base_url}/api/v1/export/compute.csv")
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req_csv_unauth)
            self.assertEqual(ctx.exception.code, 401)

            # 4. POST request -> 405 Method Not Allowed (read-only guarantee)
            req_post = urllib.request.Request(
                f"{base_url}/api/v1/runs",
                data=b'{"action": "kill"}',
                headers={"Authorization": f"Bearer {self.auth_token}", "Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req_post)
            self.assertEqual(ctx.exception.code, 405)

        finally:
            server.shutdown()

    def test_empty_states_contract_for_compute_and_sessions(self):
        """Verify empty states contract when no compute samples or active sessions exist."""
        collector = AccountUsageCollector(ledger_path=self.ledger_path)
        registry = ManagedSessionRegistry(self.registry_path)
        emitter = TelemetryEmitter(ledger_path=self.telemetry_path, run_id="r-empty")

        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=collector,
            registry=registry,
            telemetry=emitter,
            auth_token=self.auth_token,
            allow_external=False,
        )
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"

        try:
            headers = {"Authorization": f"Bearer {self.auth_token}", "Accept": "application/json"}

            req_samples = urllib.request.Request(f"{base_url}/api/v1/compute/samples", headers=headers)
            with urllib.request.urlopen(req_samples) as resp:
                samples = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(samples, [])

            req_sessions = urllib.request.Request(f"{base_url}/api/v1/sessions", headers=headers)
            with urllib.request.urlopen(req_sessions) as resp:
                sessions = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(sessions, [])

            req_summary = urllib.request.Request(f"{base_url}/api/v1/summary", headers=headers)
            with urllib.request.urlopen(req_summary) as resp:
                summary = json.loads(resp.read().decode("utf-8"))
                self.assertIsNone(summary["current_balance_cu"])
                self.assertEqual(summary["active_assignments"], 0)

            # Invariant: Empty states match explicit UI contract labels
            self.assertIn("No historical compute samples observed yet", DASHBOARD_HTML)
            self.assertIn("No active managed Colab sessions", DASHBOARD_HTML)
            self.assertIn("Unavailable", DASHBOARD_HTML)

        finally:
            server.shutdown()

    def test_gpu_telemetry_measured_vs_unmeasured_contract(self):
        """Verify GPU telemetry is only present when explicitly measured by instrumentation."""
        collector = AccountUsageCollector(ledger_path=self.ledger_path)
        registry = ManagedSessionRegistry(self.registry_path)
        emitter = TelemetryEmitter(
            ledger_path=self.telemetry_path,
            run_id="run-gpu-meas-01",
            project_id="vision_v2",
        )
        # Emit an event with real GPU metrics
        emitter.emit(
            "metric",
            epoch=1,
            step=20,
            metrics={"loss": 0.31, "gpu_util_pct": 82.5, "gpu_mem_mb": 9420},
        )

        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=collector,
            registry=registry,
            telemetry=emitter,
            auth_token=self.auth_token,
            allow_external=False,
        )
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"

        try:
            headers = {"Authorization": f"Bearer {self.auth_token}", "Accept": "application/json"}
            req_metrics = urllib.request.Request(f"{base_url}/api/v1/runs/run-gpu-meas-01/metrics", headers=headers)
            with urllib.request.urlopen(req_metrics) as resp:
                metrics_list = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(len(metrics_list), 1)
                pt = metrics_list[0]
                self.assertEqual(pt["metrics"]["gpu_util_pct"], 82.5)
                self.assertEqual(pt["metrics"]["gpu_mem_mb"], 9420)

            # Invariant: UI JavaScript contains logic to render measured GPU vs "GPU utilization: not measured"
            self.assertIn("GPU utilization: not measured", DASHBOARD_HTML)
            self.assertIn("GPU Utilization:", DASHBOARD_HTML)
            self.assertIn("gpu_util_pct", DASHBOARD_HTML)

        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
