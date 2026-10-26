"""Drive publication status must distinguish NOT RECORDED, PENDING and PUBLISHED (M8 defect).

Unit tests on the derivation helper, plus HTTP checks through /api/v1/runs and /api/v1/runs/<id>/drive.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from colab_adapter.monitor.server import (
    DRIVE_STATUS_NOT_RECORDED,
    DRIVE_STATUS_PENDING,
    DRIVE_STATUS_PUBLISHED,
    _drive_publication_state,
    start_monitor_server,
)
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.telemetry import TelemetryEmitter

TOKEN = "drive-status-test-token-abcdef"


def _local_artifact(name="model.pt", sha="a" * 64):
    return {
        "artifact_id": f"p/r/{name}@sha256_{sha}",
        "logical_path": name,
        "path": f"/x/{name}",
        "sha256": sha,
        "storage_provider": "local",
        "remote_file_id": None,
        "artifact_type": "checkpoint",
    }


class DerivationTests(unittest.TestCase):
    def test_no_drive_information_is_not_recorded(self):
        self.assertEqual(_drive_publication_state(False, None, []), DRIVE_STATUS_NOT_RECORDED)

    def test_in_progress_manifest_default_alone_is_not_pending(self):
        manifest = {"status": "IN_PROGRESS", "artifacts": {"x": _local_artifact()}}
        self.assertEqual(_drive_publication_state(False, manifest, []), DRIVE_STATUS_NOT_RECORDED)

    def test_recorded_interrupted_upload_is_pending(self):
        art = _local_artifact()
        art["metadata"] = {"drive_publication_status": "UPLOAD_PENDING"}
        manifest = {"status": "IN_PROGRESS", "artifacts": {"x": art}}
        self.assertEqual(_drive_publication_state(False, manifest, []), DRIVE_STATUS_PENDING)

    def test_recorded_retry_in_telemetry_is_pending(self):
        events = [{"event_type": "upload", "details": {"transfer_status": "RETRY_PENDING"}}]
        self.assertEqual(_drive_publication_state(False, None, events), DRIVE_STATUS_PENDING)

    def test_failed_publication_manifest_is_pending(self):
        self.assertEqual(
            _drive_publication_state(False, {"status": "PARTIAL", "artifacts": {}}, []),
            DRIVE_STATUS_PENDING,
        )

    def test_drive_backed_artifact_is_published_even_after_interruption(self):
        art = _local_artifact()
        art["metadata"] = {"drive_publication_status": "RETRY_PENDING"}
        self.assertEqual(_drive_publication_state(True, {"artifacts": {"x": art}}, []), DRIVE_STATUS_PUBLISHED)

    def test_published_value_is_backward_compatible(self):
        self.assertEqual(DRIVE_STATUS_PUBLISHED, "PUBLISHED")
        self.assertEqual(DRIVE_STATUS_PENDING, "PENDING")


class DriveStatusHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="colab_drive_status_"))
        self.artifacts = self.tmp / "artifacts"
        self.server = None

    def tearDown(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_run(self, project, run, manifest):
        run_dir = self.artifacts / "projects" / project / "runs" / run
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _get(self, path):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.server.server_port}{path}",
            headers={"Authorization": f"Bearer {TOKEN}", "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _start(self):
        self.server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=_NullCollector(self.tmp / "ledger.jsonl"),
            registry=ManagedSessionRegistry(self.tmp / "sessions.json"),
            telemetry=TelemetryEmitter(ledger_path=self.tmp / "telemetry.jsonl", run_id="r-events"),
            auth_token=TOKEN,
            artifacts_dir=self.artifacts,
            poll_interval=None,
        )
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def test_runs_list_distinguishes_three_states(self):
        self._write_run("proj", "run-local", {
            "project_id": "proj", "run_id": "run-local", "status": "IN_PROGRESS",
            "artifacts": {"a": _local_artifact("local.pt")}, "latest_by_path": {},
        })
        pending_art = _local_artifact("pend.pt", sha="b" * 64)
        pending_art["metadata"] = {"drive_publication_status": "UPLOAD_PENDING"}
        self._write_run("proj", "run-pending", {
            "project_id": "proj", "run_id": "run-pending", "status": "IN_PROGRESS",
            "artifacts": {"a": pending_art}, "latest_by_path": {},
        })
        drive_art = _local_artifact("done.pt", sha="c" * 64)
        drive_art.update({"storage_provider": "google_drive", "remote_file_id": "file-1"})
        self._write_run("proj", "run-published", {
            "project_id": "proj", "run_id": "run-published", "status": "PUBLISHED",
            "artifacts": {"a": drive_art}, "latest_by_path": {},
        })
        self._start()
        runs = {r["run_id"]: r for r in self._get("/api/v1/runs")}
        self.assertEqual(runs["run-local"]["drive_publication_status"], "NOT RECORDED")
        self.assertEqual(runs["run-pending"]["drive_publication_status"], "PENDING")
        self.assertEqual(runs["run-published"]["drive_publication_status"], "PUBLISHED")

    def test_run_drive_endpoint_reports_not_recorded_without_drive_info(self):
        self._write_run("proj", "run-only-local", {
            "project_id": "proj", "run_id": "run-only-local", "status": "IN_PROGRESS",
            "artifacts": {"a": _local_artifact("only.pt")}, "latest_by_path": {},
        })
        self._start()
        body = self._get("/api/v1/runs/run-only-local/drive?project=proj")
        self.assertEqual(body["drive_publication_status"], "NOT RECORDED")
        self.assertEqual(body["published_count"], 0)


class _NullCollector:
    """Collector stand-in with an empty ledger; never runs a CLI."""

    def __init__(self, ledger_path):
        self.ledger_path = ledger_path
        self.monitoring_since = "2026-10-01T00:00:00+00:00"
        self.backoff_level = 0
        self.is_stale = True

    def get_samples(self, **kwargs):
        return []

    def get_latest_sample(self):
        return None

    def poll_once(self, unmanaged_sessions_count=0):  # pragma: no cover - polling disabled in tests
        raise AssertionError("poll_once must not run in this test")


if __name__ == "__main__":
    unittest.main()
