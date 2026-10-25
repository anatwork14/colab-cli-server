"""Regression tests for monitor review follow-ups (project isolation hardening).

Covers: storage-internal dirs (.locks) excluded from run artifacts, single
project/run prefix on on-disk artifact ids, symlinks and large unmanifested
files in run dirs, project-aware runtime-active flag, summary project filter,
legacy keyless events not causing 409 ambiguity, and strict identifier
validation (re.fullmatch, so a trailing newline is rejected).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from colab_adapter.monitor import server as monitor_server
from colab_adapter.monitor.collector import AccountUsageCollector
from colab_adapter.monitor.server import start_monitor_server
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.telemetry import TelemetryEmitter

SHA_ALPHA = "a" * 64
SHA_BETA = "b" * 64
AUTH_TOKEN = "review-followup-token-0123456789"
MIB = 1024 * 1024


class TestMonitorReviewFollowups(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.artifacts_dir = self.tmp_dir / "artifacts"
        self.telemetry = TelemetryEmitter(
            ledger_path=self.tmp_dir / "telemetry.jsonl",
            run_id="run_001",
            project_id="alpha",
        )
        self.registry = ManagedSessionRegistry(self.tmp_dir / "managed_sessions.json")
        self.collector = AccountUsageCollector(ledger_path=self.tmp_dir / "account_usage.jsonl")

        self.telemetry.emit(event_type="train_step", epoch=1, step=1, metrics={"loss": 0.1},
                            run_id="run_001", project_id="alpha")
        self.telemetry.emit(event_type="train_step", epoch=1, step=1, metrics={"loss": 0.9},
                            run_id="run_001", project_id="beta")
        self.telemetry.emit(event_type="train_step", epoch=2, step=5, metrics={"loss": 0.5},
                            run_id="run_002", project_id="alpha")
        self._write_manifest("alpha", "run_001", SHA_ALPHA)
        self._write_manifest("beta", "run_001", SHA_BETA)
        self._write_manifest("alpha", "run_002", SHA_ALPHA)

        self.server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=self.collector,
            registry=self.registry,
            telemetry=self.telemetry,
            auth_token=AUTH_TOKEN,
            allow_external=False,
            artifacts_dir=self.artifacts_dir,
        )
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # ---- helpers -------------------------------------------------------

    def _run_dir(self, project, run_id):
        return self.artifacts_dir / "projects" / project / "runs" / run_id

    def _write_manifest(self, project, run_id, sha):
        run_dir = self._run_dir(project, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "created_at": "2026-10-20T00:00:00+00:00",
            "artifacts": {
                "model.pt": {
                    "artifact_id": f"model.pt@sha256_{sha}",
                    "logical_path": "model.pt",
                    "sha256": sha,
                    "artifact_type": "checkpoint",
                    "path": str(run_dir / "model.pt"),
                }
            },
            "latest_by_path": {"model.pt": f"model.pt@sha256_{sha}"},
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _get(self, path, token=AUTH_TOKEN):
        req_headers = {}
        if token is not None:
            req_headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(f"{self.base_url}{path}", headers=req_headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8")
            try:
                return exc.code, json.loads(body)
            except ValueError:
                return exc.code, body

    def _local_artifacts(self, project, run_id):
        status, body = self._get(f"/api/v1/runs/{run_id}/artifacts?project={project}")
        self.assertEqual(status, 200)
        return body["artifacts"]

    # ---- 1. .locks excluded, single id prefix --------------------------

    def test_locks_dir_is_not_listed_and_ids_are_prefixed_once(self):
        run_dir = self._run_dir("alpha", "run_001")
        (run_dir / ".locks").mkdir()
        (run_dir / ".locks" / "manifest.lock").write_text("", encoding="utf-8")
        (run_dir / "out").mkdir()
        (run_dir / "out" / "result.txt").write_text("hello", encoding="utf-8")

        arts = self._local_artifacts("alpha", "run_001")
        ids = [a["artifact_id"] for a in arts]
        self.assertTrue(all(i.startswith("alpha/run_001/") for i in ids), ids)
        self.assertFalse([i for i in ids if ".locks" in i], ids)
        self.assertFalse([i for i in ids if "run_001/run_001" in i], ids)
        self.assertIn("alpha/run_001/out/result.txt", ids)

    # ---- 2. symlinks and unmanifested hashing --------------------------

    def test_symlinked_file_in_run_dir_is_not_listed_or_hashed(self):
        secret = self.tmp_dir / "outside_secret.bin"
        secret.write_bytes(b"beta-secret-bytes")
        os.symlink(secret, self._run_dir("alpha", "run_001") / "leak.bin")

        arts = self._local_artifacts("alpha", "run_001")
        self.assertFalse([a for a in arts if "leak.bin" in (a.get("artifact_id") or "")], arts)
        leaked_sha = hashlib.sha256(b"beta-secret-bytes").hexdigest()
        self.assertNotIn(leaked_sha, {a.get("sha256") for a in arts})

    def test_large_unmanifested_file_reports_size_without_sha(self):
        big = self._run_dir("alpha", "run_001") / "big" / "blob.bin"
        big.parent.mkdir(parents=True)
        with open(big, "wb") as fh:
            fh.truncate(9 * MIB)

        small = self._run_dir("alpha", "run_001") / "out" / "small.txt"
        small.parent.mkdir(parents=True)
        small.write_bytes(b"small")

        arts = {a["artifact_id"]: a for a in self._local_artifacts("alpha", "run_001")}
        big_art = arts["alpha/run_001/big/blob.bin"]
        self.assertEqual(big_art["size_bytes"], 9 * MIB)
        self.assertIsNone(big_art["sha256"])
        small_art = arts["alpha/run_001/out/small.txt"]
        self.assertEqual(small_art["sha256"], hashlib.sha256(b"small").hexdigest())

    # ---- 3. runtime-active is project-aware ----------------------------

    def test_same_run_id_in_two_projects_is_not_both_active(self):
        self.registry.register("run_001", "job-1", "T4", "alpha")
        self.registry.update_state("run_001", "RUNNING")
        self.registry.register("run_002", "job-2", "T4", "alpha")
        self.registry.update_state("run_002", "RUNNING")

        status, runs = self._get("/api/v1/runs")
        self.assertEqual(status, 200)
        by_key = {(r["project_id"], r["run_id"]): r for r in runs}
        self.assertFalse(by_key[("alpha", "run_001")]["is_runtime_active"])
        self.assertFalse(by_key[("beta", "run_001")]["is_runtime_active"])
        self.assertTrue(by_key[("alpha", "run_002")]["is_runtime_active"])

    # ---- 4. summary honours project filter for run counts --------------

    def test_summary_total_runs_honours_project_filter(self):
        for i, project in enumerate(("alpha", "beta", "gamma")):
            self.registry.register(f"sess-{i}", f"job-{i}", "T4", project)
            self.registry.update_state(f"sess-{i}", "RUNNING")

        status, body = self._get("/api/v1/summary?project=beta")
        self.assertEqual(status, 200)
        self.assertEqual(body["total_runs"], 1)
        self.assertEqual(body["managed_counts_scope"], "account")

        status, body = self._get("/api/v1/summary")
        self.assertEqual(status, 200)
        self.assertEqual(body["total_runs"], 3)

    # ---- 5. legacy keyless events do not cause 409 --------------------

    def test_legacy_keyless_event_does_not_make_named_run_ambiguous(self):
        legacy = {
            "schema_version": 1, "event_id": "evt-legacy-2", "run_id": "run_002",
            "emitted_at": "2026-10-20T00:00:00+00:00", "event_type": "train_step",
            "epoch": 1, "step": 1, "metrics": {"loss": 0.7}, "details": {},
        }
        with open(self.tmp_dir / "telemetry.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(legacy) + "\n")

        status, body = self._get("/api/v1/runs/run_002")
        self.assertEqual(status, 200, body)
        self.assertEqual(body["project_id"], "alpha")

        status, metrics = self._get("/api/v1/runs/run_002/metrics")
        self.assertEqual(status, 200)
        self.assertEqual([p["metrics"].get("loss") for p in metrics], [0.5])

    # ---- 6. strict identifier validation -------------------------------

    def test_identifier_with_trailing_newline_is_rejected(self):
        self.assertFalse(monitor_server._is_valid_identifier("run\n"))
        self.assertFalse(monitor_server._is_valid_identifier("proj\n"))
        self.assertTrue(monitor_server._is_valid_identifier("run_001"))
        self.assertFalse(monitor_server._is_valid_identifier("rün"))

    def test_project_with_trailing_newline_is_400(self):
        for path in (
            "/api/v1/runs?project=alpha%0A",
            "/api/v1/summary?project=alpha%0A",
            "/api/v1/runs/run_001?project=alpha%0A",
            "/api/v1/runs/run_001/metrics?project=beta%0A",
        ):
            with self.subTest(path=path):
                status, _ = self._get(path)
                self.assertEqual(status, 400)


if __name__ == "__main__":
    unittest.main()
