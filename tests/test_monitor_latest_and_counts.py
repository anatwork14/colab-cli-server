"""Regression tests for monitor latest-revision derivation and active-run counts.

P2-a: ``is_latest`` must come only from an authoritative manifest head
(``latest_by_path``). Without a matching head the flag is unknown (None), never
guessed from timestamps, steps or ordering, and the /runs list must not name a
guessed latest checkpoint.

P2-b: legacy keyless (project '') telemetry keys that share a run id with a named
project must not be counted as extra runs, so /runs and /summary agree and the
runtime-active flag is not suppressed by a phantom duplicate.
"""
from __future__ import annotations

import json
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

AUTH_TOKEN = "latest-counts-token-0123456789abcdef"
SHA_A = "1" * 64
SHA_B = "2" * 64
SHA_C = "3" * 64
SHA_ABSENT = "9" * 64
LOGICAL = "checkpoints/model.pt"


class TestMonitorLatestAndCounts(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.artifacts_dir = self.tmp_dir / "artifacts"
        self.ledger = self.tmp_dir / "telemetry.jsonl"
        self.telemetry = TelemetryEmitter(ledger_path=self.ledger, run_id="run_001", project_id="alpha")
        self.registry = ManagedSessionRegistry(self.tmp_dir / "managed_sessions.json")
        self.collector = AccountUsageCollector(ledger_path=self.tmp_dir / "account_usage.jsonl")
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

    def _write_checkpoint_manifest(self, project, run_id, revisions, latest_by_path=None):
        """revisions: list of (sha, created_at, step) in the order they appear in the manifest."""
        run_dir = self._run_dir(project, run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {}
        for sha, created_at, step in revisions:
            key = f"{LOGICAL}@sha256_{sha[:12]}"
            artifacts[key] = {
                "artifact_id": key,
                "logical_path": LOGICAL,
                "sha256": sha,
                "artifact_type": "checkpoint",
                "created_at": created_at,
                "metadata": {"step": step},
                "path": str(run_dir / "checkpoints" / "model.pt"),
            }
        manifest = {"created_at": "2026-10-20T00:00:00+00:00", "artifacts": artifacts}
        if latest_by_path is not None:
            manifest["latest_by_path"] = latest_by_path
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return artifacts

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

    def _artifacts(self, project, run_id):
        status, body = self._get(f"/api/v1/runs/{run_id}/artifacts?project={project}")
        self.assertEqual(status, 200, body)
        return body["artifacts"]

    def _runs(self):
        status, runs = self._get("/api/v1/runs")
        self.assertEqual(status, 200, runs)
        return runs

    def _three_revisions(self):
        # C is the newest by created_at and step; B is deliberately older than C.
        return [
            (SHA_A, "2026-10-19T00:00:00+00:00", 100),
            (SHA_B, "2026-10-19T06:00:00+00:00", 200),
            (SHA_C, "2026-10-19T12:00:00+00:00", 300),
        ]

    # ---- P2-a: is_latest only from an authoritative manifest head -------

    def test_a_no_head_record_makes_every_revision_unknown(self):
        self._write_checkpoint_manifest("alpha", "run_ck", self._three_revisions(), latest_by_path=None)

        arts = self._artifacts("alpha", "run_ck")
        self.assertEqual(len(arts), 3, arts)
        for a in arts:
            self.assertIsNone(a.get("is_latest"), a)
            self.assertEqual(a.get("latest_state"), "unknown", a)

        runs = {r["run_id"]: r for r in self._runs()}
        run = runs["run_ck"]
        self.assertEqual(run["checkpoint_count"], 3)
        self.assertFalse(run.get("latest_checkpoint_name"), run)
        self.assertFalse(run.get("latest_checkpoint_sha256"), run)
        self.assertEqual(run.get("latest_checkpoint_state"), "unknown", run)

    def test_b_manifest_head_is_the_only_latest_even_when_older(self):
        head = f"{LOGICAL}@sha256_{SHA_B[:12]}"
        self._write_checkpoint_manifest("alpha", "run_ck", self._three_revisions(),
                                        latest_by_path={LOGICAL: head})

        by_sha = {a["sha256"]: a for a in self._artifacts("alpha", "run_ck")}
        self.assertIs(by_sha[SHA_B]["is_latest"], True)
        self.assertIs(by_sha[SHA_A]["is_latest"], False)
        self.assertIs(by_sha[SHA_C]["is_latest"], False)
        self.assertEqual(by_sha[SHA_B]["latest_state"], "known")

        run = {r["run_id"]: r for r in self._runs()}["run_ck"]
        self.assertEqual(run["latest_checkpoint_name"], "model.pt")
        self.assertEqual(run["latest_checkpoint_sha256"], SHA_B)
        self.assertEqual(run.get("latest_checkpoint_state"), "known")

    def test_c_head_matching_no_revision_makes_every_revision_unknown(self):
        self._write_checkpoint_manifest(
            "alpha", "run_ck", self._three_revisions(),
            latest_by_path={LOGICAL: f"{LOGICAL}@sha256_{SHA_ABSENT[:12]}"},
        )

        arts = self._artifacts("alpha", "run_ck")
        for a in arts:
            self.assertIsNone(a.get("is_latest"), a)
            self.assertEqual(a.get("latest_state"), "unknown", a)

        run = {r["run_id"]: r for r in self._runs()}["run_ck"]
        self.assertFalse(run.get("latest_checkpoint_name"), run)
        self.assertFalse(run.get("latest_checkpoint_sha256"), run)
        self.assertEqual(run.get("latest_checkpoint_state"), "unknown", run)

    def test_d_ambiguous_head_makes_every_revision_unknown(self):
        # The head string equals one entry's artifact_id and another entry's sha256.
        ambiguous_target = f"{LOGICAL}@sha256_{SHA_B[:12]}"
        revisions = self._three_revisions()
        run_dir = self._run_dir("alpha", "run_ck")
        run_dir.mkdir(parents=True, exist_ok=True)
        artifacts = {
            "k_a": {"artifact_id": f"{LOGICAL}@sha256_{SHA_A[:12]}", "logical_path": LOGICAL,
                    "sha256": SHA_A, "artifact_type": "checkpoint", "created_at": revisions[0][1]},
            "k_b": {"artifact_id": ambiguous_target, "logical_path": LOGICAL,
                    "sha256": SHA_B, "artifact_type": "checkpoint", "created_at": revisions[1][1]},
            "k_c": {"artifact_id": "other-id", "logical_path": LOGICAL,
                    "sha256": ambiguous_target, "artifact_type": "checkpoint", "created_at": revisions[2][1]},
        }
        manifest = {"created_at": "2026-10-20T00:00:00+00:00", "artifacts": artifacts,
                    "latest_by_path": {LOGICAL: ambiguous_target}}
        (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        arts = self._artifacts("alpha", "run_ck")
        self.assertEqual(len(arts), 3, arts)
        for a in arts:
            self.assertIsNone(a.get("is_latest"), a)
            self.assertEqual(a.get("latest_state"), "unknown", a)

    # ---- P2-b: legacy keyless phantoms do not double-count runs --------

    def _emit_legacy(self, run_id):
        legacy = {
            "schema_version": 1, "event_id": f"evt-legacy-{run_id}", "run_id": run_id,
            "emitted_at": "2026-10-20T00:00:00+00:00", "event_type": "train_step",
            "epoch": 1, "step": 1, "metrics": {"loss": 0.7}, "details": {},
        }
        with open(self.ledger, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(legacy) + "\n")

    def test_e_named_run_with_legacy_phantom_is_active_and_counted_once(self):
        self.telemetry.emit(event_type="train_step", epoch=1, step=1, metrics={"loss": 0.3},
                            run_id="run_001", project_id="alpha")
        self._emit_legacy("run_001")
        self.registry.register("run_001", "job-1", "T4", "alpha")
        self.registry.update_state("run_001", "RUNNING")

        runs = self._runs()
        matching = [r for r in runs if r["run_id"] == "run_001"]
        self.assertEqual(len(matching), 1, runs)
        self.assertEqual(matching[0]["project_id"], "alpha")
        self.assertIs(matching[0]["is_runtime_active"], True)

        status, summary = self._get("/api/v1/summary")
        self.assertEqual(status, 200)
        self.assertEqual(summary["total_runs"], len(runs), (summary, runs))

    def test_f_legacy_only_run_is_still_listed(self):
        self.telemetry.emit(event_type="train_step", epoch=1, step=1, metrics={"loss": 0.3},
                            run_id="run_001", project_id="alpha")
        self._emit_legacy("run_legacy")

        runs = {(r["project_id"], r["run_id"]): r for r in self._runs()}
        self.assertIn(("", "run_legacy"), runs)
        self.assertIn(("alpha", "run_001"), runs)
        self.assertEqual(len(runs), 2)

        status, summary = self._get("/api/v1/summary")
        self.assertEqual(status, 200)
        self.assertEqual(summary["total_runs"], 2)


if __name__ == "__main__":
    unittest.main()
