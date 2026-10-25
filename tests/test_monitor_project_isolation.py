"""Monitor project isolation tests (C4: RunKey=(project_id, run_id)).

Two projects ("alpha" and "beta") both contain a run called ``run_001`` with
different losses and different checkpoint hashes. The monitor must keep them
apart in the run list, in every run-scoped detail endpoint and in telemetry
filtering. Unqualified legacy URLs only work when unambiguous.
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from colab_adapter.monitor import ui as monitor_ui
from colab_adapter.monitor.collector import AccountUsageCollector
from colab_adapter.monitor.server import start_monitor_server
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.telemetry import TelemetryEmitter

SHA_ALPHA = "a" * 64
SHA_BETA = "b" * 64
AUTH_TOKEN = "isolation-test-token-0123456789"


class TestMonitorProjectIsolation(unittest.TestCase):
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

        # Project alpha: run_001 with loss 0.1 and checkpoint sha AAA...
        self.telemetry.emit(event_type="train_step", epoch=1, step=1, metrics={"loss": 0.1},
                            run_id="run_001", project_id="alpha")
        self.telemetry.emit(event_type="checkpoint_saved", epoch=1, step=1,
                            run_id="run_001", project_id="alpha")
        self._write_manifest("alpha", "run_001", SHA_ALPHA)

        # Project beta: run_001 with loss 0.9 and checkpoint sha BBB...
        self.telemetry.emit(event_type="train_step", epoch=1, step=1, metrics={"loss": 0.9},
                            run_id="run_001", project_id="beta")
        self.telemetry.emit(event_type="checkpoint_saved", epoch=1, step=1,
                            run_id="run_001", project_id="beta")
        self._write_manifest("beta", "run_001", SHA_BETA)

        # Project alpha only: run_002 (used for the unambiguous legacy URL case)
        self.telemetry.emit(event_type="train_step", epoch=2, step=5, metrics={"loss": 0.5},
                            run_id="run_002", project_id="alpha")
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

    def _write_manifest(self, project, run_id, sha):
        run_dir = self.artifacts_dir / "projects" / project / "runs" / run_id
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

    def _get(self, path, token=AUTH_TOKEN, headers=None):
        req_headers = dict(headers or {})
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

    @staticmethod
    def _artifact_shas(artifacts):
        return {a.get("sha256") for a in artifacts}

    # ---- list ----------------------------------------------------------

    def test_run_list_keeps_same_run_id_from_two_projects_apart(self):
        status, runs = self._get("/api/v1/runs")
        self.assertEqual(status, 200)
        keyed = {(r["project_id"], r["run_id"]): r for r in runs}
        self.assertIn(("alpha", "run_001"), keyed)
        self.assertIn(("beta", "run_001"), keyed)
        self.assertEqual(sum(1 for r in runs if r["run_id"] == "run_001"), 2)
        self.assertEqual(keyed[("alpha", "run_001")]["latest_metrics"], {"loss": 0.1})
        self.assertEqual(keyed[("beta", "run_001")]["latest_metrics"], {"loss": 0.9})
        for r in runs:
            self.assertTrue(r.get("project_id"), "every run entry must carry project_id")

    def test_run_list_project_filter_returns_only_that_project(self):
        status, runs = self._get("/api/v1/runs?project=beta")
        self.assertEqual(status, 200)
        self.assertEqual([(r["project_id"], r["run_id"]) for r in runs], [("beta", "run_001")])

    # ---- detail endpoints with explicit project -----------------------

    def test_project_selector_isolates_metrics_artifacts_and_events(self):
        for project, loss, sha, other_sha in (
            ("alpha", 0.1, SHA_ALPHA, SHA_BETA),
            ("beta", 0.9, SHA_BETA, SHA_ALPHA),
        ):
            with self.subTest(project=project):
                status, metrics = self._get(f"/api/v1/runs/run_001/metrics?project={project}")
                self.assertEqual(status, 200)
                losses = [p["metrics"].get("loss") for p in metrics]
                self.assertEqual(losses, [loss])

                status, arts = self._get(f"/api/v1/runs/run_001/artifacts?project={project}")
                self.assertEqual(status, 200)
                shas = self._artifact_shas(arts["artifacts"])
                self.assertIn(sha, shas)
                self.assertNotIn(other_sha, shas)

                status, detail = self._get(f"/api/v1/runs/run_001?project={project}")
                self.assertEqual(status, 200)
                self.assertNotIn(other_sha, self._artifact_shas(detail["artifacts"]))
                self.assertIn(sha, self._artifact_shas(detail["artifacts"]))
                self.assertTrue(detail["events"])
                for ev in detail["events"]:
                    self.assertEqual(ev["project_id"], project)
                    self.assertEqual(ev["run_id"], "run_001")

                status, logs = self._get(f"/api/v1/runs/run_001/logs?project={project}")
                self.assertEqual(status, 200)

                status, drive = self._get(f"/api/v1/runs/run_001/drive?project={project}")
                self.assertEqual(status, 200)
                self.assertNotIn(other_sha, {a.get("sha256") for a in drive["artifacts"]})

    def test_project_selector_on_manifest_endpoint_is_scoped(self):
        status, manifest = self._get("/api/v1/runs/run_001/manifest?project=beta")
        self.assertEqual(status, 200)
        self.assertEqual(manifest["artifacts"]["model.pt"]["sha256"], SHA_BETA)

    def test_unknown_project_for_existing_run_is_404(self):
        status, body = self._get("/api/v1/runs/run_001?project=gamma")
        self.assertEqual(status, 404)
        self.assertEqual(body.get("run_id"), "run_001")

    # ---- unqualified legacy URL ---------------------------------------

    def test_unqualified_ambiguous_run_returns_409_with_projects(self):
        status, body = self._get("/api/v1/runs/run_001")
        self.assertEqual(status, 409)
        self.assertEqual(body.get("error"), "ambiguous_run")
        self.assertEqual(sorted(body.get("projects")), ["alpha", "beta"])

        status, _ = self._get("/api/v1/runs/run_001/metrics")
        self.assertEqual(status, 409)

    def test_unqualified_unambiguous_legacy_url_still_200(self):
        status, body = self._get("/api/v1/runs/run_002")
        self.assertEqual(status, 200)
        self.assertEqual(body["run_id"], "run_002")
        self.assertIn(SHA_ALPHA, self._artifact_shas(body["artifacts"]))

        status, metrics = self._get("/api/v1/runs/run_002/metrics")
        self.assertEqual(status, 200)
        self.assertEqual([p["metrics"].get("loss") for p in metrics], [0.5])

    def test_unknown_run_is_404(self):
        status, _ = self._get("/api/v1/runs/run_missing")
        self.assertEqual(status, 404)
        status, _ = self._get("/api/v1/runs/run_missing?project=alpha")
        self.assertEqual(status, 404)

    # ---- selector validation and auth ---------------------------------

    def test_invalid_project_selector_rejected_400(self):
        for bad in ("..", "../beta", "a%2Fb", "x" * 200, "bad%20name%3B"):
            with self.subTest(project=bad):
                status, _ = self._get(f"/api/v1/runs/run_001?project={bad}")
                self.assertEqual(status, 400)
                status, _ = self._get(f"/api/v1/runs?project={bad}")
                self.assertEqual(status, 400)

    def test_auth_still_required_for_project_scoped_endpoints(self):
        self.assertEqual(self._get("/api/v1/runs?project=alpha", token=None)[0], 401)
        self.assertEqual(self._get("/api/v1/runs/run_001?project=alpha", token=None)[0], 401)
        self.assertEqual(self._get("/api/v1/runs/run_001/metrics?project=beta", token="wrong")[0], 401)

    # ---- telemetry emitter filter -------------------------------------

    def test_telemetry_get_events_filters_by_project_and_run(self):
        events = self.telemetry.get_events(run_id="run_001", project_id="beta")
        self.assertEqual(len(events), 2)
        self.assertTrue(all(e["project_id"] == "beta" and e["run_id"] == "run_001" for e in events))
        self.assertEqual(len(self.telemetry.get_events(run_id="run_001")), 4)

    # ---- D10 leakage paths --------------------------------------------

    def _stream_events(self, path):
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            headers={"Authorization": f"Bearer {AUTH_TOKEN}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8")

    def test_sse_stream_is_filtered_by_project(self):
        status, body = self._stream_events("/api/v1/events/stream?project=beta")
        self.assertEqual(status, 200)
        data_lines = [json.loads(line[len("data: "):]) for line in body.splitlines() if line.startswith("data: ")]
        self.assertEqual(len(data_lines), 2)
        self.assertTrue(all(ev["project_id"] == "beta" for ev in data_lines))

        status, _ = self._stream_events("/api/v1/events/stream?project=..")
        self.assertEqual(status, 400)

    def test_summary_is_filtered_by_project_and_validated(self):
        status, body = self._get("/api/v1/summary?project=beta")
        self.assertEqual(status, 200)
        self.assertEqual(body["total_runs"], 1)
        status, body = self._get("/api/v1/summary")
        self.assertEqual(status, 200)
        self.assertEqual(body["total_runs"], 3)
        status, _ = self._get("/api/v1/summary?project=../x")
        self.assertEqual(status, 400)

    def test_artifact_ids_include_project_and_run(self):
        status, body = self._get("/api/v1/runs/run_001/artifacts?project=alpha")
        self.assertEqual(status, 200)
        ids = [a["artifact_id"] for a in body["artifacts"]]
        self.assertTrue(ids)
        self.assertTrue(all(i.startswith("alpha/run_001/") for i in ids), ids)

    def test_legacy_event_without_project_id_never_matches_named_project(self):
        legacy = {
            "schema_version": 1, "event_id": "evt-legacy-1", "run_id": "run_legacy",
            "emitted_at": "2026-10-20T00:00:00+00:00", "event_type": "train_step",
            "epoch": 1, "step": 1, "metrics": {"loss": 0.7}, "details": {},
        }
        with open(self.tmp_dir / "telemetry.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(legacy) + "\n")

        status, runs = self._get("/api/v1/runs?project=default")
        self.assertEqual(status, 200)
        self.assertFalse([r for r in runs if r["run_id"] == "run_legacy"])
        self.assertEqual(self._get("/api/v1/runs/run_legacy?project=default")[0], 404)

        status, body = self._get("/api/v1/runs/run_legacy")
        self.assertEqual(status, 200)
        self.assertEqual(body["event_count"], 1)

    def test_symlinked_project_dir_is_rejected(self):
        os.symlink(self.artifacts_dir / "projects" / "alpha", self.artifacts_dir / "projects" / "gamma")
        status, body = self._get("/api/v1/runs/run_001?project=gamma")
        self.assertEqual(status, 404)
        status, runs = self._get("/api/v1/runs")
        self.assertFalse([r for r in runs if r["project_id"] == "gamma"])
        status, body = self._get("/api/v1/runs/run_001")
        self.assertEqual(status, 409)
        self.assertEqual(body["projects"], ["alpha", "beta"])

    def test_manifest_with_mismatched_ids_is_ignored(self):
        run_dir = self.artifacts_dir / "projects" / "delta" / "runs" / "run_001"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps({"project_id": "alpha", "run_id": "run_001",
                        "artifacts": {"model.pt": {"sha256": "c" * 64, "logical_path": "model.pt"}}}),
            encoding="utf-8",
        )
        status, body = self._get("/api/v1/runs/run_001/artifacts?project=delta")
        self.assertEqual(status, 200)
        self.assertEqual(body["artifacts"], [])
        status, body = self._get("/api/v1/runs/run_001")
        self.assertEqual(status, 409)
        self.assertIn("delta", body["projects"])

    # ---- UI source contract -------------------------------------------

    def test_ui_source_keeps_project_id_and_no_inner_html(self):
        ui_source = Path(monitor_ui.__file__).read_text(encoding="utf-8")
        self.assertIn("projectId", ui_source)
        self.assertIn("encodeURIComponent(safeProjectId)", ui_source)
        self.assertIn("?project=", ui_source)
        self.assertNotIn(".innerHTML", ui_source)


if __name__ == "__main__":
    unittest.main()
