"""Offline tests for scripts/live/monitor_check.py.

The checker runs as a subprocess (as an operator would run it) against:
- a real MonitorServer bound to 127.0.0.1 with seeded telemetry and manifests
  (two checkpoint revisions, manifest head = the newer one), and
- deliberately broken fake servers and leaked-token logs, to prove the checker
  can fail.

All tokens and directories are throwaway. Nothing contacts Colab or Google.
"""
from __future__ import annotations

import http.server
import json
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from colab_adapter.monitor.collector import AccountUsageCollector
from colab_adapter.monitor.server import start_monitor_server
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.telemetry import TelemetryEmitter

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "live" / "monitor_check.py"
TOKEN_ENV = "COLAB_MONITOR_TOKEN"
OLD_SHA = "a" * 64
NEW_SHA = "b" * 64
OTHER_SHA = "c" * 64


def _write_manifest(run_dir: Path, revisions: list, head_sha: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for sha in revisions:
        artifacts[f"model.pt@sha256_{sha}"] = {
            "artifact_id": f"model.pt@sha256_{sha}",
            "logical_path": "model.pt",
            "sha256": sha,
            "artifact_type": "checkpoint",
            "path": str(run_dir / "model.pt"),
        }
    manifest = {
        "created_at": "2026-10-20T00:00:00+00:00",
        "artifacts": artifacts,
        "latest_by_path": {"model.pt": f"model.pt@sha256_{head_sha}"},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


class _CheckerRunner:
    """Runs the checker subprocess with the token supplied only via the environment."""

    def __init__(self, tmp: Path) -> None:
        self.tmp = tmp

    def run(self, base_url: str, token: str, extra: list, env_token: bool = True):
        env = {k: v for k, v in os.environ.items() if k != TOKEN_ENV}
        if env_token:
            env[TOKEN_ENV] = token
        cmd = [sys.executable, "-I", str(CHECKER), "--base-url", base_url,
               "--expect-project", "alpha", "--expect-run", "run_001"] + extra
        return subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=120)


class _SeededMonitorMixin:
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.token = secrets.token_urlsafe(24)
        artifacts_dir = self.tmp / "artifacts"
        telemetry = TelemetryEmitter(ledger_path=self.tmp / "telemetry.jsonl", run_id="run_001", project_id="alpha")
        telemetry.emit(event_type="train_step", epoch=1, step=1, metrics={"loss": 0.4},
                       run_id="run_001", project_id="alpha")
        telemetry.emit(event_type="checkpoint_saved", epoch=1, step=1, run_id="run_001", project_id="alpha")
        telemetry.emit(event_type="train_step", epoch=1, step=2, metrics={"loss": 0.1},
                       run_id="run_001", project_id="beta")
        # Same run id in a second project: the checker must key by (project, run).
        _write_manifest(artifacts_dir / "projects" / "alpha" / "runs" / "run_001",
                        [OLD_SHA, NEW_SHA], head_sha=NEW_SHA)
        _write_manifest(artifacts_dir / "projects" / "beta" / "runs" / "run_001",
                        [OTHER_SHA], head_sha=OTHER_SHA)
        self.server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=AccountUsageCollector(ledger_path=self.tmp / "account_usage.jsonl"),
            registry=ManagedSessionRegistry(self.tmp / "managed_sessions.json"),
            telemetry=telemetry,
            auth_token=self.token,
            allow_external=False,
            artifacts_dir=artifacts_dir,
        )
        thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

        def _stop() -> None:
            self.server.shutdown()
            self.server.server_close()
            thread.join(timeout=5)

        self.addCleanup(_stop)
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.runner = _CheckerRunner(self.tmp)
        self.expect_args = ["--expect-latest-sha256", NEW_SHA, "--expect-revisions", "2"]


class TestCheckerAgainstSeededMonitor(_SeededMonitorMixin, unittest.TestCase):
    def _run_with_evidence(self, extra: list, log_file=None):
        evidence = self.tmp / "evidence.json"
        args = list(extra) + ["--evidence-file", str(evidence)]
        if log_file is not None:
            args += ["--log-file", str(log_file)]
        result = self.runner.run(self.base_url, self.token, args)
        self.assertNotIn(self.token, result.stdout)
        self.assertNotIn(self.token, result.stderr)
        data = json.loads(evidence.read_text(encoding="utf-8")) if evidence.exists() else None
        if data is not None:
            self.assertNotIn(self.token, json.dumps(data))
        return result, data

    def _check(self, data: dict, check_id: str) -> dict:
        matches = [c for c in data["checks"] if c["id"] == check_id]
        self.assertEqual(len(matches), 1, f"expected exactly one check {check_id}")
        return matches[0]

    def test_full_pass_and_patch_is_405(self):
        # PATCH used to fall through to the stock handler's 501; the monitor now refuses it with 405.
        result, data = self._run_with_evidence(self.expect_args)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(data["overall"], "PASS")
        d1 = self._check(data, "d1_mutating_methods_refused")
        self.assertEqual(d1["details"]["PATCH /api/v1/runs"], 405)
        self.assertEqual(d1["details"]["POST /api/v1/runs"], 405)
        for check in data["checks"]:
            self.assertEqual(check["status"], "PASS", check["id"])

    def test_leaked_token_in_log_file_fails(self):
        log = self.tmp / "monitor.log"
        log.write_text(f"2026-10-25 starting monitor auth_token={self.token}\n", encoding="utf-8")
        result, data = self._run_with_evidence(self.expect_args, log_file=log)
        self.assertEqual(result.returncode, 1)
        e2 = self._check(data, "e2_no_secret_in_log")
        self.assertEqual(e2["status"], "FAIL")
        self.assertTrue(e2["details"]["exact_token_found"])

    def test_secret_shaped_value_in_log_file_fails_without_exact_token(self):
        log = self.tmp / "monitor.log"
        log.write_text("refresh_token: 1234567890abcdefXYZ\nBearer abcdefgh12345678\n", encoding="utf-8")
        result, data = self._run_with_evidence(self.expect_args, log_file=log)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._check(data, "e2_no_secret_in_log")["status"], "FAIL")
        self.assertNotIn("1234567890abcdefXYZ", json.dumps(data))

    def test_clean_log_file_passes_log_scan(self):
        log = self.tmp / "monitor.log"
        log.write_text("monitor started on 127.0.0.1\nrefresh_token=[REDACTED]\n", encoding="utf-8")
        _, data = self._run_with_evidence(self.expect_args, log_file=log)
        self.assertEqual(self._check(data, "e2_no_secret_in_log")["status"], "PASS")

    def test_wrong_expected_head_fails_revision_check(self):
        result, data = self._run_with_evidence(
            ["--expect-latest-sha256", OLD_SHA, "--expect-revisions", "2"])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(self._check(data, "c2_revision_latest_state")["status"], "FAIL")
        self.assertEqual(self._check(data, "c3_run_list_latest_matches_head")["status"], "FAIL")

    def test_unknown_expected_pair_fails_identity_check(self):
        evidence = self.tmp / "evidence.json"
        result = self.runner.run(self.base_url, self.token,
                                 ["--expect-run", "run_999", "--evidence-file", str(evidence)])
        self.assertEqual(result.returncode, 1)
        data = json.loads(evidence.read_text(encoding="utf-8"))
        self.assertEqual(self._check(data, "c1_run_key_project_and_run")["status"], "FAIL")

    def test_missing_token_env_is_usage_error_without_network(self):
        result = self.runner.run(self.base_url, self.token, self.expect_args, env_token=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn(TOKEN_ENV, result.stderr)

    def test_token_on_argv_is_rejected(self):
        env = {k: v for k, v in os.environ.items() if k != TOKEN_ENV}
        cmd = [sys.executable, "-I", str(CHECKER), "--base-url", self.base_url,
               "--expect-project", "alpha", "--expect-run", "run_001", "--token", self.token]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(self.token, result.stdout + result.stderr)

    def test_token_value_passed_as_positional_is_refused_without_echo(self):
        env = {k: v for k, v in os.environ.items() if k != TOKEN_ENV}
        env[TOKEN_ENV] = self.token
        cmd = [sys.executable, "-I", str(CHECKER), "--base-url", self.base_url,
               "--expect-project", "alpha", "--expect-run", self.token]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn(self.token, result.stdout + result.stderr)

    def test_non_loopback_host_is_refused(self):
        env = {k: v for k, v in os.environ.items() if k != TOKEN_ENV}
        env[TOKEN_ENV] = self.token
        cmd = [sys.executable, "-I", str(CHECKER), "--base-url", "http://192.0.2.10:8080",
               "--expect-project", "alpha", "--expect-run", "run_001"]
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 2)
        self.assertIn("loopback", result.stderr)


class _FakeHandler(http.server.BaseHTTPRequestHandler):
    """Deliberately insecure monitor: serves everything, echoes request headers."""

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        pass

    def _reply(self):
        body = json.dumps({"path": self.path, "auth": self.headers.get("Authorization", "")}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _reply
    do_POST = _reply
    do_PUT = _reply
    do_DELETE = _reply
    do_PATCH = _reply


class TestCheckerAgainstTamperedServer(_SeededMonitorMixin, unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        fake = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeHandler)
        thread = threading.Thread(target=fake.serve_forever, daemon=True)
        thread.start()

        def _stop() -> None:
            fake.shutdown()
            fake.server_close()
            thread.join(timeout=5)

        self.addCleanup(_stop)
        self.fake_url = f"http://127.0.0.1:{fake.server_port}"

    def test_unauthenticated_and_query_token_access_fails(self):
        evidence = self.tmp / "fake_evidence.json"
        result = self.runner.run(self.fake_url, self.token,
                                 self.expect_args + ["--evidence-file", str(evidence)])
        self.assertEqual(result.returncode, 1)
        self.assertNotIn(self.token, result.stdout + result.stderr)
        data = json.loads(evidence.read_text(encoding="utf-8"))
        self.assertEqual(data["overall"], "FAIL")
        for check_id in ("a1_unauthenticated_401", "a2_query_token_rejected", "a3_wrong_bearer_401"):
            matches = [c for c in data["checks"] if c["id"] == check_id]
            self.assertEqual(matches[0]["status"], "FAIL", check_id)
        self.assertNotIn(self.token, json.dumps(data))

    def test_response_echoing_authorization_fails_secret_scan(self):
        evidence = self.tmp / "fake_evidence.json"
        self.runner.run(self.fake_url, self.token,
                        self.expect_args + ["--evidence-file", str(evidence)])
        data = json.loads(evidence.read_text(encoding="utf-8"))
        e1 = [c for c in data["checks"] if c["id"] == "e1_no_secret_in_responses"][0]
        self.assertEqual(e1["status"], "FAIL")
        self.assertNotIn(self.token, json.dumps(data))


class TestSessionSnapshotAssertion(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def _write(self, name: str, payload) -> Path:
        path = self.tmp / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def _assert(self, before, after):
        before_path = self._write("before.json", before)
        after_path = self._write("after.json", after)
        evidence = self.tmp / "sessions_evidence.json"
        result = subprocess.run(
            [sys.executable, "-I", str(CHECKER), "--assert-no-new-sessions",
             "--sessions-before", str(before_path), "--sessions-after", str(after_path),
             "--evidence-file", str(evidence)],
            capture_output=True, text=True, timeout=60)
        data = json.loads(evidence.read_text(encoding="utf-8")) if evidence.exists() else None
        return result, data

    def test_unchanged_session_set_passes(self):
        snap = [{"session_id": "managed-1", "state": "READY"}]
        result, data = self._assert(snap, snap)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(data["overall"], "PASS")

    def test_new_managed_session_fails(self):
        before = [{"session_id": "managed-1"}]
        after = [{"session_id": "managed-1"}, {"session_id": "managed-2"}]
        result, data = self._assert(before, after)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(data["checks"][0]["details"]["new_session_count"], 1)
        self.assertEqual(data["overall"], "FAIL")

    def test_session_swap_counts_as_growth(self):
        result, _ = self._assert(["managed-1"], ["managed-2"])
        self.assertEqual(result.returncode, 1)

    def test_malformed_snapshot_is_usage_error(self):
        result, data = self._assert([{"name": "no-id"}], [])
        self.assertEqual(result.returncode, 2)
        self.assertIsNone(data)


if __name__ == "__main__":
    unittest.main()
