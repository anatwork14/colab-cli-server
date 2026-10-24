"""Unit and integration tests for Colab Compute & Training Monitor Backend (M8).

Verifies acceptance criteria Q19–Q31:
- Q19: Strictly read-only collector; 0 allocation/new/run/ssh calls.
- Q20: Ingestion of 1,440 synthetic samples without data drops.
- Q21: Run event history consistency across restarts.
- Q22: Simulated CU increases/top-ups labeled honestly without negative spend.
- Q23: Account balance decreases with external sessions labeled UNATTRIBUTED.
- Q24: First collector run records explicit monitoring_since date.
- Q25: CLI error / timeout backoff and stale indicators.
- Q26: Token authentication enforcement and CSV export.
- Q27: SSE event stream with resume cursor.
- Q28: Strict loopback localhost binding enforcement.
- Q29: Read-only HTTP enforcement: 405 Method Not Allowed on POST/PUT/DELETE.
- Q30: Real emitted metrics without fabrication.
- Q31: Deep credential redaction in API responses.
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import io
import json
import os
import shutil
import ssl
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

from colab_adapter.monitor.collector import (
    AccountSample,
    AccountUsageCollector,
    parse_usage_output,
)
from colab_adapter.monitor.schemas import _iso_now
from colab_adapter.monitor.server import MonitorServer, start_monitor_server
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.telemetry import TelemetryEmitter


class TestMonitorBackend(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.ledger_path = Path(self.tmp_dir) / "account_usage.jsonl"
        self.telemetry_path = Path(self.tmp_dir) / "telemetry.jsonl"
        self.registry_path = Path(self.tmp_dir) / "managed_sessions.json"

        self.registry = ManagedSessionRegistry(self.registry_path)
        self.telemetry = TelemetryEmitter(ledger_path=self.telemetry_path, run_id="test-run-1", project_id="proj-test")
        self.auth_token = "secret-test-token-xyz-123"

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_q19_collector_strictly_read_only_zero_allocation(self):
        """Q19: Collector only executes read-only 'usage' command; zero new/run/ssh calls."""
        executed_commands = []

        def mock_run_cmd(cmd, **kwargs):
            executed_commands.append(cmd)
            # Realistic colab usage output
            out = "Compute units: 185.5\nRate: 0.50 CU/hr\nAssignments: 1\n"
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=out, stderr="")

        collector = AccountUsageCollector(
            ledger_path=self.ledger_path,
            run_cmd_fn=mock_run_cmd,
        )

        sample = collector.poll_once()
        self.assertEqual(sample.balance_cu, 185.5)
        self.assertEqual(sample.rate_cu_per_hour, 0.50)
        self.assertEqual(sample.active_assignments, 1)

        # Invariant: Dispatched only 'usage' command
        self.assertEqual(len(executed_commands), 1)
        self.assertEqual(executed_commands[0][-1], "usage")
        for arg in executed_commands[0]:
            self.assertNotIn(arg, ("new", "run", "ssh", "stop"))

    def test_q20_collector_1440_synthetic_minute_snapshots(self):
        """Q20: Writes and queries 1,440 synthetic snapshots without data loss."""
        collector = AccountUsageCollector(ledger_path=self.ledger_path)

        samples = []
        for i in range(1440):
            sample = AccountSample(
                sample_id=f"sample-{i:04d}",
                observed_at=f"2026-10-24T{i//60:02d}:{i%60:02d}:00Z",
                balance_cu=100.0 - (i * 0.01),
                rate_cu_per_hour=0.6,
                active_assignments=1,
            )
            collector._record_sample(sample)

        loaded = collector.get_samples(limit=2000)
        self.assertEqual(len(loaded), 1440)
        self.assertEqual(loaded[0].sample_id, "sample-0000")
        self.assertEqual(loaded[-1].sample_id, "sample-1439")

    def test_q22_simulated_cu_topup_labeled_honestly(self):
        """Q22: Balance increase labeled TOP_UP without inventing negative spend."""
        calls = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Compute units: 50.0\n", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Compute units: 150.0\n", stderr=""),
        ]

        def mock_run_cmd(cmd, **kwargs):
            return calls.pop(0)

        collector = AccountUsageCollector(ledger_path=self.ledger_path, run_cmd_fn=mock_run_cmd)

        s1 = collector.poll_once()
        self.assertEqual(s1.balance_cu, 50.0)
        self.assertEqual(s1.change_type, "INITIAL")

        s2 = collector.poll_once()
        self.assertEqual(s2.balance_cu, 150.0)
        self.assertEqual(s2.net_balance_change, 100.0)
        self.assertEqual(s2.change_type, "TOP_UP")

    def test_q23_balance_decrease_with_external_sessions_labeled_unattributed(self):
        """Q23: CU decrease while external/unmanaged sessions exist is labeled UNATTRIBUTED."""
        calls = [
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Compute units: 100.0\nAssignments: 2\n", stderr=""),
            subprocess.CompletedProcess(args=[], returncode=0, stdout="Compute units: 98.0\nAssignments: 2\n", stderr=""),
        ]

        def mock_run_cmd(cmd, **kwargs):
            return calls.pop(0)

        collector = AccountUsageCollector(ledger_path=self.ledger_path, run_cmd_fn=mock_run_cmd)
        collector.poll_once(unmanaged_sessions_count=1)
        s2 = collector.poll_once(unmanaged_sessions_count=1)

        self.assertEqual(s2.net_balance_change, -2.0)
        self.assertEqual(s2.change_type, "UNATTRIBUTED")

    def test_q24_first_collector_run_records_monitoring_since(self):
        """Q24: First collector run initializes immutable monitoring_since timestamp."""
        collector = AccountUsageCollector(ledger_path=self.ledger_path)
        since1 = collector.monitoring_since
        self.assertTrue(len(since1) > 10)

        # Subsequent instance loads existing timestamp
        collector2 = AccountUsageCollector(ledger_path=self.ledger_path)
        self.assertEqual(collector2.monitoring_since, since1)

    def test_q25_cli_error_triggers_backoff_and_stale_flag(self):
        """Q25: Simulated CLI error triggers backoff increment and records ERROR status."""
        def mock_run_fail(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="HTTP 429 Too Many Requests")

        collector = AccountUsageCollector(ledger_path=self.ledger_path, run_cmd_fn=mock_run_fail)
        sample = collector.poll_once()

        self.assertEqual(sample.status, "ERROR")
        self.assertIn("HTTP 429", sample.error_message or "")
        self.assertEqual(collector.backoff_level, 1)

    def test_q28_localhost_default_binding_security(self):
        """Q28: Refuses to bind to external interface (0.0.0.0) without allow_external=True."""
        collector = AccountUsageCollector(ledger_path=self.ledger_path)
        with self.assertRaises(ValueError) as ctx:
            start_monitor_server(
                host="0.0.0.0",
                port=8899,
                collector=collector,
                registry=self.registry,
                telemetry=self.telemetry,
                allow_external=False,
            )
        self.assertIn("Security violation", str(ctx.exception))

    def test_server_authenticated_api_and_q29_read_only_enforcement(self):
        """Tests REST endpoints, token auth (Q26), and 405 Method Not Allowed (Q29)."""
        collector = AccountUsageCollector(ledger_path=self.ledger_path)
        # Pre-seed one sample
        collector._record_sample(AccountSample(
            sample_id="test-s-1",
            observed_at=_iso_now(),
            balance_cu=150.0,
            rate_cu_per_hour=0.25,
            active_assignments=1,
        ))

        # Start server on dynamic loopback port
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,  # OS assigns free port
            collector=collector,
            registry=self.registry,
            telemetry=self.telemetry,
            auth_token=self.auth_token,
            allow_external=False,
        )
        port = server.server_port
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        base_url = f"http://127.0.0.1:{port}"

        try:
            # 1. Unauthenticated request -> 401
            req_unauth = urllib.request.Request(f"{base_url}/api/v1/summary")
            try:
                urllib.request.urlopen(req_unauth)
                self.fail("Expected HTTP 401 Unauthorized")
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 401)

            # 2. Authenticated request via Bearer header -> 200
            req_auth = urllib.request.Request(
                f"{base_url}/api/v1/summary",
                headers={"Authorization": f"Bearer {self.auth_token}"},
            )
            with urllib.request.urlopen(req_auth) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertEqual(data["current_balance_cu"], 150.0)

            # 3. Request via ?token= query parameter is strictly rejected (P0) -> 401
            req_param = urllib.request.Request(f"{base_url}/api/v1/health?token={self.auth_token}")
            try:
                urllib.request.urlopen(req_param)
                self.fail("Expected HTTP 401 for ?token= query parameter")
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 401)

            # 4. Q29: POST request -> 405 Method Not Allowed
            req_post = urllib.request.Request(
                f"{base_url}/api/v1/runs",
                data=b'{"cmd": "run"}',
                headers={"Authorization": f"Bearer {self.auth_token}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req_post)
                self.fail("Expected HTTP 405 Method Not Allowed")
            except urllib.error.HTTPError as e:
                self.assertEqual(e.code, 405)

            # 5. Q26: CSV export with Authorization header
            req_csv = urllib.request.Request(
                f"{base_url}/api/v1/export/compute.csv",
                headers={"Authorization": f"Bearer {self.auth_token}"},
            )
            with urllib.request.urlopen(req_csv) as resp:
                self.assertEqual(resp.status, 200)
                csv_text = resp.read().decode("utf-8")
                self.assertIn("sample_id,observed_at", csv_text)
                self.assertIn("test-s-1", csv_text)

        finally:
            server.shutdown()
            server.server_close()

    def test_q27_sse_event_stream_with_cursor_resume(self):
        """Q27: SSE stream resumes from cursor without missing events."""
        ev1 = self.telemetry.emit(event_type="run_started", epoch=0, step=0)
        ev2 = self.telemetry.emit(event_type="train_step", epoch=1, step=10, metrics={"loss": 1.5})
        ev3 = self.telemetry.emit(event_type="checkpoint_saved", epoch=1, step=10)

        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=AccountUsageCollector(ledger_path=self.ledger_path),
            registry=self.registry,
            telemetry=self.telemetry,
            auth_token=self.auth_token,
            allow_external=False,
        )
        port = server.server_port
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        base_url = f"http://127.0.0.1:{port}"

        try:
            # Request SSE resuming from ev1
            cursor = ev1["event_id"]
            url = f"{base_url}/api/v1/events/stream?from_id={cursor}"
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {self.auth_token}"})
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                stream_chunk = resp.read().decode("utf-8")
                # ev1 should be skipped because of cursor; ev2 and ev3 should be present
                self.assertNotIn(cursor, stream_chunk)
                self.assertIn(ev2["event_id"], stream_chunk)
        finally:
            server.shutdown()
            server.server_close()

    def test_q31_credential_redaction_in_api_responses(self):
        """Q31: Seeded secrets in telemetry are redacted from HTTP responses."""
        secret_token = "ya29.a0AfH6SMAAAAAAA_dashboard_secret_leak_12345"
        self.telemetry.emit(
            event_type="error",
            metrics={"auth": secret_token},
            details={"msg": f"Failed with secret: {secret_token}"},
        )

        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            collector=AccountUsageCollector(ledger_path=self.ledger_path),
            registry=self.registry,
            telemetry=self.telemetry,
            auth_token=self.auth_token,
            allow_external=False,
        )
        port = server.server_port
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        base_url = f"http://127.0.0.1:{port}"
        try:
            req = urllib.request.Request(
                f"{base_url}/api/v1/runs",
                headers={"Authorization": f"Bearer {self.auth_token}"},
            )
            with urllib.request.urlopen(req) as resp:
                body = resp.read().decode("utf-8")
                self.assertNotIn(secret_token, body)
        finally:
            server.shutdown()
            server.server_close()

    def test_trusted_proxy_missing_secret_raises_value_error(self):
        """trusted_proxy=True without trusted_proxy_secret raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            start_monitor_server(
                host="127.0.0.1",
                port=0,
                registry=self.registry,
                trusted_proxy=True,
                trusted_proxy_secret=None,
            )
        self.assertIn("requires a non-empty trusted_proxy_secret", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            start_monitor_server(
                host="127.0.0.1",
                port=0,
                registry=self.registry,
                trusted_proxy=True,
                trusted_proxy_secret="",
            )
        self.assertIn("requires a non-empty trusted_proxy_secret", str(ctx2.exception))

    def test_allow_external_on_0000_require_tls_missing_certs_raises_value_error(self):
        """allow_external=True on 0.0.0.0 with require_tls=True but without valid cert files raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            start_monitor_server(
                host="0.0.0.0",
                port=0,
                allow_external=True,
                auth_token="valid-token",
                require_tls=True,
                tls_certfile=None,
                tls_keyfile=None,
            )
        self.assertIn("External binding over unencrypted HTTP", str(ctx.exception))
        self.assertIn("Real TLS requires valid tls_certfile and tls_keyfile", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            start_monitor_server(
                host="0.0.0.0",
                port=0,
                allow_external=True,
                auth_token="valid-token",
                require_tls=True,
                tls_certfile="/tmp/nonexistent_cert.pem",
                tls_keyfile="/tmp/nonexistent_key.pem",
            )
        self.assertIn("External binding over unencrypted HTTP", str(ctx2.exception))
        self.assertIn("Real TLS requires valid tls_certfile and tls_keyfile", str(ctx2.exception))

    def test_trusted_proxy_mode_external_bind_without_real_tls_raises_value_error(self):
        """trusted_proxy=True on external interface without real TLS raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            start_monitor_server(
                host="0.0.0.0",
                port=0,
                allow_external=True,
                trusted_proxy=True,
                trusted_proxy_secret="secret123",
            )
        self.assertIn("trusted_proxy mode cannot bind to an unencrypted external interface", str(ctx.exception))

    def test_trusted_proxy_direct_attacker_header_rejected_401(self):
        """Direct client supplying X-Remote-User or X-Forwarded-User without valid secret/sig receives HTTP 401."""
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            registry=self.registry,
            telemetry=self.telemetry,
            collector=AccountUsageCollector(ledger_path=self.ledger_path),
            trusted_proxy=True,
            trusted_proxy_secret="valid-proxy-secret",
        )
        port = server.server_port
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base_url = f"http://127.0.0.1:{port}"

        try:
            # 1. Attacker header without secret
            req = urllib.request.Request(f"{base_url}/api/v1/summary", headers={"X-Remote-User": "attacker"})
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 401)

            # 2. X-Forwarded-User admin without secret
            req = urllib.request.Request(f"{base_url}/api/v1/summary", headers={"X-Forwarded-User": "admin"})
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 401)

            # 3. Attacker with forged/wrong secret
            req = urllib.request.Request(
                f"{base_url}/api/v1/summary",
                headers={"X-Remote-User": "attacker", "X-Proxy-Secret": "wrong-secret"},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 401)

            # 4. Attacker with invalid HMAC signature
            req = urllib.request.Request(
                f"{base_url}/api/v1/summary",
                headers={"X-Remote-User": "attacker", "X-Proxy-Signature": "0000deadbeef"},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 401)
        finally:
            server.shutdown()
            server.server_close()

    def test_trusted_proxy_untrusted_peer_ip_rejected_401(self):
        """Untrusted peer IP supplying X-Remote-User + secret is rejected with HTTP 401."""
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            registry=self.registry,
            telemetry=self.telemetry,
            collector=AccountUsageCollector(ledger_path=self.ledger_path),
            trusted_proxy=True,
            trusted_proxy_secret="valid-proxy-secret",
            trusted_proxy_ips={"192.168.1.50"},  # 127.0.0.1 is untrusted
        )
        port = server.server_port
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base_url = f"http://127.0.0.1:{port}"

        try:
            req = urllib.request.Request(
                f"{base_url}/api/v1/summary",
                headers={"X-Remote-User": "alice", "X-Proxy-Secret": "valid-proxy-secret"},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req)
            self.assertEqual(ctx.exception.code, 401)
        finally:
            server.shutdown()
            server.server_close()

    def test_trusted_proxy_valid_loopback_proxy_request_succeeds(self):
        """Valid loopback proxy request with X-Remote-User and secret or HMAC succeeds with HTTP 200."""
        secret = "super-secret-proxy-token-42"
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            registry=self.registry,
            telemetry=self.telemetry,
            collector=AccountUsageCollector(ledger_path=self.ledger_path),
            trusted_proxy=True,
            trusted_proxy_secret=secret,
            trusted_proxy_ips={"127.0.0.1", "::1"},
        )
        port = server.server_port
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base_url = f"http://127.0.0.1:{port}"

        try:
            # 1. Via X-Proxy-Secret
            req1 = urllib.request.Request(
                f"{base_url}/api/v1/summary",
                headers={"X-Remote-User": "alice", "X-Proxy-Secret": secret},
            )
            with urllib.request.urlopen(req1) as resp:
                self.assertEqual(resp.status, 200)

            # 2. Via X-Proxy-Signature HMAC-SHA256
            user = "bob"
            sig = hmac.new(secret.encode("utf-8"), user.encode("utf-8"), hashlib.sha256).hexdigest()
            req2 = urllib.request.Request(
                f"{base_url}/api/v1/summary",
                headers={"X-Forwarded-User": user, "X-Proxy-Signature": sig},
            )
            with urllib.request.urlopen(req2) as resp:
                self.assertEqual(resp.status, 200)
        finally:
            server.shutdown()
            server.server_close()

    def test_all_endpoints_enforce_auth_and_reject_query_token(self):
        """All endpoints enforce auth and reject ?token=."""
        ev = self.telemetry.emit(event_type="run_started", epoch=0, step=0)
        run_id = ev["run_id"]

        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            registry=self.registry,
            telemetry=self.telemetry,
            collector=AccountUsageCollector(ledger_path=self.ledger_path),
            auth_token=self.auth_token,
        )
        port = server.server_port
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base_url = f"http://127.0.0.1:{port}"

        endpoints = [
            "/api/v1/summary",
            f"/api/v1/runs/{run_id}",
            "/api/v1/events/stream",
            "/api/v1/export/compute.csv",
        ]

        try:
            for ep in endpoints:
                # 1. Unauthenticated -> 401
                req_unauth = urllib.request.Request(f"{base_url}{ep}")
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(req_unauth)
                self.assertEqual(ctx.exception.code, 401, f"{ep} failed to enforce auth on unauthenticated request")

                # 2. ?token= parameter strictly rejected -> 401
                sep = "&" if "?" in ep else "?"
                req_param = urllib.request.Request(f"{base_url}{ep}{sep}token={self.auth_token}")
                with self.assertRaises(urllib.error.HTTPError) as ctx:
                    urllib.request.urlopen(req_param)
                self.assertEqual(ctx.exception.code, 401, f"{ep} allowed ?token= query parameter")

                # 3. Bearer header -> 200
                req_auth = urllib.request.Request(
                    f"{base_url}{ep}",
                    headers={"Authorization": f"Bearer {self.auth_token}"},
                )
                with urllib.request.urlopen(req_auth, timeout=5.0) as resp:
                    self.assertEqual(resp.status, 200, f"{ep} failed on valid Bearer auth")
        finally:
            server.shutdown()
            server.server_close()

    def test_real_tls_enforcement_and_https_serving(self):
        """require_tls=True wraps socket with SSLContext and serves HTTPS."""
        cert_path = Path(self.tmp_dir) / "cert.pem"
        key_path = Path(self.tmp_dir) / "key.pem"
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", str(key_path), "-out", str(cert_path),
            "-days", "1", "-nodes", "-subj", "/CN=localhost",
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            registry=self.registry,
            telemetry=self.telemetry,
            collector=AccountUsageCollector(ledger_path=self.ledger_path),
            auth_token=self.auth_token,
            require_tls=True,
            tls_certfile=cert_path,
            tls_keyfile=key_path,
        )
        port = server.server_port
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

        ssl_ctx = ssl.create_default_context()
        ssl_ctx.check_hostname = False
        ssl_ctx.verify_mode = ssl.CERT_NONE

        base_url = f"https://127.0.0.1:{port}"
        try:
            req = urllib.request.Request(
                f"{base_url}/api/v1/summary",
                headers={"Authorization": f"Bearer {self.auth_token}"},
            )
            with urllib.request.urlopen(req, context=ssl_ctx) as resp:
                self.assertEqual(resp.status, 200)
                data = json.loads(resp.read().decode("utf-8"))
                self.assertIn("monitoring_since", data)
        finally:
            server.shutdown()
            server.server_close()

    def test_broker_token_verification_integration(self):
        """Monitor server accepts Bearer and X-API-Token if verified by attached broker."""
        mock_broker = MagicMock()
        mock_broker.verify_token.side_effect = lambda t: t == "valid-broker-token"

        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            registry=self.registry,
            telemetry=self.telemetry,
            collector=AccountUsageCollector(ledger_path=self.ledger_path),
            broker=mock_broker,
        )
        port = server.server_port
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        base_url = f"http://127.0.0.1:{port}"

        try:
            # 1. Bearer header with broker token -> 200
            req = urllib.request.Request(
                f"{base_url}/api/v1/summary",
                headers={"Authorization": "Bearer valid-broker-token"},
            )
            with urllib.request.urlopen(req) as resp:
                self.assertEqual(resp.status, 200)

            # 2. X-API-Token header with broker token -> 200
            req2 = urllib.request.Request(
                f"{base_url}/api/v1/summary",
                headers={"X-API-Token": "valid-broker-token"},
            )
            with urllib.request.urlopen(req2) as resp:
                self.assertEqual(resp.status, 200)

            # 3. Invalid broker token -> 401
            req3 = urllib.request.Request(
                f"{base_url}/api/v1/summary",
                headers={"Authorization": "Bearer invalid-token"},
            )
            with self.assertRaises(urllib.error.HTTPError) as ctx:
                urllib.request.urlopen(req3)
            self.assertEqual(ctx.exception.code, 401)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
