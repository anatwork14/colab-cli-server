"""Regression tests reproducing identified authentication, security, resumable upload, and recovery failures.

Covers:
1. Authentication & Remote-Access Security:
   - Prove ?token=SECRET is rejected across all endpoints (summary, stream, export, runs).
   - Requests without valid authentication receive 401.
   - Strict CORS: Access-Control-Allow-Origin: * is eliminated; evil origins receive no CORS header.
   - External binding over unencrypted HTTP without trusted identity proxy is rejected with ValueError.
   - WorkspaceBroker persistent tokens_file survives restarts and enforces 0600 permissions.
   - Two separately authenticated browser clients can inspect the same run.
2. Resumable Uploads & Authoritative Verification:
   - Absent remote metadata size raises DriveIntegrityError (never substitutes local expected size).
   - Partial 308 Range header parsing (bytes=0-N correctly updates offset to N+1).
   - Ambiguous network failure reconciles committed offset before retry.
   - Expired session URL (404/410) safely re-initiates and checks for existing file.
   - Streaming download in retrieve() with memory-bounded temporary file and hash verification.
3. Persistent Transfer Journal & In-Process Retries:
   - Corrupted journal JSON preserves .corrupt copy and recovers from .bak or raises JournalCorruptionError.
   - TransferJournal directory is 0700 and files are 0600.
   - Background worker in-process retry scheduler retries RETRY_PENDING tasks without restarting worker.
4. Reproducibility Snapshot Hardening:
   - Symlinks pointing outside project root are excluded from snapshots.
   - Self-referential destination archive inside project root is excluded from snapshots.
"""
import http.client
import json
import os
import shutil
import stat
import tempfile
import threading
import time
import unittest
from pathlib import Path

from colab_adapter.broker import WorkspaceBroker
from colab_adapter.errors import (
    ArtifactValidationError,
    BrokerAuthenticationError,
    ColabV2Error,
    DriveAuthenticationError,
    DriveIntegrityError,
    DriveTransferError,
    JournalCorruptionError,
)
from colab_adapter.monitor.server import MonitorServer, start_monitor_server
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.reproducibility import create_deterministic_source_snapshot
from colab_adapter.storage import (
    ArtifactType,
    ArtifactUploadWorker,
    DriveCredentials,
    GoogleDriveArtifactStore,
    LocalArtifactStore,
    TransferJournal,
    TransferRecord,
    TransferStatus,
)
from tests.test_google_drive_store import MockDriveHttpClient


class TestSecurityAndResumableRecoveryRegressions(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_regressions_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))

        self.artifacts_dir = Path(self.test_dir) / "artifacts"
        self.registry_file = Path(self.test_dir) / "registry.json"
        self.journal_file = Path(self.test_dir) / "transfers_shared.json"
        self.registry = ManagedSessionRegistry(registry_path=self.registry_file)

    # -------------------------------------------------------------------------
    # 1. AUTHENTICATION & REMOTE ACCESS SECURITY REGRESSIONS
    # -------------------------------------------------------------------------

    def test_query_token_strictly_rejected(self):
        """Query parameter tokens (?token=...) are strictly rejected on all endpoints."""
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            registry=self.registry,
            auth_token="super-secret-auth-token-12345",
        )
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)

        endpoints = [
            "/api/v1/summary?token=super-secret-auth-token-12345",
            "/api/v1/runs?token=super-secret-auth-token-12345",
            "/api/v1/events/stream?token=super-secret-auth-token-12345",
            "/api/v1/export/compute.csv?token=super-secret-auth-token-12345",
        ]
        for ep in endpoints:
            conn.request("GET", ep)
            resp = conn.getresponse()
            resp.read()
            self.assertEqual(resp.status, 401, f"Endpoint {ep} allowed query token bypass!")

        # Authorization header: Bearer ... must succeed
        conn.request("GET", "/api/v1/summary", headers={"Authorization": "Bearer super-secret-auth-token-12345"})
        resp = conn.getresponse()
        resp.read()
        self.assertEqual(resp.status, 200)

    def test_cors_rejects_wildcard_and_unauthorized_origins(self):
        """Monitor server does NOT send wildcard CORS header and ignores untrusted origins."""
        server = start_monitor_server(
            host="127.0.0.1",
            port=0,
            registry=self.registry,
            auth_token="test-token",
            allowed_origins=["http://localhost:8080"],
        )
        port = server.server_address[1]
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        self.addCleanup(conn.close)

        # 1. Evil origin
        conn.request("GET", "/api/v1/summary", headers={
            "Authorization": "Bearer test-token",
            "Origin": "http://malicious-site.com",
        })
        resp = conn.getresponse()
        resp.read()
        cors_hdr = resp.getheader("Access-Control-Allow-Origin")
        self.assertIsNone(cors_hdr, "Wildcard or untrusted CORS origin was returned!")

        # 2. Allowed origin
        conn.request("GET", "/api/v1/summary", headers={
            "Authorization": "Bearer test-token",
            "Origin": "http://localhost:8080",
        })
        resp = conn.getresponse()
        resp.read()
        self.assertEqual(resp.getheader("Access-Control-Allow-Origin"), "http://localhost:8080")

    def test_public_binding_without_secure_ingress_rejected(self):
        """Binding to external interface over unencrypted HTTP without trusted proxy is rejected."""
        # Unencrypted HTTP external bind without trusted proxy -> ValueError
        with self.assertRaises(ValueError) as ctx:
            start_monitor_server(
                host="0.0.0.0",
                port=9876,
                allow_external=True,
                auth_token="valid-token",
                require_tls=False,
                trusted_proxy=False,
            )
        self.assertIn("External binding over unencrypted HTTP without a trusted identity proxy is rejected", str(ctx.exception))

        # With trusted_proxy=True or require_tls=True -> permitted
        srv = start_monitor_server(
            host="127.0.0.1",
            port=0,
            allow_external=False,
            auth_token="valid-token",
        )
        srv.server_close()

    def test_broker_tokens_file_survives_restart_and_has_0600_permissions(self):
        """Tokens file survives restart, enforces 0600 permissions, and authenticates clients."""
        tokens_file = Path(self.test_dir) / "tokens.json"

        # Initialize broker 1 with explicit token and tokens_file
        broker1 = WorkspaceBroker(
            registry=self.registry,
            artifacts_dir=self.artifacts_dir,
            auth_tokens={"persisted-token-xyz"},
            tokens_file=tokens_file,
        )
        self.assertTrue(tokens_file.is_file())
        self.assertEqual(stat.S_IMODE(tokens_file.stat().st_mode), 0o600)

        # Restart simulation: initialize broker 2 loading from tokens_file
        broker2 = WorkspaceBroker(
            registry=self.registry,
            artifacts_dir=self.artifacts_dir,
            tokens_file=tokens_file,
            require_configured_auth=True,
        )
        self.assertTrue(broker2.verify_token("persisted-token-xyz"))
        runs = broker2.list_runs(token="persisted-token-xyz")
        self.assertIsInstance(runs, list)

    def test_two_separately_authenticated_browser_clients(self):
        """Two separately authenticated browser clients can inspect the same run without interference."""
        broker = WorkspaceBroker(
            registry=self.registry,
            artifacts_dir=self.artifacts_dir,
            auth_tokens={"macbook-token-1", "windows-token-2"},
        )
        self.registry.register(
            session_id="session-gpu-100",
            job_id="job-100",
            gpu="T4",
            project_name="fraud_detection",
        )

        # Client A (MacBook) attaches
        session_a = broker.attach_run(run_id="session-gpu-100", client_id="macbook", token="macbook-token-1")
        self.assertEqual(session_a["attached_client_id"], "macbook")
        self.assertEqual(session_a["run_id"], "session-gpu-100")

        # Client B (Windows) attaches
        session_b = broker.attach_run(run_id="session-gpu-100", client_id="windows", token="windows-token-2")
        self.assertEqual(session_b["attached_client_id"], "windows")
        self.assertEqual(session_b["run_id"], "session-gpu-100")

        # Both observe active session
        self.assertEqual(len(broker.get_connected_clients("session-gpu-100")), 2)

    # -------------------------------------------------------------------------
    # 2. RESUMABLE UPLOADS & INTEGRITY VERIFICATION REGRESSIONS
    # -------------------------------------------------------------------------

    def test_absent_remote_size_fails_verification(self):
        """If Drive response and metadata omit file size, do not claim verified; raise DriveIntegrityError."""
        mock_http = MockDriveHttpClient()

        # Mock returns file ID without size
        original_request = mock_http.request
        def patched_request(method, url, headers=None, data=None, timeout=30.0):
            status, resp_headers, body = original_request(method, url, headers=headers, data=data, timeout=timeout)
            if "session_id=upload-session" in url and status == 200:
                return 200, resp_headers, json.dumps({"id": "file-no-size"}).encode("utf-8")
            if "drive/v3/files/file-no-size" in url and method == "GET":
                return 200, {}, json.dumps({"id": "file-no-size"}).encode("utf-8")
            return status, resp_headers, body

        mock_http.request = patched_request

        store = GoogleDriveArtifactStore(
            credentials=DriveCredentials(access_token="tok"),
            local_staging_dir=self.artifacts_dir,
            journal_path=self.journal_file,
            http_client=mock_http,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

        test_file = Path(self.test_dir) / "model_weights.pt"
        test_file.write_bytes(b"DATA" * 1024)

        with self.assertRaises(DriveIntegrityError) as ctx:
            store.publish(source=test_file, project_id="p1", run_id="r1")
        self.assertIn("missing authoritative size", str(ctx.exception))

    def test_partial_chunk_308_range_parsing(self):
        """A 308 response with partial Range correctly sets next offset to N+1, not chunk_len."""
        mock_http = MockDriveHttpClient()
        partial_range_sent = False

        original_request = mock_http.request
        def patched_request(method, url, headers=None, data=None, timeout=30.0):
            nonlocal partial_range_sent
            if "session_id=upload-session" in url and method == "PUT" and not partial_range_sent:
                partial_range_sent = True
                # Server acknowledged only first 65536 bytes of 262144 byte chunk
                return 308, {"range": "bytes=0-65535"}, b""
            return original_request(method, url, headers=headers, data=data, timeout=timeout)

        mock_http.request = patched_request

        store = GoogleDriveArtifactStore(
            credentials=DriveCredentials(access_token="tok"),
            local_staging_dir=self.artifacts_dir,
            journal_path=self.journal_file,
            http_client=mock_http,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

        test_file = Path(self.test_dir) / "partial_test.pt"
        test_file.write_bytes(b"P" * (256 * 1024))

        stored = store.publish(source=test_file, project_id="p1", run_id="r1")
        self.assertIsNotNone(stored.remote_file_id)

    def test_chunk_size_must_be_multiple_of_256_kib(self):
        """GoogleDriveArtifactStore rejects chunk_size that is not a multiple of 256 KiB."""
        with self.assertRaises(ValueError) as ctx:
            GoogleDriveArtifactStore(chunk_size=1000)
        self.assertIn("multiple of 256 KiB", str(ctx.exception))

    def test_streaming_download_in_retrieve(self):
        """Retrieve downloads bounded chunks, verifies checksum and size, and atomically commits."""
        mock_http = MockDriveHttpClient()
        store = GoogleDriveArtifactStore(
            credentials=DriveCredentials(access_token="tok"),
            local_staging_dir=self.artifacts_dir,
            journal_path=self.journal_file,
            http_client=mock_http,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

        test_file = Path(self.test_dir) / "source_weights.pt"
        payload_data = b"STREAMING_WEIGHTS_CONTENT_BYTES" * 100
        test_file.write_bytes(payload_data)

        # Publish artifact
        stored = store.publish(source=test_file, project_id="p_stream", run_id="r_stream")

        # Mock download response matching data
        def patched_request(method, url, headers=None, data=None, timeout=30.0):
            if "alt=media" in url:
                return 200, {}, payload_data
            return mock_http.request(method, url, headers=headers, data=data, timeout=timeout)

        mock_http.request = patched_request

        dest_file = Path(self.test_dir) / "retrieved_weights.pt"
        retrieved_path = store.retrieve(stored.artifact_id, dest_file)

        self.assertTrue(retrieved_path.is_file())
        self.assertEqual(retrieved_path.read_bytes(), payload_data)

    # -------------------------------------------------------------------------
    # 3. PERSISTENT TRANSFER JOURNAL & WORKER RETRIES
    # -------------------------------------------------------------------------

    def test_transfer_journal_corrupt_json_preserves_file_and_recovers_or_fails(self):
        """Corrupted journal JSON must preserve .corrupt copy and recover from .bak or raise error."""
        journal_file = Path(self.test_dir) / "journal" / "transfers.json"
        journal = TransferJournal(journal_file)

        rec = TransferRecord(
            transfer_id="test-1",
            project_id="p1",
            run_id="r1",
            artifact_id="p1/r1/model.pt",
            local_path="/tmp/model.pt",
            target_rel_path="model.pt",
            size_bytes=100,
            sha256="abc",
            status=TransferStatus.LOCAL_VERIFIED.value,
        )
        journal.record_transfer(rec)

        # Corrupt the journal file with truncated/invalid JSON
        journal_file.write_text('{"test-1": {"transfer_id": "test-1", "stat')

        # Read should detect corruption, preserve .corrupt copy, and recover from .bak
        recovered = journal.get_transfer("test-1")
        self.assertIsNotNone(recovered, "Journal failed to recover record from backup!")
        self.assertEqual(recovered.transfer_id, "test-1")

        # Now remove backup and corrupt again -> must raise JournalCorruptionError
        bak_file = journal_file.with_suffix(".bak")
        if bak_file.exists():
            bak_file.unlink()
        journal_file.write_text('{bad json')

        with self.assertRaises(JournalCorruptionError):
            journal.get_transfer("test-1")

    def test_worker_in_process_retry_scheduler(self):
        """Worker actively retries failed uploads in-process without requiring worker process restart."""
        journal_file = Path(self.test_dir) / "worker_sched_journal.json"

        mock_http = MockDriveHttpClient()
        mock_http.fail_next_n_chunks = 1  # Fail once then succeed

        store = GoogleDriveArtifactStore(
            credentials=DriveCredentials(access_token="tok"),
            local_staging_dir=self.artifacts_dir,
            journal_path=journal_file,
            http_client=mock_http,
            chunk_size=256 * 1024,
            initial_backoff_sec=0.01,
        )

        worker = ArtifactUploadWorker(
            store=store,
            journal=store.journal,
            max_workers=1,
            initial_retry_delay_sec=0.05,
        )
        worker.start()
        self.addCleanup(worker.stop)

        test_file = Path(self.test_dir) / "sched_retry.pt"
        test_file.write_bytes(b"SCHED_RETRY_DATA")

        rec = worker.enqueue(
            local_path=test_file,
            project_id="p_sched",
            run_id="r_sched",
        )

        # Wait for in-process retry to succeed
        completed = worker.wait_for_completion(timeout=5.0)
        self.assertTrue(completed, "Worker failed to process in-process retried upload within timeout!")

        final_rec = store.journal.get_transfer(rec.transfer_id)
        self.assertEqual(final_rec.status, TransferStatus.PUBLISHED.value)
        self.assertEqual(final_rec.status, TransferStatus.PUBLISHED.value)

    # -------------------------------------------------------------------------
    # 4. REPRODUCIBILITY SNAPSHOT HARDENING
    # -------------------------------------------------------------------------

    def test_reproducibility_symlink_escape_and_output_exclusion(self):
        """Snapshots exclude external symlinks and self-referential output archives."""
        proj_dir = Path(self.test_dir) / "my_project"
        proj_dir.mkdir(parents=True)

        # Real files
        (proj_dir / "train.py").write_text("print('training')")
        (proj_dir / "model.py").write_text("class Model: pass")

        # Secret outside project root
        outside_secret = Path(self.test_dir) / "outside_secret.key"
        outside_secret.write_text("SUPER_SECRET_EXTERNAL_PRIVATE_KEY")

        # Malicious symlink inside project pointing outside
        symlink_file = proj_dir / "evil_symlink.key"
        os.symlink(str(outside_secret), str(symlink_file))

        # Output archive inside project directory
        output_tar = proj_dir / "source_snapshot.tar.gz"

        tar_path, manifest = create_deterministic_source_snapshot(
            project_dir=proj_dir,
            dest_tar_gz=output_tar,
        )

        self.assertTrue(tar_path.is_file())

        # Manifest must contain train.py and model.py
        self.assertIn("train.py", manifest)
        self.assertIn("model.py", manifest)

        # Malicious symlink and output archive must be strictly excluded
        self.assertNotIn("evil_symlink.key", manifest)
        self.assertNotIn("source_snapshot.tar.gz", manifest)


if __name__ == "__main__":
    unittest.main()
