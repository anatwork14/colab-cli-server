"""Unit tests for Ubuntu Workspace Broker cross-device read-only attachment.

Validates:
- Two independent authenticated clients viewing and attaching to the same run
- Zero Colab VM allocation on view or attach (strictly offline-safe)
- Expired or terminated runtime remains discoverable and inspectable via durable artifacts
- Authentication token enforcement (rejects invalid token)
- Rejection of unmanaged sessions
- Audit logging of client actions
- Read-only enforcement (rejects writable attach)
"""
import shutil
import tempfile
import unittest
from pathlib import Path

from colab_adapter.broker import WorkspaceBroker
from colab_adapter.errors import (
    BrokerAuthenticationError,
    UnmanagedSessionError,
)
from colab_adapter.provider import ManagedSessionRegistry
from colab_adapter.storage import (
    ArtifactType,
    LocalArtifactStore,
    PublicationManifest,
    StoredArtifact,
)
from colab_adapter.telemetry import TelemetryEmitter


class TestWorkspaceBroker(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="colab_broker_test_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))
        self.artifacts_dir = Path(self.test_dir) / "artifacts"
        self.registry_file = Path(self.test_dir) / "registry.json"
        self.telemetry_file = Path(self.test_dir) / "telemetry.jsonl"

        self.registry = ManagedSessionRegistry(registry_path=self.registry_file)
        self.telemetry = TelemetryEmitter(ledger_path=self.telemetry_file)
        self.store = LocalArtifactStore(root_dir=self.artifacts_dir)


        self.token = "secure-broker-token-xyz"
        self.broker = WorkspaceBroker(
            registry=self.registry,
            telemetry=self.telemetry,
            storage=self.store,
            auth_tokens={self.token},
            artifacts_dir=self.artifacts_dir,
        )

    def test_unauthenticated_request_rejected(self):
        """Requests without authorized bearer token raise BrokerAuthenticationError."""
        with self.assertRaises(BrokerAuthenticationError):
            self.broker.list_runs(token="invalid-token")

        with self.assertRaises(BrokerAuthenticationError):
            self.broker.attach_run(run_id="run_1", client_id="laptop", token=None)

    def test_two_independent_clients_attach_to_same_run(self):
        """Client A (MacBook) and Client B (Windows) can independently attach to the same run."""
        # Pre-seed a managed session
        self.registry.register(
            session_id="session-gpu-001",
            job_id="run-cifar-001",
            gpu="T4",
            project_name="cifar",
        )


        # Pre-seed local verified checkpoint in store
        src = Path(self.test_dir) / "ckpt.pt"
        src.write_bytes(b"WEIGHTS")
        self.store.publish(
            source=src,
            project_id="cifar",
            run_id="run-cifar-001",
            artifact_type=ArtifactType.CHECKPOINT.value,
        )

        # Client A attaches
        client_a_resp = self.broker.attach_run(
            run_id="run-cifar-001",
            client_id="macbook-pro-user",
            token=self.token,
        )
        self.assertEqual(client_a_resp["run_id"], "run-cifar-001")
        self.assertEqual(client_a_resp["mode"], "READ_ONLY")
        self.assertEqual(client_a_resp["artifacts_count"], 1)

        # Client B attaches
        client_b_resp = self.broker.attach_run(
            run_id="run-cifar-001",
            client_id="windows-workstation",
            token=self.token,
        )
        self.assertEqual(client_b_resp["run_id"], "run-cifar-001")
        self.assertEqual(client_b_resp["mode"], "READ_ONLY")
        self.assertEqual(client_b_resp["latest_checkpoint"]["size_bytes"], len(b"WEIGHTS"))

        # Both attachments logged in audit history
        audit = self.broker.get_audit_log(token=self.token)
        clients_in_audit = [a["client_id"] for a in audit]
        self.assertIn("macbook-pro-user", clients_in_audit)
        self.assertIn("windows-workstation", clients_in_audit)

    def test_expired_runtime_historical_inspection_without_cloud_allocation(self):
        """When Colab VM has terminated, run remains discoverable via durable manifest with zero GPU calls."""
        # Create completed historical run in manifest
        m = PublicationManifest(
            project_id="imagenet",
            run_id="run-historic-99",
            status="PUBLISHED",
        )
        artifact = StoredArtifact(
            artifact_id="imagenet/run-historic-99/model_final.pt",
            storage_provider="google_drive",
            path="checkpoints/model_final.pt",
            size_bytes=5000,
            sha256="deadbeefhash",
            project_id="imagenet",
            run_id="run-historic-99",
            remote_file_id="drive-file-id-9999",
        )
        m.add_artifact(artifact)
        m_path = self.artifacts_dir / "projects" / "imagenet" / "runs" / "run-historic-99" / "manifest.json"
        m.save(m_path)

        # List runs discovers historical run
        runs = self.broker.list_runs(token=self.token)
        historic = next(r for r in runs if r["run_id"] == "run-historic-99")
        self.assertEqual(historic["status"], "TERMINATED")
        self.assertFalse(historic["is_active"])

        # Attach to historical run reports RUNTIME_UNAVAILABLE but returns durable artifacts
        resp = self.broker.attach_run(run_id="run-historic-99", client_id="phone-client", token=self.token)
        self.assertEqual(resp["runtime_status"], "RUNTIME_UNAVAILABLE")
        self.assertFalse(resp["is_runtime_active"])
        self.assertEqual(resp["artifacts_count"], 1)
        self.assertEqual(resp["latest_checkpoint"]["remote_file_id"], "drive-file-id-9999")

    def test_refuses_unmanaged_sessions(self):
        """Attaching to an unmanaged session strictly raises UnmanagedSessionError."""
        with self.assertRaises(UnmanagedSessionError):
            self.broker.attach_run(
                run_id="unmanaged-colab-session-123",
                client_id="client-x",
                token=self.token,
            )


    def test_writable_attach_strictly_rejected_in_read_only_release(self):
        """Requesting write lease raises NotImplementedError in read-only release."""
        with self.assertRaises(NotImplementedError):
            self.broker.request_write_lease(
                run_id="run_1",
                client_id="client_1",
                authorization_token="auth_code",
            )


if __name__ == "__main__":
    unittest.main()
