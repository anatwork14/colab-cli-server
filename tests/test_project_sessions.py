"""Tests for Milestone M5: Project Session Lifecycle and Lease Management.

Validates:
- State machine invariants and illegal transition rejection.
- Collision-resistant session ID generation.
- Offline planning without remote allocation.
- Cross-process locking and atomic intent registration in registry.
- Idempotent session reuse during active valid leases.
- Fail-closed refusal of unmanaged sessions for attach, exec, and stop (Q08).
- Lease expiration detection and lock-aware reconciliation.
"""
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

from colab_adapter.errors import (
    AllocationError,
    CleanupError,
    InvalidStateTransitionError,
    LeaseExpiredError,
    UnmanagedSessionError,
)
from colab_adapter.project_manifest import ProjectSpec
from colab_adapter.project_sessions import (
    ProjectSession,
    ProjectSessionManager,
    ProjectSessionState,
    VALID_TRANSITIONS,
)
from colab_adapter.provider import ColabProcessLock, ManagedSessionRegistry
from tests.approval_helpers import grant_v2


class TestProjectSessions(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.tmp_dir) / "managed_sessions.json"
        self.lock_path = Path(self.tmp_dir) / "colab.lock"
        self.registry = ManagedSessionRegistry(self.registry_path)
        self.lock = ColabProcessLock(self.lock_path)
        # The plan fingerprint covers the project tree, so keep the project apart from the registry and approvals.
        project_root = Path(self.tmp_dir) / "project"
        project_root.mkdir()
        (project_root / "train.py").write_text("print('train')\n", encoding="utf-8")
        self.spec = ProjectSpec(
            name="resnet_cifar",
            root=project_root,
            entrypoint="train.py",
            gpu="T4",
            timeout_seconds=1800.0,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_state_machine_legal_and_illegal_transitions(self):
        """Validates that valid transitions succeed and illegal transitions raise InvalidStateTransitionError."""
        session = ProjectSession(
            session_id="colab-v2-test-1111",
            project_id="proj-12345",
            project_name="resnet_cifar",
            state=ProjectSessionState.PLANNED.value,
            accelerator="T4",
            created_at=datetime.now(timezone.utc).isoformat(),
            updated_at=datetime.now(timezone.utc).isoformat(),
            lease_seconds=3600.0,
            expires_at=datetime.now(timezone.utc).isoformat(),
            remote_root="/content/workspace",
        )

        # Illegal: PLANNED -> RUNNING
        with self.assertRaises(InvalidStateTransitionError) as ctx:
            session.transition_to(ProjectSessionState.RUNNING)
        self.assertIn("Illegal session state transition", str(ctx.exception))

        # Legal: PLANNED -> PROVISIONING -> READY -> RUNNING -> READY -> STOPPING -> TERMINATED
        session.transition_to(ProjectSessionState.PROVISIONING)
        self.assertEqual(session.state, ProjectSessionState.PROVISIONING.value)

        session.transition_to(ProjectSessionState.READY)
        self.assertEqual(session.state, ProjectSessionState.READY.value)

        session.transition_to(ProjectSessionState.RUNNING)
        self.assertEqual(session.state, ProjectSessionState.RUNNING.value)

        session.transition_to(ProjectSessionState.READY)
        self.assertEqual(session.state, ProjectSessionState.READY.value)

        session.transition_to(ProjectSessionState.STOPPING)
        self.assertEqual(session.state, ProjectSessionState.STOPPING.value)

        session.transition_to(ProjectSessionState.TERMINATED)
        self.assertEqual(session.state, ProjectSessionState.TERMINATED.value)

        # Terminal state: cannot transition out of TERMINATED
        with self.assertRaises(InvalidStateTransitionError):
            session.transition_to(ProjectSessionState.READY)

    def test_offline_plan_session_does_not_invoke_cli_or_registry(self):
        """Q03: Planning a session must generate collision-resistant IDs and touch 0 cloud resources."""
        mock_run = MagicMock()
        manager = ProjectSessionManager(
            registry=self.registry,
            lock=self.lock,
            run_cmd_fn=mock_run,
        )

        session = manager.plan_session(self.spec, lease_seconds=1800.0)
        self.assertTrue(session.session_id.startswith("colab-v2-resnet_cifar-"))
        self.assertEqual(len(session.session_id.split("-")[-1]), 8)  # 8 hex chars
        self.assertEqual(session.state, ProjectSessionState.PLANNED.value)
        self.assertEqual(session.lease_seconds, 1800.0)
        self.assertTrue(session.is_lease_active())

        # No subprocess calls made
        mock_run.assert_not_called()
        # Not yet registered in registry
        self.assertFalse(self.registry.is_managed(session.session_id))

    def test_up_provisions_under_lock_and_registers_intent_first(self):
        """Session provisioning registers PROVISIONING intent before executing CLI command."""
        recorded_states = []

        def mock_run_cmd(cmd, **kwargs):
            # Check registry state during CLI execution
            all_managed = self.registry.get_all_managed()
            for s_id, info in all_managed.items():
                recorded_states.append(info.get("state"))
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Session created", stderr="")

        manager = ProjectSessionManager(
            registry=self.registry,
            lock=self.lock,
            run_cmd_fn=mock_run_cmd,
        )

        session = manager.up(self.spec, lease_seconds=3600.0, approval_id=grant_v2(manager, self.spec, 3600.0))
        self.assertEqual(session.state, ProjectSessionState.READY.value)
        self.assertIn(ProjectSessionState.PROVISIONING.value, recorded_states)

        # Registry should now reflect READY state
        stored = self.registry.get_all_managed()[session.session_id]
        self.assertEqual(stored["state"], ProjectSessionState.READY.value)
        self.assertEqual(stored["accelerator"], "T4")
        self.assertTrue(self.registry.is_managed(session.session_id))

    def test_up_allocation_failure_records_failed_state(self):
        """Failed CLI allocation transitions session to FAILED and raises AllocationError."""
        def mock_run_fail(cmd, **kwargs):
            return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="Quota exceeded for GPU T4")

        manager = ProjectSessionManager(
            registry=self.registry,
            lock=self.lock,
            run_cmd_fn=mock_run_fail,
        )

        with self.assertRaises(AllocationError) as ctx:
            manager.up(self.spec, approval_id=grant_v2(manager, self.spec))
        self.assertIn("Quota exceeded", str(ctx.exception))

        all_managed = self.registry.get_all_managed()
        self.assertEqual(len(all_managed), 1)
        info = list(all_managed.values())[0]
        self.assertEqual(info["state"], ProjectSessionState.FAILED.value)

    def test_up_reuses_existing_active_session_idempotently(self):
        """Multiple calls to up() for the same project reuse active valid lease without re-allocating."""
        mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="ok", stderr=""))
        manager = ProjectSessionManager(
            registry=self.registry,
            lock=self.lock,
            run_cmd_fn=mock_run,
        )

        s1 = manager.up(self.spec, lease_seconds=3600.0, approval_id=grant_v2(manager, self.spec, 3600.0))
        self.assertEqual(mock_run.call_count, 1)

        # Second call to up with reuse_active=True (default)
        s2 = manager.up(self.spec, lease_seconds=3600.0, reuse_active=True)
        self.assertEqual(s1.session_id, s2.session_id)
        # Verify no second CLI allocation was made
        self.assertEqual(mock_run.call_count, 1)

    def test_q08_attach_refuses_unmanaged_session(self):
        """Q08: Refuses to attach to unmanaged or unknown sessions."""
        manager = ProjectSessionManager(registry=self.registry, lock=self.lock)

        with self.assertRaises(UnmanagedSessionError) as ctx:
            manager.attach("colab-unmanaged-external-session")
        self.assertIn("is not registered or owned", str(ctx.exception))

    def test_attach_expired_lease_raises_lease_expired_error(self):
        """Attaching to a session whose lease expired raises LeaseExpiredError."""
        session_id = "colab-v2-expired-1"
        expired_dt = datetime.fromtimestamp(time.time() - 100, timezone.utc).isoformat()
        with self.registry._lock():
            data = self.registry._read_locked()
            data[session_id] = {
                "session_id": session_id,
                "project_id": "proj-12345",
                "project_name": "resnet_cifar",
                "state": ProjectSessionState.READY.value,
                "accelerator": "T4",
                "created_at": expired_dt,
                "updated_at": expired_dt,
                "lease_seconds": 60.0,
                "expires_at": expired_dt,
                "remote_root": "/content/workspace",
            }
            self.registry._atomic_write_locked(data)

        manager = ProjectSessionManager(registry=self.registry, lock=self.lock)
        with self.assertRaises(LeaseExpiredError) as ctx:
            manager.attach(session_id)
        self.assertIn("lease expired", str(ctx.exception))

    def test_exec_lifecycle_and_state_tracking(self):
        """exec() transitions session state from READY to RUNNING and back to READY."""
        # Setup managed session
        session_id = "colab-v2-exec-test-1"
        now_iso = datetime.now(timezone.utc).isoformat()
        future_iso = datetime.fromtimestamp(time.time() + 3600, timezone.utc).isoformat()
        with self.registry._lock():
            data = self.registry._read_locked()
            data[session_id] = {
                "session_id": session_id,
                "project_id": "proj-12345",
                "project_name": "resnet_cifar",
                "state": ProjectSessionState.READY.value,
                "accelerator": "T4",
                "created_at": now_iso,
                "updated_at": now_iso,
                "lease_seconds": 3600.0,
                "expires_at": future_iso,
                "remote_root": "/content/workspace",
            }
            self.registry._atomic_write_locked(data)

        mock_transport = MagicMock()
        mock_transport.exec_code.return_value = {
            "success": True,
            "output": "Training loss: 0.12",
            "has_traceback": False,
        }

        manager = ProjectSessionManager(
            registry=self.registry,
            lock=self.lock,
            transport=mock_transport,
        )

        res = manager.exec(session_id, "python train.py", working_dir="/content/workspace")
        self.assertTrue(res["success"])
        self.assertIn("Training loss", res["output"])
        mock_transport.exec_code.assert_called_once()
        # In registry, state is back to READY
        self.assertEqual(self.registry.get_all_managed()[session_id]["state"], ProjectSessionState.READY.value)

    def test_q08_stop_refuses_unmanaged_session(self):
        """Q08: stop() strictly refuses to touch unmanaged sessions; 0 CLI commands dispatched."""
        mock_run = MagicMock()
        manager = ProjectSessionManager(
            registry=self.registry,
            lock=self.lock,
            run_cmd_fn=mock_run,
        )

        with self.assertRaises(UnmanagedSessionError) as ctx:
            manager.stop("unmanaged-colab-session-777")
        self.assertIn("Refusing to stop unmanaged session", str(ctx.exception))
        mock_run.assert_not_called()

    def test_stop_managed_session_transitions_to_terminated(self):
        """Stopping a managed session executes CLI stop and transitions state to TERMINATED."""
        session_id = "colab-v2-stop-test-1"
        now_iso = datetime.now(timezone.utc).isoformat()
        future_iso = datetime.fromtimestamp(time.time() + 3600, timezone.utc).isoformat()
        with self.registry._lock():
            data = self.registry._read_locked()
            data[session_id] = {
                "session_id": session_id,
                "project_id": "proj-12345",
                "project_name": "resnet_cifar",
                "state": ProjectSessionState.READY.value,
                "accelerator": "T4",
                "created_at": now_iso,
                "updated_at": now_iso,
                "lease_seconds": 3600.0,
                "expires_at": future_iso,
                "remote_root": "/content/workspace",
            }
            self.registry._atomic_write_locked(data)

        mock_run = MagicMock(return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""))
        manager = ProjectSessionManager(
            registry=self.registry,
            lock=self.lock,
            run_cmd_fn=mock_run,
        )

        ok = manager.stop(session_id)
        self.assertTrue(ok)
        mock_run.assert_called_once_with(
            ["colab", "stop", "-s", session_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60.0,
        )
        self.assertEqual(self.registry.get_all_managed()[session_id]["state"], ProjectSessionState.TERMINATED.value)

    def test_reconcile_leases_auto_stops_expired_sessions(self):
        """reconcile_leases identifies expired leases and stops only managed expired sessions."""
        expired_id = "colab-v2-expired-rec"
        active_id = "colab-v2-active-rec"

        now = time.time()
        expired_dt = datetime.fromtimestamp(now - 120, timezone.utc).isoformat()
        future_dt = datetime.fromtimestamp(now + 1800, timezone.utc).isoformat()

        with self.registry._lock():
            data = self.registry._read_locked()
            data[expired_id] = {
                "session_id": expired_id,
                "project_id": "proj-1",
                "project_name": "expired_proj",
                "state": ProjectSessionState.READY.value,
                "accelerator": "T4",
                "created_at": expired_dt,
                "updated_at": expired_dt,
                "lease_seconds": 60.0,
                "expires_at": expired_dt,
                "remote_root": "/content/p1",
            }
            data[active_id] = {
                "session_id": active_id,
                "project_id": "proj-2",
                "project_name": "active_proj",
                "state": ProjectSessionState.READY.value,
                "accelerator": "T4",
                "created_at": expired_dt,
                "updated_at": expired_dt,
                "lease_seconds": 3600.0,
                "expires_at": future_dt,
                "remote_root": "/content/p2",
            }
            self.registry._atomic_write_locked(data)

        stopped_calls = []

        def mock_run_cmd(cmd, **kwargs):
            stopped_calls.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

        manager = ProjectSessionManager(
            registry=self.registry,
            lock=self.lock,
            run_cmd_fn=mock_run_cmd,
        )

        report = manager.reconcile_leases(auto_stop=True)
        self.assertEqual(report["stopped"], [expired_id])
        self.assertEqual(report["active"], [active_id])
        self.assertEqual(stopped_calls, [["colab", "stop", "-s", expired_id]])

        # Expired is now terminated, active remains ready
        self.assertEqual(self.registry.get_all_managed()[expired_id]["state"], ProjectSessionState.TERMINATED.value)
        self.assertEqual(self.registry.get_all_managed()[active_id]["state"], ProjectSessionState.READY.value)


if __name__ == "__main__":
    unittest.main()
