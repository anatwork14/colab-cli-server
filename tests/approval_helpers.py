"""Offline test helpers that grant operator approvals without a terminal.

Only tests import this module. Production code never reads a phrase from anywhere but /dev/tty.
"""
from __future__ import annotations

from unittest import mock

APPROVAL_READER = "colab_adapter.approval.read_tty_phrase"


def grant_record(store, record):
    """Approve a pending record by answering its exact phrase through the mocked terminal reader."""
    with mock.patch(APPROVAL_READER, return_value=record["phrase"]):
        store.approve(record["approval_id"])
    return record["approval_id"]


def grant_v1(provider, prep):
    """Approval id for a prepared V1 job, granted through the mocked terminal reader."""
    phrase = f"ALLOCATE {prep.spec.gpu.upper()} {prep.approval_nonce}"
    with mock.patch(APPROVAL_READER, return_value=phrase):
        return provider.approve_job_interactive(prep).approval_id


def grant_v2(manager, spec, lease_seconds=3600.0):
    """Approval id for a V2 project plan, requested and granted through the mocked terminal reader."""
    record = manager.request_approval(spec, lease_seconds=lease_seconds)
    return grant_record(manager.approvals, record)


def fake_guard_spawner(test_case):
    """spawn_fn for ProjectSessionManager in offline tests.

    Starts a harmless sleeper whose argv carries 'lease-guard' and the session id, so the
    manager's pid and cmdline checks treat it like a real guard. It never runs the real guard
    and never touches a real colab binary. Sleepers are killed in test cleanup.
    """
    import subprocess
    import sys
    from datetime import datetime, timezone

    started = []

    def spawn(cmd, **kwargs):
        session_id = cmd[cmd.index("--session-id") + 1]
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(3600)", "lease-guard", session_id],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        started.append(proc)
        # Mirror the real guard's startup marker so the manager's confirmation step succeeds.
        from colab_adapter.lease_guard import merge_guard_metadata
        from colab_adapter.provider import ManagedSessionRegistry
        from pathlib import Path

        merge_guard_metadata(
            ManagedSessionRegistry(Path(cmd[cmd.index("--registry-path") + 1])),
            session_id,
            {"guard_started_at": datetime.now(timezone.utc).isoformat()},
        )
        return proc

    def _cleanup():
        for proc in started:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    test_case.addCleanup(_cleanup)
    return spawn
