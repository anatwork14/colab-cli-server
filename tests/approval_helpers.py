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
