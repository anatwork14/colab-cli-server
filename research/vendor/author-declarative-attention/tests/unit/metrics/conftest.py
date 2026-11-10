"""Fixtures over the Lane C fake engine."""

from __future__ import annotations

import signal
from typing import Any

import pytest
from fakes import FakeRunner, FakeTorch


@pytest.fixture
def fake_torch() -> type[FakeTorch]:
    FakeTorch.reset()
    return FakeTorch


@pytest.fixture
def runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture(autouse=True)
def no_signal_leak() -> Any:
    """Hand SIGTERM and SIGINT back after every test in this package.

    `install_from_env` with a record directory arms a drain on both signals from the first
    forward, and a test that never calls `uninstall` would otherwise leave that handler on
    the pytest PROCESS and change what the next test observes. Same class of defect as the
    statlog monkeypatch leak in `tests/unit/bench/conftest.py`: a process-global installed
    by a unit under test is the unit's to undo, and the suite's to guarantee.
    """
    saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    yield
    for sig, handler in saved.items():
        signal.signal(sig, handler)
