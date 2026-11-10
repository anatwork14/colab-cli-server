"""Fixtures for the Lane D CLI tests.

Everything here is synthetic. The one place a real dac tree is used is
`dac_outputs`, which is opt-in through `$DATTN_DAC_OUTPUTS` and skipped otherwise, so the
suite runs on a box that has never seen the reference repository (`PROVENANCE.md`).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fakes import judgment_row, response_row, write_cell


@pytest.fixture
def dac_outputs() -> Path:
    """The recorded dac output root, when the operator has pointed at one."""
    env = os.environ.get("DATTN_DAC_OUTPUTS")
    if not env:
        pytest.skip("set DATTN_DAC_OUTPUTS to a dac outputs root to run this test")
    path = Path(env)
    if not path.is_dir():
        pytest.skip(f"DATTN_DAC_OUTPUTS={env} is not a directory")
    return path


@pytest.fixture
def make_cell():
    return write_cell


@pytest.fixture
def row():
    return response_row


@pytest.fixture
def verdict_row():
    return judgment_row
