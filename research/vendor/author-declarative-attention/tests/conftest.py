"""Shared fixtures. Keeps the whole unit directory under 60 s (DESIGN.md section 7.2)."""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT / "tests"))
PACKAGE = SRC / "dattn"

if str(SRC) not in sys.path:  # `uv run pytest` without an editable install
    sys.path.insert(0, str(SRC))


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def package_root() -> Path:
    return PACKAGE


@pytest.fixture(scope="session")
def profiles_toml(repo_root: Path) -> Path:
    return repo_root / "research" / "backend_profiles.toml"
