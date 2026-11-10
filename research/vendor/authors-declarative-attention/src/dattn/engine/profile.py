"""Explicit engine kwargs per model and use, with no profile inheritance."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from dattn.contracts import BackendProfile, DattnError

PROFILE_PATH_ENV = "DATTN_BACKEND_PROFILES"


def profile_path() -> Path:
    return Path(
        os.environ.get(PROFILE_PATH_ENV, Path(__file__).resolve().parents[3] / "research/backend_profiles.toml")
    )


def profile_row(model_id: str, capability: str = "sm100", use: str = "parity") -> dict:
    with profile_path().open("rb") as f:
        rows = tomllib.load(f)
    try:
        return rows[f"{model_id}/{use}"]
    except KeyError as exc:
        raise DattnError(f"No configuration for {model_id}/{use}") from exc


def load_profile(model_id: str, capability: str = "sm100", use: str = "parity") -> BackendProfile:
    row = profile_row(model_id, capability, use)
    return BackendProfile(**row)
