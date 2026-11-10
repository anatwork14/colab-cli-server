"""Packaging contract: one general plugin, no logits-processor entry point.

DESIGN.md section 2.7 gives two independent reasons for registering the logits processor
by fully qualified class name instead: only the CLI route sets
`logitsprocs_need_output_token_ids`, and the MRV2 demotion check counts INSTALLED entry
points, which would make every arm's model runner depend on what is in the venv.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from dattn.engine.plugin import BACKEND_CLASS_PATH, LOGITS_PROCESSOR_FQCN


def _pyproject(repo_root: Path) -> dict:
    with (repo_root / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_the_general_plugin_entry_point_is_declared(repo_root: Path) -> None:
    eps = _pyproject(repo_root)["project"]["entry-points"]
    assert eps["vllm.general_plugins"] == {"dattn": "dattn.engine.plugin:install_plugin"}


def test_no_logits_processor_entry_point_is_declared(repo_root: Path) -> None:
    eps = _pyproject(repo_root)["project"]["entry-points"]
    assert "vllm.logits_processors" not in eps, (
        "shipping a vllm.logits_processors entry point would demote the model runner "
        "based on what is installed in the venv (DESIGN.md 2.7)"
    )


def test_the_console_script_is_the_cli(repo_root: Path) -> None:
    assert _pyproject(repo_root)["project"]["scripts"]["dattn"] == "dattn.cli:main"


def test_the_fqcn_constants_resolve_to_real_objects() -> None:
    import importlib

    module, _, attr = LOGITS_PROCESSOR_FQCN.partition(":")
    assert getattr(importlib.import_module(module), attr).__name__ == "DattnLogitsProcessor"
    module, _, attr = BACKEND_CLASS_PATH.rpartition(".")
    # DattnBackend is built at plugin load from the pinned base class (WP5), so only the
    # module must exist here.
    assert importlib.import_module(module).__name__ == "dattn.engine.backend"
    assert attr == "DattnBackend"
