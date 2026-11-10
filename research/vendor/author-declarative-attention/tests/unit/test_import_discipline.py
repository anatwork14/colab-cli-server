"""The import-discipline test of DESIGN.md section 1.

Walks the AST of every module under `src/dattn` and asserts the layering:

    dattn.contracts    imports nothing outside the stdlib
    dattn.fsm          stdlib and dattn.contracts only
    dattn.prompt       may add transformers
    dattn.engine / dattn.metrics / dattn.bench   may import vllm or torch
    dattn.eval         may import dattn.engine.api but not vllm

This is what makes 90% of the suite run on a CPU box in under a minute.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

STDLIB = frozenset(sys.stdlib_module_names)
ANY = None  # a sentinel meaning "no restriction"


@dataclass(frozen=True)
class Rule:
    #: Third-party top-level packages this layer may import. `ANY` lifts the check.
    third_party: frozenset[str] | None
    #: `dattn.` prefixes this layer may import. `ANY` lifts the check.
    internal: tuple[str, ...] | None


_FSM = ("dattn.contracts", "dattn.fsm")
_PROMPT = (*_FSM, "dattn.prompt")

# Longest matching prefix wins.
RULES: dict[str, Rule] = {
    "dattn": Rule(frozenset(), ("dattn.contracts",)),
    "dattn.contracts": Rule(frozenset(), ()),
    "dattn.fsm": Rule(frozenset(), _FSM),
    "dattn.prompt": Rule(frozenset({"transformers"}), _PROMPT),
    "dattn.engine": Rule(ANY, (*_PROMPT, "dattn.engine")),
    "dattn.metrics": Rule(ANY, ("dattn.contracts", "dattn.metrics", "dattn.engine.api")),
    "dattn.bench": Rule(ANY, ANY),
    # WP8 (Lane D) adds `datasets`: reproducing dac's row selection means reproducing
    # `Dataset.filter -> shuffle(seed) -> select` exactly, and only `dattn.eval.datasets`
    # imports it, inside the one function that needs it. The transport deliberately stays
    # stdlib (`urllib.request`), so no HTTP client is on this list.
    "dattn.eval": Rule(
        frozenset({"transformers", "numpy", "pandas", "datasets"}),
        ("dattn.contracts", "dattn.fsm", "dattn.prompt", "dattn.eval", "dattn.engine.api"),
    ),
    # Campaign orchestration is CPU-only. Engines are subprocesses through the CLI.
    "dattn.campaign": Rule(
        frozenset({"transformers"}),
        ("dattn.campaign", "dattn.eval", "dattn.prompt", "dattn.cli"),
    ),
    "dattn.cli": Rule(ANY, ANY),
}

#: Never importable from `dattn.eval`, whatever else the rule allows.
EVAL_FORBIDDEN = frozenset({"vllm", "torch"})


def _modules(package_root: Path) -> list[tuple[str, Path]]:
    out = []
    for path in sorted(package_root.rglob("*.py")):
        parts = path.relative_to(package_root.parent).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        out.append((".".join(parts), path))
    return out


def _imports(path: Path, module: str) -> list[str]:
    """Every absolute module name this file imports, relative imports resolved."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    pkg = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = ".".join(pkg.split(".")[: len(pkg.split(".")) - node.level + 1])
                names.append(f"{base}.{node.module}" if node.module else base)
            elif node.module:
                names.append(node.module)
    return names


def _rule_for(module: str) -> Rule:
    best = ""
    for prefix in RULES:
        if (module == prefix or module.startswith(prefix + ".")) and len(prefix) > len(best):
            best = prefix
    assert best, f"no import rule covers {module}; add one to RULES"
    return RULES[best]


def test_every_module_has_a_rule(package_root: Path) -> None:
    mods = _modules(package_root)
    assert mods, "no modules found under src/dattn"
    for module, _ in mods:
        _rule_for(module)


def test_layering(package_root: Path) -> None:
    failures: list[str] = []
    for module, path in _modules(package_root):
        rule = _rule_for(module)
        for imported in _imports(path, module):
            top = imported.split(".")[0]
            if top == "dattn":
                if rule.internal is ANY:
                    continue
                if not any(imported == p or imported.startswith(p + ".") for p in rule.internal):
                    failures.append(f"{module} may not import {imported} (allowed: {rule.internal})")
            elif top not in STDLIB:
                if rule.third_party is ANY:
                    continue
                if top not in rule.third_party:
                    failures.append(f"{module} may not import third-party {imported}")
    assert not failures, "import discipline violated:\n  " + "\n  ".join(failures)


def test_eval_never_imports_vllm_or_torch(package_root: Path) -> None:
    for module, path in _modules(package_root):
        if not module.startswith("dattn.eval"):
            continue
        for imported in _imports(path, module):
            assert imported.split(".")[0] not in EVAL_FORBIDDEN, f"{module} imports {imported}"


def test_metrics_never_imports_engine_internals(package_root: Path) -> None:
    """WP6 couples to Lane B through `dattn.engine.api` and the `_REC` slot, nothing else."""
    for module, path in _modules(package_root):
        if not module.startswith("dattn.metrics"):
            continue
        for imported in _imports(path, module):
            if imported.startswith("dattn.engine"):
                assert imported == "dattn.engine.api", f"{module} imports {imported}"


def test_contracts_imports_only_stdlib_at_runtime(package_root: Path) -> None:
    """`import dattn.contracts` succeeds with NO third-party package importable.

    `-S` drops site-packages (and every `.pth` hook in it) off `sys.path`, so this is the
    "no third-party packages installed" environment of the WP0 acceptance criterion.
    """
    code = "import sys, json; import dattn.contracts; print(json.dumps(sorted({m.split('.')[0] for m in sys.modules})))"
    env_src = str(package_root.parent)
    proc = subprocess.run(
        [sys.executable, "-S", "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONPATH": env_src},
    )
    loaded = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    extra = {m for m in loaded if not m.startswith("_") and m not in STDLIB and m != "dattn"}
    assert not extra, f"importing dattn.contracts pulled in {sorted(extra)}"
