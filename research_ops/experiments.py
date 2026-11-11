"""Freeze an experiment cell before executing any real GPU inference."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from research_ops.assets import read_manifest, verify, hash_file
from research_ops.engine import plan as engine_plan, verify as engine_verify

SLUG = re.compile(r"^[a-z][a-z0-9_-]{2,63}$")
RUN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,79}$")


def read_experiment(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema_version") != 1:
        raise ValueError("Invalid experiment schema")
    for key in ("study_id", "experiment_id", "research_question", "hypothesis"):
        if not isinstance(doc.get(key), str) or not doc[key]:
            raise ValueError("Missing experiment field " + key)
    for key in ("study_id", "experiment_id"):
        if not SLUG.fullmatch(doc[key]):
            raise ValueError("Invalid experiment " + key)
    if doc.get("kind") != "inference_engine_benchmark":
        raise ValueError("This workflow never trains/fine-tunes or creates datasets")
    if doc.get("synthetic_data") is not False or doc.get("training") is not False:
        raise ValueError("Research cells must assert synthetic_data=false and training=false")
    cells = doc.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("At least one experimental cell needed")
    identifiers = set()
    for c in cells:
        if not isinstance(c, dict) or not SLUG.fullmatch(str(c.get("id", ""))):
            raise ValueError("Cell needs safe ID")
        if c["id"] in identifiers:
            raise ValueError("Duplicate experimental cell")
        identifiers.add(c["id"])
        engine = c.get("engine")
        if (not isinstance(engine, dict) or
                engine.get("mode") not in ("stock", "source_python", "source_compiled")):
            raise ValueError("Engine mode must be stock/source_python/source_compiled")
        for label in ("model_asset_id", "dataset_asset_id"):
            if not isinstance(c.get(label), str) or not SLUG.fullmatch(c[label]):
                raise ValueError("Cell missing pinned " + label)
        if not isinstance(c.get("parameters"), dict):
            raise ValueError("Cell parameters must be a JSON object")
        if not isinstance(c.get("metrics"), list) or not c["metrics"]:
            raise ValueError("Metrics must be declared before run")
    return doc


def validate_experiment(experiment, assets):
    spec = read_experiment(experiment)
    asset_doc = read_manifest(assets)
    by_id = {a["id"]: a for a in asset_doc["assets"]}
    for cell in spec["cells"]:
        mid, did = cell["model_asset_id"], cell["dataset_asset_id"]
        if mid not in by_id or by_id[mid]["role"] != "model":
            raise ValueError("Missing model asset in pinned assets manifest: " + mid)
        if did not in by_id or by_id[did]["role"] != "dataset":
            raise ValueError("Missing dataset asset in pinned assets manifest: " + did)
        engine = cell["engine"]
        if engine.get("mode") == "stock" and (engine.get("source_root") or engine.get("source_revision")):
            raise ValueError("Stock arm cannot claim custom engine changes")
        if engine["mode"] != "stock":
            rev = engine.get("source_revision")
            if not isinstance(rev, str) or not re.fullmatch(r"[0-9a-f]{40}", rev):
                raise ValueError("Modified engine requires a full frozen commit SHA")
            if (not isinstance(engine.get("source_root"), str) or
                    not Path(engine["source_root"]).is_absolute()):
                raise ValueError("Modified engine requires an absolute source_root")
    return spec


def plan_experiment(experiment, assets, scratch):
    doc = validate_experiment(experiment, assets)
    return {"status": "OFFLINE_PLAN_NO_ALLOCATION", "study_id": doc["study_id"],
            "experiment_id": doc["experiment_id"],
            "cells": [{"id": c["id"], "engine": c["engine"]["mode"],
                       "model": c["model_asset_id"], "dataset": c["dataset_asset_id"]}
                      for c in doc["cells"]],
            "assets_staging_root": str(scratch),
            "real_data_only": True, "model_training": False}


def freeze(experiment, assets, scratch, output, run_id, require_gpu=False, cell_id=None):
    doc = validate_experiment(experiment, assets)
    if not RUN.fullmatch(run_id):
        raise ValueError("Invalid run ID")
    selected = [c for c in doc["cells"] if cell_id is None or c["id"] == cell_id]
    if len(selected) != 1:
        raise ValueError("Freeze exactly one cell per active engine installation; supply --cell-id")
    observations = {}
    for cell in selected:
        engine = cell["engine"]
        engine_status = engine_verify(engine["mode"], engine.get("source_root"),
                                      engine.get("source_revision"), require_gpu=require_gpu)
        model = verify(assets, cell["model_asset_id"], scratch)
        dataset = verify(assets, cell["dataset_asset_id"], scratch)
        observations[cell["id"]] = {
            "engine": engine_status, "model_asset": model, "dataset_asset": dataset,
            "parameters": cell["parameters"], "planned_metrics": cell["metrics"],
            "status": "FROZEN_NOT_EXECUTED",
        }
    root = Path(output).expanduser().resolve() / doc["study_id"] / doc["experiment_id"] / run_id
    root.mkdir(parents=True, exist_ok=False)
    manifest = {"schema_version": 1, "status": "FROZEN_NOT_EXECUTED",
                "kind": "inference_engine_benchmark", "run_id": run_id,
                "study_id": doc["study_id"], "experiment_id": doc["experiment_id"],
                "experiment_sha256": hash_file(experiment),
                "asset_manifest_sha256": hash_file(assets),
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "cells": observations, "results": None, "official_score": False,
                "note": "Freezing inputs is not evidence of GPU execution."}
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {"status": manifest["status"], "path": str(root / "run_manifest.json"),
            "cells_frozen": len(observations)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("plan", "freeze"))
    p.add_argument("--experiment", required=True)
    p.add_argument("--assets", required=True)
    p.add_argument("--scratch", default="/content/research-assets")
    p.add_argument("--output", default="/content/research-runs")
    p.add_argument("--run-id")
    p.add_argument("--cell-id", help="Exactly one cell; required for multi-arm studies")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    if a.action == "plan":
        out = plan_experiment(a.experiment, a.assets, a.scratch)
    else:
        if not a.execute or not a.run_id:
            p.error("freeze requires --execute and --run-id; never implies benchmark completion")
        out = freeze(a.experiment, a.assets, a.scratch, a.output, a.run_id, a.require_gpu, a.cell_id)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
