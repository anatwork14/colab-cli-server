"""Agent-safe read-only campaign planner: checks real files, never allocates Colab."""
import argparse
import json
from pathlib import Path

from research_ops.assets import read_manifest, verify as verify_asset
from .datasets import audit, KIND


def campaign_plan(path):
    cfg=json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg.get("schema_version")!=1 or cfg.get("training") is not False or cfg.get("synthetic_data") is not False:
        raise ValueError("Campaign must be schema 1, training=false, synthetic_data=false")
    studies=cfg.get("datasets")
    if not isinstance(studies,list) or len(studies)!=2:
        raise ValueError("Campaign must explicitly include exactly two datasets")
    if set(x.get("kind") for x in studies)!={"memlens","mmlongbench_doc"}:
        raise ValueError("Campaign requires MemLens and MMLongBench-Doc")
    all_assets=cfg.get("assets_manifest")
    root=cfg.get("asset_scratch")
    reasons=[]
    asset_inventory=[]
    if not all_assets or not Path(all_assets).is_file():
        reasons.append("No real pinned asset manifest found")
    else:
        try:
            manifest=read_manifest(all_assets)
            asset_inventory=[a["id"] for a in manifest["assets"]]
        except Exception as e:
            reasons.append("Invalid pinned asset manifest: "+type(e).__name__)
    if not root:
        reasons.append("No asset scratch directory")
    datasets=[]
    for x in studies:
        ident=x["kind"]
        report={"kind":ident,"status":"BLOCKED_UNSTAGED_REAL_DATA"}
        asset=x.get("asset_id")
        samples_rel=x.get("samples_rel")
        expected=x.get("expected_count")
        if not asset or not samples_rel or not root or not all_assets:
            report["blocker"]="Asset ID, samples_rel, scratch and manifest required"
        elif not Path(all_assets).exists():
            report["blocker"]="Asset manifest not present"
        else:
            try:
                receipt=verify_asset(all_assets,asset,root)
                dataset_root=Path(receipt["staged_path"])
                sample=dataset_root/samples_rel
                report=audit(ident,sample,dataset_root,expected_count=expected,
                             deep_pdf=bool(x.get("deep_pdf")))
            except Exception as e:
                report["blocker"]=type(e).__name__+": "+str(e)[:160]
        if report.get("status")!="COMPLETE_FOR_ENGINEERING_PILOT":
            reasons.append(ident+" integrity or staging not verified")
        datasets.append(report)
    model_id=cfg.get("model_asset_id")
    model_receipt=None
    if all_assets and root and model_id and Path(all_assets).is_file():
        try:
            m=verify_asset(all_assets,model_id,root)
            model_receipt={"asset_id":model_id,"tree_sha256":m["tree_sha256"],
                           "status":"VERIFIED_STAGED_ASSET"}
        except Exception as e:
            reasons.append("Model weights not verified staged: "+type(e).__name__)
    else:
        reasons.append("Model asset ID/manifest/scratch missing")
    gpu={
        "status":"NOT_QUERIED_NO_ALLOCATION",
        "requires_operator_approved_colab_project_up":True,
        "requires_actual_vllm_server_and_engine_verification":True,
    }
    return {
        "status":"OFFLINE_DATA_PREFLIGHT_PASS_GPU_UNVERIFIED" if not reasons
            else "BLOCKED_BEFORE_GPU",
        "reasons":sorted(set(reasons)),
        "datasets":datasets,"model":model_receipt,"gpu":gpu,
        "official_memlens_judge":"NOT_RUN",
        "official_mmlong_answer_extractor":"NOT_RUN",
        "official_qualifications":"BLOCKED_UNTIL_REAL_EVALUATOR_PARITY",
        "no_cloud_action":True,"run_ready":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("plan",))
    p.add_argument("--campaign",required=True)
    p.add_argument("--output")
    args=p.parse_args()
    result=campaign_plan(args.campaign)
    if args.output:
        out=Path(args.output)
        out.parent.mkdir(parents=True,exist_ok=True)
        with out.open("x",encoding="utf8") as f:
            json.dump(result,f,indent=2)
            f.write("\n")
    print(json.dumps(result,indent=2))
    if result["status"]=="BLOCKED_BEFORE_GPU":
        raise SystemExit(2)


if __name__=="__main__":
    main()
