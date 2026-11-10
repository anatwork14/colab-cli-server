"""Collect only fully qualified finite CCU cells; never infer missing scores."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

from .plan import read_config,cells,summary


FIELDS=("id","model","arm","input_tokens","ccu","status",
        "attempt_id","requests","warmups","success_count","failure_count",
        "duration_s","ttft_p50_ms","ttft_p95_ms","e2e_p50_ms",
        "e2e_p95_ms","tpot_estimate_p50_ms","tpot_estimate_p95_ms",
        "completed_requests_per_sec","output_tokens_per_sec",
        "input_length_verified_on_server","output_length_verified_on_server",
        "protocol_sha256","model_commit","model_tree_sha256","seed")


def _result_row(obj):
    cell=obj["cell"]
    server=obj.get("server",{})
    return {
        "id":cell["id"],"model":cell["model"],"arm":cell["arm"],
        "input_tokens":cell["input_tokens"],"ccu":cell["ccu"],
        "requests":cell["requests"],"warmups":cell["warmups"],
        "status":obj.get("status"),"attempt_id":obj.get("attempt_id"),
        "success_count":obj.get("success_count"),
        "failure_count":obj.get("failure_count"),
        "duration_s":obj.get("duration_s"),
        "ttft_p50_ms":obj.get("ttft_p50_ms"),
        "ttft_p95_ms":obj.get("ttft_p95_ms"),
        "e2e_p50_ms":obj.get("e2e_p50_ms"),
        "e2e_p95_ms":obj.get("e2e_p95_ms"),
        "tpot_estimate_p50_ms":obj.get("tpot_estimate_p50_ms"),
        "tpot_estimate_p95_ms":obj.get("tpot_estimate_p95_ms"),
        "completed_requests_per_sec":obj.get("completed_requests_per_sec"),
        "output_tokens_per_sec":obj.get("output_tokens_per_sec"),
        "input_length_verified_on_server":obj.get("input_length_verified_on_server"),
        "output_length_verified_on_server":obj.get("output_length_verified_on_server"),
        "protocol_sha256":obj.get("protocol_sha256"),
        "model_commit":server.get("model_commit"),
        "model_tree_sha256":server.get("model_tree_sha256"),
        "seed":obj.get("seed"),
    }


def collect(root,config_path):
    config=read_config(config_path)
    from research_ops.assets import hash_file
    frozen=hash_file(config_path)
    all_cells={c["id"]:c for c in cells(config)}
    outputs=[]
    blocked=[]
    by_cell=defaultdict(list)
    for p in sorted(Path(root).glob("*/summary.json")):
        try:
            obj=json.loads(p.read_text(encoding="utf8"))
            cell=obj["cell"]
            name=cell["id"]
            if name not in all_cells or cell != all_cells[name]:
                raise ValueError("Run cell differs from original 252-cell matrix")
            if obj.get("protocol_sha256")!=frozen:
                raise ValueError("Result contains different protocol revision")
            if (obj.get("status")!="ENGINEERING_COMPLETE" or
                obj.get("success_count")!=cell["requests"] or
                obj.get("warmup_count")!=cell["warmups"] or
                obj.get("failure_count")!=0 or
                obj.get("input_length_verified_on_server") is not True or
                obj.get("output_length_verified_on_server") is not True):
                raise ValueError("Incomplete or unverified request population")
            if cell["arm"]!="base":
                raise ValueError("H2O and DA GPU qualification remains blocked")
            by_cell[name].append(_result_row(obj))
        except (KeyError,TypeError,ValueError,IndexError) as exc:
            blocked.append({"path":str(p),"reason":str(exc)[:180]})
    for cell_id,entries in sorted(by_cell.items()):
        if len(entries)!=1:
            blocked.append({"cell":cell_id,"reason":"Multiple complete attempts; avoid selection bias"})
        else:
            outputs.append(entries[0])
    completed={row["id"] for row in outputs}
    base={c["id"] for c in all_cells.values() if c["arm"]=="base"}
    report={
        "protocol":config["id"],"protocol_sha256":frozen,
        "total_grid_cells":len(all_cells),"base_cells_expected":len(base),
        "base_cells_valid":len(completed),"base_missing":sorted(base-completed),
        "h2o_gpu_cells_qualified":0,"da_gpu_cells_qualified":0,
        "blocked_attempts":blocked,
        "valid_base_campaign":len(completed)==len(base) and not blocked,
        "all_252_cells_ready":False,
        "official_accuracy":None,"official_hallucination":None,
        "source":"only_real_GPU_receipts_no_inferred_data",
    }
    return outputs,report


def main():
    p=argparse.ArgumentParser(description="Aggregate verified original GPU throughput cell receipts")
    p.add_argument("--root",required=True,help="Path of host-downloaded GPU result directories")
    p.add_argument("--protocol",default=str(__import__("research_ops.throughput.plan",fromlist=["DEFAULT_CONFIG"]).DEFAULT_CONFIG))
    p.add_argument("--csv",required=True)
    p.add_argument("--report",required=True)
    a=p.parse_args()
    rows,report=collect(a.root,a.protocol)
    out=Path(a.csv).resolve()
    target=Path(a.report).resolve()
    if out.exists() or target.exists():
        raise FileExistsError("Refuse overwrite of previous benchmark aggregates")
    out.parent.mkdir(parents=True,exist_ok=True)
    target.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",newline="",encoding="utf8") as stream:
        writer=csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with target.open("x",encoding="utf8") as stream:
        json.dump(report,stream,indent=2,sort_keys=True)
        stream.write("\n")
    print(json.dumps({"csv":str(out),"report_path":str(target),
                      "validated_base_cells":len(rows),
                      "blocked_attempts":len(report["blocked_attempts"]),
                      "campaign_ready":report["valid_base_campaign"]},indent=2))


if __name__=="__main__":
    main()
