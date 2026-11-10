"""Read-only 2-model x 3-arm x 7-length x 6-CCU GPU-sweep planner.

Pure Python stdlib, no external resources, allocation or synthetic training
data. A synthetic random-token *load* request is not a benchmark dataset.
"""
import argparse
import json
from pathlib import Path

DEFAULT_CONFIG=Path(__file__).resolve().parents[2]/"research"/"benchmarks"/"qwen35_inference_sweep.json"
CCU=(1,2,4,8,16,32)
LENGTHS=(128,512,1024,2048,4096,8192,16384)
ARMS=("base","h2o","da")
MODELS=("qwen35_08b","qwen35_4b")


def read_config(path=DEFAULT_CONFIG):
    obj=json.loads(Path(path).read_text(encoding="utf8"))
    if obj.get("schema_version")!=1 or obj.get("engine_version")!="0.29.0":
        raise ValueError("Unsupported frozen research protocol")
    if tuple(obj.get("concurrency",[]))!=CCU or tuple(obj.get("input_lengths",[]))!=LENGTHS:
        raise ValueError("Input/CCU grid changed; create a new protocol version")
    if tuple(x["key"] for x in obj.get("models",[]))!=MODELS:
        raise ValueError("Wrong two Qwen3.5 model profiles")
    if tuple(x["key"] for x in obj.get("arms",[]))!=ARMS:
        raise ValueError("Base/H2O/DA arm list changed")
    if obj.get("output_tokens")!=256 or obj.get("stream") is not True or obj.get("thinking") is not False:
        raise ValueError("256-token streaming non-thinking protocol drift")
    server=obj.get("server",{})
    if server.get("max_num_seqs")!=32 or server.get("enable_prefix_caching") is not False:
        raise ValueError("Missing max-num-seqs 32 / disabled prefix cache")
    if server.get("max_model_len",0)<max(LENGTHS)+obj["output_tokens"]+256:
        raise ValueError("Server model length cannot fit longest inputs plus chat header/output")
    if obj.get("endpoint")!="/v1/chat/completions":
        raise ValueError("Wrong measured endpoint")
    sampling=obj.get("sampling",{})
    if sampling.get("temperature")!=0 or sampling.get("ignore_eos") is not True or sampling.get("min_tokens")!=256:
        raise ValueError("Requested fixed-output decoding contract missing")
    return obj


def cells(config=None, model=None, arm=None, context=None, ccu=None):
    c=config or read_config()
    if model is not None and model not in MODELS:
        raise ValueError("Invalid model")
    if arm is not None and arm not in ARMS:
        raise ValueError("Invalid arm")
    if context is not None and context not in LENGTHS:
        raise ValueError("Invalid context")
    if ccu is not None and ccu not in CCU:
        raise ValueError("Invalid CCU")
    out=[]
    for m in c["models"]:
        if model and m["key"]!=model:continue
        for a in c["arms"]:
            if arm and a["key"]!=arm:continue
            for length in LENGTHS:
                if context and context!=length:continue
                for concurrency in CCU:
                    if ccu and ccu!=concurrency:continue
                    cell_id="{}_{}_i{:05d}_c{:02d}".format(
                        m["key"],a["key"],length,concurrency)
                    out.append({
                        "id":cell_id,"model":m["key"],"model_id":m["id"],
                        "arm":a["key"],"input_tokens":length,
                        "ccu":concurrency,
                        "requests":max(16,4*concurrency),"warmups":concurrency,
                        "output_tokens":256,"source_revision_required":True,
                        "status":a["qualification"],"gpu_run_completed":False,
                    })
    return out


def summary(items):
    by_arm={key:sum(int(x["arm"]==key) for x in items) for key in ARMS}
    return {
        "cells":len(items),"cells_by_arm":by_arm,
        "measured_requests":sum(x["requests"] for x in items),
        "warmup_requests":sum(x["warmups"] for x in items),
        "all_requests":sum(x["requests"]+x["warmups"] for x in items),
        "full_run_qualified":False,"source":"offline_plan_only",
    }


def main():
    p=argparse.ArgumentParser(description="Offline 252-cell research plan; NO GPU allocation")
    p.add_argument("--protocol",default=str(DEFAULT_CONFIG))
    p.add_argument("--model",choices=MODELS)
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--context",type=int)
    p.add_argument("--ccu",type=int)
    p.add_argument("--output")
    a=p.parse_args()
    config=read_config(a.protocol)
    result={"protocol":config["id"],
            "items":cells(config,a.model,a.arm,a.context,a.ccu)}
    result["summary"]=summary(result["items"])
    if a.output:
        path=Path(a.output).resolve()
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",encoding="utf8") as f:
            json.dump(result,f,indent=2,sort_keys=True)
            f.write("\n")
    print(json.dumps(result if (a.model and a.context and a.ccu) else
        {"protocol":result["protocol"],"summary":result["summary"],
         "first_three":result["items"][:3]},indent=2))


if __name__=="__main__":
    main()
