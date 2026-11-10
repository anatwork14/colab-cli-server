"""Launch ONLY a verified local vLLM process within an approved GPU VM.

This does not create a Colab session, authenticate, or own any provider lifecycle.
The PID start-time guard prevents killing an unrelated PID after reuse.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time

from research_ops.assets import verify as verify_asset
from research_ops.engine import verify as verify_engine
from .runner import _atomic_json, server_check

SAFE=re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,79}$")


def _start_ticks(pid):
    try:
        # /proc/<pid>/stat: starttime is field 22 (index 19 after comm).
        data=Path("/proc/"+str(pid)+"/stat").read_text()
        after=data.rsplit(")",1)[1].strip().split()
        return int(after[19])
    except (OSError,IndexError,ValueError):
        return None


def _owned(record):
    pid=record.get("pid")
    return (type(pid) is int and pid>0 and record.get("start_ticks") is not None
            and _start_ticks(pid)==record.get("start_ticks"))


def start(args):
    if not args.execute:
        raise ValueError("Explicit --execute required to start a GPU-serving process")
    if not SAFE.fullmatch(args.run_id):
        raise ValueError("Invalid process run ID")
    if not (1<=args.port<=65535 and 0.1<=args.gpu_memory_utilization<=0.96
            and args.max_model_len>=128 and 1<=args.max_images<=1024):
        raise ValueError("Invalid inference-serving limits")
    receipt=verify_asset(args.assets,args.model_asset,args.scratch)
    engine=verify_engine(args.engine_mode,args.source_root,args.source_revision,require_gpu=True)
    target=Path(args.output_root).resolve()/args.run_id
    target.mkdir(parents=True,exist_ok=False)
    output=target/"server.log"
    model_path=Path(receipt["staged_path"])
    argv=["vllm","serve",str(model_path),
          "--served-model-name",args.model,
          "--host","127.0.0.1","--port",str(args.port),
          "--max-model-len",str(args.max_model_len),
          "--gpu-memory-utilization",str(args.gpu_memory_utilization),
          "--limit-mm-per-prompt",json.dumps({"image":args.max_images})]
    if args.speculative_config_json:
        cfg=json.loads(args.speculative_config_json)
        if not isinstance(cfg,dict) or cfg.get("method") not in (
                "mtp","qwen3_5_mtp","gemma4_mtp"):
            raise ValueError("Unsupported speculative decoding method")
        n=cfg.get("num_speculative_tokens")
        if type(n) is not int or n<1 or n>16:
            raise ValueError("num_speculative_tokens must be 1..16")
        argv.extend(["--speculative-config",json.dumps(cfg,separators=(",",":"))])
    with output.open("x",encoding="utf8") as log:
        p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
    # The command doesn't wait for model loading, which can exceed the
    # Colab adapter's per-exec timeout. A separate status poll is required.
    record={
        "run_id":args.run_id,"pid":p.pid,"start_ticks":_start_ticks(p.pid),
        "gpu_process":"STARTING_UNQUALIFIED","serving_model":args.model,
        "engine":engine,"model_asset_sha256":receipt["tree_sha256"],
        "argv_without_tokens":argv,"log":str(output),
        "local_only":True,
    }
    _atomic_json(target/"server_process.json",record)
    return record


def status(args):
    target=Path(args.output_root).resolve()/args.run_id
    record=json.loads((target/"server_process.json").read_text(encoding="utf8"))
    if not _owned(record):
        return {"status":"NOT_RUNNING","run_id":args.run_id,"owned_pid_alive":False}
    try:
        check=server_check("http://127.0.0.1:"+str(args.port),record["serving_model"])
        return {"status":"READY_VERSION_VERIFIED","run_id":args.run_id,
                "owned_pid_alive":True,"server":check}
    except Exception as exc:
        return {"status":"STARTING_OR_FAILED","run_id":args.run_id,
                "owned_pid_alive":True,"reason_type":type(exc).__name__}


def stop(args):
    target=Path(args.output_root).resolve()/args.run_id
    record=json.loads((target/"server_process.json").read_text(encoding="utf8"))
    if not _owned(record):
        return {"status":"NOT_RUNNING_OR_PID_REUSED","stopped":False}
    if not args.execute:
        raise ValueError("stop requires explicit --execute")
    pgid=os.getpgid(record["pid"])
    if pgid==os.getpgrp():
        raise RuntimeError("Refusing to terminate own controller process group")
    os.killpg(pgid,signal.SIGTERM)
    return {"status":"STOP_REQUESTED","pid":record["pid"],"stopped":True,
            "termination_verified":False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("start","status","stop"))
    p.add_argument("--assets")
    p.add_argument("--model-asset")
    p.add_argument("--scratch",default="/content/research-assets")
    p.add_argument("--engine-mode",default="stock",
                   choices=("stock","source_python","source_compiled"))
    p.add_argument("--source-root")
    p.add_argument("--source-revision")
    p.add_argument("--model")
    p.add_argument("--port",type=int,default=8000)
    p.add_argument("--max-model-len",type=int,default=32768)
    p.add_argument("--max-images",type=int,default=160)
    p.add_argument("--gpu-memory-utilization",type=float,default=0.85)
    p.add_argument("--speculative-config-json")
    p.add_argument("--output-root",default="/content/research-vllm-servers")
    p.add_argument("--run-id",required=True)
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    if a.action=="start" and (not a.assets or not a.model_asset or not a.model):
        p.error("start requires pinned --assets, --model-asset and --model")
    print(json.dumps({"start":start,"status":status,"stop":stop}[a.action](a),
                     indent=2,default=str))


if __name__=="__main__":
    main()
