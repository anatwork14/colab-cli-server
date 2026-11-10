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
from research_ops.vllm_engine.matrix import MODES, METHODS, make_arm, command_for_arm

SAFE=re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,79}$")

def _assert_model_origin(receipt, expected_repo_id):
    """Bind local Drive/archive copies to a pinned original HF model revision."""
    source=receipt.get("source")
    if receipt.get("role") != "model" or not isinstance(source, dict):
        raise ValueError("Experiment checkpoints must be verified model assets")
    if source.get("type") == "huggingface":
        repo_id, revision=source.get("repo_id"),source.get("revision")
    elif source.get("type") in ("directory", "archive"):
        repo_id, revision=source.get("origin_repo_id"),source.get("origin_revision")
        source_tree=source.get("origin_tree_sha256")
        staged_tree=receipt.get("tree_sha256")
        if (not isinstance(source_tree,str) or
                not re.fullmatch(r"[a-f0-9]{64}",source_tree) or
                source_tree!=staged_tree):
            raise ValueError(
                "Local model mirror needs an original HF snapshot tree SHA256 "
                "matching the exact staged bytes; labels alone are insufficient")
    else:
        raise ValueError("Unsupported model source provenance")
    if repo_id != expected_repo_id or not isinstance(revision,str) or not re.fullmatch(r"[a-f0-9]{40}",revision):
        raise ValueError("Wrong or unpinned original Hugging Face checkpoint source")
    return {"repo_id":repo_id,"revision":revision}




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
    if not (1 <= args.port <= 65535 and
            0.1 <= args.gpu_memory_utilization <= 0.96 and
            args.max_model_len >= 128 and 1 <= args.max_images <= 1024):
        raise ValueError("Invalid inference-serving limits")
    # The H2O repository's plugin is a registration canary, not H2O.
    # Co-loading it with DA uses the same singleton CUSTOM backend slot.
    if os.environ.get("H2O_VLLM_MODE","disabled").strip().lower() not in ("","disabled"):
        raise RuntimeError("Benchmark process must disable H2O CUSTOM canary; use isolated engines")
    if getattr(args,"experiment_mode",None) and any(
            os.environ.get(key) for key in ("DATTN_MODEL","DATTN_BASE_BACKEND")):
        raise RuntimeError(
            "Normal/speculative benchmark must not inherit author DA plugin activation; "
            "use a clean isolated engine process")
    if getattr(args,"h2o_policy_manifest",None):
        from research_ops.vllm_engine.h2o_bridge import (
            validate_h2o_policy_receipt, reject_h2o_gpu_execution)
        validate_h2o_policy_receipt(args.h2o_policy_manifest)
        reject_h2o_gpu_execution()
    receipt=verify_asset(args.assets,args.model_asset,args.scratch)
    model_path=Path(receipt["staged_path"])
    arm=None
    draft_receipt=None

    # Keep legacy call sites intact unless a research mode is explicitly used.
    if getattr(args,"experiment_mode",None):
        arm=make_arm(args.model_profile,args.experiment_mode,args.spec_method)
        if arm.da_roles:
            raise RuntimeError(
                "Author dattn vLLM 0.29.0 DA is not Qwen3.5-VLM/vision-layout verified. "
                "Cannot run stock attention and claim Declarative Attention.")
        if args.speculative_config_json:
            raise ValueError("Cannot combine research arm and legacy speculative configuration")
        target_origin=_assert_model_origin(receipt, arm.target_id)
        if arm.method in ("dflash","eagle3"):
            if not args.draft_asset:
                raise ValueError("DFlash/EAGLE3 requires pinned draft asset ID")
            draft_receipt=verify_asset(args.assets,args.draft_asset,args.scratch)
            draft_origin=_assert_model_origin(draft_receipt,arm.draft_id)
            if draft_receipt["tree_sha256"] == receipt["tree_sha256"]:
                raise ValueError("Draft and target checkpoints unexpectedly identical")
        elif args.draft_asset:
            raise ValueError("MTP uses target-native layers, no external draft")
        argv=command_for_arm(arm,str(model_path),
              draft_path=(str(Path(draft_receipt["staged_path"])) if draft_receipt else None),
              served_name=args.model,max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_images=args.max_images)
        argv.extend(["--port",str(args.port)])
    else:
        if getattr(args,"spec_method",None) or getattr(args,"draft_asset",None):
            raise ValueError("Use --experiment-mode when selecting research drafts")
        argv=["vllm","serve",str(model_path),"--served-model-name",args.model,
              "--host","127.0.0.1","--port",str(args.port),
              "--max-model-len",str(args.max_model_len),
              "--gpu-memory-utilization",str(args.gpu_memory_utilization),
              "--limit-mm-per-prompt",json.dumps({"image":args.max_images})]
        if args.speculative_config_json:
            cfg=json.loads(args.speculative_config_json)
            if not isinstance(cfg,dict) or cfg.get("method") not in (
                    "mtp","qwen3_5_mtp","gemma4_mtp"):
                raise ValueError("Legacy speculative method unsupported")
            n=cfg.get("num_speculative_tokens")
            if type(n) is not int or not 1<=n<=16:
                raise ValueError("Invalid speculative token count")
            argv.extend(["--speculative-config",json.dumps(cfg,separators=(",",":"))])
    engine=verify_engine(args.engine_mode,args.source_root,args.source_revision,require_gpu=True)
    target=Path(args.output_root).resolve()/args.run_id
    target.mkdir(parents=True,exist_ok=False)
    output=target/"server.log"
    with output.open("x",encoding="utf8") as log:
        p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
    record={
        "run_id":args.run_id,"pid":p.pid,"start_ticks":_start_ticks(p.pid),
        "gpu_process":"STARTING_UNQUALIFIED","serving_model":args.model,
        "engine":engine,"model_asset_sha256":receipt["tree_sha256"],
        "argv_without_tokens":argv,"log":str(output),"local_only":True,
        "experiment_arm":dict(arm.to_dict(), offline_plan_only=False,
                               server_process_start_requested=True) if arm else None,
        "target_origin":target_origin if arm else None,
        "draft_origin":draft_origin if draft_receipt else None,
        "draft_asset_tree_sha256":(
            draft_receipt["tree_sha256"] if draft_receipt else None),
        "da_hook_runtime_verified":False,
    }
    _atomic_json(target/"server_process.json",record)
    return record


def _port_from_record(record):
    """The caller cannot pick an unrelated local vLLM service at status time."""
    argv=record.get("argv_without_tokens")
    if not isinstance(argv,list) or argv.count("--port") != 1:
        raise ValueError("Frozen server receipt must record exactly one --port")
    idx=argv.index("--port")
    try:
        port=int(argv[idx+1])
    except (IndexError,TypeError,ValueError):
        raise ValueError("Invalid frozen port in server receipt")
    if not 1<=port<=65535 or "--host" not in argv:
        raise ValueError("Frozen server must declare localhost bind and valid port")
    host_index=argv.index("--host")
    if host_index+1>=len(argv) or argv[host_index+1]!="127.0.0.1":
        raise ValueError("Frozen server unexpectedly binds to a non-loopback host")
    return port


def status(args):
    target=Path(args.output_root).resolve()/args.run_id
    record=json.loads((target/"server_process.json").read_text(encoding="utf8"))
    if not _owned(record):
        return {"status":"NOT_RUNNING","run_id":args.run_id,"owned_pid_alive":False}
    try:
        frozen_port=_port_from_record(record)
    except ValueError:
        return {"status":"INVALID_FROZEN_SERVER_RECEIPT","run_id":args.run_id,"owned_pid_alive":True}
    if frozen_port!=args.port:
        return {"status":"FROZEN_PORT_MISMATCH","run_id":args.run_id,
                "owned_pid_alive":True,"frozen_port":frozen_port}
    try:
        check=server_check("http://127.0.0.1:"+str(frozen_port),record["serving_model"])
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
    if pgid==os.getpgrp() or pgid!=record["pid"] or not _owned(record):
        raise RuntimeError("Refusing termination of non-owned or recycled server process group")
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
    p.add_argument("--experiment-mode",choices=MODES,
                   help="Opt into strict six-mode research experiment")
    p.add_argument("--spec-method",choices=METHODS,
                   help="For speculative and speculative+DA categories")
    p.add_argument("--model-profile",default="qwen35_4b",
                   help="Qwen3.5 model profile in pinned checkpoint catalog")
    p.add_argument("--draft-asset",
                   help="Verified staged HF draft asset ID for DFlash/EAGLE3")
    p.add_argument("--h2o-policy-manifest",
                   help="Optional H2O policy-only receipt; GPU execution is intentionally blocked")
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
