"""One finite streaming-chat throughput cell on a pre-authorized local vLLM GPU.

No Colab allocation, no training, no official dataset scoring. A cell is
write-once; retries are separate attempt IDs and receive fresh random seeds.
All inputs (warmup included) are original random token-ID draws decoded
and verified against the effective chat template length before timing.
"""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import subprocess
import tempfile
import time
from urllib.parse import urlsplit

from .plan import read_config, cells, MODELS, ARMS
from .workload import build_prompt, child_seed, request_stream
from research_ops.benchmarks.serve import _owned, _port_from_record
from research_ops.benchmarks.runner import server_check

RUN_ID=re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{2,79}$")


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    from research_ops.assets import hash_file
    return hash_file(path)


def _write_once(file, obj):
    path=Path(file)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf8") as output:
        json.dump(obj,output,ensure_ascii=True,sort_keys=True,indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def _pct(v, percent):
    if not v:return None
    a=sorted(v)
    idx=(len(a)-1)*percent/100
    lo=int(math.floor(idx));hi=int(math.ceil(idx))
    return round(a[lo]*(hi-idx)+a[hi]*(idx-lo),4) if hi!=lo else round(a[lo],4)


def _option(argv,name):
    if not isinstance(argv,list) or argv.count(name)!=1:
        raise RuntimeError("Frozen vLLM serving command lacks one "+name)
    i=argv.index(name)+1
    if i>=len(argv):
        raise RuntimeError("Frozen vLLM option has no value "+name)
    return argv[i]


def gpu_info():
    proc=subprocess.run(
        ["nvidia-smi","--query-gpu=name,memory.total,memory.used,driver_version",
         "--format=csv,noheader,nounits"],
        capture_output=True,text=True,check=True,timeout=20)
    if not proc.stdout.strip():
        raise RuntimeError("No NVIDIA GPU found for actual throughput test")
    return proc.stdout.strip().splitlines()


def attest_server(receipt_path,model_key,arm_key,config,url):
    if arm_key=="h2o":
        raise RuntimeError(
            "BLOCKED: h2o-vllm is CPU policy + FlashAttention passthrough; "
            "real physical H2O KV eviction is not implemented.")
    if arm_key=="da":
        raise RuntimeError(
            "BLOCKED: author DA on Qwen3.5 0.8B/4B VLM is not GPU "
            "correctness-qualified; no DA performance label is allowed.")
    if arm_key!="base":
        raise ValueError("Unknown mode")
    from research_ops.vllm_engine.matrix import make_arm
    model=next(x for x in config["models"] if x["key"]==model_key)
    record=json.loads(Path(receipt_path).read_text(encoding="utf8"))
    if not _owned(record):
        raise RuntimeError("Original vLLM server process is not live/owned")
    arm=make_arm(model_key,"normal")
    emitted=record.get("experiment_arm")
    if (not isinstance(emitted,dict) or emitted.get("id")!=arm.slug or
            emitted.get("offline_plan_only") is not False):
        raise RuntimeError("Server is not the specified live base-model research arm")
    if record.get("serving_model")!=model["id"]:
        raise RuntimeError("Server served-model alias does not match selected checkpoint")
    original=record.get("target_origin")
    if (not isinstance(original,dict) or original.get("repo_id")!=model["id"] or
            not re.fullmatch(r"[0-9a-f]{40}",str(original.get("revision","")))):
        raise RuntimeError("Full HF publisher model checkpoint identity absent")
    engine=record.get("engine",{})
    if engine.get("installed_version")!="0.29.0":
        raise RuntimeError("GPU server is not vLLM 0.29.0")
    argv=record.get("argv_without_tokens",[])
    required=("--no-enable-prefix-caching",)
    if any(name not in argv for name in required):
        raise RuntimeError("Prefix caching must be explicitly disabled at server startup")
    if "--speculative-config" in argv:
        raise RuntimeError("Base-model server must not use speculative decoding")
    if "--default-chat-template-kwargs" not in argv:
        raise RuntimeError("Server must freeze thinking-off default chat kwargs")
    chat_kwargs=json.loads(_option(argv,"--default-chat-template-kwargs"))
    if chat_kwargs.get("enable_thinking") is not False:
        raise RuntimeError("Server does not default to thinking off")
    if int(_option(argv,"--max-num-seqs"))!=32:
        raise RuntimeError("Server must allow 32 scheduled sequences")
    max_model_len=int(_option(argv,"--max-model-len"))
    if max_model_len<config["server"]["max_model_len"]:
        raise RuntimeError("Server context budget is too small for 16K+256 chat requests")
    u=urlsplit(url)
    if (u.scheme!="http" or u.hostname not in ("localhost","127.0.0.1") or
            u.username or u.password or u.query or u.fragment or u.path.rstrip("/") not in
            ("","/v1")):
        raise ValueError("Only the actual local vLLM server is permitted")
    port=u.port or 80
    if port != _port_from_record(record):
        raise RuntimeError("Server receipt port does not match target localhost URL")
    check=server_check("http://127.0.0.1:{}".format(port),model["id"])
    if check["vllm_version"]!="0.29.0":
        raise RuntimeError("Unexpected live engine version")
    model_path=Path(argv[2])
    if len(argv)<3 or argv[0:2]!=["vllm","serve"] or not model_path.is_dir():
        raise RuntimeError("Server did not stage the local actual checkpoint directory")
    return {
        "status":"BASE_ENGINE_IDENTITY_VERIFIED_NOT_GPU_BENCHMARK_PASS",
        "server":check,"port":port,"model_path":str(model_path),
        "server_pid":record["pid"],
        "server_start_ticks":record["start_ticks"],
        "model_id":model["id"],
        "model_commit":original["revision"],
        "model_tree_sha256":record["model_asset_sha256"],
        "engine":engine,
        "argv":argv,
        "max_model_len":max_model_len,
        "max_num_seqs":32,"prefix_caching":False,
        "arm_id":arm.slug,
    }


def prepare_prompts(tokenizer,ctx,count,master_seed):
    """All warmup and measured requests must have unique, verified prompts."""
    results=[];seen=set()
    for i in range(count):
        seed=child_seed(master_seed,i)
        data=build_prompt(tokenizer,ctx,seed)
        if data["prompt_tokens"]!=ctx or data["prompt_sha256"] in seen:
            raise RuntimeError("Prompt invalid, duplicate, or wrong effective input length")
        seen.add(data["prompt_sha256"])
        data["request_index"]=i
        results.append(data)
    return results


def _one(index,kind,model,url,prompt,seed,timeout):
    try:
        observation=request_stream(url,model,prompt,seed,timeout=timeout)
        observation["status"]="SUCCESS"
    except Exception as exc:
        observation={"status":"FAILED","error_type":type(exc).__name__,
                     "error_message":str(exc)[:250]}
    observation.update({"request_index":index,"phase":kind,
                        "prompt_sha256":prompt["prompt_sha256"],
                        "prompt_tokens_target":prompt["prompt_tokens"],
                        "seed":seed})
    return observation


def _phase(entries,kind,ccu,model,url,timeout,output_dir):
    """Requests start only when a client concurrency slot is free."""
    observations=[None]*len(entries)
    with ThreadPoolExecutor(max_workers=ccu) as ex:
        jobs=[ex.submit(_one,i,kind,model,url,entry,
                        entry["seed"],timeout)
              for i,entry in enumerate(entries)]
        for job in as_completed(jobs):
            result=job.result()
            observations[result["request_index"]]=result
            # Immutable per-request receipt survives abrupt GPU VM loss until
            # the agent downloads it to the dedicated host.
            _write_once(Path(output_dir)/"requests"/kind/
                        "{:04d}.json".format(result["request_index"]),result)
    return observations


def summarize(cell,measured,elapsed,warmup,seed,hardware,server,protocol_sha):
    good=[x for x in measured if x["status"]=="SUCCESS"]
    failed=len(measured)-len(good)
    value=lambda key:[r[key] for r in good
                      if type(r.get(key)) in (float,int)]
    result={
        "schema_version":1,"cell":cell,"status":(
            "ENGINEERING_COMPLETE" if failed==0 and len(warmup)==cell["warmups"]
            else "INCOMPLETE_NOT_COMPARABLE"),
        "protocol_sha256":protocol_sha,
        "random_input_kind":"random_tokens_decoded_into_chat_text",
        "raw_prompt_token_ids_sent_to_chat":False,
        "all_prompts_unique":True,
        "seed":seed,"measured_count":len(measured),
        "success_count":len(good),"failure_count":failed,
        "warmup_count":len(warmup),
        "duration_s":round(elapsed,5),
        "input_length_verified_on_server":all(
            r.get("prompt_tokens")==cell["input_tokens"] for r in good) if good else False,
        "output_length_verified_on_server":all(
            r.get("completion_tokens")==256 for r in good) if good else False,
        "ttft_p50_ms":_pct(value("ttft_ms"),50),
        "ttft_p95_ms":_pct(value("ttft_ms"),95),
        "e2e_p50_ms":_pct(value("e2e_ms"),50),
        "e2e_p95_ms":_pct(value("e2e_ms"),95),
        "tpot_estimate_p50_ms":_pct(value("tpot_ms_estimate"),50),
        "tpot_estimate_p95_ms":_pct(value("tpot_ms_estimate"),95),
        "tpot_is_server_token_timing":False,
        "output_tokens_per_sec":round(sum(
            r["completion_tokens"] for r in good)/elapsed,5) if elapsed>0 else None,
        "completed_requests_per_sec":round(len(good)/elapsed,5) if elapsed>0 else None,
        "gpu":hardware,"server":server,
        "peak_vram_bytes":None,"physical_kv_bytes_saved":None,
        "official_memlens_accuracy":None,"official_mmlongbench_accuracy":None,
        "synthetic_workload_only":True,
        "completed_utc":_utc(),
    }
    if not result["input_length_verified_on_server"] or not result["output_length_verified_on_server"]:
        result["status"]="INCOMPLETE_LENGTH_VERIFICATION"
    return result


def run_cell(args):
    config=read_config(args.protocol)
    target=cells(config,args.model,args.arm,args.context,args.ccu)
    if len(target)!=1:
        raise ValueError("Select exactly one model/arm/context/CCU")
    if not args.execute:
        raise ValueError("Explicit --execute required for real GPU workloads")
    if not RUN_ID.fullmatch(args.run_id):
        raise ValueError("Invalid unique attempt ID")
    cell=target[0]
    # Fail-fast exclusive managed-GPU lock, held by the child runner across
    # prompt preparation, warmup and measured loads. Prevents two agent jobs
    # from contaminating one server throughput measurement.
    # One lock across alternate output directories so agents cannot bypass
    # per-GPU exclusivity by pointing two jobs at different result roots.
    lock_path=Path("/tmp/colab-cli-server-single-gpu-throughput.lock")
    lock_handle=lock_path.open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError("Another GPU throughput cell is already active") from exc
    server=attest_server(args.server_receipt,args.model,args.arm,config,args.url)
    hardware=gpu_info()
    if not args.timeout or args.timeout<1 or args.timeout>21600:
        raise ValueError("Per-request timeout outside supported bounds")
    # One cell = one new seed; this seed and every child request seed persist.
    seed=args.seed if args.seed is not None else secrets.randbits(63)
    if type(seed) is not int or not 0<=seed<(1<<63):
        raise ValueError("Invalid 63-bit immutable master seed")
    dest=Path(args.output_root).resolve()/args.run_id
    dest.mkdir(parents=True,exist_ok=False)
    protocol_sha=_sha(args.protocol)
    manifest={
        "schema_version":1,"status":"STARTED_NOT_QUALIFIED",
        "cell":cell,"attempt_id":args.run_id,
        "protocol_sha256":protocol_sha,"server":server,
        "hardware":hardware,"random_seed":seed,
        "started_utc":_utc(),
        "source":"synthetic_load_only_not_official_accuracy",
    }
    _write_once(dest/"attempt_manifest.json",manifest)
    # Loading frozen model tokenizer stays OUTSIDE timed request throughput.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(server["model_path"],
                         local_files_only=True,trust_remote_code=False)
    prompts=prepare_prompts(tokenizer,args.context,
                             cell["warmups"]+cell["requests"],seed)
    _write_once(dest/"prompt_inventory.json",{
        "total":len(prompts),
        "verified_prompt_tokens":args.context,
        "unique_prompt_hashes":[p["prompt_sha256"] for p in prompts],
        "warmup_request_seeds":[p["seed"] for p in prompts[:cell["warmups"]]],
        "measured_request_seeds":[p["seed"] for p in prompts[cell["warmups"]:]],
        "no_original_prompts_persisted":True,
    })
    url="http://127.0.0.1:{}/v1/chat/completions".format(server["port"])
    pre=prompts[:cell["warmups"]]
    measured=prompts[cell["warmups"]:]
    warmup_observations=_phase(pre,"warmup",args.ccu,server["model_id"],url,
                               args.timeout,dest)
    if any(r["status"]!="SUCCESS" for r in warmup_observations):
        _write_once(dest/"summary.json",{
            "status":"BLOCKED_WARMUP_FAILED","attempt_id":args.run_id,
            "warmup_attempts":len(pre),"warmup_failure_count":sum(
                r["status"]!="SUCCESS" for r in warmup_observations),
            "measured_count":0,"no_fake_metrics":True})
        raise RuntimeError("At least one unique warmup failed; measured run withheld")
    start=time.perf_counter()
    observations=_phase(measured,"measured",args.ccu,server["model_id"],url,
                        args.timeout,dest)
    elapsed=time.perf_counter()-start
    outcome=summarize(cell,observations,elapsed,warmup_observations,seed,
                      hardware,server,protocol_sha)
    outcome["attempt_id"]=args.run_id
    _write_once(dest/"summary.json",outcome)
    print(json.dumps({"output":str(dest),"summary":outcome},indent=2))
    if outcome["status"]!="ENGINEERING_COMPLETE":
        raise SystemExit(4)
    # The lock remains held until Python closes lock_handle on process exit.


def main():
    p=argparse.ArgumentParser(description="One isolated, authentic Qwen GPU throughput cell")
    p.add_argument("--protocol",default=str(__import__("research_ops.throughput.plan",fromlist=["DEFAULT_CONFIG"]).DEFAULT_CONFIG))
    p.add_argument("--model",choices=MODELS,required=True)
    p.add_argument("--arm",choices=ARMS,required=True)
    p.add_argument("--context",type=int,required=True)
    p.add_argument("--ccu",type=int,required=True)
    p.add_argument("--server-receipt",required=True)
    p.add_argument("--url",default="http://127.0.0.1:8000")
    p.add_argument("--output-root",default="/content/qwen35-throughput-runs")
    p.add_argument("--run-id",required=True)
    p.add_argument("--timeout",type=int,default=1200)
    p.add_argument("--seed",type=int)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    run_cell(args)


if __name__=="__main__":
    main()
