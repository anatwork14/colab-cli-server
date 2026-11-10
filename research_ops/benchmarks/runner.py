"""Resumable, real VLM benchmark client for one already-approved local GPU VM.

Does not provision Colab, install models, train, generate dataset rows, or
produce pseudo-official metrics. Requests are sent only to loopback.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from research_ops.assets import hash_file, verify as verify_asset
from research_ops.engine import verify as verify_engine
from .datasets import audit, identity, is_unanswerable, evidence_pages, load_records, safe_relative
from .messages import build

SLUG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{2,79}$")


def utc():
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(file, item):
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=str(file.parent),
                                     prefix=".receipt-", suffix=".tmp",
                                     encoding="utf-8", delete=False) as h:
        temp = h.name
        json.dump(item, h, ensure_ascii=False, sort_keys=True, indent=2)
        h.write("\n")
        h.flush()
        os.fsync(h.fileno())
    os.replace(temp,file)


def _endpoint(base):
    u = urlsplit(base)
    if u.scheme != "http" or u.hostname not in ("localhost","127.0.0.1","::1"):
        raise ValueError("Benchmark requests may only target a loopback vLLM server")
    if u.username or u.password or u.query or u.fragment or u.path.rstrip("/") not in ("","/v1"):
        raise ValueError("Use http://127.0.0.1:8000 or http://127.0.0.1:8000/v1")
    return u.scheme + "://" + u.netloc


def _get_json(url):
    with urlopen(url, timeout=15) as response:
        return json.load(response)


def server_check(url, model):
    prefix = _endpoint(url)
    info = _get_json(prefix + "/version")
    if info.get("version") != "0.29.0":
        raise RuntimeError("Remote server is not pinned to vLLM 0.29.0")
    models = _get_json(prefix + "/v1/models").get("data",[])
    if not isinstance(models,list) or model not in [m.get("id") for m in models]:
        raise RuntimeError("Requested served-model name absent from local vLLM /v1/models")
    return {"vllm_version":info["version"],"served_model":model}


def _parse_text(delta):
    chunk = delta.get("content")
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, list):
        return "".join(x.get("text","") for x in chunk if isinstance(x,dict))
    return ""


def streaming_completion(base, model, messages, *, max_tokens, timeout_s):
    payload = json.dumps({
        "model":model,"messages":messages,"max_tokens":max_tokens,
        "temperature":0,"stream":True,"stream_options":{"include_usage":True}
    },ensure_ascii=False).encode("utf-8")
    endpoint = _endpoint(base) + "/v1/chat/completions"
    start = time.perf_counter()
    req = Request(endpoint, data=payload, method="POST",
                  headers={"Content-Type":"application/json",
                           "Accept":"text/event-stream"})
    pieces, times = [], []
    usage, finish_reason = None, None
    with urlopen(req,timeout=timeout_s) as response:
        for line in response:
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if event.get("error"):
                raise RuntimeError("vLLM streamed an error")
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices",[]):
                part = _parse_text(choice.get("delta") or {})
                if part:
                    if not times:
                        times.append(time.perf_counter())
                    else:
                        times.append(time.perf_counter())
                    pieces.append(part)
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
    end = time.perf_counter()
    if not pieces:
        raise RuntimeError("No generated text in model stream")
    result = {"prediction":"".join(pieces),
              "ttft_ms":(times[0]-start)*1000,
              "e2e_ms":(end-start)*1000,
              "chunk_count":len(times),
              "prompt_tokens":usage.get("prompt_tokens") if isinstance(usage,dict) else None,
              "completion_tokens":usage.get("completion_tokens") if isinstance(usage,dict) else None,
              "finish_reason":finish_reason,
              "request_payload_bytes":len(payload),
              "ITL_ms":"NOT_MEASURED_NON_TOKEN_ALIGNED_SSE"}
    tokens = result["completion_tokens"]
    result["tpot_ms_estimate"] = (((end-times[0])*1000)/(tokens-1)
                                   if type(tokens) is int and tokens>1 else None)
    return result


def _pct(values, percent):
    if not values:
        return None
    v=sorted(values)
    index=(len(v)-1)*percent/100
    lo,hi=int(math.floor(index)),int(math.ceil(index))
    return v[lo]*(hi-index)+v[hi]*(index-lo) if hi!=lo else v[lo]


def _norm(s):
    return re.sub(r"\s+"," ",str(s).casefold().strip().strip(".!?\"' "))


def summarize(kind, rows, receipts, elapsed_s, *, subset):
    n=len(rows)
    done=[x for x in receipts if x and x.get("status")=="SUCCESS"]
    success_count=len(done)
    fail=n-success_count
    times=lambda key:[r[key] for r in done if type(r.get(key)) in (float,int)]
    groups={}
    for i,row in enumerate(rows):
        group = (str(row.get("question_type","UNLABELLED")) if kind=="memlens"
                 else ("unanswerable" if is_unanswerable(row) else
                       ("cross_page" if len(set(evidence_pages(row)))>1 else "single_page")))
        entry=groups.setdefault(group,{"n":0,"completed":0,"proxy_exact_match":0})
        entry["n"]+=1
        rec=receipts[i]
        if rec and rec.get("status")=="SUCCESS":
            entry["completed"]+=1
            entry["proxy_exact_match"]+=int(_norm(rec["prediction"])==_norm(row["answer"]))
    p50 = lambda key:_pct(times(key),50)
    p95 = lambda key:_pct(times(key),95)
    proxy=sum(x["proxy_exact_match"] for x in groups.values())/n if n else None
    wrong_unanswerable = 0
    n_unanswerable = 0
    if kind == "mmlongbench_doc":
        for i,row in enumerate(rows):
            if is_unanswerable(row):
                n_unanswerable += 1
                rec=receipts[i]
                if rec and rec.get("status")=="SUCCESS" and not is_unanswerable(
                    {"answer":rec["prediction"]}):
                    wrong_unanswerable+=1
    return {
        "status":"ENGINEERING_SUBSET_NOT_OFFICIAL" if subset else "ENGINEERING_MEASURED_OFFICIAL_SCORER_PENDING",
        "dataset":kind,"denominator":n,"success":success_count,"failures":fail,
        "failure_rate":fail/n if n else None,
        "proxy_exact_match_including_failures":proxy,
        "proxy_groups":{k:dict(v,proxy_exact_match_accuracy=v["proxy_exact_match"]/v["n"])
                        for k,v in groups.items()},
        "unanswerable_false_assertion_rate_proxy":(
            wrong_unanswerable/n_unanswerable if n_unanswerable else None),
        "official_accuracy":None,"official_f1":None,
        "official_hallucination_rate":None,
        "official_judge": "NOT_RUN_OR_NOT_QUALIFIED",
        "ttft_p50_ms":p50("ttft_ms"),"ttft_p95_ms":p95("ttft_ms"),
        "e2e_p50_ms":p50("e2e_ms"),"e2e_p95_ms":p95("e2e_ms"),
        "tpot_p50_estimate_ms":p50("tpot_ms_estimate"),
        "tpot_p95_estimate_ms":p95("tpot_ms_estimate"),
        "tpot_quality":"CLIENT_SSE_ESTIMATE_FROM_COMPLETION_USAGE_NOT_SERVER_TOKEN_TIMING",
        "itl":"NOT_MEASURED","physical_kv_hbm":"NOT_MEASURED",
        "throughput_completed_requests_per_second":success_count/elapsed_s if elapsed_s>0 else None,
        "throughput_all_attempts_per_second":n/elapsed_s if elapsed_s>0 else None,
        "wall_seconds":elapsed_s,"measurement_includes_client_preprocessing":True,
        "no_training":True,"no_synthetic_benchmark_data":True
    }


def _slug_receipt(run_root,sample_id):
    return Path(run_root)/"receipts"/(hashlib.sha256(sample_id.encode("utf8")).hexdigest()+".json")


def _hardware():
    p=subprocess.run(["nvidia-smi","--query-gpu=name,memory.total,driver_version",
                     "--format=csv,noheader"],capture_output=True,text=True,
                     check=True,timeout=20)
    if not p.stdout.strip():
        raise RuntimeError("No NVIDIA GPU available")
    return p.stdout.strip().splitlines()


def run(args):
    if not SLUG.fullmatch(args.run_id):
        raise ValueError("Invalid run ID")
    if (args.concurrency<1 or args.concurrency>32 or args.limit<0 or
            args.max_tokens<1 or args.timeout<1 or args.warmup<0 or args.warmup>10):
        raise ValueError("Invalid concurrency, limit or generation budget")
    if args.kind=="mmlongbench_doc" and not args.deep_pdf:
        raise ValueError("MMLongBench-Doc GPU execution requires --deep-pdf preflight")
    if args.engine_mode!="stock" and not (args.source_root and args.source_revision):
        raise ValueError("Modified vLLM needs exact source path and commit")
    scratch=Path(args.scratch).resolve()
    model_receipt=verify_asset(args.assets,args.model_asset,scratch)
    dataset_receipt=verify_asset(args.assets,args.dataset_asset,scratch)
    dataset_root=Path(dataset_receipt["staged_path"])
    samples=dataset_root / safe_relative(args.samples_rel)
    qualification=audit(args.kind,samples,dataset_root,
                        expected_count=args.expected_count,deep_pdf=args.deep_pdf)
    if qualification["status"]!="COMPLETE_FOR_ENGINEERING_PILOT":
        raise ValueError("Dataset integrity/denominator BLOCKED; see offline audit")
    engine_receipt=verify_engine(args.engine_mode,args.source_root,
                                 args.source_revision,require_gpu=True)
    hardware=_hardware()
    # A model id can be an alias for arbitrary weights. Tie this server process
    # to the exact staged asset, committed engine and local TCP port.
    from .serve import _owned
    srv=json.loads(Path(args.server_receipt).read_text(encoding="utf8"))
    if not _owned(srv):
        raise RuntimeError("No live process matching the pinned server process receipt")
    if srv.get("model_asset_sha256") != model_receipt["tree_sha256"]:
        raise RuntimeError("Served weight source differs from frozen model")
    if srv.get("engine") != engine_receipt:
        raise RuntimeError("Served engine does not match frozen runtime provenance")
    if srv.get("serving_model") != args.model:
        raise RuntimeError("Served-model alias differs from frozen model name")
    command=srv.get("argv_without_tokens", [])
    expected_port=urlsplit(args.url).port or 80
    try:
        command_port=int(command[command.index("--port")+1])
    except (ValueError,IndexError,TypeError):
        raise RuntimeError("Server receipt has no recorded localhost port")
    if command_port != expected_port:
        raise RuntimeError("Server receipt port does not match requested vLLM endpoint")
    server=server_check(args.url,args.model)
    if args.limit and args.limit>=args.expected_count:
        raise ValueError("--limit is a pilot subset knob, not full population")
    rows=load_records(args.kind,samples)
    selected=rows[:args.limit] if args.limit else rows
    ids=[identity(args.kind,row,i) for i,row in enumerate(selected)]
    out=Path(args.output_root).resolve()/args.run_id
    config={
        "schema_version":1,"run_id":args.run_id,"kind":args.kind,
        "expected_count":args.expected_count,"question_file_sha256":hash_file(samples),
        "model_asset_tree_sha256":model_receipt["tree_sha256"],
        "dataset_asset_tree_sha256":dataset_receipt["tree_sha256"],
        "asset_manifest_sha256":hash_file(args.assets),
        "server":server,"engine":engine_receipt,
        "gpu":hardware,"limit":args.limit,
        "max_tokens":args.max_tokens,"max_images":args.max_images,
        "max_pages":args.max_pages,"dpi":args.dpi,
        "concurrency":args.concurrency,"temperature":0,
        "warmup_requests":args.warmup,
        "server_pid_start_ticks":srv["start_ticks"],
        "subset":bool(args.limit),"run_id_header":args.run_id,
    }
    meta=out/"run_config.json"
    if out.exists():
        if not args.resume or not meta.exists():
            raise FileExistsError("Run exists. Use --resume with identical frozen configuration")
        if json.loads(meta.read_text(encoding="utf8"))!=config:
            raise ValueError("Cannot resume: source, engine, model, sample IDs or settings changed")
    else:
        out.mkdir(parents=True)
        _atomic_json(meta,config)
    indexed = list(enumerate(selected))
    pending=[]
    receipts=[None]*len(indexed)
    for i,row in indexed:
        receipt_path=_slug_receipt(out,ids[i])
        if receipt_path.exists():
            old=json.loads(receipt_path.read_text(encoding="utf8"))
            if old.get("sample_id")!=ids[i]:
                raise ValueError("Receipt SHA collision or corrupted sample ID")
            receipts[i]=old
        else:
            pending.append((i,row))

    def eval_one(i,row):
        name=ids[i]
        try:
            messages,prep=build(args.kind,row,dataset_root,
                                max_images=args.max_images,max_pages=args.max_pages,
                                dpi=args.dpi)
            response=streaming_completion(args.url,args.model,messages,
                                         max_tokens=args.max_tokens,timeout_s=args.timeout)
            result={"sample_id":name,"original_index":i,"status":"SUCCESS",
                    "received_utc":utc(),**prep,**response}
        except Exception as exc:
            # Failed rows are retained in the declared denominator, never dropped.
            result={"sample_id":name,"original_index":i,"status":"FAILED",
                    "received_utc":utc(),"error_type":type(exc).__name__,
                    "error_reason":str(exc)[:220],
                    "prediction":""}
        _atomic_json(_slug_receipt(out,name),result)
        return i,result

    # Warmup runs use ORIGINAL dataset questions. Exclude warmup output from
    # quality/latency and explicitly record count before measured requests.
    if pending and args.warmup:
        for k in range(args.warmup):
            warm_messages,_=build(args.kind,selected[k%len(selected)],dataset_root,
                max_images=args.max_images,max_pages=args.max_pages,dpi=args.dpi)
            streaming_completion(args.url,args.model,warm_messages,
                                 max_tokens=args.max_tokens,timeout_s=args.timeout)
        _atomic_json(out/"warmup_status.json",{
            "status":"REAL_DATA_WARMUP_COMPLETED","count":args.warmup,
            "excluded_from_score":True,"original_question_ids":[
                ids[k%len(selected)] for k in range(args.warmup)]})
    started=time.perf_counter()
    if pending:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for future in as_completed([pool.submit(eval_one,i,row) for i,row in pending]):
                i,result=future.result()
                receipts[i]=result

    duration=max(time.perf_counter()-started,1e-9)
    summary=summarize(args.kind,selected,receipts,duration,subset=bool(args.limit))
    # Resume may reuse previous receipts measured hours earlier; the current
    # wall clock cannot truthfully describe throughput of all those requests.
    if len(pending)!=len(indexed):
        summary["throughput_completed_requests_per_second"]=None
        summary["throughput_all_attempts_per_second"]=None
        summary["wall_seconds"]=None
        summary["throughput_status"]="NOT_MEASURED_ON_RESUME_MIXED_TIME_WINDOWS"
    else:
        summary["throughput_status"]="MEASURED_ON_THIS_CLIENT_RUN"
    if summary["failures"]:
        summary["status"]="FAILED_REQUESTS_OFFICIAL_SCORER_PENDING"
    summary["run_id"]=args.run_id
    summary["question_file_sha256"]=hash_file(samples)
    summary["model_asset_tree_sha256"]=model_receipt["tree_sha256"]
    summary["dataset_asset_tree_sha256"]=dataset_receipt["tree_sha256"]
    summary["dataset_audit_sha256"]=hashlib.sha256(
        json.dumps(qualification,sort_keys=True).encode()).hexdigest()
    summary["resume_used"]=args.resume
    summary["warmup_requests"]=args.warmup
    summary["existing_receipts_before_run"]=len(indexed)-len(pending)
    summary["completed_utc"]=utc()
    _atomic_json(out/"engineering_summary.json",summary)
    print(json.dumps({"results":str(out),"summary":summary},indent=2))
    if summary["failures"]:
        raise SystemExit(4)


def main():
    p=argparse.ArgumentParser(description="Authorized GPU-only benchmark client: NO Colab allocation.")
    p.add_argument("--kind",choices=("memlens","mmlongbench_doc"),required=True)
    p.add_argument("--assets",required=True)
    p.add_argument("--model-asset",required=True)
    p.add_argument("--dataset-asset",required=True)
    p.add_argument("--scratch",default="/content/research-assets")
    p.add_argument("--samples-rel",required=True)
    p.add_argument("--expected-count",type=int,required=True)
    p.add_argument("--model",required=True,help="Exact served model id")
    p.add_argument("--server-receipt",required=True,help="Verified live server_process.json path")
    p.add_argument("--url",default="http://127.0.0.1:8000")
    p.add_argument("--engine-mode",choices=("stock","source_python","source_compiled"),default="stock")
    p.add_argument("--source-root")
    p.add_argument("--source-revision")
    p.add_argument("--output-root",default="/content/research-benchmark-runs")
    p.add_argument("--run-id",required=True)
    p.add_argument("--max-tokens",type=int,default=256)
    p.add_argument("--max-images",type=int,default=160)
    p.add_argument("--max-pages",type=int,default=128)
    p.add_argument("--dpi",type=int,default=72)
    p.add_argument("--concurrency",type=int,default=1)
    p.add_argument("--warmup",type=int,default=1,help="Real-question warmups, excluded from score")
    p.add_argument("--limit",type=int,default=0,help="Real-row engineering subset; never official")
    p.add_argument("--timeout",type=int,default=180)
    p.add_argument("--deep-pdf",action="store_true")
    p.add_argument("--resume",action="store_true")
    p.add_argument("--execute",action="store_true")
    a=p.parse_args()
    if not a.execute:
        p.error("No allocation from this command. Add --execute ONLY inside approved GPU runtime")
    run(a)

if __name__=="__main__":
    main()
