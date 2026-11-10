"""Finite detached ONE-CELL GPU jobs under an ALREADY-approved Colab session.

No scheduler, no Colab allocation or keepalive. An agent can invoke launch
via short colab CLI shell exec, poll the job receipt, collect results and
stop the existing session. Detach continuity must be proven on the real VM.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from .plan import DEFAULT_CONFIG,MODELS,ARMS,read_config,cells
from .runner import attest_server,gpu_info,_write_once,RUN_ID
from research_ops.benchmarks.serve import _start_ticks,_owned

DEFAULT_JOB_ROOT="/content/qwen35-throughput-jobs"


def _job_dir(args):
    if not RUN_ID.fullmatch(args.run_id):
        raise ValueError("Invalid bounded job ID")
    return Path(args.job_root).resolve()/args.run_id


def start(args):
    if not args.execute:
        raise RuntimeError("Explicit --execute required for any GPU job")
    config=read_config(args.protocol)
    selected=cells(config,args.model,args.arm,args.context,args.ccu)
    if len(selected)!=1:
        raise ValueError("Only one cell may start per managed GPU job")
    # Fail BEFORE child allocation when H2O / DA cannot honestly run.
    attested=attest_server(args.server_receipt,args.model,args.arm,config,args.url)
    gpu=gpu_info()
    if not RUN_ID.fullmatch(args.run_id):
        raise ValueError("Invalid job ID")
    root=_job_dir(args)
    root.mkdir(parents=True,exist_ok=False)
    argv=[
        sys.executable,"-m","research_ops.throughput.runner",
        "--protocol",str(Path(args.protocol).resolve()),
        "--model",args.model,"--arm",args.arm,
        "--context",str(args.context),"--ccu",str(args.ccu),
        "--server-receipt",str(Path(args.server_receipt).resolve()),
        "--url",args.url,
        "--output-root",str(Path(args.output_root).resolve()),
        "--run-id",args.run_id,"--timeout",str(args.timeout),"--execute",
    ]
    if args.seed is not None:
        argv.extend(["--seed",str(args.seed)])
    log=root/"runner.log"
    with log.open("x",encoding="utf8") as dest:
        p=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=dest,
                           stderr=subprocess.STDOUT,start_new_session=True)
    record={"schema_version":1,"run_id":args.run_id,"pid":p.pid,
        "start_ticks":_start_ticks(p.pid),"argv":argv,"cell":selected[0],
        "server_pid":attested["server_pid"],"gpu":gpu,
        "status":"START_REQUESTED_REMOTE_DETACH_NOT_QUALIFIED",
        "log":str(log),"results":str(Path(args.output_root).resolve()/args.run_id),
        "no_gpu_allocation":True,"human_approved_colab_session_required":True}
    _write_once(root/"job.json",record)
    return record


def status(args):
    root=_job_dir(args)
    job=json.loads((root/"job.json").read_text(encoding="utf8"))
    out=Path(job["results"])/"summary.json"
    if out.is_file():
        score=json.loads(out.read_text(encoding="utf8"))
        return {"run_id":args.run_id,"status":score["status"],
                "summary":str(out),"pid_owned":_owned(job),
                "gpu_qualification":score.get("status")=="ENGINEERING_COMPLETE"}
    running=_owned(job)
    return {"run_id":args.run_id,
            "status":"RUNNING" if running else "STOPPED_WITHOUT_SUMMARY",
            "pid_owned":running,"log":job["log"],
            "continuity_across_colab_executions_proven":False}


def stop(args):
    job=json.loads((_job_dir(args)/"job.json").read_text(encoding="utf8"))
    if not _owned(job):
        return {"status":"NOT_RUNNING_OR_PID_REUSED","stop_requested":False}
    if not args.execute:
        raise RuntimeError("Explicit --execute required to stop owned GPU job")
    pid=job["pid"]
    pgid=os.getpgid(pid)
    if pgid!=pid or pgid==os.getpgrp() or not _owned(job):
        raise RuntimeError("Unsafe process group: refuse to stop another job")
    os.killpg(pgid,signal.SIGTERM)
    return {"status":"SIGTERM_REQUESTED_NOT_CONFIRMED","pid":pid}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("start","status","stop"))
    p.add_argument("--protocol",default=str(DEFAULT_CONFIG))
    p.add_argument("--model",choices=MODELS)
    p.add_argument("--arm",choices=ARMS)
    p.add_argument("--context",type=int)
    p.add_argument("--ccu",type=int)
    p.add_argument("--run-id",required=True)
    p.add_argument("--server-receipt")
    p.add_argument("--url",default="http://127.0.0.1:8000")
    p.add_argument("--output-root",default="/content/qwen35-throughput-runs")
    p.add_argument("--job-root",default=DEFAULT_JOB_ROOT)
    p.add_argument("--timeout",type=int,default=1200)
    p.add_argument("--seed",type=int)
    p.add_argument("--execute",action="store_true")
    args=p.parse_args()
    if args.action=="start" and (None in (args.model,args.arm,args.context,args.ccu) or
                                  not args.server_receipt):
        p.error("start requires explicit model, arm, context, ccu and server receipt")
    result={"start":start,"status":status,"stop":stop}[args.action](args)
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
