"""Plain subprocess driver for the two explicitly gated step 3 passes.

Run with python -m dattn.campaign.driver. Preparation and progress are CPU-only.
Generation reuses dattn eval and its row locks, append_records and torn-tail recovery.
A committed row is never regenerated. A row in flight at process death is retried.
Disjoint shards share the cell's responses.jsonl under a POSIX flock, not a Python
thread lock. Complete lines are fsynced. See eval.persistence for crash semantics.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import secrets
import select
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, suppress
from pathlib import Path
from queue import Empty, Queue

from dattn.campaign import compare
from dattn.campaign.population import (
    ARMS,
    DAC_ROOT,
    DEFAULT_ROOT,
    HEADLINE_SUBSETS,
    HOURS,
    MODELS,
    PILOT,
    ROWS_ROOT,
    SLUGS,
    digest,
    identity,
    load_population,
    shards,
)
from dattn.eval.datasets import EvalRow, SelectionSpec
from dattn.eval.persistence import atomic_json, failed, lock, read_records
from dattn.eval.runner import RunPaths, run_rollouts
from dattn.eval.sampling import RolloutConfig
from dattn.prompt.template_policy import resolve_template, template_metadata

# Keep the venv directory, do not resolve the Python symlink to its base install.
DATTN = str(Path(sys.executable).with_name("dattn"))
VLLM = str(Path(sys.executable).with_name("vllm"))


def pidfd_open(pid):
    """Use host libc because the managed Python build omits Linux pidfd APIs."""
    function = ctypes.CDLL(None, use_errno=True).pidfd_open
    function.argtypes = [ctypes.c_int, ctypes.c_uint]
    function.restype = ctypes.c_int
    result = function(pid, 0)
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def pidfd_signal(fd, sig):
    function = ctypes.CDLL(None, use_errno=True).pidfd_send_signal
    function.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint]
    function.restype = ctypes.c_int
    if function(fd, sig, None, 0) < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def check_pidfds():
    """Refuse an unsupported host before spawning any engine."""
    fd = pidfd_open(os.getpid())
    try:
        pidfd_signal(fd, 0)
    finally:
        os.close(fd)


def output_root(path):
    path = path.resolve()
    repo = Path(__file__).resolve().parents[3]
    if path == repo or repo in path.parents:
        raise ValueError("campaign outputs must be outside the code tree, including runs/")
    return path


def config(model, arm):
    return RolloutConfig(
        model_id=MODELS[model],
        arm=arm,
        prompt_format="vanilla" if arm == "vanilla" else "chunk_multi_turn_tool",
        dattn_enabled=arm in ("da", "da_eqseg"),
        vanilla_system="" if arm == "vanilla" else None,
        segmenter="equal_length" if arm == "da_eqseg" else "semantic",
    ).resolved()


def tokenizer(model):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, local_files_only=True)


def provenance(model, arm, tok):
    from dattn.prompt import segment

    cfg = config(model, arm)
    return {
        "rollout": cfg.to_dict(),
        "rendering": template_metadata(resolve_template(tok, cfg.model_id, cfg.template_policy), cfg.template_policy),
        "segmenter_code_sha256": hashlib.sha256(Path(segment.__file__).read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest(),
    }


def ensure_json(path, expected):
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError(f"resume metadata mismatch: {path}")
    else:
        atomic_json(path, expected)


def prepare(args, arms):
    """Validate every cell before any engine boot, including empty resumed cells."""
    jobs = []
    for model in MODELS:
        if args.command == "smoke" and args.model != model:
            continue
        rows = load_population(model, args.rows_root, args.dac_root)
        if args.command == "smoke":
            rows = [r for r in rows if r["subset"] == PILOT[0]][:10]
        tok = tokenizer(MODELS[model])
        source = args.root / "population" / f"{model}{'-smoke' if args.command == 'smoke' else ''}.json"
        ensure_json(source, rows)
        for arm in arms:
            cell = args.root / args.repeat / SLUGS[model] / arm
            cell.mkdir(parents=True, exist_ok=True)
            manifest = {
                "schema": "step3.cell.v1",
                "repeat": args.repeat,
                "model": model,
                "arm": arm,
                "population": str(source),
                "selection_sha256": digest(rows),
                "row_keys": [list(identity(r)) for r in rows],
                **provenance(model, arm, tok),
            }
            if (cell / "responses.jsonl").exists() and not (cell / "campaign.json").exists():
                raise ValueError(f"{cell}: responses without campaign metadata")
            ensure_json(cell / "campaign.json", manifest)
            check_cell(cell, manifest)
            parts = [("smoke", rows, 10)] if args.command == "smoke" else shards(rows, HOURS[model][arm])
            for name, part, minutes in parts:
                shard = cell / "shards" / name
                shard.mkdir(parents=True, exist_ok=True)
                # Shard membership can change safely between restarts. Row claims
                # and cell-level provenance, not shard IDs, define done work.
                atomic_json(shard / "keys.json", [list(identity(r)) for r in part])
                jobs.append(
                    {
                        "cell": str(cell),
                        "shard": str(shard),
                        "name": name,
                        "model": model,
                        "arm": arm,
                        "minutes": minutes,
                    }
                )
    return jobs


def check_cell(cell, manifest):
    if (cell / "meta.json").exists():
        meta = json.loads((cell / "meta.json").read_text())
        if meta.get("rollout") != manifest["rollout"] or meta.get("rendering") != manifest["rendering"]:
            raise ValueError(f"{cell}: rollout policy differs from campaign metadata")
    records = read_records(cell / "responses.jsonl")
    allowed = {tuple(k) for k in manifest["row_keys"]}
    if any(identity(r) not in allowed for r in records):
        raise ValueError(f"{cell}: row selection changed")


def update_progress(cell, shard=None, state="running", **extra):
    with lock(cell / ".progress.lock"):
        responses = read_records(cell / "responses.jsonl")
        done = {identity(r) for r in responses if not failed(r)}
        manifest = json.loads((cell / "campaign.json").read_text())
        judgments = read_records(cell / "judgments.jsonl")
        judged = {identity(r) for r in judgments if not failed(r)}
        eligible = {identity(r) for r in responses if not failed(r) and r.get("format_correct")}
        atomic_json(
            cell / "progress.json",
            {
                "schema": "step3.progress.v1",
                "repeat": manifest["repeat"],
                "completed": len(done),
                "selected": len(manifest["row_keys"]),
                "transport_pending": sum(failed(r) for r in responses),
                "judged": len(judged),
                "judge_pending": len(eligible - judged),
                **extra,
            },
        )
        if shard:
            keys = {tuple(k) for k in json.loads((shard / "keys.json").read_text())}
            atomic_json(
                shard / "progress.json",
                {
                    "schema": "step3.progress.v1",
                    "state": state,
                    "completed": len(done & keys),
                    "selected": len(keys),
                    **extra,
                },
            )


def eval_shard(args):
    """Existing dattn eval's explicit campaign-row path, never a new sampler."""
    from dattn.cli import _server

    cell, shard = args.campaign_cell, args.campaign_shard
    manifest = json.loads((cell / "campaign.json").read_text())
    model, arm = manifest["model"], manifest["arm"]
    is_samples = manifest["repeat"] == "samples"
    if is_samples:
        from dattn.campaign.samples import safe_root, sample_config, sample_provenance

        safe_root(cell.parents[2])
        if cell.parents[1].name != "samples":
            raise ValueError("sample cell must be under samples")
    if args.model != MODELS[model] or args.arm != arm:
        raise ValueError("CLI model/arm differs from campaign cell")
    if (
        args.template_policy != manifest["rollout"]["template_policy"]
        or args.max_tokens != 8192
        or args.seed != 42
        or args.n_per_subset != 128
        or args.last_answer
        or args.no_replay_metrics
        or args.subsets
    ):
        raise ValueError("CLI policy/selection differs from pinned campaign")
    if not shard.resolve().is_relative_to((cell / "shards").resolve()):
        raise ValueError("shard must belong to the requested campaign cell")
    rows = json.loads(Path(manifest["population"]).read_text())
    if digest(rows) != manifest["selection_sha256"]:
        raise ValueError("campaign row selection digest changed")
    tok = tokenizer(MODELS[model])
    current = sample_provenance(model, arm, tok) if is_samples else provenance(model, arm, tok)
    if any(manifest[k] != v for k, v in current.items()):
        raise ValueError("campaign template, tokenizer or segmenter changed")
    check_cell(cell, manifest)
    keys = {tuple(k) for k in json.loads((shard / "keys.json").read_text())}
    if is_samples:
        if any(k[2] not in range(1, 5) for k in keys):
            raise ValueError("samples indices must be 1 through 4")
        keys = {(k[0], k[1], 0) for k in keys}
    if not keys <= {identity(r) for r in rows}:
        raise ValueError("shard contains foreign rows")
    selected = [EvalRow.from_dict(r) for r in rows if identity(r) in keys]
    # Stable task ordering after selection, never another draw.
    from dattn.campaign.population import pilot_order

    selected = [EvalRow.from_dict(r) for r in pilot_order([r.to_dict() for r in selected])]
    spec = SelectionSpec(
        model_id=MODELS[model],
        subsets=HEADLINE_SUBSETS,
        n_per_subset=128,
        seed=42,
        min_context_tokens=4096,
        max_context_tokens=249856,
    )

    def progress(message):
        print(message, flush=True)
        update_progress(cell, shard)
        if "failed in transport" in message:
            raise RuntimeError("rollout transport incident, stop and notify the operator")

    run_rollouts(
        selected,
        tokenizer=tok,
        config=sample_config(model, arm) if is_samples else config(model, arm),
        server=_server(args),
        paths=RunPaths(cell),
        spec=spec,
        chunk_size=args.chunk_size,
        progress=progress,
        meta_extra={"notes": [f"step3 {manifest['repeat']}", f"selection {manifest['selection_sha256']}"]},
        **({"response_index_start": 1} if is_samples else {}),
    )
    update_progress(cell, shard, "complete")
    if pending({"cell": str(cell), "shard": str(shard)}):
        raise RuntimeError("rollout transport failure, stop and notify the operator")
    return 0


def pending(job):
    keys = {tuple(k) for k in json.loads((Path(job["shard"]) / "keys.json").read_text())}
    done = {identity(r) for r in read_records(Path(job["cell"]) / "responses.jsonl") if not failed(r)}
    return keys - done


def gpu_processes():
    listing = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"], text=True
    )
    indices = {line.split(",")[0].strip(): line.split(",")[1].strip() for line in listing.splitlines() if line.strip()}
    listing = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True
    )
    procs = {uuid: set() for uuid in indices.values()}
    for line in listing.splitlines():
        if line.strip():
            uuid, pid = (x.strip() for x in line.split(","))
            procs.setdefault(uuid, set()).add(int(pid))
    return indices, procs


def assert_free(gpu, owned=None):
    indices, procs = gpu_processes()
    uuid = indices.get(gpu, gpu)
    if uuid not in procs:
        raise ValueError(f"unknown requested GPU {gpu}")
    # An exited child's PID may now name a foreign process. Only live pidfds
    # establish ownership, checked after the GPU process snapshot.
    live = {pid for pid, fd in (owned or {}).items() if not select.select([fd], [], [], 0)[0]}
    foreign = procs[uuid] - live
    if foreign:
        raise RuntimeError(f"GPU {gpu} REFUSED, foreign PIDs {sorted(foreign)}")


def wait_all(gpus):
    while True:
        indices, procs = gpu_processes()
        uuids = [indices.get(g, g) for g in gpus]
        if len(set(uuids)) != len(gpus) or any(u not in procs for u in uuids):
            raise ValueError("GPU list contains aliases, duplicates or unknown devices")
        busy = {g: sorted(procs[u]) for g, u in zip(gpus, uuids, strict=True) if procs[u]}
        if not busy:
            return
        print(f"WAIT all requested GPUs must be free: {busy}", flush=True)
        time.sleep(15)


class Owned:
    """Track only this Popen tree via /proc ancestry and pidfds, never process names.

    pidfds prevent signaling a reused PID. Descendants must be observed while their
    ancestry is live. Unknown orphan processes are refused at reuse, never killed.
    """

    def __init__(self, process):
        self.process = process
        self.fds = {process.pid: pidfd_open(process.pid)}
        self.scan()

    @staticmethod
    def _parent(path):
        return int(path.read_text().rsplit(")", 1)[1].split()[1])

    def scan(self):
        parents = {}
        for path in Path("/proc").glob("[0-9]*/stat"):
            try:
                parents[int(path.parent.name)] = self._parent(path)
            except (OSError, ValueError, IndexError):
                continue
        changed = True
        while changed:
            changed = False
            for pid, parent in parents.items():
                if parent in self.fds and pid not in self.fds and not select.select([self.fds[parent]], [], [], 0)[0]:
                    try:
                        fd = pidfd_open(pid)
                    except ProcessLookupError:
                        continue
                    accepted = False
                    try:
                        # Revalidate after opening. Between the snapshot and open,
                        # the PID could have been reused by an unrelated process.
                        if (
                            self._parent(Path(f"/proc/{pid}/stat")) == parent
                            and not select.select([fd, self.fds[parent]], [], [], 0)[0]
                        ):
                            self.fds[pid] = fd
                            accepted = True
                            changed = True
                    except (OSError, ValueError, IndexError):
                        pass
                    finally:
                        if not accepted:
                            os.close(fd)

    def close(self):
        self.scan()
        for sig in (signal.SIGTERM, signal.SIGKILL):
            for fd in reversed(list(self.fds.values())):
                with suppress(ProcessLookupError):
                    pidfd_signal(fd, sig)
            if sig == signal.SIGTERM:
                with suppress(subprocess.TimeoutExpired):
                    self.process.wait(timeout=10)
        self.process.wait()
        for fd in self.fds.values():
            os.close(fd)


def cpu_env():
    return {
        **{k: v for k, v in os.environ.items() if not k.startswith("DATTN_")},
        "CUDA_VISIBLE_DEVICES": "",
        "VLLM_PLUGINS": "",
        "PYTHONUNBUFFERED": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }


@contextmanager
def server(args, gpu, port, model, log, stop, *, judge=False):
    assert_free(gpu)
    env = cpu_env()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    # vLLM 0.29 seeds the worker RNG from ModelConfig.seed (default zero).
    # Fresh entropy at every boot keeps unseeded requests fresh across repeats.
    engine_seed = secrets.randbits(32)
    if judge:
        # dac's _build_judge_pool(enable_dac=False) uses the stock tokenizer
        # template and no reasoning parser. Pin that cached template explicitly.
        tok = tokenizer("Qwen/Qwen3.5-4B")
        template = tok.get_chat_template()
        template_path = args.root / "judge_template.jinja"
        with lock(args.root / ".judge-template.lock"):
            if template_path.exists() and template_path.read_text() != template:
                raise ValueError("judge template changed")
            template_path.write_text(template)
            ensure_json(
                args.root / "judge_template.json",
                {
                    "model": "Qwen/Qwen3.5-4B",
                    "sha256": hashlib.sha256(template.encode()).hexdigest(),
                    "reasoning_parser": None,
                    "tokenizer_sha256": hashlib.sha256(tok.backend_tokenizer.to_str().encode()).hexdigest(),
                },
            )
        command = [
            VLLM,
            "serve",
            "Qwen/Qwen3.5-4B",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--tensor-parallel-size",
            "1",
            "--max-model-len",
            "65536",
            "--gpu-memory-utilization",
            "0.85",
            "--max-num-seqs",
            "32",
            "--chat-template",
            str(template_path),
            "--seed",
            str(engine_seed),
        ]
    else:
        env["VLLM_PLUGINS"] = "dattn"
        command = [
            DATTN,
            "serve",
            "--model",
            MODELS[model],
            "--profile",
            "parity_prefix" if getattr(args, "repeat", "") == "samples" else "parity",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--exec",
            "--extra",
            "--seed",
            str(engine_seed),
        ]
    # Refuse an occupied port, rather than accepting somebody else's health response.
    import socket

    with socket.socket() as sock:
        # Allow the previous engine's TIME_WAIT sockets, but still reject a listener.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    assert_free(gpu)
    log.parent.mkdir(parents=True, exist_ok=True)
    launch = {"command": command, "gpu": gpu, "port": port, "engine_seed": engine_seed}
    atomic_json(log.with_suffix(".launch.json"), launch)
    with log.open("ab", buffering=0) as stream:
        stream.write((json.dumps(launch) + "\n").encode())
        os.fsync(stream.fileno())
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        owner = Owned(process)
        try:
            deadline = time.monotonic() + 1200
            while True:
                owner.scan()
                if stop.is_set() or process.poll() is not None:
                    raise RuntimeError(f"engine boot interrupted, see {log}")
                if time.monotonic() > deadline:
                    raise RuntimeError(f"engine readiness timeout, see {log}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                stop.wait(2)
            yield owner
        finally:
            owner.close()
            os.fsync(stream.fileno())


def client(command, log, owner, stop, gpu):
    owner.scan()
    assert_free(gpu, owner.fds)
    with log.open("ab", buffering=0) as stream:
        process = subprocess.Popen(command, env=cpu_env(), stdout=stream, stderr=subprocess.STDOUT)
        child = Owned(process)
        try:
            while process.poll() is None:
                owner.scan()
                child.scan()
                if stop.wait(1) or owner.process.poll() is not None:
                    raise RuntimeError("campaign incident or engine exit")
            if process.returncode:
                raise RuntimeError(f"command failed ({process.returncode}), see {log}")
        finally:
            child.close()
            os.fsync(stream.fileno())


class WorkerIncident(RuntimeError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(detail["error"])


class CampaignIncident(RuntimeError):
    def __init__(self, failures):
        self.failures = failures
        super().__init__(" | ".join(f"GPU {f['gpu']} {f['cell']} {f['shard']}: {f['error']}" for f in failures))


def worker_incident(exc, *, cell, shard, gpu, log, offset):
    """Bound log reads and retain this attempt's errors even after noisy teardown."""
    detail = {
        "cell": str(cell) if cell is not None else None,
        "shard": str(shard) if shard is not None else None,
        "gpu": gpu,
        "error": f"{type(exc).__name__}: {exc}",
        "engine_log": str(log),
        "engine_log_tail": [],
        "engine_errors": [],
    }
    try:
        with log.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(offset, stream.tell() - 262144))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        detail["engine_log_tail"] = lines[-40:]
        errors = [i for i, line in enumerate(lines) if any(s in line.lower() for s in ("error", "traceback", "oom"))]
        # Keep the root OOM even when later API tracebacks fill the error tail.
        oom = [i for i in errors if "outofmemory" in lines[i].lower() or "out of memory" in lines[i].lower()]
        detail["engine_errors"] = [lines[i] for i in sorted(set(errors[-80:] + oom[-8:]))]
    except OSError as log_error:
        detail["log_read_error"] = str(log_error)
    return WorkerIncident(detail)


def collect_workers(futures):
    failures = []
    for future in as_completed(futures):
        try:
            future.result()
        except WorkerIncident as exc:
            failures.append(exc.detail)
    if failures:
        raise CampaignIncident(sorted(failures, key=lambda f: (f["gpu"], f["cell"] or "")))


def generate(args, jobs, stop):
    queue = Queue()
    for job in sorted(jobs, key=lambda j: (j["name"] != "pilot", -j["minutes"], j["cell"], j["name"])):
        if pending(job):
            queue.put(job)
        else:
            update_progress(Path(job["cell"]), Path(job["shard"]), "complete")

    def worker(slot, gpu):
        while not stop.is_set():
            try:
                job = queue.get_nowait()
            except Empty:
                return
            cell, shard = Path(job["cell"]), Path(job["shard"])
            log = shard / "engine.log"
            offset = 0
            try:
                offset = log.stat().st_size if log.exists() else 0
                update_progress(cell, shard, "booting", gpu=gpu)
                with server(args, gpu, args.port_base + slot, job["model"], log, stop) as owner:
                    update_progress(cell, shard, "running", gpu=gpu, engine_pid=owner.process.pid)
                    client(
                        [
                            DATTN,
                            "eval",
                            "--model",
                            MODELS[job["model"]],
                            "--arm",
                            job["arm"],
                            "--campaign-cell",
                            str(cell),
                            "--campaign-shard",
                            str(shard),
                            "--base-url",
                            f"http://127.0.0.1:{args.port_base + slot}",
                        ],
                        shard / "eval.log",
                        owner,
                        stop,
                        gpu,
                    )
                if pending(job):
                    raise RuntimeError("shard exited with pending rows")
                update_progress(cell, shard, "complete")
            except BaseException as exc:
                stop.set()
                failure = worker_incident(exc, cell=cell, shard=shard, gpu=gpu, log=log, offset=offset)
                try:
                    update_progress(cell, shard, "incident", gpu=gpu)
                except Exception as progress_error:
                    failure.detail["progress_error"] = str(progress_error)
                raise failure from exc

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, slot, gpu) for slot, gpu in enumerate(args.gpus)]
        collect_workers(futures)


def judge(args, cells, stop):
    queue = Queue()
    for cell in cells:
        queue.put(cell)

    def worker(slot, gpu):
        cell = None
        log = args.root / args.repeat / f"judge-{slot}.log"
        offset = 0
        try:
            offset = log.stat().st_size if log.exists() else 0
            with server(args, gpu, args.port_base + slot, "", log, stop, judge=True) as owner:
                while not stop.is_set():
                    try:
                        cell = queue.get_nowait()
                    except Empty:
                        return
                    client(
                        [
                            DATTN,
                            "judge",
                            "--dir",
                            str(cell),
                            "--base-url",
                            f"http://127.0.0.1:{args.port_base + slot}",
                        ],
                        cell / "judge.log",
                        owner,
                        stop,
                        gpu,
                    )
                    responses = read_records(cell / "responses.jsonl")
                    tasks = tuple(t for t in HEADLINE_SUBSETS if any(r["subset"] == t for r in responses))
                    if not tasks:
                        raise RuntimeError("judge cell has no responses")
                    compare.summarize(responses, read_records(cell / "judgments.jsonl"), tasks, candidate=True)
                    client(
                        [
                            DATTN,
                            "analyse",
                            "--dirs",
                            str(cell),
                            "--json-out",
                            str(cell / "analysis.json"),
                        ],
                        cell / "analyse.log",
                        owner,
                        stop,
                        gpu,
                    )
                    update_progress(cell)
        except BaseException as exc:
            stop.set()
            raise worker_incident(exc, cell=cell, shard=None, gpu=gpu, log=log, offset=offset) from exc

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures = [pool.submit(worker, i, gpu) for i, gpu in enumerate(args.gpus)]
        collect_workers(futures)


def comparison(args, stage, *, ablation=False):
    base = args.root / args.repeat
    command = [
        "--root",
        str(args.root),
        "--dac-root",
        str(args.dac_root),
        "--repeats",
        args.repeat,
        "--stage",
        stage,
        "--verdict",
        str(base / f"{stage}.verdict.json"),
        "--markdown",
        str(base / f"{stage}.md"),
    ]
    if ablation:
        command += ["--ablation"]
        # Keep pass1 gate immutable when adding the final ablation report.
        command[command.index("--verdict") + 1] = str(base / "final.verdict.json")
        command[command.index("--markdown") + 1] = str(base / "final.md")
    if stage == "pilot" and not args.pilot_task_stop:
        command += ["--no-per-task-stop"]
    result = compare.main(command)
    if result and not (result == 1 and ablation and args.override_gate_b):
        raise RuntimeError(f"Gate {'A' if stage == 'pilot' else 'B'} STOP, see {base / (stage + '.md')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("pass1", "pass2", "judge", "progress", "prepare", "smoke", "samples"))
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--repeat", choices=("rep0", "rep1", "rep2", "samples"), default="rep0")
    p.add_argument("--gpus", nargs="+")
    p.add_argument("--rows-root", type=Path, default=ROWS_ROOT)
    p.add_argument("--dac-root", type=Path, default=DAC_ROOT)
    p.add_argument("--port-base", type=int, default=18400)
    p.add_argument("--pass", dest="pass_number", choices=(1, 2), type=int, default=1)
    p.add_argument("--override-gate-b", action="store_true", help="only after an explicit operator override of Gate B")
    p.add_argument("--pilot-task-stop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--model", choices=tuple(MODELS), default="gemma")
    p.add_argument("--n", type=int, default=4)
    args = p.parse_args(argv)
    args.root = output_root(args.root)
    if args.command == "samples" or args.repeat == "samples":
        if args.repeat != "samples" or args.command not in ("samples", "judge") or args.pass_number != 1:
            p.error("samples requires --repeat samples and command samples or judge --pass 1")
        if args.n != 4 or not args.gpus or len(args.gpus) != 4 or len(set(args.gpus)) != 4:
            p.error("samples requires --n 4 and four distinct GPUs")
        from dattn.campaign.samples import run

        return run(args)
    if args.command == "smoke" and args.root == DEFAULT_ROOT:
        p.error("smoke requires a separate --root, for example ../results/step3-smoke")
    if args.command == "progress":
        for manifest in sorted(args.root.glob("rep*/*/*/campaign.json")):
            cell = manifest.parent
            update_progress(cell)
            print(str(cell.relative_to(args.root)), (cell / "progress.json").read_text().strip())
            for shard in sorted(cell.glob("shards/*/progress.json")):
                print(str(shard.relative_to(args.root)), shard.read_text().strip())
        return 0
    pass2 = args.command == "pass2" or args.pass_number == 2
    if pass2 and args.repeat != "rep0":
        p.error("ablation repeats are not authorized")
    arms = ["da_eqseg"] if pass2 else list(ARMS)
    if args.command == "smoke":
        arms = ["da"]
    if args.command not in ("prepare",) and (
        not args.gpus
        or len(args.gpus) != (1 if args.command == "smoke" else 4)
        or len(set(args.gpus)) != len(args.gpus)
    ):
        p.error("provide exactly four distinct GPUs (one for smoke)")
    args.root.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def interrupt(signum, frame):
        del signum, frame
        stop.set()
        raise KeyboardInterrupt

    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        with lock(args.root / ".driver.lock", blocking=False):
            if not args.pilot_task_stop:
                atomic_json(
                    args.root / args.repeat / "pilot_task_stop_override.json",
                    {"explicit_override": True, "pilot_task_stop": False},
                )
            if pass2:
                compare.require_pass(
                    args.root / args.repeat / "full.verdict.json",
                    repeat=args.repeat,
                    root=args.root,
                    override=args.override_gate_b,
                    dac_root=args.dac_root,
                )
                if args.override_gate_b:
                    atomic_json(args.root / args.repeat / "gate_b_override.json", {"explicit_override": True})
                else:
                    recorded_pilot = json.loads((args.root / args.repeat / "pilot.verdict.json").read_text())
                    current_pilot, _ = compare.report(
                        args.root,
                        [args.repeat],
                        args.dac_root,
                        PILOT,
                        stage="pilot",
                        macro_pp=8,
                        task_pp=15,
                        task_se=3,
                        per_task_stop=args.pilot_task_stop,
                    )
                    if recorded_pilot != current_pilot or recorded_pilot.get("status") != "pass":
                        raise ValueError("Gate A REFUSED: no matching recorded pilot pass")
            jobs = prepare(args, arms)
            if args.command == "prepare":
                print(json.dumps(jobs, indent=2))
                return 0
            cells = sorted({Path(job["cell"]) for job in jobs})
            check_pidfds()
            wait_all(args.gpus)
            if args.command == "judge":
                judge(args, cells, stop)
                return 0
            if args.command == "pass1":
                generate(args, [j for j in jobs if j["name"] == "pilot"], stop)
                judge(args, cells, stop)
                comparison(args, "pilot")
                generate(args, [j for j in jobs if j["name"] != "pilot"], stop)
            elif args.command == "pass2":
                # Dispatch pilot shards first, then fill all four devices from the queue.
                generate(args, jobs, stop)
            else:
                generate(args, jobs, stop)
                rows = read_records(cells[0] / "responses.jsonl")
                if len(rows) != 10 or any(failed(r) or not r.get("format_correct") for r in rows):
                    raise RuntimeError("bad smoke: requires 10 successful, format-correct rows")
                print(f"SMOKE PASS {args.model}: 10/10 responses, 10/10 format-correct, 0 transport errors")
                return 0
            judge(args, cells, stop)
            comparison(args, "full", ablation=pass2)
            atomic_json(
                args.root / args.repeat / ("pass2.complete.json" if pass2 else "pass1.complete.json"),
                {
                    "schema": "step3.complete.v1",
                    "status": "complete",
                    "repeat": args.repeat,
                    "override": args.override_gate_b,
                },
            )
            print(f"{args.command} {args.repeat} COMPLETE", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        stop.set()
        incident = {"command": args.command, "error": str(exc)}
        if isinstance(exc, CampaignIncident):
            incident["failures"] = exc.failures
        atomic_json(args.root / args.repeat / "incident.json", incident)
        print(f"STOP and notify the operator: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
