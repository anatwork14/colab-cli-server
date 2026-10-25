#!/usr/bin/env python3
"""Tiny deterministic regression run for the live-smoke Colab project.

Runs remotely on Colab (Python 3.10+, PyTorch preinstalled). No downloads and no
pip installs. Synthetic data is generated in code from a fixed seed (1234).

Outputs:
  stdout  "TELEMETRY {json}" lines (same schema as colab_adapter.telemetry
          TelemetryEmitter events), then "DONE_OK ..." on success.
  ledger  when COLAB_TELEMETRY_LEDGER is set, the same events are appended as
          JSON lines (O_APPEND, file lock, mode 0600).
  files   <out-dir>/model.pt (torch.save) and <out-dir>/model.pt.sha256
          (sha256sum format, full 64-hex digest).

Exit codes:
  0  finished, DONE_OK printed
  1  bad configuration or training failure
  2  --require-gpu and CUDA is unavailable
  3  torch is not importable (nothing is trained or written)

torch is imported lazily, so the helper functions can be imported and tested
without PyTorch installed.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import uuid
from datetime import datetime, timezone

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX hosts only
    fcntl = None

SEED = 1234
DEFAULT_STEPS = 60
MAX_STEPS = 200
NUM_SAMPLES = 256
NUM_FEATURES = 8
HIDDEN = 16
BATCH_SIZE = 32
STEPS_PER_EPOCH = NUM_SAMPLES // BATCH_SIZE
LEARNING_RATE = 0.05
PROJECT_ID = "live-smoke"
EVENT_SOURCE = "trainer_callback"  # matches TelemetryEmitter.emit
CHECKPOINT_NAME = "model.pt"
DEFAULT_OUT_DIR = os.path.join("artifacts", "checkpoints")
LEDGER_ENV = "COLAB_TELEMETRY_LEDGER"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REQUIRE_GPU = 2
EXIT_NO_TORCH = 3


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def make_event(event_type, run_id, project_id, epoch=None, step=None, total_steps=None,
               metrics=None, gpu_info=None, details=None):
    """Build one telemetry event with the TelemetryEmitter key set."""
    return {
        "schema_version": 1,
        "event_id": "evt-" + uuid.uuid4().hex[:12],
        "run_id": run_id,
        "project_id": project_id,
        "emitted_at": utc_now_iso(),
        "event_type": event_type,
        "epoch": epoch,
        "step": step,
        "total_steps": total_steps,
        "metrics": metrics or {},
        "gpu_info": gpu_info,
        "source": EVENT_SOURCE,
        "details": details or {},
    }


def append_ledger_line(ledger_path, line):
    """Append one JSON line with O_APPEND, an exclusive flock, and mode 0600."""
    fd = os.open(ledger_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(fd, fcntl.LOCK_EX)
        os.write(fd, (line + "\n").encode("utf-8"))
    finally:
        if fcntl is not None:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


def emit(event, ledger_path=None):
    """Print a TELEMETRY line to stdout and, if configured, append to the ledger."""
    line = json.dumps(event, ensure_ascii=False, sort_keys=True)
    print("TELEMETRY " + line, flush=True)
    if ledger_path:
        append_ledger_line(ledger_path, line)
    return event


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sidecar(sidecar_path):
    """Return (digest, filename) from a sha256sum-format sidecar file."""
    with open(sidecar_path, "r", encoding="utf-8") as handle:
        parts = handle.read().split()
    if len(parts) != 2 or len(parts[0]) != 64:
        raise ValueError("malformed sha256 sidecar: %s" % sidecar_path)
    return parts[0], parts[1]


def make_dataset(torch):
    gen = torch.Generator().manual_seed(SEED)
    x = torch.randn(NUM_SAMPLES, NUM_FEATURES, generator=gen)
    w_true = torch.randn(NUM_FEATURES, 1, generator=gen)
    noise = 0.05 * torch.randn(NUM_SAMPLES, 1, generator=gen)
    y = x @ w_true + noise
    return x, y


def build_model(torch):
    torch.manual_seed(SEED)
    return torch.nn.Sequential(
        torch.nn.Linear(NUM_FEATURES, HIDDEN),
        torch.nn.ReLU(),
        torch.nn.Linear(HIDDEN, 1),
    )


def train(torch, device, steps, run_id, ledger_path, gpu_info):
    x_cpu, y_cpu = make_dataset(torch)
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    model = build_model(torch).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.MSELoss()
    perm_gen = torch.Generator().manual_seed(SEED + 1)
    perm = None
    for step in range(1, steps + 1):
        epoch = (step - 1) // STEPS_PER_EPOCH + 1
        position = (step - 1) % STEPS_PER_EPOCH
        if position == 0:
            perm = torch.randperm(NUM_SAMPLES, generator=perm_gen).to(device)
        idx = perm[position * BATCH_SIZE:(position + 1) * BATCH_SIZE]
        optimizer.zero_grad()
        loss = loss_fn(model(x[idx]), y[idx])
        loss.backward()
        optimizer.step()
        step_loss = float(loss.item())
        if not math.isfinite(step_loss):
            raise FloatingPointError("non-finite loss at step %d" % step)
        emit(make_event("metric", run_id, PROJECT_ID, epoch=epoch, step=step,
                        total_steps=steps, metrics={"loss": step_loss}, gpu_info=gpu_info),
             ledger_path)

    model.eval()
    with torch.no_grad():
        final_loss = float(loss_fn(model(x), y).item())
    return model, final_loss


def save_checkpoint(torch, model, out_dir, steps, final_loss):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, CHECKPOINT_NAME)
    tmp_path = path + ".tmp"
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    torch.save({
        "model_state_dict": state,
        "seed": SEED,
        "step": steps,
        "final_loss": final_loss,
        "num_features": NUM_FEATURES,
        "hidden": HIDDEN,
    }, tmp_path)
    os.replace(tmp_path, path)
    digest = sha256_file(path)
    with open(path + ".sha256", "w", encoding="utf-8") as handle:
        handle.write("%s  %s\n" % (digest, CHECKPOINT_NAME))
    return path, digest


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Tiny deterministic MLP regression run.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="optimizer steps, 1..%d (default %d)" % (MAX_STEPS, DEFAULT_STEPS))
    parser.add_argument("--require-gpu", action="store_true",
                        help="exit 2 when CUDA is unavailable")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="checkpoint directory (default %s)" % DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not 1 <= args.steps <= MAX_STEPS:
        print("TRAIN_CONFIG_ERROR: --steps must be between 1 and %d (got %d)"
              % (MAX_STEPS, args.steps))
        return EXIT_ERROR

    ledger_path = os.environ.get(LEDGER_ENV) or None
    run_id = os.environ.get("COLAB_RUN_ID") or "run-" + uuid.uuid4().hex[:8]

    try:
        import torch
    except ImportError as exc:
        print("TORCH_UNAVAILABLE: %s" % exc)
        if args.require_gpu:
            print("TRAIN_REQUIRE_GPU_FAILED: torch is not installed")
            return EXIT_REQUIRE_GPU
        print("TRAIN_NO_TORCH: cannot train without torch; nothing was written")
        return EXIT_NO_TORCH

    cuda_ok = bool(torch.cuda.is_available())
    print("CUDA available: " + str(cuda_ok))
    if args.require_gpu and not cuda_ok:
        print("TRAIN_REQUIRE_GPU_FAILED: CUDA is not available")
        return EXIT_REQUIRE_GPU

    device = torch.device("cuda:0" if cuda_ok else "cpu")
    gpu_info = {
        "device": str(device),
        "name": torch.cuda.get_device_name(device) if cuda_ok else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }
    print("Device: %s" % device)
    emit(make_event("run_started", run_id, PROJECT_ID, total_steps=args.steps,
                    gpu_info=gpu_info,
                    details={"seed": SEED, "steps": args.steps, "device": str(device)}),
         ledger_path)

    try:
        model, final_loss = train(torch, device, args.steps, run_id, ledger_path, gpu_info)
        ckpt_path, digest = save_checkpoint(torch, model, args.out_dir, args.steps, final_loss)
        emit(make_event("checkpoint_saved", run_id, PROJECT_ID, step=args.steps,
                        total_steps=args.steps, metrics={"loss": final_loss},
                        gpu_info=gpu_info,
                        details={"checkpoint_file": ckpt_path, "sha256": digest}),
             ledger_path)
        emit(make_event("run_finished", run_id, PROJECT_ID, step=args.steps,
                        total_steps=args.steps, metrics={"final_loss": final_loss},
                        gpu_info=gpu_info, details={"status": "COMPLETED"}),
             ledger_path)
    except Exception as exc:
        emit(make_event("error", run_id, PROJECT_ID, total_steps=args.steps,
                        gpu_info=gpu_info,
                        details={"error_type": type(exc).__name__, "error_message": str(exc)}),
             ledger_path)
        print("TRAIN_FAILED: %s: %s" % (type(exc).__name__, exc))
        return EXIT_ERROR

    print("FINAL_LOSS %.8f" % final_loss)
    print("DONE_OK steps=%d device=%s final_loss=%.8f checkpoint=%s sha256=%s"
          % (args.steps, device, final_loss, ckpt_path, digest))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
