#!/usr/bin/env python3
"""CUDA probe for the live-smoke Colab project (runs remotely on Colab).

Prints environment facts, runs one small matmul on CUDA and prints
GPU_PROBE_OK with an exact integer checksum (verified against a CPU reference).
Inputs are small integers, so the checksum is exact on CPU and GPU.

Exit codes:
  0  probe finished (GPU_PROBE_OK on CUDA, GPU_PROBE_NO_CUDA without --require-gpu)
  1  CUDA matmul failed or disagreed with the CPU reference
  2  --require-gpu and CUDA (or torch) is unavailable
  3  torch is not importable and --require-gpu was not given
"""
import argparse
import platform
import sys

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REQUIRE_GPU = 2
EXIT_NO_TORCH = 3
SEED = 1234
MATRIX_SIZE = 64


def main(argv=None):
    parser = argparse.ArgumentParser(description="Probe CUDA with one tiny deterministic matmul.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="exit 2 when CUDA is unavailable instead of reporting GPU_PROBE_NO_CUDA",
    )
    args = parser.parse_args(argv)

    print("Python: " + sys.version.split()[0])
    print("Platform: " + platform.platform())
    try:
        import torch
    except ImportError as exc:
        print("Torch: unavailable (%s)" % exc)
        print("CUDA available: False")
        if args.require_gpu:
            print("GPU_PROBE_REQUIRE_GPU_FAILED: torch is not installed")
            return EXIT_REQUIRE_GPU
        print("GPU_PROBE_NO_TORCH: torch is not installed; probe cannot run here")
        return EXIT_NO_TORCH

    print("Torch: " + torch.__version__)
    print("CUDA build version: " + str(torch.version.cuda))
    cuda_ok = bool(torch.cuda.is_available())
    print("CUDA available: " + str(cuda_ok))
    if not cuda_ok:
        if args.require_gpu:
            print("GPU_PROBE_REQUIRE_GPU_FAILED: CUDA is not available")
            return EXIT_REQUIRE_GPU
        print("GPU_PROBE_NO_CUDA: CUDA is not available; no GPU work was done")
        return EXIT_OK

    device = torch.device("cuda:0")
    print("GPU count: %d" % torch.cuda.device_count())
    print("GPU name: " + torch.cuda.get_device_name(device))
    try:
        gen = torch.Generator().manual_seed(SEED)
        a = torch.randint(-3, 4, (MATRIX_SIZE, MATRIX_SIZE), generator=gen).float()
        b = torch.randint(-3, 4, (MATRIX_SIZE, MATRIX_SIZE), generator=gen).float()
        product = a.to(device) @ b.to(device)
        torch.cuda.synchronize(device)
        checksum = int(product.double().sum().item())
        reference = int((a.double() @ b.double()).sum().item())
    except Exception as exc:  # fail closed: never print a success marker
        print("GPU_PROBE_FAILED: %s: %s" % (type(exc).__name__, exc))
        return EXIT_ERROR

    if checksum != reference:
        print("GPU_PROBE_MISMATCH: gpu=%d cpu_reference=%d" % (checksum, reference))
        return EXIT_ERROR
    print("GPU_PROBE_OK device=%s checksum=%d" % (device, checksum))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
