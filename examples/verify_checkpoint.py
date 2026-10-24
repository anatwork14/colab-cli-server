"""Validates downloaded PyTorch model checkpoint and manifest locally."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn

    class SmallMLP(nn.Module):
        def __init__(self, in_features: int = 4, hidden: int = 16, out_features: int = 1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_features),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
except ImportError:
    torch = None
    SmallMLP = None


def verify_checkpoint(artifacts_dir: Path) -> bool:
    artifacts_dir = Path(artifacts_dir).resolve()
    checkpoint_path = artifacts_dir / "model_checkpoint.pt"
    status_path = artifacts_dir / "status.json"

    print(f"=== Validating Checkpoint in: {artifacts_dir} ===")

    # 1. Existence check
    if not checkpoint_path.is_file():
        print(f"FAILED: Checkpoint file missing: {checkpoint_path}", file=sys.stderr)
        return False
    if not status_path.is_file():
        print(f"FAILED: Status manifest missing: {status_path}", file=sys.stderr)
        return False

    size_bytes = checkpoint_path.stat().st_size
    print(f"Checkpoint size: {size_bytes} bytes")
    if size_bytes == 0:
        print("FAILED: Checkpoint file is empty", file=sys.stderr)
        return False

    # 2. Check manifest and SHA256
    try:
        manifest = json.loads(status_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"FAILED: Invalid status JSON manifest: {e}", file=sys.stderr)
        return False

    expected_hash = manifest.get("sha256")
    hasher = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    actual_hash = hasher.hexdigest()

    if expected_hash != actual_hash:
        print(
            f"FAILED: SHA256 mismatch! Expected {expected_hash}, got {actual_hash}",
            file=sys.stderr,
        )
        return False
    print(f"SHA256 Checksum Verified: {actual_hash}")

    # 3. Checkpoint format inspection (Standard Library zipfile/pickle)
    is_zip = False
    try:
        import zipfile

        if zipfile.is_zipfile(str(checkpoint_path)):
            is_zip = True
            with zipfile.ZipFile(str(checkpoint_path), "r") as zf:
                namelist = zf.namelist()
                print(f"Zip archive entries in .pt file: {namelist[:5]}")
                has_data_or_pkl = any("data" in n or "pkl" in n for n in namelist)
                if not has_data_or_pkl:
                    print(f"FAILED: Expected PyTorch archive format with data/pickle", file=sys.stderr)
                    return False
    except Exception as e:
        print(f"FAILED: Could not inspect archive structure: {e}", file=sys.stderr)
        return False

    # 4. Optional PyTorch load if torch is present in current environment
    try:
        import torch

        data = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = data.get("model_state_dict")
        if not state_dict:
            print("FAILED: model_state_dict missing in checkpoint", file=sys.stderr)
            return False

        model = SmallMLP()
        model.load_state_dict(state_dict)
        model.eval()

        dummy_input = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
        with torch.no_grad():
            out = model(dummy_input)

        print(f"Model successfully loaded via torch.load! Output: {out.numpy().tolist()}")
    except ImportError:
        print("Note: local PyTorch not installed; verified checkpoint via ZIP format and SHA256 manifest.")
    except Exception as e:
        print(f"FAILED: torch.load failed on checkpoint: {e}", file=sys.stderr)
        return False
    print(f"Training device: {manifest.get('device')} ({manifest.get('gpu_name')})")
    print(f"Recorded loss: {manifest.get('final_loss')}")
    print("ALL CHECKPOINT VERIFICATION CHECKS PASSED.")
    return True


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "artifacts"
    success = verify_checkpoint(target)
    sys.exit(0 if success else 1)
