"""PyTorch training job executed remotely on Colab GPU backend.

Reads /content/synthetic_data.csv, trains an MLP on GPU,
and writes:
- /content/model_checkpoint.pt (weights & architecture state)
- /content/status.json (metadata, hash, and metrics manifest)
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


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


def run_training():
    print(f"=== Starting Colab PyTorch Training ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    print(f"Using Device: {device} ({gpu_name})")

    # Load synthetic dataset
    dataset_path = Path("/content/synthetic_data.csv")
    if not dataset_path.exists():
        # Fallback to local file if run locally
        dataset_path = Path("./synthetic_data.csv")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Input dataset not found at {dataset_path}")

    print(f"Loading dataset from: {dataset_path}")
    inputs = []
    targets = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row:
                continue
            inputs.append([float(x) for x in row[:4]])
            targets.append([float(row[4])])

    x_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
    print(f"Loaded {len(x_tensor)} samples.")

    # Model and training
    model = SmallMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("Beginning training iterations...")
    model.train()
    start_t = time.time()
    final_loss = 0.0
    for epoch in range(1, 21):
        optimizer.zero_grad()
        preds = model(x_tensor)
        loss = criterion(preds, y_tensor)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/20 - Loss: {final_loss:.4f}")

    duration = time.time() - start_t
    print(f"Training completed in {duration:.2f}s. Final MSE Loss: {final_loss:.4f}")

    # Save checkpoint
    checkpoint_dir = Path("/content") if Path("/content").exists() else Path("./artifacts")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model_checkpoint.pt"

    checkpoint_payload = {
        "model_state_dict": model.state_dict(),
        "input_dim": 4,
        "hidden_dim": 16,
        "output_dim": 1,
        "final_loss": final_loss,
        "device": str(device),
        "gpu_name": gpu_name,
        "timestamp": time.time(),
    }
    torch.save(checkpoint_payload, str(checkpoint_path))
    print(f"Checkpoint saved to: {checkpoint_path}")

    # Compute SHA256 checksum
    hasher = hashlib.sha256()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    checkpoint_sha256 = hasher.hexdigest()

    # Write status JSON manifest
    status_path = checkpoint_dir / "status.json"
    manifest = {
        "status": "SUCCESS",
        "model_file": str(checkpoint_path.name),
        "sha256": checkpoint_sha256,
        "final_loss": final_loss,
        "num_samples": len(x_tensor),
        "device": str(device),
        "gpu_name": gpu_name,
        "duration_seconds": round(duration, 3),
        "pytorch_version": torch.__version__,
    }
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Manifest saved to: {status_path}")
    print(f"STATUS: SUCCESS | SHA256: {checkpoint_sha256}")


if __name__ == "__main__":
    run_training()
