"""Small PyTorch CUDA smoke test for an approved Colab GPU runtime."""
import platform
import sys

print("Python:", sys.version.split()[0])
print("Platform:", platform.platform())
try:
    import torch
except ImportError:
    print("PyTorch unavailable in this runtime.")
    raise SystemExit(1)

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("GPU unavailable despite request")

device = torch.device("cuda:0")
print("GPU:", torch.cuda.get_device_name(device))
print("VRAM GiB:", round(torch.cuda.get_device_properties(device).total_memory / 2**30, 2))
tensor = torch.ones((256, 256), device=device)
assert (tensor @ tensor).shape == (256, 256)
torch.cuda.synchronize()
print("GPU smoke test passed")
