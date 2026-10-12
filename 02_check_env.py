
import sys
import subprocess

print("=== Python ===")
print("Python:", sys.version)
print("Executable:", sys.executable)

print("\n=== NVIDIA GPU ===")
try:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv",
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    print("Return code:", result.returncode)
    print(result.stdout.strip())

    if result.stderr.strip():
        print(result.stderr.strip())

except FileNotFoundError:
    print("nvidia-smi was not found.")
except (OSError, subprocess.TimeoutExpired) as exc:
    print("GPU check failed:", exc)

print("\n=== PyTorch ===")
try:
    import torch

    print("PyTorch:", torch.__version__)
    print("PyTorch CUDA version:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

except Exception as exc:
    print("PyTorch check failed:", type(exc).__name__, str(exc))