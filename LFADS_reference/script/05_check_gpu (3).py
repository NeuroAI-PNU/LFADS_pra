"""Read-only GPU / Python / PyTorch diagnostics for a Slurm cluster."""
import os
import shutil
import socket
import subprocess
import sys


def main():
    print("Host:", socket.gethostname())
    print("Python:", sys.version.split(" ")[0])
    print("Executable:", sys.executable)
    print("SLURM_JOB_ID:", os.environ.get("SLURM_JOB_ID", "UNSET"))
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "UNSET"))
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            result = subprocess.run(
                [smi, "--query-gpu=name,driver_version,memory.total", "--format=csv"],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            print("nvidia-smi return code:", result.returncode)
            print(result.stdout.strip() or result.stderr.strip())
        except (OSError, subprocess.TimeoutExpired) as exc:
            print("nvidia-smi failed:", exc)
    else:
        print("nvidia-smi: unavailable")

    try:
        import torch
    except ImportError:
        print("PyTorch: not installed in this Python environment")
        return
    print("PyTorch:", torch.__version__)
    print("PyTorch built CUDA:", torch.version.cuda)
    try:
        available = torch.cuda.is_available()
        print("GPU available:", available)
        if available:
            print("GPU name:", torch.cuda.get_device_name(0))
            x = torch.tensor([[1., 2.], [3., 4.]], device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            print("Actual CUDA matrix test: PASS", tuple(y.shape))
        else:
            print("Cannot use GPU from this Python process; check Slurm allocation and torch build.")
    except Exception as exc:
        print("CUDA runtime test failed:", type(exc).__name__, str(exc))
        raise


if __name__ == "__main__":
    main()
