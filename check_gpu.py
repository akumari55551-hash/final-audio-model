"""Verify the active PyTorch installation and run a small CUDA computation.

Run from the activated project virtual environment with: python check_gpu.py
A non-zero exit status means CUDA could not be verified; CPU fallback is not
reported as a successful GPU test.
"""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except ImportError as exc:
        print(f"FAIL: PyTorch is not installed in this interpreter: {exc}")
        print(f"Interpreter: {sys.executable}")
        return 1

    print(f"Python: {sys.version.split()[0]}")
    print(f"Interpreter: {sys.executable}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda or 'CPU-only build'}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("FAIL: CUDA is unavailable. GPU verification did not pass.")
        print("Check the CUDA-enabled PyTorch wheel and NVIDIA driver; see README.md.")
        return 1

    try:
        device = torch.device("cuda:0")
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"GPU capability: {torch.cuda.get_device_capability(device)}")

        left = torch.randn((512, 512), device=device)
        right = torch.randn((512, 512), device=device)
        result = left @ right
        torch.cuda.synchronize(device)

        if result.device.type != "cuda" or not torch.isfinite(result).all().item():
            print("FAIL: CUDA tensor allocation or computation returned an invalid result.")
            return 1

        print(f"Tensor allocation: PASS ({left.device}, {tuple(left.shape)})")
        print(f"GPU matrix multiplication: PASS (result shape {tuple(result.shape)})")
        print("CUDA verification: PASS")
        return 0
    except Exception as exc:  # Report runtime/driver failures with a nonzero exit code.
        print(f"FAIL: CUDA allocation or computation raised {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
