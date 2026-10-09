"""Read-only diagnostic for the isolated RecallFace 2.0 workspace.

This script checks Python packages, GPU availability, and expected baseline assets.
It does not download models, modify files, or run inference.
"""
from __future__ import annotations

import importlib.util
import platform
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_MODULES = (
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "numpy",
    "PIL",
    "streamlit",
)

EXPECTED_ASSETS = (
    "pretrained/diffae/checkpoints/last.ckpt",
    "pretrained/diffae/checkpoints/latent.pkl",
    "data/starter_set/starter_xT.pt",
    "checkpoints/PHASE7_FINAL/projector_best.pt",
    "checkpoints/PHASE7_FINAL/sampler_best.pt",
)


def main() -> int:
    print("RecallFace 2.0 — Environment Check")
    print("=" * 48)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Project root: {PROJECT_ROOT}")

    print("\nPython modules:")
    missing = []
    for module in REQUIRED_MODULES:
        available = importlib.util.find_spec(module) is not None
        print(f"  [{'OK' if available else 'MISSING'}] {module}")
        if not available:
            missing.append(module)

    print("\nPyTorch/CUDA:")
    try:
        import torch

        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA build: {torch.version.cuda}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gib = props.total_memory / (1024 ** 3)
                print(f"  GPU {index}: {props.name} ({gib:.2f} GiB)")
        else:
            print("  GPU inference will not be available through CUDA in this runtime.")
    except Exception as exc:
        print(f"  Could not inspect PyTorch: {exc}")
        missing.append("usable torch installation")

    print("\nExpected local baseline assets:")
    missing_assets = []
    for relative in EXPECTED_ASSETS:
        exists = (PROJECT_ROOT / relative).is_file()
        print(f"  [{'OK' if exists else 'MISSING'}] {relative}")
        if not exists:
            missing_assets.append(relative)

    print("\nSummary:")
    if missing:
        print("  Missing modules/runtime items:", ", ".join(missing))
    if missing_assets:
        print("  Baseline assets missing:", ", ".join(missing_assets))
    if not missing and not missing_assets:
        print("  Environment and expected baseline assets look ready for a controlled run.")
        return 0

    print("  This diagnostic did not run model inference.")
    print("  Resolve the listed gaps before running the baseline benchmark.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
