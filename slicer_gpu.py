#!/usr/bin/env python
"""
GPU-accelerated slicer.

Sets AUTOGEOSTEERING_EXECUTOR=gpu and runs standard slicer.
No monkey-patching - executor selected in WellProcessor._create_executor().

    AUTOGEOSTEERING_EXECUTOR=gpu    # GPU multi-population DE (default here)
    AUTOGEOSTEERING_EXECUTOR=python # Python scipy DE
    AUTOGEOSTEERING_EXECUTOR=cpu    # C++ daemon
    AUTOGEOSTEERING_EXECUTOR=auto   # Auto-select (GPU if CUDA available)

GPU configuration via .env:
    GPU_N_POPULATIONS=10    # Number of parallel populations
    GPU_POPSIZE_EACH=500    # Individuals per population
    GPU_MAXITER=500         # Iterations

Usage:
    python slicer_gpu.py --de --starsteer-dir <path>
"""
import os
import sys
from pathlib import Path

# Add cpu_baseline to path
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

# Set default executor to GPU if not specified
if 'AUTOGEOSTEERING_EXECUTOR' not in os.environ:
    os.environ['AUTOGEOSTEERING_EXECUTOR'] = 'gpu'

# Import and run slicer
from slicer import main as slicer_main

executor_type = os.environ.get('AUTOGEOSTEERING_EXECUTOR', 'gpu')
print("=" * 60)
print(f"GPU SLICER: AUTOGEOSTEERING_EXECUTOR={executor_type}")
print("=" * 60)


if __name__ == "__main__":
    slicer_main()
