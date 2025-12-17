#!/usr/bin/env python
"""
GPU-accelerated slicer.

Clean implementation using GpuEmulatorProcessor with GPU executor.
No monkey-patching - executor selected via .env:

    AUTOGEOSTEERING_EXECUTOR=gpu    # GPU multi-population DE (default)
    AUTOGEOSTEERING_EXECUTOR=python # Python scipy DE
    AUTOGEOSTEERING_EXECUTOR=cpu    # C++ daemon

GPU configuration:
    GPU_N_POPULATIONS=10    # Number of parallel populations
    GPU_POPSIZE_EACH=500    # Individuals per population
    GPU_MAXITER=500         # Iterations

Usage:
    python slicer_gpu.py --de --starsteer-dir <path>

    Or set in .env and run without args:
    python slicer_gpu.py
"""
import os
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))
sys.path.insert(0, str(Path(__file__).parent))

# Set default executor to GPU if not specified
if 'AUTOGEOSTEERING_EXECUTOR' not in os.environ:
    os.environ['AUTOGEOSTEERING_EXECUTOR'] = 'gpu'

# Import AFTER setting env var
from slicer import StarSteerSlicerOrchestrator, main as slicer_main, _CLI_CONFIG
from emulator_processor_gpu import GpuEmulatorProcessor

# Patch the emulator to use GpuEmulatorProcessor
import emulator as emulator_module

# Store original class
_OriginalEmulatorProcessor = emulator_module.EmulatorProcessor if hasattr(emulator_module, 'EmulatorProcessor') else None


def patch_emulator():
    """Patch emulator to use GpuEmulatorProcessor."""
    # The DrillingEmulator creates EmulatorProcessor internally
    # We need to patch the import in emulator module
    from emulator_processor_gpu import GpuEmulatorProcessor
    import emulator_processor
    emulator_processor.EmulatorProcessor = GpuEmulatorProcessor

    # Also patch in emulator module if it imports EmulatorProcessor
    if hasattr(emulator_module, 'EmulatorProcessor'):
        emulator_module.EmulatorProcessor = GpuEmulatorProcessor


# Apply patch
patch_emulator()

executor_type = os.environ.get('AUTOGEOSTEERING_EXECUTOR', 'gpu')
print("=" * 60)
print(f"GPU SLICER: AUTOGEOSTEERING_EXECUTOR={executor_type}")
print("Clean implementation - no monkey-patching of optimizer_fit")
print("=" * 60)


if __name__ == "__main__":
    slicer_main()
