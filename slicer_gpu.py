#!/usr/bin/env python
"""
GPU-accelerated slicer wrapper.

Monkey-patches optimizer_fit to use GPU version, then runs original slicer.
Usage: python slicer_gpu.py --de --starsteer-dir <path>
"""
import sys
import os
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))
sys.path.insert(0, str(Path(__file__).parent))

# Import GPU optimizer_fit BEFORE importing anything from cpu_baseline
from gpu_optimizer_fit import gpu_optimizer_fit

# Now import the module we want to patch
import ag_numerical.ag_func_optimizer as ag_optimizer_module

# Monkey-patch: replace CPU optimizer_fit with GPU version
_original_optimizer_fit = ag_optimizer_module.optimizer_fit

def patched_optimizer_fit(*args, **kwargs):
    """GPU-accelerated optimizer_fit wrapper."""
    # Add GPU-specific defaults
    kwargs.setdefault('device', 'cuda')
    kwargs.setdefault('verbose', os.getenv('GPU_VERBOSE', 'false').lower() == 'true')
    return gpu_optimizer_fit(*args, **kwargs)

ag_optimizer_module.optimizer_fit = patched_optimizer_fit

# CRITICAL: Also patch the executor module which imports optimizer_fit directly
# The "from ... import optimizer_fit" creates a separate binding that we must also patch
import optimizers.python_autogeosteering_executor as executor_module
executor_module.optimizer_fit = patched_optimizer_fit

print("=" * 60)
print("GPU SLICER: optimizer_fit patched to use GPU acceleration")
print("Patched modules: ag_numerical.ag_func_optimizer, optimizers.python_autogeosteering_executor")
print("=" * 60)

# Now run the original slicer
if __name__ == "__main__":
    # Import slicer module (will use patched optimizer_fit)
    from slicer import main
    main()
