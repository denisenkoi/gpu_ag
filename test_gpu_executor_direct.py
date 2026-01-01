#!/usr/bin/env python3
"""
Test gpu_executor directly - EXACT same path as true_gpu_slicer.py.
NO custom functions - pure orchestration using existing code.
"""

import os
import sys
from pathlib import Path

# Set NORMALIZATION_MODE before imports
os.environ['NORMALIZATION_MODE'] = 'ORIGINAL'

sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

# Use EXACT same functions as true_gpu_slicer.py
from true_gpu_slicer import (
    slice_data_to_md, build_manual_interpretation_to_md, interpolate_shift_at_md
)
from gpu_executor import GpuAutoGeosteeringExecutor

WELL_NAME = "Well162~EGFDL"
METERS_TO_FEET = 3.28084


def main():
    print(f"=== Testing gpu_executor - EXACT same as true_gpu_slicer.py ===")
    print(f"NORMALIZATION_MODE = {os.environ.get('NORMALIZATION_MODE')}")

    # Load dataset (same as true_gpu_slicer.py)
    dataset_path = Path(__file__).parent / "dataset" / "gpu_ag_dataset.pt"
    dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)
    data = dataset[WELL_NAME]
    data['well_name'] = WELL_NAME

    # Get start MD (same as true_gpu_slicer.py run_slicing)
    start_md = data.get('detected_start_md') or data.get('start_md', 0.0)
    if start_md is None:
        start_md = 0.0
    print(f"start_md: {start_md:.1f}m ({start_md*METERS_TO_FEET:.0f}ft)")

    # First slice to ~19600ft (5974m) like telescope
    first_slice_length = 5974 - start_md  # Go to 19600ft
    current_md = start_md + first_slice_length
    print(f"current_md (first slice): {current_md:.1f}m ({current_md*METERS_TO_FEET:.0f}ft)")

    # Build manual interpretation up to start_md (same as true_gpu_slicer.py)
    work_interpretation = build_manual_interpretation_to_md(data, start_md)
    print(f"Manual interpretation: {len(work_interpretation)} segments")

    # Build well_data (same as true_gpu_slicer.py)
    well_data = slice_data_to_md(data, current_md)
    well_data['interpretation'] = {'segments': work_interpretation}

    print(f"\nwell_data keys: {list(well_data.keys())}")
    print(f"tvdTypewellShift: {well_data['tvdTypewellShift']}")
    print(f"typeLog points: {len(well_data['typeLog']['tvdSortedPoints'])}")
    print(f"reference segments: {len(well_data['referenceInterpretation']['segments'])}")

    # Create GpuExecutor (same as true_gpu_slicer.py)
    work_dir = Path(__file__).parent / "results" / "test_direct"
    work_dir.mkdir(parents=True, exist_ok=True)
    executor = GpuAutoGeosteeringExecutor(work_dir=work_dir, results_dir=work_dir)

    # Call initialize_well (same as true_gpu_slicer.py iteration 1)
    print("\n=== Calling executor.initialize_well() ===")
    result = executor.initialize_well(well_data)

    # Look at internal state
    print(f"\n=== Executor internal state ===")
    print(f"tvd_to_typewell_shift (normalized): {executor.tvd_to_typewell_shift:.6f}")
    print(f"fixed_md_range: {executor.fixed_md_range:.1f}")
    print(f"fixed_min_md: {executor.fixed_min_md:.1f}")
    print(f"Reference segments: {len(executor.reference_segments)}")

    # Current interpretation
    print(f"\n=== Current interpretation ===")
    print(f"Segments: {len(executor.interpretation)}")
    if executor.interpretation:
        last_seg = executor.interpretation[-1]
        last_end_m = last_seg.end_md * executor.fixed_md_range + executor.fixed_min_md
        print(f"Last segment end_md: {last_end_m:.0f}m ({last_end_m*METERS_TO_FEET:.0f}ft)")


if __name__ == '__main__':
    main()
