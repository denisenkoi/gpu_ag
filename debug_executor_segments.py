#!/usr/bin/env python3
"""
Check what segments gpu_executor creates and whether VS matches.
"""
import os
import sys
from pathlib import Path

os.environ['NORMALIZATION_MODE'] = 'ORIGINAL'
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from true_gpu_slicer import slice_data_to_md, build_manual_interpretation_to_md
from gpu_executor import GpuAutoGeosteeringExecutor
from numpy_funcs.converters import segments_to_numpy
from torch_funcs.converters import segments_numpy_to_torch
from torch_funcs.projection import calc_horizontal_projection_batch_torch

WELL_NAME = "Well162~EGFDL"
METERS_TO_FEET = 3.28084
TARGET_MD_FT = 19800

def main():
    print("=== GPU Executor segment check ===\n")

    dataset_path = Path(__file__).parent / "dataset" / "gpu_ag_dataset.pt"
    dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)
    data = dataset[WELL_NAME]
    data['well_name'] = WELL_NAME

    target_md_m = TARGET_MD_FT / METERS_TO_FEET

    # Slice to include target
    start_md = data.get('detected_start_md') or data.get('start_md', 0.0) or 0.0
    current_md = target_md_m + 100  # Go past target

    work_interpretation = build_manual_interpretation_to_md(data, start_md)
    well_data_json = slice_data_to_md(data, current_md)
    well_data_json['interpretation'] = {'segments': work_interpretation}

    work_dir = Path(__file__).parent / "results" / "debug_exec_seg"
    work_dir.mkdir(parents=True, exist_ok=True)
    executor = GpuAutoGeosteeringExecutor(work_dir=work_dir, results_dir=work_dir)
    result = executor.initialize_well(well_data_json)

    well = executor.ag_well
    segments = executor.interpretation
    md_range = executor.fixed_md_range
    min_md = executor.fixed_min_md

    print(f"Executor params:")
    print(f"  fixed_md_range: {md_range:.2f}m")
    print(f"  fixed_min_md: {min_md:.2f}m")
    print(f"  well.min_vs: {well.min_vs:.2f}m")
    print(f"  Segments: {len(segments)}")
    print()

    # Find segment containing target
    target_md_norm = (target_md_m - min_md) / md_range
    target_idx = None
    for i in range(len(well.measured_depth)):
        if well.measured_depth[i] >= target_md_norm:
            target_idx = i
            break

    print(f"Target: MD={target_md_m:.2f}m, norm={target_md_norm:.6f}, idx={target_idx}")
    print()

    # Check last 4 segments (optimization region)
    print("=== Last 4 segments (optimization) ===")
    for i, seg in enumerate(segments[-4:]):
        seg_start_md_m = seg.start_md * md_range + min_md
        seg_end_md_m = seg.end_md * md_range + min_md

        # VS from segment
        seg_start_vs = seg.start_vs
        seg_end_vs = seg.end_vs

        # VS from well at same indices
        well_start_vs = well.vs_thl[seg.start_idx]
        well_end_vs = well.vs_thl[seg.end_idx]

        # Check match
        vs_start_match = abs(seg_start_vs - well_start_vs) < 1e-10
        vs_end_match = abs(seg_end_vs - well_end_vs) < 1e-10

        print(f"Seg {len(segments)-4+i}: MD={seg_start_md_m:.1f}-{seg_end_md_m:.1f}m, idx={seg.start_idx}-{seg.end_idx}")
        print(f"  seg.start_vs={seg_start_vs:.6f}, well.vs[start_idx]={well_start_vs:.6f}, match={vs_start_match}")
        print(f"  seg.end_vs={seg_end_vs:.6f}, well.vs[end_idx]={well_end_vs:.6f}, match={vs_end_match}")
        print(f"  shift: {seg.start_shift*md_range:.2f} -> {seg.end_shift*md_range:.2f}m")

        # Is target in this segment?
        if seg.start_idx <= target_idx <= seg.end_idx:
            print(f"  ** TARGET idx={target_idx} IN THIS SEGMENT **")

    # Now compute projection with executor's data and check target
    print("\n=== Projection with executor segments ===")
    from numpy_funcs.converters import well_to_numpy, typewell_to_numpy
    from torch_funcs.converters import numpy_to_torch

    well_np = well_to_numpy(well)
    typewell_np = typewell_to_numpy(executor.ag_typewell)
    segments_np = segments_to_numpy(segments, well)

    well_torch = numpy_to_torch(well_np, device='cpu')
    typewell_torch = numpy_to_torch(typewell_np, device='cpu')
    segments_torch = segments_numpy_to_torch(segments_np, device='cpu').unsqueeze(0)

    success_mask, tvt_batch, synt_batch, first_start_idx = calc_horizontal_projection_batch_torch(
        well_torch, typewell_torch, segments_torch, executor.tvd_to_typewell_shift
    )

    if target_idx is None:
        print("ERROR: target_idx not found")
        return

    rel_idx = target_idx - first_start_idx
    if rel_idx < 0 or rel_idx >= tvt_batch.shape[1]:
        print(f"ERROR: rel_idx {rel_idx} out of range [0, {tvt_batch.shape[1]})")
        return

    tvt_norm = tvt_batch[0, rel_idx].item()
    synt_norm = synt_batch[0, rel_idx].item()

    # Denormalize
    typewell = executor.ag_typewell
    tvt_m = tvt_norm * md_range + typewell.wells_min_depth
    synt_denorm = synt_norm * well.max_curve_value

    print(f"first_start_idx: {first_start_idx}, target_idx: {target_idx}, rel_idx: {rel_idx}")
    print(f"TVT (norm): {tvt_norm:.6f}")
    print(f"TVT (m): {tvt_m:.4f}m")
    print(f"synt (norm): {synt_norm:.6f}")
    print(f"synt (denorm): {synt_denorm:.4f}")
    print()

    # Compare with expected
    print("=== COMPARISON ===")
    expected_tvt = 3824.3138
    expected_synt = 73.4961
    print(f"Expected TVT: {expected_tvt:.4f}m, Got: {tvt_m:.4f}m, Diff: {expected_tvt - tvt_m:.4f}m")
    print(f"Expected synt: {expected_synt:.4f}, Got: {synt_denorm:.4f}, Diff: {expected_synt - synt_denorm:.4f}")

if __name__ == '__main__':
    main()
