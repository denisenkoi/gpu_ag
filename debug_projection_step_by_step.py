#!/usr/bin/env python3
"""
Step-by-step comparison: visualizer (numpy) vs gpu_executor (torch)
Prints intermediate values to find where they diverge.
"""
import os
import sys
from pathlib import Path

os.environ['NORMALIZATION_MODE'] = 'ORIGINAL'
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

WELL_NAME = "Well162~EGFDL"
METERS_TO_FEET = 3.28084
TARGET_MD_FT = 19800  # Check at this MD

def main():
    print("=== Step-by-step projection comparison ===\n")

    # Load dataset
    dataset_path = Path(__file__).parent / "dataset" / "gpu_ag_dataset.pt"
    dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)
    data = dataset[WELL_NAME]

    # Raw data (same for both)
    well_md = data['well_md'].cpu().numpy()
    well_tvd = data['well_tvd'].cpu().numpy()
    well_ns = data['well_ns'].cpu().numpy()
    well_ew = data['well_ew'].cpu().numpy()
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_shifts = data['ref_shifts'].cpu().numpy()

    # Calculate VS
    well_vs = np.zeros(len(well_ns))
    for i in range(1, len(well_ns)):
        dx = well_ns[i] - well_ns[i-1]
        dy = well_ew[i] - well_ew[i-1]
        well_vs[i] = well_vs[i-1] + np.sqrt(dx*dx + dy*dy)

    # TVD shift
    tvd_typewell_shift = data.get('tvd_typewell_shift', 0.0)
    if isinstance(tvd_typewell_shift, torch.Tensor):
        tvd_typewell_shift = tvd_typewell_shift.item()

    # Find target MD index
    target_md_m = TARGET_MD_FT / METERS_TO_FEET
    target_idx = np.searchsorted(well_md, target_md_m)
    target_idx = min(target_idx, len(well_md) - 1)

    print(f"Target: MD={target_md_m:.2f}m ({TARGET_MD_FT}ft), idx={target_idx}")
    print(f"tvd_typewell_shift = {tvd_typewell_shift:.4f}m")
    print()

    # Find which segment contains this point
    seg_idx = None
    for i in range(len(ref_mds) - 1):
        if ref_mds[i] <= target_md_m <= ref_mds[i+1]:
            seg_idx = i
            break

    if seg_idx is None:
        print(f"ERROR: Target MD not in any segment!")
        return

    # Segment data
    seg_start_md = ref_mds[seg_idx]
    seg_end_md = ref_mds[seg_idx + 1]
    seg_start_shift = ref_shifts[seg_idx - 1] if seg_idx > 0 else 0.0
    seg_end_shift = ref_shifts[seg_idx]

    seg_start_idx = np.searchsorted(well_md, seg_start_md)
    seg_end_idx = np.searchsorted(well_md, seg_end_md)

    print(f"=== Segment {seg_idx} ===")
    print(f"MD: {seg_start_md:.2f} - {seg_end_md:.2f}m")
    print(f"Shift: {seg_start_shift:.4f} - {seg_end_shift:.4f}m")
    print(f"Indices: {seg_start_idx} - {seg_end_idx}")
    print()

    # Values at segment boundaries
    seg_start_vs = well_vs[seg_start_idx]
    seg_end_vs = well_vs[seg_end_idx]

    print(f"=== Segment VS (from well) ===")
    print(f"VS at start_idx: {seg_start_vs:.4f}m")
    print(f"VS at end_idx: {seg_end_vs:.4f}m")
    print(f"delta_VS: {seg_end_vs - seg_start_vs:.4f}m")
    print()

    # Values at target point
    target_vs = well_vs[target_idx]
    target_tvd = well_tvd[target_idx]

    print(f"=== Target point ===")
    print(f"VS: {target_vs:.4f}m")
    print(f"TVD: {target_tvd:.4f}m")
    print()

    # Calculate shift interpolation (NUMPY way - like visualizer)
    delta_vs = seg_end_vs - seg_start_vs
    depth_shift = seg_end_shift - seg_start_shift

    # Fraction through segment
    frac = (target_vs - seg_start_vs) / delta_vs
    shift_at_target = seg_start_shift + depth_shift * frac

    print(f"=== Shift interpolation ===")
    print(f"fraction = (target_vs - start_vs) / delta_vs")
    print(f"fraction = ({target_vs:.4f} - {seg_start_vs:.4f}) / {delta_vs:.4f}")
    print(f"fraction = {frac:.6f}")
    print(f"shift = start_shift + depth_shift * fraction")
    print(f"shift = {seg_start_shift:.4f} + {depth_shift:.4f} * {frac:.6f}")
    print(f"shift = {shift_at_target:.4f}m")
    print()

    # Calculate TVT
    tvt = target_tvd - shift_at_target - tvd_typewell_shift

    print(f"=== TVT calculation ===")
    print(f"TVT = TVD - shift - tvd_typewell_shift")
    print(f"TVT = {target_tvd:.4f} - {shift_at_target:.4f} - {tvd_typewell_shift:.4f}")
    print(f"TVT = {tvt:.4f}m")
    print()

    # Check typewell bounds
    type_tvd = data['type_tvd'].cpu().numpy()
    print(f"=== Typewell bounds ===")
    print(f"Typewell TVD range: {type_tvd.min():.4f} - {type_tvd.max():.4f}m")
    print(f"TVT {tvt:.4f}m in range? {type_tvd.min() <= tvt <= type_tvd.max()}")

    print("\n" + "="*60)
    print("=== Now GPU Executor path ===")
    print("="*60 + "\n")

    from true_gpu_slicer import slice_data_to_md, build_manual_interpretation_to_md
    from gpu_executor import GpuAutoGeosteeringExecutor

    # Slice data like true_gpu_slicer does
    start_md = data.get('detected_start_md') or data.get('start_md', 0.0) or 0.0
    current_md = seg_end_md + 50  # Go past target segment

    data['well_name'] = WELL_NAME
    work_interpretation = build_manual_interpretation_to_md(data, start_md)
    well_data_json = slice_data_to_md(data, current_md)
    well_data_json['interpretation'] = {'segments': work_interpretation}

    # Create executor
    work_dir = Path(__file__).parent / "results" / "debug_step"
    work_dir.mkdir(parents=True, exist_ok=True)
    executor = GpuAutoGeosteeringExecutor(work_dir=work_dir, results_dir=work_dir)
    result = executor.initialize_well(well_data_json)

    # Get normalized well
    well = executor.ag_well
    print(f"well.normalized: {well.normalized}")
    print(f"fixed_md_range: {executor.fixed_md_range:.4f}m")
    print(f"fixed_min_md: {executor.fixed_min_md:.4f}m")
    print(f"tvd_to_typewell_shift (normalized): {executor.tvd_to_typewell_shift:.6f}")
    print(f"tvd_to_typewell_shift (meters): {executor.tvd_to_typewell_shift * executor.fixed_md_range:.4f}m")
    print()

    # Find segment that covers our target
    segments = executor.interpretation
    target_seg = None
    for i, seg in enumerate(segments):
        seg_start_md_m = seg.start_md * executor.fixed_md_range + executor.fixed_min_md
        seg_end_md_m = seg.end_md * executor.fixed_md_range + executor.fixed_min_md
        if seg_start_md_m <= target_md_m <= seg_end_md_m:
            target_seg = seg
            print(f"Found segment {i}: MD {seg_start_md_m:.2f} - {seg_end_md_m:.2f}m")
            break

    if target_seg is None:
        print("ERROR: Target not in any executor segment!")
        return

    print(f"\n=== Segment values (normalized) ===")
    print(f"start_idx: {target_seg.start_idx}")
    print(f"end_idx: {target_seg.end_idx}")
    print(f"start_vs (norm): {target_seg.start_vs:.6f}")
    print(f"end_vs (norm): {target_seg.end_vs:.6f}")
    print(f"start_shift (norm): {target_seg.start_shift:.6f}")
    print(f"end_shift (norm): {target_seg.end_shift:.6f}")

    # Denormalize
    md_range = executor.fixed_md_range
    start_vs_m = target_seg.start_vs * md_range + well.min_vs
    end_vs_m = target_seg.end_vs * md_range + well.min_vs
    start_shift_m = target_seg.start_shift * md_range
    end_shift_m = target_seg.end_shift * md_range

    print(f"\n=== Segment values (denormalized) ===")
    print(f"start_vs: {start_vs_m:.4f}m")
    print(f"end_vs: {end_vs_m:.4f}m")
    print(f"start_shift: {start_shift_m:.4f}m")
    print(f"end_shift: {end_shift_m:.4f}m")

    # What about well.vs at those indices?
    print(f"\n=== Well VS at segment indices ===")
    well_start_vs_norm = well.vs_thl[target_seg.start_idx]
    well_end_vs_norm = well.vs_thl[target_seg.end_idx]
    well_start_vs_m = well_start_vs_norm * md_range + well.min_vs
    well_end_vs_m = well_end_vs_norm * md_range + well.min_vs
    print(f"well.vs[start_idx] (norm): {well_start_vs_norm:.6f}")
    print(f"well.vs[end_idx] (norm): {well_end_vs_norm:.6f}")
    print(f"well.vs[start_idx] (m): {well_start_vs_m:.4f}m")
    print(f"well.vs[end_idx] (m): {well_end_vs_m:.4f}m")

    print(f"\n=== COMPARISON ===")
    print(f"segment.start_vs == well.vs[start_idx]? {abs(target_seg.start_vs - well_start_vs_norm) < 1e-10}")
    print(f"segment.end_vs == well.vs[end_idx]? {abs(target_seg.end_vs - well_end_vs_norm) < 1e-10}")

    # Now check TVT range for optimization segments
    print(f"\n=== TVT range check for optimization ===")
    from numpy_funcs.converters import segments_to_numpy
    from numpy_funcs.projection import calc_horizontal_projection_numpy

    # Get typewell data
    typewell = executor.ag_typewell
    print(f"Typewell TVD (norm) range: [{typewell.tvd.min():.6f}, {typewell.tvd.max():.6f}]")

    # Get well data
    print(f"Well TVD (norm) range: [{well.true_vertical_depth.min():.6f}, {well.true_vertical_depth.max():.6f}]")
    print(f"tvd_to_typewell_shift (norm): {executor.tvd_to_typewell_shift:.6f}")

    # Calculate TVT at start and end of optimization region
    opt_start_idx = segments[-4].start_idx  # First optimization segment
    opt_end_idx = segments[-1].end_idx      # Last optimization segment

    tvd_start = well.true_vertical_depth[opt_start_idx]
    tvd_end = well.true_vertical_depth[opt_end_idx]

    shift_start = segments[-4].start_shift
    shift_end = segments[-1].end_shift

    tvt_start = tvd_start - shift_start - executor.tvd_to_typewell_shift
    tvt_end = tvd_end - shift_end - executor.tvd_to_typewell_shift

    print(f"\nOptimization region TVT:")
    print(f"  Start: TVD={tvd_start:.6f} - shift={shift_start:.6f} - tvd_shift={executor.tvd_to_typewell_shift:.6f} = TVT={tvt_start:.6f}")
    print(f"  End:   TVD={tvd_end:.6f} - shift={shift_end:.6f} - tvd_shift={executor.tvd_to_typewell_shift:.6f} = TVT={tvt_end:.6f}")
    print(f"  Typewell range: [{typewell.tvd.min():.6f}, {typewell.tvd.max():.6f}]")

    in_range_start = typewell.tvd.min() <= tvt_start <= typewell.tvd.max()
    in_range_end = typewell.tvd.min() <= tvt_end <= typewell.tvd.max()
    print(f"  Start in range? {in_range_start}")
    print(f"  End in range? {in_range_end}")

if __name__ == '__main__':
    main()
