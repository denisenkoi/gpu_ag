#!/usr/bin/env python3
"""
Compare EXACT synt_curve value at target point:
- Visualizer (numpy)
- GPU Executor (torch)
"""
import os
import sys
from pathlib import Path

os.environ['NORMALIZATION_MODE'] = 'ORIGINAL'
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from numpy_funcs.projection import calc_horizontal_projection_numpy

WELL_NAME = "Well162~EGFDL"
METERS_TO_FEET = 3.28084
TARGET_MD_FT = 19800

def calc_vs(ns, ew):
    vs = np.zeros(len(ns))
    for i in range(1, len(ns)):
        dx = ns[i] - ns[i-1]
        dy = ew[i] - ew[i-1]
        vs[i] = vs[i-1] + np.sqrt(dx*dx + dy*dy)
    return vs

def build_segments(ref_mds, ref_shifts, well_md, well_vs):
    segments = []
    for i in range(len(ref_mds) - 1):
        start_md = ref_mds[i]
        end_md = ref_mds[i + 1]
        if abs(end_md - start_md) < 0.01:
            continue
        start_shift = ref_shifts[i - 1] if i > 0 else 0.0
        end_shift = ref_shifts[i]
        start_idx = np.searchsorted(well_md, start_md)
        end_idx = np.searchsorted(well_md, end_md)
        if start_idx >= len(well_md):
            start_idx = len(well_md) - 1
        if end_idx >= len(well_md):
            end_idx = len(well_md) - 1
        if start_idx == end_idx:
            continue
        start_vs = well_vs[start_idx]
        end_vs = well_vs[end_idx]
        segments.append([start_idx, end_idx, start_vs, end_vs, start_shift, end_shift])
    return np.array(segments, dtype=np.float64)

def main():
    print("=== synt_curve comparison ===\n")

    # Load data
    dataset_path = Path(__file__).parent / "dataset" / "gpu_ag_dataset.pt"
    dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)
    data = dataset[WELL_NAME]

    well_md = data['well_md'].cpu().numpy()
    well_tvd = data['well_tvd'].cpu().numpy()
    well_ns = data['well_ns'].cpu().numpy()
    well_ew = data['well_ew'].cpu().numpy()
    well_vs = calc_vs(well_ns, well_ew)
    log_md = data['log_md'].cpu().numpy()
    log_gr = data['log_gr'].cpu().numpy()
    well_gr = np.interp(well_md, log_md, log_gr)
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_shifts = data['ref_shifts'].cpu().numpy()
    type_tvd = data['type_tvd'].cpu().numpy()
    type_gr = data['type_gr'].cpu().numpy()
    tvd_typewell_shift = data.get('tvd_typewell_shift', 0.0)
    if isinstance(tvd_typewell_shift, torch.Tensor):
        tvd_typewell_shift = tvd_typewell_shift.item()

    target_md_m = TARGET_MD_FT / METERS_TO_FEET
    target_idx = np.searchsorted(well_md, target_md_m)
    target_idx = min(target_idx, len(well_md) - 1)

    print(f"Target: MD={target_md_m:.2f}m ({TARGET_MD_FT}ft), idx={target_idx}")
    print(f"well_gr at target: {well_gr[target_idx]:.4f}")
    print()

    # === NUMPY PATH (visualizer) ===
    print("=== NUMPY (visualizer) ===")
    tw_step = type_tvd[1] - type_tvd[0] if len(type_tvd) > 1 else 0.3048
    typewell_data = {
        'tvd': type_tvd,
        'value': type_gr,
        'min_depth': type_tvd.min(),
        'typewell_step': tw_step,
        'normalized': False,
    }
    well_data_np = {
        'md': well_md,
        'vs': well_vs,
        'tvd': well_tvd,
        'value': well_gr,
        'tvt': np.full_like(well_md, np.nan),
        'synt_curve': np.full_like(well_md, np.nan),
        'normalized': False,
    }
    segments_np = build_segments(ref_mds, ref_shifts, well_md, well_vs)

    success, well_data_np = calc_horizontal_projection_numpy(
        well_data_np, typewell_data, segments_np, tvd_to_typewell_shift=tvd_typewell_shift
    )

    numpy_tvt = well_data_np['tvt'][target_idx]
    numpy_synt = well_data_np['synt_curve'][target_idx]

    print(f"TVT at target: {numpy_tvt:.4f}")
    print(f"synt_curve at target: {numpy_synt:.4f}")
    print()

    # === TORCH PATH (gpu_executor) ===
    print("=== TORCH (gpu_executor) ===")
    from true_gpu_slicer import slice_data_to_md, build_manual_interpretation_to_md
    from gpu_executor import GpuAutoGeosteeringExecutor
    from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
    from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
    from torch_funcs.projection import calc_horizontal_projection_batch_torch

    start_md = data.get('detected_start_md') or data.get('start_md', 0.0) or 0.0
    current_md = 6100  # Past target
    data['well_name'] = WELL_NAME
    work_interpretation = build_manual_interpretation_to_md(data, start_md)
    well_data_json = slice_data_to_md(data, current_md)
    well_data_json['interpretation'] = {'segments': work_interpretation}

    work_dir = Path(__file__).parent / "results" / "debug_synt"
    work_dir.mkdir(parents=True, exist_ok=True)
    executor = GpuAutoGeosteeringExecutor(work_dir=work_dir, results_dir=work_dir)
    result = executor.initialize_well(well_data_json)

    # Get normalized data
    well = executor.ag_well
    typewell = executor.ag_typewell
    segments = executor.interpretation

    # Convert to numpy then torch (same path as gpu_executor)
    well_np = well_to_numpy(well)
    typewell_np = typewell_to_numpy(typewell)
    segments_np_opt = segments_to_numpy(segments, well)

    well_torch = numpy_to_torch(well_np, device='cpu')
    typewell_torch = numpy_to_torch(typewell_np, device='cpu')
    segments_torch = segments_numpy_to_torch(segments_np_opt, device='cpu')

    # Add batch dimension
    segments_batch = segments_torch.unsqueeze(0)

    # Run projection
    success_mask, tvt_batch, synt_batch, first_start_idx = calc_horizontal_projection_batch_torch(
        well_torch, typewell_torch, segments_batch, executor.tvd_to_typewell_shift
    )

    # Find target in torch output
    # torch indices are relative to segment range
    # Need to find where target_idx maps
    opt_start_idx = segments[-4].start_idx  # First optimization segment
    opt_end_idx = segments[-1].end_idx

    if target_idx < opt_start_idx or target_idx > opt_end_idx:
        print(f"ERROR: target_idx {target_idx} not in optimization range [{opt_start_idx}, {opt_end_idx}]")
        return

    rel_idx = target_idx - first_start_idx
    torch_tvt = tvt_batch[0, rel_idx].item()
    torch_synt = synt_batch[0, rel_idx].item()

    # Denormalize TVT
    md_range = executor.fixed_md_range
    torch_tvt_m = torch_tvt * md_range + typewell.wells_min_depth

    print(f"first_start_idx: {first_start_idx}, rel_idx: {rel_idx}")
    print(f"TVT at target (norm): {torch_tvt:.6f}")
    print(f"TVT at target (m): {torch_tvt_m:.4f}")
    print(f"synt_curve at target (norm): {torch_synt:.6f}")

    # Denormalize synt
    torch_synt_denorm = torch_synt * well.max_curve_value
    print(f"synt_curve at target (denorm): {torch_synt_denorm:.4f}")
    print()

    # === COMPARISON ===
    print("=== COMPARISON ===")
    print(f"NUMPY TVT: {numpy_tvt:.4f}m")
    print(f"TORCH TVT: {torch_tvt_m:.4f}m")
    print(f"TVT diff: {numpy_tvt - torch_tvt_m:.4f}m")
    print()
    print(f"NUMPY synt: {numpy_synt:.4f}")
    print(f"TORCH synt: {torch_synt_denorm:.4f}")
    print(f"synt diff: {numpy_synt - torch_synt_denorm:.4f}")

if __name__ == '__main__':
    main()
