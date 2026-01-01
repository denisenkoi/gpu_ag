#!/usr/bin/env python3
"""
Check normalized projection path - same as gpu_executor uses.
"""
import os
import sys
from pathlib import Path

os.environ['NORMALIZATION_MODE'] = 'ORIGINAL'
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from torch_funcs.projection import calc_horizontal_projection_batch_torch

WELL_NAME = "Well162~EGFDL"
METERS_TO_FEET = 3.28084
TARGET_MD_FT = 19800

def calc_vs(ns, ew):
    vs = np.zeros(len(ns))
    for i in range(1, len(ns)):
        vs[i] = vs[i-1] + np.sqrt((ns[i]-ns[i-1])**2 + (ew[i]-ew[i-1])**2)
    return vs

def build_segments(ref_mds, ref_shifts, well_md, well_vs):
    segments = []
    for i in range(len(ref_mds) - 1):
        start_md, end_md = ref_mds[i], ref_mds[i + 1]
        if abs(end_md - start_md) < 0.01:
            continue
        start_shift = ref_shifts[i - 1] if i > 0 else 0.0
        end_shift = ref_shifts[i]
        start_idx = min(np.searchsorted(well_md, start_md), len(well_md) - 1)
        end_idx = min(np.searchsorted(well_md, end_md), len(well_md) - 1)
        if start_idx == end_idx:
            continue
        segments.append([start_idx, end_idx, well_vs[start_idx], well_vs[end_idx], start_shift, end_shift])
    return np.array(segments, dtype=np.float64)

def main():
    print("=== Normalized projection check ===\n")

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
    tvd_typewell_shift = data.get('tvd_typewell_shift', torch.tensor(0.0))
    if isinstance(tvd_typewell_shift, torch.Tensor):
        tvd_typewell_shift = tvd_typewell_shift.item()

    target_md_m = TARGET_MD_FT / METERS_TO_FEET
    target_idx = min(np.searchsorted(well_md, target_md_m), len(well_md) - 1)

    # Build segments (raw)
    segments_raw = build_segments(ref_mds, ref_shifts, well_md, well_vs)

    # Normalization parameters (like gpu_executor)
    min_md = well_md.min()
    md_range = well_md.max() - min_md + 500  # buffer
    min_vs = well_vs.min()
    min_tvd = min(well_tvd.min(), type_tvd.min())
    max_gr = max(well_gr.max(), type_gr.max())

    print(f"Normalization params:")
    print(f"  min_md: {min_md:.2f}m, md_range: {md_range:.2f}m")
    print(f"  min_vs: {min_vs:.2f}m")
    print(f"  min_tvd: {min_tvd:.2f}m")
    print(f"  max_gr: {max_gr:.2f}")
    print()

    # Normalize well
    well_md_norm = (well_md - min_md) / md_range
    well_vs_norm = (well_vs - min_vs) / md_range
    well_tvd_norm = (well_tvd - min_tvd) / md_range
    well_gr_norm = well_gr / max_gr

    # Normalize typewell
    type_tvd_norm = (type_tvd - min_tvd) / md_range
    type_gr_norm = type_gr / max_gr
    tw_step_norm = (type_tvd[1] - type_tvd[0]) / md_range

    # Normalize segments
    segments_norm = segments_raw.copy()
    # VS: col 2,3
    segments_norm[:, 2] = (segments_norm[:, 2] - min_vs) / md_range
    segments_norm[:, 3] = (segments_norm[:, 3] - min_vs) / md_range
    # Shifts: col 4,5
    segments_norm[:, 4] = segments_norm[:, 4] / md_range
    segments_norm[:, 5] = segments_norm[:, 5] / md_range

    # Normalize tvd_typewell_shift
    tvd_shift_norm = tvd_typewell_shift / md_range

    print(f"Normalized tvd_typewell_shift: {tvd_shift_norm:.6f}")
    print()

    # Build torch data
    well_torch = {
        'md': torch.tensor(well_md_norm, dtype=torch.float64),
        'vs': torch.tensor(well_vs_norm, dtype=torch.float64),
        'tvd': torch.tensor(well_tvd_norm, dtype=torch.float64),
        'value': torch.tensor(well_gr_norm, dtype=torch.float64),
        'normalized': True,
    }
    typewell_torch = {
        'tvd': torch.tensor(type_tvd_norm, dtype=torch.float64),
        'value': torch.tensor(type_gr_norm, dtype=torch.float64),
        'normalized_min_depth': type_tvd_norm.min(),
        'normalized_typewell_step': tw_step_norm,
        'normalized': True,
    }
    segments_torch = torch.tensor(segments_norm, dtype=torch.float64).unsqueeze(0)

    # Run projection
    success_mask, tvt_batch, synt_batch, first_start_idx = calc_horizontal_projection_batch_torch(
        well_torch, typewell_torch, segments_torch, tvd_to_typewell_shift=tvd_shift_norm
    )

    # Get target value
    rel_idx = target_idx - first_start_idx
    if rel_idx < 0 or rel_idx >= tvt_batch.shape[1]:
        print(f"ERROR: rel_idx {rel_idx} out of range")
        return

    tvt_norm = tvt_batch[0, rel_idx].item()
    synt_norm = synt_batch[0, rel_idx].item()

    # Denormalize
    tvt_m = tvt_norm * md_range + min_tvd
    synt_denorm = synt_norm * max_gr

    print(f"=== NORMALIZED PATH ===")
    print(f"TVT (norm): {tvt_norm:.6f}")
    print(f"TVT (meters): {tvt_m:.4f}m")
    print(f"synt (norm): {synt_norm:.6f}")
    print(f"synt (denorm): {synt_denorm:.4f}")
    print()

    # Compare with expected
    print(f"=== COMPARISON ===")
    print(f"Expected TVT: 3824.3138m, Got: {tvt_m:.4f}m")
    print(f"Expected synt: 73.4961, Got: {synt_denorm:.4f}")
    print(f"TVT diff: {3824.3138 - tvt_m:.4f}m {'✓' if abs(3824.3138 - tvt_m) < 0.01 else '✗'}")
    print(f"synt diff: {73.4961 - synt_denorm:.4f} {'✓' if abs(73.4961 - synt_denorm) < 0.01 else '✗'}")

if __name__ == '__main__':
    main()
