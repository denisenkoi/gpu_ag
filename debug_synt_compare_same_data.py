#!/usr/bin/env python3
"""
Compare EXACT synt_curve: numpy vs torch on SAME data.
No gpu_executor - direct torch projection call.
"""
import os
import sys
from pathlib import Path

os.environ['NORMALIZATION_MODE'] = 'ORIGINAL'
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from numpy_funcs.projection import calc_horizontal_projection_numpy
from torch_funcs.projection import calc_horizontal_projection_batch_torch

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
    print("=== synt_curve: numpy vs torch (SAME DATA) ===\n")

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
    print(f"tvd_typewell_shift: {tvd_typewell_shift:.4f}m")
    print()

    # Build typewell data
    tw_step = type_tvd[1] - type_tvd[0] if len(type_tvd) > 1 else 0.3048
    typewell_data_np = {
        'tvd': type_tvd,
        'value': type_gr,
        'min_depth': type_tvd.min(),
        'typewell_step': tw_step,
        'normalized': False,
    }

    # Build well data
    well_data_np = {
        'md': well_md,
        'vs': well_vs,
        'tvd': well_tvd,
        'value': well_gr,
        'tvt': np.full_like(well_md, np.nan),
        'synt_curve': np.full_like(well_md, np.nan),
        'normalized': False,
    }

    # Build reference segments
    segments_np = build_segments(ref_mds, ref_shifts, well_md, well_vs)
    print(f"Reference segments: {len(segments_np)}")

    # === NUMPY ===
    print("\n=== NUMPY ===")
    success, well_data_np = calc_horizontal_projection_numpy(
        well_data_np, typewell_data_np, segments_np, tvd_to_typewell_shift=tvd_typewell_shift
    )
    numpy_tvt = well_data_np['tvt'][target_idx]
    numpy_synt = well_data_np['synt_curve'][target_idx]
    print(f"TVT: {numpy_tvt:.4f}m")
    print(f"synt_curve: {numpy_synt:.4f}")

    # === TORCH (same data, no normalization) ===
    print("\n=== TORCH ===")

    # Convert to torch tensors
    well_data_torch = {
        'md': torch.tensor(well_md, dtype=torch.float64),
        'vs': torch.tensor(well_vs, dtype=torch.float64),
        'tvd': torch.tensor(well_tvd, dtype=torch.float64),
        'value': torch.tensor(well_gr, dtype=torch.float64),
        'normalized': False,
    }
    typewell_data_torch = {
        'tvd': torch.tensor(type_tvd, dtype=torch.float64),
        'value': torch.tensor(type_gr, dtype=torch.float64),
        'min_depth': type_tvd.min(),
        'typewell_step': tw_step,
        'normalized': False,
    }
    segments_torch = torch.tensor(segments_np, dtype=torch.float64).unsqueeze(0)  # Add batch dim

    success_mask, tvt_batch, synt_batch, first_start_idx = calc_horizontal_projection_batch_torch(
        well_data_torch, typewell_data_torch, segments_torch, tvd_to_typewell_shift=tvd_typewell_shift
    )

    # Get values at target
    rel_idx = target_idx - first_start_idx
    if rel_idx < 0 or rel_idx >= tvt_batch.shape[1]:
        print(f"ERROR: rel_idx {rel_idx} out of range [0, {tvt_batch.shape[1]})")
        print(f"first_start_idx: {first_start_idx}, target_idx: {target_idx}")
        return

    torch_tvt = tvt_batch[0, rel_idx].item()
    torch_synt = synt_batch[0, rel_idx].item()
    print(f"first_start_idx: {first_start_idx}, rel_idx: {rel_idx}")
    print(f"TVT: {torch_tvt:.4f}m")
    print(f"synt_curve: {torch_synt:.4f}")

    # === COMPARISON ===
    print("\n=== COMPARISON ===")
    tvt_diff = numpy_tvt - torch_tvt
    synt_diff = numpy_synt - torch_synt
    print(f"TVT diff: {tvt_diff:.6f}m {'✓' if abs(tvt_diff) < 0.001 else '✗ MISMATCH!'}")
    print(f"synt diff: {synt_diff:.6f} {'✓' if abs(synt_diff) < 0.001 else '✗ MISMATCH!'}")

if __name__ == '__main__':
    main()
