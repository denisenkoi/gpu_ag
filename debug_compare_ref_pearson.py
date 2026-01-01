#!/usr/bin/env python3
"""
Compare Reference Pearson: debug script vs gpu_executor on EXACT same data.
Goal: values must match exactly.
"""
import os
import sys
from pathlib import Path

os.environ['NORMALIZATION_MODE'] = 'ORIGINAL'
sys.path.insert(0, str(Path(__file__).parent))

import torch
import numpy as np

from torch_funcs.projection import calc_horizontal_projection_batch_torch
from torch_funcs.correlations import pearson_batch_torch

WELL_NAME = "Well162~EGFDL"
METERS_TO_FEET = 3.28084

# Target range: 19600-20000ft
TARGET_MD_START_FT = 19600
TARGET_MD_END_FT = 20000

def calc_vs(ns, ew):
    vs = np.zeros(len(ns))
    for i in range(1, len(ns)):
        vs[i] = vs[i-1] + np.sqrt((ns[i]-ns[i-1])**2 + (ew[i]-ew[i-1])**2)
    return vs

def main():
    print("=== Compare Reference Pearson: debug vs gpu_executor ===\n")

    dataset_path = Path(__file__).parent / "dataset" / "gpu_ag_dataset.pt"
    dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)
    data = dataset[WELL_NAME]

    # Raw data
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

    # Target MD range in meters
    md_start_m = TARGET_MD_START_FT / METERS_TO_FEET
    md_end_m = TARGET_MD_END_FT / METERS_TO_FEET

    print(f"Target range: {TARGET_MD_START_FT}-{TARGET_MD_END_FT}ft = {md_start_m:.2f}-{md_end_m:.2f}m")
    print(f"tvd_typewell_shift: {tvd_typewell_shift:.4f}m")
    print()

    # ============================================================
    # METHOD 1: Debug script approach (raw, unnormalized)
    # ============================================================
    print("=== METHOD 1: Raw data (debug script) ===")

    # Build reference segments in range
    segments_raw = []
    for i in range(len(ref_mds) - 1):
        seg_start_md = ref_mds[i]
        seg_end_md = ref_mds[i + 1]

        if seg_end_md < md_start_m or seg_start_md > md_end_m:
            continue

        clipped_start_md = max(seg_start_md, md_start_m)
        clipped_end_md = min(seg_end_md, md_end_m)

        if clipped_end_md - clipped_start_md < 0.1:
            continue

        start_shift = ref_shifts[i - 1] if i > 0 else 0.0
        end_shift = ref_shifts[i]

        seg_len = seg_end_md - seg_start_md
        if seg_len > 0:
            if clipped_start_md > seg_start_md:
                frac = (clipped_start_md - seg_start_md) / seg_len
                clipped_start_shift = start_shift + (end_shift - start_shift) * frac
            else:
                clipped_start_shift = start_shift

            if clipped_end_md < seg_end_md:
                frac = (clipped_end_md - seg_start_md) / seg_len
                clipped_end_shift = start_shift + (end_shift - start_shift) * frac
            else:
                clipped_end_shift = end_shift
        else:
            clipped_start_shift = start_shift
            clipped_end_shift = end_shift

        start_idx = np.searchsorted(well_md, clipped_start_md)
        end_idx = np.searchsorted(well_md, clipped_end_md)
        start_idx = min(start_idx, len(well_md) - 1)
        end_idx = min(end_idx, len(well_md) - 1)

        if start_idx >= end_idx:
            continue

        segments_raw.append([
            start_idx, end_idx,
            well_vs[start_idx], well_vs[end_idx],
            clipped_start_shift, clipped_end_shift
        ])

    segments_raw = np.array(segments_raw, dtype=np.float64)
    print(f"Reference segments: {len(segments_raw)}")

    for i, seg in enumerate(segments_raw[:3]):
        print(f"  Seg {i}: idx={int(seg[0])}-{int(seg[1])}, vs={seg[2]:.2f}-{seg[3]:.2f}, shift={seg[4]:.2f}->{seg[5]:.2f}")
    if len(segments_raw) > 3:
        print(f"  ...")
        seg = segments_raw[-1]
        print(f"  Seg {len(segments_raw)-1}: idx={int(seg[0])}-{int(seg[1])}, vs={seg[2]:.2f}-{seg[3]:.2f}, shift={seg[4]:.2f}->{seg[5]:.2f}")

    # Build torch data (raw)
    tw_step = type_tvd[1] - type_tvd[0]

    well_torch = {
        'md': torch.tensor(well_md, dtype=torch.float64),
        'vs': torch.tensor(well_vs, dtype=torch.float64),
        'tvd': torch.tensor(well_tvd, dtype=torch.float64),
        'value': torch.tensor(well_gr, dtype=torch.float64),
        'normalized': False,
    }
    typewell_torch = {
        'tvd': torch.tensor(type_tvd, dtype=torch.float64),
        'value': torch.tensor(type_gr, dtype=torch.float64),
        'min_depth': type_tvd.min(),
        'typewell_step': tw_step,
        'normalized': False,
    }
    segments_torch = torch.tensor(segments_raw, dtype=torch.float64).unsqueeze(0)

    # Compute projection
    success_mask, tvt_batch, synt_batch, first_start_idx = calc_horizontal_projection_batch_torch(
        well_torch, typewell_torch, segments_torch, tvd_to_typewell_shift=tvd_typewell_shift
    )

    # Get range indices
    start_idx = int(segments_raw[0, 0])
    end_idx = int(segments_raw[-1, 1])
    n_points = end_idx - start_idx + 1

    well_gr_range = torch.tensor(well_gr[start_idx:end_idx+1], dtype=torch.float64)
    synt_range = synt_batch[0, :n_points]

    valid_mask = ~torch.isnan(synt_range)
    valid_count = valid_mask.sum().item()

    print(f"Index range: {start_idx}-{end_idx}, points: {n_points}, valid: {valid_count}")

    if valid_count > 10:
        pearson_raw = pearson_batch_torch(
            well_gr_range[valid_mask].unsqueeze(0),
            synt_range[valid_mask].unsqueeze(0)
        )
        print(f"Pearson (raw): {pearson_raw[0].item():.6f}")
    else:
        print("Not enough valid points!")
        return

    # ============================================================
    # METHOD 2: gpu_executor approach (normalized)
    # ============================================================
    print("\n=== METHOD 2: Normalized data (gpu_executor style) ===")

    # Normalization params (like gpu_executor)
    min_md = well_md.min()
    md_range = well_md.max() - min_md + 500  # buffer like gpu_executor
    min_vs = well_vs.min()
    min_tvd = min(well_tvd.min(), type_tvd.min())
    max_gr = max(well_gr.max(), type_gr.max())

    print(f"Normalization: min_md={min_md:.2f}, md_range={md_range:.2f}, min_vs={min_vs:.2f}, min_tvd={min_tvd:.2f}")

    # Normalize segments
    segments_norm = segments_raw.copy()
    segments_norm[:, 2] = (segments_norm[:, 2] - min_vs) / md_range  # VS
    segments_norm[:, 3] = (segments_norm[:, 3] - min_vs) / md_range
    segments_norm[:, 4] = segments_norm[:, 4] / md_range  # Shifts
    segments_norm[:, 5] = segments_norm[:, 5] / md_range

    # Normalize well
    well_md_norm = (well_md - min_md) / md_range
    well_vs_norm = (well_vs - min_vs) / md_range
    well_tvd_norm = (well_tvd - min_tvd) / md_range
    well_gr_norm = well_gr / max_gr

    # Normalize typewell
    type_tvd_norm = (type_tvd - min_tvd) / md_range
    type_gr_norm = type_gr / max_gr
    tw_step_norm = tw_step / md_range

    # Normalize shift
    tvd_shift_norm = tvd_typewell_shift / md_range

    well_torch_norm = {
        'md': torch.tensor(well_md_norm, dtype=torch.float64),
        'vs': torch.tensor(well_vs_norm, dtype=torch.float64),
        'tvd': torch.tensor(well_tvd_norm, dtype=torch.float64),
        'value': torch.tensor(well_gr_norm, dtype=torch.float64),
        'normalized': True,
    }
    typewell_torch_norm = {
        'tvd': torch.tensor(type_tvd_norm, dtype=torch.float64),
        'value': torch.tensor(type_gr_norm, dtype=torch.float64),
        'normalized_min_depth': type_tvd_norm.min(),
        'normalized_typewell_step': tw_step_norm,
        'normalized': True,
    }
    segments_torch_norm = torch.tensor(segments_norm, dtype=torch.float64).unsqueeze(0)

    # Compute projection
    success_mask2, tvt_batch2, synt_batch2, first_start_idx2 = calc_horizontal_projection_batch_torch(
        well_torch_norm, typewell_torch_norm, segments_torch_norm, tvd_to_typewell_shift=tvd_shift_norm
    )

    # Same range
    well_gr_norm_range = torch.tensor(well_gr_norm[start_idx:end_idx+1], dtype=torch.float64)
    synt_norm_range = synt_batch2[0, :n_points]

    valid_mask2 = ~torch.isnan(synt_norm_range)
    valid_count2 = valid_mask2.sum().item()

    print(f"Valid points: {valid_count2}")

    if valid_count2 > 10:
        pearson_norm = pearson_batch_torch(
            well_gr_norm_range[valid_mask2].unsqueeze(0),
            synt_norm_range[valid_mask2].unsqueeze(0)
        )
        print(f"Pearson (normalized): {pearson_norm[0].item():.6f}")

    # ============================================================
    # METHOD 3: gpu_executor actual call
    # ============================================================
    print("\n=== METHOD 3: gpu_executor build_reference_segments_torch ===")

    from gpu_executor import build_reference_segments_torch
    from true_gpu_slicer import slice_data_to_md, build_manual_interpretation_to_md
    from cpu_baseline.ag_objects.ag_obj_well import Well

    # Build well like gpu_executor does
    data['well_name'] = WELL_NAME
    start_md = data.get('detected_start_md') or data.get('start_md', 0.0) or 0.0
    work_interpretation = build_manual_interpretation_to_md(data, start_md)
    well_data_json = slice_data_to_md(data, md_end_m + 100)
    well_data_json['interpretation'] = {'segments': work_interpretation}

    well_obj = Well(well_data_json)

    # Get reference segments in JSON format
    ref_segments_json = well_data_json.get('interpretation', {}).get('referenceSegments', [])
    if not ref_segments_json:
        ref_segments_json = well_data_json.get('referenceInterpretation', {}).get('segments', [])

    print(f"Reference segments JSON: {len(ref_segments_json)}")

    # Fixed normalization params from gpu_executor
    fixed_min_md = well_obj.min_md
    fixed_md_range = well_obj.md_range

    print(f"Well min_md={fixed_min_md:.2f}, md_range={fixed_md_range:.2f}")

    # Build reference segments using gpu_executor function
    ref_seg_torch = build_reference_segments_torch(
        ref_segments_json, well_obj, md_start_m, md_end_m,
        fixed_md_range, fixed_min_md, device='cpu'
    )

    if ref_seg_torch is not None:
        print(f"Built {len(ref_seg_torch)} reference segments")
        for i, seg in enumerate(ref_seg_torch[:3]):
            print(f"  Seg {i}: idx={int(seg[0])}-{int(seg[1])}, vs={seg[2]:.6f}-{seg[3]:.6f}, shift={seg[4]:.6f}->{seg[5]:.6f}")

        # Compare with segments_norm
        print("\n=== COMPARISON ===")
        print(f"{'':5} | {'Method 2 (norm)':>20} | {'Method 3 (executor)':>20} | {'Diff':>12}")
        print(f"{'-'*5}-+-{'-'*20}-+-{'-'*20}-+-{'-'*12}")

        min_len = min(len(segments_norm), len(ref_seg_torch))
        for i in range(min(3, min_len)):
            seg_norm = segments_norm[i]
            seg_exec = ref_seg_torch[i].numpy()

            # Compare VS
            vs_diff = abs(seg_norm[2] - seg_exec[2])
            print(f"Seg{i} VS start | {seg_norm[2]:>20.10f} | {seg_exec[2]:>20.10f} | {vs_diff:>12.10f}")

            # Compare shift
            shift_diff = abs(seg_norm[4] - seg_exec[4])
            print(f"Seg{i} shift_s | {seg_norm[4]:>20.10f} | {seg_exec[4]:>20.10f} | {shift_diff:>12.10f}")

        # ============================================================
        # METHOD 3b: Compute Pearson using executor's normalized data
        # ============================================================
        print("\n=== METHOD 3b: Compute Pearson with executor's data ===")

        from numpy_funcs.converters import well_to_numpy, typewell_to_numpy
        from torch_funcs.converters import numpy_to_torch

        # Get typewell from gpu_executor
        from cpu_baseline.ag_objects.ag_obj_typewell import TypeWell
        typewell_obj = TypeWell(well_data_json)

        well_np = well_to_numpy(well_obj)
        typewell_np = typewell_to_numpy(typewell_obj)

        well_torch_exec = numpy_to_torch(well_np, device='cpu')
        typewell_torch_exec = numpy_to_torch(typewell_np, device='cpu')

        # tvd_typewell_shift from dataset, normalized
        tvd_shift_exec = tvd_typewell_shift / fixed_md_range

        print(f"tvd_shift_exec (normalized): {tvd_shift_exec:.6f}")

        # Compute projection
        success, tvt_exec, synt_exec, first_idx = calc_horizontal_projection_batch_torch(
            well_torch_exec, typewell_torch_exec, ref_seg_torch.unsqueeze(0),
            tvd_to_typewell_shift=tvd_shift_exec
        )

        # Get range
        exec_start_idx = int(ref_seg_torch[0, 0])
        exec_end_idx = int(ref_seg_torch[-1, 1])
        exec_n_points = exec_end_idx - exec_start_idx + 1

        print(f"Exec idx range: {exec_start_idx}-{exec_end_idx}, points: {exec_n_points}")
        print(f"first_idx from projection: {first_idx}")

        # Get well GR and synt for range
        well_value_exec = well_torch_exec['value'][exec_start_idx:exec_end_idx+1]
        synt_exec_range = synt_exec[0, :exec_n_points]

        valid_exec = ~torch.isnan(synt_exec_range)
        valid_exec_count = valid_exec.sum().item()
        print(f"Valid points: {valid_exec_count}")

        if valid_exec_count > 10:
            pearson_exec = pearson_batch_torch(
                well_value_exec[valid_exec].unsqueeze(0),
                synt_exec_range[valid_exec].unsqueeze(0)
            )
            print(f"Pearson (executor normalized): {pearson_exec[0].item():.6f}")

            # FINAL COMPARISON
            print("\n=== FINAL RESULT ===")
            print(f"Method 1 (raw):        {pearson_raw[0].item():.6f}")
            print(f"Method 2 (norm full):  {pearson_norm[0].item():.6f}")
            print(f"Method 3 (executor):   {pearson_exec[0].item():.6f}")

            diff_1_3 = abs(pearson_raw[0].item() - pearson_exec[0].item())
            print(f"Diff Method1 vs Method3: {diff_1_3:.6f} {'✓ MATCH' if diff_1_3 < 0.001 else '✗ MISMATCH'}")
    else:
        print("No reference segments built!")

if __name__ == '__main__':
    main()
