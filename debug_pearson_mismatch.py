#!/usr/bin/env python3
"""
Debug: Why Method 3 (executor) has different Pearson than Method 1/2 (raw)?
Focus on understanding NaN points and data alignment.
"""
import os
import sys
from pathlib import Path

os.environ['NORMALIZATION_MODE'] = 'ORIGINAL'
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'cpu_baseline'))

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
    print("=== Debug Pearson Mismatch ===\n")

    dataset_path = Path(__file__).parent / "dataset" / "gpu_ag_dataset.pt"
    dataset = torch.load(dataset_path, map_location='cpu', weights_only=False)
    data = dataset[WELL_NAME]

    # Raw data from dataset
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

    # Dataset stats
    print("=== Dataset Stats ===")
    print(f"well_md: {well_md.min():.2f} - {well_md.max():.2f} (len={len(well_md)})")
    print(f"well_tvd: {well_tvd.min():.2f} - {well_tvd.max():.2f}")
    print(f"type_tvd: {type_tvd.min():.2f} - {type_tvd.max():.2f} (len={len(type_tvd)})")
    tw_step = type_tvd[1] - type_tvd[0]
    print(f"type_tvd step: {tw_step:.6f}m")
    print()

    # Build reference segments in target range
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
    print(f"=== Reference segments: {len(segments_raw)} ===")

    # Calculate expected TVT range for reference
    first_seg = segments_raw[0]
    last_seg = segments_raw[-1]
    first_tvd = well_tvd[int(first_seg[0])]
    last_tvd = well_tvd[int(last_seg[1])]
    first_shift = first_seg[4]
    last_shift = last_seg[5]

    print(f"\nTVD at range start (idx={int(first_seg[0])}): {first_tvd:.2f}m")
    print(f"TVD at range end (idx={int(last_seg[1])}): {last_tvd:.2f}m")
    print(f"Shift at start: {first_shift:.2f}m")
    print(f"Shift at end: {last_shift:.2f}m")

    # Expected TVT range
    tvt_start = first_tvd - first_shift - tvd_typewell_shift
    tvt_end = last_tvd - last_shift - tvd_typewell_shift

    print(f"\nExpected TVT range: {tvt_start:.2f} - {tvt_end:.2f}m")
    print(f"Type TVD range: {type_tvd.min():.2f} - {type_tvd.max():.2f}m")

    if tvt_start < type_tvd.min():
        print(f"  WARNING: TVT start {tvt_start:.2f} < type_tvd.min {type_tvd.min():.2f}")
    if tvt_end > type_tvd.max():
        print(f"  WARNING: TVT end {tvt_end:.2f} > type_tvd.max {type_tvd.max():.2f}")

    # ============================================================
    # Method 1: Raw data, same typewell step as dataset
    # ============================================================
    print("\n=== METHOD 1: Raw data (same step as dataset) ===")

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

    success, tvt, synt, first_idx = calc_horizontal_projection_batch_torch(
        well_torch, typewell_torch, segments_torch, tvd_to_typewell_shift=tvd_typewell_shift
    )

    start_idx = int(segments_raw[0, 0])
    end_idx = int(segments_raw[-1, 1])
    n_points = end_idx - start_idx + 1

    synt_range = synt[0, :n_points]
    well_gr_range = torch.tensor(well_gr[start_idx:end_idx+1], dtype=torch.float64)

    valid = ~torch.isnan(synt_range)
    print(f"Valid points: {valid.sum().item()}/{n_points}")

    # Debug: where are NaN?
    if valid.sum() < n_points:
        nan_positions = torch.where(~valid)[0]
        print(f"NaN positions (first 10): {nan_positions[:10].tolist()}")

        # Check TVT at NaN positions
        tvt_range = tvt[0, :n_points]
        for pos in nan_positions[:5]:
            abs_idx = start_idx + pos.item()
            print(f"  pos {pos.item()}: idx={abs_idx}, tvd={well_tvd[abs_idx]:.2f}, "
                  f"tvt={tvt_range[pos].item():.2f}, type_tvd range: {type_tvd.min():.2f}-{type_tvd.max():.2f}")

    if valid.sum() > 10:
        pearson = pearson_batch_torch(
            well_gr_range[valid].unsqueeze(0),
            synt_range[valid].unsqueeze(0)
        )
        print(f"Pearson (Method 1): {pearson[0].item():.6f}")
    else:
        print("Not enough valid points!")
        return

    # ============================================================
    # Method 2: Using TypeWell object (with its default 0.03048m step)
    # ============================================================
    print("\n=== METHOD 2: Using TypeWell object (default step 0.03048m) ===")

    from cpu_baseline.ag_objects.ag_obj_typewell import TypeWell
    from true_gpu_slicer import slice_data_to_md, build_manual_interpretation_to_md
    from numpy_funcs.converters import typewell_to_numpy
    from torch_funcs.converters import numpy_to_torch

    # Create TypeWell from full data (no slice)
    data['well_name'] = WELL_NAME
    # Don't slice - use full data
    well_data_json = {
        'trajectory': {
            'points': [
                {'measuredDepth': md, 'trueVerticalDepth': tvd, 'ns': ns, 'ew': ew}
                for md, tvd, ns, ew in zip(well_md.tolist(), well_tvd.tolist(),
                                           well_ns.tolist(), well_ew.tolist())
            ]
        },
        'gammaLog': {
            'points': [
                {'measuredDepth': md, 'data': gr}
                for md, gr in zip(log_md.tolist(), log_gr.tolist())
            ]
        },
        'typeLog': {
            'tvdSortedPoints': [
                {'trueVerticalDepth': tvd, 'data': gr}
                for tvd, gr in zip(type_tvd.tolist(), type_gr.tolist())
            ]
        }
    }

    typewell_obj = TypeWell(well_data_json)
    print(f"TypeWell depth range: {typewell_obj.min_depth:.2f} - {typewell_obj.tvd.max():.2f}")
    print(f"TypeWell step: {typewell_obj.typewell_step:.6f}m")
    print(f"TypeWell data points: {len(typewell_obj.tvd)}")
    typewell_max_depth = typewell_obj.tvd.max()

    # Check if expected TVT is within TypeWell range
    print(f"\nExpected TVT range: {tvt_start:.2f} - {tvt_end:.2f}m")
    print(f"TypeWell range: {typewell_obj.min_depth:.2f} - {typewell_max_depth:.2f}m")

    if tvt_start < typewell_obj.min_depth:
        print(f"  WARNING: TVT start outside TypeWell!")
    if tvt_end > typewell_max_depth:
        print(f"  WARNING: TVT end outside TypeWell!")

    # Convert to numpy/torch
    typewell_np = typewell_to_numpy(typewell_obj)
    typewell_torch2 = numpy_to_torch(typewell_np, device='cpu')

    # Projection with TypeWell object
    success2, tvt2, synt2, first_idx2 = calc_horizontal_projection_batch_torch(
        well_torch, typewell_torch2, segments_torch, tvd_to_typewell_shift=tvd_typewell_shift
    )

    synt_range2 = synt2[0, :n_points]
    valid2 = ~torch.isnan(synt_range2)
    print(f"Valid points: {valid2.sum().item()}/{n_points}")

    if valid2.sum() < n_points:
        nan_positions = torch.where(~valid2)[0]
        print(f"NaN positions (first 10): {nan_positions[:10].tolist()}")

    if valid2.sum() > 10:
        pearson2 = pearson_batch_torch(
            well_gr_range[valid2].unsqueeze(0),
            synt_range2[valid2].unsqueeze(0)
        )
        print(f"Pearson (Method 2): {pearson2[0].item():.6f}")

    # ============================================================
    # Method 3: Sliced data (like gpu_executor)
    # ============================================================
    print("\n=== METHOD 3: Sliced data (gpu_executor style) ===")

    # Slice from start_md (optimization start) to md_end_m + 100
    # Like true_gpu_slicer does
    start_md = data.get('detected_start_md') or data.get('start_md', 0.0) or 0.0
    if isinstance(start_md, torch.Tensor):
        start_md = start_md.item()

    print(f"Slice start_md: {start_md:.2f}m")
    print(f"Slice end_md: {md_end_m + 100:.2f}m")

    # Simulate slice_data_to_md
    slice_end_md = md_end_m + 100

    # Find start index for slice
    slice_start_idx = np.searchsorted(well_md, start_md)
    slice_end_idx = np.searchsorted(well_md, slice_end_md)

    print(f"Slice indices: {slice_start_idx} - {slice_end_idx} (from total {len(well_md)})")

    # Sliced arrays
    sliced_well_md = well_md[slice_start_idx:slice_end_idx+1]
    sliced_well_tvd = well_tvd[slice_start_idx:slice_end_idx+1]
    sliced_well_vs = well_vs[slice_start_idx:slice_end_idx+1]
    sliced_well_gr = well_gr[slice_start_idx:slice_end_idx+1]

    print(f"Sliced well_md: {sliced_well_md.min():.2f} - {sliced_well_md.max():.2f} (len={len(sliced_well_md)})")

    # Recalculate VS for sliced data (from 0)
    sliced_ns = well_ns[slice_start_idx:slice_end_idx+1]
    sliced_ew = well_ew[slice_start_idx:slice_end_idx+1]
    sliced_well_vs_recalc = calc_vs(sliced_ns, sliced_ew)

    # Build segments for sliced data
    segments_sliced = []
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

        # Find indices in SLICED data
        start_idx_sliced = np.searchsorted(sliced_well_md, clipped_start_md)
        end_idx_sliced = np.searchsorted(sliced_well_md, clipped_end_md)
        start_idx_sliced = min(start_idx_sliced, len(sliced_well_md) - 1)
        end_idx_sliced = min(end_idx_sliced, len(sliced_well_md) - 1)

        if start_idx_sliced >= end_idx_sliced:
            continue

        segments_sliced.append([
            start_idx_sliced, end_idx_sliced,
            sliced_well_vs_recalc[start_idx_sliced], sliced_well_vs_recalc[end_idx_sliced],
            clipped_start_shift, clipped_end_shift
        ])

    segments_sliced = np.array(segments_sliced, dtype=np.float64)
    print(f"\nSliced segments: {len(segments_sliced)}")

    if len(segments_sliced) == 0:
        print("No segments in sliced data!")
        return

    for i, seg in enumerate(segments_sliced[:3]):
        print(f"  Seg {i}: idx={int(seg[0])}-{int(seg[1])}, vs={seg[2]:.2f}-{seg[3]:.2f}, shift={seg[4]:.2f}->{seg[5]:.2f}")

    # Build torch data for sliced
    well_torch_sliced = {
        'md': torch.tensor(sliced_well_md, dtype=torch.float64),
        'vs': torch.tensor(sliced_well_vs_recalc, dtype=torch.float64),
        'tvd': torch.tensor(sliced_well_tvd, dtype=torch.float64),
        'value': torch.tensor(sliced_well_gr, dtype=torch.float64),
        'normalized': False,
    }
    segments_torch_sliced = torch.tensor(segments_sliced, dtype=torch.float64).unsqueeze(0)

    # Projection
    success3, tvt3, synt3, first_idx3 = calc_horizontal_projection_batch_torch(
        well_torch_sliced, typewell_torch2, segments_torch_sliced, tvd_to_typewell_shift=tvd_typewell_shift
    )

    sliced_start_idx = int(segments_sliced[0, 0])
    sliced_end_idx = int(segments_sliced[-1, 1])
    sliced_n_points = sliced_end_idx - sliced_start_idx + 1

    synt_range3 = synt3[0, :sliced_n_points]
    well_gr_sliced_range = torch.tensor(sliced_well_gr[sliced_start_idx:sliced_end_idx+1], dtype=torch.float64)

    valid3 = ~torch.isnan(synt_range3)
    print(f"Valid points: {valid3.sum().item()}/{sliced_n_points}")

    if valid3.sum() > 10:
        pearson3 = pearson_batch_torch(
            well_gr_sliced_range[valid3].unsqueeze(0),
            synt_range3[valid3].unsqueeze(0)
        )
        print(f"Pearson (Method 3): {pearson3[0].item():.6f}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n=== SUMMARY ===")
    print(f"Method 1 (raw, dataset step):   Pearson = {pearson[0].item():.6f}")
    print(f"Method 2 (raw, TypeWell obj):   Pearson = {pearson2[0].item():.6f}")
    print(f"Method 3 (sliced, TypeWell):    Pearson = {pearson3[0].item():.6f}")

    diff12 = abs(pearson[0].item() - pearson2[0].item())
    diff13 = abs(pearson[0].item() - pearson3[0].item())

    print(f"\nDiff 1-2: {diff12:.6f} {'✓' if diff12 < 0.01 else '✗'}")
    print(f"Diff 1-3: {diff13:.6f} {'✓' if diff13 < 0.01 else '✗'}")

if __name__ == '__main__':
    main()
