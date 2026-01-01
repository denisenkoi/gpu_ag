#!/usr/bin/env python3
"""
Take reference segments, trim to optimization MD range, compute projection.
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

def calc_vs(ns, ew):
    vs = np.zeros(len(ns))
    for i in range(1, len(ns)):
        vs[i] = vs[i-1] + np.sqrt((ns[i]-ns[i-1])**2 + (ew[i]-ew[i-1])**2)
    return vs

def build_ref_segments_in_range(ref_mds, ref_shifts, well_md, well_vs, md_start, md_end):
    """Build reference segments that fall within [md_start, md_end]"""
    segments = []

    for i in range(len(ref_mds) - 1):
        seg_start_md = ref_mds[i]
        seg_end_md = ref_mds[i + 1]

        # Skip if entirely outside range
        if seg_end_md < md_start or seg_start_md > md_end:
            continue

        # Clip to range
        clipped_start_md = max(seg_start_md, md_start)
        clipped_end_md = min(seg_end_md, md_end)

        if clipped_end_md - clipped_start_md < 0.1:
            continue

        # Get shifts - need to interpolate if clipped
        start_shift = ref_shifts[i - 1] if i > 0 else 0.0
        end_shift = ref_shifts[i]

        # Interpolate shifts for clipped boundaries
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

        # Find indices
        start_idx = np.searchsorted(well_md, clipped_start_md)
        end_idx = np.searchsorted(well_md, clipped_end_md)

        start_idx = min(start_idx, len(well_md) - 1)
        end_idx = min(end_idx, len(well_md) - 1)

        if start_idx >= end_idx:
            continue

        segments.append([
            start_idx, end_idx,
            well_vs[start_idx], well_vs[end_idx],
            clipped_start_shift, clipped_end_shift
        ])

    return np.array(segments, dtype=np.float64)

def main():
    print("=== Reference segments in optimization range ===\n")

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

    # Optimization range from previous run
    OPT_MD_START = 4124.9
    OPT_MD_END = 6134.1

    print(f"Optimization range: {OPT_MD_START:.1f} - {OPT_MD_END:.1f}m")
    print(f"({OPT_MD_START * METERS_TO_FEET:.0f} - {OPT_MD_END * METERS_TO_FEET:.0f}ft)")
    print()

    # Build reference segments in opt range
    ref_segments = build_ref_segments_in_range(
        ref_mds, ref_shifts, well_md, well_vs, OPT_MD_START, OPT_MD_END
    )
    print(f"Reference segments in range: {len(ref_segments)}")

    if len(ref_segments) == 0:
        print("ERROR: No reference segments in range!")
        return

    # Get range indices
    start_idx = int(ref_segments[0, 0])
    end_idx = int(ref_segments[-1, 1])
    print(f"Index range: {start_idx} - {end_idx}")
    print()

    # Build torch data (raw, not normalized)
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
    segments_torch = torch.tensor(ref_segments, dtype=torch.float64).unsqueeze(0)

    # Compute projection
    success_mask, tvt_batch, synt_batch, first_start_idx = calc_horizontal_projection_batch_torch(
        well_torch, typewell_torch, segments_torch, tvd_to_typewell_shift=tvd_typewell_shift
    )

    print(f"Projection success: {success_mask[0].item()}")
    print(f"first_start_idx: {first_start_idx}")

    # Get well GR and synt for the range
    n_points = end_idx - start_idx + 1
    well_gr_range = torch.tensor(well_gr[start_idx:end_idx+1], dtype=torch.float64).unsqueeze(0)
    synt_range = synt_batch[:, :n_points]

    # Check for NaN
    valid_mask = ~torch.isnan(synt_range[0])
    valid_count = valid_mask.sum().item()
    print(f"Valid synt points: {valid_count} / {n_points}")

    if valid_count < 10:
        print("ERROR: Too few valid points!")
        return

    # Compute Pearson
    well_gr_valid = well_gr_range[0, valid_mask].unsqueeze(0)
    synt_valid = synt_range[0, valid_mask].unsqueeze(0)

    pearson = pearson_batch_torch(well_gr_valid, synt_valid)
    print(f"\n=== RESULT ===")
    print(f"Pearson (reference segments, opt range): {pearson[0].item():.4f}")

    # Also compute for sub-ranges
    print(f"\n=== Sub-range Pearson ===")
    range_start = start_idx
    for md_ft in [14000, 16000, 18000, 20000]:
        md_m = md_ft / METERS_TO_FEET
        if md_m < OPT_MD_START or md_m > OPT_MD_END:
            continue
        range_end = np.searchsorted(well_md, md_m)
        if range_end <= range_start:
            continue

        rel_start = range_start - first_start_idx
        rel_end = range_end - first_start_idx

        if rel_end > synt_batch.shape[1]:
            rel_end = synt_batch.shape[1]

        sub_synt = synt_batch[0, rel_start:rel_end]
        sub_well = torch.tensor(well_gr[range_start:range_end], dtype=torch.float64)

        valid = ~torch.isnan(sub_synt)
        if valid.sum() > 10:
            p = pearson_batch_torch(sub_well[valid].unsqueeze(0), sub_synt[valid].unsqueeze(0))
            print(f"  MD {OPT_MD_START * METERS_TO_FEET:.0f}-{md_ft}ft: Pearson = {p[0].item():.4f}")

    # Check specific range 19600-20000ft
    print(f"\n=== Specific range 19600-20000ft ===")
    md_start_ft = 19600
    md_end_ft = 20000
    md_start_m = md_start_ft / METERS_TO_FEET
    md_end_m = md_end_ft / METERS_TO_FEET

    idx_start = np.searchsorted(well_md, md_start_m)
    idx_end = np.searchsorted(well_md, md_end_m)

    rel_start = idx_start - first_start_idx
    rel_end = idx_end - first_start_idx

    if rel_start >= 0 and rel_end <= synt_batch.shape[1]:
        sub_synt = synt_batch[0, rel_start:rel_end]
        sub_well = torch.tensor(well_gr[idx_start:idx_end], dtype=torch.float64)
        valid = ~torch.isnan(sub_synt)
        if valid.sum() > 10:
            p = pearson_batch_torch(sub_well[valid].unsqueeze(0), sub_synt[valid].unsqueeze(0))
            print(f"  Pearson (19600-20000ft): {p[0].item():.4f}")
        else:
            print(f"  Not enough valid points: {valid.sum()}")
    else:
        print(f"  Range {rel_start}-{rel_end} not in batch [0, {synt_batch.shape[1]})")

if __name__ == '__main__':
    main()
