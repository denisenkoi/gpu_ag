#!/usr/bin/env python3
"""
Test that torch projection matches numpy projection.
Uses ONLY existing functions - no new code.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Use existing functions from plot_gamma_comparison.py
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from plot_gamma_comparison import (
    load_dataset, stitch_typewell, build_segments_numpy, calc_vs_from_trajectory
)
from numpy_funcs.projection import calc_horizontal_projection_numpy
from torch_funcs.projection import calc_horizontal_projection_torch
from torch_funcs.batch_objective import TorchObjectiveWrapper

WELL_NAME = "Well162~EGFDL"
MD_START_FT = 19600
MD_END_FT = 20000
METERS_TO_FEET = 3.28084


def main():
    print(f"=== Testing projection: numpy vs torch ===")
    print(f"Well: {WELL_NAME}")
    print(f"Range: {MD_START_FT}-{MD_END_FT} ft")

    # Load data (same as plot_gamma_comparison.py)
    dataset = load_dataset()
    data = dataset[WELL_NAME]

    # Get arrays
    well_md = data['well_md'].cpu().numpy()
    well_tvd = data['well_tvd'].cpu().numpy()
    well_ns = data['well_ns'].cpu().numpy()
    well_ew = data['well_ew'].cpu().numpy()
    well_vs = calc_vs_from_trajectory(well_ns, well_ew)

    log_md = data['log_md'].cpu().numpy()
    log_gr = data['log_gr'].cpu().numpy()
    well_gr = np.interp(well_md, log_md, log_gr)

    # Reference interpretation
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_shifts = data['ref_shifts'].cpu().numpy()

    # TVD shift
    tvd_typewell_shift = data.get('tvd_typewell_shift', 0.0)
    if isinstance(tvd_typewell_shift, torch.Tensor):
        tvd_typewell_shift = tvd_typewell_shift.item()
    print(f"tvd_typewell_shift: {tvd_typewell_shift:.4f}m")

    # Build ORIGINAL typewell (same as plot_gamma_comparison.py with MODE='ORIGINAL')
    tw_tvd, tw_gr = stitch_typewell(data, mode='ORIGINAL')
    tw_step = tw_tvd[1] - tw_tvd[0] if len(tw_tvd) > 1 else 0.3048
    print(f"Typewell TVD: {tw_tvd.min():.1f}-{tw_tvd.max():.1f}m, points={len(tw_tvd)}")

    # Build segments
    segments_data = build_segments_numpy(ref_mds, ref_shifts, well_md, well_vs)
    print(f"Segments: {len(segments_data)}")

    # ========== NUMPY PROJECTION ==========
    typewell_data_np = {
        'tvd': tw_tvd,
        'value': tw_gr,
        'min_depth': tw_tvd.min(),
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

    success_np, well_data_np = calc_horizontal_projection_numpy(
        well_data_np, typewell_data_np, segments_data, tvd_to_typewell_shift=tvd_typewell_shift
    )
    synt_np = well_data_np['synt_curve']
    print(f"Numpy projection success: {success_np}")

    # ========== TORCH PROJECTION ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    typewell_data_torch = {
        'tvd': torch.tensor(tw_tvd, dtype=torch.float64, device=device),
        'value': torch.tensor(tw_gr, dtype=torch.float64, device=device),
        'normalized': False,
        'min_depth': tw_tvd.min(),
        'typewell_step': tw_step,
    }
    well_data_torch = {
        'md': torch.tensor(well_md, dtype=torch.float64, device=device),
        'vs': torch.tensor(well_vs, dtype=torch.float64, device=device),
        'tvd': torch.tensor(well_tvd, dtype=torch.float64, device=device),
        'value': torch.tensor(well_gr, dtype=torch.float64, device=device),
        'tvt': torch.full((len(well_md),), float('nan'), dtype=torch.float64, device=device),
        'synt_curve': torch.full((len(well_md),), float('nan'), dtype=torch.float64, device=device),
    }
    segments_torch = torch.tensor(segments_data, dtype=torch.float64, device=device)

    success_torch, well_data_torch = calc_horizontal_projection_torch(
        well_data_torch, typewell_data_torch, segments_torch, tvd_to_typewell_shift=tvd_typewell_shift
    )
    synt_torch = well_data_torch['synt_curve'].cpu().numpy()
    print(f"Torch projection success: {success_torch}")

    # ========== COMPARE ==========
    valid_np = ~np.isnan(synt_np)
    valid_torch = ~np.isnan(synt_torch)

    print(f"\nValid points: numpy={valid_np.sum()}, torch={valid_torch.sum()}")

    # Check if valid masks match
    mask_match = np.array_equal(valid_np, valid_torch)
    print(f"Valid masks match: {mask_match}")

    # Compare values where both valid
    both_valid = valid_np & valid_torch
    if both_valid.sum() > 0:
        diff = np.abs(synt_np[both_valid] - synt_torch[both_valid])
        print(f"Max diff: {diff.max():.6f}")
        print(f"Mean diff: {diff.mean():.6f}")

    # ========== PEARSON on range 19600-20000ft ==========
    md_start_m = MD_START_FT / METERS_TO_FEET
    md_end_m = MD_END_FT / METERS_TO_FEET

    range_mask = (well_md >= md_start_m) & (well_md <= md_end_m)

    # Numpy Pearson
    np_range_valid = range_mask & valid_np
    if np_range_valid.sum() > 10:
        pearson_np = np.corrcoef(well_gr[np_range_valid], synt_np[np_range_valid])[0, 1]
        print(f"\nNumpy Pearson ({MD_START_FT}-{MD_END_FT}ft): {pearson_np:.4f}")

    # Torch Pearson
    torch_range_valid = range_mask & valid_torch
    if torch_range_valid.sum() > 10:
        pearson_torch = np.corrcoef(well_gr[torch_range_valid], synt_torch[torch_range_valid])[0, 1]
        print(f"Torch Pearson ({MD_START_FT}-{MD_END_FT}ft): {pearson_torch:.4f}")

    # Summary
    print("\n=== SUMMARY ===")
    if success_np and success_torch and mask_match and diff.max() < 0.001:
        print("✓ Projections MATCH (raw data)")
    else:
        print("✗ Projections DIFFER - investigate!")

    # ========== TEST WITH NORMALIZATION (like gpu_executor) ==========
    print("\n\n=== Testing with NORMALIZATION (like gpu_executor) ===")

    # Normalization parameters (same as gpu_executor)
    fixed_md_range = well_md.max() - well_md.min() + 1000  # buffer like gpu_executor
    min_md = well_md.min()
    wells_min_depth = min(tw_tvd.min(), well_tvd.min())  # = 0 for this well

    # Normalize tvd_typewell_shift
    tvd_shift_norm = tvd_typewell_shift / fixed_md_range
    print(f"tvd_typewell_shift normalized: {tvd_shift_norm:.6f}")

    # Normalize well data
    well_md_norm = (well_md - min_md) / fixed_md_range
    well_tvd_norm = (well_tvd - wells_min_depth) / fixed_md_range
    well_vs_norm = (well_vs - well_vs.min()) / fixed_md_range

    # Normalize typewell
    tw_tvd_norm = (tw_tvd - wells_min_depth) / fixed_md_range
    tw_step_norm = tw_step / fixed_md_range

    # Normalize segments
    segments_norm = segments_data.copy()
    # segments: [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
    # vs and shift need normalization
    segments_norm[:, 2] = (segments_norm[:, 2] - well_vs.min()) / fixed_md_range  # start_vs
    segments_norm[:, 3] = (segments_norm[:, 3] - well_vs.min()) / fixed_md_range  # end_vs
    segments_norm[:, 4] = segments_norm[:, 4] / fixed_md_range  # start_shift
    segments_norm[:, 5] = segments_norm[:, 5] / fixed_md_range  # end_shift

    # Build normalized torch data
    typewell_data_norm = {
        'tvd': torch.tensor(tw_tvd_norm, dtype=torch.float64, device=device),
        'value': torch.tensor(tw_gr, dtype=torch.float64, device=device),  # value NOT normalized
        'normalized': True,
        'normalized_min_depth': tw_tvd_norm.min(),
        'normalized_typewell_step': tw_step_norm,
    }
    well_data_norm = {
        'md': torch.tensor(well_md_norm, dtype=torch.float64, device=device),
        'vs': torch.tensor(well_vs_norm, dtype=torch.float64, device=device),
        'tvd': torch.tensor(well_tvd_norm, dtype=torch.float64, device=device),
        'value': torch.tensor(well_gr, dtype=torch.float64, device=device),  # value NOT normalized
        'tvt': torch.full((len(well_md),), float('nan'), dtype=torch.float64, device=device),
        'synt_curve': torch.full((len(well_md),), float('nan'), dtype=torch.float64, device=device),
    }
    segments_norm_torch = torch.tensor(segments_norm, dtype=torch.float64, device=device)

    success_norm, well_data_norm = calc_horizontal_projection_torch(
        well_data_norm, typewell_data_norm, segments_norm_torch, tvd_to_typewell_shift=tvd_shift_norm
    )
    synt_norm = well_data_norm['synt_curve'].cpu().numpy()
    print(f"Normalized projection success: {success_norm}")

    # Compare with raw torch
    valid_norm = ~np.isnan(synt_norm)
    print(f"Valid points: raw_torch={valid_torch.sum()}, normalized={valid_norm.sum()}")

    both_valid_norm = valid_torch & valid_norm
    if both_valid_norm.sum() > 0:
        diff_norm = np.abs(synt_torch[both_valid_norm] - synt_norm[both_valid_norm])
        print(f"Max diff (raw vs normalized): {diff_norm.max():.6f}")

    # Pearson for normalized
    norm_range_valid = range_mask & valid_norm
    if norm_range_valid.sum() > 10:
        pearson_norm = np.corrcoef(well_gr[norm_range_valid], synt_norm[norm_range_valid])[0, 1]
        print(f"Normalized Pearson ({MD_START_FT}-{MD_END_FT}ft): {pearson_norm:.4f}")

    # ========== TEST PARTIAL SEGMENTS (like telescope) ==========
    print("\n\n=== Testing PARTIAL segments (like telescope first step) ===")

    # Find segments that cover our range 19600-20000ft (5974-6096m)
    md_start_m = MD_START_FT / METERS_TO_FEET
    md_end_m = MD_END_FT / METERS_TO_FEET

    # ref_mds contains segment END MDs
    # Find which segments end in our range or before
    partial_mask = ref_mds <= md_end_m + 50  # include some buffer
    partial_count = partial_mask.sum()
    print(f"Reference segments up to {md_end_m:.0f}m: {partial_count} / {len(ref_mds)}")

    # Take last few segments only (like telescope work segments)
    last_n = 10  # Take last 10 segments
    if len(segments_data) > last_n:
        partial_segments = segments_data[-last_n:]
        print(f"Using last {last_n} segments for partial test")

        # Torch projection for partial segments
        well_data_partial = {
            'md': torch.tensor(well_md, dtype=torch.float64, device=device),
            'vs': torch.tensor(well_vs, dtype=torch.float64, device=device),
            'tvd': torch.tensor(well_tvd, dtype=torch.float64, device=device),
            'value': torch.tensor(well_gr, dtype=torch.float64, device=device),
            'tvt': torch.full((len(well_md),), float('nan'), dtype=torch.float64, device=device),
            'synt_curve': torch.full((len(well_md),), float('nan'), dtype=torch.float64, device=device),
        }
        partial_segments_torch = torch.tensor(partial_segments, dtype=torch.float64, device=device)

        success_partial, well_data_partial = calc_horizontal_projection_torch(
            well_data_partial, typewell_data_torch, partial_segments_torch,
            tvd_to_typewell_shift=tvd_typewell_shift
        )
        synt_partial = well_data_partial['synt_curve'].cpu().numpy()
        print(f"Partial projection success: {success_partial}")

        # Pearson for partial
        valid_partial = ~np.isnan(synt_partial)
        partial_range_valid = range_mask & valid_partial
        if partial_range_valid.sum() > 10:
            pearson_partial = np.corrcoef(well_gr[partial_range_valid], synt_partial[partial_range_valid])[0, 1]
            print(f"Partial Pearson ({MD_START_FT}-{MD_END_FT}ft): {pearson_partial:.4f}")
        else:
            print(f"Not enough valid points in range: {partial_range_valid.sum()}")

        # Show which MD range partial covers
        first_idx = int(partial_segments[0, 0])
        last_idx = int(partial_segments[-1, 1])
        print(f"Partial covers MD: {well_md[first_idx]:.0f} - {well_md[last_idx]:.0f}m "
              f"({well_md[first_idx]*METERS_TO_FEET:.0f} - {well_md[last_idx]*METERS_TO_FEET:.0f}ft)")

    # ========== TEST TorchObjectiveWrapper.compute_detailed ==========
    print("\n\n=== Testing TorchObjectiveWrapper.compute_detailed ===")

    # Normalize value (like gpu_executor does)
    max_curve_value = max(well_gr.max(), tw_gr.max())
    well_gr_norm = well_gr / max_curve_value
    tw_gr_norm = tw_gr / max_curve_value
    print(f"max_curve_value: {max_curve_value:.2f}")

    # Rebuild normalized data with normalized value
    typewell_data_norm_full = {
        'tvd': torch.tensor(tw_tvd_norm, dtype=torch.float64, device=device),
        'value': torch.tensor(tw_gr_norm, dtype=torch.float64, device=device),
        'normalized': True,
        'normalized_min_depth': tw_tvd_norm.min(),
        'normalized_typewell_step': tw_step_norm,
    }
    well_data_norm_full = {
        'md': torch.tensor(well_md_norm, dtype=torch.float64, device=device),
        'vs': torch.tensor(well_vs_norm, dtype=torch.float64, device=device),
        'tvd': torch.tensor(well_tvd_norm, dtype=torch.float64, device=device),
        'value': torch.tensor(well_gr_norm, dtype=torch.float64, device=device),
        'tvt': torch.full((len(well_md),), float('nan'), dtype=torch.float64, device=device),
        'synt_curve': torch.full((len(well_md),), float('nan'), dtype=torch.float64, device=device),
    }

    # Find segments that cover our range 5974-6096m (19600-20000ft)
    target_md_start = md_start_m  # 5974m
    target_md_end = md_end_m  # 6096m

    # Find segment indices that overlap with target range
    # segments_data format: [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
    segment_start_mds = well_md[segments_data[:, 0].astype(int)]
    segment_end_mds = well_md[segments_data[:, 1].astype(int)]

    # Find segments where end >= target_start AND start <= target_end
    overlap_mask = (segment_end_mds >= target_md_start) & (segment_start_mds <= target_md_end)
    overlap_indices = np.where(overlap_mask)[0]
    print(f"Segments overlapping {target_md_start:.0f}-{target_md_end:.0f}m: indices {overlap_indices}")

    if len(overlap_indices) >= 4:
        # Take 4 segments that cover target range
        wrap_indices = overlap_indices[:4]
        wrap_segments_raw = segments_data[wrap_indices]
        wrap_segments = segments_norm[wrap_indices]
        wrap_segments_torch = torch.tensor(wrap_segments, dtype=torch.float64, device=device)

        # Show which MD range these segments cover
        first_idx_s = int(wrap_segments_raw[0, 0])
        last_idx_s = int(wrap_segments_raw[-1, 1])
        print(f"Selected segments cover MD: {well_md[first_idx_s]:.0f} - {well_md[last_idx_s]:.0f}m "
              f"({well_md[first_idx_s]*METERS_TO_FEET:.0f} - {well_md[last_idx_s]*METERS_TO_FEET:.0f}ft)")

        # Extract shifts from segments (end_shift column)
        ref_shifts_for_wrap = wrap_segments[:, 5]  # end_shift, already normalized
        print(f"Reference shifts (normalized): {ref_shifts_for_wrap}")

        # Create wrapper (like gpu_executor does)
        wrapper = TorchObjectiveWrapper(
            well_data=well_data_norm_full,
            typewell_data=typewell_data_norm_full,
            segments_torch=wrap_segments_torch,
            self_corr_start_idx=0,
            pearson_power=1.0,
            mse_power=2.0,
            num_intervals_self_correlation=1,
            sc_power=1.0,
            angle_range=0.35,  # ~20 degrees
            angle_sum_power=2.0,
            min_pearson_value=0.3,
            tvd_to_typewell_shift=tvd_shift_norm,
            prev_segment_angle=None,
            device=device,
            reward_start_segment_idx=0  # No telescope skip
        )

        # Compute detailed metrics
        metrics = wrapper.compute_detailed(torch.tensor(ref_shifts_for_wrap, dtype=torch.float64, device=device))
        print(f"compute_detailed results:")
        print(f"  Pearson: {metrics['pearson']:.4f}")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  Objective: {metrics['objective']:.6f}")
        print(f"  Angles (deg): {metrics['angles_deg']}")

        # Verify - Wrapper should now cover our target range
        first_idx_w = int(wrap_segments_raw[0, 0])
        last_idx_w = int(wrap_segments_raw[-1, 1])
        print(f"Wrapper covers MD: {well_md[first_idx_w]:.0f} - {well_md[last_idx_w]:.0f}m "
              f"({well_md[first_idx_w]*METERS_TO_FEET:.0f} - {well_md[last_idx_w]*METERS_TO_FEET:.0f}ft)")


if __name__ == '__main__':
    main()
