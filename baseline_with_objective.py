#!/usr/bin/env python3
"""
Baseline + Objective optimization experiment.

Idea:
1. Start from TVT=const baseline prediction
2. Optimize last 2 segments using our objective (Pearson + MSE + angle penalties)
3. Constrain segment angles to be close to average well angle
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from torch_funcs.batch_objective import compute_detailed_metrics_torch
from torch_funcs.projection import calc_horizontal_projection_batch_torch


def compute_well_angle(well_data: dict, target_md: float, lookback: float = 100.0) -> float:
    """
    Compute angle from WELL TRAJECTORY (not interpretation).
    Angle = arctan(delta_tvd / delta_md)
    """
    well_md = np.array(well_data.get('well_md', []))
    well_tvd = np.array(well_data.get('well_tvd', []))

    if len(well_md) < 2:
        return 0.0

    # Find indices for lookback window
    end_idx = np.searchsorted(well_md, target_md)
    start_idx = np.searchsorted(well_md, target_md - lookback)

    if end_idx <= start_idx or end_idx >= len(well_md):
        return 0.0

    delta_tvd = well_tvd[end_idx] - well_tvd[start_idx]
    delta_md = well_md[end_idx] - well_md[start_idx]

    if delta_md <= 0:
        return 0.0

    return np.degrees(np.arctan(delta_tvd / delta_md))


def compute_average_angle(well_data: dict) -> float:
    """Compute average angle from reference segments (PEEKING!)."""
    mds = well_data.get('ref_segment_mds', [])
    shifts = well_data.get('ref_shifts', [])

    if len(mds) < 2 or len(shifts) < 2:
        return 0.0

    mds = np.array(mds)
    shifts = np.array(shifts)

    angles = []
    for i in range(1, len(mds)):
        md_delta = mds[i] - mds[i-1]
        shift_delta = shifts[i] - shifts[i-1]

        if md_delta > 0:
            angle_rad = np.arctan(shift_delta / md_delta)
            angles.append(np.degrees(angle_rad))

    return np.mean(angles) if angles else 0.0


def compute_baseline_shift(well_data: dict) -> tuple:
    """
    Compute TVT=const baseline prediction.
    Returns (predicted_shift_at_end, tvt_at_max_tvd)

    Formula: TVT = TVD - shift
    """
    mds = well_data.get('ref_segment_mds', [])
    shifts = well_data.get('ref_shifts', [])
    well_md = well_data.get('well_md', [])
    well_tvd = well_data.get('well_tvd', [])

    if len(mds) < 2 or len(shifts) < 2:
        return 0.0, 0.0

    mds = np.array(mds)
    shifts = np.array(shifts)
    well_md = np.array(well_md)
    well_tvd = np.array(well_tvd)

    # Interpolate TVD at segment MDs
    tvds_at_segments = np.interp(mds, well_md, well_tvd)

    # Find point with max TVD
    max_idx = np.argmax(tvds_at_segments)
    max_tvd = tvds_at_segments[max_idx]
    shift_at_max_tvd = shifts[max_idx]

    # TVT at max TVD point (TVT = TVD - shift)
    tvt_at_max = max_tvd - shift_at_max_tvd

    # End point
    tvd_end = tvds_at_segments[-1]

    # Predicted shift assuming TVT stays constant
    predicted_shift = tvd_end - tvt_at_max

    return predicted_shift, tvt_at_max


def compute_simple_gr_correlation(
    well_data: dict,
    test_shift: float,
    window_start_md: float,
    window_end_md: float,
) -> float:
    """
    Compute GR correlation for given shift in window.
    Simple version: just shift TVD and lookup typewell GR.
    """
    # Get data (use correct keys from dataset)
    md = np.array(well_data.get('log_md', well_data.get('well_md', [])))
    tvd = np.array(well_data.get('well_tvd', []))
    gr = np.array(well_data.get('log_gr', []))
    typewell_tvd = np.array(well_data.get('type_tvd', []))
    typewell_gr = np.array(well_data.get('type_gr', []))

    if len(md) == 0 or len(gr) == 0:
        return 0.0

    # Interpolate TVD at log_md positions if needed
    well_md = np.array(well_data.get('well_md', []))
    if len(well_md) > 0 and len(tvd) > 0:
        tvd_at_log = np.interp(md, well_md, tvd)
    else:
        tvd_at_log = tvd

    # Find window indices
    start_idx = np.searchsorted(md, window_start_md)
    end_idx = np.searchsorted(md, window_end_md)

    if end_idx - start_idx < 10:
        return 0.0

    # Get well GR in window
    well_gr = gr[start_idx:end_idx]
    well_tvd = tvd_at_log[start_idx:end_idx]

    # Compute TVT = TVD - shift
    tvt = well_tvd - test_shift

    # Lookup typewell GR at TVT positions
    # Simple linear interpolation
    typewell_gr_interp = np.interp(tvt, typewell_tvd, typewell_gr)

    # Compute Pearson correlation
    if np.std(well_gr) < 1e-6 or np.std(typewell_gr_interp) < 1e-6:
        return 0.0

    correlation = np.corrcoef(well_gr, typewell_gr_interp)[0, 1]
    if np.isnan(correlation):
        return 0.0

    return correlation


def optimize_endpoint_grid_search(
    well_data: dict,
    baseline_shift: float,
    avg_angle: float,
    search_range: float = 10.0,
    n_steps: int = 41,
    window_length: float = 60.0,
) -> tuple:
    """
    Grid search around baseline using GR correlation.
    Angle constraint: penalize deviations from avg_angle.

    Uses shift_at_start from reference (known data).
    """
    mds = np.array(well_data.get('ref_segment_mds', []))
    shifts = np.array(well_data.get('ref_shifts', []))

    if len(mds) < 2:
        return baseline_shift, 0.0

    end_md = mds[-1]
    start_md = end_md - window_length

    # Get shift at start of window (known data)
    shift_at_start = np.interp(start_md, mds, shifts)

    best_objective = -float('inf')
    best_shift = baseline_shift

    shifts_to_test = np.linspace(
        baseline_shift - search_range,
        baseline_shift + search_range,
        n_steps
    )

    for test_shift in shifts_to_test:
        # Compute angle from start to end
        md_delta = window_length
        shift_delta = test_shift - shift_at_start
        angle_rad = np.arctan(shift_delta / md_delta)
        angle_deg = np.degrees(angle_rad)

        # Angle penalty: squared deviation from average
        angle_penalty = (angle_deg - avg_angle) ** 2

        # GR correlation
        correlation = compute_simple_gr_correlation(
            well_data, test_shift, start_md, end_md
        )

        # Objective: correlation - angle_penalty * weight
        # Higher is better
        objective = correlation - 0.01 * angle_penalty

        if objective > best_objective:
            best_objective = objective
            best_shift = test_shift

    return best_shift, best_objective


def main():
    # Load dataset
    dataset_path = '/mnt/e/Projects/Rogii/gpu_ag/dataset/gpu_ag_dataset.pt'
    print(f"Loading dataset from {dataset_path}")
    dataset = torch.load(dataset_path, weights_only=False)
    print(f"Loaded {len(dataset)} wells")

    results = []

    for well_name, well_data in dataset.items():
        mds = well_data.get('ref_segment_mds', [])
        shifts = well_data.get('ref_shifts', [])

        if len(mds) < 3 or len(shifts) < 3:
            continue

        # Reference endpoint
        ref_shift_end = shifts[-1]

        # Baseline prediction (TVT=const)
        baseline_shift, tvt_at_max = compute_baseline_shift(well_data)
        baseline_error = baseline_shift - ref_shift_end

        # Well trajectory angle (HONEST - no peeking!)
        end_md = mds[-1]
        well_angle = compute_well_angle(well_data, end_md, lookback=100.0)

        # For comparison: angle from reference (peeking)
        ref_angle = compute_average_angle(well_data)
        avg_angle = well_angle  # Use honest angle

        # Optimize with objective + angle constraint
        optimized_shift, best_obj = optimize_endpoint_grid_search(
            well_data=well_data,
            baseline_shift=baseline_shift,
            avg_angle=avg_angle,
            search_range=10.0,
            n_steps=41,
            window_length=60.0,
        )
        optimized_error = optimized_shift - ref_shift_end

        results.append({
            'well': well_name,
            'ref_shift': ref_shift_end,
            'baseline_shift': baseline_shift,
            'baseline_error': baseline_error,
            'avg_angle': avg_angle,
            'optimized_shift': optimized_shift,
            'optimized_error': optimized_error,
            'objective': best_obj,
        })

        print(f"{well_name}: baseline={baseline_error:+.2f}m, opt={optimized_error:+.2f}m, angle={avg_angle:.2f}°, obj={best_obj:.3f}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    baseline_errors = [r['baseline_error'] for r in results]
    optimized_errors = [r['optimized_error'] for r in results]

    baseline_mae = np.mean(np.abs(baseline_errors))
    baseline_rmse = np.sqrt(np.mean(np.array(baseline_errors)**2))

    optimized_mae = np.mean(np.abs(optimized_errors))
    optimized_rmse = np.sqrt(np.mean(np.array(optimized_errors)**2))

    print(f"\nBaseline (TVT=const):")
    print(f"  MAE:  {baseline_mae:.2f}m")
    print(f"  RMSE: {baseline_rmse:.2f}m")

    print(f"\nOptimized (corr + angle constraint):")
    print(f"  MAE:  {optimized_mae:.2f}m ({optimized_mae - baseline_mae:+.2f}m)")
    print(f"  RMSE: {optimized_rmse:.2f}m ({optimized_rmse - baseline_rmse:+.2f}m)")

    # Count improvements
    improved = sum(1 for b, o in zip(baseline_errors, optimized_errors) if abs(o) < abs(b))
    print(f"\nImproved: {improved}/{len(results)} wells")

    # Angle statistics
    angles = [r['avg_angle'] for r in results]
    print(f"\nAverage angle stats:")
    print(f"  Mean: {np.mean(angles):.2f}°")
    print(f"  Std:  {np.std(angles):.2f}°")
    print(f"  Range: [{np.min(angles):.2f}°, {np.max(angles):.2f}°]")


if __name__ == '__main__':
    main()
