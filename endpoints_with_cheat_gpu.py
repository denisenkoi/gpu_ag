#!/usr/bin/env python3
"""
Endpoint prediction with cheat (GPU version) - USING REAL PROJECTION.

Uses known shift_at_start from reference, brute force over endpoint shift.
Uses calc_horizontal_projection_batch_torch for proper segment interpolation.

Run: python endpoints_with_cheat_gpu.py
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from torch_funcs.gr_metrics import compute_gr_metrics


def compute_well_angle(well_data: dict, target_md: float, lookback: float = 100.0) -> float:
    """Compute angle from well trajectory."""
    well_md = np.array(well_data.get('well_md', []))
    well_tvd = np.array(well_data.get('well_tvd', []))

    if len(well_md) < 2:
        return 0.0

    end_idx = np.searchsorted(well_md, target_md)
    start_idx = np.searchsorted(well_md, target_md - lookback)

    if end_idx <= start_idx or end_idx >= len(well_md):
        return 0.0

    delta_tvd = well_tvd[end_idx] - well_tvd[start_idx]
    delta_md = well_md[end_idx] - well_md[start_idx]

    if delta_md <= 0:
        return 0.0

    return np.degrees(np.arctan(delta_tvd / delta_md))


def compute_baseline_shift(well_data: dict) -> tuple:
    """Compute TVT=const baseline prediction."""
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

    tvds_at_segments = np.interp(mds, well_md, well_tvd)
    max_idx = np.argmax(tvds_at_segments)
    max_tvd = tvds_at_segments[max_idx]
    shift_at_max_tvd = shifts[max_idx]
    tvt_at_max = max_tvd - shift_at_max_tvd
    tvd_end = tvds_at_segments[-1]
    predicted_shift = tvd_end - tvt_at_max

    return predicted_shift, tvt_at_max


def prepare_well_data_for_metrics(well_data: dict) -> dict:
    """Prepare well data dict for compute_gr_metrics."""
    well_md = np.array(well_data.get('well_md', []))
    well_tvd = np.array(well_data.get('well_tvd', []))
    well_ns = np.array(well_data.get('well_ns', []))
    well_ew = np.array(well_data.get('well_ew', []))
    log_md = np.array(well_data.get('log_md', []))
    log_gr = np.array(well_data.get('log_gr', []))

    # Interpolate GR to well_md positions
    gr_at_well_md = np.interp(well_md, log_md, log_gr)

    return {
        'md': well_md,
        'tvd': well_tvd,
        'ns': well_ns,
        'ew': well_ew,
        'value': gr_at_well_md,
    }


def prepare_typewell_data_for_metrics(well_data: dict) -> dict:
    """Prepare typewell data dict for compute_gr_metrics."""
    type_tvd = np.array(well_data.get('type_tvd', []))
    type_gr = np.array(well_data.get('type_gr', []))

    return {
        'tvd': type_tvd,
        'value': type_gr,
    }


def optimize_endpoint_gpu(
    well_data: dict,
    baseline_shift: float,
    avg_angle: float,
    search_range: float = 10.0,
    n_steps: int = 41,
    window_length: float = 60.0,
    parallel_shift: bool = False,
    device: str = 'cuda'
) -> tuple:
    """
    Grid search using GPU projection with real segments.

    Args:
        parallel_shift: If True, move both start and end shift together (like TVD - const).
                       If False, only move end shift (interpolation within segment).
    """
    mds = np.array(well_data.get('ref_segment_mds', []))
    shifts = np.array(well_data.get('ref_shifts', []))

    if len(mds) < 2:
        return baseline_shift, 0.0

    end_md = float(mds[-1])
    start_md = end_md - window_length
    ref_shift_at_start = float(np.interp(start_md, mds, shifts))

    # Prepare data for compute_gr_metrics
    well_data_for_metrics = prepare_well_data_for_metrics(well_data)
    typewell_data = prepare_typewell_data_for_metrics(well_data)

    best_objective = -float('inf')
    best_shift = baseline_shift
    best_pearson = 0.0

    shifts_to_test = np.linspace(
        baseline_shift - search_range,
        baseline_shift + search_range,
        n_steps
    )

    for test_shift in shifts_to_test:
        if parallel_shift:
            # Move both ends together (parallel shift = TVD - const)
            delta = test_shift - baseline_shift
            start_shift = ref_shift_at_start + delta
            end_shift = float(test_shift)
        else:
            # Only move end shift (interpolation)
            start_shift = ref_shift_at_start
            end_shift = float(test_shift)

        # Create segment
        segment = {
            'startMd': start_md,
            'endMd': end_md,
            'startShift': start_shift,
            'endShift': end_shift,
        }

        # Compute GR metrics using real projection
        metrics = compute_gr_metrics(
            well_data=well_data_for_metrics,
            typewell_data=typewell_data,
            interpretation_segments=[segment],
            md_start=start_md,
            md_end=end_md,
            tvd_to_typewell_shift=0.0,
            device=device
        )

        if not metrics.get('success', False):
            continue

        pearson = metrics['pearson_raw']

        # Angle penalty (same formula as baseline_with_objective.py)
        shift_delta = test_shift - ref_shift_at_start
        angle_deg = np.degrees(np.arctan(shift_delta / window_length))
        angle_penalty = (angle_deg - avg_angle) ** 2

        # Objective: correlation - angle_penalty * weight
        objective = pearson - 0.01 * angle_penalty

        if objective > best_objective:
            best_objective = objective
            best_shift = test_shift
            best_pearson = pearson

    return best_shift, best_objective


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel', action='store_true',
                       help='Parallel shift (move both ends together, like TVD-const)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Mode: {'PARALLEL shift (both ends)' if args.parallel else 'INTERPOLATION (only end shift)'}")

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

        ref_shift_end = shifts[-1]
        baseline_shift, tvt_at_max = compute_baseline_shift(well_data)
        baseline_error = baseline_shift - ref_shift_end

        end_md = mds[-1]
        well_angle = compute_well_angle(well_data, end_md, lookback=100.0)

        # GPU optimization with real projection
        optimized_shift, best_obj = optimize_endpoint_gpu(
            well_data=well_data,
            baseline_shift=baseline_shift,
            avg_angle=well_angle,
            search_range=10.0,
            n_steps=41,
            window_length=60.0,
            parallel_shift=args.parallel,
            device=device
        )
        optimized_error = optimized_shift - ref_shift_end

        results.append({
            'well': well_name,
            'ref_shift': ref_shift_end,
            'baseline_error': baseline_error,
            'optimized_error': optimized_error,
        })

        print(f"{well_name}: baseline={baseline_error:+.2f}m, opt={optimized_error:+.2f}m")

    # Summary
    mode_name = "parallel" if args.parallel else "interpolation"
    print("\n" + "="*60)
    print(f"SUMMARY (GPU {mode_name})")
    print("="*60)

    baseline_errors = [r['baseline_error'] for r in results]
    optimized_errors = [r['optimized_error'] for r in results]

    baseline_rmse = np.sqrt(np.mean(np.array(baseline_errors)**2))
    optimized_rmse = np.sqrt(np.mean(np.array(optimized_errors)**2))

    print(f"\nBaseline (TVT=const):  RMSE {baseline_rmse:.2f}m")
    print(f"Optimized ({mode_name}):  RMSE {optimized_rmse:.2f}m ({optimized_rmse - baseline_rmse:+.2f}m)")

    improved = sum(1 for b, o in zip(baseline_errors, optimized_errors) if abs(o) < abs(b))
    print(f"\nImproved: {improved}/{len(results)} wells")


if __name__ == '__main__':
    main()
