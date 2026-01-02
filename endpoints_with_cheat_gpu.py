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
    BATCHED version - all shifts evaluated in single GPU call.
    """
    from torch_funcs.projection import calc_horizontal_projection_batch_torch
    from torch_funcs.correlations import pearson_batch_torch
    from torch_funcs.converters import GPU_DTYPE

    mds = np.array(well_data.get('ref_segment_mds', []))
    shifts = np.array(well_data.get('ref_shifts', []))

    if len(mds) < 2:
        return baseline_shift, 0.0

    end_md = float(mds[-1])
    start_md = end_md - window_length
    ref_shift_at_start = float(np.interp(start_md, mds, shifts))

    # Prepare well data
    well_md = np.array(well_data.get('well_md', []))
    well_tvd = np.array(well_data.get('well_tvd', []))
    well_ns = np.array(well_data.get('well_ns', []))
    well_ew = np.array(well_data.get('well_ew', []))
    log_md = np.array(well_data.get('log_md', []))
    log_gr = np.array(well_data.get('log_gr', []))
    type_tvd = np.array(well_data.get('type_tvd', []))
    type_gr = np.array(well_data.get('type_gr', []))

    # Calculate VS
    well_vs = np.zeros(len(well_ns))
    for i in range(1, len(well_ns)):
        well_vs[i] = well_vs[i-1] + np.sqrt((well_ns[i] - well_ns[i-1])**2 + (well_ew[i] - well_ew[i-1])**2)

    # Interpolate GR to well_md
    gr_at_well_md = np.interp(well_md, log_md, log_gr)

    # Find indices for segment
    start_idx = int(np.searchsorted(well_md, start_md))
    end_idx = int(np.searchsorted(well_md, end_md))
    start_idx = min(start_idx, len(well_md) - 1)
    end_idx = min(end_idx, len(well_md) - 1)

    if start_idx >= end_idx:
        return baseline_shift, 0.0

    start_vs = well_vs[start_idx]
    end_vs = well_vs[end_idx]

    # Generate all test shifts
    shifts_to_test = np.linspace(
        baseline_shift - search_range,
        baseline_shift + search_range,
        n_steps
    )

    # Build batch of segments: (n_steps, 1, 6)
    # Format: [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
    segments_list = []
    for test_shift in shifts_to_test:
        if parallel_shift:
            delta = test_shift - baseline_shift
            seg_start_shift = ref_shift_at_start + delta
            seg_end_shift = float(test_shift)
        else:
            seg_start_shift = ref_shift_at_start
            seg_end_shift = float(test_shift)

        segments_list.append([start_idx, end_idx, start_vs, end_vs, seg_start_shift, seg_end_shift])

    segments_np = np.array(segments_list, dtype=np.float32).reshape(n_steps, 1, 6)
    segments_torch = torch.tensor(segments_np, dtype=GPU_DTYPE, device=device)

    # Prepare torch data
    tw_step = float(type_tvd[1] - type_tvd[0])
    min_depth = float(type_tvd.min())

    well_torch = {
        'md': torch.tensor(well_md, dtype=GPU_DTYPE, device=device),
        'vs': torch.tensor(well_vs, dtype=GPU_DTYPE, device=device),
        'tvd': torch.tensor(well_tvd, dtype=GPU_DTYPE, device=device),
        'value': torch.tensor(gr_at_well_md, dtype=GPU_DTYPE, device=device),
        'normalized': False,
    }
    typewell_torch = {
        'tvd': torch.tensor(type_tvd, dtype=GPU_DTYPE, device=device),
        'value': torch.tensor(type_gr, dtype=GPU_DTYPE, device=device),
        'min_depth': min_depth,
        'typewell_step': tw_step,
        'normalized': False,
    }

    # Single batched projection call!
    success_mask, tvt_batch, synt_batch, first_idx = calc_horizontal_projection_batch_torch(
        well_torch, typewell_torch, segments_torch, tvd_to_typewell_shift=0.0
    )

    # Get well GR for range
    n_points = end_idx - start_idx + 1
    well_gr_range = torch.tensor(
        gr_at_well_md[start_idx:end_idx + 1], dtype=GPU_DTYPE, device=device
    ).unsqueeze(0).expand(n_steps, -1)  # (n_steps, n_points)

    synt_range = synt_batch[:, :n_points]  # (n_steps, n_points)

    # Compute pearson for all batches
    pearsons = pearson_batch_torch(well_gr_range, synt_range)  # (n_steps,)

    # Set failed batches to -inf
    pearsons = torch.where(success_mask, pearsons, torch.tensor(-float('inf'), device=device))

    # Compute angle penalties (vectorized)
    shifts_t = torch.tensor(shifts_to_test, dtype=GPU_DTYPE, device=device)
    shift_deltas = shifts_t - ref_shift_at_start
    angle_degs = torch.rad2deg(torch.atan(shift_deltas / window_length))
    angle_penalties = (angle_degs - avg_angle) ** 2

    # Compute objectives
    objectives = pearsons - 0.01 * angle_penalties

    # If all failed, return baseline
    if not success_mask.any():
        return baseline_shift, -float('inf')

    # Find best among successful
    best_idx = torch.argmax(objectives).item()
    best_shift = shifts_to_test[best_idx]
    best_objective = objectives[best_idx].item()

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
