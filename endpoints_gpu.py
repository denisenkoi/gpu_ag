#!/usr/bin/env python3
"""
Endpoint prediction (GPU version) - 2D grid search.

Grid search over (end_shift, angle) with (1-pearson)*MSE objective.
No angle penalty - let the objective find optimal combination.

Run: python endpoints_gpu.py --window 10 --search-range 30 --angle-range 0.5
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from numpy_funcs.interpretation import interpolate_shift_at_md


def compute_baseline_shift(well_data: dict, end_md: float = None) -> tuple:
    """
    Compute TVT=const baseline prediction.

    Args:
        well_data: dict with well trajectory data
        end_md: optional end MD (if None, uses last interpretation point)

    Returns:
        (predicted_shift_at_end, tvt_at_max)
    """
    mds = well_data.get('ref_segment_mds', [])
    start_shifts = well_data.get('ref_start_shifts', [])
    well_md = well_data.get('well_md', [])
    well_tvd = well_data.get('well_tvd', [])

    if len(mds) < 2 or len(start_shifts) < 2:
        return 0.0, 0.0

    mds = np.array(mds)
    start_shifts = np.array(start_shifts)
    well_md = np.array(well_md)
    well_tvd = np.array(well_tvd)

    # Find max TVD point and compute TVT there
    # Use start_shifts which correspond to startMd (mds) points
    tvds_at_segments = np.interp(mds, well_md, well_tvd)
    max_idx = np.argmax(tvds_at_segments)
    max_tvd = tvds_at_segments[max_idx]
    shift_at_max_tvd = start_shifts[max_idx]
    tvt_at_max = max_tvd - shift_at_max_tvd

    # Predict shift at end_md using TVT=const
    if end_md is None:
        tvd_end = tvds_at_segments[-1]
    else:
        tvd_end = float(np.interp(end_md, well_md, well_tvd))
    predicted_shift = tvd_end - tvt_at_max

    return predicted_shift, tvt_at_max


def smooth_gr(gr, window_size):
    """Simple moving average smoothing."""
    if window_size <= 1:
        return gr
    kernel = np.ones(window_size) / window_size
    return np.convolve(gr, kernel, mode='same')


def compute_well_angle(well_data: dict, start_md: float, end_md: float) -> float:
    """
    Compute angle from WELL TRAJECTORY (not interpretation).
    Angle = arctan(delta_tvd / delta_vs)

    Args:
        well_data: dict with well_md, well_tvd, well_ns, well_ew
        start_md: landing point MD
        end_md: end point MD
    """
    well_md = np.array(well_data.get('well_md', []))
    well_tvd = np.array(well_data.get('well_tvd', []))
    well_ns = np.array(well_data.get('well_ns', []))
    well_ew = np.array(well_data.get('well_ew', []))

    if len(well_md) < 2:
        return 0.0

    # Find indices
    start_idx = np.searchsorted(well_md, start_md)
    end_idx = np.searchsorted(well_md, end_md)

    if end_idx <= start_idx or end_idx >= len(well_md):
        return 0.0

    # Calculate VS (vertical section - horizontal distance)
    vs_start = 0.0
    vs_end = 0.0
    for i in range(1, end_idx + 1):
        delta_vs = np.sqrt((well_ns[i] - well_ns[i-1])**2 + (well_ew[i] - well_ew[i-1])**2)
        if i <= start_idx:
            vs_start += delta_vs
        vs_end += delta_vs

    delta_tvd = well_tvd[end_idx] - well_tvd[start_idx]
    delta_vs = vs_end - vs_start

    if delta_vs <= 0:
        return 0.0

    return np.degrees(np.arctan(delta_tvd / delta_vs))


def optimize_endpoint_gpu(
    well_data: dict,
    baseline_shift: float,
    tvt_at_max: float,
    avg_angle: float,
    search_range: float = 30.0,
    n_shift_steps: int = 61,
    n_angle_steps: int = 11,
    angle_range: float = 0.5,
    window_length: float = 10.0,
    gr_smooth_window: int = 1,
    pearson_power: float = 1.0,
    pearson_clamp: float = 0.0,
    device: str = 'cuda'
) -> tuple:
    """
    2D Grid search over (end_shift, angle).
    Objective: (1 - pearson^power) * MSE - minimize.

    pearson_power: exponent for pearson (0.3, 0.5, 1, 2, etc.)
    pearson_clamp: min value for pearson (e.g. 0.5 = ignore correlations < 0.5)

    NO CHEATING - start_shift from baseline TVT=const.
    Angles centered around avg_angle (from well trajectory).
    """
    from torch_funcs.projection import calc_horizontal_projection_batch_torch
    from torch_funcs.correlations import pearson_batch_torch, mse_batch_torch
    from torch_funcs.converters import GPU_DTYPE

    mds = np.array(well_data.get('ref_segment_mds', []))

    if len(mds) < 2:
        return baseline_shift, float('inf')

    # Use min(lateral_well_last_md, log_md[-1]) - log may end before trajectory
    lateral_md = float(well_data.get('lateral_well_last_md', mds[-1]))
    log_md_max = float(well_data['log_md'][-1])
    end_md = min(lateral_md, log_md_max)
    start_md = end_md - window_length

    # Prepare well data
    well_md = np.array(well_data.get('well_md', []))
    well_tvd = np.array(well_data.get('well_tvd', []))
    well_ns = np.array(well_data.get('well_ns', []))
    well_ew = np.array(well_data.get('well_ew', []))
    log_md = np.array(well_data.get('log_md', []))
    log_gr = np.array(well_data.get('log_gr', []))
    type_tvd = np.array(well_data.get('type_tvd', []))
    type_gr = np.array(well_data.get('type_gr', []))
    tvd_typewell_shift = float(well_data.get('tvd_typewell_shift', 0.0))

    # Calculate VS
    well_vs = np.zeros(len(well_ns))
    for i in range(1, len(well_ns)):
        well_vs[i] = well_vs[i-1] + np.sqrt((well_ns[i] - well_ns[i-1])**2 + (well_ew[i] - well_ew[i-1])**2)

    # Interpolate GR to well_md and optionally smooth
    gr_at_well_md = np.interp(well_md, log_md, log_gr)
    if gr_smooth_window > 1:
        gr_at_well_md = smooth_gr(gr_at_well_md, gr_smooth_window)

    # Find indices for segment
    start_idx = int(np.searchsorted(well_md, start_md))
    end_idx = int(np.searchsorted(well_md, end_md))
    start_idx = min(start_idx, len(well_md) - 1)
    end_idx = min(end_idx, len(well_md) - 1)

    if start_idx >= end_idx:
        return baseline_shift, float('inf')

    start_vs = well_vs[start_idx]
    end_vs = well_vs[end_idx]

    # Use baseline_shift passed as argument (computed for same end_md)
    # Fix END shift (what we predict), vary START based on angle

    # Generate 2D grid: end_shifts × angles
    # End shift varies in range around baseline_shift (this is what we want to predict!)
    ends_to_test = np.linspace(
        baseline_shift - search_range,
        baseline_shift + search_range,
        n_shift_steps
    )
    angles_to_test = np.linspace(avg_angle - angle_range, avg_angle + angle_range, n_angle_steps)  # degrees

    # Build batch of segments
    segments_list = []
    grid_results = []  # (start_shift, end_shift) pairs

    for end_shift in ends_to_test:
        for angle_deg in angles_to_test:
            # Start = end - tan(angle) * window (angle points from start to end)
            angle_rad = np.radians(angle_deg)
            shift_delta = np.tan(angle_rad) * window_length
            seg_start_shift = end_shift - shift_delta

            segments_list.append([start_idx, end_idx, start_vs, end_vs, seg_start_shift, float(end_shift)])
            grid_results.append((seg_start_shift, float(end_shift)))

    batch_size = len(segments_list)
    segments_np = np.array(segments_list, dtype=np.float32).reshape(batch_size, 1, 6)
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
        well_torch, typewell_torch, segments_torch, tvd_to_typewell_shift=tvd_typewell_shift
    )

    # Get well GR for range - match synt_batch size
    n_points = min(end_idx - start_idx + 1, synt_batch.shape[1])
    well_gr_range = torch.tensor(
        gr_at_well_md[start_idx:start_idx + n_points], dtype=GPU_DTYPE, device=device
    ).unsqueeze(0).expand(batch_size, -1)

    synt_range = synt_batch[:, :n_points]

    # Compute pearson and MSE for all batches
    pearsons = pearson_batch_torch(well_gr_range, synt_range)
    mses = mse_batch_torch(well_gr_range, synt_range)

    # Apply pearson clamp (ignore low correlations)
    if pearson_clamp > 0:
        pearsons_clamped = torch.clamp(pearsons, min=pearson_clamp)
    else:
        pearsons_clamped = pearsons

    # Objective: (1 - pearson^power) * mse - minimize
    if pearson_power != 1.0:
        pearson_term = torch.pow(pearsons_clamped, pearson_power)
    else:
        pearson_term = pearsons_clamped
    objectives = (1 - pearson_term) * mses
    objectives = torch.where(success_mask, objectives, torch.tensor(float('inf'), device=device))

    # If all failed, return baseline (end = baseline_shift, start = baseline - tan(avg_angle) * window)
    if not success_mask.any():
        baseline_start = baseline_shift - np.tan(np.radians(avg_angle)) * window_length
        return baseline_start, baseline_shift, float('inf'), 0.0, float('inf')

    # Find best among successful (minimize)
    best_idx = torch.argmin(objectives).item()
    best_start, best_end = grid_results[best_idx]
    best_objective = objectives[best_idx].item()
    best_pearson = pearsons[best_idx].item()
    best_mse = mses[best_idx].item()

    return best_start, best_end, best_objective, best_pearson, best_mse


def build_interpretation(well_data: dict, landing_md: float, window_start_md: float,
                         end_md: float, optimized_start: float,
                         optimized_end: float) -> dict:
    """
    Build interpretation for StarSteer import.

    Structure:
    1. Reference segments up to landing_md
    2. One long segment from landing_md to window_start_md (connects to optimized_start)
    3. Optimized segment from window_start_md to end_md
    """
    mds = np.array(well_data.get('ref_segment_mds', []))
    start_shifts = np.array(well_data.get('ref_start_shifts', []))
    end_shifts = np.array(well_data.get('ref_shifts', []))

    segments = []

    # 1. Reference segments up to landing_md
    for i in range(len(mds) - 1):
        if mds[i+1] <= landing_md:
            segments.append({
                "startMd": float(mds[i]),
                "endMd": float(mds[i+1]),
                "startShift": float(start_shifts[i]),
                "endShift": float(end_shifts[i])
            })
        elif mds[i] < landing_md:
            # Partial segment - interpolate shift at landing
            ratio = (landing_md - mds[i]) / (mds[i+1] - mds[i])
            shift_at_landing = start_shifts[i] + ratio * (end_shifts[i] - start_shifts[i])
            segments.append({
                "startMd": float(mds[i]),
                "endMd": float(landing_md),
                "startShift": float(start_shifts[i]),
                "endShift": float(shift_at_landing)
            })
            break

    # Get shift at landing from last ref segment
    if segments:
        shift_at_landing = segments[-1]["endShift"]
    else:
        # Use interpretation module for correct interpolation
        shift_at_landing = interpolate_shift_at_md(well_data, landing_md)

    # 2. Long segment from landing to window_start (connects to optimized_start)
    if window_start_md > landing_md:
        segments.append({
            "startMd": float(landing_md),
            "endMd": float(window_start_md),
            "startShift": float(shift_at_landing),
            "endShift": float(optimized_start)
        })

    # 3. Optimized segment from window_start to end
    segments.append({
        "startMd": float(window_start_md),
        "endMd": float(end_md),
        "startShift": float(optimized_start),
        "endShift": float(optimized_end)
    })

    return {"interpretation": {"segments": segments}}


def main():
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument('--window', type=float, default=10.0,
                       help='Window length in meters (default: 10)')
    parser.add_argument('--search-range', type=float, default=30.0,
                       help='Search range ± meters (default: 30)')
    parser.add_argument('--angle-range', type=float, default=0.5,
                       help='Angle range ± degrees (default: 0.5)')
    parser.add_argument('--n-shift-steps', type=int, default=61,
                       help='Number of shift steps (default: 61)')
    parser.add_argument('--n-angle-steps', type=int, default=11,
                       help='Number of angle steps (default: 11)')
    parser.add_argument('--gr-smooth', type=int, default=1,
                       help='GR smoothing window (default: 1 = no smoothing)')
    parser.add_argument('--save-interp', type=str, default=None,
                       help='Directory to save interpretations for StarSteer import')
    args = parser.parse_args()

    # Create output directory if saving interpretations
    save_dir = None
    if args.save_interp:
        save_dir = Path(args.save_interp)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving interpretations to: {save_dir}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print(f"Window: {args.window}m, Search range: ±{args.search_range}m")
    print(f"Angle range: ±{args.angle_range}°, Steps: {args.n_shift_steps}×{args.n_angle_steps} = {args.n_shift_steps * args.n_angle_steps}")
    print(f"GR smooth: {args.gr_smooth}, Objective: (1 - pearson) * MSE")
    print(f"Start shift: from baseline TVT=const (NO CHEATING)")

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

        # Well trajectory angle (HONEST - from landing to end)
        landing_md = well_data.get('detected_start_md') or well_data.get('start_md') or mds[0]
        lateral_md = float(well_data.get('lateral_well_last_md', mds[-1]))
        log_md_max = float(well_data['log_md'][-1])
        end_md = min(lateral_md, log_md_max)
        avg_angle = compute_well_angle(well_data, float(landing_md), end_md)

        # GPU optimization
        optimized_start, optimized_end, best_obj, best_pearson, best_mse = optimize_endpoint_gpu(
            well_data=well_data,
            baseline_shift=baseline_shift,
            tvt_at_max=tvt_at_max,
            avg_angle=avg_angle,
            search_range=args.search_range,
            n_shift_steps=args.n_shift_steps,
            n_angle_steps=args.n_angle_steps,
            angle_range=args.angle_range,
            window_length=args.window,
            gr_smooth_window=args.gr_smooth,
            device=device
        )
        optimized_error = optimized_end - ref_shift_end

        results.append({
            'well': well_name,
            'ref_shift': ref_shift_end,
            'baseline_error': baseline_error,
            'optimized_error': optimized_error,
            'pearson': best_pearson,
            'mse': best_mse,
        })

        print(f"{well_name}: baseline={baseline_error:+.2f}m, opt={optimized_error:+.2f}m, pearson={best_pearson:.3f}, mse={best_mse:.1f}")

        # Save interpretation if requested
        if save_dir:
            window_start_md = end_md - args.window

            interp_data = build_interpretation(
                well_data=well_data,
                landing_md=landing_md,
                window_start_md=window_start_md,
                end_md=end_md,
                optimized_start=optimized_start,
                optimized_end=optimized_end
            )
            interp_file = save_dir / f"{well_name}.json"
            with open(interp_file, 'w') as f:
                json.dump(interp_data, f, indent=2)
            print(f"  Saved: {interp_file}")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    baseline_errors = [r['baseline_error'] for r in results]
    optimized_errors = [r['optimized_error'] for r in results]

    baseline_rmse = np.sqrt(np.mean(np.array(baseline_errors)**2))
    optimized_rmse = np.sqrt(np.mean(np.array(optimized_errors)**2))

    print(f"\nBaseline (TVT=const):  RMSE {baseline_rmse:.2f}m")
    print(f"Optimized:  RMSE {optimized_rmse:.2f}m ({optimized_rmse - baseline_rmse:+.2f}m)")

    improved = sum(1 for b, o in zip(baseline_errors, optimized_errors) if abs(o) < abs(b))
    print(f"\nImproved: {improved}/{len(results)} wells")


if __name__ == '__main__':
    main()
