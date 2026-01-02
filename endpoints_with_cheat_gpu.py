#!/usr/bin/env python3
"""
Endpoint prediction with cheat (GPU version).

Uses known shift_at_start from reference, brute force over endpoint shift.
GPU Pearson correlation + angle penalty.

Result: RMSE 3.32m (vs baseline 6.18m), improved 47/100 wells
Run: python endpoints_with_cheat_gpu.py
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from torch_funcs.converters import GPU_DTYPE
from torch_funcs.projection import calc_horizontal_projection_batch_torch
from torch_funcs.correlations import pearson_batch_torch, mse_batch_torch


def prepare_well_data_torch(well_data: dict, device: str = 'cuda'):
    """Convert well data to torch tensors for GPU processing."""
    well_md = np.array(well_data.get('well_md', []))
    well_tvd = np.array(well_data.get('well_tvd', []))
    log_md = np.array(well_data.get('log_md', []))
    log_gr = np.array(well_data.get('log_gr', []))

    # Interpolate TVD at log positions
    log_tvd = np.interp(log_md, well_md, well_tvd)

    return {
        'md': torch.tensor(log_md, dtype=GPU_DTYPE, device=device),
        'tvd': torch.tensor(log_tvd, dtype=GPU_DTYPE, device=device),
        'value': torch.tensor(log_gr, dtype=GPU_DTYPE, device=device),
    }


def prepare_typewell_data_torch(well_data: dict, device: str = 'cuda'):
    """Convert typewell data to torch tensors."""
    type_tvd = np.array(well_data.get('type_tvd', []))
    type_gr = np.array(well_data.get('type_gr', []))

    return {
        'tvd': torch.tensor(type_tvd, dtype=GPU_DTYPE, device=device),
        'value': torch.tensor(type_gr, dtype=GPU_DTYPE, device=device),
    }


def create_single_segment_torch(
    start_md: float,
    end_md: float,
    start_shift: float,
    end_shift: float,
    well_data: dict,
    device: str = 'cuda'
):
    """
    Create a single segment tensor.
    Format: [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
    """
    well_md = np.array(well_data.get('well_md', []))
    well_tvd = np.array(well_data.get('well_tvd', []))

    # Find indices
    min_md = well_md.min()
    md_step = (well_md.max() - min_md) / (len(well_md) - 1) if len(well_md) > 1 else 1.0

    start_idx = int((start_md - min_md) / md_step)
    end_idx = int((end_md - min_md) / md_step)

    # Get VS (vertical section) = TVD at these points
    start_vs = np.interp(start_md, well_md, well_tvd)
    end_vs = np.interp(end_md, well_md, well_tvd)

    segment = torch.tensor([
        [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
    ], dtype=GPU_DTYPE, device=device)

    return segment


def compute_gr_correlation_gpu(
    well_data_torch: dict,
    typewell_data_torch: dict,
    segment_torch: torch.Tensor,
    device: str = 'cuda'
) -> tuple:
    """
    Compute GR correlation using GPU projection.
    Returns (pearson, mse).
    """
    # Simple shift-based projection (like baseline)
    # For now, just use the end_shift as constant TVT offset
    end_shift = segment_torch[0, 5].item()

    well_tvd = well_data_torch['tvd']
    well_gr = well_data_torch['value']
    type_tvd = typewell_data_torch['tvd']
    type_gr = typewell_data_torch['value']

    # TVT = TVD - shift
    tvt = well_tvd - end_shift

    # Interpolate typewell GR at TVT positions
    # Use simple torch interpolation
    tvt_clamped = torch.clamp(tvt, type_tvd.min(), type_tvd.max())

    # Find indices for interpolation
    indices = torch.searchsorted(type_tvd, tvt_clamped)
    indices = torch.clamp(indices, 1, len(type_tvd) - 1)

    # Linear interpolation
    x0 = type_tvd[indices - 1]
    x1 = type_tvd[indices]
    y0 = type_gr[indices - 1]
    y1 = type_gr[indices]

    t = (tvt_clamped - x0) / (x1 - x0 + 1e-8)
    typewell_gr_interp = y0 + t * (y1 - y0)

    # Compute Pearson and MSE
    well_gr_norm = well_gr - well_gr.mean()
    type_gr_norm = typewell_gr_interp - typewell_gr_interp.mean()

    pearson = (well_gr_norm * type_gr_norm).sum() / (
        torch.sqrt((well_gr_norm ** 2).sum() * (type_gr_norm ** 2).sum()) + 1e-8
    )

    mse = ((well_gr - typewell_gr_interp) ** 2).mean()

    return pearson.item(), mse.item()


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


def optimize_endpoint_gpu(
    well_data: dict,
    well_data_torch: dict,
    typewell_data_torch: dict,
    baseline_shift: float,
    avg_angle: float,
    search_range: float = 10.0,
    n_steps: int = 41,
    window_length: float = 60.0,
    device: str = 'cuda'
) -> tuple:
    """Grid search using GPU correlation."""
    mds = np.array(well_data.get('ref_segment_mds', []))
    shifts = np.array(well_data.get('ref_shifts', []))

    if len(mds) < 2:
        return baseline_shift, 0.0

    end_md = mds[-1]
    start_md = end_md - window_length
    shift_at_start = np.interp(start_md, mds, shifts)

    best_objective = -float('inf')
    best_shift = baseline_shift

    shifts_to_test = np.linspace(
        baseline_shift - search_range,
        baseline_shift + search_range,
        n_steps
    )

    for test_shift in shifts_to_test:
        # Compute angle
        shift_delta = test_shift - shift_at_start
        angle_rad = np.arctan(shift_delta / window_length)
        angle_deg = np.degrees(angle_rad)
        angle_penalty = (angle_deg - avg_angle) ** 2

        # Create segment and compute GPU correlation
        segment = create_single_segment_torch(
            start_md, end_md, shift_at_start, test_shift, well_data, device
        )

        pearson, mse = compute_gr_correlation_gpu(
            well_data_torch, typewell_data_torch, segment, device
        )

        # Objective: correlation - angle_penalty * weight
        objective = pearson - 0.01 * angle_penalty

        if objective > best_objective:
            best_objective = objective
            best_shift = test_shift

    return best_shift, best_objective


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

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

        # Prepare GPU data
        well_data_torch = prepare_well_data_torch(well_data, device)
        typewell_data_torch = prepare_typewell_data_torch(well_data, device)

        # GPU optimization
        optimized_shift, best_obj = optimize_endpoint_gpu(
            well_data=well_data,
            well_data_torch=well_data_torch,
            typewell_data_torch=typewell_data_torch,
            baseline_shift=baseline_shift,
            avg_angle=well_angle,
            search_range=10.0,
            n_steps=41,
            window_length=60.0,
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
    print("\n" + "="*60)
    print("SUMMARY (GPU)")
    print("="*60)

    baseline_errors = [r['baseline_error'] for r in results]
    optimized_errors = [r['optimized_error'] for r in results]

    baseline_rmse = np.sqrt(np.mean(np.array(baseline_errors)**2))
    optimized_rmse = np.sqrt(np.mean(np.array(optimized_errors)**2))

    print(f"\nBaseline (TVT=const):  RMSE {baseline_rmse:.2f}m")
    print(f"Optimized (GPU corr):  RMSE {optimized_rmse:.2f}m ({optimized_rmse - baseline_rmse:+.2f}m)")

    improved = sum(1 for b, o in zip(baseline_errors, optimized_errors) if abs(o) < abs(b))
    print(f"\nImproved: {improved}/{len(results)} wells")


if __name__ == '__main__':
    main()
