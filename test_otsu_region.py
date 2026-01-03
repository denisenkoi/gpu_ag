#!/usr/bin/env python3
"""
Test endpoint optimization on OTSU-selected region vs fixed window.

Compares:
1. Fixed window (last 200m)
2. OTSU RegionFinder region (200m around best peak density)
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from peak_detectors import OtsuPeakDetector, RegionFinder
from torch_funcs.projection import calc_horizontal_projection_batch_torch
from torch_funcs.correlations import pearson_batch_torch, mse_batch_torch
from torch_funcs.converters import GPU_DTYPE
from numpy_funcs.interpretation import interpolate_shift_at_md


def compute_baseline_shift(well_data: dict, end_md: float) -> tuple:
    """Compute TVT=const baseline prediction."""
    mds = np.array(well_data.get('ref_segment_mds', []))
    start_shifts = np.array(well_data.get('ref_start_shifts', []))
    well_md = np.array(well_data.get('well_md', []))
    well_tvd = np.array(well_data.get('well_tvd', []))

    if len(mds) < 2:
        return 0.0, 0.0

    tvds_at_segments = np.interp(mds, well_md, well_tvd)
    max_idx = np.argmax(tvds_at_segments)
    max_tvd = tvds_at_segments[max_idx]
    shift_at_max_tvd = start_shifts[max_idx]
    tvt_at_max = max_tvd - shift_at_max_tvd

    tvd_end = float(np.interp(end_md, well_md, well_tvd))
    predicted_shift = tvd_end - tvt_at_max

    return predicted_shift, tvt_at_max


def optimize_on_region(
    well_data: dict,
    start_md: float,
    end_md: float,
    baseline_shift: float,
    search_range: float = 30.0,
    n_shift_steps: int = 61,
    n_angle_steps: int = 11,
    angle_range: float = 1.0,
    device: str = 'cuda'
) -> tuple:
    """
    Optimize endpoint on specific region [start_md, end_md].

    Returns: (best_end_shift, best_objective, best_pearson)
    """
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

    # Interpolate GR to well_md
    gr_at_well_md = np.interp(well_md, log_md, log_gr)

    # Find indices for region
    start_idx = int(np.searchsorted(well_md, start_md))
    end_idx = int(np.searchsorted(well_md, end_md))
    start_idx = max(0, min(start_idx, len(well_md) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(well_md) - 1))

    if start_idx >= end_idx:
        return baseline_shift, float('inf'), 0.0

    start_vs = well_vs[start_idx]
    end_vs = well_vs[end_idx]
    window_length = end_md - start_md

    # Average angle from trajectory
    delta_tvd = well_tvd[end_idx] - well_tvd[start_idx]
    delta_vs = end_vs - start_vs
    avg_angle = np.degrees(np.arctan(delta_tvd / delta_vs)) if delta_vs > 0 else 0.0

    # Generate 2D grid
    ends_to_test = np.linspace(
        baseline_shift - search_range,
        baseline_shift + search_range,
        n_shift_steps
    )
    angles_to_test = np.linspace(avg_angle - angle_range, avg_angle + angle_range, n_angle_steps)

    segments_list = []
    grid_results = []

    for end_shift in ends_to_test:
        for angle_deg in angles_to_test:
            angle_rad = np.radians(angle_deg)
            shift_delta = np.tan(angle_rad) * window_length
            seg_start_shift = end_shift - shift_delta
            segments_list.append([start_idx, end_idx, start_vs, end_vs, seg_start_shift, float(end_shift)])
            grid_results.append((seg_start_shift, float(end_shift)))

    batch_size = len(segments_list)
    segments_np = np.array(segments_list, dtype=np.float32).reshape(batch_size, 1, 6)
    segments_torch = torch.tensor(segments_np, dtype=GPU_DTYPE, device=device)

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

    success_mask, tvt_batch, synt_batch, first_idx = calc_horizontal_projection_batch_torch(
        well_torch, typewell_torch, segments_torch, tvd_to_typewell_shift=tvd_typewell_shift
    )

    n_points = min(end_idx - start_idx + 1, synt_batch.shape[1])
    well_gr_range = torch.tensor(
        gr_at_well_md[start_idx:start_idx + n_points], dtype=GPU_DTYPE, device=device
    ).unsqueeze(0).expand(batch_size, -1)

    synt_range = synt_batch[:, :n_points]

    pearson = pearson_batch_torch(well_gr_range, synt_range)
    mse = mse_batch_torch(well_gr_range, synt_range)

    # Objective: (1 - pearson) * MSE
    objective = (1.0 - pearson) * mse

    # Mask failed projections
    success_1d = success_mask.view(-1) if success_mask.dim() > 1 else success_mask
    objective = torch.where(success_1d, objective, torch.tensor(float('inf'), device=device))

    # Find best
    best_idx = torch.argmin(objective).item()
    best_end_shift = grid_results[best_idx][1]
    best_obj = objective[best_idx].item()
    best_pearson = pearson[best_idx].item()

    return best_end_shift, best_obj, best_pearson


def main():
    print("=" * 60)
    print("OTSU Region vs Fixed Window - Endpoint Optimization Test")
    print("=" * 60)

    ds = torch.load('dataset/gpu_ag_dataset.pt', weights_only=False)
    well_name = 'Well1221~EGFDL'
    well = ds[well_name]

    log_md = well['log_md'].numpy()
    log_gr = well['log_gr'].numpy()

    # End MD (where we predict)
    lateral_md = float(well.get('lateral_well_last_md', log_md[-1]))
    log_md_max = float(log_md[-1])
    end_md = min(lateral_md, log_md_max)

    # Reference shift at end
    ref_shift_end = interpolate_shift_at_md(well, end_md)

    # Baseline
    baseline_shift, tvt_at_max = compute_baseline_shift(well, end_md)
    baseline_error = baseline_shift - ref_shift_end

    print(f"\nWell: {well_name}")
    print(f"End MD: {end_md:.1f}m")
    print(f"Reference shift: {ref_shift_end:.2f}m")
    print(f"Baseline shift: {baseline_shift:.2f}m (error: {baseline_error:+.2f}m)")

    # === Fixed window (last 200m) ===
    window_fixed = 200.0
    start_md_fixed = end_md - window_fixed

    print(f"\n--- Fixed Window (last {window_fixed:.0f}m) ---")
    print(f"Region: {start_md_fixed:.1f} - {end_md:.1f}m")

    opt_shift_fixed, obj_fixed, pearson_fixed = optimize_on_region(
        well, start_md_fixed, end_md, baseline_shift
    )
    error_fixed = opt_shift_fixed - ref_shift_end
    print(f"Optimized shift: {opt_shift_fixed:.2f}m (error: {error_fixed:+.2f}m)")
    print(f"Pearson: {pearson_fixed:.3f}")

    # === OTSU Region ===
    print(f"\n--- OTSU RegionFinder (200m window) ---")

    # Find best region in last 600m
    search_start_md = log_md_max - 600
    detector = OtsuPeakDetector()
    finder = RegionFinder(detector, search_fraction=1.0)
    result = finder.find_best_region(log_gr, log_md, region_length_m=200.0, start_md=search_start_md)

    # Region from center
    center_md = result.best_md
    start_md_otsu = center_md - 100
    end_md_otsu = center_md + 100

    print(f"OTSU center: {center_md:.1f}m (score: {result.score:.2f})")
    print(f"Region: {start_md_otsu:.1f} - {end_md_otsu:.1f}m")
    print(f"Significant peaks in search zone: {len(result.significant_peaks)}")

    # Baseline for OTSU region end
    baseline_shift_otsu, _ = compute_baseline_shift(well, end_md_otsu)
    ref_shift_otsu = interpolate_shift_at_md(well, end_md_otsu)

    opt_shift_otsu, obj_otsu, pearson_otsu = optimize_on_region(
        well, start_md_otsu, end_md_otsu, baseline_shift_otsu
    )
    error_otsu = opt_shift_otsu - ref_shift_otsu
    print(f"Optimized shift: {opt_shift_otsu:.2f}m (error: {error_otsu:+.2f}m)")
    print(f"Pearson: {pearson_otsu:.3f}")

    # === Comparison ===
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"{'Method':<25} {'Error':>10} {'Pearson':>10}")
    print("-" * 50)
    print(f"{'Baseline (TVT=const)':<25} {baseline_error:>+10.2f}m {'-':>10}")
    print(f"{'Fixed 200m window':<25} {error_fixed:>+10.2f}m {pearson_fixed:>10.3f}")
    print(f"{'OTSU region 200m':<25} {error_otsu:>+10.2f}m {pearson_otsu:>10.3f}")


if __name__ == '__main__':
    main()
