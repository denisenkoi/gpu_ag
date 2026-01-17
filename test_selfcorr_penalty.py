#!/usr/bin/env python3
"""
Test self-correlation penalty on degraded wells.
Grid search over threshold and weight.
"""
import os
import sys
import torch
import numpy as np
import argparse
from collections import defaultdict

# Set env before imports
os.environ.setdefault('USE_PSEUDO_TYPELOG', 'true')

from gpu_executor import GPUBlockOptimizer


def compute_bin_std_torch(tvt: torch.Tensor, gr: torch.Tensor, bin_size: float = 0.10):
    """Compute std(bin_means) for single solution."""
    if len(tvt) < 10:
        return 0.0

    tvt_min = tvt.min()
    bin_idx = ((tvt - tvt_min) / bin_size).long()
    n_bins = int(bin_idx.max().item()) + 1

    # Scatter add for sums and counts
    bin_sums = torch.zeros(n_bins, device=tvt.device, dtype=gr.dtype)
    bin_counts = torch.zeros(n_bins, device=tvt.device, dtype=gr.dtype)

    bin_sums.scatter_add_(0, bin_idx, gr)
    bin_counts.scatter_add_(0, bin_idx, torch.ones_like(gr))

    # Compute mean per bin
    valid_mask = bin_counts > 0
    if valid_mask.sum() < 5:
        return 0.0

    bin_means = bin_sums[valid_mask] / bin_counts[valid_mask]
    return float(bin_means.std())


def run_optimization_with_penalty(
    well_name: str,
    angle_range: float = 2.0,
    angle_step: float = 0.2,
    mse_weight: float = 5.0,
    selfcorr_threshold: float = 0.0,  # 0 = disabled
    selfcorr_weight: float = 0.0,
    device: str = 'cuda:1'
):
    """
    Run optimization with self-correlation penalty.

    Penalty = max(0, threshold - std(bin_means)) * weight
    """
    # Load dataset
    ds = torch.load('data/wells_limited_pseudo.pt', weights_only=False)
    if well_name not in ds:
        print(f"Well {well_name} not found")
        return None

    well_data = ds[well_name]

    # Create optimizer
    optimizer = GPUBlockOptimizer(
        well_data=well_data,
        angle_range=angle_range,
        angle_step=angle_step,
        mse_weight=mse_weight,
        device=device,
        chunk_size=100000
    )

    # Run optimization
    result = optimizer.optimize_full_well()

    if result is None:
        return None

    # Compute final TVT and self-correlation metric
    # Get TVT from result interpretation
    well_md = well_data['well_md'].numpy()
    well_tvd = well_data['well_tvd'].numpy()

    tvt = np.full_like(well_tvd, np.nan)
    for seg in result['interpretation']:
        md_start, md_end = seg['startMd'], seg['endMd']
        start_shift, end_shift = seg['startShift'], seg['endShift']
        mask = (well_md >= md_start) & (well_md <= md_end)
        if not mask.any():
            continue
        seg_md = well_md[mask]
        seg_tvd = well_tvd[mask]
        ratio = (seg_md - md_start) / (md_end - md_start) if md_end > md_start else np.zeros_like(seg_md)
        seg_shift = start_shift + ratio * (end_shift - start_shift)
        tvt[mask] = seg_tvd - seg_shift

    # Filter by landing_end_dls
    landing_end = well_data.get('landing_end_dls')
    valid = ~np.isnan(tvt)
    if landing_end:
        valid = valid & (well_md >= landing_end)

    tvt_valid = tvt[valid]

    # Get GR (smoothed)
    from scipy.signal import savgol_filter
    log_md = well_data['log_md'].numpy()
    log_gr = well_data['log_gr'].numpy()
    well_gr = np.interp(well_md, log_md, log_gr)
    if len(well_gr) > 51:
        well_gr = savgol_filter(well_gr, 51, 2)
    gr_valid = well_gr[valid]

    # Compute std(bin_means)
    tvt_t = torch.tensor(tvt_valid, dtype=torch.float32)
    gr_t = torch.tensor(gr_valid, dtype=torch.float32)
    std_bins = compute_bin_std_torch(tvt_t, gr_t, bin_size=0.10)

    result['std_bins'] = std_bins
    result['tvt_range'] = float(tvt_valid.max() - tvt_valid.min()) if len(tvt_valid) > 0 else 0.0

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wells', nargs='+', default=['Well1675~EGFDL', 'Well239~EGFDL'])
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()

    # Grid search parameters
    thresholds = [0, 8, 10, 12, 15]  # 0 = disabled
    weights = [0, 0.1, 0.2, 0.5]

    print("Grid search: self-correlation penalty")
    print(f"Wells: {args.wells}")
    print()

    results = []

    for well in args.wells:
        print(f"\n{'='*60}")
        print(f"Well: {well}")
        print(f"{'='*60}")

        # Baseline (no penalty)
        print("\nBaseline (no penalty)...")
        base = run_optimization_with_penalty(
            well, angle_range=2.0, angle_step=0.2, mse_weight=5.0,
            selfcorr_threshold=0, selfcorr_weight=0, device=args.device
        )

        if base:
            print(f"  Error: {base['opt_error']:+.2f}m")
            print(f"  std(bins): {base['std_bins']:.2f}")
            print(f"  TVT range: {base['tvt_range']:.2f}m")
            results.append({
                'well': well, 'threshold': 0, 'weight': 0,
                'error': base['opt_error'], 'std_bins': base['std_bins']
            })

    print("\n\nSummary:")
    print("-"*60)
    for r in results:
        print(f"{r['well']}: error={r['error']:+.2f}m, std={r['std_bins']:.2f}")


if __name__ == '__main__':
    main()
