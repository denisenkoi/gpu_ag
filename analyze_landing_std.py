#!/usr/bin/env python3
"""
Analyze STD(bin_means) for landing vs post-landing parts of REF interpretations.
Uses preprocessing module for consistent normalization.
"""

import os
import sys
import torch
import numpy as np
from scipy.signal import savgol_filter

# Add cpu_baseline to path
sys.path.insert(0, '/mnt/e/Projects/Rogii/gpu_ag')
from cpu_baseline.preprocessing import prepare_data


def compute_tvt_from_ref(well_md, well_tvd, ref_segment_mds, ref_start_shifts, ref_shifts):
    """
    Compute TVT using REF interpretation shifts.
    TVT = TVD - shift
    """
    n_pts = len(well_md)
    tvt = np.full(n_pts, np.nan)
    n_segs = len(ref_segment_mds)

    for i in range(n_segs):
        start_md = ref_segment_mds[i]
        if i < n_segs - 1:
            end_md = ref_segment_mds[i + 1]
        else:
            end_md = well_md.max() + 1.0

        start_shift = ref_start_shifts[i]
        end_shift = ref_shifts[i]

        mask = (well_md >= start_md) & (well_md < end_md)
        if not np.any(mask):
            continue

        md_pts = well_md[mask]
        if end_md > start_md:
            ratio = (md_pts - start_md) / (end_md - start_md)
        else:
            ratio = np.zeros_like(md_pts)
        shifts = start_shift + ratio * (end_shift - start_shift)

        tvd_pts = well_tvd[mask]
        tvt[mask] = tvd_pts - shifts

    return tvt


def compute_std_bin_means(tvt, gr, bin_size=0.10, smooth_window=51):
    """
    Compute std(bin_means) for GR projected onto TVT axis.
    Uses savgol filter with window=51 (optimal from grid search).
    """
    valid = ~np.isnan(tvt)
    if valid.sum() < 10:
        return np.nan, 0

    tvt_valid = tvt[valid]
    gr_valid = gr[valid]

    # Smooth GR with optimal parameters
    if smooth_window > 1 and len(gr_valid) >= smooth_window:
        gr_smooth = savgol_filter(gr_valid, smooth_window, 3)
    else:
        gr_smooth = gr_valid

    # Create bins
    tvt_min, tvt_max = tvt_valid.min(), tvt_valid.max()
    n_bins = int((tvt_max - tvt_min) / bin_size) + 1
    if n_bins < 5:
        return np.nan, n_bins

    # Bin means
    bin_idx = ((tvt_valid - tvt_min) / bin_size).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    np.add.at(bin_sums, bin_idx, gr_smooth)
    np.add.at(bin_counts, bin_idx, 1)

    valid_bins = bin_counts > 0
    if valid_bins.sum() < 5:
        return np.nan, valid_bins.sum()

    bin_means = bin_sums[valid_bins] / bin_counts[valid_bins]
    return np.std(bin_means), valid_bins.sum()


def main():
    print("Loading dataset...")
    dataset = torch.load('/mnt/e/Projects/Rogii/gpu_ag/data/wells_limited_pseudo.pt', weights_only=False)

    print(f"\n{'Well':<25} {'Landing STD':>12} {'Post-land STD':>14} {'Ratio':>8} {'Land bins':>10} {'Post bins':>10}")
    print("=" * 85)

    results = []

    for well_name, well_data in sorted(dataset.items()):
        # Use preprocessing module for consistent normalization
        prepared = prepare_data(well_data, use_pseudo=False, normalize_0_100=True)

        well_md = prepared.well_md
        well_tvd = prepared.well_tvd
        well_gr = prepared.well_gr  # Normalized 0-100

        # Get landing_end_dls
        landing_end_md = well_data.get('landing_end_dls')
        if landing_end_md is None:
            continue
        landing_end_md = float(landing_end_md)

        # Get REF interpretation
        ref_segment_mds = well_data.get('ref_segment_mds')
        ref_start_shifts = well_data.get('ref_start_shifts')
        ref_shifts = well_data.get('ref_shifts')

        if ref_segment_mds is None or len(ref_segment_mds) == 0:
            continue

        ref_segment_mds = np.asarray(ref_segment_mds)
        ref_start_shifts = np.asarray(ref_start_shifts)
        ref_shifts = np.asarray(ref_shifts)

        # Compute TVT for all points using REF interpretation
        tvt = compute_tvt_from_ref(well_md, well_tvd, ref_segment_mds, ref_start_shifts, ref_shifts)

        # Split into landing and post-landing
        landing_mask = well_md <= landing_end_md
        post_mask = well_md > landing_end_md

        # Compute STD for each part (bin=10cm, window=51 - optimal params)
        if landing_mask.sum() > 0:
            std_landing, bins_landing = compute_std_bin_means(tvt[landing_mask], well_gr[landing_mask])
        else:
            std_landing, bins_landing = np.nan, 0

        if post_mask.sum() > 0:
            std_post, bins_post = compute_std_bin_means(tvt[post_mask], well_gr[post_mask])
        else:
            std_post, bins_post = np.nan, 0

        # Ratio
        if not np.isnan(std_landing) and not np.isnan(std_post) and std_landing > 0:
            ratio = std_post / std_landing
        else:
            ratio = np.nan

        results.append({
            'well': well_name,
            'std_landing': std_landing,
            'std_post': std_post,
            'ratio': ratio,
            'bins_landing': bins_landing,
            'bins_post': bins_post
        })

        # Print row
        std_l_str = f"{std_landing:.2f}" if not np.isnan(std_landing) else "N/A"
        std_p_str = f"{std_post:.2f}" if not np.isnan(std_post) else "N/A"
        ratio_str = f"{ratio:.2f}" if not np.isnan(ratio) else "N/A"

        print(f"{well_name:<25} {std_l_str:>12} {std_p_str:>14} {ratio_str:>8} {bins_landing:>10} {bins_post:>10}")

    # Summary statistics
    print("\n" + "=" * 85)
    print("SUMMARY STATISTICS (normalized 0-100, savgol window=51, bin=10cm)")
    print("=" * 85)

    valid_results = [r for r in results if not np.isnan(r['std_landing']) and not np.isnan(r['std_post'])]

    if valid_results:
        std_landings = [r['std_landing'] for r in valid_results]
        std_posts = [r['std_post'] for r in valid_results]
        ratios = [r['ratio'] for r in valid_results if not np.isnan(r['ratio'])]

        print(f"\nLanding STD:     mean={np.mean(std_landings):.2f}, std={np.std(std_landings):.2f}, min={np.min(std_landings):.2f}, max={np.max(std_landings):.2f}")
        print(f"Post-land STD:   mean={np.mean(std_posts):.2f}, std={np.std(std_posts):.2f}, min={np.min(std_posts):.2f}, max={np.max(std_posts):.2f}")
        print(f"Ratio (post/land): mean={np.mean(ratios):.2f}, std={np.std(ratios):.2f}, min={np.min(ratios):.2f}, max={np.max(ratios):.2f}")

        print(f"\nLanding STD percentiles: p10={np.percentile(std_landings, 10):.2f}, p25={np.percentile(std_landings, 25):.2f}, p50={np.percentile(std_landings, 50):.2f}, p75={np.percentile(std_landings, 75):.2f}, p90={np.percentile(std_landings, 90):.2f}")
        print(f"Post-land STD percentiles: p10={np.percentile(std_posts, 10):.2f}, p25={np.percentile(std_posts, 25):.2f}, p50={np.percentile(std_posts, 50):.2f}, p75={np.percentile(std_posts, 75):.2f}, p90={np.percentile(std_posts, 90):.2f}")

        post_higher = sum(1 for r in valid_results if r['std_post'] > r['std_landing'])
        print(f"\nPost-land STD > Landing STD: {post_higher}/{len(valid_results)} ({100*post_higher/len(valid_results):.1f}%)")


if __name__ == '__main__':
    main()
