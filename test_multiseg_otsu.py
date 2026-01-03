#!/usr/bin/env python3
"""
Test multi-segment optimization on OTSU-selected region.

Uses SmartSegmenter to find boundaries, then optimizes angles with penalty.
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from smart_segmenter import SmartSegmenter, detect_peaks_valleys
from numpy_funcs.interpretation import interpolate_shift_at_md


def get_trajectory_angle(well_tvd, well_md, start_idx, end_idx):
    """Get average trajectory angle from TVD/MD."""
    delta_tvd = well_tvd[end_idx] - well_tvd[start_idx]
    delta_md = well_md[end_idx] - well_md[start_idx]
    if delta_md < 0.1:
        return 0.0
    return np.degrees(np.arctan(delta_tvd / delta_md))


def project_segments(segments, well_md, well_tvd, type_tvd, type_gr, tvd_shift):
    """
    Project multiple segments and return synthetic GR.

    segments: list of (start_idx, end_idx, start_shift, end_shift)
    """
    synthetic = np.zeros(len(well_md))
    type_tvd_shifted = type_tvd + tvd_shift

    for start_idx, end_idx, start_shift, end_shift in segments:
        if end_idx <= start_idx:
            continue

        seg_md = well_md[start_idx:end_idx]
        seg_tvd = well_tvd[start_idx:end_idx]

        # Interpolate shift
        md_start = well_md[start_idx]
        md_end = well_md[end_idx - 1]
        if md_end > md_start:
            ratio = (seg_md - md_start) / (md_end - md_start)
        else:
            ratio = np.zeros_like(seg_md)
        seg_shift = start_shift + ratio * (end_shift - start_shift)

        # TVT and project
        tvt = seg_tvd - seg_shift
        seg_gr = np.interp(tvt, type_tvd_shifted, type_gr)
        synthetic[start_idx:end_idx] = seg_gr

    return synthetic


def pearson(a, b):
    if len(a) < 3:
        return 0.0
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom < 1e-10:
        return 0.0
    return np.sum(a * b) / denom


def optimize_multiseg(
    segment_indices,
    baseline_shift,
    trajectory_angle,
    well_md, well_tvd, well_gr,
    type_tvd, type_gr, tvd_shift,
    angle_range=1.0,
    angle_steps=11,
):
    """
    Optimize angles for multiple connected segments.

    Each segment has its own angle, constrained to trajectory_angle ± angle_range.
    Segments are connected: end_shift[i] = start_shift[i+1].
    """
    n_seg = len(segment_indices)

    # Generate angle candidates for each segment
    angles = np.linspace(
        trajectory_angle - angle_range,
        trajectory_angle + angle_range,
        angle_steps
    )

    # Start with baseline shift at first segment start
    first_start_idx = segment_indices[0][0]
    first_start_shift = baseline_shift

    # Full zone for Pearson calculation
    last_end_idx = segment_indices[-1][1]
    zone_gr = well_gr[first_start_idx:last_end_idx]

    best_pearson = -1
    best_angles = [trajectory_angle] * n_seg
    best_segments = []

    # Grid search over all angle combinations
    # For N segments, this is O(angle_steps^N) - limit to max 5 segments
    from itertools import product

    for angle_combo in product(range(angle_steps), repeat=n_seg):
        seg_angles = [angles[i] for i in angle_combo]

        # Build segments with connected shifts
        segments = []
        current_shift = first_start_shift

        for i, (start_idx, end_idx) in enumerate(segment_indices):
            seg_md_len = well_md[end_idx] - well_md[start_idx]
            seg_angle = seg_angles[i]
            shift_delta = np.tan(np.radians(seg_angle)) * seg_md_len
            end_shift = current_shift + shift_delta

            segments.append((start_idx, end_idx, current_shift, end_shift))
            current_shift = end_shift  # Connect to next segment

        # Project and calculate Pearson
        synthetic = project_segments(
            segments, well_md, well_tvd, type_tvd, type_gr, tvd_shift
        )
        zone_synt = synthetic[first_start_idx:last_end_idx]
        p = pearson(zone_gr, zone_synt)

        if p > best_pearson:
            best_pearson = p
            best_angles = seg_angles.copy()
            best_segments = segments.copy()

    return best_pearson, best_angles, best_segments


def find_otsu_zone(log_gr, log_md, search_fraction=0.33, window_m=200.0):
    """
    Find best OTSU zone in last search_fraction of well.
    Returns (zone_start_md, zone_end_md).
    """
    from peak_detectors import OtsuPeakDetector, RegionFinder

    detector = OtsuPeakDetector()
    finder = RegionFinder(detector, search_fraction=search_fraction)
    result = finder.find_best_region(log_gr, log_md, region_length_m=window_m)

    # Region around center
    center_md = result.best_md
    zone_start = center_md - window_m / 2
    zone_end = center_md + window_m / 2

    # Clamp to log bounds
    zone_start = max(zone_start, log_md[0])
    zone_end = min(zone_end, log_md[-1])

    return zone_start, zone_end


def process_well(well_name, well, settings, angle_range=1.0, angle_steps=11):
    """
    Process single well and return results dict.
    """
    # Extract data
    well_md = well['well_md'].numpy() if hasattr(well['well_md'], 'numpy') else np.array(well['well_md'])
    well_tvd = well['well_tvd'].numpy() if hasattr(well['well_tvd'], 'numpy') else np.array(well['well_tvd'])
    log_md = well['log_md'].numpy() if hasattr(well['log_md'], 'numpy') else np.array(well['log_md'])
    log_gr = well['log_gr'].numpy() if hasattr(well['log_gr'], 'numpy') else np.array(well['log_gr'])
    type_tvd = well['type_tvd'].numpy() if hasattr(well['type_tvd'], 'numpy') else np.array(well['type_tvd'])
    type_gr = well['type_gr'].numpy() if hasattr(well['type_gr'], 'numpy') else np.array(well['type_gr'])
    tvd_shift = float(well.get('tvd_typewell_shift', 0.0))

    # Interpolate GR to well_md
    well_gr = np.interp(well_md, log_md, log_gr)

    # Find OTSU zone
    try:
        otsu_start_md, otsu_end_md = find_otsu_zone(log_gr, log_md, search_fraction=0.33, window_m=200.0)
    except Exception as e:
        return {'error': f'OTSU failed: {e}'}

    # Get baseline shift at zone start
    baseline_shift = interpolate_shift_at_md(well, otsu_start_md)

    # Get trajectory angle in zone
    start_idx = int(np.searchsorted(well_md, otsu_start_md))
    end_idx = int(np.searchsorted(well_md, otsu_end_md))

    if end_idx - start_idx < 50:
        return {'error': 'Zone too short'}

    traj_angle = get_trajectory_angle(well_tvd, well_md, start_idx, end_idx)

    # Use SmartSegmenter to find boundaries
    zone_data = {
        'well_md': torch.tensor(well_md),
        'log_md': torch.tensor(log_md),
        'log_gr': torch.tensor(log_gr),
        'start_md': otsu_start_md,
        'detected_start_md': otsu_start_md,
    }

    try:
        segmenter = SmartSegmenter(zone_data, settings)

        all_boundaries = []
        while not segmenter.is_finished:
            result = segmenter.next_slice()
            for bp in result.boundaries:
                if bp.md <= otsu_end_md:
                    all_boundaries.append(bp)
            if result.is_final or (result.boundaries and result.boundaries[-1].md >= otsu_end_md):
                break
    except Exception as e:
        return {'error': f'SmartSegmenter failed: {e}'}

    # Create segment indices (max 5 segments)
    segment_indices = []
    prev_idx = start_idx

    for bp in all_boundaries[:4]:
        bp_idx = int(np.searchsorted(well_md, bp.md))
        if bp_idx > prev_idx:
            segment_indices.append((prev_idx, bp_idx))
            prev_idx = bp_idx

    if prev_idx < end_idx:
        segment_indices.append((prev_idx, end_idx))

    if not segment_indices:
        segment_indices = [(start_idx, end_idx)]

    n_segments = len(segment_indices)

    # Reference shift at zone end
    ref_shift_end = interpolate_shift_at_md(well, otsu_end_md)

    # Optimize multi-segment
    try:
        multi_pearson, multi_angles, multi_segments = optimize_multiseg(
            segment_indices,
            baseline_shift,
            traj_angle,
            well_md, well_tvd, well_gr,
            type_tvd, type_gr, tvd_shift,
            angle_range=angle_range,
            angle_steps=angle_steps,
        )
        multi_end_shift = multi_segments[-1][3] if multi_segments else baseline_shift
        multi_error = multi_end_shift - ref_shift_end
    except Exception as e:
        return {'error': f'Multi-seg optimization failed: {e}'}

    # Optimize single segment
    try:
        single_seg = [(start_idx, end_idx)]
        single_pearson, single_angles, single_segments = optimize_multiseg(
            single_seg,
            baseline_shift,
            traj_angle,
            well_md, well_tvd, well_gr,
            type_tvd, type_gr, tvd_shift,
            angle_range=angle_range,
            angle_steps=21,
        )
        single_end_shift = single_segments[0][3] if single_segments else baseline_shift
        single_error = single_end_shift - ref_shift_end
    except Exception as e:
        single_pearson, single_error = 0.0, 999.0

    return {
        'well': well_name,
        'n_segments': n_segments,
        'zone_start': otsu_start_md,
        'zone_end': otsu_end_md,
        'traj_angle': traj_angle,
        'ref_shift_end': ref_shift_end,
        'single_pearson': single_pearson,
        'single_error': single_error,
        'multi_pearson': multi_pearson,
        'multi_error': multi_error,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--well', type=str, help='Single well to process')
    parser.add_argument('--all', action='store_true', help='Process all wells')
    args = parser.parse_args()

    print("=" * 60)
    print("Multi-Segment Optimization on OTSU Region")
    print("=" * 60)

    ds = torch.load('dataset/gpu_ag_dataset.pt', weights_only=False)

    settings = {
        'min_segment_length': 30.0,
        'overlap_segments': 0,
        'pelt_penalty': 10.0,
        'min_distance': 20.0,
        'segments_count': 5,
    }

    if args.well:
        wells = [args.well]
    elif args.all:
        wells = list(ds.keys())
    else:
        wells = list(ds.keys())  # Default: all

    print(f"\nProcessing {len(wells)} wells...")
    print(f"Settings: angle_range=±1°, max_segments=5")
    print()

    results = []
    errors = []

    for i, well_name in enumerate(wells):
        well = ds[well_name]
        result = process_well(well_name, well, settings)

        if 'error' in result:
            errors.append((well_name, result['error']))
            print(f"[{i+1:3d}/{len(wells)}] {well_name}: ERROR - {result['error']}")
        else:
            results.append(result)
            improved = abs(result['multi_error']) < abs(result['single_error'])
            marker = '✓' if improved else ' '
            print(f"[{i+1:3d}/{len(wells)}] {well_name}: "
                  f"1seg={result['single_error']:+.2f}m, "
                  f"{result['n_segments']}seg={result['multi_error']:+.2f}m {marker}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results:
        single_errors = np.array([r['single_error'] for r in results])
        multi_errors = np.array([r['multi_error'] for r in results])

        single_rmse = np.sqrt(np.mean(single_errors**2))
        multi_rmse = np.sqrt(np.mean(multi_errors**2))

        single_mae = np.mean(np.abs(single_errors))
        multi_mae = np.mean(np.abs(multi_errors))

        improved = np.sum(np.abs(multi_errors) < np.abs(single_errors))

        print(f"\nProcessed: {len(results)} wells (errors: {len(errors)})")
        print()
        print(f"{'Metric':<20} {'1 Segment':>12} {'Multi-Seg':>12} {'Δ':>10}")
        print("-" * 60)
        print(f"{'RMSE':<20} {single_rmse:>12.2f}m {multi_rmse:>12.2f}m {multi_rmse - single_rmse:>+10.2f}m")
        print(f"{'MAE':<20} {single_mae:>12.2f}m {multi_mae:>12.2f}m {multi_mae - single_mae:>+10.2f}m")
        print(f"{'Improved':<20} {'-':>12} {improved:>12}/{len(results)}")

        # Segment count distribution
        seg_counts = [r['n_segments'] for r in results]
        print(f"\nSegment count distribution:")
        for n in range(1, 6):
            count = seg_counts.count(n)
            if count > 0:
                print(f"  {n} segments: {count} wells")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for name, err in errors[:5]:
            print(f"  {name}: {err}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")


if __name__ == '__main__':
    main()
