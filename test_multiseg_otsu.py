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


def main():
    print("=" * 60)
    print("Multi-Segment Optimization on OTSU Region")
    print("=" * 60)

    ds = torch.load('dataset/gpu_ag_dataset.pt', weights_only=False)
    well_name = 'Well1221~EGFDL'
    well = ds[well_name]

    # Extract data
    well_md = well['well_md'].numpy()
    well_tvd = well['well_tvd'].numpy()
    log_md = well['log_md'].numpy()
    log_gr = well['log_gr'].numpy()
    type_tvd = well['type_tvd'].numpy()
    type_gr = well['type_gr'].numpy()
    tvd_shift = float(well.get('tvd_typewell_shift', 0.0))

    # Interpolate GR to well_md
    well_gr = np.interp(well_md, log_md, log_gr)

    # OTSU zone start
    otsu_start_md = 6069.5
    otsu_end_md = 6269.4

    # Get baseline shift
    from numpy_funcs.interpretation import interpolate_shift_at_md
    baseline_shift = interpolate_shift_at_md(well, otsu_start_md)

    # Get trajectory angle in zone
    start_idx = np.searchsorted(well_md, otsu_start_md)
    end_idx = np.searchsorted(well_md, otsu_end_md)
    traj_angle = get_trajectory_angle(well_tvd, well_md, start_idx, end_idx)

    print(f"\nWell: {well_name}")
    print(f"OTSU zone: {otsu_start_md:.1f} - {otsu_end_md:.1f}m")
    print(f"Baseline shift at start: {baseline_shift:.2f}m")
    print(f"Trajectory angle in zone: {traj_angle:.2f}°")

    # Use SmartSegmenter to find boundaries in zone
    # Create subset data for SmartSegmenter
    zone_data = {
        'well_md': torch.tensor(well_md),
        'log_md': torch.tensor(log_md),
        'log_gr': torch.tensor(log_gr),
        'start_md': otsu_start_md,
        'detected_start_md': otsu_start_md,
    }

    settings = {
        'min_segment_length': 30.0,  # At least 30m per segment
        'overlap_segments': 0,
        'pelt_penalty': 10.0,
        'min_distance': 20.0,
        'segments_count': 5,  # Max 5 segments
    }

    segmenter = SmartSegmenter(zone_data, settings)

    # Get all boundaries up to zone end
    all_boundaries = []
    while not segmenter.is_finished:
        result = segmenter.next_slice()
        for bp in result.boundaries:
            if bp.md <= otsu_end_md:
                all_boundaries.append(bp)
        if result.is_final or (result.boundaries and result.boundaries[-1].md >= otsu_end_md):
            break

    print(f"\nSmartSegmenter found {len(all_boundaries)} boundaries:")
    for bp in all_boundaries[:10]:
        print(f"  MD={bp.md:.1f}m, {bp.type}, score={bp.score:.1f}")

    # Create segment indices from boundaries (max 5 segments)
    segment_indices = []
    prev_idx = start_idx

    for bp in all_boundaries[:4]:  # Max 4 boundaries = 5 segments
        bp_idx = np.searchsorted(well_md, bp.md)
        if bp_idx > prev_idx:
            segment_indices.append((prev_idx, bp_idx))
            prev_idx = bp_idx

    # Last segment to zone end
    if prev_idx < end_idx:
        segment_indices.append((prev_idx, end_idx))

    print(f"\nSegments to optimize: {len(segment_indices)}")
    for i, (s, e) in enumerate(segment_indices):
        print(f"  Seg {i+1}: MD {well_md[s]:.1f} - {well_md[e]:.1f}m ({well_md[e]-well_md[s]:.1f}m)")

    # Reference shift at zone end for error calculation
    ref_shift_end = interpolate_shift_at_md(well, otsu_end_md)
    print(f"\nReference shift at zone end: {ref_shift_end:.2f}m")

    # === Optimize ===
    print("\n" + "=" * 60)
    print("OPTIMIZATION (angle range ±1°)")
    print("=" * 60)

    best_pearson, best_angles, best_segments = optimize_multiseg(
        segment_indices,
        baseline_shift,
        traj_angle,
        well_md, well_tvd, well_gr,
        type_tvd, type_gr, tvd_shift,
        angle_range=1.0,
        angle_steps=11,
    )

    print(f"\nBest Pearson: {best_pearson:.3f}")
    print("Best angles:")
    for i, (angle, (s, e)) in enumerate(zip(best_angles, segment_indices)):
        print(f"  Seg {i+1}: {angle:.2f}° (MD {well_md[s]:.1f}-{well_md[e]:.1f}m)")

    # Calculate error
    if best_segments:
        opt_end_shift = best_segments[-1][3]  # end_shift of last segment
        error = opt_end_shift - ref_shift_end
        print(f"\nOptimized end shift: {opt_end_shift:.2f}m")
        print(f"Reference end shift: {ref_shift_end:.2f}m")
        print(f"Error: {error:+.2f}m")

    # === Compare with single segment ===
    print("\n" + "=" * 60)
    print("COMPARISON: 1 segment vs multi-segment")
    print("=" * 60)

    # Single segment
    single_seg = [(start_idx, end_idx)]
    single_pearson, single_angles, single_segments = optimize_multiseg(
        single_seg,
        baseline_shift,
        traj_angle,
        well_md, well_tvd, well_gr,
        type_tvd, type_gr, tvd_shift,
        angle_range=1.0,
        angle_steps=21,
    )
    single_end_shift = single_segments[0][3]
    single_error = single_end_shift - ref_shift_end

    print(f"\n{'Method':<25} {'Pearson':>10} {'Error':>10}")
    print("-" * 50)
    print(f"{'1 segment':<25} {single_pearson:>10.3f} {single_error:>+10.2f}m")
    print(f"{f'{len(segment_indices)} segments':<25} {best_pearson:>10.3f} {error:>+10.2f}m")

    # Reference Pearson
    ref_synthetic = project_segments(
        [(s, e, interpolate_shift_at_md(well, well_md[s]), interpolate_shift_at_md(well, well_md[e]))
         for s, e in segment_indices],
        well_md, well_tvd, type_tvd, type_gr, tvd_shift
    )
    ref_pearson = pearson(well_gr[start_idx:end_idx], ref_synthetic[start_idx:end_idx])
    print(f"{'Reference (actual)':<25} {ref_pearson:>10.3f} {0.0:>+10.2f}m")


if __name__ == '__main__':
    main()
