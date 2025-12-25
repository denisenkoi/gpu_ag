#!/usr/bin/env python3
"""
Analyze GR variance around segment boundaries.

Builds a table showing variance of normalized GR around each segment boundary.
"""

import json
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from gr_utils import apply_gr_smoothing, get_smoothing_params

load_dotenv()

# Source data - use saved raw file from json_comparison (has both well and interpretation)
SOURCE_WELL = Path("/mnt/e/Projects/Rogii/gpu_ag/json_comparison/new_slicing_well_raw_5032.9.json")
CSV_OUTPUT = Path("/mnt/e/Projects/Rogii/gpu_ag/variance_analysis.csv")

# Conversion
METERS_TO_FEET = 3.28084


def load_data():
    """Load interpretation and well GR from source file"""
    with open(SOURCE_WELL, 'r', encoding='utf-8') as f:
        source_data = json.load(f)

    # Get interpretation segments
    segments = source_data.get('interpretation', {}).get('segments', [])

    # Get well GR curve from wellLog (NOT well.points which is trajectory)
    well_points = source_data.get('wellLog', {}).get('points', [])

    # AG start MD
    ag_start_md = source_data.get('autoGeosteeringParameters', {}).get('startMd', 0)

    return segments, well_points, ag_start_md


def build_gr_array(well_points, start_md):
    """Build MD and GR arrays from well points, starting from start_md"""
    md_list = []
    gr_list = []

    for p in well_points:
        md = p.get('measuredDepth', 0)
        gr = p.get('data')
        # Skip points before start_md or with None data
        if md >= start_md and gr is not None:
            md_list.append(md)
            gr_list.append(gr)

    return np.array(md_list), np.array(gr_list)


def normalize_gr(gr_array):
    """Normalize GR to 0-1 range (min-max scaling)"""
    gr_min = gr_array.min()
    gr_max = gr_array.max()
    if gr_max - gr_min < 1e-6:
        return np.zeros_like(gr_array)
    return (gr_array - gr_min) / (gr_max - gr_min)


def find_gr_in_window(md_array, gr_array, center_md, half_left, half_right):
    """Find GR values in window [center_md - half_left, center_md + half_right]"""
    mask = (md_array >= center_md - half_left) & (md_array <= center_md + half_right)
    return gr_array[mask]


def main():
    segments, well_points, ag_start_md = load_data()

    print(f"AG start MD: {ag_start_md:.2f}m ({ag_start_md * METERS_TO_FEET:.2f}ft)")
    print(f"Total segments: {len(segments)}")
    print()

    if not segments:
        print("No segments found!")
        return

    # Use lookback from .env
    lookback = 200.0  # LOOKBACK_DISTANCE
    analysis_start_md = ag_start_md - lookback

    print(f"Analysis start MD: {analysis_start_md:.2f}m ({analysis_start_md * METERS_TO_FEET:.2f}ft)")
    print()

    # Build GR array from analysis start
    md_array, gr_array = build_gr_array(well_points, analysis_start_md)

    if len(gr_array) == 0:
        print("No GR data found!")
        return

    # Apply GR smoothing (same as in optimizer)
    smoothing = get_smoothing_params()
    if smoothing['enabled']:
        print(f"GR smoothing: Savitzky-Golay (window={smoothing['window']}, order={smoothing['order']})")
        gr_array = apply_gr_smoothing(gr_array)
    else:
        print("GR smoothing: disabled")

    # Normalize GR
    gr_norm = normalize_gr(gr_array)

    print(f"GR range: {gr_array.min():.2f} - {gr_array.max():.2f}")
    print(f"GR points: {len(gr_array)}")
    print()

    # Build table of boundaries
    print("=" * 90)
    print(f"{'#':>3} | {'MD (m)':>10} | {'MD (ft)':>10} | {'SegLen(m)':>9} | {'Window(m)':>9} | {'Var×100':>10} | {'Std×10':>8}")
    print("-" * 90)

    for i, seg in enumerate(segments):
        start_md = seg.get('startMd', 0)
        # end_md = startMd of next segment (segments only have startMd)
        if i + 1 < len(segments):
            end_md = segments[i + 1].get('startMd', 0)
        else:
            # Last segment - use well end or skip
            continue
        seg_len = end_md - start_md

        # Skip segments before analysis start
        if end_md < analysis_start_md:
            continue

        # Boundary is at end of segment (= start of next)
        boundary_md = end_md

        # Window: half segment left + half segment right
        if i + 2 < len(segments):
            next_seg_end = segments[i + 2].get('startMd', 0)
            next_seg_len = next_seg_end - end_md
            half_right = next_seg_len / 2
        else:
            half_right = seg_len / 2

        half_left = seg_len / 2
        window_size = half_left + half_right

        # Get GR in window
        gr_window = find_gr_in_window(md_array, gr_norm, boundary_md, half_left, half_right)

        if len(gr_window) < 2:
            variance = 0
            std = 0
        else:
            variance = np.var(gr_window)
            std = np.std(gr_window)

        # Scale for readability
        var_scaled = variance * 100
        std_scaled = std * 10

        print(f"{i:>3} | {boundary_md:>10.2f} | {boundary_md * METERS_TO_FEET:>10.2f} | {seg_len:>9.2f} | {window_size:>9.2f} | {var_scaled:>10.4f} | {std_scaled:>8.4f}")

    print("=" * 90)
    print()
    print("Legend:")
    print("  MD - boundary at end of segment (= start of next)")
    print("  SegLen - length of current segment")
    print("  Window - half left + half right around boundary")
    print("  Var×100 - variance of normalized GR × 100")
    print("  Std×10 - standard deviation × 10")

    # Save to CSV
    csv_path = CSV_OUTPUT
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("seg_idx,boundary_md_m,boundary_md_ft,seg_len_m,window_m,var_x100,std_x10\n")
        for i, seg in enumerate(segments):
            start_md = seg.get('startMd', 0)
            if i + 1 < len(segments):
                end_md = segments[i + 1].get('startMd', 0)
            else:
                continue
            seg_len = end_md - start_md
            if end_md < analysis_start_md:
                continue
            boundary_md = end_md
            if i + 2 < len(segments):
                next_seg_end = segments[i + 2].get('startMd', 0)
                next_seg_len = next_seg_end - end_md
                half_right = next_seg_len / 2
            else:
                half_right = seg_len / 2
            half_left = seg_len / 2
            window_size = half_left + half_right
            gr_window = find_gr_in_window(md_array, gr_norm, boundary_md, half_left, half_right)
            if len(gr_window) < 2:
                variance = std = 0
            else:
                variance = np.var(gr_window)
                std = np.std(gr_window)
            f.write(f"{i},{boundary_md:.2f},{boundary_md * METERS_TO_FEET:.2f},{seg_len:.2f},{window_size:.2f},{variance*100:.4f},{std*10:.4f}\n")
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
