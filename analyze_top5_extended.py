#!/usr/bin/env python3
"""
Extended analysis of top 5 degraded wells.
Shows: receiver MD, ref angle, advisor angle, was correct angle achievable?
"""

import sys
import os
sys.path.insert(0, '/mnt/e/Projects/Rogii/gpu_ag')

import torch
import pandas as pd
import numpy as np
from neighbor_angle_advisor import NeighborAngleAdvisor

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/gpu_ag_dataset.pt')
SMOOTHING = 100.0
MAX_DIST = 1000.0
ANGLE_RANGE = 2.5
M_TO_FT = 3.28084

data = torch.load(DATASET_PATH, weights_only=False)
advisor = NeighborAngleAdvisor(DATASET_PATH)

# Load results
adv_csv = '/mnt/e/Projects/Rogii/gpu_ag/results/full_well_20260111_192314.csv'
bf_csv = '/mnt/e/Projects/Rogii/gpu_ag/results/full_well_20260111_082236.csv'

df_adv = pd.read_csv(adv_csv)
df_bf = pd.read_csv(bf_csv)

# Top 5 degraded wells
wells_info = [
    ('Well1620~ASTNL', 'Well1416~EGFDU'),
    ('Well1586~EGFDL', 'Well579~EGFDL'),
    ('Well522~EGFDL', 'Well1884~EGFDL'),
    ('Well32~EGFDL', 'Well531~EGFDL'),
    ('Well576~EGFDL', 'Well916~EGFDL'),
]

def get_ref_angle_at_md(well_name, md):
    """Get ref interpretation angle at given MD"""
    well_data = data[well_name]
    ref_mds = well_data['ref_segment_mds'].numpy()
    ref_start_shifts = well_data['ref_start_shifts'].numpy()
    ref_shifts = well_data['ref_shifts'].numpy()
    last_md = well_data['lateral_well_last_md']

    for i in range(len(ref_mds)):
        seg_start = ref_mds[i]
        seg_end = ref_mds[i + 1] if i + 1 < len(ref_mds) else last_md

        if seg_start <= md <= seg_end:
            dmd = seg_end - seg_start
            dshift = ref_shifts[i] - ref_start_shifts[i]
            if dmd > 0:
                return np.degrees(np.arctan2(dshift, dmd))
    return None

print("=" * 120)
print("EXTENDED ANALYSIS: TOP 5 DEGRADED WELLS")
print("=" * 120)

for well_name, neighbor_name in wells_info:
    print(f"\n{'='*120}")
    print(f"WELL: {well_name} (neighbor: {neighbor_name})")
    print("=" * 120)

    well_data = data.get(well_name)
    if well_data is None:
        print("  NOT FOUND")
        continue

    # Get perch_md and detected_start_md
    perch_md = well_data.get('perch_md', 0)
    detected_start_md = well_data.get('detected_start_md', 0)

    print(f"  perch_md: {perch_md * M_TO_FT:.0f}ft, detected_start_md: {detected_start_md * M_TO_FT:.0f}ft")

    # Get segments with large errors
    segs = df_adv[(df_adv['well_name'] == well_name) & (df_adv['row_type'] == 'segment')]
    bf_segs = df_bf[(df_bf['well_name'] == well_name) & (df_bf['row_type'] == 'segment')]

    bad_segs = segs[abs(segs['end_error']) > 3.0].sort_values('end_error', key=abs, ascending=False)

    print(f"\n  Segments with |error| > 10ft:")
    print(f"  {'Seg':>4} {'MD_recv':>10} {'Error':>10} {'ADV_ang':>8} {'BF_ang':>8} {'Ref_ang':>8} | Neighbor: MD_src, dip | Range OK?")
    print(f"  {'-'*110}")

    for _, seg in bad_segs.head(5).iterrows():
        seg_idx = int(seg['seg_idx'])
        md_recv = seg['md_end']
        err = seg['end_error']
        adv_angle = seg['angle_deg']

        # Get BF angle for same segment
        bf_seg = bf_segs[bf_segs['seg_idx'] == seg_idx]
        bf_angle = bf_seg['angle_deg'].values[0] if len(bf_seg) > 0 else float('nan')

        # Get ref angle at this MD
        ref_angle = get_ref_angle_at_md(well_name, md_recv)
        ref_angle_str = f"{ref_angle:+.2f}°" if ref_angle is not None else "N/A"

        # Get neighbor info at this MD
        neighbors = advisor.find_neighbors(well_name, md_recv, max_neighbors=3, max_distance=MAX_DIST, smoothing_range=SMOOTHING)

        target_neighbor = None
        for n in neighbors:
            if n.well_name == neighbor_name:
                target_neighbor = n
                break

        if target_neighbor:
            md_src = target_neighbor.md
            neigh_dip = target_neighbor.dip_angle_deg
            neigh_str = f"{md_src * M_TO_FT:.0f}ft, {neigh_dip:+.1f}°"

            # Check if correct angle was achievable
            # Advisor center is avg of neighbors
            avg_dip = sum(n.dip_angle_deg for n in neighbors) / len(neighbors)
            range_min = avg_dip - ANGLE_RANGE
            range_max = avg_dip + ANGLE_RANGE

            # Was BF angle in advisor range?
            if range_min <= bf_angle <= range_max:
                range_ok = "✓ YES"
            else:
                gap = min(abs(bf_angle - range_min), abs(bf_angle - range_max))
                range_ok = f"✗ NO (gap {gap:.1f}°)"
        else:
            neigh_str = "not in top 3"
            range_ok = "?"

        print(f"  {seg_idx:>4} {md_recv * M_TO_FT:>10.0f}ft {err * M_TO_FT:>+10.1f}ft {adv_angle:>+8.2f}° {bf_angle:>+8.2f}° {ref_angle_str:>8} | {neigh_str:<20} | {range_ok}")

    # For Well32, compare trend angle
    if well_name == 'Well32~EGFDL':
        print(f"\n  Special analysis for Well32:")
        # Get trend angle from first segment
        first_seg = segs.iloc[0] if len(segs) > 0 else None
        if first_seg is not None:
            # trend_angle is typically stored or can be computed
            ref_mds = well_data['ref_segment_mds'].numpy()
            perch_md = well_data.get('perch_md', ref_mds[0])

            # Compute trend from perch to zone_start (first ~300m)
            zone_start_md = perch_md + 300
            ref_start_shifts = well_data['ref_start_shifts'].numpy()
            ref_shifts = well_data['ref_shifts'].numpy()

            # Find shift at perch and zone_start
            for i, md in enumerate(ref_mds):
                if md >= perch_md:
                    shift_at_perch = ref_start_shifts[i]
                    break

            for i, md in enumerate(ref_mds):
                if md >= zone_start_md:
                    shift_at_zone = ref_shifts[i-1] if i > 0 else ref_start_shifts[i]
                    break
            else:
                shift_at_zone = ref_shifts[-1]

            trend = np.degrees(np.arctan2(shift_at_zone - shift_at_perch, zone_start_md - perch_md))
            print(f"    Trend angle (perch to +300m): {trend:+.2f}°")

            # Compare with BF angles
            bf_angles = bf_segs['angle_deg'].values[:10]
            print(f"    BF angles (first 10 segs): {[f'{a:+.1f}' for a in bf_angles]}")

print("\n" + "=" * 120)
print("DONE")
