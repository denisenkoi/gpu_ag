#!/usr/bin/env python3
"""
Analyze degraded wells: compare advisor recommendations with BF baseline.
Find where advisor pulls solution away from trend.
"""

import sys
import os
sys.path.insert(0, '/mnt/e/Projects/Rogii/gpu_ag')

import torch
import pandas as pd
import numpy as np
from neighbor_angle_advisor import NeighborAngleAdvisor

# Parameters
DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/gpu_ag_dataset.pt')
ANGLE_RANGE = 2.0
SMOOTHING = 100.0
MAX_DIST = 1000.0
SEGS_PER_BLOCK = 5

# Load data
data = torch.load(DATASET_PATH, weights_only=False)
advisor = NeighborAngleAdvisor(DATASET_PATH)

# Top degraded wells from advisor smooth=100 test
wells_to_analyze = [
    ('Well1002_landing~EGFDL', -9.21, +10.65),  # (well, base, adv)
    ('Well1890~EGFDL', -4.09, -16.30),
    ('Well1620~ASTNL', +0.08, -11.81),
    ('Well1682~EGFDL', +0.95, +9.37),
    ('Well1150~EGFDL', +1.81, +8.39),
]

print("=" * 90)
print("ANALYSIS: Where does advisor pull solution away from optimal?")
print("=" * 90)

for well_name, base_err, adv_err in wells_to_analyze:
    print(f"\n{'='*90}")
    print(f"WELL: {well_name}")
    print(f"Base error: {base_err:+.2f}m, Advisor error: {adv_err:+.2f}m, Degradation: {abs(adv_err)-abs(base_err):+.2f}m")
    print("=" * 90)

    well_data = data.get(well_name)
    if well_data is None:
        print("  NOT FOUND")
        continue

    # Get segment MDs
    ref_segment_mds = well_data['ref_segment_mds'].numpy()
    perch_md = well_data.get('perch_md', ref_segment_mds[0])
    n_seg = len(ref_segment_mds)

    # Find first lateral segment (MD > perch_md)
    lateral_start = 0
    for i, md in enumerate(ref_segment_mds):
        if md > perch_md:
            lateral_start = i
            break

    print(f"\n  Total segments: {n_seg}, Lateral start: seg {lateral_start} (perch_md={perch_md:.0f}m)")

    # Analyze blocks
    n_blocks = (n_seg - lateral_start + SEGS_PER_BLOCK - 1) // SEGS_PER_BLOCK

    print(f"\n  {'Block':>5} {'MD_start':>10} {'MD_end':>10} {'AdvCenter':>10} {'Neighbors':>40}")
    print(f"  {'-'*80}")

    for block_idx in range(min(n_blocks, 20)):  # First 20 blocks
        start_seg = lateral_start + block_idx * SEGS_PER_BLOCK
        end_seg = min(start_seg + SEGS_PER_BLOCK, n_seg)

        if end_seg <= start_seg:
            break

        # Block center MD
        md_start = ref_segment_mds[start_seg]
        md_end = ref_segment_mds[end_seg - 1]
        block_center_md = (md_start + md_end) / 2

        # Get advisor recommendation
        neighbors = advisor.find_neighbors(
            well_name, block_center_md,
            max_neighbors=3,
            max_distance=MAX_DIST,
            smoothing_range=SMOOTHING
        )

        if neighbors:
            advisor_dip = sum(n.dip_angle_deg for n in neighbors) / len(neighbors)
            neighbor_str = ", ".join([f"{n.well_name.split('~')[0]}:{n.dip_angle_deg:+.1f}°({n.distance_3d:.0f}m)" for n in neighbors[:2]])
        else:
            advisor_dip = 0.0
            neighbor_str = "NO NEIGHBORS"

        # Range that advisor would use
        range_min = advisor_dip - ANGLE_RANGE
        range_max = advisor_dip + ANGLE_RANGE

        print(f"  {block_idx:>5} {md_start:>10.0f} {md_end:>10.0f} {advisor_dip:>+10.2f}° {neighbor_str}")

    # Check overall pattern
    print(f"\n  Advisor dip trend across well:")
    mid_points = np.linspace(ref_segment_mds[lateral_start], ref_segment_mds[-1], 10)
    for md in mid_points:
        neighbors = advisor.find_neighbors(well_name, md, max_neighbors=3, max_distance=MAX_DIST, smoothing_range=SMOOTHING)
        if neighbors:
            avg_dip = sum(n.dip_angle_deg for n in neighbors) / len(neighbors)
            print(f"    MD={md:>8.0f}m: advisor_dip={avg_dip:+.2f}° from {len(neighbors)} neighbors")
        else:
            print(f"    MD={md:>8.0f}m: NO NEIGHBORS FOUND")

print("\n" + "=" * 90)
print("DONE")
