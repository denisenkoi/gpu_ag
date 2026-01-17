#!/usr/bin/env python3
"""
Analyze top degraded wells - find where bad recommendations came from.
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

data = torch.load(DATASET_PATH, weights_only=False)
advisor = NeighborAngleAdvisor(DATASET_PATH)

# Load results
adv_csv = '/mnt/e/Projects/Rogii/gpu_ag/results/full_well_20260111_192314.csv'  # Advisor ±2.5°
df = pd.read_csv(adv_csv)
wells_df = df[df['row_type'] == 'well']

# Calculate degradation (opt worse than baseline)
wells_df = wells_df.copy()
wells_df['degradation'] = abs(wells_df['opt_error']) - abs(wells_df['baseline_error'])

# Top 5 degraded
top5 = wells_df.nlargest(5, 'degradation')[['well_name', 'baseline_error', 'opt_error', 'degradation']]

print("=" * 100)
print("TOP 5 DEGRADED WELLS (Advisor ±2.5°)")
print("=" * 100)

for _, row in top5.iterrows():
    well_name = row['well_name']
    base_err = row['baseline_error']
    opt_err = row['opt_error']
    deg = row['degradation']

    print(f"\n{'='*100}")
    print(f"WELL: {well_name}")
    print(f"Baseline: {base_err:+.2f}m → Optimized: {opt_err:+.2f}m (degradation: {deg:+.2f}m)")
    print("=" * 100)

    well_data = data.get(well_name)
    if well_data is None:
        print("  NOT FOUND")
        continue

    ref_segment_mds = well_data['ref_segment_mds'].numpy()
    perch_md = well_data.get('perch_md', ref_segment_mds[0])
    n_seg = len(ref_segment_mds)

    # Get segments from CSV
    segs_df = df[(df['well_name'] == well_name) & (df['row_type'] == 'segment')]

    # Find segments with large errors (|end_error| > 5m)
    bad_segs = segs_df[abs(segs_df['end_error']) > 3.0].sort_values('end_error', key=abs, ascending=False)

    print(f"\n  Segments with |error| > 3m:")
    print(f"  {'Seg':>4} {'MD':>10} {'EndErr':>8} {'Angle':>8} | Neighbors (well:dip@MD, dist)")
    print(f"  {'-'*90}")

    for _, seg in bad_segs.head(10).iterrows():
        seg_idx = int(seg['seg_idx'])
        md = seg['md_end']
        err = seg['end_error']
        angle = seg['angle_deg']

        # Get neighbors at this MD
        neighbors = advisor.find_neighbors(
            well_name, md,
            max_neighbors=3,
            max_distance=MAX_DIST,
            smoothing_range=SMOOTHING
        )

        if neighbors:
            neigh_str = ", ".join([
                f"{n.well_name.split('~')[0]}:{n.dip_angle_deg:+.1f}°@{n.md:.0f}m ({n.distance_3d:.0f}m)"
                for n in neighbors[:2]
            ])
        else:
            neigh_str = "NO NEIGHBORS"

        print(f"  {seg_idx:>4} {md:>10.0f} {err:>+8.2f}m {angle:>+8.2f}° | {neigh_str}")

    # Summary: which neighbors contributed most
    print(f"\n  Neighbor summary (most frequent):")
    neighbor_counts = {}
    for md in ref_segment_mds[10:]:  # skip landing
        if md < perch_md:
            continue
        neighbors = advisor.find_neighbors(well_name, float(md), max_neighbors=3, max_distance=MAX_DIST, smoothing_range=SMOOTHING)
        for n in neighbors:
            key = n.well_name
            if key not in neighbor_counts:
                neighbor_counts[key] = {'count': 0, 'dips': [], 'dists': []}
            neighbor_counts[key]['count'] += 1
            neighbor_counts[key]['dips'].append(n.dip_angle_deg)
            neighbor_counts[key]['dists'].append(n.distance_3d)

    for neigh_name, stats in sorted(neighbor_counts.items(), key=lambda x: -x[1]['count'])[:3]:
        avg_dip = np.mean(stats['dips'])
        avg_dist = np.mean(stats['dists'])
        # Get neighbor's baseline error
        neigh_row = wells_df[wells_df['well_name'] == neigh_name]
        if len(neigh_row) > 0:
            neigh_base = neigh_row['baseline_error'].values[0]
        else:
            neigh_base = float('nan')
        print(f"    {neigh_name}: {stats['count']}x, avg_dip={avg_dip:+.1f}°, avg_dist={avg_dist:.0f}m, baseline_err={neigh_base:+.1f}m")

print("\n" + "=" * 100)
print("DONE")
