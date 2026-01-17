#!/usr/bin/env python3
"""
Analyze where advisor angles differ from BF angles
and if range ±2° is insufficient.
"""

import sys
import os
sys.path.insert(0, '/mnt/e/Projects/Rogii/gpu_ag')

import torch
import pandas as pd
from pathlib import Path
from neighbor_angle_advisor import NeighborAngleAdvisor

# Load data
dataset_path = os.environ.get('DATASET_PATH', 'dataset/gpu_ag_dataset.pt')
data = torch.load(dataset_path, weights_only=False)
advisor = NeighborAngleAdvisor(dataset_path)

# Load BF results (baseline with mse=5, range=2.5)
bf_csv = '/mnt/e/Projects/Rogii/gpu_ag/results/full_well_20260111_063018.csv'  # BF mse=5
bf_df = pd.read_csv(bf_csv)

# Advisor results
adv_csv = '/mnt/e/Projects/Rogii/gpu_ag/results/full_well_20260111_171521.csv'  # Advisor smooth=100
adv_df = pd.read_csv(adv_csv)

# Top degraded wells
wells_to_analyze = [
    'Well1002_landing~EGFDL',  # base=-9.21, adv=+10.65
    'Well1890~EGFDL',          # base=-4.09, adv=-16.30
    'Well1620~ASTNL',          # base=+0.08, adv=-11.81
    'Well1682~EGFDL',          # base=+0.95, adv=+9.37
    'Well1150~EGFDL',          # base=+1.81, adv=+8.39
]

angle_range = 2.0
smoothing_range = 100.0
max_distance = 1000.0

print("=" * 80)
print("ANALYSIS: Advisor vs BF angles, checking range sufficiency")
print("=" * 80)

for well_name in wells_to_analyze:
    print(f"\n{'='*80}")
    print(f"WELL: {well_name}")
    print(f"{'='*80}")

    # Get well data (dataset format: dict with well_name keys directly)
    well_data = data.get(well_name)
    if well_data is None:
        print(f"  Well not found in dataset")
        continue

    segments = well_data['segments']
    n_seg = len(segments)

    # Get BF and ADV final errors
    bf_row = bf_df[bf_df['well'] == well_name]
    adv_row = adv_df[adv_df['well'] == well_name]

    if len(bf_row) == 0 or len(adv_row) == 0:
        print(f"  Not found in CSV")
        continue

    bf_error = bf_row['optimized_error'].values[0]
    adv_error = adv_row['optimized_error'].values[0]
    base_error = bf_row['baseline_error'].values[0]

    print(f"  Base error: {base_error:+.2f}m")
    print(f"  BF error:   {bf_error:+.2f}m")
    print(f"  ADV error:  {adv_error:+.2f}m")
    print(f"  Degradation: {abs(adv_error) - abs(base_error):+.2f}m")

    # Analyze blocks (5 segments each)
    segs_per_block = 5
    n_blocks = (n_seg + segs_per_block - 1) // segs_per_block

    print(f"\n  Analyzing {n_blocks} blocks ({segs_per_block} segments each):")
    print(f"  {'Block':>5} {'MD':>8} {'TrendAngle':>10} {'AdvCenter':>10} {'Δ':>6} {'Range':>14}")
    print(f"  {'-'*60}")

    issues = []

    for block_idx in range(n_blocks):
        start_seg = block_idx * segs_per_block
        end_seg = min(start_seg + segs_per_block, n_seg)

        # Block center MD
        md_start = segments[start_seg]['md_start']
        md_end = segments[end_seg - 1]['md_end']
        block_center_md = (md_start + md_end) / 2

        # Get trend angle
        trend_angle = segments[start_seg]['trend_angle']

        # Get advisor recommended center
        advisor_dip = advisor.get_recommended_dip(
            well_name, block_center_md,
            max_neighbors=3,
            max_distance=max_distance,
            smoothing_range=smoothing_range
        )

        if advisor_dip is None:
            advisor_center = trend_angle
            center_diff = 0
        else:
            advisor_center = advisor_dip
            center_diff = advisor_center - trend_angle

        # Range
        range_min = advisor_center - angle_range
        range_max = advisor_center + angle_range

        print(f"  {block_idx:>5} {block_center_md:>8.0f} {trend_angle:>+10.2f}° {advisor_center:>+10.2f}° {center_diff:>+6.2f}° [{range_min:>+5.1f}°,{range_max:>+5.1f}°]")

        # Check if trend is outside advisor range
        if trend_angle < range_min or trend_angle > range_max:
            issues.append({
                'block': block_idx,
                'md': block_center_md,
                'trend': trend_angle,
                'adv_center': advisor_center,
                'gap': min(abs(trend_angle - range_min), abs(trend_angle - range_max))
            })

    if issues:
        print(f"\n  ⚠️ ISSUES: Trend outside advisor range in {len(issues)} blocks:")
        for issue in issues:
            print(f"     Block {issue['block']}: trend={issue['trend']:+.2f}°, adv_range=[{issue['adv_center']-angle_range:+.1f}°,{issue['adv_center']+angle_range:+.1f}°], gap={issue['gap']:.2f}°")
    else:
        print(f"\n  ✓ All blocks: trend within ±{angle_range}° of advisor center")

    # Check neighbors
    print(f"\n  Neighbors at mid-well:")
    mid_md = (segments[0]['md_start'] + segments[-1]['md_end']) / 2
    neighbors = advisor.find_neighbors(
        well_name, mid_md,
        max_neighbors=5,
        max_distance=max_distance,
        smoothing_range=smoothing_range
    )
    for n in neighbors:
        print(f"    {n.well_name}: dist={n.distance:.0f}m, dip={n.dip_angle:+.2f}°")

print("\n" + "=" * 80)
print("DONE")
