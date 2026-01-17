#!/usr/bin/env python3
"""
Trim pseudoTypeLog to TVD at landing point to remove data leakage.

Usage:
    python trim_pseudo_tvd.py --well Well162~EGFDL --dry-run
    python trim_pseudo_tvd.py --all
"""

import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Tuple


def get_tvd_at_landing(well_data: dict, use_dls: bool = True) -> float:
    """Get TVD at landing point from dataset."""
    if use_dls and 'landing_end_dls' in well_data:
        landing_md = well_data['landing_end_dls']
    else:
        landing_md = well_data['landing_end_87_200']

    # Find TVD at landing MD from well trajectory
    mds = well_data['well_md']
    tvds = well_data['well_tvd']
    mds = mds.numpy() if torch.is_tensor(mds) else np.array(mds)
    tvds = tvds.numpy() if torch.is_tensor(tvds) else np.array(tvds)

    idx = np.argmin(np.abs(mds - landing_md))
    return float(tvds[idx])


def trim_pseudo_in_json(json_path: Path, tvd_limit: float, dry_run: bool = False) -> Tuple[int, int, float, float]:
    """
    Trim pseudoTypeLog points to TVD limit.

    Returns: (before_count, after_count, before_tvd_max, after_tvd_max)
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    pseudo = data.get('pseudoTypeLog', {})
    points = pseudo.get('points', [])
    tvd_sorted = pseudo.get('tvdSortedPoints', [])

    if not tvd_sorted:
        return 0, 0, 0, 0

    before_count = len(tvd_sorted)
    before_tvd_max = max(p['trueVerticalDepth'] for p in tvd_sorted)

    # Filter points by TVD
    new_tvd_sorted = [p for p in tvd_sorted if p['trueVerticalDepth'] <= tvd_limit]
    new_points = [{'data': p['data'], 'measuredDepth': p['measuredDepth']} for p in new_tvd_sorted]

    after_count = len(new_tvd_sorted)
    after_tvd_max = max(p['trueVerticalDepth'] for p in new_tvd_sorted) if new_tvd_sorted else 0

    if not dry_run:
        pseudo['points'] = new_points
        pseudo['tvdSortedPoints'] = new_tvd_sorted

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    return before_count, after_count, before_tvd_max, after_tvd_max


def main():
    parser = argparse.ArgumentParser(description='Trim pseudoTypeLog to TVD at landing')
    parser.add_argument('--well', type=str, help='Single well name')
    parser.add_argument('--all', action='store_true', help='Process all wells')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be done')
    parser.add_argument('--use-87-200', action='store_true', help='Use 87_200 instead of DLS landing')
    args = parser.parse_args()

    if not args.well and not args.all:
        parser.error('Specify --well or --all')

    # Load dataset
    dataset_path = Path('/mnt/e/Projects/Rogii/gpu_ag/data/wells_fresh.pt')
    dataset = torch.load(dataset_path, weights_only=False)

    json_dir = Path('/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de/AG_DATA/InitialData')

    wells = [args.well] if args.well else list(dataset.keys())

    total_removed = 0
    processed = 0

    for well_name in wells:
        if well_name not in dataset:
            print(f"SKIP {well_name}: not in dataset")
            continue

        json_path = json_dir / f"{well_name}.json"
        if not json_path.exists():
            print(f"SKIP {well_name}: JSON not found")
            continue

        well_data = dataset[well_name]
        tvd_limit = get_tvd_at_landing(well_data, use_dls=not args.use_87_200)

        before, after, before_max, after_max = trim_pseudo_in_json(
            json_path, tvd_limit, dry_run=args.dry_run
        )

        removed = before - after
        total_removed += removed
        processed += 1

        if removed > 0 or args.well:
            leakage_before = before_max - tvd_limit
            leakage_after = after_max - tvd_limit
            print(f"{well_name}: {before} -> {after} points (-{removed}), "
                  f"TVD max: {before_max:.1f} -> {after_max:.1f}m, "
                  f"leakage: {leakage_before:.1f}m -> {leakage_after:.1f}m")

    print(f"\nTotal: processed {processed} wells, removed {total_removed} points")
    if args.dry_run:
        print("(dry run - no files modified)")


if __name__ == '__main__':
    main()
