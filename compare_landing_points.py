#!/usr/bin/env python3
"""
Compare different landing point detection algorithms:
1. 87_200 - first point where incl >= 87° + 200m
2. DLS-based - longest segment with INCL↑ after DLS>5, + 600ft

Runs on 100 wells from JSON files.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch


@dataclass
class LandingComparison:
    well_name: str
    landing_87_200: float  # MD
    landing_dls: float     # MD
    delta: float           # dls - 87_200


def compute_dls(md: np.ndarray, tvd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute DLS (Dog Leg Severity) and inclination for trajectory.

    DLS in °/100ft. Formula: angle_change / distance * 30.48 (since 100ft = 30.48m)
    Inclination: angle from vertical (0° = vertical, 90° = horizontal)

    Returns:
        dls: DLS array (°/100ft)
        incl: inclination array (degrees)
    """
    n = len(md)
    dls = np.zeros(n)
    incl = np.zeros(n)

    for i in range(1, n):
        delta_md = md[i] - md[i-1]
        delta_tvd = tvd[i] - tvd[i-1]

        if delta_md > 0.1:
            # Inclination: angle from vertical
            # tan(incl) = horizontal_displacement / vertical_displacement
            # For simple 2D: incl = 90 - atan(|delta_tvd| / delta_md)
            incl[i] = 90 - np.degrees(np.arctan(abs(delta_tvd) / delta_md))

            # DLS: rate of inclination change per 100ft
            if i > 1:
                angle_change = abs(incl[i] - incl[i-1])
                # Convert to °/100ft (100ft = 30.48m)
                dls[i] = angle_change / delta_md * 30.48

    return dls, incl


def find_landing_dls(md: np.ndarray, tvd: np.ndarray) -> float:
    """
    DLS-based landing detection:
    1. Find first point where DLS > 5
    2. After DLS > 5, continue while INCL increases
    3. Wait until INCL reaches ~87° AND then stops increasing (or drops)
    4. landing_end = point where INCL stabilizes after reaching near-horizontal
    5. target = landing_end + 600ft (182.88m)

    Returns:
        target_md: landing end + 600ft
    """
    dls, incl = compute_dls(md, tvd)
    n = len(md)

    # Step 1: Find first point where DLS > 5
    dls_start_idx = None
    for i in range(1, n):
        if dls[i] > 5.0:
            dls_start_idx = i
            break

    if dls_start_idx is None:
        return md[n // 2] + 182.88

    # Step 2: From DLS>5, find where incl reaches ~87° and then stops increasing
    # Look for: incl >= 87° AND (incl stabilizes OR incl decreases)
    landing_end_idx = dls_start_idx
    reached_high_angle = False
    window = 10  # points to check for stability

    for i in range(dls_start_idx, n - window):
        # First, wait until we reach high angle (87°+)
        if incl[i] >= 87:
            reached_high_angle = True

        if reached_high_angle:
            # Check if incl has stabilized or started decreasing
            incl_change = incl[i + window] - incl[i]

            # Landing end when: incl stops increasing significantly OR starts dropping
            if incl_change < 0.3:  # less than 0.3° increase over window = stabilized
                landing_end_idx = i
                break

            # Also detect "overshoot" - when incl goes above 90 then drops
            if incl[i] > 89.5 and incl[i + window] < incl[i]:
                landing_end_idx = i
                break
    else:
        # Never stabilized, find point of max incl
        landing_end_idx = np.argmax(incl)

    landing_end_md = md[landing_end_idx]
    target_md = landing_end_md + 182.88  # 600ft in meters

    return target_md


def find_landing_87_200(md: np.ndarray, tvd: np.ndarray) -> float:
    """
    87_200 algorithm: first point where incl >= 87° + 200m
    """
    n = len(md)
    incl87_md = None

    for i in range(1, n):
        delta_tvd = tvd[i] - tvd[i-1]
        delta_md = md[i] - md[i-1]

        if delta_md > 0.1:
            incl = 90 - np.degrees(np.arctan(abs(delta_tvd) / delta_md))
            if incl >= 87:
                incl87_md = md[i]
                break

    if incl87_md is None:
        incl87_md = md[n // 2]

    return incl87_md + 200


def process_well(well_path: Path) -> Optional[LandingComparison]:
    """Process single well JSON file"""
    try:
        with open(well_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        well_name = data.get('wellName', well_path.stem)
        points = data['well']['points']

        md = np.array([p['measuredDepth'] for p in points])
        tvd = np.array([p['trueVerticalDepth'] for p in points])

        landing_87_200 = find_landing_87_200(md, tvd)
        landing_dls = find_landing_dls(md, tvd)

        return LandingComparison(
            well_name=well_name,
            landing_87_200=landing_87_200,
            landing_dls=landing_dls,
            delta=landing_dls - landing_87_200
        )
    except Exception as e:
        print(f"Error processing {well_path.name}: {e}")
        return None


def main():
    # Find wells directory (same as json_to_torch.py uses)
    wells_dir = Path('/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de/AG_DATA/InitialData/')
    if not wells_dir.exists():
        print(f"Wells directory not found: {wells_dir}")
        return

    well_files = sorted(wells_dir.glob('*.json'))
    print(f"Found {len(well_files)} wells")
    print("=" * 80)

    results: List[LandingComparison] = []

    for well_file in well_files:
        result = process_well(well_file)
        if result:
            results.append(result)

    # Sort by delta (largest difference first)
    results.sort(key=lambda x: abs(x.delta), reverse=True)

    # Print results
    print(f"\n{'Well':<30} {'87_200':>10} {'DLS':>10} {'Delta':>10}")
    print("-" * 62)

    for r in results:
        print(f"{r.well_name:<30} {r.landing_87_200:>10.1f} {r.landing_dls:>10.1f} {r.delta:>+10.1f}")

    # Statistics
    deltas = [r.delta for r in results]
    print("\n" + "=" * 62)
    print(f"Statistics (N={len(results)}):")
    print(f"  Mean delta:   {np.mean(deltas):+.1f}m")
    print(f"  Std delta:    {np.std(deltas):.1f}m")
    print(f"  Min delta:    {np.min(deltas):+.1f}m")
    print(f"  Max delta:    {np.max(deltas):+.1f}m")
    print(f"  Median delta: {np.median(deltas):+.1f}m")

    # How many DLS > 87_200 vs DLS < 87_200
    dls_later = sum(1 for d in deltas if d > 0)
    dls_earlier = sum(1 for d in deltas if d < 0)
    same = sum(1 for d in deltas if abs(d) < 1)

    print(f"\n  DLS later than 87_200:   {dls_later}")
    print(f"  DLS earlier than 87_200: {dls_earlier}")
    print(f"  Within 1m:               {same}")


if __name__ == '__main__':
    main()
