#!/usr/bin/env python3
"""
Debug script to visualize landing trajectory and detection points
"""

import json
import numpy as np
from pathlib import Path
from typing import Tuple


def compute_dls_incl(md: np.ndarray, tvd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute DLS and inclination"""
    n = len(md)
    dls = np.zeros(n)
    incl = np.zeros(n)

    for i in range(1, n):
        delta_md = md[i] - md[i-1]
        delta_tvd = tvd[i] - tvd[i-1]

        if delta_md > 0.1:
            incl[i] = 90 - np.degrees(np.arctan(abs(delta_tvd) / delta_md))
            if i > 1:
                angle_change = abs(incl[i] - incl[i-1])
                dls[i] = angle_change / delta_md * 30.48  # °/100ft

    return dls, incl


def find_landing_87_200(md: np.ndarray, incl: np.ndarray) -> Tuple[float, int]:
    """Find 87_200 point, return (target_md, incl87_idx)"""
    n = len(md)
    for i in range(1, n):
        if incl[i] >= 87:
            return md[i] + 200, i
    return md[n // 2] + 200, n // 2


def find_landing_dls(md: np.ndarray, dls: np.ndarray, incl: np.ndarray) -> Tuple[float, int, int]:
    """Find DLS-based point, return (target_md, dls_start_idx, landing_end_idx)"""
    n = len(md)

    # Find first DLS > 5
    dls_start_idx = None
    for i in range(1, n):
        if dls[i] > 5.0:
            dls_start_idx = i
            break

    if dls_start_idx is None:
        return md[n // 2] + 182.88, n // 2, n // 2

    # Find where incl reaches 87° and then stops increasing
    landing_end_idx = dls_start_idx
    reached_high_angle = False
    window = 10

    for i in range(dls_start_idx, n - window):
        if incl[i] >= 87:
            reached_high_angle = True

        if reached_high_angle:
            incl_change = incl[i + window] - incl[i]
            if incl_change < 0.3:
                landing_end_idx = i
                break
            if incl[i] > 89.5 and incl[i + window] < incl[i]:
                landing_end_idx = i
                break
    else:
        landing_end_idx = np.argmax(incl)

    return md[landing_end_idx] + 182.88, dls_start_idx, landing_end_idx


def analyze_well(well_path: Path):
    """Analyze single well trajectory"""
    with open(well_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    well_name = data.get('wellName', well_path.stem)
    points = data['well']['points']

    md = np.array([p['measuredDepth'] for p in points])
    tvd = np.array([p['trueVerticalDepth'] for p in points])

    dls, incl = compute_dls_incl(md, tvd)

    # Find landing points
    target_87_200, incl87_idx = find_landing_87_200(md, incl)
    target_dls, dls_start_idx, dls_end_idx = find_landing_dls(md, dls, incl)

    print(f"\n{'='*80}")
    print(f"Well: {well_name}")
    print(f"{'='*80}")

    print(f"\n87_200 algorithm:")
    print(f"  incl87 point: MD={md[incl87_idx]:.1f}m, incl={incl[incl87_idx]:.2f}°")
    print(f"  target (incl87 + 200m): MD={target_87_200:.1f}m")

    print(f"\nDLS algorithm:")
    print(f"  DLS>5 first: MD={md[dls_start_idx]:.1f}m, incl={incl[dls_start_idx]:.2f}°, DLS={dls[dls_start_idx]:.2f}°/100ft")
    print(f"  landing_end: MD={md[dls_end_idx]:.1f}m, incl={incl[dls_end_idx]:.2f}°")
    print(f"  target (end + 600ft): MD={target_dls:.1f}m")

    print(f"\nDelta: {target_dls - target_87_200:+.1f}m")

    # Print trajectory around landing zone
    print(f"\n--- Trajectory around landing ---")
    print(f"{'MD':>10} {'Incl':>8} {'DLS':>8}  Notes")
    print("-" * 50)

    # Find range to display
    start_display = max(0, incl87_idx - 20)
    end_display = min(len(md), max(incl87_idx, dls_end_idx) + 30)

    for i in range(start_display, end_display):
        notes = []
        if i == incl87_idx:
            notes.append("← incl87")
        if i == dls_start_idx:
            notes.append("← DLS>5 start")
        if i == dls_end_idx:
            notes.append("← DLS end (incl stable)")

        # Find target points (approximately)
        if abs(md[i] - target_87_200) < 10:
            notes.append("≈ 87_200 target")
        if abs(md[i] - target_dls) < 10:
            notes.append("≈ DLS target")

        note_str = " ".join(notes)
        print(f"{md[i]:>10.1f} {incl[i]:>8.2f} {dls[i]:>8.2f}  {note_str}")

    # Show max incl and where it occurs
    max_incl_idx = np.argmax(incl)
    print(f"\nMax incl: {incl[max_incl_idx]:.2f}° at MD={md[max_incl_idx]:.1f}m")

    # Show incl after DLS end
    print(f"\n--- Incl after DLS end (next 20 points) ---")
    for i in range(dls_end_idx, min(dls_end_idx + 20, len(md))):
        print(f"  MD={md[i]:.1f}m: incl={incl[i]:.2f}°")


def main():
    wells_dir = Path('/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de/AG_DATA/InitialData/')

    # Pick a few wells with different deltas
    wells_to_analyze = [
        'Well162~EGFDL.json',      # typical
        'Well522~EGFDL.json',      # problematic
        'Well1898~EGFDU.json',     # worst error
        'Well446_landing~EGFDL.json',  # small delta
    ]

    for well_file in wells_to_analyze:
        well_path = wells_dir / well_file
        if well_path.exists():
            analyze_well(well_path)
        else:
            print(f"Not found: {well_file}")


if __name__ == '__main__':
    main()
