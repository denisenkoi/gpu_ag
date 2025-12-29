#!/usr/bin/env python3
"""Compare ag_well/ag_typewell dumps from StarSteer and GPU slicer."""

import json
import sys
from pathlib import Path

def compare_dumps(ss_path: str, gpu_path: str, tolerance: float = 0.01):
    """Compare two dump files and report differences."""

    with open(ss_path) as f:
        ss = json.load(f)
    with open(gpu_path) as f:
        gpu = json.load(f)

    print("=" * 60)
    print("DUMP COMPARISON")
    print("=" * 60)
    print(f"StarSteer: {ss_path}")
    print(f"GPU slicer: {gpu_path}")
    print()

    all_match = True

    # Compare ag_well
    print("ag_well:")
    for key in ['md_len', 'gr_len', 'step']:
        ss_val = ss['ag_well'].get(key)
        gpu_val = gpu['ag_well'].get(key)
        match = ss_val == gpu_val
        status = "✓" if match else "❌"
        print(f"  {key}: SS={ss_val}, GPU={gpu_val} {status}")
        if not match:
            all_match = False

    # Compare ranges with tolerance
    for key in ['md_range', 'gr_range']:
        ss_val = ss['ag_well'].get(key)
        gpu_val = gpu['ag_well'].get(key)

        if ss_val and gpu_val:
            diff_min = abs(ss_val[0] - gpu_val[0])
            diff_max = abs(ss_val[1] - gpu_val[1])
            match = diff_min < tolerance and diff_max < tolerance
            status = "✓" if match else "❌"
            print(f"  {key}: SS=[{ss_val[0]:.2f}, {ss_val[1]:.2f}], "
                  f"GPU=[{gpu_val[0]:.2f}, {gpu_val[1]:.2f}] "
                  f"(diff: [{diff_min:.4f}, {diff_max:.4f}]) {status}")
            if not match:
                all_match = False

    print()

    # Compare ag_typewell
    print("ag_typewell:")
    for key in ['tvd_len', 'gr_len']:
        ss_val = ss['ag_typewell'].get(key)
        gpu_val = gpu['ag_typewell'].get(key)
        match = ss_val == gpu_val
        status = "✓" if match else "❌"
        print(f"  {key}: SS={ss_val}, GPU={gpu_val} {status}")
        if not match:
            all_match = False

    for key in ['tvd_range', 'gr_range']:
        ss_val = ss['ag_typewell'].get(key)
        gpu_val = gpu['ag_typewell'].get(key)

        if ss_val and gpu_val:
            diff_min = abs(ss_val[0] - gpu_val[0])
            diff_max = abs(ss_val[1] - gpu_val[1])
            match = diff_min < tolerance and diff_max < tolerance
            status = "✓" if match else "❌"
            print(f"  {key}: SS=[{ss_val[0]:.2f}, {ss_val[1]:.2f}], "
                  f"GPU=[{gpu_val[0]:.2f}, {gpu_val[1]:.2f}] "
                  f"(diff: [{diff_min:.4f}, {diff_max:.4f}]) {status}")
            if not match:
                all_match = False

    print()

    # Compare tvd_shift
    ss_shift = ss.get('tvd_shift', 0)
    gpu_shift = gpu.get('tvd_shift', 0)
    diff = abs(ss_shift - gpu_shift)
    match = diff < tolerance
    status = "✓" if match else "❌"
    print(f"tvd_shift: SS={ss_shift:.2f}, GPU={gpu_shift:.2f} (diff: {diff:.4f}) {status}")
    if not match:
        all_match = False

    print()
    print("=" * 60)
    if all_match:
        print("✓ ALL MATCH!")
    else:
        print("❌ DIFFERENCES FOUND")
    print("=" * 60)

    return all_match


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python compare_dumps.py <starsteer_dump.json> <gpu_dump.json> [tolerance]")
        sys.exit(1)

    ss_path = sys.argv[1]
    gpu_path = sys.argv[2]
    tolerance = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

    success = compare_dumps(ss_path, gpu_path, tolerance)
    sys.exit(0 if success else 1)
