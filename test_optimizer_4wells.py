#!/usr/bin/env python3
"""
Test full_well_optimizer on 4 target wells with new typelog_preprocessing.
"""

import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from full_well_optimizer import optimize_full_well
from numpy_funcs.interpretation import interpolate_shift_at_md

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/gpu_ag_dataset.pt')

def test_wells():
    wells = [
        'Well1898~EGFDU',
        'Well1416~EGFDU',
        'Well1143~ASTNL',
        'Well1675~EGFDL',
    ]

    print("Loading dataset...")
    ds = torch.load(DATASET_PATH, weights_only=False)

    print("\n" + "=" * 70)
    print("Full Well Optimizer Test - 4 Wells")
    print("=" * 70)
    print(f"{'Well':<20} {'Baseline':<12} {'Optimized':<12} {'Delta':<10} {'Time'}")
    print("-" * 70)

    errors = []
    baseline_errors = []

    for well_name in wells:
        if well_name not in ds:
            print(f"{well_name:<20} NOT FOUND")
            continue

        well_data = ds[well_name]
        well_md = well_data['well_md'].numpy()
        well_tvd = well_data['well_tvd'].numpy()
        log_md = well_data['log_md'].numpy()

        # Reference end shift
        ref_end_md = min(well_md[-1], log_md[-1])
        ref_end_shift = interpolate_shift_at_md(well_data, ref_end_md)

        # Baseline (TVT=const from 87° + 200m)
        baseline_md = float(well_data.get('landing_end_87_200', well_md[len(well_md)//2]))
        baseline_idx = min(int(np.searchsorted(well_md, baseline_md)), len(well_md) - 1)
        tvt_at_baseline = well_tvd[baseline_idx] - interpolate_shift_at_md(well_data, well_md[baseline_idx])
        baseline_shift = well_tvd[np.searchsorted(well_md, ref_end_md) - 1] - tvt_at_baseline
        baseline_error = baseline_shift - ref_end_shift

        # Optimize
        t0 = time.time()
        segments = optimize_full_well(
            well_name, well_data,
            angle_range=2.0,
            angle_step=0.2,
            verbose=False,
        )
        t1 = time.time()

        if not segments:
            print(f"{well_name:<20} FAILED")
            continue

        opt_end_shift = segments[-1].end_shift
        opt_error = opt_end_shift - ref_end_shift

        errors.append(opt_error)
        baseline_errors.append(baseline_error)

        delta = opt_error - baseline_error
        status = "✓" if abs(opt_error) < abs(baseline_error) else "✗"

        print(f"{well_name:<20} {baseline_error:>+10.2f}m {opt_error:>+10.2f}m {delta:>+8.2f}m {status} {t1-t0:.1f}s")

    print("-" * 70)

    if errors:
        rmse_baseline = np.sqrt(np.mean(np.array(baseline_errors)**2))
        rmse_opt = np.sqrt(np.mean(np.array(errors)**2))
        improved = sum(1 for b, o in zip(baseline_errors, errors) if abs(o) < abs(b))

        print(f"\nBaseline RMSE: {rmse_baseline:.2f}m")
        print(f"Optimized RMSE: {rmse_opt:.2f}m")
        print(f"Improved: {improved}/{len(errors)}")


if __name__ == '__main__':
    test_wells()
