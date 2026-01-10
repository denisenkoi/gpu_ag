#!/usr/bin/env python3
"""
Test typelog_preprocessing module - verify overlap metrics.

Expected results:
- Pearson > 0.8 in overlap zone
- Overlap > 100m
"""

import sys
import torch
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from cpu_baseline.preprocessing import prepare_typelog

logging.basicConfig(level=logging.INFO, format='%(message)s')

def test_wells():
    """Test on 4 target wells from specification."""
    wells = [
        'Well1898~EGFDU',
        'Well1416~EGFDU',
        'Well1143~ASTNL',
        'Well1675~EGFDL',
    ]

    print("Loading dataset...")
    ds = torch.load('dataset/gpu_ag_dataset.pt', weights_only=False)

    print("\n" + "=" * 70)
    print("TypeLog Preprocessing - Overlap Metrics Test")
    print("=" * 70)
    print(f"{'Well':<20} {'Overlap':<10} {'Pearson':<10} {'RMSE':<10} {'tvd_shift':<10}")
    print("-" * 70)

    for well_name in wells:
        if well_name not in ds:
            print(f"{well_name:<20} NOT FOUND")
            continue

        well_data = ds[well_name]

        # Get typelog with new preprocessing
        type_tvd, type_gr, meta = prepare_typelog(
            well_data,
            use_pseudo=True,
            apply_smoothing=False
        )

        overlap = meta['overlap_metrics']

        print(f"{well_name:<20} "
              f"{overlap['overlap_length']:>7.0f}m "
              f"{overlap['pearson']:>9.3f} "
              f"{overlap['rmse']:>9.2f} "
              f"{meta['tvd_shift']:>+9.1f}")

    print("-" * 70)
    print("\nExpected: Pearson > 0.8, Overlap > 100m")


if __name__ == '__main__':
    test_wells()
