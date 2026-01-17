#!/usr/bin/env python3
"""
Regenerate JSON files with limited pseudoTypeLog (no leakage).

Sets pseudo_log_end_md = TVD_at_landing + offset for each well.
"""

import json
import time
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from starsteer_import import StarSteerImporter

# Offset to compensate for meters->feet conversion in project
UNIT_OFFSET = 73.0  # meters


class PseudoRegenerator(StarSteerImporter):
    def __init__(self):
        super().__init__(results_dir=Path('/tmp'))
        self.ag_config_path = self.starsteer_dir / "ag_config.json"

    def get_tvd_at_landing(self, well_data: dict) -> float:
        """Get TVD at landing point."""
        if 'landing_end_dls' in well_data:
            landing_md = well_data['landing_end_dls']
        else:
            landing_md = well_data['landing_end_87_200']

        mds = well_data['well_md']
        tvds = well_data['well_tvd']
        mds = mds.numpy() if torch.is_tensor(mds) else np.array(mds)
        tvds = tvds.numpy() if torch.is_tensor(tvds) else np.array(tvds)

        idx = np.argmin(np.abs(mds - landing_md))
        return float(tvds[idx])

    def set_pseudo_limit(self, well_name: str, tvd_at_landing: float):
        """Set pseudo_log_end_md in ag_config.json."""
        if self.ag_config_path.exists():
            with open(self.ag_config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}

        # pseudo_log_end_md = TVD at landing + offset
        pseudo_limit = tvd_at_landing + UNIT_OFFSET

        if well_name not in config:
            config[well_name] = {}
        config[well_name]['pseudo_log_end_md'] = pseudo_limit

        with open(self.ag_config_path, 'w') as f:
            json.dump(config, f, indent=2)

        return pseudo_limit

    def get_non_starred_interp(self) -> Optional[str]:
        """Get first non-starred interpretation UUID."""
        interp_list = self.starsteer_dir / "interpretations_list.json"
        if not interp_list.exists():
            return None
        with open(interp_list, 'r') as f:
            data = json.load(f)
        for i in data.get('interpretations', []):
            if not i.get('is_starred'):
                return i['uuid']
        return None

    def regenerate_well(self, well_name: str, tvd_at_landing: float) -> bool:
        """Regenerate JSON for one well with limited pseudo."""
        # Set limit
        pseudo_limit = self.set_pseudo_limit(well_name, tvd_at_landing)

        # Select well
        self._send_command('SELECT_WELL', {'wellName': well_name})
        time.sleep(1.5)

        if not self._wait_for_interp_list_update(well_name):
            print(f"  ERROR: interp list not updated")
            return False

        # Get non-starred interpretation
        interp_uuid = self.get_non_starred_interp()
        if not interp_uuid:
            print(f"  ERROR: no non-starred interpretation")
            return False

        # Select interpretation
        self._send_command('SELECT_INTERPRETATION', {'interpretationUuid': interp_uuid})
        time.sleep(0.5)

        # Configure AG (this reads pseudo_log_end_md from ag_config.json)
        self._send_command('CONFIGURE_AG_SETTINGS', {'useGridSlice': False})
        time.sleep(0.5)

        # Start AG to generate new JSON
        self._send_command('START_AG', {})
        time.sleep(2)

        # Stop AG
        self._send_command('STOP_AG', {}, timeout=30)
        time.sleep(0.5)

        return True


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--well', type=str, help='Single well')
    parser.add_argument('--all', action='store_true', help='All wells')
    args = parser.parse_args()

    if not args.well and not args.all:
        parser.error('Specify --well or --all')

    # Load dataset
    dataset = torch.load('/mnt/e/Projects/Rogii/gpu_ag/data/wells_fresh.pt', weights_only=False)

    regenerator = PseudoRegenerator()
    json_dir = regenerator.starsteer_dir / 'AG_DATA/InitialData'

    wells = [args.well] if args.well else list(dataset.keys())

    success = 0
    failed = 0

    for i, well_name in enumerate(wells):
        if well_name not in dataset:
            print(f"[{i+1}/{len(wells)}] {well_name}: not in dataset, skip")
            continue

        json_path = json_dir / f"{well_name}.json"
        if not json_path.exists():
            print(f"[{i+1}/{len(wells)}] {well_name}: JSON not found, skip")
            continue

        well_data = dataset[well_name]
        tvd_at_landing = regenerator.get_tvd_at_landing(well_data)

        print(f"[{i+1}/{len(wells)}] {well_name}: TVD@landing={tvd_at_landing:.1f}m")

        # Get before stats
        with open(json_path, 'r') as f:
            before = json.load(f)
        pseudo_before = before.get('pseudoTypeLog', {}).get('tvdSortedPoints', [])
        tvd_max_before = pseudo_before[-1]['trueVerticalDepth'] if pseudo_before else 0

        if regenerator.regenerate_well(well_name, tvd_at_landing):
            # Get after stats
            with open(json_path, 'r') as f:
                after = json.load(f)
            pseudo_after = after.get('pseudoTypeLog', {}).get('tvdSortedPoints', [])
            tvd_max_after = pseudo_after[-1]['trueVerticalDepth'] if pseudo_after else 0

            leakage_before = tvd_max_before - tvd_at_landing
            leakage_after = tvd_max_after - tvd_at_landing

            print(f"  TVD max: {tvd_max_before:.1f} -> {tvd_max_after:.1f}m, leakage: {leakage_before:+.1f} -> {leakage_after:+.1f}m")
            success += 1
        else:
            print(f"  FAILED")
            failed += 1

    print(f"\nDone: {success} success, {failed} failed")


if __name__ == '__main__':
    main()
