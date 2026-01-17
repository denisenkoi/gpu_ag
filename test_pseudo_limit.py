#!/usr/bin/env python3
"""
Test limiting pseudoTypeLog to landing point to avoid data leakage.

Usage:
    python test_pseudo_limit.py --well Well162~EGFDL
"""

import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from starsteer_import import StarSteerImporter
import torch


class PseudoLimitTester(StarSteerImporter):
    """Test pseudoTypeLog limiting."""

    def __init__(self, well_name: str):
        super().__init__(results_dir=Path("/tmp"))
        self.well_name = well_name
        self.json_dir = self.starsteer_dir / "AG_DATA" / "InitialData"
        self.json_path = self.json_dir / f"{well_name}.json"
        self.ag_config_path = self.starsteer_dir / "ag_config.json"

    def _get_non_starred_interpretation_uuid(self) -> tuple:
        """Get first non-starred interpretation UUID."""
        interp_list_file = self.starsteer_dir / "interpretations_list.json"
        if not interp_list_file.exists():
            return None, None

        with open(interp_list_file, 'r') as f:
            data = json.load(f)

        for interp in data.get("interpretations", []):
            if not interp.get("is_starred", False):
                return interp.get("uuid"), interp.get("name")
        return None, None

    def get_pseudo_stats(self) -> dict:
        """Get current pseudoTypeLog statistics."""
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        pseudo = data.get('pseudoTypeLog', {})
        pts = pseudo.get('points', [])
        tvd_pts = pseudo.get('tvdSortedPoints', [])

        if not pts:
            return {'error': 'No pseudoTypeLog points'}

        mds = [p['measuredDepth'] for p in pts]
        tvds = [p['trueVerticalDepth'] for p in tvd_pts] if tvd_pts else []

        return {
            'n_points': len(pts),
            'md_min': min(mds),
            'md_max': max(mds),
            'tvd_min': min(tvds) if tvds else None,
            'tvd_max': max(tvds) if tvds else None,
        }

    def set_pseudo_log_end_md(self, md: float):
        """Set pseudoLogEndMd in ag_config.json."""
        if self.ag_config_path.exists():
            with open(self.ag_config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            config = {}

        if self.well_name not in config:
            config[self.well_name] = {}
        config[self.well_name]['pseudoLogEndMd'] = md

        with open(self.ag_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

        print(f"Set pseudoLogEndMd={md:.2f}m for {self.well_name}")

    def regenerate_json(self, timeout: int = 30) -> bool:
        """Regenerate JSON by starting/stopping AG."""
        old_mtime = self.json_path.stat().st_mtime if self.json_path.exists() else 0

        # Select well
        print(f"Selecting well: {self.well_name}")
        self._send_command("SELECT_WELL", {"wellName": self.well_name})
        time.sleep(2)

        # Wait for interpretations list
        if not self._wait_for_interp_list_update(self.well_name):
            print("interpretations_list.json not updated")
            return False

        # Get non-starred interpretation UUID (AG cannot run on starred)
        interp_uuid, interp_name = self._get_non_starred_interpretation_uuid()
        if not interp_uuid:
            print("No non-starred interpretation found")
            return False
        print(f"Using interpretation: {interp_name} ({interp_uuid})")

        # Select interpretation
        self._send_command("SELECT_INTERPRETATION", {"interpretationUuid": interp_uuid})
        time.sleep(1)

        # Configure AG settings with pseudoLogEndMd
        config = {}
        if self.ag_config_path.exists():
            with open(self.ag_config_path, 'r') as f:
                config = json.load(f)
        pseudo_md = config.get(self.well_name, {}).get('pseudoLogEndMd', None)
        if pseudo_md:
            print(f"Configuring AG with pseudoLogEndMd={pseudo_md:.1f}m")
            self._send_command("CONFIGURE_AG_SETTINGS", {
                "useGridSlice": False,
                "pseudoLogEndMd": pseudo_md
            })
            time.sleep(1)

        # Start AG
        print("Starting AG...")
        self._send_command("START_AG", {})

        # Wait for AG_RUNNING (longer timeout)
        for i in range(100):  # 30 seconds
            state = self._get_starsteer_state()
            if state == "AG_RUNNING":
                print(f"AG started after {i*0.3:.1f}s")
                break
            if i % 10 == 0:
                print(f"  waiting... state={state}")
            time.sleep(0.3)
        else:
            print(f"AG not running after 30s, state: {state}")
            return False

        # Wait for JSON update
        start = time.time()
        while time.time() - start < timeout:
            if self.json_path.exists():
                new_mtime = self.json_path.stat().st_mtime
                if new_mtime > old_mtime:
                    time.sleep(0.5)  # Let file finish writing
                    print(f"JSON updated after {time.time()-start:.1f}s")
                    break
            time.sleep(0.5)
        else:
            print("JSON not updated")

        # Stop AG
        print("Stopping AG...")
        self._send_command("STOP_AG", {}, timeout=30)
        time.sleep(1)

        return True

    def _get_starsteer_state(self) -> str:
        """Get current StarSteer state."""
        if not self.status_file.exists():
            return "NOT_RUNNING"
        with open(self.status_file, 'r') as f:
            status = json.load(f)
        return status.get("application_state", {}).get("state", "UNKNOWN")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--well', type=str, default='Well162~EGFDL')
    args = parser.parse_args()

    # Load dataset to get landing_end_dls
    dataset = torch.load('/mnt/e/Projects/Rogii/gpu_ag/data/wells_fresh.pt', weights_only=False)
    well_data = dataset.get(args.well)
    if well_data is None:
        print(f"Well {args.well} not in dataset")
        return 1

    landing_dls = well_data.get('landing_end_dls', 0)
    landing_87 = well_data.get('landing_end_87_200', 0)
    print(f"Well: {args.well}")
    print(f"landing_end_87_200: {landing_87:.1f}m")
    print(f"landing_end_dls: {landing_dls:.1f}m")

    tester = PseudoLimitTester(args.well)

    # Get BEFORE stats
    print("\n=== BEFORE ===")
    before = tester.get_pseudo_stats()
    print(f"Points: {before['n_points']}")
    print(f"MD range: {before['md_min']:.1f} - {before['md_max']:.1f}m")
    if before.get('tvd_max'):
        print(f"TVD range: {before['tvd_min']:.1f} - {before['tvd_max']:.1f}m")

    # Set limit and regenerate
    # Use incl87 point (before +200m) as the limit for vertical part
    # Actually, pseudoLogEndMd should limit the vertical section used for pseudo
    # Let's try landing_end_dls first
    limit_md = landing_dls
    print(f"\n=== Setting pseudoLogEndMd = {limit_md:.1f}m ===")
    tester.set_pseudo_log_end_md(limit_md)

    if not tester.regenerate_json():
        print("Failed to regenerate JSON")
        return 1

    # Get AFTER stats
    print("\n=== AFTER ===")
    after = tester.get_pseudo_stats()
    print(f"Points: {after['n_points']}")
    print(f"MD range: {after['md_min']:.1f} - {after['md_max']:.1f}m")
    if after.get('tvd_max'):
        print(f"TVD range: {after['tvd_min']:.1f} - {after['tvd_max']:.1f}m")

    # Compare
    print("\n=== COMPARISON ===")
    print(f"Points: {before['n_points']} -> {after['n_points']} ({after['n_points'] - before['n_points']:+d})")
    print(f"MD max: {before['md_max']:.1f} -> {after['md_max']:.1f} ({after['md_max'] - before['md_max']:+.1f}m)")
    if before.get('tvd_max') and after.get('tvd_max'):
        print(f"TVD max: {before['tvd_max']:.1f} -> {after['tvd_max']:.1f} ({after['tvd_max'] - before['tvd_max']:+.1f}m)")

    return 0


if __name__ == '__main__':
    sys.exit(main())
