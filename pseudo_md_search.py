#!/usr/bin/env python3
"""
Search for optimal pseudoLogEndMd by iterating through different MD values.

Cycle:
1. Set pseudoLogEndMd in ag_config.json
2. Start AG (StarSteer exports new JSON with trimmed pseudoTypeLog)
3. Wait for fresh JSON
4. Create dataset from JSON
5. Run optimization
6. Measure accuracy
7. Record to CSV
8. Stop AG
9. Decrease MD and repeat

Usage:
    python pseudo_md_search.py --well Well1675~EGFDL --n-points 10 --step 2.0
"""

import sys
import os
import json
import time
import csv
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

import torch
import numpy as np

from starsteer_import import StarSteerImporter
from dataset.json_to_torch import process_single_json, create_landing_detector
from python_normalization.normalization_calculator import NormalizationCalculator
from full_well_optimizer import optimize_full_well, detect_gpu_model, get_gpu_free_memory_gb
from numpy_funcs.interpretation import interpolate_shift_at_md

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PseudoMdSearcher(StarSteerImporter):
    """Search for optimal pseudoLogEndMd."""

    def __init__(self, well_name: str, env_file: Path = None):
        """
        Initialize searcher.

        Args:
            well_name: Well to search (e.g., "Well1675~EGFDL")
            env_file: Path to .env file
        """
        # Use empty results_dir since we don't import interpretations
        super().__init__(results_dir=Path("/tmp"), env_file=env_file)
        self.well_name = well_name

        # JSON file path
        self.json_dir = self.starsteer_dir / "AG_DATA" / "InitialData"
        self.json_path = self.json_dir / f"{well_name}.json"

        # AG config path
        self.ag_config_path = self.starsteer_dir / "ag_config.json"

        # Results CSV
        self.results_dir = Path(__file__).parent / "results" / "pseudo_md_search"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Detectors for dataset creation
        self.landing_detector = create_landing_detector()
        self.norm_calculator = NormalizationCalculator(interactive_mode=False)

        logger.info(f"PseudoMdSearcher initialized for {well_name}")
        logger.info(f"JSON path: {self.json_path}")
        logger.info(f"AG config: {self.ag_config_path}")

    def set_pseudo_log_end_md(self, md: float):
        """
        Set pseudoLogEndMd in ag_config.json.

        Args:
            md: MD value to set
        """
        # Read existing config
        if self.ag_config_path.exists():
            with open(self.ag_config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            config_data = {}

        # Update pseudoLogEndMd for this well
        if self.well_name not in config_data:
            config_data[self.well_name] = {}
        config_data[self.well_name]['pseudoLogEndMd'] = md

        # Write back
        with open(self.ag_config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Set pseudoLogEndMd={md:.2f}m for {self.well_name}")

    def get_json_mtime(self) -> float:
        """Get JSON file modification time."""
        if self.json_path.exists():
            return self.json_path.stat().st_mtime
        return 0

    def wait_for_fresh_json(self, old_mtime: float, timeout: int = 60) -> bool:
        """
        Wait for JSON file to be updated.

        Args:
            old_mtime: Previous modification time
            timeout: Max wait time in seconds

        Returns:
            True if JSON was updated
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            current_mtime = self.get_json_mtime()
            if current_mtime > old_mtime:
                # Wait a bit more for file to be fully written
                time.sleep(0.5)
                logger.info(f"JSON updated (age: {time.time() - current_mtime:.1f}s)")
                return True
            time.sleep(0.5)

        logger.warning(f"JSON not updated after {timeout}s")
        return False

    def create_dataset_from_json(self) -> Optional[Dict[str, Any]]:
        """
        Create dataset entry from current JSON file.

        Returns:
            dict with tensors or None if failed
        """
        if not self.json_path.exists():
            logger.error(f"JSON not found: {self.json_path}")
            return None

        result = process_single_json(
            self.json_path,
            device='cpu',
            landing_detector=self.landing_detector,
            norm_calculator=self.norm_calculator
        )

        if result:
            logger.info(f"Dataset created: pseudo_tvd={len(result['pseudo_tvd'])} pts, "
                       f"type_tvd={len(result['type_tvd'])} pts")

        return result

    def run_optimization(self, well_data: Dict[str, Any], angle_range: float = 2.0) -> tuple:
        """
        Run full well optimization.

        Args:
            well_data: Dataset entry
            angle_range: Angle range for optimization

        Returns:
            (segments, opt_error, baseline_error)
        """
        from full_well_optimizer import optimize_full_well

        # Get reference end shift
        well_md = well_data['well_md'].numpy()
        log_md = well_data['log_md'].numpy()
        ref_end_md = min(well_md[-1], log_md[-1])
        ref_end_shift = interpolate_shift_at_md(well_data, ref_end_md)

        # Baseline (TVT=const from landing_end_87_200)
        well_tvd = well_data['well_tvd'].numpy()
        baseline_md = float(well_data.get('landing_end_87_200', well_md[len(well_md)//2]))
        baseline_idx = min(int(np.searchsorted(well_md, baseline_md)), len(well_md) - 1)
        tvt_at_baseline = well_tvd[baseline_idx] - interpolate_shift_at_md(well_data, well_md[baseline_idx])
        baseline_shift = well_tvd[np.searchsorted(well_md, ref_end_md) - 1] - tvt_at_baseline
        baseline_error = baseline_shift - ref_end_shift

        # Optimize
        segments = optimize_full_well(
            self.well_name,
            well_data,
            angle_range=angle_range,
            verbose=False,
        )

        if not segments:
            return None, None, baseline_error

        opt_end_shift = segments[-1].end_shift
        opt_error = opt_end_shift - ref_end_shift

        return segments, opt_error, baseline_error

    def run_single_iteration(self, pseudo_md: float, angle_range: float = 2.0) -> Dict[str, Any]:
        """
        Run single iteration of the search.

        Args:
            pseudo_md: pseudoLogEndMd value
            angle_range: Angle range for optimization

        Returns:
            dict with results
        """
        result = {
            'pseudo_md': pseudo_md,
            'started_at': datetime.now().isoformat(),
            'status': 'pending',
        }

        try:
            # 1. Set pseudoLogEndMd in ag_config.json
            self.set_pseudo_log_end_md(pseudo_md)

            # 2. Send CONFIGURE_AG_SETTINGS to reload config
            logger.info("Reloading AG config...")
            self._send_command("CONFIGURE_AG_SETTINGS", {
                "useGridSlice": False,
                "gridName": "",
                "pseudoLogEndMd": pseudo_md  # Also pass in command
            })
            time.sleep(0.5)

            # 3. Get current JSON mtime
            old_mtime = self.get_json_mtime()

            # 4. Start AG
            logger.info("Starting AG...")
            self._send_command("START_AG", {})

            # Wait for AG to start with longer timeout
            for attempt in range(50):  # 50 * 0.3s = 15s
                state = self._get_starsteer_state()
                if state == "AG_RUNNING":
                    logger.info(f"AG started after {attempt * 0.3:.1f}s")
                    break
                if attempt == 0 or attempt % 10 == 0:
                    logger.debug(f"Waiting for AG_RUNNING, current state: {state}")
                time.sleep(0.3)
            else:
                raise TimeoutError(f"AG not running after 15s, state: {state}")

            # 4. Wait for fresh JSON
            if not self.wait_for_fresh_json(old_mtime, timeout=30):
                result['status'] = 'json_timeout'
                self._send_command("STOP_AG", {}, timeout=30)
                return result

            # 5. Stop AG (we have the JSON)
            logger.info("Stopping AG...")
            self._send_command("STOP_AG", {}, timeout=30)
            time.sleep(1)  # Let it settle

            # 6. Create dataset
            well_data = self.create_dataset_from_json()
            if well_data is None:
                result['status'] = 'dataset_failed'
                return result

            result['pseudo_tvd_len'] = len(well_data['pseudo_tvd'])
            result['type_tvd_len'] = len(well_data['type_tvd'])

            # 7. Run optimization
            t_opt_start = time.time()
            segments, opt_error, baseline_error = self.run_optimization(well_data, angle_range)
            opt_time_ms = int((time.time() - t_opt_start) * 1000)

            if segments is None:
                result['status'] = 'opt_failed'
                return result

            result['n_segments'] = len(segments)
            result['baseline_error'] = baseline_error
            result['opt_error'] = opt_error
            result['opt_time_ms'] = opt_time_ms
            result['status'] = 'success'

        except Exception as e:
            logger.error(f"Iteration failed: {e}")
            result['status'] = f'error: {e}'
            # Try to stop AG
            try:
                self._send_command("STOP_AG", {}, timeout=10)
            except:
                pass

        result['finished_at'] = datetime.now().isoformat()
        return result

    def search(self, start_md: float, step: float, n_points: int, angle_range: float = 2.0) -> list:
        """
        Run search over multiple MD values.

        Args:
            start_md: Starting MD (e.g., landing_87_200)
            step: Step to decrease MD (positive value)
            n_points: Number of points to test
            angle_range: Angle range for optimization

        Returns:
            list of result dicts
        """
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = self.results_dir / f"search_{self.well_name}_{run_id}.csv"

        logger.info("=" * 70)
        logger.info(f"Pseudo MD Search: {self.well_name}")
        logger.info(f"Start MD: {start_md:.2f}m, step: -{step}m, points: {n_points}")
        logger.info(f"Run ID: {run_id}")
        logger.info("=" * 70)

        # Ensure StarSteer is ready
        if not self._is_starsteer_running():
            logger.error("StarSteer not running! Please start it manually.")
            return []

        # Select well
        logger.info(f"Selecting well: {self.well_name}")
        self._send_command("SELECT_WELL", {"wellName": self.well_name})
        time.sleep(2)

        # Wait for interpretations_list.json to update
        if not self._wait_for_interp_list_update(self.well_name):
            logger.error("interpretations_list.json not updated")
            return []

        # Get Manual interpretation UUID
        manual_uuid = self._get_manual_interpretation_uuid()
        if not manual_uuid:
            logger.error("Manual interpretation not found")
            return []
        logger.info(f"Manual UUID: {manual_uuid}")

        # Check if AI_GPU_search exists, otherwise create it
        search_interp_name = "AI_GPU_search"
        existing_uuid = self._get_interpretation_uuid(search_interp_name)

        if existing_uuid:
            logger.info(f"Using existing {search_interp_name}: {existing_uuid}")
            target_uuid = existing_uuid
        else:
            logger.info(f"Creating {search_interp_name} from Manual...")
            self._send_command("COPY_INTERPRETATION", {
                "sourceInterpretationUuid": manual_uuid,
                "targetInterpretationName": search_interp_name
            })
            time.sleep(1)
            target_uuid = self._get_copy_result_uuid()
            if not target_uuid:
                logger.error(f"{search_interp_name} not created")
                return []
            logger.info(f"Created: {target_uuid}")

        # Select the non-starred interpretation
        logger.info(f"Selecting {search_interp_name}...")
        self._send_command("SELECT_INTERPRETATION", {"interpretationUuid": target_uuid})
        time.sleep(1)

        # Configure AG settings (minimal)
        logger.info("Configuring AG settings...")
        self._send_command("CONFIGURE_AG_SETTINGS", {
            "useGridSlice": False,
            "gridName": ""
        })

        # CSV header
        csv_header = [
            'run_id', 'well_name', 'pseudo_md', 'baseline_error', 'opt_error',
            'n_segments', 'pseudo_tvd_len', 'type_tvd_len', 'opt_time_ms',
            'status', 'started_at', 'finished_at'
        ]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            writer.writeheader()

        logger.info(f"CSV: {csv_path}")
        logger.info("-" * 70)

        results = []
        current_md = start_md

        for i in range(n_points):
            logger.info(f"\n[{i+1}/{n_points}] Testing pseudo_md={current_md:.2f}m")

            result = self.run_single_iteration(current_md, angle_range)
            result['run_id'] = run_id
            result['well_name'] = self.well_name
            results.append(result)

            # Write to CSV immediately
            with open(csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_header)
                row = {k: result.get(k, '') for k in csv_header}
                if 'baseline_error' in result and result['baseline_error'] is not None:
                    row['baseline_error'] = f"{result['baseline_error']:.2f}"
                if 'opt_error' in result and result['opt_error'] is not None:
                    row['opt_error'] = f"{result['opt_error']:.2f}"
                writer.writerow(row)

            # Print result
            if result['status'] == 'success':
                logger.info(f"  base={result['baseline_error']:+.2f}m, opt={result['opt_error']:+.2f}m, "
                           f"time={result['opt_time_ms']}ms")
            else:
                logger.info(f"  status={result['status']}")

            # Next MD
            current_md -= step

        logger.info("\n" + "=" * 70)
        logger.info(f"Search complete: {len(results)} points")
        logger.info(f"CSV: {csv_path}")

        # Print summary
        success_results = [r for r in results if r['status'] == 'success']
        if success_results:
            best = min(success_results, key=lambda r: abs(r['opt_error']))
            logger.info(f"Best: pseudo_md={best['pseudo_md']:.2f}m, opt_error={best['opt_error']:+.2f}m")

        return results


def main():
    parser = argparse.ArgumentParser(description='Search for optimal pseudoLogEndMd')
    parser.add_argument('--well', type=str, required=True,
                        help='Well name (e.g., Well1675~EGFDL)')
    parser.add_argument('--start-md', type=float, default=None,
                        help='Starting MD (default: landing_87_200 from dataset)')
    parser.add_argument('--step', type=float, default=2.0,
                        help='Step to decrease MD (default: 2.0m)')
    parser.add_argument('--n-points', type=int, default=10,
                        help='Number of points to test (default: 10)')
    parser.add_argument('--angle-range', type=float, default=2.0,
                        help='Angle range for optimization (default: 2.0)')
    parser.add_argument('--env', type=str, default=None,
                        help='Path to .env file')

    args = parser.parse_args()

    # GPU info
    gpu_model = detect_gpu_model()
    free_gb = get_gpu_free_memory_gb()
    logger.info(f"GPU: {gpu_model}, Free memory: {free_gb:.1f}GB")

    # Create searcher
    env_file = Path(args.env) if args.env else None
    searcher = PseudoMdSearcher(args.well, env_file)

    # Get start_md from pseudoTypeLog max MD if not specified
    start_md = args.start_md
    if start_md is None:
        # Get max MD from current pseudoTypeLog in JSON
        json_path = searcher.json_path
        if json_path.exists():
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            pseudo_pts = data.get('pseudoTypeLog', {}).get('points', [])
            if pseudo_pts:
                pseudo_mds = [p['measuredDepth'] for p in pseudo_pts]
                start_md = max(pseudo_mds)
                logger.info(f"Using max(pseudoTypeLog.md) from JSON: {start_md:.2f}m")
            else:
                logger.error(f"No pseudoTypeLog points in {json_path}")
                sys.exit(1)
        else:
            logger.error(f"JSON not found: {json_path}")
            sys.exit(1)

    # Run search
    results = searcher.search(
        start_md=start_md,
        step=args.step,
        n_points=args.n_points,
        angle_range=args.angle_range
    )


if __name__ == '__main__':
    main()
