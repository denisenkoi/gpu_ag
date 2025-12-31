#!/usr/bin/env python3
"""
Import interpretations from true_gpu_slicer results into StarSteer source wells.

Usage:
    python starsteer_import.py --results-dir results/optuna/trial_0011/work
    python starsteer_import.py --results-dir results/optuna/trial_0011/work --well Well162~EGFDL
    python starsteer_import.py --list --results-dir results/optuna/trial_0011/work

Process:
    1. Launch StarSteer (via trigger if WSL)
    2. Open project (SOLO_PROJECT_NAME from .env - should have _local suffix)
    3. For each well with saved interpretation:
       - SELECT_WELL by name
       - COPY_INTERPRETATION (Manual -> AI_GPU_local)
       - SELECT_INTERPRETATION
       - START_AG
       - Write interpretation.json with our segments
       - STOP_AG
"""

import argparse
import json
import os
import sys
import time
import platform
from pathlib import Path
from typing import Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add cpu_baseline for imports
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from dotenv import load_dotenv


class StarSteerImporter:
    """Import interpretations into StarSteer source wells."""

    def __init__(self, results_dir: Path, env_file: Path = None, interp_name: str = None):
        """
        Initialize importer.

        Args:
            results_dir: Directory with saved interpretations (work/)
            env_file: Path to .env file (default: gpu_ag/.env)
            interp_name: Name for imported interpretation (default: AI_GPU_local)
        """
        self.results_dir = Path(results_dir)

        # Load environment
        if env_file is None:
            env_file = Path(__file__).parent / ".env"
        load_dotenv(env_file)
        logger.info(f"Loaded .env from: {env_file}")

        # Get config from .env
        self.project_name = os.getenv("SOLO_PROJECT_NAME")
        starsteer_dir_str = os.getenv("STARSTEER_DIR", "")

        if not self.project_name:
            raise ValueError("SOLO_PROJECT_NAME not set in .env")

        logger.info(f"Project: {self.project_name}")

        # Convert Windows path to WSL if needed
        if platform.system() == 'Linux' and starsteer_dir_str.startswith("E:"):
            starsteer_dir_str = starsteer_dir_str.replace("E:", "/mnt/e").replace("\\", "/")

        self.starsteer_dir = Path(starsteer_dir_str)
        logger.info(f"StarSteer dir: {self.starsteer_dir}")

        # Control files
        self.commands_file = self.starsteer_dir / "commands.json"
        self.status_file = self.starsteer_dir / "status.json"
        self.interp_list_file = self.starsteer_dir / "interpretations_list.json"
        self.interp_file = self.starsteer_dir / "interpretation.json"

        # Interpretation name for imported results
        self.import_interp_name = interp_name or os.getenv("STARSTEER_IMPORT_NAME", "AI_GPU_local")

    def list_available_wells(self) -> List[str]:
        """List wells with saved interpretations."""
        wells = []
        if self.results_dir.exists():
            for f in self.results_dir.glob("*.json"):
                if f.stem.startswith("Well"):
                    wells.append(f.stem)
        return sorted(wells)

    def _is_starsteer_running(self) -> bool:
        """Check if StarSteer is running by checking status.json freshness."""
        if not self.status_file.exists():
            return False

        try:
            # Check if status.json is fresh (updated in last 60 seconds)
            mtime = self.status_file.stat().st_mtime
            age = time.time() - mtime
            if age > 60:
                logger.info(f"status.json is stale ({age:.0f}s old) - StarSteer not running")
                return False

            with open(self.status_file, 'r') as f:
                status = json.load(f)
            state = status.get("application_state", {}).get("state", "")
            # All valid states that indicate StarSteer is running
            valid_states = [
                "IDLE", "PROJECT_LOADED", "AG_RUNNING",
                "WAITING_PROJECT_SELECTION", "WELL_SELECTED",
                "INTERPRETATION_SELECTED", "AG_CONFIGURED"
            ]
            return state in valid_states
        except (json.JSONDecodeError, IOError):
            return False

    def _launch_starsteer_via_trigger(self):
        """Launch StarSteer via task runner trigger (for WSL)."""
        from path_utils import normalize_path

        trigger_dir = Path(normalize_path("E:/Projects/Rogii/sc/task_queue"))
        trigger_dir.mkdir(parents=True, exist_ok=True)

        timestamp = int(time.time())
        task_id = f"task_{timestamp}_starsteer"
        trigger_file = trigger_dir / f"{task_id}.json"

        # Select BAT file
        starsteer_dir_str = str(self.starsteer_dir)
        if "slicer_de_2" in starsteer_dir_str:
            bat_file = "runss_slicer_de_2.bat"
        else:
            bat_file = "runss_slicer_de.bat"

        trigger_data = {
            "task_id": task_id,
            "type": "run_bat",
            "script_path": bat_file,
            "bot_id": "SSAndAG",
            "created_at": timestamp
        }

        with open(trigger_file, 'w') as f:
            json.dump(trigger_data, f)

        logger.info(f"Created StarSteer trigger: {trigger_file}")
        logger.info("Waiting for task runner to pick up trigger...")

        # Wait for trigger to be processed
        wait_start = time.time()
        while trigger_file.exists():
            if time.time() - wait_start > 30:
                logger.warning("Trigger not picked up in 30s, continuing anyway...")
                break
            time.sleep(1)

        logger.info("StarSteer trigger processed")

    def _wait_starsteer_ready(self, timeout: int = 180):
        """Wait for StarSteer to be ready."""
        logger.info("Waiting for StarSteer to be ready...")

        # Initial wait for StarSteer to start (like in slicer.py)
        time.sleep(30)

        start_time = time.time()

        while time.time() - start_time < timeout:
            if self._is_starsteer_running():
                logger.info("✓ StarSteer is ready")
                return
            time.sleep(2)

        raise TimeoutError(f"StarSteer not ready after {timeout}s")

    def _close_starsteer(self, timeout: int = 60):
        """Close StarSteer via CLOSE_APP command."""
        if not self._is_starsteer_running():
            logger.info("StarSteer not running, nothing to close")
            return

        logger.info("Closing StarSteer via CLOSE_APP...")

        # Delete commands.json first
        if self.commands_file.exists():
            self.commands_file.unlink()

        # Write CLOSE_APP command
        cmd_data = {
            "command": "CLOSE_APP",
            "params": {"save_changes": False}
        }
        with open(self.commands_file, 'w') as f:
            json.dump(cmd_data, f, indent=2)

        # Wait for StarSteer to close (status.json becomes stale)
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self._is_starsteer_running():
                logger.info("✓ StarSteer closed")
                # Clean up commands.json
                if self.commands_file.exists():
                    try:
                        self.commands_file.unlink()
                    except:
                        pass
                return
            time.sleep(2)

        logger.warning(f"StarSteer not closed after {timeout}s")

    def _ensure_starsteer_running(self):
        """Ensure StarSteer is running, launch if needed."""
        # First close any existing StarSteer to avoid conflicts
        if self._is_starsteer_running():
            logger.info("Closing existing StarSteer to avoid conflicts...")
            self._close_starsteer()
            time.sleep(5)  # Wait a bit after closing

        # Now launch fresh StarSteer
        logger.info("Launching fresh StarSteer...")
        if platform.system() == 'Linux':
            self._launch_starsteer_via_trigger()
        else:
            raise NotImplementedError("Direct Windows launch not implemented")
        self._wait_starsteer_ready()

    def _send_command(self, command: str, params: dict, timeout: int = 300):
        """Send command to StarSteer via commands.json."""
        logger.info(f"Sending command: {command}")

        # Delete file first (StarSteer watches for file creation)
        if self.commands_file.exists():
            self.commands_file.unlink()

        cmd_data = {
            "command": command,
            "params": params
        }

        # Write command file
        with open(self.commands_file, 'w') as f:
            json.dump(cmd_data, f, indent=2)

        # Wait for command to be processed (file deleted by StarSteer)
        start_time = time.time()
        while self.commands_file.exists():
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Command {command} not processed in {timeout}s")
            time.sleep(0.5)

        # Wait for status.json to be updated with result of THIS command (10 retries * 0.1s = 1s max)
        for _ in range(10):
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                result = status.get("last_command_result", {})
                if result.get("command") == command:
                    return
            except (json.JSONDecodeError, IOError):
                pass
            time.sleep(0.1)

    def _get_starsteer_state(self) -> str:
        """Get current StarSteer state from status.json."""
        if not self.status_file.exists():
            return "UNKNOWN"

        try:
            with open(self.status_file, 'r') as f:
                status = json.load(f)
            return status.get("application_state", {}).get("state", "UNKNOWN")
        except (json.JSONDecodeError, IOError):
            return "UNKNOWN"

    def _wait_for_state(self, expected_state: str, timeout: int = 300):
        """Wait for StarSteer to reach expected state."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            current_state = self._get_starsteer_state()
            if current_state == expected_state:
                return
            time.sleep(2)

        raise TimeoutError(f"Did not reach state {expected_state} in {timeout}s")

    def _wait_for_project_loaded(self, timeout: int = 300):
        """Wait for project to be loaded (any valid state)."""
        project_loaded_states = [
            "PROJECT_LOADED", "WELL_SELECTED",
            "INTERPRETATION_SELECTED", "AG_CONFIGURED", "AG_RUNNING"
        ]
        start_time = time.time()

        while time.time() - start_time < timeout:
            current_state = self._get_starsteer_state()
            if current_state in project_loaded_states:
                return
            time.sleep(2)

        raise TimeoutError(f"Project not loaded after {timeout}s")

    def _open_project(self):
        """Open project in StarSteer."""
        current_state = self._get_starsteer_state()

        # States that indicate project is already loaded
        project_loaded_states = [
            "PROJECT_LOADED", "WELL_SELECTED",
            "INTERPRETATION_SELECTED", "AG_CONFIGURED", "AG_RUNNING"
        ]

        if current_state in project_loaded_states:
            logger.info(f"✓ Project already loaded (state: {current_state})")
        elif current_state == "WAITING_PROJECT_SELECTION":
            logger.info(f"Opening project: {self.project_name}")

            self._send_command("OPEN_PROJECT", {
                "projectName": self.project_name
            }, timeout=600)

            # Wait for any state that indicates project loaded
            self._wait_for_project_loaded(timeout=300)
            logger.info("✓ Project loaded")
        else:
            logger.warning(f"Unknown state: {current_state}, trying to open project anyway")
            self._send_command("OPEN_PROJECT", {
                "projectName": self.project_name
            }, timeout=600)
            self._wait_for_project_loaded(timeout=300)
            logger.info("✓ Project loaded")

    def _get_manual_interpretation_uuid(self) -> Optional[str]:
        """Get Manual interpretation UUID from current well."""
        if not self.interp_list_file.exists():
            return None

        with open(self.interp_list_file, 'r') as f:
            data = json.load(f)

        for interp in data.get("interpretations", []):
            name = interp.get("name", "")
            if name.lower() == "manual":
                return interp.get("uuid")

        return None

    def _get_interpretation_uuid(self, name: str) -> Optional[str]:
        """Get UUID of interpretation by name from interpretations_list.json."""
        if not self.interp_list_file.exists():
            return None

        try:
            with open(self.interp_list_file, 'r') as f:
                data = json.load(f)

            for interp in data.get("interpretations", []):
                if interp.get("name") == name:
                    return interp.get("uuid")
        except (json.JSONDecodeError, IOError):
            pass
        return None

    def _get_copy_result_uuid(self) -> Optional[str]:
        """Get new interpretation UUID from COPY_INTERPRETATION result in status.json."""
        try:
            with open(self.status_file, 'r') as f:
                status = json.load(f)

            result = status.get("last_command_result", {})
            if result.get("command") == "COPY_INTERPRETATION" and result.get("status") == "SUCCESS":
                data = result.get("data", {})
                return data.get("new_uuid")
        except (json.JSONDecodeError, IOError):
            pass
        return None

    def _wait_for_interp_list_update(self, well_name: str, retries: int = 20, delay: float = 0.2) -> bool:
        """Wait for interpretations_list.json to update for the new well."""
        for _ in range(retries):
            if self.interp_list_file.exists():
                try:
                    with open(self.interp_list_file, 'r') as f:
                        data = json.load(f)
                    if data.get("well_name") == well_name:
                        return True
                except (json.JSONDecodeError, IOError):
                    pass
            time.sleep(delay)
        return False

    def _wait_for_interpretation(self, name: str, timeout: float = 10) -> Optional[str]:
        """Wait for interpretation to appear in list."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.interp_list_file.exists():
                with open(self.interp_list_file, 'r') as f:
                    data = json.load(f)

                for interp in data.get("interpretations", []):
                    if interp.get("name") == name:
                        return interp.get("uuid")

            time.sleep(0.5)

        return None

    def _wait_for_interpretation_active(self, uuid: str, retries: int = 20, delay: float = 0.2) -> bool:
        """Wait for interpretation to become active (is_active=true)."""
        for _ in range(retries):
            if self.interp_list_file.exists():
                try:
                    with open(self.interp_list_file, 'r') as f:
                        data = json.load(f)
                    for interp in data.get("interpretations", []):
                        if interp.get("uuid") == uuid and interp.get("is_active"):
                            return True
                except (json.JSONDecodeError, IOError):
                    pass
            time.sleep(delay)
        return False

    def _wait_for_ag_running(self, retries: int = 20, delay: float = 0.2):
        """Wait for AG to start running with retry logic."""
        for _ in range(retries):
            state = self._get_starsteer_state()
            if state == "AG_RUNNING":
                return
            time.sleep(delay)

        raise TimeoutError(f"AG not running after {retries} retries")

    def _load_interpretation(self, well_name: str) -> Optional[Dict]:
        """Load saved interpretation for well."""
        interp_file = self.results_dir / f"{well_name}.json"
        if not interp_file.exists():
            return None

        with open(interp_file, 'r') as f:
            return json.load(f)

    def _write_interpretation(self, interp_data: Dict):
        """Write interpretation.json to trigger QFileSystemWatcher."""
        tmp_file = self.interp_file.with_suffix('.tmp')

        with open(tmp_file, 'w') as f:
            json.dump(interp_data, f, indent=2, ensure_ascii=False)

        tmp_file.replace(self.interp_file)
        segments = interp_data.get('interpretation', {}).get('segments', [])
        logger.info(f"✓ interpretation.json written ({len(segments)} segments)")

    def import_well(self, well_name: str) -> bool:
        """
        Import interpretation for single well.

        Args:
            well_name: Well name (e.g., "Well162~EGFDL")

        Returns:
            True if successful
        """
        logger.info("=" * 60)
        logger.info(f"Importing: {well_name}")
        logger.info("=" * 60)

        # Load saved interpretation
        interp_data = self._load_interpretation(well_name)
        if not interp_data:
            logger.error(f"No interpretation found for {well_name}")
            return False

        segments = interp_data.get('interpretation', {}).get('segments', [])
        logger.info(f"Loaded {len(segments)} segments")

        try:
            # Step 0: Stop AG if running (from previous well)
            state = self._get_starsteer_state()
            if state == "AG_RUNNING":
                logger.info("Step 0: Stopping AG from previous well...")
                self._send_command("STOP_AG", {}, timeout=30)

            # Step 1: SELECT_WELL by name
            logger.info("Step 1: Selecting well...")
            self._send_command("SELECT_WELL", {"wellName": well_name})

            # Wait for interpretations_list.json to update for this well
            if not self._wait_for_interp_list_update(well_name):
                logger.error(f"interpretations_list.json not updated for {well_name}")
                return False

            # Step 2: Get Manual interpretation UUID
            manual_uuid = self._get_manual_interpretation_uuid()
            if not manual_uuid:
                logger.error("Manual interpretation not found")
                return False
            logger.info(f"Manual UUID: {manual_uuid}")

            # Step 3: Check if AI_GPU_local already exists, otherwise COPY_INTERPRETATION
            existing_uuid = self._get_interpretation_uuid(self.import_interp_name)
            if existing_uuid:
                logger.info(f"Using existing {self.import_interp_name}: {existing_uuid}")
                import_uuid = existing_uuid
            else:
                logger.info(f"Step 2: Copying interpretation -> {self.import_interp_name}")
                self._send_command("COPY_INTERPRETATION", {
                    "sourceInterpretationUuid": manual_uuid,
                    "targetInterpretationName": self.import_interp_name
                })

                # Get UUID from command result (status.json)
                import_uuid = self._get_copy_result_uuid()
                if not import_uuid:
                    logger.error(f"{self.import_interp_name} not created (no UUID in result)")
                    return False
                logger.info(f"Created: {import_uuid}")

            # Step 4: SELECT_INTERPRETATION
            logger.info("Step 3: Selecting interpretation...")
            self._send_command("SELECT_INTERPRETATION", {
                "interpretationUuid": import_uuid
            })

            # Wait for interpretation to become active (is_active=true)
            if not self._wait_for_interpretation_active(import_uuid):
                logger.error(f"Interpretation {import_uuid} not active after SELECT")
                return False

            # Step 5: CONFIGURE_AG_SETTINGS (required before START_AG)
            logger.info("Step 4: Configuring AG settings...")
            ag_params = {
                "useGridSlice": False,
                "gridName": ""
            }
            self._send_command("CONFIGURE_AG_SETTINGS", ag_params)

            # Step 6: START_AG
            logger.info("Step 5: Starting AG...")
            self._send_command("START_AG", {})
            self._wait_for_ag_running()

            # Step 6: Write interpretation.json
            logger.info("Step 5: Writing interpretation...")
            self._write_interpretation(interp_data)

            # Step 7: STOP_AG
            logger.info("Step 6: Stopping AG...")
            self._send_command("STOP_AG", {}, timeout=30)

            logger.info(f"✓ {well_name} imported successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to import {well_name}: {e}")
            return False

    def run(self, wells: List[str] = None):
        """
        Run import for all wells.

        Args:
            wells: List of wells to import (None = all available)
        """
        # Get available wells
        available = self.list_available_wells()
        if not available:
            logger.error(f"No wells found in {self.results_dir}")
            return

        # Filter to requested wells
        if wells:
            to_import = [w for w in wells if w in available]
            missing = [w for w in wells if w not in available]
            if missing:
                logger.warning(f"Wells not found: {missing}")
        else:
            to_import = available

        logger.info(f"Wells to import: {len(to_import)}")

        # Ensure StarSteer running
        self._ensure_starsteer_running()

        # Open project
        self._open_project()

        # Import each well
        success = 0
        failed = 0

        for i, well_name in enumerate(to_import, 1):
            logger.info(f"\n[{i}/{len(to_import)}] {well_name}")
            if self.import_well(well_name):
                success += 1
            else:
                failed += 1

        logger.info("=" * 60)
        logger.info(f"DONE: {success} success, {failed} failed")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Import interpretations to StarSteer")
    parser.add_argument('--results-dir', type=str, required=True,
                        help='Directory with saved interpretations (work/)')
    parser.add_argument('--well', type=str, nargs='*',
                        help='Specific well(s) to import')
    parser.add_argument('--list', action='store_true',
                        help='List available wells')
    parser.add_argument('--env', type=str, default=None,
                        help='Path to .env file')
    parser.add_argument('--name', type=str, default=None,
                        help='Interpretation name in StarSteer (default: AI_GPU_local)')

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    env_file = Path(args.env) if args.env else None

    importer = StarSteerImporter(results_dir, env_file, interp_name=args.name)

    if args.list:
        wells = importer.list_available_wells()
        print(f"Available wells in {results_dir}:")
        for w in wells:
            print(f"  {w}")
        print(f"\nTotal: {len(wells)}")
        return

    importer.run(args.well)


if __name__ == '__main__':
    main()
