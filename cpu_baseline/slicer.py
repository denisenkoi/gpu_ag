#!/usr/bin/env python3
"""
StarSteer Slicer Integration - Standalone Orchestrator
=======================================================

Manages StarSteer Slicer + Night Guard for automated well slicing.

Architecture:
    slicer.py (this) → manages → Night Guard (unchanged)
                    ↓
                StarSteer Slicer

Execution:
    python slicer.py  (standalone, not called from main.py)

Phase 1 (ME-10): INIT
- Launch StarSteer
- Open project
- INIT_WELL_SLICER for each active well
- Night Guard INIT (via import)

Phase 2 (ME-11): Slicing loop
- Will be implemented in ME-11

Author: Claude Code
Date: 2025-11-12
Task: ME-10
Spec: docs/251112_14_starsteer_slicer_integration_spec.md
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dotenv import load_dotenv

# Load .env before any other imports that might read env vars
load_dotenv()

# Import Night Guard infrastructure
from main import load_configuration, setup_logging
from emulator import DrillingEmulator
from slicer_quality import SlicerQualityAnalyzer

logger = logging.getLogger(__name__)

# Default StarSteer directory for slicer (used when no --starsteer-dir argument)
DEFAULT_SLICER_STARSTEER_DIR = "E:/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_ag"

# Global config set by argparse (will override env vars)
_CLI_CONFIG = {
    'starsteer_dir': None,
    'python_executor': None,
    'max_iterations': None
}


class StarSteerSlicerOrchestrator:
    """
    Orchestrator for StarSteer Slicer + Night Guard integration

    Manages StarSteer lifecycle and calls Night Guard for processing.
    Night Guard has NO knowledge of slicer - works as-is.
    """

    def __init__(self):
        """Initialize orchestrator"""
        # Load Night Guard configuration
        self.config = load_configuration()

        # Initialize paths
        self.starsteer_dir = None
        self.commands_file = None
        self.slice_command_file = None
        self.status_file = None
        self.wells_list_file = None
        self.slicer_status_file = None
        self.ag_data_dir = None
        self.task_queue_dir = None
        self.project_name = None

        # Setup paths
        self._setup_paths()

        # Add starsteer_dir to config for emulator
        self.config['starsteer_dir'] = str(self.starsteer_dir)

        # Commands logging setup
        self.log_commands = os.getenv('LOG_STARSTEER_COMMANDS', 'false').lower() == 'true'
        self.commands_log_file = Path(os.getenv('STARSTEER_COMMANDS_LOG', 'starsteer_commands.jsonl'))
        if self.log_commands:
            logger.info(f"StarSteer commands logging enabled: {self.commands_log_file}")

        # Initialize quality analyzer for real-time monitoring with parameters
        dip_range = float(os.getenv('DIP_ANGLE_RANGE_DEGREE', 0))
        lookback = float(os.getenv('LOOKBACK_DISTANCE', 0))
        smoothness = float(os.getenv('SMOOTHNESS', 0))

        self.quality_analyzer = SlicerQualityAnalyzer(
            self.config,
            dip_range=dip_range if dip_range > 0 else None,
            lookback=lookback if lookback > 0 else None,
            smoothness=smoothness if smoothness > 0 else None
        )

        # Slice ID counter for duplicate detection (STARSTEER slice_id)
        self.slice_id_counter = 0

        # Debug mode - skip StarSteer init, assume AG already running
        self.debug_mode = os.getenv('SLICER_DEBUG_MODE', 'false').lower() == 'true'
        if self.debug_mode:
            logger.info("=" * 70)
            logger.info("DEBUG MODE ENABLED - skipping StarSteer init")
            logger.info("=" * 70)

    def _setup_paths(self):
        """Setup StarSteer paths.

        Priority: CLI argument > DEFAULT_SLICER_STARSTEER_DIR
        """
        from path_utils import normalize_path

        # ONLY CLI argument, no fallback!
        starsteer_path = _CLI_CONFIG.get('starsteer_dir')
        if not starsteer_path:
            raise ValueError("--starsteer-dir is required!")

        # Auto-convert path for current platform (Windows <-> WSL)
        starsteer_path = normalize_path(starsteer_path)

        self.starsteer_dir = Path(starsteer_path)

        logger.info(f"StarSteer directory: {self.starsteer_dir}")

        # Control files
        self.commands_file = self.starsteer_dir / "commands.json"
        self.slice_command_file = self.starsteer_dir / "slice_command.json"
        self.status_file = self.starsteer_dir / "status.json"
        self.wells_list_file = self.starsteer_dir / "wells_list.json"
        self.slicer_status_file = self.starsteer_dir / "slicer_status.json"
        self.ag_data_dir = self.starsteer_dir / "AG_DATA" / "InitialData"

        # Task queue (source directory) - Windows path
        self.task_queue_dir = Path("E:/Projects/Rogii/sc/starsteer/task_queue")

        # Project name
        self.project_name = os.getenv('SOLO_PROJECT_NAME')
        if not self.project_name:
            raise ValueError("SOLO_PROJECT_NAME not set in .env")

    def run(self):
        """
        Main entry point - Phase 1: INIT only

        Phase 2 (ME-11) will add slicing loop
        """
        if not self.debug_mode:
            # Ensure ALL StarSteer instances are closed before starting
            # This prevents race conditions when multiple StarSteer are running
            self._ensure_all_starsteer_closed()

            logger.info("=" * 70)
            logger.info("StarSteer Slicer Orchestrator - Phase 1: INIT")
            logger.info("=" * 70)

            # Cleanup previous run artifacts (files only, StarSteer already closed)
            self._cleanup_previous_run(cleanup_starsteer=False)

            # Ensure StarSteer running
            self._ensure_starsteer_running()

            # Ensure project loaded
            self._ensure_project_loaded()
        else:
            logger.info("=" * 70)
            logger.info("DEBUG MODE - StarSteer init skipped, AG assumed running")
            logger.info("=" * 70)

        # Step 1: Load wells config (process=true only)
        logger.info("Loading wells configuration...")
        wells_to_process = self._load_wells_config()
        logger.info(f"Found {len(wells_to_process)} wells with process=true")

        # Step 2: Load wells_list.json ONCE
        logger.info("Loading wells list from StarSteer...")
        wells_uuid_map = self._load_wells_uuid_map()
        logger.info(f"Loaded {len(wells_uuid_map)} wells from StarSteer")

        # Step 3: Enrich with UUIDs
        logger.info("Enriching wells config with UUIDs...")
        enriched_wells, missing_wells = self._enrich_with_uuids(
            wells_to_process, wells_uuid_map
        )

        # Step 4: Log processing summary
        self._log_processing_summary(enriched_wells, missing_wells)

        if missing_wells:
            logger.warning(f"Skipping {len(missing_wells)} wells without UUIDs")

        if not enriched_wells:
            logger.error("No wells to process!")
            return

        # Process each well (Phase 1: INIT only)
        processed_count = self._process_all_wells_init(enriched_wells)

        # Summary
        logger.info("=" * 70)
        logger.info("Phase 1 (INIT) COMPLETED!")
        logger.info(f"Processed wells: {processed_count}/{len(enriched_wells)}")
        logger.info("=" * 70)

    # ========================================================================
    # Cleanup Methods
    # ========================================================================

    def _ensure_all_starsteer_closed(self, max_retries: int = 5, stale_timeout: int = 35):
        """
        Ensure ALL StarSteer instances are closed before starting.

        Prevents race conditions when multiple StarSteer are running.
        Uses retry loop: send CLOSE_APP, wait for status.json to become stale (>30s).

        Args:
            max_retries: Maximum CLOSE_APP attempts
            stale_timeout: Seconds to wait for status.json to become stale
        """
        logger.info("Ensuring all StarSteer instances are closed...")

        for attempt in range(max_retries):
            # Check if status.json is stale (no updates for 30+ seconds)
            if not self.status_file.exists():
                logger.info("  ✓ No status.json - no StarSteer running")
                # Clean commands.json just in case
                if self.commands_file.exists():
                    self.commands_file.unlink()
                    logger.info("  ✓ Cleared stale commands.json")
                return

            mtime = self.status_file.stat().st_mtime
            age_seconds = time.time() - mtime

            if age_seconds >= 30:
                # Double-check: wait a bit and verify still stale
                logger.info(f"  status.json is stale ({age_seconds:.0f}s old), verifying...")
                time.sleep(5)

                new_mtime = self.status_file.stat().st_mtime
                if new_mtime == mtime:
                    logger.info("  ✓ Confirmed: no active StarSteer (status.json unchanged)")
                    # Clean commands.json just in case
                    if self.commands_file.exists():
                        self.commands_file.unlink()
                        logger.info("  ✓ Cleared stale commands.json")
                    return
                else:
                    logger.info("  status.json was updated during verification, retrying...")
                    continue

            # StarSteer is running - send CLOSE_APP
            logger.info(f"  StarSteer detected (status.json {age_seconds:.0f}s old), sending CLOSE_APP... (attempt {attempt + 1}/{max_retries})")

            # Write CLOSE_APP command
            command_data = {
                "command": "CLOSE_APP",
                "params": {"save_changes": False}
            }
            with open(self.commands_file, 'w', encoding='utf-8') as f:
                json.dump(command_data, f, indent=2)

            # Wait for status.json to become stale (StarSteer stopped updating)
            wait_start = time.time()
            last_mtime = mtime

            while time.time() - wait_start < stale_timeout:
                time.sleep(3)

                if not self.status_file.exists():
                    logger.info("  ✓ status.json deleted - StarSteer closed")
                    # Clean commands.json to prevent new StarSteer from reading stale CLOSE_APP
                    if self.commands_file.exists():
                        self.commands_file.unlink()
                        logger.info("  ✓ Cleared commands.json (prevent race condition)")
                    return

                current_mtime = self.status_file.stat().st_mtime
                current_age = time.time() - current_mtime

                if current_age >= 30:
                    logger.info(f"  ✓ status.json stale ({current_age:.0f}s) - StarSteer closed")
                    # Clean commands.json to prevent new StarSteer from reading stale CLOSE_APP
                    if self.commands_file.exists():
                        self.commands_file.unlink()
                        logger.info("  ✓ Cleared commands.json (prevent race condition)")
                    # Extra wait to be safe
                    time.sleep(5)
                    return

                if current_mtime != last_mtime:
                    logger.info(f"    status.json updated ({current_age:.0f}s ago), waiting...")
                    last_mtime = current_mtime

            logger.warning(f"  CLOSE_APP attempt {attempt + 1} did not fully close StarSteer, retrying...")

        # After all retries, check final state
        if self.status_file.exists():
            final_age = time.time() - self.status_file.stat().st_mtime
            if final_age < 30:
                raise RuntimeError(
                    f"Failed to close all StarSteer instances after {max_retries} attempts. "
                    f"status.json still fresh ({final_age:.0f}s). "
                    "Please close StarSteer manually."
                )

        logger.info("  ✓ All StarSteer instances closed")

    def _cleanup_previous_run(self, cleanup_starsteer: bool = True):
        """
        Cleanup artifacts from previous slicer runs.

        Clears:
        - json_comparison directory (raw/step files)
        - Commands log file (starsteer_commands.jsonl)

        Args:
            cleanup_starsteer: If True, also close StarSteer (default True for backward compat)
        """
        import shutil

        logger.info("Cleaning up previous run artifacts...")

        # 1. Clear json_comparison directory
        comparison_dir = os.getenv('JSON_COMPARISON_DIR')
        if comparison_dir:
            comparison_path = Path(comparison_dir)
            if comparison_path.exists():
                # Remove all files but keep directory
                file_count = 0
                for f in comparison_path.iterdir():
                    if f.is_file():
                        f.unlink()
                        file_count += 1
                logger.info(f"  Cleared {file_count} files from {comparison_dir}")
            else:
                comparison_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"  Created {comparison_dir}")

        # 2. Clear commands log file
        if self.log_commands and self.commands_log_file.exists():
            self.commands_log_file.unlink()
            logger.info(f"  Cleared {self.commands_log_file}")

        # 3. Close any running StarSteer instance (soft kill via CLOSE_APP)
        if cleanup_starsteer:
            self._close_starsteer_if_running()

        logger.info("Cleanup complete")

    def _close_starsteer_if_running(self):
        """
        Close StarSteer via CLOSE_APP command (soft kill).

        Unlike taskkill which kills ALL StarSteer processes system-wide,
        CLOSE_APP targets only the instance controlled by this slicer
        (via its commands.json file).
        """
        if not self._is_starsteer_running():
            logger.info("  StarSteer not running, skip CLOSE_APP")
            return

        logger.info("  Closing StarSteer via CLOSE_APP command...")

        # Write CLOSE_APP command
        command_data = {
            "command": "CLOSE_APP",
            "params": {"save_changes": False}
        }

        with open(self.commands_file, 'w', encoding='utf-8') as f:
            json.dump(command_data, f, indent=2)

        # Wait for StarSteer to close (status.json becomes stale)
        timeout = 30
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self._is_starsteer_running():
                logger.info("  ✓ StarSteer closed")
                return
            time.sleep(2)

        logger.warning("  StarSteer did not close within timeout, continuing anyway")

    # ========================================================================
    # StarSteer Control Methods
    # ========================================================================

    def _ensure_starsteer_running(self):
        """Check if StarSteer running, launch if needed"""
        if not self._is_starsteer_running():
            logger.info("StarSteer not running, launching...")
            self._launch_starsteer()
            self._wait_starsteer_ready()
        else:
            logger.info("✓ StarSteer already running")

    def _is_starsteer_running(self) -> bool:
        """Check if StarSteer running (status.json fresh < 30 sec and not CLOSE_APP SUCCESS)"""
        if not self.status_file.exists():
            return False

        mtime = self.status_file.stat().st_mtime
        age_seconds = time.time() - mtime

        if age_seconds >= 30:
            return False

        # Check if status.json indicates app closed
        try:
            with open(self.status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)
            # If CLOSE_APP succeeded, StarSteer is NOT running
            if status.get("command") == "CLOSE_APP" and status.get("status") == "SUCCESS":
                return False
        except (json.JSONDecodeError, IOError):
            pass

        return True

    def _launch_starsteer(self):
        """Launch StarSteer - via trigger on WSL, direct on Windows"""
        import subprocess
        import platform

        # Check if running on Linux (WSL) - use trigger system
        if platform.system() == 'Linux':
            logger.info(f"WSL detected - launching StarSteer via trigger")
            self._launch_starsteer_via_trigger()
        else:
            # Windows - direct launch
            starsteer_exe = self.starsteer_dir / "StarSteer.exe"

            if not starsteer_exe.exists():
                raise FileNotFoundError(f"StarSteer.exe not found: {starsteer_exe}")

            logger.info(f"Launching StarSteer: {starsteer_exe}")
            subprocess.Popen(
                [str(starsteer_exe)],
                cwd=str(self.starsteer_dir),
                creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info("StarSteer launched")

    def _launch_starsteer_via_trigger(self):
        """Launch StarSteer via task runner trigger (for WSL)"""
        import time as time_module

        # Trigger directory - use sc project task_queue
        from path_utils import normalize_path
        trigger_dir = Path(normalize_path("E:/Projects/Rogii/sc/task_queue"))
        trigger_dir.mkdir(parents=True, exist_ok=True)

        # Create trigger file
        timestamp = int(time_module.time())
        task_id = f"task_{timestamp}_starsteer"
        trigger_file = trigger_dir / f"{task_id}.json"

        trigger_data = {
            "task_id": task_id,
            "type": "run_bat",
            "script_path": "runss_slicer_de.bat",
            "bot_id": "SSAndAG",
            "created_at": timestamp
        }

        with open(trigger_file, 'w') as f:
            json.dump(trigger_data, f)

        logger.info(f"Created StarSteer trigger: {trigger_file}")
        logger.info("Waiting for task runner to pick up trigger...")

        # Wait for trigger to be processed (file will be deleted by task runner)
        wait_start = time_module.time()
        while trigger_file.exists():
            if time_module.time() - wait_start > 30:
                logger.warning("Trigger not picked up in 30s, continuing anyway...")
                break
            time_module.sleep(1)

        logger.info("StarSteer trigger processed, waiting for application...")

    def _wait_starsteer_ready(self, timeout: int = 60):
        """Wait for StarSteer ready"""
        logger.info("Waiting for StarSteer...")
        start_time = time.time()

        time.sleep(30)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("StarSteer launch timeout")

            if self.status_file.exists():
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    status = json.load(f)

                state = status.get("application_state", {}).get("state")
                if state:
                    logger.info(f"✓ StarSteer ready: {state}")
                    return

            time.sleep(2)

    def _ensure_project_loaded(self):
        """Ensure project loaded"""
        current_state = self._get_starsteer_state()

        if current_state != "PROJECT_LOADED":
            logger.info(f"Opening project: {self.project_name} (may take up to 3 minutes)")

            # Remember wells_list.json mtime before opening
            initial_mtime = 0
            if self.wells_list_file.exists():
                initial_mtime = self.wells_list_file.stat().st_mtime
                logger.info(f"wells_list.json current mtime: {initial_mtime}")

            # Send OPEN_PROJECT command (may take up to 20 minutes)
            self._send_command("OPEN_PROJECT", {
                "projectName": self.project_name
            }, timeout=1200)

            # Wait for state change (timeout 600 sec = 10 minutes with margin)
            self._wait_for_state("PROJECT_LOADED", timeout=600)

            # Additional validation: wait for wells_list.json update
            logger.info("Verifying wells_list.json updated...")
            self._wait_for_wells_list_update(initial_mtime, timeout=30)

            logger.info("✓ Project loaded and verified")
        else:
            logger.info(f"✓ Project already loaded: {self.project_name}")

    def _get_starsteer_state(self) -> str:
        """Get current StarSteer state"""
        with open(self.status_file, 'r', encoding='utf-8') as f:
            status = json.load(f)
        return status.get("application_state", {}).get("state", "UNKNOWN")

    def _send_command(self, command: str, params: dict, timeout: int = 300):
        """Send command to StarSteer"""
        logger.info(f"Sending command: {command}")

        if self.commands_file.exists():
            self.commands_file.unlink()

        command_data = {
            "command": command,
            "params": params
        }

        # Log command to JSONL if enabled
        if self.log_commands:
            log_record = {
                "timestamp": time.time(),
                "datetime": time.strftime("%Y-%m-%d %H:%M:%S"),
                "command": command,
                "params": params
            }
            with open(self.commands_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_record, ensure_ascii=False) + '\n')
            logger.debug(f"Logged command to {self.commands_log_file}")

        with open(self.commands_file, 'w', encoding='utf-8') as f:
            json.dump(command_data, f, indent=2)

        # Wait for processing (file deleted) with heartbeat
        start_time = time.time()
        last_heartbeat = start_time

        while self.commands_file.exists():
            elapsed = time.time() - start_time

            if elapsed > timeout:
                raise TimeoutError(f"Command {command} timeout after {timeout}s")

            # Heartbeat every 30 seconds
            if time.time() - last_heartbeat >= 30:
                logger.info(f"⏱️  Heartbeat: {command} processing... {elapsed:.0f}s elapsed")
                last_heartbeat = time.time()

            time.sleep(2)

        logger.info(f"✓ Command {command} processed")

    def _wait_for_state(self, expected_state: str, timeout: int = 30):
        """Wait for StarSteer state"""
        logger.info(f"Waiting for state: {expected_state}")
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"State {expected_state} timeout")

            with open(self.status_file, 'r', encoding='utf-8') as f:
                status = json.load(f)

            current_state = status.get("application_state", {}).get("state")
            if current_state == expected_state:
                logger.info(f"✓ Reached state: {expected_state}")
                return

            time.sleep(2)

    def _wait_for_wells_list_update(self, initial_mtime: float, timeout: int = 30):
        """
        Wait for wells_list.json to be updated (mtime changed)

        This provides additional validation that project load completed
        and wells_list.json was exported by StarSteer.
        """
        start_time = time.time()

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError("wells_list.json not updated after project load")

            if not self.wells_list_file.exists():
                logger.warning("wells_list.json does not exist yet")
                time.sleep(1)
                continue

            current_mtime = self.wells_list_file.stat().st_mtime
            if current_mtime > initial_mtime:
                logger.info(f"✓ wells_list.json updated (mtime: {current_mtime})")
                return

            time.sleep(1)

    def _load_wells_config(self) -> List[Dict]:
        """Load wells from wells_config_full.json with process=true"""
        config_file = Path("wells_config_full.json")

        if not config_file.exists():
            raise FileNotFoundError(f"Config not found: {config_file}")

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        all_wells = config.get("wells", [])
        active_wells = [w for w in all_wells if w.get("process") == True]

        return active_wells

    def _load_wells_uuid_map(self) -> Dict[str, str]:
        """
        Load wells_list.json ONCE and create name→UUID mapping

        Returns: {well_name: uuid}
        """
        wells_list_file = self.starsteer_dir / "wells_list.json"

        if not wells_list_file.exists():
            raise FileNotFoundError(f"wells_list.json not found: {wells_list_file}")

        with open(wells_list_file, 'r', encoding='utf-8') as f:
            wells_data = json.load(f)

        # Build name→UUID mapping
        uuid_map = {}
        for well in wells_data.get("laterals", []):
            name = well.get("name")
            uuid = well.get("uuid")
            if name and uuid:
                uuid_map[name] = uuid

        return uuid_map

    def _enrich_with_uuids(
        self,
        wells_config: List[Dict],
        uuid_map: Dict[str, str]
    ) -> Tuple[List[Dict], List[str]]:
        """
        Add UUIDs to wells config

        Returns:
            - enriched_wells: Wells with UUIDs added
            - missing_wells: Well names without UUIDs
        """
        enriched = []
        missing = []

        for well in wells_config:
            well_name = well['well_name']

            if well_name in uuid_map:
                # Add UUID to well config
                well_with_uuid = well.copy()
                well_with_uuid['uuid'] = uuid_map[well_name]
                enriched.append(well_with_uuid)
            else:
                missing.append(well_name)

        return enriched, missing

    def _log_processing_summary(
        self,
        enriched_wells: List[Dict],
        missing_wells: List[str]
    ):
        """Log processing summary before starting"""
        total = len(enriched_wells) + len(missing_wells)
        found = len(enriched_wells)
        missing_count = len(missing_wells)

        logger.info("=" * 70)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total wells to process: {total}")
        logger.info(f"Found UUIDs for: {found} wells")

        if missing_count > 0:
            logger.warning(f"Missing UUIDs for: {missing_count} wells")
            logger.warning(f"Wells without UUIDs: {missing_wells}")
        else:
            logger.info("✓ All wells have UUIDs")

        logger.info("=" * 70)
        logger.info("Wells to process:")
        for i, well in enumerate(enriched_wells, 1):
            logger.info(f"  {i}. {well['well_name']} → {well['uuid']}")
        logger.info("=" * 70)

    # ========================================================================
    # Well Processing (Phase 1: INIT only)
    # ========================================================================

    def _process_all_wells_init(self, active_wells: list) -> int:
        """
        Process all active wells - Phase 1: INIT only

        For each well:
        1. Get reference interpretation UUID (from source well)
        2. INIT_WELL_SLICER
        3. Copy interpretation (starred → Assisted on target well)
        4. Call Night Guard INIT via emulator.setup_well_processing()
        5. Cleanup

        Returns: Number of processed wells
        """
        TARGET_WELL_NAME = "slicing_well"
        processed_count = 0
        total_wells = len(active_wells)

        for well_idx, well_info in enumerate(active_wells, 1):
            source_name = well_info['well_name']
            source_uuid = well_info['uuid']

            logger.info("=" * 70)
            logger.info(f"Processing well: {source_name} ({well_idx}/{total_wells})")
            logger.info(f"UUID: {source_uuid}")
            logger.info("=" * 70)

            try:
                if not self.debug_mode:
                    # Step 1: Get reference interpretation UUID (BEFORE INIT_WELL_SLICER)
                    # Must be done while still on source well!
                    logger.info("Step 1: Getting reference interpretation UUID from source well...")
                    ref_interp_uuid = self._get_reference_interpretation_uuid(source_uuid)
                    logger.info(f"✓ Reference UUID: {ref_interp_uuid}")

                    # Step 2: INIT_WELL_SLICER (creates target well and activates it)
                    logger.info("Step 2: Initializing well slicer...")
                    target_uuid, landing_end_md = self._init_well_slicer(source_uuid, TARGET_WELL_NAME)
                    logger.info(f"✓ Target well created and activated: {target_uuid}")
                    logger.info(f"✓ Landing end MD extracted: {landing_end_md:.2f}m")

                    # Step 3: Copy interpretation (starred → Assisted on target well)
                    logger.info("Step 3: Copying interpretation to target well...")
                    assisted_uuid = self._copy_interpretation_to_target()
                    logger.info(f"✓ Interpretation copied: Assisted UUID={assisted_uuid}")

                    # ========================================================================
                    # TESTING: SKIP AG CONFIGURATION but START AG
                    # ========================================================================
                    # Testing: Can AG start WITHOUT explicit configuration?
                    # SKIP: Configure AG config file and Configure AG settings
                    # KEEP: Start AG command
                    # ========================================================================

                    # Step 4: Configure AG config file (passive JSON write)
                    logger.info("Step 4: Configuring AG config file...")
                    self._configure_ag_config(TARGET_WELL_NAME, ref_interp_uuid)

                    # Step 5: Configure AG settings (triggers ag_config.json reload)
                    logger.info("Step 5: Configuring AG settings...")
                    self._configure_ag_settings(well_info, landing_end_md)

                    # Step 6: Start AG (with default settings)
                    # Note: AG start MD is configured in StarSteer, not passed as parameter
                    logger.info("Step 6: Starting AG with default settings...")
                    self._start_ag(TARGET_WELL_NAME)
                else:
                    logger.info("DEBUG MODE - skipping StarSteer commands (steps 1-6)")
                    logger.info("Assuming AG already running on slicing_well")

                # Load initial data
                initial_well_data = self._load_well_json(TARGET_WELL_NAME)

                # Step 5: Night Guard INIT (via emulator)
                logger.info("Step 4: Calling Night Guard INIT...")
                setup_data = self._call_night_guard_init(initial_well_data, TARGET_WELL_NAME)

                logger.info(f"✓ Well prepared: {TARGET_WELL_NAME}")
                logger.info(f"  Start MD: {setup_data['current_md']:.2f}m")
                logger.info(f"  Max MD: {setup_data['max_md']:.2f}m")
                if not self.debug_mode:
                    logger.info(f"  Reference UUID: {ref_interp_uuid}")

                # Run slicing loop (skip in debug mode - only INIT)
                if not self.debug_mode:
                    self._run_slicing_loop(setup_data, TARGET_WELL_NAME, source_name)
                else:
                    logger.info("DEBUG MODE - skipping slicing loop, only INIT done")
                    # Export interpretation to StarSteer after INIT
                    emulator = setup_data['emulator']
                    executor = setup_data['executor']
                    emulator._export_interpretation_to_starsteer(TARGET_WELL_NAME, executor)
                    logger.info("DEBUG MODE - exported interpretation to StarSteer")

                # Cleanup AFTER loop
                if setup_data.get('visualizer'):
                    setup_data['visualizer'].close()
                if setup_data.get('executor'):
                    setup_data['executor'].stop_daemon()

                processed_count += 1
                logger.info(f"✓ Completed: {source_name}")

            except Exception as e:
                logger.error(f"Failed: {source_name}: {e}", exc_info=True)
                # Save typewells_list.json for failed well (contains correct typewell/log)
                typewells_file = Path(self.starsteer_dir) / "typewells_list.json"
                if typewells_file.exists():
                    save_dir = Path(self.starsteer_dir) / "failed_wells_typewells"
                    save_dir.mkdir(exist_ok=True)
                    safe_name = source_name.replace("~", "_").replace("/", "_")
                    save_path = save_dir / f"{safe_name}_typewells.json"
                    import shutil
                    shutil.copy(typewells_file, save_path)
                    logger.info(f"Saved typewells config to: {save_path}")
                raise  # Fall through - all wells should pass now

        return processed_count

    def _get_reference_interpretation_uuid(self, source_uuid: str) -> str:
        """
        Get Manual interpretation UUID from source well

        This must be called BEFORE INIT_WELL_SLICER to get reference
        interpretation UUID while still on source well.

        Args:
            source_uuid: Source well UUID

        Returns:
            UUID of Manual interpretation on source well
        """
        logger.info(f"Getting Manual interpretation UUID from source well: {source_uuid}")

        # SELECT_WELL for source well to get its interpretations
        logger.info("Selecting source well...")
        self._send_command("SELECT_WELL", {
            "wellUuid": source_uuid
        })

        # Wait for SELECT_WELL to complete and export interpretations_list.json
        time.sleep(1)

        # Read interpretations_list.json (auto-exported by SELECT_WELL)
        interp_list_file = self.starsteer_dir / "interpretations_list.json"

        if not interp_list_file.exists():
            raise FileNotFoundError("interpretations_list.json not found after SELECT_WELL")

        with open(interp_list_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Find Manual interpretation (case-insensitive)
        interpretations = data.get("interpretations", [])
        logger.info(f"Found {len(interpretations)} interpretations on source well")

        for interp in interpretations:
            name = interp.get("name")
            uuid = interp.get("uuid")
            logger.info(f"  - {name}: {uuid}")

            # Case-insensitive comparison
            if name and name.lower() == "manual":
                logger.info(f"✓ Found Manual interpretation: {uuid}")
                return uuid

        # If Manual not found, raise error
        available_names = [i.get("name") for i in interpretations]
        raise ValueError(f"Manual interpretation not found. Available: {available_names}")

    def _copy_interpretation_to_target(self) -> str:
        """
        Copy starred interpretation to Assisted on current (target) well

        This must be called AFTER target well is activated (SELECT_WELL).
        Reads interpretations_list.json to find starred interpretation.
        If Assisted already exists, uses existing one without copying.

        Returns:
            UUID of copied Assisted interpretation
        """
        logger.info("Copying starred interpretation to Assisted on target well")

        # Read interpretations_list.json (should be for target well)
        interp_list_file = self.starsteer_dir / "interpretations_list.json"

        if not interp_list_file.exists():
            raise FileNotFoundError("interpretations_list.json not found")

        with open(interp_list_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Find starred interpretation and check if Assisted exists
        interpretations = data.get("interpretations", [])
        logger.info(f"Found {len(interpretations)} interpretations on target well")

        starred_uuid = None
        starred_name = None
        existing_assisted_uuid = None

        for interp in interpretations:
            name = interp.get("name")
            uuid = interp.get("uuid")
            is_starred = interp.get("is_starred", False)

            logger.info(f"  - {name}: {uuid} (starred={is_starred})")

            if is_starred:
                starred_uuid = uuid
                starred_name = name
            if name == "Assisted":
                existing_assisted_uuid = uuid

        # If Assisted already exists, use it
        if existing_assisted_uuid:
            logger.info(f"✓ Assisted interpretation already exists: {existing_assisted_uuid}")
            assisted_uuid = existing_assisted_uuid
        else:
            # Need to copy from starred
            if not starred_uuid:
                raise ValueError("No starred interpretation found on target well")

            logger.info(f"Found starred interpretation: {starred_name} ({starred_uuid})")

            # COPY_INTERPRETATION (starred → Assisted)
            logger.info("Sending COPY_INTERPRETATION command...")
            self._send_command("COPY_INTERPRETATION", {
                "sourceInterpretationUuid": starred_uuid,
                "targetInterpretationName": "Assisted"
            })

            # Wait for copy to complete
            time.sleep(2)

            # Re-read interpretations_list.json to find Assisted
            with open(interp_list_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            interpretations = data.get("interpretations", [])
            assisted_uuid = None

            for interp in interpretations:
                if interp.get("name") == "Assisted":
                    assisted_uuid = interp.get("uuid")
                    logger.info(f"✓ Found Assisted interpretation: {assisted_uuid}")
                    break

            if not assisted_uuid:
                raise ValueError("Assisted interpretation not created after COPY_INTERPRETATION")

        # SELECT_INTERPRETATION (Assisted)
        logger.info("Selecting Assisted interpretation...")
        self._send_command("SELECT_INTERPRETATION", {
            "interpretationUuid": assisted_uuid
        })

        # Wait for selection
        time.sleep(1)

        logger.info(f"✓ Interpretation copied and selected: {assisted_uuid}")
        return assisted_uuid

    def _init_well_slicer(self, source_uuid: str, target_name: str) -> Tuple[str, float]:
        """
        Initialize Well Slicer

        Returns:
            Tuple of (target_well_uuid, landing_end_md)
        """
        logger.info(f"INIT_WELL_SLICER: {source_uuid} → {target_name}")

        # additionalMD value used for cutoff calculation
        # Changed from 50.0 to 90.0 to match old emulator init MD (~3720m instead of ~3680m)
        additional_md = 90.0

        # Log cutoff calculation parameters for debugging
        logger.info("INIT_WELL_SLICER cutoff parameters:")
        logger.info("  cutoffMD: 0 (automatic landing detection)")
        logger.info(f"  additionalMD: {additional_md}m (ME-16 fix)")
        logger.info(f"  Expected: landing_end + 200.0m + {additional_md}m")

        self._send_command("INIT_WELL_SLICER", {
            "sourceWellUuid": source_uuid,
            "targetWellName": target_name,
            "cutoffMD": 0,  # Automatic landing detection
            "additionalMD": additional_md  # Fix ME-16: 50m instead of hardcoded 1.0m (requires STARSTEER-35)
        })

        # Wait for initialization
        time.sleep(1)

        if not self.slicer_status_file.exists():
            raise RuntimeError("slicer_status.json not created")

        with open(self.slicer_status_file, 'r', encoding='utf-8') as f:
            slicer_status = json.load(f)

        if not slicer_status.get("initialized", False):
            raise RuntimeError("Slicer initialization failed")

        # Parse cutoff details from slicer_status
        config = slicer_status.get("configuration", {})
        cutoff_md = config.get("cutoff_md_meters")
        if cutoff_md is None:
            raise RuntimeError("cutoff_md_meters not found in slicer_status configuration")

        logger.info(f"✓ Slicer initialized: cutoffMD={cutoff_md:.2f}m")

        # Calculate landing_end_md from cutoff (cutoff = landing_end + 200m + additionalMD)
        # So: landing_end = cutoff - 200m - additionalMD
        # But we only subtract additionalMD because we want to start AG BEFORE the 200m offset
        landing_end_md = cutoff_md - additional_md

        logger.info(f"  Calculated landing_end_md: {landing_end_md:.2f}m")
        logger.info(f"  Formula: cutoff_md ({cutoff_md:.2f}m) - additionalMD ({additional_md}m)")
        logger.info(f"  This will be used as AG start MD")

        # Activate the sliced well (SELECT_WELL)
        target_uuid = slicer_status.get("target_well", {}).get("uuid")
        if not target_uuid:
            raise RuntimeError("target_well UUID not found in slicer_status")

        logger.info(f"Activating sliced well: {target_uuid}")
        self._send_command("SELECT_WELL", {
            "wellUuid": target_uuid
        })

        # Wait for SELECT_WELL to complete and export interpretations_list.json
        time.sleep(1)

        logger.info(f"✓ Sliced well activated: {target_name}")
        return target_uuid, landing_end_md

    def _configure_ag_config(self, target_well_name: str, ref_interp_uuid: str):
        """
        Write AG config file with all required fields

        Args:
            target_well_name: Name of target well (slicing_well)
            ref_interp_uuid: UUID of reference Manual interpretation
        """
        logger.info(f"Configuring AG config for: {target_well_name}")

        # AG config file path - MUST be in StarSteer directory!
        ag_config_path = Path(self.starsteer_dir) / "ag_config.json"

        # Create config data
        well_config = {
            "startInterpretation": "manual",           # NEW! (lowercase!)
            "landingInterpretation": "manual",         # NEW! (lowercase!)
            "referenceInterpretation": ref_interp_uuid
        }

        # Read existing config if exists
        if ag_config_path.exists():
            with open(ag_config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            config_data = {}

        # Update with new well config
        config_data[target_well_name] = well_config

        # Write config
        with open(ag_config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"✓ AG config updated: {target_well_name}")
        logger.info(f"  startInterpretation: manual")
        logger.info(f"  landingInterpretation: manual")
        logger.info(f"  referenceInterpretation: {ref_interp_uuid}")

    def _update_pseudo_log_end_md(self, well_name: str, current_md: float):
        """
        Update pseudoLogEndMd in ag_config.json for current step.

        This value tells C++ AG where to cut off typeLog data to prevent
        data leakage from interpretation zone.

        Args:
            well_name: Well name (key in ag_config.json)
            current_md: Current MD in meters
        """
        lookback_distance = float(os.getenv('LOOKBACK_DISTANCE', '200.0'))
        pseudo_log_end_md = current_md - lookback_distance

        if pseudo_log_end_md <= 0:
            return

        ag_config_path = Path(self.starsteer_dir) / "ag_config.json"

        # Read existing config
        if ag_config_path.exists():
            with open(ag_config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            config_data = {}

        # Update pseudoLogEndMd for this well
        if well_name not in config_data:
            config_data[well_name] = {}
        config_data[well_name]['pseudoLogEndMd'] = pseudo_log_end_md

        # Write back
        with open(ag_config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)

        logger.debug(f"Updated pseudoLogEndMd: {pseudo_log_end_md:.2f}m (current_md={current_md:.2f} - lookback={lookback_distance})")

    def _save_typelog_snapshot(self, well_data: Dict[str, Any], current_md: float):
        """
        Save typeLog snapshot to history file for comparing across STEP iterations.

        Used to verify that pseudoLogEndMd affects typeLog data.

        Args:
            well_data: Well data dict containing typeLog
            current_md: Current MD in meters
        """
        history_path = Path(self.starsteer_dir) / "typeLog_history.json"

        # Load existing history
        if history_path.exists():
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
        else:
            history = {}

        # Get typeLog points
        typelog = well_data.get('typeLog', {})
        points = typelog.get('tvdSortedPoints', [])

        if not points:
            logger.debug("No typeLog points to save")
            return

        # Create snapshot: count + first/last 5 points for comparison
        snapshot = {
            'points_count': len(points),
            'md_range': [points[0].get('measuredDepth', 0), points[-1].get('measuredDepth', 0)] if points else [],
            'first_5': points[:5],
            'last_5': points[-5:] if len(points) >= 5 else points
        }

        # Save with MD as key
        history[f"{current_md:.2f}"] = snapshot

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved typeLog snapshot: MD={current_md:.2f}m, points={len(points)}")

    def _configure_ag_settings(self, well_config: Dict[str, Any], landing_end_md: float):
        """
        Send CONFIGURE_AG_SETTINGS command with parameters from well config and .env

        Args:
            well_config: Well configuration dict with lateral_log_name, typewell_name,
                        typewell_log_name, grid_name fields
            landing_end_md: Landing end MD from slicer (used as AG start MD)
        """
        logger.info("Configuring AG settings...")

        # Read dipRange from .env (REQUIRED, no default)
        dip_range_str = os.getenv('DIP_ANGLE_RANGE_DEGREE')

        if dip_range_str is None:
            raise ValueError(
                "DIP_ANGLE_RANGE_DEGREE not set in .env!\n"
                "Set to 0 to skip AG configuration, or specify angle (e.g., 1.0, 2.0, 5.0)"
            )

        dip_range = float(dip_range_str)

        # If 0 - skip AG configuration
        if dip_range == 0:
            logger.info("DIP_ANGLE_RANGE_DEGREE=0, skipping AG configuration")
            return

        if dip_range < 0:
            raise ValueError(f"DIP_ANGLE_RANGE_DEGREE must be >= 0, got: {dip_range}")

        # Build command params: dipRange + agStartMD + optional parameters
        ag_params = {
            "dipRange": dip_range,
            "agStartMD": landing_end_md
        }

        logger.info(f"  agStartMD: {landing_end_md:.2f}m (from landing detection)")

        # Add lateral log name if specified in well config
        lateral_log = well_config.get('lateral_log_name')
        if lateral_log:
            ag_params["logName"] = lateral_log
            logger.info(f"  logName: {lateral_log} (from well config)")
        else:
            logger.info(f"  logName: using GUI default (not specified in well config)")

        # Add typewell log parameters (both must be present or none)
        typewell_name = well_config.get('typewell_name')
        typewell_log = well_config.get('typewell_log_name')

        if typewell_name and typewell_log:
            ag_params["typelogWellName"] = typewell_name
            ag_params["typelogName"] = typewell_log
            logger.info(f"  typelogWellName: {typewell_name} (from well config)")
            logger.info(f"  typelogName: {typewell_log} (from well config)")
        else:
            logger.info(f"  typewell logs: using GUI defaults (not fully specified in well config)")
            if typewell_name and not typewell_log:
                logger.warning(f"  typewell_name present but typewell_log_name missing - both required!")
            if typewell_log and not typewell_name:
                logger.warning(f"  typewell_log_name present but typewell_name missing - both required!")

        # Add smoothness if specified in .env
        smoothness_str = os.getenv('SMOOTHNESS')
        if smoothness_str:
            smoothness = float(smoothness_str)
            ag_params["horizonSmoothness"] = smoothness
            logger.info(f"  horizonSmoothness: {smoothness} (from .env)")

        # Add grid: priority to well config, fallback to .env
        grid_name = well_config.get('grid_name')
        if grid_name:
            ag_params["gridName"] = grid_name
            logger.info(f"  gridName: {grid_name} (from well config)")
        else:
            grid_name_env = os.getenv('SOLO_GRID_NAME')
            if grid_name_env:
                ag_params["gridName"] = grid_name_env
                logger.info(f"  gridName: {grid_name_env} (from .env)")

        # Add pseudo_log_end_md to prevent data leakage
        # TypeLog should not contain data from interpretation zone
        lookback_distance = float(os.getenv('LOOKBACK_DISTANCE', '200.0'))
        ag_start_md = ag_params.get("agStartMD", landing_end_md)
        pseudo_log_end_md = max(ag_start_md, landing_end_md) - lookback_distance
        if pseudo_log_end_md > 0:
            ag_params["pseudoLogEndMd"] = pseudo_log_end_md
            logger.info(f"  pseudoLogEndMd: {pseudo_log_end_md:.2f}m (max(agStart, landing) - lookback)")

        # Send CONFIGURE_AG_SETTINGS command
        self._send_command("CONFIGURE_AG_SETTINGS", ag_params, timeout=60)

        # Validate AG settings response
        with open(self.status_file, 'r', encoding='utf-8') as f:
            status_response = json.load(f)

        if status_response.get("status") == "ERROR":
            error_msg = status_response.get("message", "Unknown error")
            error_details = status_response.get("error", {})
            logger.error(f"CONFIGURE_AG_SETTINGS failed: {error_msg}")
            logger.error(f"  Error details: {json.dumps(error_details, indent=2)}")
            # Build detailed error message with available objects
            details_list = []
            for err in error_details.get("errors", []):
                obj_type = err.get("object_type", "unknown")
                obj_name = err.get("object_name", "unknown")
                details = err.get("details", "")
                available = err.get("available_objects", [])
                msg = f"{obj_type} '{obj_name}': {details}"
                if available:
                    msg += f" Available: {', '.join(available)}"
                details_list.append(msg)
            details_str = "; ".join(details_list) if details_list else "no details"
            raise ValueError(f"AG configuration failed: {error_msg}. Details: {details_str}")

        logger.info(f"✓ AG settings configured")
        logger.info(f"  dipRange: {dip_range} (from .env)")

    def _start_ag(self, target_well_name: str):
        """
        Send START_AG command and wait for AG_RUNNING state
        Then wait for fresh data file in InitialData

        Args:
            target_well_name: Name of target well to monitor data file

        Note: AG start MD is configured via CONFIGURE_AG_SETTINGS (agStartMD parameter)
              START_AG command does not accept parameters
        """
        # Check if AG was configured (dipRange != 0)
        dip_range_str = os.getenv('DIP_ANGLE_RANGE_DEGREE')
        if dip_range_str and float(dip_range_str) == 0:
            logger.info("DIP_ANGLE_RANGE_DEGREE=0, skipping AG start")
            return

        logger.info("Starting AG with configured settings...")

        # Remember time before START_AG
        start_time = time.time()

        self._send_command("START_AG", {}, timeout=60)

        # Wait for AG_RUNNING state
        logger.info("Waiting for AG_RUNNING state...")
        self._wait_for_state("AG_RUNNING", timeout=30)

        logger.info("✓ AG state: AG_RUNNING")

        # Wait for fresh data file
        logger.info(f"Waiting for fresh data file: {target_well_name}.json...")
        self._wait_for_fresh_data_file(target_well_name, start_time, timeout=30)

        logger.info("✓ AG started and running with fresh data")

    def _wait_for_fresh_data_file(self, well_name: str, start_time: float, timeout: float = 30):
        """
        Wait for fresh data file in AG_DATA/InitialData after AG start

        Args:
            well_name: Well name (e.g., "slicing_well")
            start_time: Timestamp when START_AG was sent
            timeout: Maximum time to wait (seconds)

        Raises:
            TimeoutError: If fresh file doesn't appear within timeout
        """
        data_file_path = self.ag_data_dir / f"{well_name}.json"
        check_interval = 0.5  # Check every 0.5 second
        max_retries = 10  # Retry on race condition (atomic write: delete + rename)

        elapsed = 0
        while elapsed < timeout:
            # Retry loop for race condition when StarSteer atomically rewrites file
            for retry in range(max_retries):
                try:
                    if data_file_path.exists():
                        file_mtime = data_file_path.stat().st_mtime
                        if file_mtime > start_time:
                            logger.info(f"✓ Fresh data file detected (modified {file_mtime - start_time:.1f}s after AG start)")
                            return
                    break  # File exists but not fresh yet, exit retry loop
                except FileNotFoundError:
                    # Race condition: file deleted during atomic write, retry
                    if retry < max_retries - 1:
                        logger.debug(f"Race condition on {well_name}.json, retry {retry + 1}/{max_retries}")
                        time.sleep(0.5)
                    else:
                        logger.warning(f"Race condition persists after {max_retries} retries")
                        break

            time.sleep(check_interval)
            elapsed += check_interval

        raise TimeoutError(f"Fresh data file {well_name}.json not found after {timeout}s")

    def _load_well_json(self, well_name: str, save_raw: bool = True) -> Dict[str, Any]:
        """Load well data from AG_DATA/InitialData with retry for race condition.

        StarSteer uses atomic write: delete old file + rename temp file.
        Between delete and rename, file doesn't exist (race condition window).
        We retry 3 times with 0.5s delay to handle this.
        """
        well_json_path = self.ag_data_dir / f"{well_name}.json"

        max_retries = 3
        retry_delay = 0.5  # seconds

        for attempt in range(max_retries):
            if well_json_path.exists():
                break
            if attempt < max_retries - 1:
                logger.warning(f"File not found (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        if not well_json_path.exists():
            raise FileNotFoundError(f"Well JSON not found after {max_retries} attempts: {well_json_path}")

        logger.info(f"Reading: {well_json_path.name}")

        # Retry read in case file disappears during read (atomic write in progress)
        for attempt in range(max_retries):
            try:
                with open(well_json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                break
            except (FileNotFoundError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Read failed (attempt {attempt + 1}/{max_retries}): {e}, retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    raise

        # Log MD from all sources
        well_md = max(p['measuredDepth'] for p in data['well']['points']) if data.get('well', {}).get('points') else 0
        log_md = max(p['measuredDepth'] for p in data['wellLog']['points']) if data.get('wellLog', {}).get('points') else 0
        grid_md = max(p['measuredDepth'] for p in data['gridSlice']['points']) if data.get('gridSlice', {}).get('points') else 0

        logger.info(f"  RAW from StarSteer: well_md={well_md:.1f} log_md={log_md:.1f} grid_md={grid_md:.1f}")

        # Check consistency
        if abs(well_md - log_md) > 1.0:
            logger.warning(f"  MD MISMATCH: well vs log delta={well_md - log_md:.1f}m")
        if abs(well_md - grid_md) > 1.0:
            logger.warning(f"  MD MISMATCH: well vs grid delta={well_md - grid_md:.1f}m")

        # Save raw copy to comparison dir
        if save_raw:
            self._save_raw_starsteer_file(data, well_name, well_md)

        return data

    def _save_raw_starsteer_file(self, data: Dict[str, Any], well_name: str, current_md: float):
        """Save raw StarSteer file to comparison directory"""
        comparison_dir = os.getenv('JSON_COMPARISON_DIR')
        if not comparison_dir:
            return

        comparison_path = Path(comparison_dir)
        comparison_path.mkdir(exist_ok=True, parents=True)

        version_tag = os.getenv('VERSION_TAG', 'unknown')
        filename = f"{version_tag}_{well_name}_raw_{current_md:.1f}.json"

        with open(comparison_path / filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Saved raw StarSteer: {filename}")

    def _call_night_guard_init(self, well_data: Dict[str, Any], well_name: str) -> Dict[str, Any]:
        """
        Call Night Guard INIT logic

        Creates emulator and calls setup_well_processing() directly.
        Does NOT start monitoring loop (Phase 2).
        """
        # Create emulator instance with alerts disabled
        emulator = DrillingEmulator(self.config, disable_alerts=True)

        # Call setup (Night Guard INIT logic)
        setup_data = emulator.setup_well_processing(well_data, well_name)

        # Add emulator to setup_data for loop access
        setup_data['emulator'] = emulator

        return setup_data

    def _run_slicing_loop(self, setup_data: Dict, target_well_name: str, source_well_name: str):
        """
        Slicing loop: SLICE → wait → process → export → repeat

        Args:
            setup_data: From setup_well_processing() (has emulator, executor)
            target_well_name: e.g. "slicing_well"
            source_well_name: For logging
        """
        logger.info("=" * 70)
        logger.info(f"SLICING LOOP START: {source_well_name}")
        logger.info("=" * 70)

        # Extract from setup_data
        emulator = setup_data['emulator']
        executor = setup_data['executor']

        # Params - CLI argument overrides env var
        slice_distance = float(os.getenv('SLICE_DISTANCE_METERS', '30.0'))
        max_iterations = _CLI_CONFIG.get('max_iterations') or int(os.getenv('MAX_SLICING_ITERATIONS', '1000'))

        # JSON path
        json_path = self.ag_data_dir / f"{target_well_name}.json"

        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            logger.info(f"--- Iteration {iteration}/{max_iterations} ---")

            # Remember mtime BEFORE slice
            old_mtime = json_path.stat().st_mtime

            # 1. SLICE
            self._send_slice_command(slice_distance)

            # 1.5. CHECK if slice was successful and if well is complete
            # StarSteer writes atomically, minimal delay for status update
            time.sleep(0.1)
            slice_status = self._check_slice_completion()

            # Check if we've reached the end of the well
            if not slice_status["can_add_more"]:
                logger.info("=" * 70)
                logger.info(f"✓ WELL COMPLETE: can_add_more=False")
                logger.info(f"  Total slices: {slice_status['slice_count']}")
                logger.info(f"  Remaining MD: {slice_status['remaining_md']:.2f}m")
                logger.info("=" * 70)
                break

            if slice_status["remaining_md"] < slice_distance:
                logger.info("=" * 70)
                logger.info(f"✓ WELL COMPLETE: remaining MD ({slice_status['remaining_md']:.2f}m) < slice distance ({slice_distance}m)")
                logger.info(f"  Total slices: {slice_status['slice_count']}")
                logger.info("=" * 70)
                break

            # 2. WAIT for JSON update (mtime change)
            # Increased timeout to 300s (5 min) as StarSteer may take time to process slice
            try:
                self._wait_for_fresh_data_file(target_well_name, old_mtime, timeout=300)
            except TimeoutError:
                logger.warning("JSON not updated after 300s, checking if end of well...")
                # Double-check with slicer_status before breaking
                slice_status = self._check_slice_completion()
                if not slice_status["can_add_more"]:
                    logger.info("Confirmed: end of well (can_add_more=False)")
                else:
                    logger.error("Timeout but can_add_more=True - possible issue!")
                break

            # 3. LOAD new data
            updated_data = self._load_well_json(target_well_name)

            # 3.5. VALIDATE auto interpretation (fail fast if extreme values from EXTRAPOLATE bug)
            self._validate_auto_interpretation(updated_data, iteration)

            # 4. APPLY normalization (SAME AS NIGHT GUARD!)
            if emulator.processor.python_normalization_enabled:
                updated_data = emulator.processor._apply_normalization(updated_data)

            # 4.5. REMOVE reference data for STEP (prevent full recalculation)
            # These fields trigger full AG recalculation instead of incremental update.
            # They should only be sent on INIT, not on every step.
            # Reference: emulator.py:858-865 (old slicer behavior)
            if 'typeLog' in updated_data:
                # Save snapshot BEFORE removing (for pseudoLogEndMd verification)
                step_current_md = max(p['measuredDepth'] for p in updated_data['well']['points'])
                self._save_typelog_snapshot(updated_data, step_current_md)
                del updated_data['typeLog']
                logger.debug("Removed typeLog for step (prevents full recalc)")
            if 'tops' in updated_data:
                del updated_data['tops']
                logger.debug("Removed tops for step (prevents full recalc)")
            # NOTE: gridSlice NOT removed - C++ code modified to not trigger full recalc
            # See: AutoGeosteeringCalculation.cpp:278 - isFullRecalculationRequired commented out
            if 'tvdTypewellShift' in updated_data:
                del updated_data['tvdTypewellShift']
                logger.debug("Removed tvdTypewellShift for step")

            # 5. UPDATE executor (incremental!)
            result = executor.update_well_data(updated_data)

            # 6. GET interpretation
            interpretation_data = executor.get_interpretation_from_result(result)

            # 6.1. Save OUTPUT to comparison directory
            current_md_for_output = max(p['measuredDepth'] for p in updated_data['well']['points'])
            executor._copy_output_to_comparison_dir(interpretation_data, target_well_name, "step", current_md_for_output)

            # 6.2. Update pseudoLogEndMd for this step (for future C++ use)
            self._update_pseudo_log_end_md(target_well_name, current_md_for_output)

            # 6.5. QUALITY ANALYSIS: Compare StarSteer MANUAL vs Python interpretation
            if interpretation_data:
                # Get CORRECT reference (starredInterpretation, NOT 'interpretation'!)
                starsteer_segments = self._get_reference_interpretation(updated_data, iteration)
                python_segments = interpretation_data['interpretation']['segments']
                current_md_value = max(p['measuredDepth'] for p in updated_data['well']['points'])

                metrics = self.quality_analyzer.analyze_step(
                    starsteer_segments,
                    python_segments,
                    current_md_value,
                    source_well_name
                )

                if metrics:
                    self.quality_analyzer.append_to_csv(metrics, source_well_name, current_md_value)
                    self.quality_analyzer.log_metrics(metrics, source_well_name, current_md_value)

            # 7. EXPORT to StarSteer
            emulator._export_interpretation_to_starsteer(target_well_name, executor)

            # 8. SAVE result
            current_md = max(p['measuredDepth'] for p in updated_data['well']['points'])
            emulator.processor.save_step_result(
                target_well_name,
                current_md,
                result,
                "current",
                interpretation_data
            )

            logger.info(f"✓ Processed MD={current_md:.1f}m")

        logger.info("=" * 70)
        logger.info(f"SLICING LOOP DONE: {source_well_name} ({iteration} iterations)")
        logger.info("=" * 70)

    def _validate_auto_interpretation(self, well_data: Dict[str, Any], iteration: int):
        """
        Validate StarSteer automatic 'interpretation' field for NaN/infinity/extreme values.

        This is the AUTO-GENERATED interpretation that may contain EXTRAPOLATE bugs.
        Should NOT be used as reference for quality comparison!

        Args:
            well_data: Data from StarSteer JSON
            iteration: Current iteration number for error reporting

        Raises:
            ValueError: If interpretation contains NaN or extreme values
        """
        if 'interpretation' not in well_data:
            logger.warning(f"Iteration {iteration}: No 'interpretation' field in StarSteer data")
            return

        segments = well_data['interpretation'].get('segments', [])
        if not segments:
            logger.warning(f"Iteration {iteration}: Empty segments in 'interpretation'")
            return

        import math

        # Check ALL segments for extreme values (not just last!)
        extreme_segments = []
        for i, seg in enumerate(segments):
            start_shift = seg.get('startShift')
            end_shift = seg.get('endShift')

            for field_name, value in [('startShift', start_shift), ('endShift', end_shift)]:
                if value is None:
                    extreme_segments.append((i, field_name, 'NULL'))
                elif not math.isfinite(value):
                    extreme_segments.append((i, field_name, f'NaN/Inf: {value}'))
                elif abs(value) > 1e100:
                    extreme_segments.append((i, field_name, f'EXTREME: {value:.6e}'))

        if extreme_segments:
            warning_lines = [f"Iteration {iteration}: StarSteer 'interpretation' contains INVALID values (EXTRAPOLATE bug)"]
            warning_lines.append(f"Found {len(extreme_segments)} bad values:")
            for seg_idx, field, issue in extreme_segments[:5]:  # Show first 5
                warning_lines.append(f"  Segment {seg_idx}, {field}: {issue}")
            if len(extreme_segments) > 5:
                warning_lines.append(f"  ... and {len(extreme_segments) - 5} more")
            warning_lines.append("NOTE: This field is NOT used for quality comparison - using referenceInterpretation instead")

            logger.warning('\n'.join(warning_lines))
            return  # Continue execution - this field is not used anyway!

        logger.debug(f"Iteration {iteration}: Auto 'interpretation' validated OK ({len(segments)} segments)")

    def _get_reference_interpretation(self, well_data: Dict[str, Any], iteration: int) -> List[Dict[str, float]]:
        """
        Get MANUAL reference interpretation from SOURCE well FOR QUALITY COMPARISON.

        CRITICAL: Must use 'referenceInterpretation' from SOURCE well!
                  DO NOT use 'starredInterpretation' - it's from TARGET well (partial after slicing)!
                  DO NOT use 'interpretation' - it's auto-generated (may contain EXTRAPOLATE bugs)!

        Args:
            well_data: Data from StarSteer JSON
            iteration: Current iteration number for error reporting

        Returns:
            List of segments from reference interpretation (SOURCE well)

        Raises:
            ValueError: If referenceInterpretation is null or invalid
        """
        # ONLY use referenceInterpretation (from SOURCE well)
        if 'referenceInterpretation' not in well_data or well_data['referenceInterpretation'] is None:
            raise ValueError(
                f"Iteration {iteration}: referenceInterpretation is NULL!\n"
                f"\n"
                f"Cannot compare quality without reference from SOURCE well.\n"
                f"\n"
                f"Possible causes:\n"
                f"1. AG was not restarted after ag_config.json was updated\n"
                f"2. Reference UUID not found in source well\n"
                f"3. Config file missing or has empty referenceInterpretation field\n"
                f"\n"
                f"Solution: Restart AG for this well to reload config."
            )

        reference = well_data['referenceInterpretation']
        if not isinstance(reference, dict):
            raise ValueError(
                f"Iteration {iteration}: referenceInterpretation is not a dict: {type(reference)}"
            )

        segments = reference.get('segments', [])
        if not segments:
            raise ValueError(
                f"Iteration {iteration}: referenceInterpretation has no segments!"
            )

        logger.debug(f"Iteration {iteration}: Using referenceInterpretation from SOURCE well ({len(segments)} segments)")
        return segments

    def _send_slice_command(self, distance_meters: float):
        """
        Send slice command to StarSteer via slice_command.json

        Uses WellSlicerController mechanism (not commands.json)

        Note: Uses slice_id for duplicate detection in StarSteer.
        No delay needed - slice_id prevents double processing from QFileSystemWatcher.
        """
        # Increment slice ID counter for duplicate detection
        self.slice_id_counter += 1
        logger.info(f"Sending slice command: {distance_meters}m (slice_id={self.slice_id_counter})")

        slice_data = {
            "action": "add_slice",
            "slice_id": self.slice_id_counter,  # For duplicate detection in StarSteer
            "slice_distance_meters": distance_meters
        }

        # Write slice command (atomic overwrite, no delay needed with slice_id)
        with open(self.slice_command_file, 'w', encoding='utf-8') as f:
            json.dump(slice_data, f, indent=2)

        logger.info(f"✓ Slice command written to {self.slice_command_file.name}")

    def _check_slice_completion(self) -> dict:
        """
        Read slicer_status.json to check if well slicing is complete

        Returns:
            dict with keys:
                - can_add_more (bool): Whether more slices can be added
                - remaining_md (float): Remaining MD in meters
                - slice_count (int): Number of slices processed
        """
        # Polling: 10 attempts with 100ms interval (1 second total)
        max_attempts = 10
        for attempt in range(max_attempts):
            if not self.slicer_status_file.exists():
                if attempt < max_attempts - 1:
                    time.sleep(0.1)
                    continue
                logger.warning("slicer_status.json not found after 10 attempts, assuming can continue")
                return {"can_add_more": True, "remaining_md": float('inf'), "slice_count": 0}

            try:
                with open(self.slicer_status_file, 'r', encoding='utf-8') as f:
                    status = json.load(f)
                break  # Success
            except json.JSONDecodeError:
                if attempt < max_attempts - 1:
                    time.sleep(0.1)
                    continue
                raise  # Re-raise on last attempt

        streaming = status.get("streaming", {})
        config = status.get("configuration", {})

        result = {
            "can_add_more": streaming.get("can_add_more", True),
            "remaining_md": config.get("remaining_md_meters", float('inf')),
            "slice_count": streaming.get("slice_count", 0)
        }

        logger.info(f"Slicer status: can_add_more={result['can_add_more']}, "
                   f"remaining_md={result['remaining_md']:.2f}m, "
                   f"slice_count={result['slice_count']}")

        return result


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='StarSteer Slicer - run AG slicing with EXE or Python DE optimizer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python slicer.py                           # Default: EXE mode, default StarSteer dir
  python slicer.py --de                      # DE mode: Python executor, 1 iteration
  python slicer.py --de --max-iterations 5   # DE mode with 5 iterations
  python slicer.py --starsteer-dir "E:/..."  # Custom StarSteer directory
        """
    )

    parser.add_argument(
        '--starsteer-dir',
        type=str,
        default=None,
        help=f'StarSteer directory path (default: {DEFAULT_SLICER_STARSTEER_DIR})'
    )

    parser.add_argument(
        '--de', '--differential-evolution',
        action='store_true',
        dest='use_de',
        help='Use Python Differential Evolution optimizer instead of C++ EXE'
    )

    parser.add_argument(
        '--max-iterations',
        type=int,
        default=None,
        help='Maximum slicing iterations (default: 1 for DE mode, 1000 for EXE mode)'
    )

    return parser.parse_args()


def main():
    """Entry point for standalone execution"""
    global _CLI_CONFIG

    # Parse arguments
    args = parse_args()

    # Setup logging first
    setup_logging(os.getenv('LOG_LEVEL', 'INFO'))

    # Apply CLI config
    _CLI_CONFIG['starsteer_dir'] = args.starsteer_dir

    # DE mode: enable Python executor and default to 1 iteration
    if args.use_de:
        os.environ['PYTHON_EXECUTOR_ENABLED'] = 'true'
        _CLI_CONFIG['python_executor'] = True
        # 0 means unlimited (use large number), None means default to 1
        _CLI_CONFIG['max_iterations'] = args.max_iterations if args.max_iterations is not None else 1
        if _CLI_CONFIG['max_iterations'] == 0:
            _CLI_CONFIG['max_iterations'] = 10000  # effectively unlimited
        logger.info("=" * 60)
        logger.info("DE MODE: Python Differential Evolution optimizer")
        logger.info(f"  Max iterations: {_CLI_CONFIG['max_iterations']}")
        logger.info("=" * 60)
    else:
        # EXE mode: use env setting for python_executor
        _CLI_CONFIG['max_iterations'] = args.max_iterations  # None = use env default (1000)
        logger.info("EXE MODE: C++ AutoGeosteering daemon")

    logger.info("Starting StarSteer Slicer Orchestrator")

    try:
        orchestrator = StarSteerSlicerOrchestrator()
        orchestrator.run()
        return 0
    except Exception as e:
        logger.error(f"Orchestrator failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
