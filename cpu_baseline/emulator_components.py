#!/usr/bin/env python3
"""
Base components for AutoGeosteering Emulator
Contains reusable classes and functions for both emulator and nightguard modes
"""

import json
import time
import subprocess
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from copy import deepcopy
import os
import numpy as np

logger = logging.getLogger(__name__)


class WellDataSlicer:
    """Slices well data up to a specific measured depth"""

    def slice_well_data(self, well_data: Dict[str, Any], max_md: float) -> Dict[str, Any]:
        """Slice well data up to max_md"""
        sliced_data = deepcopy(well_data)  # CRITICAL: Deep copy to avoid modifying original

        # Track original counts for logging
        original_trajectory_count = len(sliced_data['well']['points'])
        original_log_count = len(sliced_data['wellLog']['points'])

        # Slice trajectory points
        if 'well' in sliced_data and 'points' in sliced_data['well']:
            sliced_points = [
                point for point in sliced_data['well']['points']
                if point['measuredDepth'] <= max_md
            ]
            sliced_data['well']['points'] = sliced_points

        # Slice well log data
        if 'wellLog' in sliced_data and 'points' in sliced_data['wellLog']:
            sliced_log_points = [
                point for point in sliced_data['wellLog']['points']
                if point['measuredDepth'] <= max_md
            ]
            sliced_data['wellLog']['points'] = sliced_log_points

        # TypeLog data is NOT sliced - reference data for entire well

        # Log the slicing results
        well_name = well_data['wellName']
        start_md = well_data['autoGeosteeringParameters']['startMd']
        sliced_count = len(sliced_data['wellLog']['points'])
        logger.info(
            f"Well: {well_name}, StartMD: {start_md}, Sliced to MD={max_md}: "
            f"{original_log_count} -> {sliced_count} points, Status: preparing data")

        return sliced_data


class AutoGeosteeringExecutor:
    """C++ daemon executor for AutoGeosteering calculations"""

    def __init__(self, exe_path: str, work_dir: str):
        self.exe_path = exe_path
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)

        # File paths for communication with daemon
        self.input_file = self.work_dir / "input.json"
        self.output_file = self.work_dir / "output.json"
        self.data_ready_flag = self.work_dir / "data_ready.flag"
        self.interpretation_ready_flag = self.work_dir / "interpretation_ready.flag"
        self.exit_flag = self.work_dir / "exit.flag"

        self.process = None
        self.timeout = int(os.getenv('DAEMON_TIMEOUT_SECONDS', '300'))

        # Get interpretation directory from exe
        self.interpretation_dir = self._get_interpretation_path_from_exe()

    def _get_interpretation_path_from_exe(self) -> str:
        """Get interpretation directory path dynamically from exe output"""
        logger.info(f"Detecting interpretation directory from exe...\nexe: {self.exe_path}\npath:{self.work_dir}")

        # Run exe without parameters to get test interpretation path
        result = subprocess.run([self.exe_path],
                                capture_output=True,
                                text=True,
                                timeout=30,
                                cwd=self.work_dir)

        assert result.returncode == 0, f"Exe failed with return code {result.returncode}"

        # Parse output for "Full path: ..."
        for line in result.stdout.split('\n'):
            if 'Full path:' in line:
                full_path = line.split('Full path:')[1].strip()
                # Extract directory from full path to testwell.json
                interpretation_dir = str(Path(full_path).parent)
                logger.info(f"Detected interpretation directory: {interpretation_dir}")
                return interpretation_dir

        assert False, "Could not find 'Full path:' in exe output"

    def start_daemon(self):
        """Test interpretation path only, don't start persistent process"""
        logger.info("Testing interpretation synchronization...")
        self._test_interpretation_sync()
        self._cleanup_flags()
        logger.info("Ready for daemon initialization")

    def _test_interpretation_sync(self):
        """Test that interpretation directory is properly synchronized"""
        logger.info("Testing interpretation synchronization...")

        # Test file should already exist from path detection
        test_file = Path(self.interpretation_dir) / "testwell.json"

        assert test_file.exists(), f"Test interpretation not found at: {test_file.absolute()}"

        # Verify file is not empty and contains valid JSON
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            assert test_data['interpretation'], "Test interpretation file is invalid"

        logger.info("✅ Interpretation synchronization test passed")
        logger.info(f"Test file verified at: {test_file.absolute()}")

    def initialize_well(self, well_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize well with first dataset"""
        logger.info("Initializing well data")

        # 0. Clear old interpretation file
        well_name = well_data['wellName']
        Path(self.interpretation_dir, f"{well_name}.json").unlink(missing_ok=True)

        # 1. Write input.json BEFORE starting daemon
        self._write_json(self.input_file, well_data)

        # 1a. Copy to comparison directory
        self._copy_to_comparison_dir(well_data, "init")

        # 2. Start daemon with parameters
        self.process = subprocess.Popen(
            [self.exe_path, str(self.input_file), str(self.output_file)],
            cwd=self.work_dir
        )

        # 3. Wait for first calculation completion
        start_time = time.time()
        while not self.interpretation_ready_flag.exists():
            assert time.time() - start_time <= self.timeout, \
                f"Daemon initialization timeout after {self.timeout} seconds"
            time.sleep(0.1)

        # 4. Read result and clean duplicates
        result = self._read_json(self.output_file)
        result = self._clean_cpp_interpretation(result)

        self.interpretation_ready_flag.unlink(missing_ok=True)

        return result

    def update_well_data(self, well_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update well with new data"""
        return self._send_command("update", well_data)

    def _send_command(self, command: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Send command to daemon and wait for response"""
        # Write data directly without wrapper - C++ expects unwrapped data
        if input_data is not None:
            self._write_json(self.input_file, input_data)

            # Copy to comparison directory
            self._copy_to_comparison_dir(input_data, "step")

        # Python writes atomically via .tmp + replace, no delay needed

        # Signal daemon that data is ready
        self.data_ready_flag.touch()

        # Wait for daemon to process
        start_time = time.time()
        while not self.interpretation_ready_flag.exists():
            assert time.time() - start_time <= self.timeout, \
                f"Daemon timeout after {self.timeout} seconds"
            time.sleep(0.1)

        # Read result and clean duplicates
        result = self._read_json(self.output_file)
        result = self._clean_cpp_interpretation(result)

        # Clean up flags
        self.interpretation_ready_flag.unlink(missing_ok=True)

        return result

    def _clean_cpp_interpretation(self, result_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean duplicate startMd in results from C++ daemon"""
        if 'interpretation' not in result_data:
            return result_data

        interpretation = result_data['interpretation']
        if 'segments' not in interpretation or not interpretation['segments']:
            return result_data

        segments = interpretation['segments']

        # Group segments by startMd
        segments_by_md = {}
        for segment in segments:
            start_md = segment['startMd']
            if start_md not in segments_by_md:
                segments_by_md[start_md] = []
            segments_by_md[start_md].append(segment)

        # Keep last segment for each startMd
        cleaned_segments = []
        duplicate_count = 0

        for start_md in sorted(segments_by_md.keys()):
            segment_group = segments_by_md[start_md]
            if len(segment_group) > 1:
                duplicate_count += len(segment_group) - 1
                logger.warning(
                    f"Found {len(segment_group)} segments from C++ with startMd={start_md}, keeping the last one")

            # Take last segment from the group
            cleaned_segments.append(segment_group[-1])

        # Update data
        result_data['interpretation']['segments'] = cleaned_segments

        if duplicate_count > 0:
            logger.info(f"Cleaned {duplicate_count} duplicate segments from C++ daemon result")
            logger.info(f"C++ segments count: {len(segments)} -> {len(cleaned_segments)}")

        return result_data

    def _write_json(self, file_path: Path, data: Dict[str, Any]):
        """Safely write JSON file using temporary file"""
        temp_path = file_path.with_suffix('.json.tmp')
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        temp_path.replace(file_path)

    def _read_json(self, file_path: Path) -> Dict[str, Any]:
        """Read JSON file"""
        with open(file_path, 'r', encoding="utf-8") as f:
            return json.load(f)

    def _copy_to_comparison_dir(self, well_data: Dict[str, Any], step_type: str):
        """Copy input JSON to comparison directory for debugging"""
        comparison_dir = os.getenv('JSON_COMPARISON_DIR')
        if not comparison_dir:
            return

        comparison_path = Path(comparison_dir)
        comparison_path.mkdir(exist_ok=True, parents=True)

        version_tag = os.getenv('VERSION_TAG', 'unknown')
        well_name = well_data['wellName']

        # Clean old files for this well on init
        if step_type == "init":
            pattern = f"{version_tag}_{well_name}_*.json"
            for old_file in comparison_path.glob(pattern):
                old_file.unlink()
                logger.debug(f"Removed old file: {old_file.name}")
            logger.info(f"Cleaned old comparison files: {pattern}")

        # Get current MD from well data
        trajectory_points = well_data['well']['points']
        current_md = max(p['measuredDepth'] for p in trajectory_points) if trajectory_points else 0

        # Format: {version}_{wellname}_{step}_{md}.json
        filename = f"{version_tag}_{well_name}_{step_type}_{current_md:.1f}.json"
        comparison_file = comparison_path / filename

        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(well_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Copied to comparison: {filename}")

    def _copy_output_to_comparison_dir(self, interpretation_data: Dict[str, Any], well_name: str, step_type: str, current_md: float):
        """Copy OUTPUT interpretation JSON to comparison directory for debugging"""
        comparison_dir = os.getenv('JSON_COMPARISON_DIR')
        if not comparison_dir:
            return

        comparison_path = Path(comparison_dir)
        comparison_path.mkdir(exist_ok=True, parents=True)

        version_tag = os.getenv('VERSION_TAG', 'unknown')

        # Format: {version}_{wellname}_{step}_{md}_output.json
        filename = f"{version_tag}_{well_name}_{step_type}_{current_md:.1f}_output.json"
        comparison_file = comparison_path / filename

        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(interpretation_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Copied OUTPUT to comparison: {filename}")

    def _cleanup_flags(self):
        """Remove all flag files"""
        for flag_file in [self.data_ready_flag, self.interpretation_ready_flag, self.exit_flag]:
            flag_file.unlink(missing_ok=True)

    def stop_daemon(self):
        """Stop the daemon process"""
        if self.process:
            logger.info("Stopping daemon")

            # Signal daemon to exit
            self.exit_flag.touch()

            # Wait for graceful shutdown
            self.process.wait(timeout=5)

            self._cleanup_flags()
            logger.info("Daemon stopped")

    def get_interpretation_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract interpretation from C++ executor result"""
        relative_path = result['interpretationPath']
        file_name = Path(relative_path).name
        absolute_path = Path(self.interpretation_dir) / file_name

        logger.debug(f"INTERP_DEBUG: Reading interpretation from: {absolute_path}")
        time.sleep(0.1)
        interpretation_data = self._read_json(absolute_path)
        segments_count = len(interpretation_data.get('interpretation', {}).get('segments', []))
        logger.debug(f"INTERP_DEBUG: Loaded interpretation with {segments_count} segments")

        return interpretation_data

    def __enter__(self):
        """Context manager entry"""
        self.start_daemon()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_daemon()


def setup_cpp_env_config(work_dir: str):
    """Copy .env to work directory once for C++ daemon to read"""
    project_env = Path('.env')
    exe_env = Path(work_dir) / '.env'

    assert project_env.exists(), f"Project .env file not found: {project_env.absolute()}"

    shutil.copy2(project_env, exe_env)
    logger.info(f"Copied .env to C++ work directory: {exe_env}")
