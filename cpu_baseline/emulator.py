#!/usr/bin/env python3
"""
Enhanced Multi-Drilling Emulator with unified processing for both modes
Supports both file-based emulation and real-time PAPI nightguard monitoring
"""

import json
import logging
import time
import traceback
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment and setup logging
load_dotenv()
log_level = os.getenv('LOG_LEVEL', 'INFO')
numeric_level = getattr(logging, log_level.upper(), logging.INFO)
logging.basicConfig(
    level=numeric_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import unified components
from emulator_processor import WellProcessor
from emulator_components import WellDataSlicer

# Import state management
from wells_state_manager import (
    load_processed_wells_state,
    save_processed_wells_state,
    scan_results_directory,
    sync_wells_state,
    check_timeouts,
    get_next_pending_well,
    update_well_status,
    print_state_summary
)

# Import PAPI loader for nightguard mode
from papi_loader import PAPILoader

# Import PAPI uploader if enabled
from papi_export.uploaders.interpretation_uploader import InterpretationUploader

# Import Night Guard Alert System
from alerts.alert_analyzer import AlertAnalyzer
from alerts.starsteer_target_line_interface import StarSteerTargetLineInterface
from night_guard_logger import NightGuardLogger

# Import AG visualization if enabled
from ag_visualization.ag_visual_edit import InteractivePlot
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json
from copy import deepcopy


class DrillingEmulator:
    """Main drilling emulator with unified processing for both modes"""
    
    def __init__(self, config: Dict[str, Any], disable_alerts: bool = False):
        """
        Initialize emulator with configuration

        Args:
            config: Configuration dict
            disable_alerts: If True, force disable alerts regardless of .env
                           Used by slicer.py to avoid manual .env changes
        """
        # Store configuration
        self.config = config
        self.exe_path = config['exe_path']
        self.wells_dir = config['wells_dir']
        self.work_dir = config['work_dir']
        self.results_dir = config['results_dir']
        self.md_step = config['md_step_meters']
        self.quality_threshold = config['quality_thresholds_meters']
        
        # Operating mode from .env
        self.mode = os.getenv('EMULATOR_MODE', 'emulator')
        logger.info(f"Operating mode: {self.mode}")
        
        # Continue mode configuration (for emulator mode)
        self.continue_mode = config.get('continue_mode', False)
        
        # Detect display availability for visualization
        self.interactive_mode = self._detect_display()
        logger.info(f"Interactive mode: {self.interactive_mode}")
        
        # Create directories
        Path(self.work_dir).mkdir(exist_ok=True)
        Path(self.results_dir).mkdir(exist_ok=True)
        
        # Initialize unified processor
        processor_config = {
            'exe_path': self.exe_path,
            'work_dir': self.work_dir,
            'results_dir': self.results_dir,
            'md_step_meters': self.md_step,
            'mode': self.mode,
            'landing_detection_enabled': config.get('landing_detection_enabled', False),
            'landing_offset_meters': config.get('landing_offset_meters', 100.0),
            'landing_alternative_offset_meters': config.get('landing_alternative_offset_meters', 50.0),
            'landing_start_angle_deg': config.get('landing_start_angle_deg', 60.0),
            'landing_max_length_meters': config.get('landing_max_length_meters', 200.0),
            'landing_perch_stability_threshold': config.get('landing_perch_stability_threshold', 0.5),
            'landing_perch_min_angle_deg': config.get('landing_perch_min_angle_deg', 30.0),
            'landing_perch_stability_window': config.get('landing_perch_stability_window', 5)
        }
        
        self.processor = WellProcessor(processor_config)
        self.slicer = WellDataSlicer()

        # NIGHT GUARD FIX: Initialize shared PAPI loader for thread-safe instance reuse
        self._shared_papi_loader = None
        self._papi_loader_lock = threading.Lock()

        # Initialize PAPI components based on mode
        if self.mode == 'nightguard':
            # Read well name from StarSteer status.json
            starsteer_dir = Path(os.getenv('STARSTEER_DIR', ''))
            if not starsteer_dir:
                raise ValueError("STARSTEER_DIR not set in .env for nightguard mode")

            status_json_path = starsteer_dir / 'status.json'
            if not status_json_path.exists():
                raise FileNotFoundError(f"StarSteer status.json not found: {status_json_path}")

            try:
                with open(status_json_path, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                well_name = status_data.get('application_state', {}).get('well', {}).get('name', '')
                if not well_name:
                    raise ValueError(f"Well name not found in status.json: {status_json_path}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in status.json: {e}")

            logger.info(f"Read well name from status.json: {well_name}")

            # Save well name as instance attribute for later use
            self.nightguard_well_name = well_name

            # Create config dict from .env variables
            config_dict = {
                'project_name': os.getenv('SOLO_PROJECT_NAME'),
                'lateral_log_name': os.getenv('SOLO_LATERAL_LOG_NAME'),
                'typewell_log_name': os.getenv('SOLO_TYPEWELL_LOG_NAME'),
                'typewell_name': os.getenv('TYPEWELL_NAME'),
                'horizon_name': os.getenv('SOLO_HORIZON_NAME'),
                'grid_name': os.getenv('SOLO_GRID_NAME'),
                'enable_log_normalization': True,
                'process': True
            }

            # NOTE: Removed self.papi_loader - using shared_papi_loader for consistency
            self.nightguard_well_name = well_name
            self.polling_interval = int(os.getenv('NIGHTGUARD_POLLING_INTERVAL', '30'))
            self.auto_exit_timeout_minutes = int(os.getenv('NIGHT_GUARD_AUTO_EXIT_MINUTES', '10'))
            self.debug_save_json = os.getenv('NIGHTGUARD_DEBUG_SAVE_JSON', 'false').lower() == 'true'
            
            # Track last MD values for change detection
            self.last_trajectory_md = None
            self.last_log_md = None
            self._last_papi_output_path = None
            
            logger.info(f"Nightguard mode initialized for well: {well_name}")
            logger.info(f"Polling interval: {self.polling_interval}s")

            # Initialize Night Guard Debug Logger FIRST
            self.ng_logger = NightGuardLogger()

            # Check data source - initialize PAPI only if needed
            data_source = os.getenv('NIGHTGUARD_DATA_SOURCE', 'papi').lower()
            logger.info(f"NIGHT_GUARD_DEBUG: Data source: {data_source}")

            if data_source == 'papi':
                # Initialize shared PAPI Loader with full interpretation data from API
                logger.info(f"NIGHT_GUARD_DEBUG: Initializing shared PAPI Loader for Night Guard mode")
                shared_papi_loader = self._get_papi_loader()
                if shared_papi_loader:
                    # Load full interpretation data from PAPI API (includes horizon_tvdss)
                    interpretation_name = os.getenv('PAPI_SOURCE_INTERPRETATION_NAME', '')
                    try:
                        shared_papi_loader.run(interpretation_name=interpretation_name)
                        logger.info(f"NIGHT_GUARD_DEBUG: Shared PAPI Loader initialization completed")
                    except ValueError as e:
                        logger.error(f"CRITICAL: Failed to initialize PAPI Loader - {e}")

                        # Log to Night Guard for visibility
                        if self.ng_logger:
                            self.ng_logger.log_error("papi_loader_initialization", str(e), {
                                "well_name": well_name,
                                "interpretation_name": interpretation_name or "starred"
                            })

                        raise  # Re-raise for initialization failure
                else:
                    logger.error("NIGHT_GUARD_DEBUG: Failed to create shared PAPI loader")
            else:
                logger.info(f"NIGHT_GUARD_DEBUG: StarSteer mode - skipping PAPI Loader initialization")

            # Nightguard flag initialization - Phase 1: CUT
            wait_for_cut = os.getenv('WAIT_FOR_CUT_FLAG')
            if wait_for_cut is None:
                raise ValueError("WAIT_FOR_CUT_FLAG not set in .env - must be 'true' or 'false'")

            if wait_for_cut.lower() == 'true':
                self._send_cut_flag_and_wait()
            else:
                logger.info("Skipping CUT flag coordination (WAIT_FOR_CUT_FLAG=false)")
        
        # Initialize PAPI uploader if enabled
        self.papi_upload_enabled = os.getenv('PAPI_UPLOAD_ENABLED', 'false').lower() == 'true'
        self.papi_uploader = None
        self._last_upload_time = None  # Track last upload timestamp for batching
        # Note: InterpretationUploader will be initialized later with project_measure_unit from PAPI
        if self.papi_upload_enabled:
            logger.info("PAPI interpretation upload ENABLED (will initialize uploader after getting project units)")
        else:
            logger.info("PAPI interpretation upload disabled")

        # NEW: Initialize Night Guard Alert System (only for nightguard mode)
        # Can be disabled via disable_alerts parameter (used by slicer.py)
        alerts_from_env = os.getenv('ALERT_ANALYZER_ENABLED', 'false').lower() == 'true'
        self.alert_analyzer_enabled = (
            self.mode == 'nightguard' and
            alerts_from_env and
            not disable_alerts  # Override: force disable if True
        )
        self.alert_analyzer = None
        self.starsteer_interface = None

        # Initialize Night Guard Debug Logger (if not already initialized in nightguard mode)
        if not hasattr(self, 'ng_logger'):
            self.ng_logger = NightGuardLogger() if self.mode == 'nightguard' else None

        if self.ng_logger:
            self.ng_logger.log_initialization("mode", True, {"mode": self.mode})
            self.ng_logger.log_initialization("alert_analyzer_enabled", self.alert_analyzer_enabled)

        if self.alert_analyzer_enabled:
            alert_config = {
                'BASE_TVD_METERS': os.getenv('BASE_TVD_METERS', '4000.0'),
                'ALERT_THRESHOLD_FEET': os.getenv('ALERT_THRESHOLD_FEET', '7.5'),
                'ALERTS_DIR': os.getenv('ALERTS_DIR', 'alerts/'),
                'SMS_ENABLED': os.getenv('SMS_ENABLED', 'false'),
                'TWILIO_ACCOUNT_SID': os.getenv('TWILIO_ACCOUNT_SID', ''),
                'TWILIO_AUTH_TOKEN': os.getenv('TWILIO_AUTH_TOKEN', ''),
                'TWILIO_FROM_NUMBER': os.getenv('TWILIO_FROM_NUMBER', ''),
                'TWILIO_TO_NUMBERS': os.getenv('TWILIO_TO_NUMBERS', ''),
                'NOTIFICATION_RATE_LIMIT_MINUTES': os.getenv('NOTIFICATION_RATE_LIMIT_MINUTES', '10'),
                'TELEGRAM_ENABLED': os.getenv('TELEGRAM_ENABLED', 'false'),
                'TG_BOT_KEY': os.getenv('TG_BOT_KEY', ''),
                'TG_CHAT_ID': os.getenv('TG_CHAT_ID', '')
            }
            self.alert_analyzer = AlertAnalyzer(alert_config)
            logger.info("Night Guard Alert System initialized")

            # Prepare StarSteer config (will initialize interface later with project_measure_unit)
            self._starsteer_config = {
                'STARSTEER_DIRECTORY': os.getenv('STARSTEER_DIRECTORY', 'SS_slicer'),
                'STARSTEER_TIMEOUT_SECONDS': os.getenv('STARSTEER_TIMEOUT_SECONDS', '60'),
                'STARSTEER_POLL_INTERVAL': os.getenv('STARSTEER_POLL_INTERVAL', '2.0'),
                'STARSTEER_MAX_RETRIES': os.getenv('STARSTEER_MAX_RETRIES', '3'),
                'NIGHTGUARD_WELL_NAME': self.nightguard_well_name,
                'TARGET_LINE_NAME': os.getenv('TARGET_LINE_NAME'),
                'PROJECT_MEASURE_UNIT': os.getenv('PROJECT_MEASURE_UNIT')
            }

            # Initialize StarSteer if PROJECT_MEASURE_UNIT is already in environment
            if self._starsteer_config['PROJECT_MEASURE_UNIT']:
                self.starsteer_interface = StarSteerTargetLineInterface(self._starsteer_config, self.ng_logger)
                target_line_source = os.getenv('TARGET_LINE_SOURCE', 'papi').lower()
                logger.info(f"StarSteer Target Line Interface initialized (TARGET_LINE_SOURCE={target_line_source})")
            else:
                self.starsteer_interface = None
                logger.info("StarSteer initialization deferred until project_measure_unit is available from PAPI")

            if self.ng_logger:
                self.ng_logger.log_initialization("alert_analyzer", True, alert_config)
                self.ng_logger.log_initialization("starsteer_interface", self.starsteer_interface is not None, self._starsteer_config)
        else:
            if self.mode == 'nightguard':
                logger.info("Night Guard Alert System disabled (ALERT_ANALYZER_ENABLED=false)")
                if self.ng_logger:
                    self.ng_logger.log_initialization("alert_analyzer", False, {"reason": "ALERT_ANALYZER_ENABLED=false"})
            else:
                logger.debug(f"Night Guard not applicable for mode: {self.mode}")
        
        logger.info(f"MD range step: {self.md_step}m")
        logger.info(f"Continue mode: {self.continue_mode}")

    # NIGHT GUARD FIX: Shared PAPI loader management methods
    def _get_papi_loader(self):
        """Get shared PAPI Loader instance with thread safety

        Returns:
            PAPILoader: Shared PAPI Loader instance for Night Guard mode
        """
        if self.mode != 'nightguard':
            # For non-nightguard modes, create individual instances as before
            return None

        with self._papi_loader_lock:
            if self._shared_papi_loader is None:
                # Create shared instance for nightguard mode
                well_name = self.nightguard_well_name

                # Get source interpretation name - OPTIONAL (can be None/empty for starred)
                # NOTE: This is NOT added to config_dict - will be passed to run() instead

                config_dict = {
                    'project_name': os.getenv('SOLO_PROJECT_NAME'),
                    'lateral_log_name': os.getenv('SOLO_LATERAL_LOG_NAME'),
                    'typewell_log_name': os.getenv('SOLO_TYPEWELL_LOG_NAME'),
                    'horizon_name': os.getenv('SOLO_HORIZON_NAME'),
                    'grid_name': os.getenv('SOLO_GRID_NAME'),
                    'enable_log_normalization': True,
                    'process': True
                }
                self._shared_papi_loader = PAPILoader(well_name=well_name, config_dict=config_dict, ng_logger=self.ng_logger)
                logger.info(f"NIGHT_GUARD_DEBUG: Created shared PAPI Loader instance ID: {id(self._shared_papi_loader)}")

                # Night Guard checkpoint
                if self.ng_logger:
                    self.ng_logger.log_checkpoint("papi_loader_created", {
                        "well_name": well_name,
                        "project_name": config_dict['project_name']
                    })
            return self._shared_papi_loader

    def _validate_papi_loader_cache(self) -> bool:
        """Validate shared PAPI loader cache integrity

        Returns:
            bool: True if cache is available and valid, False otherwise
        """
        if self.mode != 'nightguard':
            logger.info("NIGHT_GUARD_DEBUG: Cache validation skipped - not in nightguard mode")
            return True  # Not applicable for non-nightguard modes

        logger.info("NIGHT_GUARD_DEBUG: Starting cache validation...")

        papi_loader = self._shared_papi_loader
        if not papi_loader:
            logger.error("NIGHT_GUARD_DEBUG: No shared PAPI loader instance available")
            return False

        logger.info(f"NIGHT_GUARD_DEBUG: Found shared PAPI loader instance ID: {id(papi_loader)}")

        if not hasattr(papi_loader, '_last_interpretation_data'):
            logger.error("NIGHT_GUARD_DEBUG: PAPI Loader missing '_last_interpretation_data' attribute")
            return False

        logger.info("NIGHT_GUARD_DEBUG: PAPI Loader has '_last_interpretation_data' attribute")

        if not papi_loader._last_interpretation_data:
            logger.error("NIGHT_GUARD_DEBUG: PAPI Loader interpretation cache is None or empty")
            logger.info(f"NIGHT_GUARD_DEBUG: Cache value: {papi_loader._last_interpretation_data}")
            return False

        logger.info("NIGHT_GUARD_DEBUG: PAPI Loader cache is not empty")

        # Detailed cache content validation
        cache_data = papi_loader._last_interpretation_data
        logger.info(f"NIGHT_GUARD_DEBUG: Cache data type: {type(cache_data)}")
        logger.info(f"NIGHT_GUARD_DEBUG: Cache keys: {list(cache_data.keys()) if isinstance(cache_data, dict) else 'Not a dict'}")

        required_keys = ['horizon_tvdss', 'segments']

        for key in required_keys:
            if key not in cache_data:
                logger.error(f"NIGHT_GUARD_DEBUG: Missing required cache key: '{key}' - Available keys: {list(cache_data.keys())}")
                return False

        horizon_count = len(cache_data['horizon_tvdss'])
        segments_count = len(cache_data['segments'])

        logger.info(f"NIGHT_GUARD_DEBUG: Cache validation - horizon_tvdss: {horizon_count} points")
        logger.info(f"NIGHT_GUARD_DEBUG: Cache validation - segments: {segments_count} items")

        if horizon_count == 0:
            logger.error("NIGHT_GUARD_DEBUG: Cache validation FAILED - No horizon points available")
            return False

        logger.info(f"NIGHT_GUARD_DEBUG: Cache validation SUCCESSFUL - {horizon_count} horizon points, {segments_count} segments")
        return True

    def _detect_display(self) -> bool:
        """Detect if display is available for visualization"""
        import matplotlib
        backend = matplotlib.get_backend().lower()
        return 'agg' not in backend
    
    def setup_well_processing(self, well_data: Dict[str, Any], well_name: str):
        """
        Setup and prepare single well for processing (without iterations)

        Args:
            well_data: Complete well data from JSON or PAPI
            well_name: Name of the well
        """
        logger.info(f"Processing well: {well_name} in {self.mode} mode")

        # Initialize InterpretationUploader if enabled and not yet initialized (for emulator mode)
        if self.papi_upload_enabled and self.papi_uploader is None:
            from papi_export.uploaders.interpretation_uploader import InterpretationUploader

            # Try to get project_measure_unit from well_data (exported JSON)
            project_measure_unit = well_data.get('_metadata', {}).get('project_measure_unit')

            if not project_measure_unit:
                logger.warning("project_measure_unit not found in well_data metadata.")
                # Use shared PAPI Loader (already loaded) to get project_measure_unit
                if self._shared_papi_loader and hasattr(self._shared_papi_loader, 'project_measure_unit'):
                    project_measure_unit = self._shared_papi_loader.project_measure_unit
                    logger.info(f"Retrieved project_measure_unit from shared PAPI Loader: {project_measure_unit}")
                else:
                    logger.error("shared_papi_loader not available or project_measure_unit not set")
                    project_measure_unit = None

            if project_measure_unit:
                self.papi_uploader = InterpretationUploader(project_measure_unit=project_measure_unit)
                logger.info(f"InterpretationUploader initialized with project units: {project_measure_unit}")
            else:
                logger.error("Cannot initialize InterpretationUploader without project_measure_unit")
                self.papi_upload_enabled = False

        # Clean up interpretations once at the start (if PAPI upload enabled)
        if self.papi_upload_enabled and self.papi_uploader:
            self._cleanup_interpretations_for_well(well_name)
        
        # UNIFIED PREPARATION - same for both modes
        preparation = self.processor.prepare_well_for_processing(well_data)

        # VALIDATE reference interpretation (fail fast if corrupted)
        self._validate_reference_interpretation_initial(well_data, well_name)

        # Extract preparation results
        start_md = preparation['start_md']
        max_md = preparation['max_md']
        current_md = preparation['initial_md']
        executor = preparation['executor']
        interpretation_data = preparation['initial_interpretation']
        prepared_well_data = preparation['prepared_well_data']
        
        # Create AG visualizer if in interactive mode (emulator only)
        visualizer = None
        if self.interactive_mode and self.mode == 'emulator':
            visualizer = self._create_ag_visualizer(
                well_data,
                well_name,
                preparation['original_start_md'],
                preparation['detected_start_md']
            )
            # Update with initial interpretation
            if visualizer and interpretation_data:
                visualizer.update_interpretation(interpretation_data, current_md)
        
        # Save initial result
        self.processor.save_step_result(
            well_name,
            current_md,
            preparation['initial_result'],
            "init",
            interpretation_data
        )
        
        # Upload initial interpretation to PAPI if enabled
        if self.papi_upload_enabled:
            if self.papi_uploader and interpretation_data:
                self._upload_interpretation(well_data, interpretation_data, current_md)
            else:
                raise ("interpretation_data and papi_uploader must be set")


        logger.info(f"Well setup completed: {well_name}")

        # Return setup data for use in processing modes
        return {
            'prepared_well_data': prepared_well_data,
            'current_md': current_md,
            'max_md': max_md,
            'executor': executor,
            'interpretation_data': interpretation_data,
            'visualizer': visualizer
        }
    
    def _process_emulator_iterations(
        self,
        prepared_well_data: Dict[str, Any],
        well_name: str,
        current_md: float,
        max_md: float,
        executor,
        interpretation_data: Dict[str, Any],
        visualizer: Optional[InteractivePlot]
    ):
        """Process emulator mode iterations with MD steps"""
        step_count = 1
        
        while current_md < max_md:
            current_md += self.md_step
            if current_md > max_md:
                current_md = max_md
            
            step_count += 1
            logger.info(f"Step {step_count}: MD={current_md:.1f}")
            
            # Slice data up to current MD
            sliced_data = self.slicer.slice_well_data(prepared_well_data, current_md)
            
            # Replace interpretation with result from previous step
            if interpretation_data:
                sliced_data['interpretation'] = interpretation_data['interpretation']
            
            # Send update to daemon
            result = executor.update_well_data(sliced_data)
            
            # Get interpretation from result
            new_interpretation_data = executor.get_interpretation_from_result(result)
            
            # Update interpretation_data for next iteration
            interpretation_data = new_interpretation_data
            
            # Update visualization if available
            if visualizer and interpretation_data:
                visualizer.update_interpretation(interpretation_data, current_md)
            
            # Upload to PAPI if enabled
            if self.papi_upload_enabled and self.papi_uploader and interpretation_data:
                self._upload_interpretation(prepared_well_data, interpretation_data, current_md)
            
            # Save step result
            self.processor.save_step_result(
                well_name,
                current_md,
                result,
                "step",
                interpretation_data
            )
            
            # Log interpretation status
            if interpretation_data:
                segments_count = len(interpretation_data['interpretation']['segments'])
                logger.info(f"Step {step_count}: {segments_count} segments")
        
        logger.info(f"Completed {step_count} steps for well {well_name}")

    def _send_cut_flag_and_wait(self):
        """Send CUT flag and wait for response with intelligent timing"""
        import time
        import random

        logger.info("🚦 Phase 1: Sending CUT flag...")

        # Send CUT command
        with open('drilling_control.flag', 'w') as f:
            f.write('CUT')

        # Wait for response flag with timeout
        max_wait = 120
        check_interval = 2
        waited = 0

        logger.info("⏳ Waiting for CUT completion (checking for CUT_DONE)...")

        while waited < max_wait:
            time.sleep(check_interval)
            waited += check_interval

            try:
                with open('drilling_control.flag', 'r', encoding='utf-8') as f:
                    current_flag = f.read().strip()

                if current_flag == 'CUT_DONE':
                    logger.info(f"✅ CUT completed after {waited}s")
                    # Add small random delay (0-5s) to avoid race conditions
                    random_delay = random.uniform(0, 5)
                    time.sleep(random_delay)
                    return

            except FileNotFoundError:
                pass

        logger.warning(f"⚠️ CUT timeout after {max_wait}s, proceeding anyway")

    def _send_process_flag_and_wait(self):
        """Send PROCESS flag and wait for streaming confirmation"""
        import time

        logger.info("🚦 Phase 2: Sending PROCESS flag...")

        # Send PROCESS command
        with open('drilling_control.flag', 'w') as f:
            f.write('PROCESS')

        # Wait for streaming to start
        max_wait = 30
        check_interval = 1
        waited = 0

        logger.info("⏳ Waiting for streaming start (checking for PROCESSING)...")

        while waited < max_wait:
            time.sleep(check_interval)
            waited += check_interval

            try:
                with open('drilling_control.flag', 'r', encoding='utf-8') as f:
                    current_flag = f.read().strip()

                if current_flag == 'PROCESSING':
                    logger.info(f"✅ Streaming started after {waited}s")
                    # Wait 5s for data to be guaranteed in SOLO
                    logger.info("⏳ Waiting 5s for data upload to SOLO...")
                    time.sleep(5)
                    return

            except FileNotFoundError:
                pass

        logger.warning(f"⚠️ PROCESSING timeout after {max_wait}s, proceeding anyway")
    
    def _get_last_papi_output_path(self) -> str:
        """Get the last PAPI output path from _detect_new_papi_data()"""
        return self._last_papi_output_path

    def _detect_new_papi_data(self) -> bool:
        """Detect new data from PAPI or StarSteer for nightguard mode"""
        logger.debug("Checking for new data...")

        # Check data source mode
        data_source = os.getenv('NIGHTGUARD_DATA_SOURCE', 'papi').lower()

        if data_source == 'starsteer':
            # StarSteer mode: read JSON from {STARSTEER_DIR}/{WELLS_DATA_SUBDIR}/{well_name}.json
            starsteer_dir = Path(os.getenv('STARSTEER_DIR', ''))
            wells_data_subdir = os.getenv('WELLS_DATA_SUBDIR', 'AG_DATA/InitialData')

            if not starsteer_dir:
                logger.error("STARSTEER_DIR not set in .env for StarSteer mode")
                return False

            # Read well name from status.json
            status_json_path = starsteer_dir / 'status.json'
            if not status_json_path.exists():
                logger.error(f"StarSteer status.json not found: {status_json_path}")
                return False

            try:
                with open(status_json_path, 'r', encoding='utf-8') as f:
                    status_data = json.load(f)
                well_name = status_data.get('application_state', {}).get('well', {}).get('name', '')
                if not well_name:
                    logger.error(f"Well name not found in status.json: {status_json_path}")
                    return False
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in status.json: {e}")
                return False

            wells_dir = starsteer_dir / wells_data_subdir
            output_path = wells_dir / f"{well_name}.json"

            if not output_path.exists():
                logger.error(f"StarSteer JSON file not found: {output_path}")
                return False

            logger.debug(f"StarSteer mode: reading from {output_path}")
            self._last_papi_output_path = output_path

        else:
            # PAPI mode: use shared PAPI Loader
            logger.debug("PAPI mode: fetching data via PAPI API")
            shared_papi_loader = self._get_papi_loader()

            # Get interpretation name from .env for THIS specific call (allows dynamic changes)
            interpretation_name = os.getenv('PAPI_SOURCE_INTERPRETATION_NAME', '')

            # Handle interpretation errors gracefully for Night Guard resilience
            try:
                output_path = shared_papi_loader.run(interpretation_name=interpretation_name)
                self._last_papi_output_path = output_path  # Store for reuse
            except ValueError as e:
                logger.error(f"Interpretation error in _detect_new_papi_data(): {e}")
                logger.warning("Skipping this cycle - Night Guard will continue monitoring")

                # Night Guard checkpoint
                if self.ng_logger:
                    self.ng_logger.log_checkpoint("interpretation_error_detected", {
                        "error": str(e),
                        "interpretation_name": interpretation_name or "starred",
                        "action": "skip_cycle"
                    })

                # Don't update _last_papi_output_path - keep previous valid data
                # Return False to indicate no new data (continue to next cycle)
                return False

        # Read JSON data from file (same for both modes)
        with open(output_path, 'r', encoding='utf-8') as f:
            well_data = json.load(f)

        # Override AG parameters from .env if in StarSteer mode
        if data_source == 'starsteer':
            ag_params = well_data.get('autoGeosteeringParameters', {})

            # Override lookBackDistance (StarSteer hardcodes it to 304.8m = 1000ft)
            lookback = os.getenv('LOOKBACK_DISTANCE')
            if lookback:
                ag_params['lookBackDistance'] = float(lookback)
                logger.info(f"Override lookBackDistance: {lookback} (from .env)")

            # Override smoothness (horizon smoothness for Python processing)
            smoothness = os.getenv('SMOOTHNESS')
            if smoothness:
                ag_params['smoothness'] = float(smoothness)
                logger.info(f"Override smoothness: {smoothness} (from .env)")

        # Save well_data with overrides for later use (avoid re-reading JSON)
        self._last_well_data = well_data

        # Extract current MD values
        trajectory_points = well_data['well']['points']
        current_trajectory_md = max(p['measuredDepth'] for p in trajectory_points) if trajectory_points else 0.0

        welllog_points = well_data['wellLog']['points']
        current_log_md = max(p['measuredDepth'] for p in welllog_points) if welllog_points else 0.0
        
        # Initialize on first call
        if self.last_trajectory_md is None:
            self.last_trajectory_md = current_trajectory_md
            self.last_log_md = current_log_md
            logger.info(f"Initial MD values: trajectory={current_trajectory_md:.1f}, log={current_log_md:.1f}")
            return False
        
        # Check for changes (tolerance 0.1)
        trajectory_new = (current_trajectory_md - self.last_trajectory_md) > 0.1
        log_new = (current_log_md - self.last_log_md) > 0.1
        
        if trajectory_new or log_new:
            logger.info(f"New data detected: trajectory {self.last_trajectory_md:.1f} -> {current_trajectory_md:.1f}, "
                       f"log {self.last_log_md:.1f} -> {current_log_md:.1f}")

            # Night Guard checkpoint: new data detected
            if self.ng_logger:
                self.ng_logger.log_checkpoint("new_data_detected", {
                    "trajectory_old": self.last_trajectory_md,
                    "trajectory_new": current_trajectory_md,
                    "log_old": self.last_log_md,
                    "log_new": current_log_md,
                    "trajectory_new_flag": trajectory_new,
                    "log_new_flag": log_new
                })

            self.last_trajectory_md = current_trajectory_md
            self.last_log_md = current_log_md
            return True
        
        logger.debug(f"No new data: trajectory={current_trajectory_md:.1f}, log={current_log_md:.1f}")
        return False
    
    def run_continuous_processing(self):
        """Main entry point supporting both modes"""
        if self.mode == 'nightguard':
            if self.ng_logger:
                self.ng_logger.log_checkpoint("starting_nightguard_mode")
            self.process_single_well_watcher()
        else:
            self.process_single_well_self_driven()
    
    def process_single_well_watcher(self):
        """Process single well in watcher mode - wait for external data"""
        logger.info("Starting Night Guard monitoring mode")
        logger.info(f"Monitoring well: {self.nightguard_well_name}")
        logger.info(f"Polling interval: {self.polling_interval}s")

        # INITIAL SETUP: Get first data and do full preparation
        logger.info("Getting initial data for preparation...")

        # Use detect method for first data load (initializes MD tracking and applies overrides)
        # First call returns False (initializes MD tracking) but still loads _last_well_data
        self._detect_new_papi_data()

        # Verify data was loaded
        if not hasattr(self, '_last_well_data') or self._last_well_data is None:
            raise RuntimeError("Failed to load initial well data - check StarSteer/PAPI configuration")

        logger.info("Initial data loaded successfully")

        # Use cached well_data from _detect_new_papi_data (already has overrides applied!)
        initial_well_data = self._last_well_data

        well_name = initial_well_data['wellName']

        # FULL PREPARATION (once)
        logger.info("Performing initial well preparation...")
        setup_data = self.setup_well_processing(initial_well_data, well_name)

        # Nightguard flag initialization - Phase 2: PROCESS
        wait_for_process = os.getenv('WAIT_FOR_PROCESS_FLAG')
        if wait_for_process is None:
            raise ValueError("WAIT_FOR_PROCESS_FLAG not set in .env - must be 'true' or 'false'")

        if wait_for_process.lower() == 'true':
            self._send_process_flag_and_wait()
        else:
            logger.info("Skipping PROCESS flag coordination (WAIT_FOR_PROCESS_FLAG=false)")

        logger.info("🎯 All systems ready, starting monitoring...")
        logger.info(f"Auto-exit timeout: {self.auto_exit_timeout_minutes} minutes")

        # Initialize timeout tracking
        last_data_time = datetime.now()

        # INCREMENTAL MONITORING LOOP
        while True:
            # Check for new data
            if self._detect_new_papi_data():
                logger.info("Processing new PAPI data...")
                # Update last data time
                last_data_time = datetime.now()

                # Use cached well_data from _detect_new_papi_data (already has overrides applied!)
                updated_well_data = self._last_well_data

                # Keep this for error logging context
                json_file_path = self._get_last_papi_output_path()

                # Validate JSON structure (skip JSONDecodeError since we already parsed it)
                try:
                    # Basic validation that required fields exist
                    if 'wellName' not in updated_well_data:
                        raise ValueError("Missing required field: wellName")
                except (ValueError, KeyError) as e:
                    # Full traceback logging
                    logger.error("JSON decode error detected!")
                    logger.error(f"File: {json_file_path}")
                    logger.error(f"Error: {e}")
                    logger.error(f"Full traceback:\n{traceback.format_exc()}")

                    # Create fault directory if it doesn't exist
                    fault_dir = Path("fault_json_files")
                    fault_dir.mkdir(exist_ok=True)

                    # Copy fault file with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fault_file = fault_dir / f"fault_{timestamp}_{Path(json_file_path).name}"
                    shutil.copy2(json_file_path, fault_file)
                    logger.error(f"Corrupted file copied to: {fault_file}")

                    # Re-raise to stop processing
                    raise

                # Save debug JSON if enabled
                if self.debug_save_json:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    debug_file = f"debug_well_data_{timestamp}.json"
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        json.dump(updated_well_data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Debug: saved well data to {debug_file}")

                # Incremental processing of new data using existing executor
                # Apply cached normalization to updated data
                if self.processor.python_normalization_enabled:
                    updated_well_data = self.processor._apply_normalization(updated_well_data)

                # Update executor with new data (preserves interpretation state)
                result = setup_data['executor'].update_well_data(updated_well_data)

                # Get updated interpretation
                interpretation_data = setup_data['executor'].get_interpretation_from_result(result)

                # Export interpretation to StarSteer (atomic copy for nginterpretation.json)
                lateral_last_md = updated_well_data['lateralWellLastMD']
                logger.info(f"TRIM_DEBUG: lateralWellLastMD={lateral_last_md}")
                self._export_interpretation_to_starsteer(well_name, setup_data['executor'], lateral_last_md)

                # Get current max MD for saving
                current_max_md = self.processor._get_max_md(updated_well_data)

                # Night Guard checkpoint: data processed
                if self.ng_logger:
                    segments = interpretation_data['interpretation']['segments']
                    has_interpretation = len(segments) > 0
                    self.ng_logger.log_checkpoint("data_processed", {
                        "max_md": current_max_md,
                        "has_interpretation": has_interpretation,
                        "segments_count": len(segments)
                    })

                # Save current result
                self.processor.save_step_result(
                    well_name,
                    current_max_md,
                    result,
                    "current",
                    interpretation_data
                )

                # Upload to PAPI if enabled
                if self.papi_upload_enabled and self.papi_uploader and interpretation_data:
                    self._upload_interpretation(updated_well_data, interpretation_data, current_max_md)

                # NEW: Night Guard Alert Analysis (only if enabled and in nightguard mode)
                if self.alert_analyzer_enabled:
                    if interpretation_data:
                        if self.ng_logger:
                            self.ng_logger.log_checkpoint("night_guard_triggered", {
                                "well_name": well_name,
                                "max_md": current_max_md,
                                "alert_analyzer_enabled": True
                            })
                        self._run_night_guard_analysis(
                            well_name=well_name,
                            updated_well_data=updated_well_data,
                            interpretation_data=interpretation_data
                        )
                    else:
                        if self.ng_logger:
                            self.ng_logger.log_checkpoint("night_guard_skipped_no_interpretation", {
                                "well_name": well_name,
                                "max_md": current_max_md,
                                "reason": "no_interpretation_data"
                            })
                else:
                    if self.ng_logger:
                        self.ng_logger.log_checkpoint("night_guard_disabled", {
                            "alert_analyzer_enabled": self.alert_analyzer_enabled
                        })

                logger.info(f"Processed incremental data up to MD={current_max_md:.1f}")
            
            # Check for end flag
            if self._check_end_flag() and False:
                logger.info("END_REACHED detected - shutting down Night Guard")
                break

            # Check timeout - exit if no new data for configured period
            time_since_last_data = (datetime.now() - last_data_time).total_seconds() / 60
            if time_since_last_data > self.auto_exit_timeout_minutes:
                logger.info(f"Auto-exit timeout reached: {time_since_last_data:.1f} minutes since last data")
                logger.info(f"Configured timeout: {self.auto_exit_timeout_minutes} minutes")
                if self.ng_logger:
                    self.ng_logger.log_checkpoint("auto_exit_timeout", {
                        "timeout_minutes": self.auto_exit_timeout_minutes,
                        "time_since_last_data_minutes": time_since_last_data
                    })
                break

            # Sleep before next check
            logger.debug(f"Sleeping for {self.polling_interval} seconds...")
            time.sleep(self.polling_interval)

        # Cleanup
        if setup_data and setup_data['visualizer']:
            setup_data['visualizer'].close()
        if setup_data and setup_data['executor']:
            setup_data['executor'].stop_daemon()

        logger.info("Night Guard monitoring stopped")
    
    def process_single_well_self_driven(self):
        """Process wells in self-driven mode - iterate through files with MD steps"""
        logger.info("Starting emulator mode with file processing")
        logger.info(f"Monitoring directory: {self.wells_dir}")
        
        timeout_minutes = 20
        sleep_seconds = 10
        
        while True:
            # Load current state
            state = load_processed_wells_state(self.wells_dir)
            
            # Scan directory for JSON files
            current_files = scan_results_directory(self.wells_dir)
            
            # Sync state with current files
            state = sync_wells_state(current_files, state)
            
            # Check for timeouts
            check_timeouts(state, timeout_minutes)
            
            # Save updated state
            save_processed_wells_state(state, self.wells_dir)
            
            # Print summary
            print_state_summary(self.wells_dir)
            
            # Find next pending well
            next_well = get_next_pending_well(state)
            
            if next_well:
                filename = next_well['filename']
                
                # Mark as in_progress
                update_well_status(state, filename, 'in_progress')
                save_processed_wells_state(state, self.wells_dir)
                
                # Process the well
                file_path = Path(self.wells_dir) / filename
                well_name = self._extract_well_name_from_file(file_path)
                
                # Clean old results if continue mode
                if self.continue_mode:
                    self._clean_well_results(well_name)
                
                # Load and process well
                well_data = self._load_well_data(str(file_path))
                
                if well_data:
                    # Setup well processing
                    setup_data = self.setup_well_processing(well_data, well_name)

                    # Self-driven iterations with MD steps
                    self._process_emulator_iterations(
                        setup_data['prepared_well_data'],
                        well_name,
                        setup_data['current_md'],
                        setup_data['max_md'],
                        setup_data['executor'],
                        setup_data['interpretation_data'],
                        setup_data['visualizer']
                    )

                    # Cleanup
                    if setup_data and setup_data['visualizer']:
                        setup_data['visualizer'].close()
                    if setup_data and setup_data['executor']:
                        setup_data['executor'].stop_daemon()

                    update_well_status(state, filename, 'completed')
                    logger.info(f"✅ Well {filename} completed successfully")
                else:
                    update_well_status(state, filename, 'error', "Failed to load well data")
                    logger.error(f"❌ Well {filename} failed to load")
                
                # Save final state
                save_processed_wells_state(state, self.wells_dir)
                
            else:
                logger.info("No pending wells found, waiting for new files...")
            
            # Sleep before next cycle
            time.sleep(sleep_seconds)
    
    def _check_end_flag(self) -> bool:
        """Check for END_REACHED flag file"""
        flag_file = Path('drilling_control.flag')
        if flag_file.exists():
            with open(flag_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                return content == 'END_REACHED'
        return False
    
    def _extract_well_name_from_file(self, well_file: Path) -> str:
        """Extract well name from well file"""
        with open(well_file, 'r', encoding='utf-8') as f:
            well_data = json.load(f)
        return well_data['wellName']
    
    def _load_well_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load well data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _clean_well_results(self, well_name: str):
        """Remove all result files for specific well"""
        results_path = Path(self.results_dir)
        if not results_path.exists():
            return
        
        # Remove all files for this well
        pattern_files = list(results_path.glob(f"{well_name}_*.json"))
        removed_count = 0
        for file_path in pattern_files:
            file_path.unlink()
            removed_count += 1
        
        if removed_count > 0:
            logger.info(f"Cleaned {removed_count} old result files for well {well_name}")
    
    def _cleanup_interpretations_for_well(self, well_name: str):
        """Clean up all interpretations for well once at the start"""
        if not self.papi_upload_enabled or not self.papi_uploader:
            return
        
        self.papi_uploader._ensure_connection()
        
        # Find project and well
        project = self.papi_uploader.api.get_project_by_name(self.papi_uploader.project_name)
        if not project:
            logger.warning(f"Project '{self.papi_uploader.project_name}' not found for cleanup")
            return
        
        well = self.papi_uploader.api.get_well_by_name(project['uuid'], well_name)
        if not well:
            logger.warning(f"Well '{well_name}' not found for cleanup")
            return
        
        # Delete all interpretations with base name prefix
        deleted_count = self.papi_uploader.api.delete_interpretations_by_prefix(
            well['uuid'],
            self.papi_uploader.base_interpretation_name
        )
        logger.info(f"Pre-cleaned {deleted_count} interpretations for well {well_name}")
    
    def _should_upload_now(self) -> bool:
        """Check if enough time has passed for next PAPI upload (10-minute batching)"""
        if self._last_upload_time is None:
            return True  # First upload

        current_time = time.time()
        time_since_last_upload = current_time - self._last_upload_time
        return time_since_last_upload >= 600  # 10 minutes = 600 seconds

    def _upload_interpretation(
        self,
        well_data: Dict[str, Any],
        interpretation_data: Dict[str, Any],
        current_md: float
    ):
        """Upload interpretation to PAPI with 10-minute batching"""
        if not self._should_upload_now():
            logger.debug(f"PAPI upload skipped - batching (last upload: {time.time() - self._last_upload_time:.1f}s ago)")
            return

        logger.info("Uploading interpretation to PAPI...")
        self.papi_uploader.upload_interpretation(well_data, interpretation_data, current_md)
        self._last_upload_time = time.time()
        logger.info("PAPI upload completed, next upload in 10 minutes")
    
    def _create_ag_visualizer(
        self,
        well_data: Dict[str, Any],
        well_name: str,
        original_start_md: float,
        detected_start_md: float
    ) -> Optional[InteractivePlot]:
        """Create AG visualizer from well data"""
        if not self.interactive_mode:
            logger.info("Running in headless mode - skipping visualization")
            return None
        
        logger.info("Creating AG visualizer from well data")
        
        # Create AG objects from JSON
        well = Well(well_data)
        typewell = TypeWell(well_data)
        tvd_to_typewell_shift = well_data['tvdTypewellShift']
        
        # Normalization
        max_curve_value = max(well.max_curve, typewell.value.max())
        well_normalized = deepcopy(well)
        typewell_normalized = deepcopy(typewell)
        
        well_normalized.normalize(max_curve_value, typewell.min_depth)
        typewell_normalized.normalize(max_curve_value, well.min_depth, well.md_range)
        
        # Denormalized copies
        well_denorm = well
        typewell_denorm = typewell
        
        # Extract manual interpretation from JSON
        manual_interpretation_json = well_data.get('interpretation', {}).get('segments', [])
        manual_segments_denorm = None
        manual_segments_norm = None
        well_manual_interpretation_denorm = None
        well_manual_interpretation_norm = None
        
        if manual_interpretation_json:
            from ag_objects.ag_obj_interpretation import normalize_segments
            
            # For manual interpretation use end of well as endMd
            max_md = well_denorm.measured_depth[-1]
            
            # Create denormalized segments
            manual_segments_denorm = create_segments_from_json(manual_interpretation_json, well_denorm, max_md)
            
            # Create normalized segments
            manual_segments_norm = normalize_segments(manual_segments_denorm, well_denorm.md_range)
            
            # Create well objects with manual interpretation
            well_manual_interpretation_denorm = deepcopy(well_denorm)
            well_manual_interpretation_norm = deepcopy(well_normalized)
            
            # Calculate projection for manual interpretation
            well_manual_interpretation_denorm.calc_horizontal_projection(
                typewell_denorm,
                manual_segments_denorm,
                tvd_to_typewell_shift
            )
            
            well_manual_interpretation_norm.calc_horizontal_projection(
                typewell_normalized,
                manual_segments_norm,
                tvd_to_typewell_shift / well.md_range
            )
            
            logger.info(f"Loaded manual interpretation with {len(manual_segments_denorm)} segments")
        
        # Empty segments for start
        empty_segments = []
        
        # Parameters for InteractivePlot
        well_calc_params = {
            'well_name': well_name,
            'pearson_power': 2.0,
            'mse_power': 0.001,
            'num_intervals_sc': 20,
            'sc_power': 1.15,
            'min_pearson_value': -1
        }
        
        # Create InteractivePlot
        visualizer = InteractivePlot(
            well_denorm=well_denorm,
            well=well_normalized,
            segments=empty_segments,
            type_well_denorm=typewell_denorm,
            type_well=typewell_normalized,
            well_manual_interpretation_denorm=well_manual_interpretation_denorm,
            well_manual_interpretation=well_manual_interpretation_norm,
            tvd_to_typewell_shift=tvd_to_typewell_shift,
            landing_end_md_manual=original_start_md,
            landing_end_md_auto=detected_start_md,
            well_calc_params=well_calc_params
        )
        
        # Show plot initially
        visualizer.show_initial()
        
        logger.info("AG visualizer created successfully")
        return visualizer

    def _validate_reference_interpretation_initial(self, well_data: Dict[str, Any], well_name: str):
        """
        Validate StarSteer reference interpretation at initial load.

        Fails fast with clear error message if interpretation is corrupted.

        Args:
            well_data: Initial well data from StarSteer
            well_name: Well name for error reporting

        Raises:
            ValueError: If interpretation contains NaN or extreme values
        """
        if 'interpretation' not in well_data:
            logger.warning(f"Well {well_name}: No reference interpretation in initial data")
            return

        segments = well_data['interpretation'].get('segments', [])
        if not segments:
            logger.warning(f"Well {well_name}: Empty segments in reference interpretation")
            return

        # Check ALL segments for extreme values
        import math
        extreme_segments = []
        for i, seg in enumerate(segments):
            end_shift = seg.get('endShift')

            if end_shift is None:
                raise ValueError(
                    f"Well {well_name}: Segment {i} has None endShift!\n"
                    f"Segment: {seg}\n"
                    f"StarSteer failed to create valid interpretation."
                )

            if not math.isfinite(end_shift) or abs(end_shift) > 1e100:
                extreme_segments.append((i, end_shift, seg))

        if extreme_segments:
            error_msg = f"Well {well_name}: Reference interpretation contains EXTREME/NaN values!\n\n"
            for i, end_shift, seg in extreme_segments:
                error_msg += f"Segment {i}: endShift={end_shift:.6e}\n"
                error_msg += f"  startMd={seg.get('startMd')}, startShift={seg.get('startShift')}\n"

            error_msg += "\nRoot cause: StarSteer extrapolation bug (logEndMD=DBL_MAX).\n"
            error_msg += "Check StarSteer logs for 'EXTRAPOLATE' and 'logEndMD=1.79769e+308'.\n"
            error_msg += "This happens when log is not found during interpretation creation."

            raise ValueError(error_msg)

        logger.info(f"Well {well_name}: Reference interpretation validated OK ({len(segments)} segments)")

    def _export_interpretation_to_starsteer(self, well_name: str, executor, lateral_last_md: float = None):
        """
        Atomically copy interpretation file to StarSteer nginterpretation.json

        This allows StarSteer to load the interpretation via QFileSystemWatcher
        (see STARSTEER-22 for details)

        If lateral_last_md is provided, trims interpretation to that MD
        (StarSteer cannot handle interpretation beyond trajectory end)

        Args:
            well_name: Well name for finding source interpretation file
            executor: Executor instance with interpretation_dir path
            lateral_last_md: If provided, trim interpretation to this MD
        """
        try:
            # Get StarSteer dir from config (passed from caller: slicer.py, main.py)
            starsteer_dir = self.config.get('starsteer_dir', '')
            if not starsteer_dir:
                logger.debug("starsteer_dir not in config - skipping interpretation export")
                return

            starsteer_path = Path(starsteer_dir)
            if not starsteer_path.exists():
                logger.warning(f"STARSTEER_DIR does not exist: {starsteer_dir} - skipping export")
                return

            # Get source interpretation file from executor
            source_file = Path(executor.interpretation_dir) / f"{well_name}.json"
            if not source_file.exists():
                logger.warning(f"Source interpretation file not found: {source_file}")
                return

            # Atomic write: write to .tmp then rename
            dest_file = starsteer_path / "interpretation.json"
            tmp_file = starsteer_path / "interpretation.json.tmp"

            # Read interpretation data
            with open(source_file, 'r', encoding='utf-8') as f:
                interp_data = json.load(f)

            # Trim segments to lateral_last_md if provided (StarSteer can't handle beyond trajectory)
            logger.info(f"TRIM_DEBUG: lateral_last_md={lateral_last_md}, has_interpretation={'interpretation' in interp_data}")
            if lateral_last_md is not None and 'interpretation' in interp_data:
                segments = interp_data['interpretation'].get('segments', [])
                if segments:
                    last_seg_md = segments[-1].get('startMd', 0) if segments else 0
                    logger.info(f"TRIM_DEBUG: {len(segments)} segments, last_startMd={last_seg_md}, cutoff={lateral_last_md}")
                    original_count = len(segments)
                    trimmed_segments = []
                    for seg in segments:
                        start_md = seg.get('startMd', 0)
                        if start_md <= lateral_last_md:
                            trimmed_segments.append(seg)
                        else:
                            # Interpolate last segment to end at lateral_last_md
                            if trimmed_segments:
                                prev = trimmed_segments[-1]
                                prev_md = prev.get('startMd', 0)
                                prev_end_shift = prev.get('endShift', 0)
                                curr_start_shift = seg.get('startShift', 0)
                                curr_end_shift = seg.get('endShift', 0)
                                # Linear interpolation for shift at lateral_last_md
                                if start_md != prev_md:
                                    ratio = (lateral_last_md - prev_md) / (start_md - prev_md)
                                    interp_shift = prev_end_shift + ratio * (curr_start_shift - prev_end_shift)
                                    # Update previous segment's endShift
                                    trimmed_segments[-1] = dict(prev)
                                    trimmed_segments[-1]['endShift'] = interp_shift
                                    logger.info(f"TRIM_DEBUG: interpolated endShift={interp_shift:.4f} at cutoff MD")
                            break

                    if len(trimmed_segments) < original_count:
                        logger.info(f"TRIM_DEBUG: Trimmed {original_count} -> {len(trimmed_segments)} segments at MD={lateral_last_md}")
                    else:
                        logger.info(f"TRIM_DEBUG: No trimming needed, all {original_count} segments within MD={lateral_last_md}")
                    interp_data['interpretation']['segments'] = trimmed_segments

            # Write to temporary file
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(interp_data, f, indent=2, ensure_ascii=False)

            # Atomic replace (overwrites existing file atomically, works on Windows and Linux)
            os.replace(tmp_file, dest_file)

            logger.info(f"✅ Interpretation exported to StarSteer: {dest_file}")
            logger.debug(f"Source: {source_file} -> Dest: {dest_file}")

        except Exception as e:
            logger.error(f"Failed to export interpretation to StarSteer: {e}")
            logger.error(f"Traceback:\n{traceback.format_exc()}")

    def _build_starsteer_night_guard_data(
        self,
        well_name: str,
        updated_well_data: Dict[str, Any],
        interpretation_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Build night_guard_data for StarSteer mode from local data

        In StarSteer mode, all data (targetLineTVD, corridorTop, corridorBase)
        is already in well.points[] - no need to fetch from external sources.
        Units are always in meters.
        """
        logger.info("STARSTEER_MODE: Building night_guard_data from local well data")

        # Get current max MD from well data
        well_points = updated_well_data.get('well', {}).get('points', [])
        if not well_points:
            logger.error("STARSTEER_MODE: No well points found in well data")
            return None

        current_max_md = max(point['measuredDepth'] for point in well_points)
        logger.info(f"STARSTEER_MODE: current_max_md = {current_max_md:.1f}")

        # Build minimal night_guard_data - target/corridor data is in well.points
        night_guard_data = {
            'current_max_md': current_max_md,
            'night_guard_ready': True,
            'project_measure_unit': 'METER'  # StarSteer always uses meters
        }

        logger.info("STARSTEER_MODE: night_guard_data built successfully")
        return night_guard_data

    def _run_night_guard_analysis(
        self,
        well_name: str,
        updated_well_data: Dict[str, Any],
        interpretation_data: Dict[str, Any]
    ):
        """Run Night Guard alert analysis - supports both PAPI and StarSteer modes"""

        logger.debug(f"Running Night Guard analysis for {well_name}")

        if self.ng_logger:
            self.ng_logger.log_checkpoint("night_guard_analysis_start", {"well_name": well_name})

        # Check data source mode
        import os
        data_source = os.getenv('NIGHTGUARD_DATA_SOURCE', 'papi').lower()
        logger.info(f"NIGHT_GUARD_DEBUG: Data source mode: {data_source}")

        # Get well UUID
        well_uuid = updated_well_data['well']['uuid']

        # Branch based on data source
        if data_source == 'starsteer':
            # ========== STARSTEER MODE: Build night_guard_data locally ==========
            logger.info("NIGHT_GUARD_DEBUG: Using StarSteer mode - building night_guard_data locally")

            night_guard_data = self._build_starsteer_night_guard_data(
                well_name, updated_well_data, interpretation_data
            )

            if not night_guard_data:
                logger.error("NIGHT_GUARD_DEBUG: Failed to build StarSteer night_guard_data")
                return

        else:
            # ========== PAPI MODE: Use PAPI Loader ==========
            logger.info("NIGHT_GUARD_DEBUG: Using PAPI mode - fetching data from PAPI Loader")

            # Get enhanced Night Guard data from PAPI Loader
            # NIGHT GUARD FIX: Use shared PAPI loader instance with cached interpretation
            shared_papi_loader = self._get_papi_loader()
            if not shared_papi_loader:
                logger.error("NIGHT_GUARD_DEBUG: Failed to get shared PAPI loader for analysis")
                return

            # NIGHT GUARD FIX: Validate cache integrity before analysis
            if not self._validate_papi_loader_cache():
                logger.error("NIGHT_GUARD_DEBUG: Cache validation FAILED before analysis - interpretation data not available")
                if self.ng_logger:
                    self.ng_logger.log_checkpoint("night_guard_cache_validation_failed", {
                        "well_name": well_name,
                        "cache_available": False,
                        "reason": "interpretation_data_missing"
                    })
                return

            instance_id = id(shared_papi_loader)
            logger.info(f"NIGHT_GUARD_DEBUG: Using SHARED PAPI Loader for analysis (ID: {instance_id}) - Cache validated")

            # Update PAPI Loader cache with current interpretation data for Night Guard
            # IMPORTANT: Preserve 'horizon_tvdss' (MD→VS lookup table) from initial load, only update segments
            if interpretation_data:
                if shared_papi_loader._last_interpretation_data and 'horizon_tvdss' in shared_papi_loader._last_interpretation_data:
                    # Save MD→VS lookup table from first load
                    md_vs_lookup = shared_papi_loader._last_interpretation_data['horizon_tvdss']
                    logger.info(f"NIGHT_GUARD_DEBUG: Preserving MD→VS lookup table with {len(md_vs_lookup)} points from initial load")

                    # Update cache with new interpretation data
                    shared_papi_loader._last_interpretation_data = interpretation_data

                    # Restore MD→VS lookup table
                    shared_papi_loader._last_interpretation_data['horizon_tvdss'] = md_vs_lookup
                    segments_count = len(interpretation_data['interpretation']['segments'])
                    logger.info(f"NIGHT_GUARD_DEBUG: Updated cache with {segments_count} new segments, preserved MD→VS table")
                else:
                    # First update or no MD→VS table available
                    shared_papi_loader._last_interpretation_data = interpretation_data
                    segments_count = len(interpretation_data['interpretation']['segments'])
                    logger.info(f"NIGHT_GUARD_DEBUG: Updated cache with interpretation ({segments_count} segments)")
            else:
                logger.warning(f"NIGHT_GUARD_DEBUG: No interpretation_data to update in PAPI Loader cache")

            night_guard_data = shared_papi_loader.get_night_guard_data(well_uuid, updated_well_data, self.ng_logger)

        # Initialize StarSteer and InterpretationUploader if project_measure_unit is available
        if night_guard_data and 'project_measure_unit' in night_guard_data:
            project_measure_unit = night_guard_data['project_measure_unit']

            # Initialize StarSteer if not yet initialized
            if self.starsteer_interface is None and hasattr(self, '_starsteer_config'):
                from alerts.starsteer_target_line_interface import StarSteerTargetLineInterface
                self._starsteer_config['PROJECT_MEASURE_UNIT'] = project_measure_unit
                self.starsteer_interface = StarSteerTargetLineInterface(self._starsteer_config, self.ng_logger)
                logger.info(f"StarSteer Target Line Interface initialized with project units: {project_measure_unit}")
            elif self.starsteer_interface:
                # Update units if already initialized
                self.starsteer_interface.project_measure_unit = project_measure_unit
                logger.debug(f"Updated StarSteer units: {project_measure_unit}")

            # Initialize InterpretationUploader if enabled and not yet initialized
            if self.papi_upload_enabled and self.papi_uploader is None:
                from papi_export.uploaders.interpretation_uploader import InterpretationUploader
                self.papi_uploader = InterpretationUploader(project_measure_unit=project_measure_unit)
                logger.info(f"InterpretationUploader initialized with project units: {project_measure_unit}")

        if not night_guard_data or not night_guard_data['night_guard_ready']:
            # Создаем алерт вместо пропуска анализа
            current_md = night_guard_data.get('current_max_md') if night_guard_data else None
            self._generate_data_not_ready_alert(well_name, night_guard_data, current_md)
            return

        # Extract data for alert analysis
        current_md = night_guard_data['current_max_md']

        # Check data source mode
        import os
        data_source = os.getenv('NIGHTGUARD_DATA_SOURCE', 'papi').lower()

        # Get target TVD based on data source
        target_tvd = None
        vs_coordinate = None  # For PAPI mode

        if data_source == 'starsteer':
            # ========== STARSTEER MODE: Get target from well.points ==========
            logger.info("STARSTEER_MODE: Getting target TVD from well.points")

            # Find point at current_md (or closest)
            well_points = updated_well_data.get('well', {}).get('points', [])
            if not well_points:
                logger.error("STARSTEER_MODE: No well points found")
                return

            # Find the point with MD closest to current_md
            point_at_md = min(well_points, key=lambda p: abs(p['measuredDepth'] - current_md))
            md_diff = abs(point_at_md['measuredDepth'] - current_md)

            if md_diff > 1.0:  # More than 1 meter difference
                logger.warning(f"STARSTEER_MODE: Closest point is {md_diff:.2f}m away from current MD")

            target_tvd = point_at_md.get('targetLineTVD')
            if target_tvd is None:
                logger.error(f"STARSTEER_MODE: targetLineTVD is null at MD={current_md:.1f}")
                return

            logger.info(f"STARSTEER_MODE: target_tvd = {target_tvd:.3f} at MD={point_at_md['measuredDepth']:.1f}")

            # Initialize target_line_info (optional metadata for alerts)
            target_line_info = None  # StarSteer mode - target data already in well.points

        else:
            # ========== PAPI MODE: Use PAPI target line data ==========
            vs_coordinate = night_guard_data.get('vs_coordinate')
            target_line_info = night_guard_data.get('target_line_data')

            target_line_source = os.getenv('TARGET_LINE_SOURCE', 'papi').lower()

            if target_line_source == 'papi':
                # Use PAPI target TVD from night_guard_data
                target_tvd = night_guard_data.get('target_tvd_at_end')
                if target_tvd:
                    logger.info(f"PAPI mode: target_tvd = {target_tvd:.3f} @ VS={vs_coordinate:.2f}")
                else:
                    logger.error(f"PAPI mode enabled but no target TVD available from PAPI")
            else:
                logger.error(f"Unknown TARGET_LINE_SOURCE in PAPI mode: {target_line_source}")

        # Check if we have valid target TVD for analysis
        if target_tvd is None:
            logger.error(f"Cannot perform alert analysis - no valid target TVD available")
            if self.ng_logger:
                self.ng_logger.log_checkpoint("night_guard_target_tvd_failed", {
                    "well_name": well_name,
                    "current_md": current_md,
                    "vs_coordinate": vs_coordinate,
                    "reason": "both_starsteer_and_papi_failed"
                })
            return

        if self.ng_logger:
            self.ng_logger.log_checkpoint("night_guard_data_ready", {
                "well_name": well_name,
                "current_md": current_md,
                "target_tvd": target_tvd,
                "vs_coordinate": vs_coordinate,
                "has_target_line": target_line_info is not None,
                "target_tvd_source": "starsteer" if self.starsteer_interface else "papi"
            })

        # Perform alert analysis
        project_measure_unit = night_guard_data.get('project_measure_unit')
        if not project_measure_unit:
            logger.error("CRITICAL: project_measure_unit missing from night_guard_data!")
            raise ValueError("project_measure_unit is required for alert analysis")

        # For StarSteer mode: skip deviation alert (use horizon breach only)
        # For PAPI mode: use deviation alert
        alert_data = None
        if data_source == 'starsteer':
            logger.info("STARSTEER_MODE: Skipping deviation alert (using horizon breach alerts only)")
        else:
            # PAPI mode - perform deviation analysis
            alert_data = self.alert_analyzer.analyze_deviation(
                interpretation_data=interpretation_data,
                current_md=current_md,
                target_tvd=target_tvd,
                vs_coordinate=vs_coordinate,
                well_name=well_name,
                target_line_info=target_line_info,
                project_measure_unit=project_measure_unit
            )

            if alert_data and alert_data['alert']:
                logger.warning(f"🚨 Night Guard ALERT triggered for {well_name} at MD={current_md:.1f}")
                if self.ng_logger:
                    self.ng_logger.log_alert(well_name, current_md, alert_data.get('deviation', 0), True)
            else:
                logger.debug(f"✅ Night Guard: {well_name} within acceptable limits at MD={current_md:.1f}")
                if self.ng_logger:
                    self.ng_logger.log_alert(well_name, current_md, alert_data.get('deviation', 0) if alert_data else 0, False)

        # Check for horizon breach alerts (StarSteer mode with horizons)
        if updated_well_data.get('hasStarredHorizons') and updated_well_data.get('hasTargetLine'):
            logger.debug("Checking horizon breach alerts...")

            # Find last valid point with targetLineTVD
            well_points = updated_well_data.get('well', {}).get('points', [])
            last_valid_point = None
            for point in reversed(well_points):
                if point.get('targetLineTVD') is not None:
                    last_valid_point = point
                    break

            if last_valid_point:
                # Get interpretation segments and tvd_typewell_shift for horizon shift calculation
                interpretation_segments = interpretation_data['interpretation']['segments']
                tvd_typewell_shift = updated_well_data.get('tvdTypewellShift', 0.0)

                horizon_alert = self.alert_analyzer.check_horizon_breach(
                    point_data=last_valid_point,
                    horizons=updated_well_data.get('horizons'),
                    interpretation_segments=interpretation_segments,
                    tvd_typewell_shift=tvd_typewell_shift
                )

                # Send notifications ALWAYS (for testing) - both RED and GREEN alerts
                if horizon_alert:
                    if horizon_alert.get('has_alert'):
                        # RED alert - breach detected
                        logger.warning(f"🚨 HORIZON BREACH ALERT: {horizon_alert['message']}")
                        alert_level = 'HORIZON_BREACH'
                        alert_status = True
                    else:
                        # GREEN alert - corridor within limits
                        logger.info(f"✅ HORIZON OK: {horizon_alert['message']}")
                        alert_level = 'HORIZON_OK'
                        alert_status = False

                    # Log checkpoint for Night Guard
                    if self.ng_logger:
                        self.ng_logger.log_checkpoint("horizon_check", {
                            "has_alert": horizon_alert.get('has_alert'),
                            "breach_type": horizon_alert.get('breach_type'),
                            "md": horizon_alert['details']['md'],
                            "message": horizon_alert['message'],
                            "upper_corridor": horizon_alert['details']['upper_corridor'],
                            "lower_corridor": horizon_alert['details']['lower_corridor'],
                            "top_horizon": horizon_alert['details']['top_horizon'],
                            "bottom_horizon": horizon_alert['details']['bottom_horizon']
                        })

                    # Send notifications via existing notification manager
                    # Format alert_data structure for notification system
                    horizon_alert_data = {
                        'timestamp': alert_data.get('timestamp') if alert_data else None,
                        'well_name': well_name,
                        'measured_depth': horizon_alert['details']['md'],
                        'alert': alert_status,
                        'alert_level': alert_level,
                        'breach_type': horizon_alert.get('breach_type'),
                        'message': horizon_alert['message'],
                        'details': horizon_alert['details'],
                        'project_measure_unit': project_measure_unit
                    }

                    # Send notifications
                    notification_results = self.alert_analyzer.notifier.send_alert(horizon_alert_data)
                    logger.info(f"Horizon notification results: {notification_results}")
                else:
                    logger.debug(f"No horizon alert data available")
            else:
                logger.debug("No valid points with targetLineTVD found for horizon check")
        else:
            logger.debug("Horizon checks skipped (hasStarredHorizons or hasTargetLine not set)")

    def _generate_data_not_ready_alert(self, well_name: str, night_guard_data: dict, current_md: float):
        """Generate alert for situations when Night Guard data is not ready (VS out of range, etc.)"""

        import os
        from datetime import datetime

        # Determine the specific reason why data is not ready
        if not night_guard_data:
            reason = "night_guard_data_missing"
            explanation = "No night guard data available - PAPI connection or data processing error"
            vs_coordinate = None
            target_range = None
        else:
            # Analyze what went wrong based on available data
            vs_coordinate = night_guard_data.get('vs_coordinate')
            target_line_data = night_guard_data.get('target_line_data')

            if vs_coordinate is not None and target_line_data:
                # VS coordinate is outside target line range
                origin_vs = target_line_data.get('origin_vs', 0)
                target_vs = target_line_data.get('target_vs', 0)
                vs_min = min(origin_vs, target_vs)
                vs_max = max(origin_vs, target_vs)
                target_range = f"[{vs_min:.1f}, {vs_max:.1f}]"

                reason = "vs_coordinate_out_of_range"
                explanation = f"VS coordinate {vs_coordinate:.1f}m is outside target line range {target_range}m"
            else:
                reason = "data_processing_error"
                explanation = "Could not determine VS coordinate or target line data"
                target_range = None

        # Create alert data structure
        alert_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "well_name": well_name,
            "measured_depth": current_md if current_md else 0.0,
            "vs_coordinate": vs_coordinate,
            "alert": True,
            "alert_level": "SYSTEM_ERROR",
            "alert_type": "DATA_NOT_READY",
            "reason": reason,
            "explanation": explanation,
            "target_line_range": target_range,
            "interpretation_tvd": None,
            "target_tvd": None,
            "deviation": None,
            "threshold": None
        }

        # Save alert to file
        alerts_dir = os.getenv('ALERTS_DIR', 'alert_data')
        if not os.path.exists(alerts_dir):
            os.makedirs(alerts_dir)

        md_str = f"{current_md:.1f}" if current_md else "unknown"
        alert_filename = f"alert_{well_name}_{md_str}_data_not_ready.json"
        alert_filepath = os.path.join(alerts_dir, alert_filename)

        with open(alert_filepath, 'w', encoding='utf-8') as f:
            json.dump(alert_data, f, indent=2, ensure_ascii=False)

        logger.warning(f"🚨 SYSTEM ALERT: {explanation}")
        logger.info(f"Alert saved to: {alert_filepath}")

        # Log to Night Guard logger
        if self.ng_logger:
            self.ng_logger.log_checkpoint("system_alert_generated", {
                "well_name": well_name,
                "alert_type": "DATA_NOT_READY",
                "reason": reason,
                "vs_coordinate": vs_coordinate,
                "target_range": target_range,
                "alert_file": alert_filename
            })

        # Send SMS notification if enabled
        if hasattr(self, 'sms_notifier') and self.sms_notifier:
            try:
                self.sms_notifier.send_system_alert(alert_data)
            except Exception as e:
                logger.error(f"Failed to send system alert SMS: {e}")


def main():
    """Main entry point for the emulator"""
    logger.info("Starting Enhanced Multi-Drilling Emulator")
    
    # Load configuration from environment
    config = {
        'exe_path': os.getenv('EXE_PATH'),
        'wells_dir': os.getenv('WELLS_DIR'),
        'work_dir': os.getenv('WORK_DIR'),
        'results_dir': os.getenv('RESULTS_DIR'),
        'md_step_meters': float(os.getenv('MD_STEP_METERS', '30')),
        'quality_thresholds_meters': float(os.getenv('QUALITY_THRESHOLDS_METERS', '3.0')),
        'continue_mode': os.getenv('CONTINUE_MODE', 'false').lower() == 'true',
        'landing_detection_enabled': os.getenv('LANDING_DETECTION_ENABLED', 'false').lower() == 'true',
        'landing_offset_meters': float(os.getenv('LANDING_OFFSET_METERS', '100.0')),
        'landing_alternative_offset_meters': float(os.getenv('LANDING_ALTERNATIVE_OFFSET_METERS', '200.0')),
        'landing_start_angle_deg': float(os.getenv('LANDING_START_ANGLE_DEG', '60.0')),
        'landing_max_length_meters': float(os.getenv('LANDING_MAX_LENGTH_METERS', '200.0')),
        'landing_perch_stability_threshold': float(os.getenv('LANDING_PERCH_STABILITY_THRESHOLD', '0.5')),
        'landing_perch_min_angle_deg': float(os.getenv('LANDING_PERCH_MIN_ANGLE_DEG', '30.0')),
        'landing_perch_stability_window': int(os.getenv('LANDING_PERCH_STABILITY_WINDOW', '5'))
    }
    
    # Create and run emulator
    emulator = DrillingEmulator(config)
    emulator.run_continuous_processing()


if __name__ == "__main__":
    main()
