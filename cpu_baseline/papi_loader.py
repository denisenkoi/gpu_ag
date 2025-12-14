import sys
import os
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Optional
import json
import pandas as pd
from datetime import datetime
import time
import random
from urllib3.exceptions import ProtocolError
from requests.exceptions import ConnectionError, ChunkedEncodingError

from papi_export.config import PAPIConfig
from papi_export.api.data_fetch_connector import DataFetchConnector
from papi_export.fetchers.lateral_fetcher import LateralFetcher
from papi_export.fetchers.typewell_fetcher import TypewellFetcher
from papi_export.fetchers.tops_fetcher import TopsFetcher
from papi_export.fetchers.interpretation_fetcher import InterpretationFetcher
from papi_export.fetchers.grid_fetcher import GridFetcher
from papi_export.fetchers.targetline_fetcher import TargetLineFetcher
from papi_export.processors.trajectory_processor import TrajectoryProcessor
from papi_export.json_builder import JSONBuilder

# NEW IMPORT - using refactored comparator module
from papi_export.comparator import compare_json_files

# Setup logging with reduced verbosity for external libraries
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Reduce logging noise from external libraries
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests_oauthlib').setLevel(logging.WARNING)
logging.getLogger('oauthlib').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class PAPILoader:
    """Main orchestrator for PAPI data export with TrajectoryProcessor coordination"""
    
    # Retry configuration
    MAX_RETRY_ATTEMPTS = 5
    BASE_DELAY = 2  # Base delay in seconds
    MAX_DELAY = 150  # Maximum delay in seconds
    
    def __init__(self, well_name: str, wells_config_file: str = None, config_dict: dict = None, ng_logger=None):
        """Initialize loader with configuration"""
        logger.info("Initializing PAPI Loader")

        # Night Guard logger
        self.ng_logger = ng_logger

        # Store config parameters for connection reuse
        self.wells_config_file = wells_config_file
        self.api = None  # Will be initialized once
        
        # Initialize logging files
        self.reports_dir = Path('./papi_export_reports')
        self.reports_dir.mkdir(exist_ok=True)
        
        # Load configuration - prioritize config_dict over file
        if config_dict:
            # Create config from dict parameters
            self.config = PAPIConfig(well_name=well_name, config_dict=config_dict)
            self._init_api_connection()
            self._init_fetchers()
        elif wells_config_file:
            # Load config from file (existing logic)
            self.config = PAPIConfig(wells_config_file=wells_config_file, well_name=well_name)
            self._init_api_connection()
            self._init_fetchers()
        else:
            raise ValueError("Either config_dict or wells_config_file must be provided")
        
    def _init_api_connection(self):
        """Initialize API connection (called once for connection reuse)"""
        # Set log level from config
        if self.config.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
            
        # Initialize API connector
        if self.api is None:
            self.api = DataFetchConnector()
            logger.info("🔌 API connection initialized (will be reused for all wells)")
        
    def _init_fetchers(self):
        """Initialize fetchers for current config"""
        # Initialize fetchers (now return raw data only)
        self.lateral_fetcher = LateralFetcher(self.config, self.api)
        self.typewell_fetcher = TypewellFetcher(self.config, self.api)
        self.tops_fetcher = TopsFetcher(self.config, self.api)
        self.interpretation_fetcher = InterpretationFetcher(self.config, self.api, self.ng_logger)
        self.grid_fetcher = GridFetcher(self.config, self.api)

        # NEW: Target line fetcher for Night Guard (only creates, doesn't use yet)
        self.targetline_fetcher = TargetLineFetcher(self.config, self.api)
        
        # Initialize trajectory processor
        self.trajectory_processor = TrajectoryProcessor(self.config)
        
        # Initialize JSON builder
        self.json_builder = JSONBuilder(self.config)
        
    def _is_connection_error(self, exception: Exception) -> bool:
        """Check if exception is a temporary connection error that should be retried"""
        error_str = str(exception).lower()
        
        # Check for common connection error patterns
        connection_patterns = [
            'connection aborted',
            'remote end closed connection',
            'connection reset',
            'connection timeout',
            'timeout',
            'network is unreachable',
            'temporary failure in name resolution',
            'read timeout',
            'connection broken'
        ]
        
        # Check exception type
        if isinstance(exception, (ConnectionError, ChunkedEncodingError, ProtocolError)):
            return True
            
        # Check error message patterns
        for pattern in connection_patterns:
            if pattern in error_str:
                return True
                
        return False
        
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter"""
        # Exponential backoff: 2^attempt * base_delay
        delay = self.BASE_DELAY * (2 ** (attempt - 1))
        
        # Cap at maximum delay
        delay = min(delay, self.MAX_DELAY)
        
        # Add random jitter (±25%)
        jitter = random.uniform(0.75, 1.25)
        final_delay = delay * jitter
        
        return final_delay
        
    def run_all_wells(self):
        """Process all wells from config file"""
        if not self.wells_config_file:
            raise ValueError("Wells config file not provided")
            
        # Load wells list
        wells_list = PAPIConfig.load_wells_list(self.wells_config_file)
        logger.info(f"📋 Processing {len(wells_list)} wells from config")
        
        # Initialize API connection once
        if self.api is None:
            # Use first well config to initialize API
            temp_config = PAPIConfig(self.wells_config_file, wells_list[0])
            self.config = temp_config
            self._init_api_connection()
        
        results = []
        for i, well_name in enumerate(wells_list, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"🏗️  PROCESSING WELL {i}/{len(wells_list)}: {well_name}")
            logger.info(f"{'='*80}")
            
            # Retry logic with exponential backoff
            success = False
            last_exception = None
            
            for attempt in range(1, self.MAX_RETRY_ATTEMPTS + 1):
                try:
                    # Load config for this well (reusing API connection)
                    self.config = PAPIConfig(self.wells_config_file, well_name)
                    self._init_fetchers()
                    
                    # Process well
                    output_path = self.run()
                    results.append({'well_name': well_name, 'status': 'success', 'output_path': output_path})
                    
                    logger.info(f"✅ {well_name} completed successfully")
                    success = True
                    break
                    
                except Exception as e:
                    last_exception = e
                    
                    # Check if this is a retryable connection error
                    if self._is_connection_error(e) and attempt < self.MAX_RETRY_ATTEMPTS:
                        delay = self._calculate_retry_delay(attempt)
                        logger.warning(f"⚠️  {well_name} connection error (attempt {attempt}/{self.MAX_RETRY_ATTEMPTS}): {str(e)}")
                        logger.info(f"🔄 Retrying after {delay:.1f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        # Non-retryable error or max attempts reached
                        if attempt >= self.MAX_RETRY_ATTEMPTS:
                            logger.error(f"❌ {well_name} failed after {self.MAX_RETRY_ATTEMPTS} attempts: {str(e)}")
                        else:
                            logger.error(f"❌ {well_name} failed (non-retryable error): {str(e)}")
                        break
            
            # Handle final failure
            if not success:
                error_details = {
                    'error': str(last_exception),
                    'traceback': traceback.format_exc(),
                    'retry_attempts': attempt,
                    'is_connection_error': self._is_connection_error(last_exception)
                }
                results.append({'well_name': well_name, 'status': 'failed', 'error': str(last_exception), 'details': error_details})
                
                # Log error details to file
                self._log_error_to_file(well_name, error_details)
                
        # Summary
        successful = len([r for r in results if r['status'] == 'success'])
        failed = len([r for r in results if r['status'] == 'failed'])
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 FINAL SUMMARY: {successful} successful, {failed} failed")
        logger.info(f"{'='*80}")
        
        if successful > 0:
            logger.info("Successful wells:")
            for result in results:
                if result['status'] == 'success':
                    logger.info(f"  ✅ {result['well_name']}")
                    
        if failed > 0:
            logger.warning("Failed wells:")
            for result in results:
                if result['status'] == 'failed':
                    logger.warning(f"  ❌ {result['well_name']}: {result['error']}")
                    
        # Save summary report to file
        self._save_summary_report(results)
        
        return results
    
    def _log_error_to_file(self, well_name: str, error_details: dict):
        """Log error details to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        error_file = self.reports_dir / f'errors_{timestamp}.log'
        
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"WELL: {well_name}\n")
            f.write(f"TIME: {datetime.now().isoformat()}\n")
            f.write(f"ERROR: {error_details['error']}\n")
            f.write(f"TRACEBACK:\n{error_details['traceback']}\n")
            f.write(f"{'='*80}\n")
    
    def _sanitize_for_json(self, obj):
        """Convert Path objects to strings for JSON serialization"""
        if hasattr(obj, '__fspath__'):  # Path-like object
            return str(obj)
        elif isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        return obj
    
    def _save_summary_report(self, results: List[dict]):
        """Save processing summary to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.reports_dir / f'processing_summary_{timestamp}.json'
        
        successful = len([r for r in results if r['status'] == 'success'])
        failed = len([r for r in results if r['status'] == 'failed'])
        
        # Create lists of successful and failed wells
        successful_wells = [r['well_name'] for r in results if r['status'] == 'success']
        failed_wells = [{
            'well_name': r['well_name'], 
            'error': str(r['error'])
        } for r in results if r['status'] == 'failed']
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_wells': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': f"{successful/len(results)*100:.1f}%" if results else "0%",
            'successful_wells': successful_wells,
            'failed_wells': failed_wells,
            'results': [self._sanitize_for_json(r) for r in results]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
            
        logger.info(f"📄 Summary report saved to: {report_file}")
        
        # Also save failed wells list for easy analysis
        if failed > 0:
            failed_file = self.reports_dir / f'failed_wells_{timestamp}.txt'
            with open(failed_file, 'w', encoding='utf-8') as f:
                f.write(f"Failed Wells Report - {datetime.now().isoformat()}\n")
                f.write(f"Total: {failed}/{len(results)}\n\n")
                for result in results:
                    if result['status'] == 'failed':
                        f.write(f"{result['well_name']}: {result['error']}\n")
            logger.info(f"📝 Failed wells list saved to: {failed_file}")
        
    def run(self, interpretation_name: Optional[str] = None):
        """Main execution flow with optional interpretation name override"""
        logger.info("="*60)
        logger.info("Starting PAPI data export with TrajectoryProcessor")
        logger.info(f"Project: {self.config.project_name}")
        logger.info(f"Well: {self.config.well_name}")
        logger.info(f"Target horizon: {self.config.horizon_name}")
        logger.info("="*60)
        
        # Step 1: API connection is automatic in DataFetchConnector
        logger.info("Step 1: Connecting to PAPI")
        
        # Get project
        logger.info("Step 2: Finding project")
        project = self.api.get_project_by_name(self.config.project_name)
        assert project, f"Project '{self.config.project_name}' not found"
        project_uuid = project['uuid']

        # Night Guard checkpoint
        if self.ng_logger:
            self.ng_logger.log_checkpoint("papi_project_found", {
                "project_name": self.config.project_name,
                "project_uuid": project_uuid
            })
        
        # Step 3: Fetch raw lateral data
        logger.info("Step 3: Fetching raw lateral well data")
        raw_lateral_data = self.lateral_fetcher.fetch_raw_lateral_data(
            project_uuid, self.config.well_name
        )
        well_uuid = raw_lateral_data['well_basic']['uuid']
        project_measure_unit = raw_lateral_data['project_measure_unit']

        # Store project units for Night Guard
        self.project_measure_unit = project_measure_unit

        logger.info(f"Raw lateral data: {len(raw_lateral_data['survey_trajectory'])} survey points, "
                   f"units: {project_measure_unit}")

        # Night Guard checkpoint
        if self.ng_logger:
            self.ng_logger.log_checkpoint("papi_well_found", {
                "well_name": self.config.well_name,
                "well_uuid": well_uuid,
                "survey_points": len(raw_lateral_data['survey_trajectory']),
                "project_units": project_measure_unit
            })
        
        # Step 4: Process lateral trajectory with proper interpolation
        logger.info("Step 4: Processing lateral trajectory with TrajectoryProcessor")
        processed_trajectory = self.trajectory_processor.process_lateral_trajectory(
            raw_survey_points=raw_lateral_data['survey_trajectory'],
            well_metadata=raw_lateral_data['well_metadata'],
            project_measure_unit=project_measure_unit
        )
        
        # Log trajectory characteristics
        self._log_trajectory_info(processed_trajectory)

        # Night Guard checkpoint
        if self.ng_logger:
            md_range = [processed_trajectory[0]['measuredDepth'], processed_trajectory[-1]['measuredDepth']] if processed_trajectory else [0, 0]
            self.ng_logger.log_checkpoint("papi_trajectory_processed", {
                "points_count": len(processed_trajectory),
                "md_range": md_range
            })
        
        # Step 5: Process lateral log data with correct sequence
        logger.info("Step 5: Processing lateral log data with correct sequence")
        welllog_data = self._build_welllog_structure(
            raw_lateral_data['log_data'],
            processed_trajectory
        )
        
        # Log well log characteristics
        self._log_welllog_info(welllog_data)

        # Night Guard checkpoint
        if self.ng_logger:
            log_md_range = [welllog_data['points'][0]['measuredDepth'], welllog_data['points'][-1]['measuredDepth']] if welllog_data.get('points') else [0, 0]
            self.ng_logger.log_checkpoint("papi_welllog_processed", {
                "log_name": raw_lateral_data['log_data'].get('log_name', 'unknown'),
                "points_count": len(welllog_data.get('points', [])),
                "md_range": log_md_range
            })
        
        # Step 6: Fetch raw typewell data
        logger.info("Step 6: Fetching raw typewell data")
        raw_typewell_data = self.typewell_fetcher.fetch_raw_typewell_data(well_uuid)
        
        if raw_typewell_data:
            shift_value = raw_typewell_data['shift_value']
            logger.info(f"Found typewell with shift: {shift_value}")
            
            # Step 7: Process typewell trajectory
            # Step 7: Process typewell trajectory
            logger.info("Step 7: Processing typewell trajectory")
            processed_typewell_trajectory, typewell_log_with_tvd = self.trajectory_processor.process_complete_well_trajectory_with_log(
                raw_survey_points=raw_typewell_data['raw_trajectory'],
                raw_log_points=raw_typewell_data['log_data']['log_points'],
                well_metadata=raw_typewell_data['typewell_metadata'],
                project_measure_unit=project_measure_unit
            )
            
            # Process typewell log data with correct sequence
            typelog_data = self._build_typelog_structure(
                raw_typewell_data['log_data'],
                processed_typewell_trajectory
            )
            
            typewell_data = {
                'typewell_uuid': raw_typewell_data['typewell_basic']['uuid'],
                'typewell_name': raw_typewell_data['typewell_basic']['name'],
                'shift': shift_value,
                'typeLog': typelog_data,
                'metadata': raw_typewell_data['typewell_metadata'],
                'trajectory_points': processed_typewell_trajectory
            }
            
            self._log_typewell_info(typewell_data)

            # Night Guard checkpoint
            if self.ng_logger:
                self.ng_logger.log_checkpoint("papi_typewell_processed", {
                    "typewell_name": typewell_data['typewell_name'],
                    "shift": shift_value,
                    "log_points": len(typelog_data.get('points', []))
                })
        else:
            logger.warning("No typewell found")
            typewell_data = None

            # Night Guard checkpoint
            if self.ng_logger:
                self.ng_logger.log_checkpoint("papi_typewell_not_found", {
                    "reason": "no_typewell_in_papi"
                })
            
        # Convert processed trajectory to simple format for other fetchers
        simple_trajectory = []
        for point in processed_trajectory:
            simple_trajectory.append({
                'md': point['measuredDepth'],
                'tvd': point['trueVerticalDepth'],
                'ns': point['northSouth'],
                'ew': point['eastWest']
            })
        
        # Step 8: Fetch tops data
        logger.info("Step 8: Fetching tops data")
        tops_data = self.tops_fetcher.fetch_tops_data(well_uuid, simple_trajectory)
        logger.info(f"Found {len(tops_data)} tops")
        self._log_tops_info(tops_data)

        # Night Guard checkpoint
        if self.ng_logger:
            self.ng_logger.log_checkpoint("papi_tops_found", {
                "tops_count": len(tops_data)
            })
        
        # Step 9: Fetch interpretation data with dynamic search
        # Determine interpretation name: priority to parameter, fallback to config
        target_interpretation_name = interpretation_name
        if target_interpretation_name is None:
            target_interpretation_name = getattr(self.config, 'source_interpretation_name', None)

        # Log what we're searching for
        if target_interpretation_name is None or target_interpretation_name.strip() == "" or target_interpretation_name.lower() == "starred":
            logger.info("Step 9: Will search for starred interpretation")
        else:
            logger.info(f"Step 9: Will search for interpretation by name: '{target_interpretation_name}'")

        interpretation_data = self.interpretation_fetcher.fetch_interpretation_data(
            well_uuid, simple_trajectory, tops_data,
            interpretation_name=target_interpretation_name  # PASS PARAMETER
        )

        # NEW: Cache interpretation data for Night Guard use
        self._last_interpretation_data = interpretation_data

        if interpretation_data:
            horizon_points = interpretation_data['horizon_tvdss']
            available_horizons = interpretation_data.get('available_horizons', [])

            # Validate that horizon was found
            if not horizon_points:
                horizons_list = ', '.join(available_horizons) if available_horizons else 'None'
                error_msg = (
                    f"Horizon '{self.config.horizon_name}' not found in interpretation "
                    f"'{interpretation_data['interpretation_name']}'. "
                    f"Available horizons: {horizons_list}"
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            if horizon_points:
                logger.info(f"Found {len(horizon_points)} horizon points")
                
            segments = interpretation_data['segments']
            if segments:
                logger.info(f"Applied shift correction and got {len(segments)} corrected segments")
                logger.info(f"Shift offset applied: {interpretation_data['shift_offset']}")
                self._log_interpretation_info(segments)

            # Night Guard checkpoint
            if self.ng_logger:
                horizon_md_range = [horizon_points[0]['md'], horizon_points[-1]['md']] if horizon_points else [0, 0]
                self.ng_logger.log_checkpoint("papi_interpretation_found", {
                    "horizon_points": len(horizon_points),
                    "segments_count": len(segments),
                    "horizon_md_range": horizon_md_range,
                    "shift_offset": interpretation_data['shift_offset']
                })
                
            # Save horizon data for analysis
            if self.config.save_intermediate:
                self._save_horizon_analysis(horizon_points)
                self._save_shift_analysis(interpretation_data)
                
        else:
            logger.error("Manual interpretation not found - cannot proceed")

            # Night Guard checkpoint
            if self.ng_logger:
                self.ng_logger.log_checkpoint("papi_interpretation_not_found", {
                    "reason": "no_manual_interpretation_in_papi"
                }, success=False)

            return None
            
        # Step 10: Fetch grid data
        # Step 10: Fetch grid data
        logger.info("Step 10: Fetching grid and calculating slice")
        x_surface = raw_lateral_data['well_basic']['xySurface']['xSurface']
        y_surface = raw_lateral_data['well_basic']['xySurface']['ySurface']
        grid_slice_data = self.grid_fetcher.fetch_grid_slice(
            project_uuid, simple_trajectory, x_surface, y_surface
        )
        if grid_slice_data:
            logger.info(f"Calculated grid slice with {len(grid_slice_data['points'])} points")
            self._log_grid_info(grid_slice_data)

            # Night Guard checkpoint
            if self.ng_logger:
                self.ng_logger.log_checkpoint("papi_grid_processed", {
                    "grid_points": len(grid_slice_data['points'])
                })
        else:
            logger.warning("No grid data found")

            # Night Guard checkpoint
            if self.ng_logger:
                self.ng_logger.log_checkpoint("papi_grid_not_found", {
                    "reason": "no_grid_in_papi"
                })
            
        # Step 11: Build lateral data structure
        logger.info("Step 11: Building lateral data structure")
        lateral_data = {
            'well': raw_lateral_data['well_basic'],
            'wellLog': welllog_data,
            'metadata': raw_lateral_data['well_metadata'],
            'project_measure_unit': project_measure_unit,
            'survey_trajectory': raw_lateral_data['survey_trajectory']
        }
        
        # Add processed trajectory points to well structure
        lateral_data['well']['points'] = processed_trajectory
        
        # Save intermediate raw data if configured
        if self.config.save_intermediate:
            logger.info("Step 12: Saving intermediate raw data")
            self.json_builder.save_intermediate_raw_data(
                lateral_data,
                typewell_data,
                tops_data,
                interpretation_data,
                grid_slice_data
            )
            
        # Build final JSON
        logger.info("Step 13: Building final JSON structure")
        final_json = self.json_builder.build_complete_json(
            lateral_data,
            typewell_data,
            tops_data,
            interpretation_data,
            grid_slice_data
        )

        # NIGHT GUARD FIX: Convert and save interpretation data
        if interpretation_data:
            # Convert horizon_tvdss from feet to meters ONLY if project is in feet
            if 'horizon_tvdss' in interpretation_data and interpretation_data['horizon_tvdss']:
                if self.project_measure_unit == 'FOOT':
                    for point in interpretation_data['horizon_tvdss']:
                        point['md'] = point['md'] * 0.3048  # feet to meters
                        point['vs'] = point['vs'] * 0.3048  # feet to meters
                    logger.info(f"NIGHT_GUARD_DEBUG: Converted {len(interpretation_data['horizon_tvdss'])} horizon points from feet to meters")
                    first_point = interpretation_data['horizon_tvdss'][0]
                    logger.info(f"NIGHT_GUARD_DEBUG: First horizon point after conversion: {first_point}")
                else:
                    logger.info(f"NIGHT_GUARD_DEBUG: Project units are {self.project_measure_unit}, no conversion needed for horizon_tvdss")
            self._last_interpretation_data = interpretation_data
            logger.info(f"NIGHT_GUARD_DEBUG: Saved interpretation data")

        # Save final JSON
        logger.info("Step 14: Saving final JSON")

        # Check if curve replacement is enabled - replace typeLog with alternative typewell
        curve_replacement_enabled = os.getenv('CURVE_REPLACEMENT_ENABLED', 'false').lower() == 'true'
        if curve_replacement_enabled:
            logger.info("🔄 CURVE_REPLACEMENT_ENABLED=true detected")
            logger.info("   Loading alternative typewell from cache to replace PAPI typewell")

            # Import storage module
            from self_correlation.alternative_typewell_storage import AlternativeTypewellStorage

            storage = AlternativeTypewellStorage()
            well_name = final_json['wellName']
            nightguard_well_name = os.getenv('NIGHTGUARD_WELL_NAME')

            # Use NIGHTGUARD_WELL_NAME if set, otherwise use well_name from JSON
            typewell_key = nightguard_well_name if nightguard_well_name else well_name

            if storage.exists(typewell_key):
                # Load alternative typewell and replace in JSON
                alternative_typewell = storage.load(typewell_key)
                final_json['typeLog'] = alternative_typewell
                logger.info(f"✅ Replaced PAPI typewell with alternative typewell for: {typewell_key}")
                logger.info("💾 Saving JSON with ALTERNATIVE typewell (with curve replacement)")
            else:
                logger.warning(f"⚠️ Alternative typewell not found for '{typewell_key}', using PAPI typewell")
                logger.warning("   Note: Alternative typewell will be created on first emulator run")
                logger.info("💾 Saving JSON with PAPI typewell (alternative not yet created)")
        else:
            logger.info("💾 Saving JSON with PAPI typewell (CURVE_REPLACEMENT_ENABLED=false)")

        output_path = self.json_builder.save_json(final_json)
        
        # Auto-compare with reference file (if enabled)
        comparison_report = None
        if os.getenv('COMPARISON_ENABLED', 'true').lower() == 'true':
            logger.info("Step 15: Auto-comparing with reference file")
            comparison_report = self._auto_compare_with_reference(final_json['wellName'])
        else:
            logger.info("Step 15: Comparison disabled by configuration")
        
        logger.info("="*60)
        logger.info("Export completed successfully!")
        logger.info(f"Output saved to: {output_path}")
        if interpretation_data:
            logger.info(f"Manual interpretation used: {interpretation_data['interpretation_name']}")
            logger.info(f"Shift correction applied: {interpretation_data['shift_offset']}")
        if comparison_report:
            if comparison_report['comparison_passed']:
                logger.info("✅ Reference comparison: PASSED")
            else:
                logger.warning(f"❌ Reference comparison: FAILED ({comparison_report['total_differences']} differences)")
        logger.info("="*60)
        
        return output_path
        
    def _build_welllog_structure(
        self, 
        raw_log_data: Optional[Dict], 
        processed_trajectory: List[Dict]
    ) -> Dict:
        """Build wellLog structure using correct sequence - interpolate log to grid first"""
        
        welllog_structure = {
            'uuid': '',
            'points': [],
            'tvdSortedPoints': []
        }
        
        if not raw_log_data:
            return welllog_structure
            
        log_points = raw_log_data.get('log_points', [])
        if not log_points:
            return welllog_structure
            
        welllog_structure['uuid'] = raw_log_data.get('log_uuid', '')
        
        # Debug: Log structure of first few points
        if len(log_points) > 0:
            logger.debug(f"First log point structure: {list(log_points[0].keys())}")
            if len(log_points) > 0:
                first_point = log_points[0]
                logger.debug(f"First point sample: md={first_point.get('md', 'MISSING')}, data={first_point.get('data', 'MISSING')}")
        
        # Process points array (no TVD)
        from papi_export.utils import extract_val

        for point in log_points:
            try:
                # Handle different data structures using extract_val
                md = extract_val(point['md']) if isinstance(point['md'], dict) else point['md']
                value = extract_val(point['data'])

                # Skip null values
                if value is None:
                    continue

                welllog_structure['points'].append({
                    'measuredDepth': md,
                    'data': value
                })
            except KeyError as e:
                logger.error(f"Missing key in log point: {e}. Point structure: {point.keys()}")
                raise

        # Check for duplicate MD values in wellLog.points
        from collections import Counter
        all_md = [p['measuredDepth'] for p in welllog_structure['points']]
        md_counts = Counter(all_md)
        duplicates = {md: count for md, count in md_counts.items() if count > 1}

        if duplicates:
            logger.error(f"DUPLICATE_CHECK_WELLLOG: Found {len(duplicates)} duplicate MD in wellLog.points")
            for dup_md, count in duplicates.items():
                indices = [i for i, p in enumerate(welllog_structure['points']) if p['measuredDepth'] == dup_md]
                logger.error(f"  MD={dup_md} appears {count} times at indices: {indices}")

            assert False, f"Duplicate MD values found in wellLog.points: {len(duplicates)} duplicates"
            
        # CORRECT SEQUENCE: Process log with trajectory using new method
        log_with_tvd = self.trajectory_processor.process_log_with_trajectory(
            raw_log_points=log_points,
            processed_trajectory=processed_trajectory
        )
        
        # Filter for monotonic TVD growth
        monotonic_tvd_points = self.trajectory_processor.filter_monotonic_tvd_points(log_with_tvd)
        
        # Sort by TVD
        monotonic_tvd_points.sort(key=lambda x: x['trueVerticalDepth'])
        welllog_structure['tvdSortedPoints'] = monotonic_tvd_points
        
        logger.info(f"Processed {len(log_points)} log points, {len(monotonic_tvd_points)} with monotonic TVD")
        
        return welllog_structure
        
    def _build_typelog_structure(
        self, 
        raw_log_data: Optional[Dict], 
        processed_trajectory: List[Dict]
    ) -> Dict:
        """Build typeLog structure using correct sequence - interpolate log to grid first"""
        
        typelog_structure = {
            'uuid': '',
            'points': [],
            'tvdSortedPoints': []
        }
        
        if not raw_log_data:
            logger.warning("No log data for typewell")
            return typelog_structure
            
        log_points = raw_log_data.get('log_points', [])
        if not log_points:
            logger.warning("Empty log points for typewell")
            return typelog_structure
            
        typelog_structure['uuid'] = raw_log_data.get('log_uuid', '')
        
        # Process points array (no TVD)
        from papi_export.utils import extract_val
        for point in log_points:
            try:
                # Handle different data structures using extract_val
                md = extract_val(point['md']) if isinstance(point['md'], dict) else point['md']
                value = extract_val(point['data'])
                
                # Skip null values
                if value is None:
                    continue
                    
                typelog_structure['points'].append({
                    'measuredDepth': md,
                    'data': value
                })
            except KeyError as e:
                logger.error(f"Missing key in typewell log point: {e}. Point structure: {point.keys()}")
                raise
            
        # CORRECT SEQUENCE: Process log with trajectory using new method
        # Filter log points with valid data (handle both formats)
        valid_log_points = []
        for p in log_points:
            data_val = p.get('data')
            if isinstance(data_val, dict) and 'val' in data_val:
                if data_val['val'] is not None:
                    valid_log_points.append(p)
            elif isinstance(data_val, (int, float)) and data_val is not None:
                valid_log_points.append(p)
        
        log_with_tvd = self.trajectory_processor.process_log_with_trajectory(
            raw_log_points=valid_log_points,
            processed_trajectory=processed_trajectory
        )
        
        # Filter for monotonic TVD growth
        monotonic_tvd_points = self.trajectory_processor.filter_monotonic_tvd_points(log_with_tvd)
        
        # Sort by TVD
        monotonic_tvd_points.sort(key=lambda x: x['trueVerticalDepth'])
        typelog_structure['tvdSortedPoints'] = monotonic_tvd_points
        
        logger.info(f"Processed typewell log: {len(log_points)} points, {len(monotonic_tvd_points)} with monotonic TVD")
        
        return typelog_structure
        
    def _save_horizon_analysis(self, horizon_points: list):
        """Save horizon points for segment analysis"""
        output_file = self.config.output_dir / 'intermediate' / 'horizon_analysis.json'
        
        # Calculate slopes for analysis
        analysis = {
            'horizon_name': self.config.horizon_name,
            'points_count': len(horizon_points),
            'points': horizon_points,
            'slopes': []
        }
        
        # Calculate slopes between consecutive points
        for i in range(1, len(horizon_points)):
            md_diff = horizon_points[i]['md'] - horizon_points[i-1]['md']
            tvdss_diff = horizon_points[i]['tvdss'] - horizon_points[i-1]['tvdss']
            
            if md_diff > 0:
                slope = tvdss_diff / md_diff
                analysis['slopes'].append({
                    'start_md': horizon_points[i-1]['md'],
                    'end_md': horizon_points[i]['md'],
                    'slope': slope,
                    'tvdss_change': tvdss_diff
                })
                
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        logger.debug(f"Saved horizon analysis to {output_file}")
        
    def _save_shift_analysis(self, interpretation_data: dict):
        """Save shift correction analysis"""
        output_file = self.config.output_dir / 'intermediate' / 'shift_correction_analysis.json'
        
        analysis = {
            'interpretation_name': interpretation_data['interpretation_name'],
            'horizon_name': self.config.horizon_name,
            'shift_offset_applied': interpretation_data['shift_offset'],
            'original_segments_count': len(interpretation_data.get('segments', [])),
            'corrected_segments': interpretation_data.get('segments', [])
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
            
        logger.debug(f"Saved shift correction analysis to {output_file}")
        
    def _log_trajectory_info(self, trajectory_points: List[Dict]):
        """Log trajectory characteristics for diagnostics"""
        if not trajectory_points:
            logger.warning("Empty trajectory data")
            return
            
        count = len(trajectory_points)
        first_md = trajectory_points[0]['measuredDepth']
        last_md = trajectory_points[-1]['measuredDepth']
        first_tvd = trajectory_points[0]['trueVerticalDepth']
        last_tvd = trajectory_points[-1]['trueVerticalDepth']
        
        # Calculate average MD step
        md_steps = []
        for i in range(1, min(20, count)):  # Check first 20 steps
            md_step = trajectory_points[i]['measuredDepth'] - trajectory_points[i-1]['measuredDepth']
            md_steps.append(md_step)
            
        avg_md_step = sum(md_steps) / len(md_steps) if md_steps else 0
        
        logger.info(f"📊 TRAJECTORY: {count} points, MD: {first_md:.1f}→{last_md:.1f}, "
                   f"TVD: {first_tvd:.1f}→{last_tvd:.1f}, avg_step: {avg_md_step:.2f}")
                   
    def _log_welllog_info(self, welllog_data: Dict):
        """Log well log characteristics for diagnostics"""
        points = welllog_data.get('points', [])
        tvd_points = welllog_data.get('tvdSortedPoints', [])
        
        if not points:
            logger.warning("Empty well log data")
            return
            
        count = len(points)
        tvd_count = len(tvd_points)
        first_md = points[0]['measuredDepth']
        last_md = points[-1]['measuredDepth']
        
        # Data value range (extract safely)
        values = []
        for p in points:
            data_val = p.get('data')
            if isinstance(data_val, dict) and 'val' in data_val:
                values.append(data_val['val'])
            elif isinstance(data_val, (int, float)):
                values.append(data_val)
        
        if values:
            min_val = min(values)
            max_val = max(values)
        else:
            min_val = max_val = 0.0
        
        logger.info(f"📈 WELL_LOG: {count} points ({tvd_count} with TVD), MD: {first_md:.1f}→{last_md:.1f}, "
                   f"values: {min_val:.1f}→{max_val:.1f}")
                   
    def _log_typewell_info(self, typewell_data: Dict):
        """Log typewell characteristics for diagnostics"""
        typelog = typewell_data.get('typeLog', {})
        points = typelog.get('points', [])
        tvd_points = typelog.get('tvdSortedPoints', [])
        shift = typewell_data.get('shift', 0)
        
        if not points:
            logger.warning("Empty typewell log data")
            return
            
        count = len(points)
        tvd_count = len(tvd_points)
        first_md = points[0]['measuredDepth']
        last_md = points[-1]['measuredDepth']
        
        # Data value range (extract safely)
        values = []
        for p in points:
            data_val = p.get('data')
            if isinstance(data_val, dict) and 'val' in data_val:
                values.append(data_val['val'])
            elif isinstance(data_val, (int, float)):
                values.append(data_val)
        
        if values:
            min_val = min(values)
            max_val = max(values)
        else:
            min_val = max_val = 0.0
        
        logger.info(f"🔄 TYPEWELL: {count} points ({tvd_count} with TVD), MD: {first_md:.1f}→{last_md:.1f}, "
                   f"values: {min_val:.1f}→{max_val:.1f}, shift: {shift:.3f}")
                   
    def _log_tops_info(self, tops_data: List[Dict]):
        """Log tops characteristics for diagnostics"""
        if not tops_data:
            logger.warning("No tops data")
            return
            
        logger.info(f"🏔️  TOPS: {len(tops_data)} markers:")
        for top in tops_data:
            name = top['name']
            md = top['measuredDepth']
            tvd = top['trueVerticalDepth']
            logger.info(f"    {name}: MD={md:.1f}, TVD={tvd:.1f}")
            
    def _log_interpretation_info(self, segments: List[Dict]):
        """Log interpretation segments characteristics for diagnostics"""
        if not segments:
            logger.warning("No interpretation segments")
            return
            
        count = len(segments)
        first_md = segments[0]['startMd']
        last_md = segments[-1]['startMd']
        
        shifts = [s['startShift'] for s in segments] + [s['endShift'] for s in segments]
        min_shift = min(shifts)
        max_shift = max(shifts)
        
        logger.info(f"🔀 INTERPRETATION: {count} segments, MD: {first_md:.1f}→{last_md:.1f}, "
                   f"shifts: {min_shift:.3f}→{max_shift:.3f}")
                   
    def _log_grid_info(self, grid_data: Dict):
        """Log grid slice characteristics for diagnostics"""
        points = grid_data.get('points', [])
        
        if not points:
            logger.warning("No grid slice data")
            return
            
        count = len(points)
        first_md = points[0]['measuredDepth']
        last_md = points[-1]['measuredDepth']
        first_vs = points[0]['verticalSection']
        last_vs = points[-1]['verticalSection']
        
        # Calculate coordinate ranges
        ns_coords = [p['northSouth'] for p in points]
        ew_coords = [p['eastWest'] for p in points]
        
        ns_range = f"{min(ns_coords):.1f}→{max(ns_coords):.1f}"
        ew_range = f"{min(ew_coords):.1f}→{max(ew_coords):.1f}"
        
        logger.info(f"🗺️  GRID_SLICE: {count} points, MD: {first_md:.1f}→{last_md:.1f}, "
                   f"VS: {first_vs:.1f}→{last_vs:.1f}, NS: {ns_range}, EW: {ew_range}")
                   
    def _auto_compare_with_reference(self, well_name: str) -> Optional[Dict]:
        """Automatically compare with reference file based on well name"""
        import os

        # USING NEW REFACTORED COMPARATOR
        from papi_export.comparator import compare_json_files
        
        wells_dir = os.getenv('WELLS_DIR')
        if not wells_dir:
            logger.warning("WELLS_DIR not set - skipping reference comparison")
            return None
            
        reference_file = Path(wells_dir) / f"{well_name}.json"
        
        if not reference_file.exists():
            logger.warning(f"Reference file not found: {reference_file} - skipping comparison")
            return None
            
        # Find generated file
        generated_files = list(self.config.output_dir.glob("*.json"))
        generated_files = [f for f in generated_files if not f.name.startswith("intermediate")]
        
        if not generated_files:
            logger.error("No generated JSON file found for comparison")
            return None
            
        generated_file = generated_files[-1]  # Use most recent
        
        logger.info(f"Comparing with reference: {reference_file}")
        
        # Run comparison with refactored comparator
        report = compare_json_files(str(reference_file), str(generated_file), tolerance=0.001)
        
        # Save comparison report
        report_file = self.config.output_dir / 'comparison_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Comparison report saved to: {report_file}")
        return report
        
    def compare_with_reference(self, reference_file: str, tolerance: float = 0.001):
        """Compare generated JSON with reference file (legacy method)"""
        # USING NEW REFACTORED COMPARATOR
        from papi_export.comparator import compare_json_files
        
        # Find the generated file
        generated_files = list(self.config.output_dir.glob("*.json"))
        generated_files = [f for f in generated_files if not f.name.startswith("intermediate")]
        
        if not generated_files:
            logger.error("No generated JSON file found")
            return None
            
        generated_file = generated_files[-1]  # Use most recent
        
        logger.info(f"Comparing with reference file: {reference_file}")
        report = compare_json_files(str(reference_file), str(generated_file), tolerance)
        
        # Save comparison report
        report_file = self.config.output_dir / 'comparison_report.json'
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Comparison report saved to: {report_file}")
        return report

    def get_night_guard_data(self, well_uuid: str, current_well_data: Dict, ng_logger=None) -> Optional[Dict]:
        """
        Get enhanced data for Night Guard alert system

        Args:
            well_uuid: UUID of the well
            current_well_data: Already processed well data from normal flow

        Returns:
            Dict with Night Guard specific data or None if not available
        """
        logger.info("Fetching Night Guard enhanced data")

        night_guard_data = {
            'target_line_data': None,
            'current_max_md': None,
            'vs_coordinate': None,
            'target_tvd_at_end': None,
            'night_guard_ready': False,
            'project_measure_unit': None
        }

        # Get current maximum MD from well data
        current_max_md = self._get_max_md_from_well_data(current_well_data)
        logger.info(f"NIGHT_GUARD_DEBUG: Step 1 - Max MD: {current_max_md}")
        if not current_max_md:
            logger.warning("NIGHT_GUARD_DEBUG: FAILED at Step 1 - Could not determine current max MD")
            return night_guard_data

        night_guard_data['current_max_md'] = current_max_md
        logger.info(f"NIGHT_GUARD_DEBUG: Step 1 SUCCESS - current_max_md = {current_max_md}")

        # Add project measure unit - REQUIRED, no default!
        if hasattr(self, 'project_measure_unit'):
            night_guard_data['project_measure_unit'] = self.project_measure_unit
            logger.info(f"NIGHT_GUARD_DEBUG: Project units: {self.project_measure_unit}")
        else:
            logger.error("NIGHT_GUARD_DEBUG: CRITICAL - project_measure_unit not available! Cannot proceed with unit conversion.")
            raise ValueError("project_measure_unit is required but not set. Ensure it's passed to PAPILoader constructor.")

        import os
        target_line_source = os.getenv('TARGET_LINE_SOURCE', 'papi').lower()

        if target_line_source == 'papi':
            target_line_data = self.targetline_fetcher.fetch_target_line_data(well_uuid)
            if not target_line_data:
                logger.warning(f"NIGHT_GUARD_DEBUG: FAILED at Step 2 - No target line data available")
                return night_guard_data

            if self.project_measure_unit == 'FOOT':
                logger.info(f"NIGHT_GUARD_DEBUG: Converting target line from feet to meters")
                target_line_data['origin_vs'] = target_line_data['origin_vs'] * 0.3048
                target_line_data['target_vs'] = target_line_data['target_vs'] * 0.3048
                target_line_data['origin_tvd'] = target_line_data['origin_tvd'] * 0.3048
                target_line_data['target_tvd'] = target_line_data['target_tvd'] * 0.3048

            night_guard_data['target_line_data'] = target_line_data

        elif target_line_source == 'starsteer':
            logger.info(f"NIGHT_GUARD_DEBUG: Step 2 - Using StarSteer mode, skipping PAPI target line")

        else:
            logger.error(f"Unknown TARGET_LINE_SOURCE: {target_line_source}")
            return night_guard_data

        logger.info(f"NIGHT_GUARD_DEBUG: Step 3 - Looking for VS coordinate for MD={current_max_md}")
        vs_coordinate = self._get_vs_coordinate_from_last_interpretation(current_max_md)
        if vs_coordinate is None:
            logger.warning(f"NIGHT_GUARD_DEBUG: FAILED at Step 3 - Could not determine VS coordinate")
            return night_guard_data

        night_guard_data['vs_coordinate'] = vs_coordinate

        if target_line_source == 'papi':
            target_tvd_at_end = self.targetline_fetcher.interpolate_target_tvd_at_vs(
                target_line_data, vs_coordinate
            )
            if target_tvd_at_end is None:
                logger.warning("Could not calculate target TVD from PAPI")
                return night_guard_data

            night_guard_data['target_tvd_at_end'] = target_tvd_at_end
            night_guard_data['night_guard_ready'] = True

        elif target_line_source == 'starsteer':
            night_guard_data['target_tvd_at_end'] = None
            night_guard_data['night_guard_ready'] = True

        return night_guard_data

    def _get_max_md_from_well_data(self, well_data: Dict) -> Optional[float]:
        """Extract maximum MD from well data structure"""

        well_points = well_data.get('well', {}).get('points', [])
        if not well_points:
            logger.warning("No well points found in well data")
            return None

        max_md = max(point['measuredDepth'] for point in well_points)
        logger.debug(f"Maximum MD from well points: {max_md:.1f}")
        return max_md

    def _get_vs_coordinate_from_last_interpretation(self, target_md: float) -> Optional[float]:
        """
        Get VS coordinate using interpretation data from last successful run

        This method accesses cached interpretation data from the last run() call
        to avoid duplicate PAPI API calls
        """

        # Check if we have interpretation data from last run
        logger.info(f"NIGHT_GUARD_DEBUG: Step 3.1 - Checking cached interpretation data")
        logger.info(f"NIGHT_GUARD_DEBUG: Step 3.1 - hasattr(_last_interpretation_data): {hasattr(self, '_last_interpretation_data')}")

        if hasattr(self, '_last_interpretation_data'):
            logger.info(f"NIGHT_GUARD_DEBUG: Step 3.1 - _last_interpretation_data is not None: {self._last_interpretation_data is not None}")
            if self._last_interpretation_data:
                logger.info(f"NIGHT_GUARD_DEBUG: Step 3.1 - _last_interpretation_data keys: {list(self._last_interpretation_data.keys())}")

        if not hasattr(self, '_last_interpretation_data') or not self._last_interpretation_data:
            logger.error("NIGHT_GUARD_DEBUG: Step 3.1 FAILED - No cached interpretation data available for VS coordinate lookup")
            return None

        interpretation_data = self._last_interpretation_data
        logger.info(f"NIGHT_GUARD_DEBUG: Step 3.2 - interpretation_data type: {type(interpretation_data)}")
        logger.info(f"NIGHT_GUARD_DEBUG: Step 3.2 - interpretation_data keys: {list(interpretation_data.keys()) if isinstance(interpretation_data, dict) else 'Not a dict'}")

        horizon_points = interpretation_data.get('horizon_tvdss')

        logger.info(f"NIGHT_GUARD_DEBUG: Step 3.2 - horizon_points found: {horizon_points is not None}")
        logger.info(f"NIGHT_GUARD_DEBUG: Step 3.2 - horizon_points type: {type(horizon_points) if horizon_points is not None else 'None'}")

        if horizon_points:
            logger.info(f"NIGHT_GUARD_DEBUG: Step 3.2 - horizon_points count: {len(horizon_points)}")
            if len(horizon_points) > 0:
                first_point = horizon_points[0]
                last_point = horizon_points[-1]
                min_md = min(p['md'] for p in horizon_points)
                max_md = max(p['md'] for p in horizon_points)
                logger.info(f"NIGHT_GUARD_DEBUG: Step 3.2 - MD range: {min_md:.1f} to {max_md:.1f} meters")
                logger.info(f"NIGHT_GUARD_DEBUG: Step 3.2 - first horizon point: MD={first_point['md']:.1f}, VS={first_point['vs']:.1f}")
                logger.info(f"NIGHT_GUARD_DEBUG: Step 3.2 - last horizon point: MD={last_point['md']:.1f}, VS={last_point['vs']:.1f}")
                logger.info(f"NIGHT_GUARD_DEBUG: Step 3.2 - Target MD: {target_md:.1f}")

        if not horizon_points or len(horizon_points) == 0:
            logger.error("NIGHT_GUARD_DEBUG: Step 3.2 FAILED - No horizon data available for VS coordinate lookup")
            logger.error(f"NIGHT_GUARD_DEBUG: Step 3.2 - horizon_points value: {horizon_points}")
            return None

        # Check target MD against horizon range with small tolerance for discretization errors
        min_md = min(p['md'] for p in horizon_points)
        max_md = max(p['md'] for p in horizon_points)
        tolerance = 0.02  # 2 cm tolerance for discretization errors

        if target_md < (min_md - tolerance):
            raise AssertionError(f"CRITICAL: Target MD={target_md:.6f} is before horizon range. "
                                f"Target={target_md:.6f} < min_horizon_MD={min_md:.6f} (tolerance={tolerance:.6f})")

        if target_md > (max_md + tolerance):
            raise AssertionError(f"CRITICAL: Target MD={target_md:.6f} is beyond horizon range. "
                                f"Target={target_md:.6f} > max_horizon_MD={max_md:.6f} (tolerance={tolerance:.6f})")

        # Target is within valid range - find closest point
        if target_md >= max_md:
            # Use last horizon point (at or near max)
            closest_point = max(horizon_points, key=lambda p: p['md'])
            logger.info(f"NIGHT_GUARD_DEBUG: Using last horizon point MD={closest_point['md']:.6f} for target MD={target_md:.6f}")
        else:
            # Find closest point within range
            closest_point = min(
                horizon_points,
                key=lambda p: abs(p['md'] - target_md)
            )

        vs_coordinate = closest_point['vs']
        logger.debug(f"VS coordinate for MD={target_md:.1f}: {vs_coordinate:.1f} "
                    f"(from horizon point at MD={closest_point['md']:.1f})")
        return vs_coordinate

    def cache_interpretation_data(self, interpretation_data: Dict):
        """
        Cache interpretation data for Night Guard use

        This method stores interpretation data that was generated during
        emulator initialization to be used later in Night Guard analysis
        """
        logger.info("NIGHT_GUARD_DEBUG: Caching interpretation data for Night Guard")

        if interpretation_data:
            # Check what's available in interpretation_data
            logger.info(f"NIGHT_GUARD_DEBUG: interpretation_data keys: {list(interpretation_data.keys())}")

            # Extract horizon_tvdss if available
            horizon_points = []
            if 'horizon_tvdss' in interpretation_data:
                horizon_points = interpretation_data['horizon_tvdss']
            elif 'horizon_points' in interpretation_data:
                horizon_points = interpretation_data['horizon_points']

            # Extract segments if available
            segments = []
            if 'interpretation' in interpretation_data:
                segments = interpretation_data['interpretation'].get('segments', [])
            elif 'segments' in interpretation_data:
                segments = interpretation_data['segments']

            # Store cached data
            self._last_interpretation_data = {
                'horizon_tvdss': horizon_points,
                'segments': segments
            }

            logger.info(f"NIGHT_GUARD_DEBUG: Cached {len(segments)} segments, "
                       f"{len(horizon_points)} horizon points")

            if horizon_points:
                logger.info(f"NIGHT_GUARD_DEBUG: First horizon point keys: {list(horizon_points[0].keys())}")
        else:
            logger.warning("NIGHT_GUARD_DEBUG: No interpretation data to cache")


def main():
    """Main entry point"""
    wells_config_file = "wells_config.json"
    
    # Process all wells
    loader = PAPILoader(wells_config_file)
    results = loader.run_all_wells()
    failed_count = len([r for r in results if r['status'] == 'failed'])
    
    if failed_count > 0:
        return 2
        
    return 0


if __name__ == "__main__":
    sys.exit(main())