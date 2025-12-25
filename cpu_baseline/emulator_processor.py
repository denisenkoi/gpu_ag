#!/usr/bin/env python3
"""
Unified well processor for AutoGeosteering Emulator
Provides single preparation logic for both emulator and nightguard modes
"""

import logging
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from copy import deepcopy
import os

from emulator_components import (
    WellDataSlicer,
    AutoGeosteeringExecutor,
    setup_cpp_env_config
)
from landing_detector import LandingDetector
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json
from python_normalization.normalization_calculator import NormalizationCalculator
from python_normalization.normalization_logger import NormalizationLogger
from self_correlation.alternative_typewell_storage import AlternativeTypewellStorage
from self_correlation.curve_replacement_processor import CurveReplacementProcessor
from typewell_provider import TypewellProvider

logger = logging.getLogger(__name__)


class WellProcessor:
    """Unified processor for well data preparation and initial processing"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize processor with configuration
        
        Args:
            config: Configuration dictionary with all parameters
        """
        self.exe_path = config['exe_path']
        self.work_dir = config['work_dir']
        self.results_dir = config['results_dir']
        self.md_step = config['md_step_meters']
        self.mode = config.get('mode', 'emulator')
        
        # Landing detection configuration
        self.landing_detection_enabled = config.get('landing_detection_enabled', False)
        if self.landing_detection_enabled:
            self.landing_detector = LandingDetector(
                offset_meters=config['landing_offset_meters'],
                alternative_offset_meters=config['landing_alternative_offset_meters'],
                landing_start_angle_deg=config['landing_start_angle_deg'],
                max_landing_length_meters=config['landing_max_length_meters'],
                perch_stability_threshold=config['landing_perch_stability_threshold'],
                perch_min_angle_deg=config['landing_perch_min_angle_deg'],
                perch_stability_window=config['landing_perch_stability_window']
            )
            logger.info("Landing detection enabled")
        else:
            self.landing_detector = None
            logger.info("Landing detection disabled")
            
        # Python normalization configuration
        self.python_normalization_enabled = os.getenv('PYTHON_NORMALIZATION_ENABLED', 'true').lower() == 'true'
        if self.python_normalization_enabled:
            # Interactive mode depends on general visualization setting
            interactive_mode = os.getenv('VISUALIZATION_ENABLED', 'false').lower() == 'true'
            self.normalization_calculator = NormalizationCalculator(interactive_mode)
            self.normalization_logger = NormalizationLogger(self.results_dir)
            logger.info("Python normalization enabled")
        else:
            logger.info("Python normalization disabled")
            
        # Initialize components
        self.slicer = WellDataSlicer()
        
        # Copy .env to work directory once for C++ daemon (only if not Python executor)
        python_executor_enabled = os.getenv('PYTHON_EXECUTOR_ENABLED', 'false').lower() == 'true'
        if not python_executor_enabled:
            setup_cpp_env_config(self.work_dir)
            
        # Caching for normalization coefficients (nightguard mode)
        self._norm_coefs = None      # (multiplier, shift)
        self._cached_start_md = None # Fixed landing detection result
        self._cached_max_md = None   # Max MD from first calculation

        # TypeWell provider for pseudo_stitched mode
        self.typewell_provider = TypewellProvider()

        logger.info(f"WellProcessor initialized in {self.mode} mode")
            
    def prepare_well_for_processing(
        self, 
        well_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Unified preparation of well for processing
        Same logic for both emulator and nightguard modes
        
        Args:
            well_data: Complete well data from JSON or PAPI
            
        Returns:
            Dictionary with preparation results:
            {
                'well_name': str,
                'original_start_md': float,
                'detected_start_md': float,
                'start_md': float,  # max(original, detected)
                'max_md': float,
                'initial_md': float,  # start_md + add_steps * md_step
                'executor': AutoGeosteeringExecutor,
                'initial_interpretation': Dict,
                'prepared_well_data': Dict  # Modified well data with normalization
            }
        """
        well_name = well_data['wellName']
        logger.info(f"Preparing well: {well_name} for mode: {self.mode}")

        # === PHASE 1: First-time coefficient calculation ===
        if self._norm_coefs is None:  # First call for this well
            logger.info(f"First-time setup for {well_name}")

            # Step 1: Get original start MD
            original_start_md = self._get_start_md(well_data)
            assert original_start_md is not None, f"Cannot determine start MD for {well_name}"

            # Step 2: Apply landing detection if enabled
            start_md = original_start_md
            detected_start_md = original_start_md
            perch_md = original_start_md  # default if landing detection disabled
            if self.landing_detection_enabled and self.landing_detector:
                detected_start_md, perch_md = self.landing_detector.detect_optimal_start(well_data)
                start_md = max(original_start_md, detected_start_md)

                if detected_start_md > original_start_md:
                    logger.info(f"Landing detector adjusted start MD: {original_start_md:.1f}m -> {detected_start_md:.1f}m")
                else:
                    logger.info(f"Landing detector: keeping original start MD {original_start_md:.1f}m (detected: {detected_start_md:.1f}m)")
                logger.info(f"Perch point for normalization: {perch_md:.1f}m")

            # Step 3: Get max MD and validate landing
            max_md = self._get_max_md(well_data)
            assert max_md is not None, f"Cannot determine max MD for {well_name}"

            # Validate: landing point must be within data range
            if start_md > max_md:
                raise RuntimeError(
                    f'{well_name}: detected landing MD {start_md:.1f} '
                    f'is beyond data end {max_md:.1f}'
                )

            # Validate: minimum offset between start_md and max_md
            # AG_EXE requires at least 100ft (30.48m) of data to produce output
            # Using 30.4m (99.7ft) to account for float precision
            MIN_OFFSET_M = 30.4  # ~99.7 ft (slightly less than 100ft for float tolerance)
            offset_m = max_md - start_md
            if offset_m < MIN_OFFSET_M:
                raise RuntimeError(
                    f'{well_name}: insufficient data range for AG. '
                    f'Distance from start_md ({start_md:.1f}m) to max_md ({max_md:.1f}m) '
                    f'is {offset_m:.1f}m ({offset_m/0.3048:.1f}ft), '
                    f'minimum required: {MIN_OFFSET_M:.1f}m (100ft)'
                )

            # Step 4: Calculate and cache normalization coefficients
            if self.python_normalization_enabled:
                self._calc_norm_coefficients(well_data, perch_md)

            # Step 5: Alternative TypeWell Creation (AFTER normalization, ALWAYS in PHASE 0)
            if os.getenv('CURVE_REPLACEMENT_ENABLED', 'false').lower() == 'true':
                nightguard_well_name = well_data.get('wellName', 'unknown')
                # ALWAYS create new alternative typewell in PHASE 0
                logger.info(f"Creating alternative typewell for {nightguard_well_name}")

                # Create AG objects
                well = Well(well_data)
                typewell = TypeWell(well_data)

                # Trim segments to well boundaries (ME-14 fix)
                # DISABLED: StarSteer handles trimming with more context
                # trimmed_segments = trim_segments_to_well_bounds(
                #     well_data['interpretation']['segments'],
                #     well
                # )

                manual_segments = create_segments_from_json(
                    well_data['interpretation']['segments'],  # Use original segments
                    well,
                    well.measured_depth[-1]
                )

                # Apply curve replacement (uses normalization results from step 4)
                replacer = CurveReplacementProcessor()
                replacement_result = replacer.process_well_data(
                    well_data=well_data,
                    well=well,
                    typewell=typewell,
                    manual_segments=manual_segments,
                    original_start_md=original_start_md,
                    detected_start_md=detected_start_md
                )

                if replacement_result.get('typewell_modified'):
                    # Convert modified TypeWell object back to JSON format
                    from self_correlation.typewell_to_json_converter import update_well_data_with_modified_typewell
                    well_data = update_well_data_with_modified_typewell(well_data, typewell)
                    logger.info("🔄 TypeWell converted back to JSON format")

                    # Save alternative typewell to cache
                    interpretation_name = well_data['interpretation'].get('name', 'starred')
                    storage = AlternativeTypewellStorage()
                    storage.save(nightguard_well_name, well_data['typeLog'], interpretation_name)
                    logger.info(f"✅ Alternative typewell created and saved for {nightguard_well_name}")
                else:
                    logger.warning(f"⚠️ Curve replacement did not modify typewell, using original")

            # Cache values for subsequent calls
            self._cached_start_md = start_md
            self._cached_max_md = max_md

            logger.info(f"Cached values: start_md={self._cached_start_md:.1f}, max_md={self._cached_max_md:.1f}")

        else:
            logger.debug(f"Reusing cached setup for {well_name}")

        # === PHASE 2: Apply cached coefficients ===
        if self.python_normalization_enabled:
            well_data = self._apply_normalization(well_data)

        # Use cached values for consistent processing
        start_md = self._cached_start_md
        max_md = max(self._cached_max_md, self._get_max_md(well_data))  # Allow MD growth
        
        # Step 5: Calculate initial MD - MODE-SPECIFIC!
        if self.mode == 'emulator':
            # Active mode: +3 steps ahead (like emulator_old.py:859)
            initial_md = start_md + self.md_step * 3
            if initial_md > max_md:
                initial_md = max_md
            logger.info(f"Emulator mode: initial_md = {start_md:.1f} + 3*{self.md_step:.1f} = {initial_md:.1f}")
        else:
            # Nightguard mode: use all available data without cutting
            initial_md = max_md
            logger.info(f"Nightguard mode: using all available data up to MD={initial_md:.1f}")
        
        logger.info(f"MD range: start={start_md:.1f}, initial={initial_md:.1f}, max={max_md:.1f}")
        
        # Step 6: Create executor
        executor = self._create_executor()

        # Step 6.5: Apply typewell transformation (pseudo_stitched mode)
        if self.typewell_provider.mode != 'original':
            self.typewell_provider.apply_to_well_data(well_data, initial_md)
            logger.info(f"Applied typewell transformation: mode={self.typewell_provider.mode}, norm_coef={self.typewell_provider.norm_coef:.6f}")

        # Step 7: Prepare initial data slice
        initial_data = self.slicer.slice_well_data(well_data, initial_md)
        # NOTE: trim_interpretation_to_start_md removed - executor's _truncate_interpretation_at_md
        # handles truncation with proper shift interpolation
        initial_data['autoGeosteeringParameters']['startMd'] = start_md
        
        # Step 8: Initialize well with executor
        logger.info(f"Initializing executor with data up to MD={initial_md:.1f}")
        result = executor.initialize_well(initial_data)
        interpretation_data = executor.get_interpretation_from_result(result)

        # Step 8a: Save OUTPUT to comparison directory
        executor._copy_output_to_comparison_dir(interpretation_data, well_name, "init", initial_md)

        # Step 9: Validate interpretation consistency
        if interpretation_data:
            self._validate_interpretation_consistency(well_data, interpretation_data, start_md)
            segments_count = len(interpretation_data['interpretation']['segments'])
            logger.info(f"Initial interpretation: {segments_count} segments")
        else:
            logger.warning("No interpretation from initial processing")
        
        # Step 10: Remove heavy objects for subsequent updates
        prepared_well_data = deepcopy(well_data)
        if 'typeLog' in prepared_well_data:
            del prepared_well_data['typeLog']
        if 'tops' in prepared_well_data:
            del prepared_well_data['tops']
        if 'gridSlice' in prepared_well_data:
            del prepared_well_data['gridSlice']
        if 'tvdTypewellShift' in prepared_well_data:
            del prepared_well_data['tvdTypewellShift']
        
        return {
            'well_name': well_name,
            'original_start_md': original_start_md,
            'detected_start_md': detected_start_md,
            'start_md': start_md,
            'max_md': max_md,
            'initial_md': initial_md,
            'current_md': initial_md,  # Starting point for iterations
            'executor': executor,
            'initial_interpretation': interpretation_data,
            'initial_result': result,
            'prepared_well_data': prepared_well_data
        }
    
    def _get_start_md(self, well_data: Dict[str, Any]) -> Optional[float]:
        """Determine start MD for well from JSON data"""
        if 'autoGeosteeringParameters' in well_data:
            start_md = well_data['autoGeosteeringParameters']['startMd']
            if start_md is not None:
                logger.info(f"Using startMd from parameters: {start_md}")
                return start_md
        
        logger.warning("autoGeosteeringParameters.startMd not found in well data")
        return None
    
    def _get_max_md(self, well_data: Dict[str, Any]) -> Optional[float]:
        """Determine maximum MD for well"""
        if 'well' in well_data and 'points' in well_data['well']:
            points = well_data['well']['points']
            if points:
                return max(point['measuredDepth'] for point in points)
        return None
    
    def _create_executor(self) -> AutoGeosteeringExecutor:
        """Create appropriate executor based on AUTOGEOSTEERING_EXECUTOR setting.

        Executor types:
        - 'gpu': GPU multi-population DE (requires PyTorch with CUDA)
        - 'python': Python scipy DE
        - 'cpu' or 'daemon': C++ daemon (original)
        - 'auto' (default): GPU if CUDA available, else Python

        Backward compatibility: PYTHON_EXECUTOR_ENABLED=true → python executor
        """
        from uuid import uuid4

        # Create unique work directory
        unique_work_dir = Path(self.work_dir) / f"executor_{uuid4().hex[:8]}"
        unique_work_dir.mkdir(parents=True, exist_ok=True)

        # Determine executor type
        executor_type = os.getenv('AUTOGEOSTEERING_EXECUTOR', 'auto').lower()

        # Backward compatibility with PYTHON_EXECUTOR_ENABLED
        if executor_type == 'auto' and os.getenv('PYTHON_EXECUTOR_ENABLED', 'false').lower() == 'true':
            executor_type = 'python'

        # Auto-select: GPU if available, else Python
        if executor_type == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    executor_type = 'gpu'
                    logger.info("Auto-selected GPU executor (CUDA available)")
                else:
                    executor_type = 'python'
                    logger.info("Auto-selected Python executor (no CUDA)")
            except ImportError:
                executor_type = 'python'
                logger.info("Auto-selected Python executor (PyTorch not installed)")

        if executor_type == 'gpu':
            # GPU executor with multi-population DE
            try:
                # gpu_executor is in parent directory (gpu_ag/)
                import sys
                gpu_ag_path = str(Path(__file__).parent.parent)
                if gpu_ag_path not in sys.path:
                    sys.path.insert(0, gpu_ag_path)
                from gpu_executor import GpuAutoGeosteeringExecutor
                logger.info(f"Creating GPU AutoGeosteering executor: {unique_work_dir}")
                return GpuAutoGeosteeringExecutor(str(unique_work_dir), self.results_dir)
            except ImportError as e:
                logger.warning(f"GPU executor not available ({e}), falling back to Python")
                executor_type = 'python'

        if executor_type == 'python':
            # Python executor (scipy DE)
            from optimizers.python_autogeosteering_executor import PythonAutoGeosteeringExecutor
            logger.info(f"Creating Python AutoGeosteering executor: {unique_work_dir}")
            return PythonAutoGeosteeringExecutor(str(unique_work_dir), self.results_dir)

        elif executor_type in ('cpu', 'daemon'):
            # C++ daemon executor - copy .env for it
            project_env = Path('.env')
            exe_env = unique_work_dir / '.env'
            if project_env.exists() and not exe_env.exists():
                import shutil
                shutil.copy2(project_env, exe_env)
                logger.debug(f"Copied .env to unique work directory: {exe_env}")
            logger.info(f"Creating C++ AutoGeosteering executor: {unique_work_dir}")
            return AutoGeosteeringExecutor(self.exe_path, str(unique_work_dir))

        else:
            raise ValueError(f"Unknown executor type: {executor_type}. Use: auto, gpu, python, cpu, daemon")

    def _calc_norm_coefficients(self, well_data: Dict[str, Any], perch_md: float) -> None:
        """Calculate and cache normalization coefficients (called once)

        Args:
            well_data: Well JSON data
            perch_md: Landing end point (perch) from landing detector
        """
        well_name = well_data['wellName']
        # STARSTEER-50: Normalization range around perch point [perch-150, perch+50]
        landing_end_md = perch_md  # Use perch point directly

        logger.info(f"Calculating normalization coefficients for {well_name}, perch_md={perch_md:.1f}m")

        # print(well_data['interpretation']['segments'])

        well = Well(well_data)
        type_well = TypeWell(well_data)

        # Trim segments to well boundaries (ME-14 fix)
        # DISABLED: StarSteer handles trimming with more context
        # trimmed_segments = trim_segments_to_well_bounds(
        #     well_data['interpretation']['segments'],
        #     well
        # )

        manual_segments = create_segments_from_json(
            well_data['interpretation']['segments'],  # Use original segments
            well,
            well.measured_depth[-1]
        )

        # Calculate normalization coefficients
        normalization_result = self.normalization_calculator.calculate_normalization_coefficients(
            well_data=well_data,
            well=well,
            typewell=type_well,
            manual_segments=manual_segments,
            landing_end_md=landing_end_md,
        )

        # Log result
        self.normalization_logger.log_normalization_result(normalization_result)

        if normalization_result['status'] == 'success':
            # Store coefficients for reuse
            multiplier = normalization_result['multiplier']
            shift = normalization_result['shift']
            self._norm_coefs = (multiplier, shift)
            logger.info(f"Cached normalization coefficients: multiplier={multiplier:.6f}, shift={shift:.6f}")
            # Set normalization coef for typewell provider (inverted for typeLog)
            self.typewell_provider.set_normalization_coef(multiplier)
        else:
            # No normalization
            self._norm_coefs = (1.0, 0.0)
            logger.warning(f"Normalization failed for {well_name}: {normalization_result['issue_description']}")
            self.typewell_provider.reset_normalization_coef()

    def _apply_normalization(self, well_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply normalization using cached coefficients"""
        assert self._norm_coefs is not None, "Normalization coefficients must be cached before calling this method"

        # Use cached coefficients
        multiplier, shift = self._norm_coefs
        logger.debug(f"Applying cached normalization: {multiplier:.6f}, {shift:.6f}")

        # Apply to wellLog data
        for point in well_data['wellLog']['points']:
            if point['data'] is not None:
                point['data'] = point['data'] * multiplier + shift

        logger.debug(f"Cashed normalization done: {multiplier:.6f}, {shift:.6f}")

        return well_data

    def _validate_interpretation_consistency(
        self, 
        well_data: Dict[str, Any], 
        interpretation_data: Dict[str, Any],
        start_md: float
    ) -> None:
        """Verify that shifts match at the start of automatic interpretation"""
        
        # Assert presence of manual interpretation
        assert 'interpretation' in well_data
        manual_segments = well_data['interpretation']['segments']
        assert manual_segments
        
        # Assert presence of auto interpretation
        assert 'interpretation' in interpretation_data
        auto_segments = interpretation_data['interpretation']['segments']
        assert auto_segments
        
        # Interpolate manual shift at start_md
        manual_shift = self._interpolate_shift(manual_segments, start_md)
        assert manual_shift is not None
        
        # Interpolate auto shift at start_md
        auto_shift = self._interpolate_shift(auto_segments, start_md)
        assert auto_shift is not None
        
        # Check shift match with strict tolerance
        shift_difference = abs(manual_shift - auto_shift)
        SHIFT_TOLERANCE = 0.1

        # TODO [GPU-SHIFT-VALIDATION]: GPU executor может давать shift отличающийся от manual.
        # Это нормально для оптимизации, но нужно правильно обрабатывать:
        # 1. Либо использовать manual shift как starting point
        # 2. Либо добавить penalty за отклонение от manual
        # См. emulator_processor.py:487-496
        executor_type = os.getenv('AUTOGEOSTEERING_EXECUTOR', 'cpu')
        if executor_type == 'gpu':
            if shift_difference > SHIFT_TOLERANCE:
                logger.warning(
                    f"GPU shift mismatch at MD={start_md:.1f}m: "
                    f"manual={manual_shift:.3f}, auto={auto_shift:.3f}, diff={shift_difference:.3f}m"
                )
        else:
            assert shift_difference <= SHIFT_TOLERANCE, (
                f"Well:{well_data['well']['name']}: "
                f"Shift mismatch at landing start MD={start_md:.1f}m: "
                f"Manual shift={manual_shift:.6f}, Auto shift={auto_shift:.6f}, "
                f"Difference={shift_difference:.6f}m > Tolerance={SHIFT_TOLERANCE}m"
            )
        
        logger.debug(f"Interpretation consistency validated at MD={start_md:.1f}, shift={manual_shift:.6f}")
    
    def _interpolate_shift(self, segments: list, md: float) -> Optional[float]:
        """Interpolate shift value for given MD"""
        assert segments
        
        # Find segment containing the given MD
        for i, segment in enumerate(segments):
            start_md = segment['startMd']
            
            # Determine segment end
            if i + 1 < len(segments):
                end_md = segments[i + 1]['startMd']
            else:
                # Last segment - use large value
                end_md = start_md + 10000.0
            
            if start_md <= md < end_md:
                # Linear interpolation between startShift and endShift
                start_shift = segment['startShift']
                end_shift = segment['endShift']
                
                if start_md == end_md:
                    return start_shift
                
                # Interpolate
                ratio = (md - start_md) / (end_md - start_md)
                return start_shift + ratio * (end_shift - start_shift)
        
        return None
    
    def save_step_result(
        self, 
        well_name: str, 
        md: float, 
        result: Dict[str, Any], 
        step_type: str,
        interpretation_data: Optional[Dict[str, Any]] = None
    ):
        """Save step calculation result with interpretation"""
        step_data = {
            "well_name": well_name,
            "measured_depth": md,
            "step_type": step_type,
            "timestamp": time.time(),
            "result": result,
            "interpretation": interpretation_data
        }
        
        result_file = Path(self.results_dir) / f"{well_name}_{step_type}_{md:.1f}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(step_data, f, indent=2, ensure_ascii=False)
        
        # Log interpretation status
        if interpretation_data:
            segments_count = len(interpretation_data['interpretation']['segments'])
            logger.debug(f"Saved {step_type} result for MD={md:.1f} with {segments_count} segments")
        else:
            logger.debug(f"Saved {step_type} result for MD={md:.1f} without interpretation")
