# python_autogeosteering_executor.py

"""
Python-based AutoGeosteering executor using optimization algorithms
Alternative to C++ daemon implementation
"""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from copy import deepcopy

from optimizers.base_autogeosteering_executor import BaseAutoGeosteeringExecutor
from optimizers.optimization_logger import OptimizationLogger

# AG module imports
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import (
    create_segments_from_json, segments_to_json, create_segments
)
from ag_numerical.ag_func_optimizer import optimizer_fit
from ag_rewards.ag_func_correlations import calculate_correlation

logger = logging.getLogger(__name__)


class PythonAutoGeosteeringExecutor(BaseAutoGeosteeringExecutor):
    """Python-based AutoGeosteering executor using optimization

    Architecture: ONE full interpretation stored in self.interpretation
    - INIT: manual -> truncate -> add new segments -> optimize -> self.interpretation
    - STEP: self.interpretation -> truncate -> add new segments -> optimize -> self.interpretation
    Same logic for both, no special cases.
    """

    def __init__(self, work_dir: str, results_dir: str):
        """
        Initialize Python AutoGeosteering executor

        Args:
            work_dir: Working directory for operations
            results_dir: Directory for results and logs
        """
        super().__init__(work_dir)

        # Load configuration from environment
        self.segments_count = int(os.getenv('PYTHON_SEGMENTS_COUNT', '4'))
        self.lookback_distance = float(os.getenv('PYTHON_LOOKBACK_DISTANCE', '50.0'))
        self.optimizer_method = os.getenv('PYTHON_OPTIMIZER_METHOD', 'differential_evolution')
        self.num_iterations = int(os.getenv('PYTHON_NUM_ITERATIONS', '100'))
        self.angle_range = float(os.getenv('PYTHON_ANGLE_RANGE', '10.0'))

        # Segment length constraints (in meters)
        self.max_segment_length = float(os.getenv('PYTHON_MAX_SEGMENT_LENGTH', '15.0'))  # 50 feet
        self.min_optimization_length = float(os.getenv('PYTHON_MIN_OPTIMIZATION_LENGTH', '15.0'))  # 50 feet

        # MD Normalization - fixed planning horizon for consistent normalization
        # Prevents MD drift between iterations due to changing md_range
        self.md_normalization_buffer = float(os.getenv('MD_NORMALIZATION_BUFFER', '3000.0'))
        self.fixed_md_range = None  # Calculated at INIT, used for all iterations
        self.fixed_min_md = None    # Fixed min_md from INIT

        # Additional optimization parameters
        self.pearson_power = float(os.getenv('PYTHON_PEARSON_POWER', '2.0'))
        self.mse_power = float(os.getenv('PYTHON_MSE_POWER', '0.001'))
        self.angle_sum_power = float(os.getenv('PYTHON_ANGLE_SUM_POWER', '2.0'))
        self.num_intervals_self_correlation = int(os.getenv('PYTHON_NUM_INTERVALS_SC', '20'))
        self.sc_power = float(os.getenv('PYTHON_SC_POWER', '1.15'))
        self.min_pearson_value = float(os.getenv('PYTHON_MIN_PEARSON_VALUE', '-1.0'))
        self.use_accumulative_bounds = os.getenv('PYTHON_USE_ACCUMULATIVE_BOUNDS', 'true').lower() == 'true'

        # State management - ONE full interpretation (all segments, normalized)
        self.interpretation = None  # List[Segment] - FULL interpretation
        self.ag_well = None
        self.ag_typewell = None
        self.tvd_to_typewell_shift = None
        self.well_name = None

        # Logging
        self.optimization_logger = OptimizationLogger(results_dir)

        # Interpretation directory for StarSteer export (same as work_dir)
        self.interpretation_dir = work_dir
        Path(self.interpretation_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Python executor initialized: segments={self.segments_count}, "
                    f"lookback={self.lookback_distance}m, max_segment={self.max_segment_length}m, "
                    f"method={self.optimizer_method}")

    def start_daemon(self):
        """Start Python executor (no actual daemon needed)"""
        logger.info("Python AutoGeosteering executor ready")

    def stop_daemon(self):
        """Stop Python executor and print statistics"""
        logger.info("Stopping Python AutoGeosteering executor")
        self.optimization_logger.print_statistics()

    def _normalize_manual_to_segments(self, manual_json: List[Dict], well: Well) -> List:
        """Convert manual interpretation JSON to normalized Segment objects

        Args:
            manual_json: List of JSON segments [{startMd, startShift, endShift, endMd?}, ...]
            well: Well object (already normalized)

        Returns:
            List of Segment objects with normalized coordinates
        """
        from ag_objects.ag_obj_interpretation import Segment

        segments = []
        # Use fixed_md_range for consistent normalization across iterations
        md_range = self.fixed_md_range if self.fixed_md_range else well.md_range
        min_md = self.fixed_min_md if self.fixed_min_md else well.min_md

        logger.debug(f"NORMALIZE: min_md={min_md:.2f}, max_md={well.max_md:.2f}, md_range={md_range:.2f} (fixed)")
        logger.debug(f"NORMALIZE: well.measured_depth range=[{well.measured_depth.min():.6f}, {well.measured_depth.max():.6f}], len={len(well.measured_depth)}")

        for i, seg_json in enumerate(manual_json):
            # Get MD values (in meters)
            start_md_m = seg_json['startMd']

            # endMd: use explicit or next segment's startMd
            if 'endMd' in seg_json:
                end_md_m = seg_json['endMd']
            elif i + 1 < len(manual_json):
                end_md_m = manual_json[i + 1]['startMd']
            else:
                # Last segment - extend to well end
                end_md_m = well.max_md

            # Skip segments outside well range
            if end_md_m <= min_md or start_md_m >= well.max_md:
                logger.debug(f"  Seg[{i}]: SKIP outside range, start={start_md_m:.2f}, end={end_md_m:.2f}")
                continue

            # Clamp to well range
            start_md_m = max(start_md_m, min_md)
            end_md_m = min(end_md_m, well.max_md)

            # Convert to indices (normalize MD first - well.measured_depth is normalized)
            start_md_norm = (start_md_m - min_md) / md_range
            end_md_norm = (end_md_m - min_md) / md_range
            start_idx = int(well.md2idx(start_md_norm))
            end_idx = int(well.md2idx(end_md_norm))

            logger.debug(f"  Seg[{i}]: md_m=[{start_md_m:.2f}, {end_md_m:.2f}], "
                        f"md_norm=[{start_md_norm:.6f}, {end_md_norm:.6f}], "
                        f"idx=[{start_idx}, {end_idx}], "
                        f"well_md=[{well.measured_depth[start_idx]:.6f}, {well.measured_depth[end_idx]:.6f}]")

            if start_idx >= end_idx:
                logger.debug(f"    -> SKIP start_idx >= end_idx")
                continue

            # Normalize shifts (meters -> normalized)
            start_shift = seg_json.get('startShift', 0.0) / md_range
            end_shift = seg_json.get('endShift', 0.0) / md_range

            # Create Segment using keyword arguments (required by universal constructor)
            segment = Segment(well, start_idx=start_idx, start_shift=start_shift,
                            end_idx=end_idx, end_shift=end_shift)
            segments.append(segment)

        logger.debug(f"Converted {len(manual_json)} manual JSON -> {len(segments)} normalized Segments")
        return segments

    def _truncate_interpretation_at_md(self, truncate_md: float, well: Well) -> float:
        """Truncate self.interpretation at given MD, return interpolated shift

        Removes all segments that START at or after truncate_md.
        Truncates segment that crosses truncate_md, interpolating its end_shift.

        Args:
            truncate_md: MD value (normalized) to truncate at
            well: Well object for VS lookup

        Returns:
            Interpolated shift at truncate_md (for seamless connection)
        """
        if not self.interpretation:
            return 0.0

        truncate_idx = well.md2idx(truncate_md)
        truncate_md_value = well.measured_depth[truncate_idx]
        logger.debug(f"TRUNCATE: truncate_md_norm={truncate_md:.6f}, truncate_idx={truncate_idx}, "
                    f"truncate_md_value={truncate_md_value:.6f}")

        new_interpretation = []
        stitch_shift = 0.0

        for i, segment in enumerate(self.interpretation):
            logger.debug(f"  Seg[{i}]: start_md={segment.start_md:.6f}, end_md={segment.end_md:.6f}, "
                        f"start_shift={segment.start_shift:.6f}, end_shift={segment.end_shift:.6f}")

            if segment.end_md <= truncate_md_value:
                # Segment fully before truncate point - keep as is
                new_interpretation.append(segment)
                stitch_shift = segment.end_shift
                logger.debug(f"    -> KEEP (end_md <= truncate), stitch_shift={stitch_shift:.6f}")
            elif segment.start_md < truncate_md_value:
                # Segment crosses truncate point - truncate it
                logger.debug(f"    -> TRUNCATE (start_md < truncate < end_md)")
                truncated = deepcopy(segment)
                truncated.end_idx = truncate_idx
                truncated.end_md = truncate_md_value
                truncated.end_vs = well.vs_thl[truncate_idx]

                # Interpolate end_shift at truncate point
                if segment.end_md != segment.start_md:
                    ratio = (truncate_md_value - segment.start_md) / (segment.end_md - segment.start_md)
                else:
                    ratio = 0.0
                old_end_shift = segment.end_shift
                truncated.end_shift = segment.start_shift + ratio * (segment.end_shift - segment.start_shift)
                logger.debug(f"    ratio={ratio:.6f}, interpolated end_shift: {old_end_shift:.6f} -> {truncated.end_shift:.6f}")
                truncated.calc_angle()

                new_interpretation.append(truncated)
                stitch_shift = truncated.end_shift
                break  # All subsequent segments are after truncate point
            else:
                # Segment starts at or after truncate point - skip it
                logger.debug(f"    -> SKIP (start_md >= truncate)")

        self.interpretation = new_interpretation
        logger.debug(f"Truncated interpretation at MD={truncate_md_value:.6f}, "
                     f"{len(new_interpretation)} segments remain, stitch_shift={stitch_shift:.6f}")
        return stitch_shift

    def _create_and_append_segments(self, start_idx: int, end_idx: int,
                                     start_shift: float, well: Well) -> int:
        """Create new segments and append to self.interpretation

        Args:
            start_idx: Start index for new segments
            end_idx: End index for new segments
            start_shift: Starting shift (from truncation point)
            well: Well object

        Returns:
            Number of new segments created
        """
        import math

        new_length_idx = end_idx - start_idx
        new_length_m = new_length_idx * well.horizontal_well_step

        if new_length_m < self.min_optimization_length:
            logger.debug(f"New length {new_length_m:.1f}m < min {self.min_optimization_length}m, no segments created")
            return 0

        # Dynamic segment count
        dynamic_segments = math.ceil(new_length_m / self.max_segment_length)
        actual_count = min(dynamic_segments, self.segments_count)
        actual_count = max(actual_count, 1)

        segment_length_idx = new_length_idx // actual_count

        # Create new segments
        new_segments = create_segments(
            well=well,
            segments_count=actual_count,
            segment_len=segment_length_idx,
            start_idx=start_idx,
            start_shift=start_shift
        )

        # Append to interpretation
        if self.interpretation is None:
            self.interpretation = []
        self.interpretation.extend(new_segments)

        logger.debug(f"Created {len(new_segments)} new segments [{start_idx}:{end_idx}], "
                     f"total interpretation: {len(self.interpretation)} segments")
        return len(new_segments)

    def _get_optimization_indices(self, num_new_segments: int) -> List[int]:
        """Get indices of segments to optimize (last num_new_segments)"""
        if not self.interpretation or num_new_segments <= 0:
            return []
        total = len(self.interpretation)
        return list(range(total - num_new_segments, total))

    def initialize_well(self, well_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize well with first dataset

        CONSISTENT LOGIC:
        1. Load manual interpretation -> self.interpretation (full)
        2. Truncate at optimization start point
        3. Add new segments [start -> current_md]
        4. Optimize new segments only
        5. Save full interpretation

        Args:
            well_data: Well JSON data for initialization

        Returns:
            Dict containing interpretation result
        """
        logger.info("Initializing well with Python executor")

        # Store well name
        self.well_name = well_data['wellName']

        # Create AG objects from JSON
        self.ag_well = Well(well_data)
        self.ag_typewell = TypeWell(well_data)
        self.tvd_to_typewell_shift = well_data['tvdTypewellShift']

        # Get start MD BEFORE normalization (md2idx needs meters)
        start_md_meters = well_data['autoGeosteeringParameters']['startMd']
        self.start_idx = self.ag_well.md2idx(start_md_meters)
        self.start_md_meters = start_md_meters

        # Get manual interpretation JSON
        manual_segments_json = well_data.get('interpretation', {}).get('segments', [])

        # Calculate fixed planning horizon BEFORE normalization
        # This ensures consistent md_range across all INIT/STEP iterations
        self.fixed_min_md = self.ag_well.min_md
        self.fixed_md_range = (self.ag_well.max_md + self.md_normalization_buffer) - self.fixed_min_md
        logger.info(f"Fixed planning horizon: min_md={self.fixed_min_md:.2f}, "
                    f"max_md={self.ag_well.max_md:.2f} + buffer={self.md_normalization_buffer:.0f} "
                    f"-> fixed_md_range={self.fixed_md_range:.2f}")

        # Normalize AG objects with fixed md_range
        max_curve_value = max(self.ag_well.max_curve, self.ag_typewell.value.max())
        self.ag_well.normalize(max_curve_value, self.ag_typewell.min_depth, self.fixed_md_range)
        self.ag_typewell.normalize(max_curve_value, self.ag_well.min_depth, self.fixed_md_range)

        # STEP 1: Convert manual to normalized Segment objects -> self.interpretation
        self.interpretation = self._normalize_manual_to_segments(manual_segments_json, self.ag_well)
        logger.info(f"Loaded {len(self.interpretation)} manual segments into interpretation")

        # STEP 2: Truncate at optimization start point
        start_md_norm = self.ag_well.measured_depth[self.start_idx]
        stitch_shift = self._truncate_interpretation_at_md(start_md_norm, self.ag_well)
        prefix_count = len(self.interpretation)
        logger.info(f"After truncation at start_md={start_md_meters:.1f}m: "
                    f"{prefix_count} segments, stitch_shift={stitch_shift * self.fixed_md_range:.3f}m")

        # STEP 3: Add new segments [start -> current_md]
        current_idx = len(self.ag_well.measured_depth) - 1
        num_new = self._create_and_append_segments(
            start_idx=self.start_idx,
            end_idx=current_idx,
            start_shift=stitch_shift,
            well=self.ag_well
        )

        if num_new == 0:
            logger.warning("No new segments created, returning current interpretation")
            json_segments = self._denormalize_segments_to_json(self.interpretation)
            return {'interpretation': {'segments': json_segments}}

        # STEP 4: Optimize only new segments
        optimize_indices = self._get_optimization_indices(num_new)
        segments_to_optimize = [self.interpretation[i] for i in optimize_indices]

        optimization_results = optimizer_fit(
            well=self.ag_well,
            typewell=self.ag_typewell,
            self_corr_start_idx=self.start_idx,
            segments=segments_to_optimize,
            angle_range=self.angle_range,
            angle_sum_power=self.angle_sum_power,
            segm_counts_reg=[2, 4, 6, 10],
            num_iterations=self.num_iterations,
            pearson_power=self.pearson_power,
            mse_power=self.mse_power,
            num_intervals_self_correlation=self.num_intervals_self_correlation,
            sc_power=self.sc_power,
            optimizer_method=self.optimizer_method,
            min_pearson_value=self.min_pearson_value,
            use_accumulative_bounds=self.use_accumulative_bounds,
            tvd_to_typewell_shift=self.tvd_to_typewell_shift
        )

        # Update interpretation with optimized segments
        best_result = optimization_results[0]
        corr, self_correlation, pearson, mse, num_points, optimized_segments, _ = best_result

        for i, opt_seg in zip(optimize_indices, optimized_segments):
            self.interpretation[i] = opt_seg

        # Log statistics
        self._log_optimization_result('init', self.ag_well.measured_depth[-1] * self.fixed_md_range + self.fixed_min_md,
                                      optimization_results, corr, pearson, mse,
                                      self_correlation, num_points)

        # STEP 5: Save full interpretation
        json_segments = self._denormalize_segments_to_json(self.interpretation)
        result = {'interpretation': {'segments': json_segments}}
        self._save_interpretation_file()

        logger.info(f"INIT complete: {prefix_count} prefix + {num_new} new = "
                    f"{len(self.interpretation)} total segments, corr={corr:.4f}")

        return result

    def get_interpretation_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract interpretation from Python executor result"""
        return result

    def update_well_data(self, well_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update well with new data using lookback optimization

        CONSISTENT LOGIC (same as INIT):
        1. self.interpretation already contains full interpretation
        2. Truncate at lookback point
        3. Add new segments [lookback -> current_md]
        4. Optimize new segments only
        5. Save full interpretation

        Args:
            well_data: Updated well JSON data

        Returns:
            Dict containing updated interpretation result
        """
        assert self.interpretation is not None, "Well not initialized"

        # Update AG well object with new data
        updated_well = Well(well_data)

        # Normalize with SAME fixed_md_range as initialization (prevents MD drift)
        max_curve_value = max(updated_well.max_curve, self.ag_typewell.value.max())
        updated_well.normalize(max_curve_value, self.ag_typewell.min_depth, self.fixed_md_range)
        self.ag_well = updated_well

        # Current MD is the last point in new data
        current_md = updated_well.measured_depth[-1]
        current_idx = len(updated_well.measured_depth) - 1

        # Calculate lookback point (normalized) - use fixed_md_range for consistency
        lookback_md = current_md - self.lookback_distance / self.fixed_md_range
        lookback_idx = updated_well.md2idx(lookback_md)

        # PROTECTION: Do not go before start_idx
        if lookback_idx < self.start_idx:
            logger.info(f"Lookback {lookback_idx} < start_idx {self.start_idx}, using start_idx")
            lookback_idx = self.start_idx

        lookback_md_value = updated_well.measured_depth[lookback_idx]

        # STEP 2: Truncate interpretation at lookback point
        stitch_shift = self._truncate_interpretation_at_md(lookback_md_value, updated_well)
        prefix_count = len(self.interpretation)

        # STEP 3: Add new segments [lookback -> current_md]
        num_new = self._create_and_append_segments(
            start_idx=lookback_idx,
            end_idx=current_idx,
            start_shift=stitch_shift,
            well=updated_well
        )

        if num_new == 0:
            logger.info("No new segments created, keeping current interpretation")
            json_segments = self._denormalize_segments_to_json(self.interpretation)
            return {'interpretation': {'segments': json_segments}}

        # STEP 4: Optimize only new segments
        optimize_indices = self._get_optimization_indices(num_new)
        segments_to_optimize = [self.interpretation[i] for i in optimize_indices]

        optimization_results = optimizer_fit(
            well=updated_well,
            typewell=self.ag_typewell,
            self_corr_start_idx=lookback_idx,
            segments=segments_to_optimize,
            angle_range=self.angle_range,
            angle_sum_power=self.angle_sum_power,
            segm_counts_reg=[2, 4, 6, 10],
            num_iterations=self.num_iterations,
            pearson_power=self.pearson_power,
            mse_power=self.mse_power,
            num_intervals_self_correlation=self.num_intervals_self_correlation,
            sc_power=self.sc_power,
            optimizer_method=self.optimizer_method,
            min_pearson_value=self.min_pearson_value,
            use_accumulative_bounds=self.use_accumulative_bounds,
            tvd_to_typewell_shift=self.tvd_to_typewell_shift
        )

        # Update interpretation with optimized segments
        best_result = optimization_results[0]
        corr, self_correlation, pearson, mse, num_points, optimized_segments, _ = best_result

        for i, opt_seg in zip(optimize_indices, optimized_segments):
            self.interpretation[i] = opt_seg

        # Log statistics
        self._log_optimization_result('step', current_md * self.fixed_md_range + self.fixed_min_md,
                                      optimization_results, corr, pearson, mse,
                                      self_correlation, num_points)

        # STEP 5: Save full interpretation
        json_segments = self._denormalize_segments_to_json(self.interpretation)
        result = {'interpretation': {'segments': json_segments}}
        self._save_interpretation_file()

        logger.info(f"STEP complete: {prefix_count} prefix + {num_new} new = "
                    f"{len(self.interpretation)} total segments, corr={corr:.4f}")

        return result

    def _denormalize_segments_to_json(self, segments) -> List[Dict[str, Any]]:
        """Denormalize segment shifts from normalized [0,1] to meters

        After normalization:
        - MD: (md - min_md) / md_range -> [0, 1]
        - Shifts: shift / md_range -> small normalized values

        To denormalize:
        - MD: md_norm * md_range + min_md
        - Shifts: shift_norm * md_range
        """
        # Use fixed_md_range for consistent denormalization across iterations
        md_range = self.fixed_md_range if self.fixed_md_range else self.ag_well.md_range
        min_md = self.fixed_min_md if self.fixed_min_md else self.ag_well.min_md

        denorm_segments = []
        for seg in segments:
            denorm_seg = {
                'startMd': seg.start_md * md_range + min_md,
                'endMd': seg.end_md * md_range + min_md,
                'startShift': seg.start_shift * md_range,
                'endShift': seg.end_shift * md_range
            }
            denorm_segments.append(denorm_seg)

        logger.debug(f"Denormalized {len(segments)} segments: "
                     f"shift range [{segments[0].start_shift:.6f}..{segments[-1].end_shift:.6f}] norm -> "
                     f"[{denorm_segments[0]['startShift']:.3f}..{denorm_segments[-1]['endShift']:.3f}]m")

        return denorm_segments

    def _save_interpretation_file(self):
        """Save FULL interpretation to file for StarSteer export

        Simple logic: just denormalize self.interpretation and save.
        No merging, no special cases - self.interpretation IS the full interpretation.
        """
        if not self.well_name or not self.ag_well or not self.interpretation:
            return

        import json

        filepath = Path(self.interpretation_dir) / f"{self.well_name}.json"

        # Denormalize all segments
        json_segments = self._denormalize_segments_to_json(self.interpretation)

        # Build and save
        full_result = {
            'interpretation': {
                'segments': json_segments
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_result, f, indent=2)

        # Log MD range
        if json_segments:
            first_md = json_segments[0]['startMd']
            last_md = json_segments[-1].get('endMd', json_segments[-1]['startMd'])
            logger.info(f"Saved interpretation: {len(json_segments)} segments, "
                        f"MD range: {first_md:.1f} - {last_md:.1f} m")

    def _log_optimization_result(self, step_type: str, measured_depth: float,
                                 optimization_results: List, correlation: float,
                                 pearson: float, mse: float, self_correlation: float,
                                 num_points: int):
        """Log optimization result to statistics"""

        # Extract optimization stats from first result
        best_result = optimization_results[0]
        optimized_well = best_result[6]  # The well object with optimization stats

        optimization_stats = {
            'method': self.optimizer_method,
            'segments_count': self.segments_count,
            'n_iterations': self.num_iterations,
            'final_correlation': correlation,
            'pearson_correlation': pearson,
            'mse_value': mse,
            'self_correlation': self_correlation,
            'intersections_count': getattr(optimized_well, 'intersections_count', 0),
            'success': True,  # Python optimization always succeeds or throws
            'n_function_evaluations': self.num_iterations  # Approximation
        }

        optimization_result = {
            'well_name': self.well_name,
            'measured_depth': measured_depth,
            'step_type': step_type,
            'optimization_stats': optimization_stats
        }

        # Log stats to console for visibility
        logger.info(f"Optimization [{step_type}]: corr={correlation:.4f}, pearson={pearson:.4f}, "
                    f"mse={mse:.6f}, self_corr={self_correlation:.4f}, points={num_points}")

        self.optimization_logger.log_optimization_result(optimization_result)

    def _copy_output_to_comparison_dir(self, interpretation_data: Dict[str, Any], well_name: str, step_type: str, current_md: float):
        """Copy OUTPUT interpretation JSON to comparison directory for debugging

        Python Executor version - includes differential_evolution settings in output
        """
        import json
        from pathlib import Path

        comparison_dir = os.getenv('JSON_COMPARISON_DIR')
        if not comparison_dir:
            return

        comparison_path = Path(comparison_dir)
        comparison_path.mkdir(exist_ok=True, parents=True)

        version_tag = os.getenv('VERSION_TAG', 'diff_evol')

        # Format: {version}_{wellname}_{step}_{md}_output.json
        filename = f"{version_tag}_{well_name}_{step_type}_{current_md:.1f}_output.json"
        comparison_file = comparison_path / filename

        # Add executor settings to output for traceability
        output_data = {
            'executor': 'PythonAutoGeosteeringExecutor',
            'method': self.optimizer_method,
            'settings': {
                'segments_count': self.segments_count,
                'lookback_distance': self.lookback_distance,
                'max_segment_length': self.max_segment_length,
                'min_optimization_length': self.min_optimization_length,
                'num_iterations': self.num_iterations,
                'angle_range': self.angle_range,
                'pearson_power': self.pearson_power,
                'mse_power': self.mse_power,
                'sc_power': self.sc_power
            },
            'interpretation': interpretation_data
        }

        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {self.optimizer_method} output to comparison: {filename}")