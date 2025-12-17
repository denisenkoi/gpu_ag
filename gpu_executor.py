"""
GPU-based AutoGeosteering executor using multi-population Differential Evolution.

Clean implementation without monkey-patching.
Select via .env: AUTOGEOSTEERING_EXECUTOR=gpu
"""
import os
import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from copy import deepcopy

# CPU baseline imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from optimizers.base_autogeosteering_executor import BaseAutoGeosteeringExecutor
from optimizers.optimization_logger import OptimizationLogger

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json, segments_to_json, create_segments
from ag_rewards.ag_func_correlations import calculate_correlation

# GPU imports
from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
from torch_funcs.batch_objective import TorchObjectiveWrapper
from ag_numerical.ag_optimizer_utils import calculate_optimization_bounds
from gpu_optimizer_fit import run_multi_population_de, N_POPULATIONS, POPSIZE_EACH, MAXITER

logger = logging.getLogger(__name__)


class GpuAutoGeosteeringExecutor(BaseAutoGeosteeringExecutor):
    """
    GPU-based AutoGeosteering executor using multi-population DE.

    Architecture: Same as PythonAutoGeosteeringExecutor but uses GPU for optimization.
    - INIT: manual -> truncate -> add new segments -> GPU optimize -> self.interpretation
    - STEP: self.interpretation -> truncate -> add new segments -> GPU optimize -> self.interpretation
    """

    def __init__(self, work_dir: str, results_dir: str):
        """Initialize GPU executor."""
        super().__init__(work_dir)

        # Load configuration from environment
        self.segments_count = int(os.getenv('PYTHON_SEGMENTS_COUNT', '4'))
        self.lookback_distance = float(os.getenv('PYTHON_LOOKBACK_DISTANCE', '50.0'))
        self.num_iterations = int(os.getenv('PYTHON_NUM_ITERATIONS', '500'))
        self.angle_range = float(os.getenv('PYTHON_ANGLE_RANGE', '10.0'))

        # Segment length constraints
        self.max_segment_length = float(os.getenv('PYTHON_MAX_SEGMENT_LENGTH', '15.0'))
        self.min_optimization_length = float(os.getenv('PYTHON_MIN_OPTIMIZATION_LENGTH', '15.0'))

        # MD Normalization buffer
        self.md_normalization_buffer = float(os.getenv('MD_NORMALIZATION_BUFFER', '3000.0'))
        self.fixed_md_range = None
        self.fixed_min_md = None

        # Optimization parameters
        self.pearson_power = float(os.getenv('PYTHON_PEARSON_POWER', '2.0'))
        self.mse_power = float(os.getenv('PYTHON_MSE_POWER', '0.001'))
        self.num_intervals_self_correlation = int(os.getenv('PYTHON_NUM_INTERVALS_SC', '0'))  # Disabled for GPU
        self.sc_power = float(os.getenv('PYTHON_SC_POWER', '1.15'))
        self.min_pearson_value = float(os.getenv('PYTHON_MIN_PEARSON_VALUE', '-1.0'))
        self.use_accumulative_bounds = os.getenv('PYTHON_USE_ACCUMULATIVE_BOUNDS', 'true').lower() == 'true'

        # GPU configuration
        self.device = os.getenv('GPU_DEVICE', 'cuda')
        self.n_populations = int(os.getenv('GPU_N_POPULATIONS', str(N_POPULATIONS)))
        self.popsize_each = int(os.getenv('GPU_POPSIZE_EACH', str(POPSIZE_EACH)))
        self.maxiter = int(os.getenv('GPU_MAXITER', str(MAXITER)))

        # State management
        self.interpretation = None
        self.ag_well = None
        self.ag_typewell = None
        self.tvd_to_typewell_shift = None
        self.well_name = None

        # Logging
        self.optimization_logger = OptimizationLogger(results_dir)
        self.interpretation_dir = work_dir
        Path(self.interpretation_dir).mkdir(parents=True, exist_ok=True)

        # Verify GPU availability
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("GPU executor requires CUDA but it's not available!")

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
        logger.info(f"GPU executor initialized: device={self.device}, GPU={gpu_name}")
        logger.info(f"  populations={self.n_populations}, popsize={self.popsize_each}, maxiter={self.maxiter}")

    def start_daemon(self):
        """Start GPU executor."""
        logger.info("GPU AutoGeosteering executor ready")

    def stop_daemon(self):
        """Stop GPU executor."""
        logger.info("Stopping GPU AutoGeosteering executor")
        self.optimization_logger.print_statistics()

    def _gpu_optimize_segments(self, well, typewell, segments, tvd_shift=0.0):
        """
        Optimize segments using GPU multi-population DE.

        Returns:
            Tuple of (optimized_segments, best_fun, elapsed_time)
        """
        import time

        # Calculate bounds
        bounds = calculate_optimization_bounds(segments, self.angle_range, self.use_accumulative_bounds)
        bounds_tensor = torch.tensor(bounds, dtype=torch.float64, device=self.device)

        # Initial projection
        well.calc_horizontal_projection(typewell, segments, tvd_shift)

        # Convert to numpy then torch
        well_np = well_to_numpy(well)
        typewell_np = typewell_to_numpy(typewell)
        segments_np = segments_to_numpy(segments, well)

        well_torch = numpy_to_torch(well_np, device=self.device)
        typewell_torch = numpy_to_torch(typewell_np, device=self.device)
        segments_torch = segments_numpy_to_torch(segments_np, device=self.device)

        # Create objective wrapper
        wrapper = TorchObjectiveWrapper(
            well_data=well_torch,
            typewell_data=typewell_torch,
            segments_torch=segments_torch,
            self_corr_start_idx=0,
            pearson_power=self.pearson_power,
            mse_power=self.mse_power,
            num_intervals_self_correlation=0,  # Disabled for speed
            sc_power=self.sc_power,
            angle_range=self.angle_range,
            angle_sum_power=2.0,
            min_pearson_value=self.min_pearson_value,
            tvd_to_typewell_shift=tvd_shift,
            device=self.device
        )

        # Run multi-population DE
        start_time = time.time()
        result = run_multi_population_de(
            wrapper=wrapper,
            bounds=bounds_tensor,
            n_populations=self.n_populations,
            popsize_each=self.popsize_each,
            maxiter=min(self.num_iterations, self.maxiter),
            seed=None,
            device=self.device
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        # Update segments with optimal shifts
        optimal_shifts = result['x'].cpu().tolist()
        optimized_segments = deepcopy(segments)
        for i, shift in enumerate(optimal_shifts):
            optimized_segments[i].end_shift = shift
            if i < len(optimized_segments) - 1:
                optimized_segments[i + 1].start_shift = shift

        pops_found = sum(1 for f in result['all_best_fun'] if f < 0.2)
        logger.info(f"GPU optimization: {elapsed:.2f}s, fun={result['fun']:.6f}, "
                   f"pops_global={pops_found}/{self.n_populations}")

        return optimized_segments, result['fun'], elapsed

    def _normalize_manual_to_segments(self, manual_json: List[Dict], well: Well) -> List:
        """Convert manual interpretation JSON to normalized Segment objects."""
        from ag_objects.ag_obj_interpretation import Segment

        segments = []
        md_range = self.fixed_md_range if self.fixed_md_range else well.md_range
        min_md = self.fixed_min_md if self.fixed_min_md else well.min_md

        for i, seg_json in enumerate(manual_json):
            start_md_m = seg_json['startMd']

            if 'endMd' in seg_json:
                end_md_m = seg_json['endMd']
            elif i + 1 < len(manual_json):
                end_md_m = manual_json[i + 1]['startMd']
            else:
                end_md_m = well.max_md

            if end_md_m <= min_md or start_md_m >= well.max_md:
                continue

            start_md_m = max(start_md_m, min_md)
            end_md_m = min(end_md_m, well.max_md)

            start_md_norm = (start_md_m - min_md) / md_range
            end_md_norm = (end_md_m - min_md) / md_range
            start_idx = int(well.md2idx(start_md_norm))
            end_idx = int(well.md2idx(end_md_norm))

            if start_idx >= end_idx:
                continue

            start_shift = seg_json.get('startShift', 0.0) / md_range
            end_shift = seg_json.get('endShift', 0.0) / md_range

            segment = Segment(well, start_idx=start_idx, start_shift=start_shift,
                            end_idx=end_idx, end_shift=end_shift)
            segments.append(segment)

        return segments

    def _truncate_interpretation_at_md(self, truncate_md: float, well: Well) -> float:
        """Truncate self.interpretation at given MD, return interpolated shift."""
        if not self.interpretation:
            return 0.0

        truncate_idx = well.md2idx(truncate_md)
        truncate_md_value = well.measured_depth[truncate_idx]

        new_interpretation = []
        stitch_shift = 0.0

        for segment in self.interpretation:
            if segment.end_md <= truncate_md_value:
                new_interpretation.append(segment)
                stitch_shift = segment.end_shift
            elif segment.start_md < truncate_md_value:
                truncated = deepcopy(segment)
                truncated.end_idx = truncate_idx
                truncated.end_md = truncate_md_value
                truncated.end_vs = well.vs_thl[truncate_idx]

                if segment.end_md != segment.start_md:
                    ratio = (truncate_md_value - segment.start_md) / (segment.end_md - segment.start_md)
                else:
                    ratio = 0.0
                truncated.end_shift = segment.start_shift + ratio * (segment.end_shift - segment.start_shift)
                truncated.calc_angle()

                new_interpretation.append(truncated)
                stitch_shift = truncated.end_shift
                break

        self.interpretation = new_interpretation
        return stitch_shift

    def _create_and_append_segments(self, start_idx: int, end_idx: int,
                                     start_shift: float, well: Well) -> int:
        """Create new segments and append to self.interpretation."""
        import math

        new_length_idx = end_idx - start_idx
        new_length_m = new_length_idx * well.horizontal_well_step

        if new_length_m < self.min_optimization_length:
            return 0

        dynamic_segments = math.ceil(new_length_m / self.max_segment_length)
        actual_count = min(dynamic_segments, self.segments_count)
        actual_count = max(actual_count, 1)

        segment_length_idx = new_length_idx // actual_count

        new_segments = create_segments(
            well=well,
            segments_count=actual_count,
            segment_len=segment_length_idx,
            start_idx=start_idx,
            start_shift=start_shift
        )

        if self.interpretation is None:
            self.interpretation = []
        self.interpretation.extend(new_segments)

        return len(new_segments)

    def _get_optimization_indices(self, num_new_segments: int) -> List[int]:
        """Get indices of segments to optimize."""
        if not self.interpretation or num_new_segments <= 0:
            return []
        total = len(self.interpretation)
        return list(range(total - num_new_segments, total))

    def initialize_well(self, well_data: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize well with first dataset using GPU optimization."""
        logger.info("Initializing well with GPU executor")

        self.well_name = well_data['wellName']
        self.ag_well = Well(well_data)
        self.ag_typewell = TypeWell(well_data)
        self.tvd_to_typewell_shift = well_data['tvdTypewellShift']

        start_md_meters = well_data['autoGeosteeringParameters']['startMd']
        self.start_idx = self.ag_well.md2idx(start_md_meters)
        self.start_md_meters = start_md_meters

        manual_segments_json = well_data.get('interpretation', {}).get('segments', [])

        # Calculate fixed planning horizon
        self.fixed_min_md = self.ag_well.min_md
        self.fixed_md_range = (self.ag_well.max_md + self.md_normalization_buffer) - self.fixed_min_md

        # Normalize
        max_curve_value = max(self.ag_well.max_curve, self.ag_typewell.value.max())
        self.ag_well.normalize(max_curve_value, self.ag_typewell.min_depth, self.fixed_md_range)
        self.ag_typewell.normalize(max_curve_value, self.ag_well.min_depth, self.fixed_md_range)

        # Convert manual to segments
        self.interpretation = self._normalize_manual_to_segments(manual_segments_json, self.ag_well)
        logger.info(f"Loaded {len(self.interpretation)} manual segments")

        # Truncate at start
        start_md_norm = self.ag_well.measured_depth[self.start_idx]
        stitch_shift = self._truncate_interpretation_at_md(start_md_norm, self.ag_well)
        prefix_count = len(self.interpretation)

        # Add new segments
        current_idx = len(self.ag_well.measured_depth) - 1
        num_new = self._create_and_append_segments(
            start_idx=self.start_idx,
            end_idx=current_idx,
            start_shift=stitch_shift,
            well=self.ag_well
        )

        if num_new == 0:
            json_segments = self._denormalize_segments_to_json(self.interpretation)
            return {'interpretation': {'segments': json_segments}}

        # GPU optimize
        optimize_indices = self._get_optimization_indices(num_new)
        segments_to_optimize = [self.interpretation[i] for i in optimize_indices]

        optimized_segments, best_fun, elapsed = self._gpu_optimize_segments(
            self.ag_well, self.ag_typewell, segments_to_optimize, self.tvd_to_typewell_shift
        )

        # Update interpretation
        for i, opt_seg in zip(optimize_indices, optimized_segments):
            self.interpretation[i] = opt_seg

        # Calculate final correlation
        self.ag_well.calc_horizontal_projection(self.ag_typewell, optimized_segments, self.tvd_to_typewell_shift)
        corr, _, self_corr, _, _, pearson, num_points, mse, _, _ = calculate_correlation(
            self.ag_well, self.start_idx,
            optimized_segments[0].start_idx, optimized_segments[-1].end_idx,
            float('inf'), 0, 0,
            self.pearson_power, self.mse_power,
            self.num_intervals_self_correlation, self.sc_power, self.min_pearson_value
        )

        # Log
        self._log_optimization_result('init', elapsed, best_fun, corr, pearson, mse)

        # Save
        json_segments = self._denormalize_segments_to_json(self.interpretation)
        self._save_interpretation_file()

        logger.info(f"INIT complete: {prefix_count} prefix + {num_new} new = "
                    f"{len(self.interpretation)} total, corr={corr:.4f}, time={elapsed:.2f}s")

        return {'interpretation': {'segments': json_segments}}

    def get_interpretation_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract interpretation from result."""
        return result

    def update_well_data(self, well_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update well with new data using GPU optimization."""
        assert self.interpretation is not None, "Well not initialized"

        updated_well = Well(well_data)
        max_curve_value = max(updated_well.max_curve, self.ag_typewell.value.max())
        updated_well.normalize(max_curve_value, self.ag_typewell.min_depth, self.fixed_md_range)
        self.ag_well = updated_well

        current_md = updated_well.measured_depth[-1]
        current_idx = len(updated_well.measured_depth) - 1

        lookback_md = current_md - self.lookback_distance / self.fixed_md_range
        lookback_idx = updated_well.md2idx(lookback_md)

        if lookback_idx < self.start_idx:
            lookback_idx = self.start_idx

        lookback_md_value = updated_well.measured_depth[lookback_idx]

        # Truncate
        stitch_shift = self._truncate_interpretation_at_md(lookback_md_value, updated_well)
        prefix_count = len(self.interpretation)

        # Add new segments
        num_new = self._create_and_append_segments(
            start_idx=lookback_idx,
            end_idx=current_idx,
            start_shift=stitch_shift,
            well=updated_well
        )

        if num_new == 0:
            json_segments = self._denormalize_segments_to_json(self.interpretation)
            return {'interpretation': {'segments': json_segments}}

        # GPU optimize
        optimize_indices = self._get_optimization_indices(num_new)
        segments_to_optimize = [self.interpretation[i] for i in optimize_indices]

        optimized_segments, best_fun, elapsed = self._gpu_optimize_segments(
            updated_well, self.ag_typewell, segments_to_optimize, self.tvd_to_typewell_shift
        )

        # Update
        for i, opt_seg in zip(optimize_indices, optimized_segments):
            self.interpretation[i] = opt_seg

        # Correlation
        updated_well.calc_horizontal_projection(self.ag_typewell, optimized_segments, self.tvd_to_typewell_shift)
        corr, _, self_corr, _, _, pearson, num_points, mse, _, _ = calculate_correlation(
            updated_well, lookback_idx,
            optimized_segments[0].start_idx, optimized_segments[-1].end_idx,
            float('inf'), 0, 0,
            self.pearson_power, self.mse_power,
            self.num_intervals_self_correlation, self.sc_power, self.min_pearson_value
        )

        # Log
        self._log_optimization_result('step', elapsed, best_fun, corr, pearson, mse)

        # Save
        json_segments = self._denormalize_segments_to_json(self.interpretation)
        self._save_interpretation_file()

        logger.info(f"STEP complete: {prefix_count} prefix + {num_new} new = "
                    f"{len(self.interpretation)} total, corr={corr:.4f}, time={elapsed:.2f}s")

        return {'interpretation': {'segments': json_segments}}

    def _denormalize_segments_to_json(self, segments) -> List[Dict[str, Any]]:
        """Denormalize segments to JSON."""
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

        return denorm_segments

    def _save_interpretation_file(self):
        """Save interpretation to file."""
        if not self.well_name or not self.ag_well or not self.interpretation:
            return

        import json

        filepath = Path(self.interpretation_dir) / f"{self.well_name}.json"
        json_segments = self._denormalize_segments_to_json(self.interpretation)

        full_result = {'interpretation': {'segments': json_segments}}

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_result, f, indent=2)

    def _log_optimization_result(self, step_type: str, elapsed: float, best_fun: float,
                                 corr: float, pearson: float, mse: float):
        """Log optimization result."""
        optimization_stats = {
            'method': 'gpu_multi_population_de',
            'n_populations': self.n_populations,
            'popsize_each': self.popsize_each,
            'maxiter': self.maxiter,
            'elapsed_time': elapsed,
            'best_fun': best_fun,
            'correlation': corr,
            'pearson': pearson,
            'mse': mse
        }

        optimization_result = {
            'well_name': self.well_name,
            'step_type': step_type,
            'optimization_stats': optimization_stats
        }

        self.optimization_logger.log_optimization_result(optimization_result)
