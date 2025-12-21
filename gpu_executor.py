# gpu_executor.py

"""
GPU-accelerated AutoGeosteering executor using EvoTorch SNES

Based on PythonAutoGeosteeringExecutor structure with EvoTorch SNES optimization
from test_gpu_de_correct.py.

Configuration via .env:
    GPU_N_RESTARTS=5        # Number of optimization restarts
    GPU_POPSIZE=100         # Population size per restart
    GPU_MAXITER=200         # Iterations per restart
    USE_PSEUDOTYPELOG=false # Use pseudoTypeLog instead of typeLog
"""

import os
import sys
import logging
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
from copy import deepcopy

# Add cpu_baseline to path for AG modules
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from optimizers.base_autogeosteering_executor import BaseAutoGeosteeringExecutor
from optimizers.optimization_logger import OptimizationLogger

# AG module imports
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments, Segment
from ag_numerical.ag_optimizer_utils import calculate_optimization_bounds

# Torch imports
from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
from torch_funcs.batch_objective import TorchObjectiveWrapper

# EvoTorch
from evotorch import Problem
from evotorch.algorithms import SNES

logger = logging.getLogger(__name__)


class GpuAutoGeosteeringExecutor(BaseAutoGeosteeringExecutor):
    """GPU-accelerated AutoGeosteering executor using EvoTorch SNES

    Architecture: Same as PythonAutoGeosteeringExecutor
    - ONE full interpretation stored in self.interpretation
    - INIT: manual -> truncate -> add new segments -> optimize -> self.interpretation
    - STEP: self.interpretation -> truncate -> add new segments -> optimize -> self.interpretation

    Optimization: EvoTorch SNES (Separable Natural Evolution Strategy)
    - Multiple restarts for reliability
    - GPU-accelerated batch evaluation
    """

    def __init__(self, work_dir: str, results_dir: str):
        """
        Initialize GPU AutoGeosteering executor

        Args:
            work_dir: Working directory for operations
            results_dir: Directory for results and logs
        """
        super().__init__(work_dir)

        # Load configuration from environment
        self.segments_count = int(os.getenv('PYTHON_SEGMENTS_COUNT', '4'))
        self.lookback_distance = float(os.getenv('PYTHON_LOOKBACK_DISTANCE', '50.0'))
        self.angle_range = float(os.getenv('PYTHON_ANGLE_RANGE', '10.0'))

        # Segment length constraints (in meters)
        self.max_segment_length = float(os.getenv('PYTHON_MAX_SEGMENT_LENGTH', '15.0'))
        self.min_optimization_length = float(os.getenv('PYTHON_MIN_OPTIMIZATION_LENGTH', '15.0'))

        # MD Normalization - fixed planning horizon
        self.md_normalization_buffer = float(os.getenv('MD_NORMALIZATION_BUFFER', '3000.0'))
        self.fixed_md_range = None
        self.fixed_min_md = None

        # Optimization parameters (same as Python executor for compatibility)
        self.pearson_power = float(os.getenv('PYTHON_PEARSON_POWER', '2.0'))
        self.mse_power = float(os.getenv('PYTHON_MSE_POWER', '0.001'))
        self.num_intervals_self_correlation = int(os.getenv('PYTHON_NUM_INTERVALS_SC', '0'))
        self.sc_power = float(os.getenv('PYTHON_SC_POWER', '1.15'))
        self.min_pearson_value = float(os.getenv('PYTHON_MIN_PEARSON_VALUE', '-1.0'))

        # GPU-specific parameters
        self.n_restarts = int(os.getenv('GPU_N_RESTARTS', '5'))
        self.popsize = int(os.getenv('GPU_POPSIZE', '100'))
        self.maxiter = int(os.getenv('GPU_MAXITER', '200'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # PseudoTypeLog support
        self.use_pseudo_typelog = os.getenv('USE_PSEUDOTYPELOG', 'false').lower() == 'true'

        # State management
        self.interpretation = None
        self.ag_well = None
        self.ag_typewell = None
        self.tvd_to_typewell_shift = None
        self.well_name = None
        self.start_idx = None
        self.start_md_meters = None

        # Logging
        self.optimization_logger = OptimizationLogger(results_dir)
        self.interpretation_dir = work_dir
        Path(self.interpretation_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"GPU executor initialized: device={self.device}, "
                    f"restarts={self.n_restarts}, popsize={self.popsize}, maxiter={self.maxiter}, "
                    f"use_pseudo={self.use_pseudo_typelog}")

    def start_daemon(self):
        """Start GPU executor"""
        logger.info(f"GPU AutoGeosteering executor ready (device={self.device})")
        if self.device == 'cuda':
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    def stop_daemon(self):
        """Stop GPU executor and print statistics"""
        logger.info("Stopping GPU AutoGeosteering executor")
        self.optimization_logger.print_statistics()

    def _create_typewell(self, well_data: Dict[str, Any]) -> TypeWell:
        """Create TypeWell, optionally using pseudoTypeLog"""
        if self.use_pseudo_typelog and 'pseudoTypeLog' in well_data:
            logger.info("Using pseudoTypeLog instead of typeLog (tvdTypewellShift=0)")
            well_data_for_typewell = dict(well_data)
            well_data_for_typewell['typeLog'] = well_data['pseudoTypeLog']
            return TypeWell(well_data_for_typewell)
        else:
            if self.use_pseudo_typelog:
                logger.warning("USE_PSEUDOTYPELOG=true but pseudoTypeLog not found, using typeLog")
            return TypeWell(well_data)

    def _get_tvd_shift(self, well_data: Dict[str, Any]) -> float:
        """Get TVD shift (0 for pseudoTypeLog)"""
        if self.use_pseudo_typelog and 'pseudoTypeLog' in well_data:
            return 0.0
        return well_data.get('tvdTypewellShift', 0.0)

    def _normalize_manual_to_segments(self, manual_json: List[Dict], well: Well) -> List:
        """Convert manual interpretation JSON to normalized Segment objects"""
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

        logger.debug(f"Converted {len(manual_json)} manual JSON -> {len(segments)} normalized Segments")
        return segments

    def _truncate_interpretation_at_md(self, truncate_md: float, well: Well) -> float:
        """Truncate self.interpretation at given MD, return interpolated shift"""
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
        """Create new segments and append to self.interpretation"""
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
        """Get indices of segments to optimize (last num_new_segments)"""
        if not self.interpretation or num_new_segments <= 0:
            return []
        total = len(self.interpretation)
        return list(range(total - num_new_segments, total))

    def _create_wrapper_and_bounds(self, segments: List, self_corr_start_idx: int):
        """Create TorchObjectiveWrapper and bounds for optimization"""
        bounds = calculate_optimization_bounds(segments, angle_range=self.angle_range, accumulative=True)
        bounds_tensor = torch.tensor(bounds, dtype=torch.float64, device=self.device)

        # Calculate horizontal projection
        self.ag_well.calc_horizontal_projection(self.ag_typewell, segments, self.tvd_to_typewell_shift)

        # Convert to numpy then torch
        well_np = well_to_numpy(self.ag_well)
        typewell_np = typewell_to_numpy(self.ag_typewell)
        segments_np = segments_to_numpy(segments, self.ag_well)

        well_torch = numpy_to_torch(well_np, device=self.device)
        typewell_torch = numpy_to_torch(typewell_np, device=self.device)
        segments_torch = segments_numpy_to_torch(segments_np, device=self.device)

        wrapper = TorchObjectiveWrapper(
            well_data=well_torch,
            typewell_data=typewell_torch,
            segments_torch=segments_torch,
            self_corr_start_idx=self_corr_start_idx,
            pearson_power=self.pearson_power,
            mse_power=self.mse_power,
            num_intervals_self_correlation=self.num_intervals_self_correlation,
            sc_power=self.sc_power,
            angle_range=self.angle_range,
            angle_sum_power=2.0,
            min_pearson_value=self.min_pearson_value,
            tvd_to_typewell_shift=self.tvd_to_typewell_shift,
            device=self.device
        )

        return wrapper, bounds_tensor

    def _optimize_with_snes(self, segments: List, self_corr_start_idx: int) -> tuple:
        """Run EvoTorch SNES optimization

        Returns:
            (best_fun, best_shifts, elapsed_time)
        """
        wrapper, bounds = self._create_wrapper_and_bounds(segments, self_corr_start_idx)

        K = len(segments)
        lb = bounds[:, 0].cpu().numpy()
        ub = bounds[:, 1].cpu().numpy()
        stdev_init = (ub - lb).mean() / 4.0

        # Define EvoTorch Problem
        class AGProblem(Problem):
            def __init__(inner_self):
                super().__init__(
                    objective_sense="min",
                    solution_length=K,
                    initial_bounds=(lb, ub),
                    dtype=torch.float64,
                    device=self.device
                )

            def _evaluate_batch(inner_self, solutions):
                x = solutions.values
                fitness = wrapper(x)
                solutions.set_evals(fitness)

        start_time = time.time()
        best_overall_fun = float('inf')
        best_overall_x = None

        for restart in range(self.n_restarts):
            problem = AGProblem()
            searcher = SNES(
                problem,
                popsize=self.popsize,
                stdev_init=stdev_init,
                center_learning_rate=0.5,
                stdev_learning_rate=0.1
            )

            for _ in range(self.maxiter):
                searcher.step()

            pop = searcher.population
            if pop is not None and hasattr(pop, 'evals') and pop.evals is not None:
                valid_mask = ~torch.isinf(pop.evals)
                if valid_mask.any():
                    best_idx = pop.evals.argmin()
                    best_fun = pop.evals[best_idx].item()
                    best_x = pop.values[best_idx]
                    if best_fun < best_overall_fun:
                        best_overall_fun = best_fun
                        best_overall_x = best_x.clone()

        elapsed = time.time() - start_time

        if best_overall_x is None:
            # Fallback to center of bounds
            best_overall_x = torch.tensor((lb + ub) / 2, dtype=torch.float64, device=self.device)
            best_overall_fun = wrapper(best_overall_x.unsqueeze(0)).item()

        return best_overall_fun, best_overall_x.cpu().tolist(), elapsed

    def _apply_optimized_shifts(self, segments: List, optimal_shifts: List) -> List:
        """Apply optimized shifts to segments"""
        optimized_segments = deepcopy(segments)
        for i, shift in enumerate(optimal_shifts):
            optimized_segments[i].end_shift = shift
            if i < len(optimized_segments) - 1:
                optimized_segments[i + 1].start_shift = shift
        return optimized_segments

    def _denormalize_segments_to_json(self, segments) -> List[Dict[str, Any]]:
        """Denormalize segment shifts from normalized to meters"""
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
        """Save FULL interpretation to file for StarSteer export"""
        if not self.well_name or not self.ag_well or not self.interpretation:
            return

        import json

        filepath = Path(self.interpretation_dir) / f"{self.well_name}.json"
        json_segments = self._denormalize_segments_to_json(self.interpretation)

        full_result = {
            'interpretation': {
                'segments': json_segments
            }
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_result, f, indent=2)

        if json_segments:
            first_md = json_segments[0]['startMd']
            last_md = json_segments[-1].get('endMd', json_segments[-1]['startMd'])
            logger.info(f"Saved interpretation: {len(json_segments)} segments, "
                        f"MD range: {first_md:.1f} - {last_md:.1f} m")

    def initialize_well(self, well_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize well with first dataset using GPU optimization

        Args:
            well_data: Well JSON data for initialization

        Returns:
            Dict containing interpretation result
        """
        logger.info("Initializing well with GPU executor (EvoTorch SNES)")

        self.well_name = well_data['wellName']

        # Create AG objects (with pseudoTypeLog support)
        self.ag_well = Well(well_data)
        self.ag_typewell = self._create_typewell(well_data)
        self.tvd_to_typewell_shift = self._get_tvd_shift(well_data)

        # Get start MD BEFORE normalization
        start_md_meters = well_data['autoGeosteeringParameters']['startMd']
        self.start_idx = self.ag_well.md2idx(start_md_meters)
        self.start_md_meters = start_md_meters

        manual_segments_json = well_data.get('interpretation', {}).get('segments', [])

        # Calculate fixed planning horizon
        self.fixed_min_md = self.ag_well.min_md
        self.fixed_md_range = (self.ag_well.max_md + self.md_normalization_buffer) - self.fixed_min_md
        logger.info(f"Fixed planning horizon: min_md={self.fixed_min_md:.2f}, "
                    f"fixed_md_range={self.fixed_md_range:.2f}")

        # Normalize with fixed md_range
        max_curve_value = max(self.ag_well.max_curve, self.ag_typewell.value.max())
        self.ag_well.normalize(max_curve_value, self.ag_typewell.min_depth, self.fixed_md_range)
        self.ag_typewell.normalize(max_curve_value, self.ag_well.min_depth, self.fixed_md_range)

        # Normalize TVD shift
        self.tvd_to_typewell_shift = self.tvd_to_typewell_shift / self.fixed_md_range

        # STEP 1: Convert manual to Segments
        self.interpretation = self._normalize_manual_to_segments(manual_segments_json, self.ag_well)
        logger.info(f"Loaded {len(self.interpretation)} manual segments")

        # STEP 2: Truncate at optimization start point
        start_md_norm = self.ag_well.measured_depth[self.start_idx]
        stitch_shift = self._truncate_interpretation_at_md(start_md_norm, self.ag_well)
        prefix_count = len(self.interpretation)

        # STEP 3: Add new segments
        current_idx = len(self.ag_well.measured_depth) - 1
        num_new = self._create_and_append_segments(
            start_idx=self.start_idx,
            end_idx=current_idx,
            start_shift=stitch_shift,
            well=self.ag_well
        )

        if num_new == 0:
            logger.warning("No new segments created")
            json_segments = self._denormalize_segments_to_json(self.interpretation)
            return {'interpretation': {'segments': json_segments}}

        # STEP 4: Optimize with EvoTorch SNES
        optimize_indices = self._get_optimization_indices(num_new)
        segments_to_optimize = [self.interpretation[i] for i in optimize_indices]

        best_fun, optimal_shifts, elapsed = self._optimize_with_snes(
            segments_to_optimize, self.start_idx
        )

        # Apply optimized shifts
        optimized_segments = self._apply_optimized_shifts(segments_to_optimize, optimal_shifts)
        for i, opt_seg in zip(optimize_indices, optimized_segments):
            self.interpretation[i] = opt_seg

        # Log
        self._log_optimization_result('init', self.ag_well.measured_depth[-1] * self.fixed_md_range + self.fixed_min_md,
                                      best_fun, elapsed)

        # STEP 5: Save
        json_segments = self._denormalize_segments_to_json(self.interpretation)
        result = {'interpretation': {'segments': json_segments}}
        self._save_interpretation_file()

        logger.info(f"INIT complete: {prefix_count} prefix + {num_new} new = "
                    f"{len(self.interpretation)} total, fun={best_fun:.4f}, time={elapsed:.2f}s")

        return result

    def update_well_data(self, well_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update well with new data using lookback optimization

        Args:
            well_data: Updated well JSON data

        Returns:
            Dict containing updated interpretation result
        """
        assert self.interpretation is not None, "Well not initialized"

        updated_well = Well(well_data)
        max_curve_value = max(updated_well.max_curve, self.ag_typewell.value.max())
        updated_well.normalize(max_curve_value, self.ag_typewell.min_depth, self.fixed_md_range)
        self.ag_well = updated_well

        current_md = updated_well.measured_depth[-1]
        current_idx = len(updated_well.measured_depth) - 1

        # Calculate lookback
        lookback_md = current_md - self.lookback_distance / self.fixed_md_range
        lookback_idx = updated_well.md2idx(lookback_md)

        if lookback_idx < self.start_idx:
            lookback_idx = self.start_idx

        lookback_md_value = updated_well.measured_depth[lookback_idx]

        # STEP 2: Truncate
        stitch_shift = self._truncate_interpretation_at_md(lookback_md_value, updated_well)
        prefix_count = len(self.interpretation)

        # STEP 3: Add new segments
        num_new = self._create_and_append_segments(
            start_idx=lookback_idx,
            end_idx=current_idx,
            start_shift=stitch_shift,
            well=updated_well
        )

        if num_new == 0:
            json_segments = self._denormalize_segments_to_json(self.interpretation)
            return {'interpretation': {'segments': json_segments}}

        # STEP 4: Optimize
        optimize_indices = self._get_optimization_indices(num_new)
        segments_to_optimize = [self.interpretation[i] for i in optimize_indices]

        best_fun, optimal_shifts, elapsed = self._optimize_with_snes(
            segments_to_optimize, lookback_idx
        )

        optimized_segments = self._apply_optimized_shifts(segments_to_optimize, optimal_shifts)
        for i, opt_seg in zip(optimize_indices, optimized_segments):
            self.interpretation[i] = opt_seg

        # Log
        self._log_optimization_result('step', current_md * self.fixed_md_range + self.fixed_min_md,
                                      best_fun, elapsed)

        # STEP 5: Save
        json_segments = self._denormalize_segments_to_json(self.interpretation)
        result = {'interpretation': {'segments': json_segments}}
        self._save_interpretation_file()

        logger.info(f"STEP complete: {prefix_count} prefix + {num_new} new = "
                    f"{len(self.interpretation)} total, fun={best_fun:.4f}, time={elapsed:.2f}s")

        return result

    def get_interpretation_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract interpretation from GPU executor result"""
        return result

    def _log_optimization_result(self, step_type: str, measured_depth: float,
                                 best_fun: float, elapsed: float):
        """Log optimization result to statistics"""
        optimization_stats = {
            'method': 'EvoTorch_SNES',
            'segments_count': self.segments_count,
            'n_restarts': self.n_restarts,
            'popsize': self.popsize,
            'maxiter': self.maxiter,
            'final_fun': best_fun,
            'elapsed_time': elapsed,
            'device': self.device,
            'success': True,
            # Fields required by optimization_logger
            'n_iterations': self.n_restarts * self.maxiter,
            'n_function_evaluations': self.n_restarts * self.maxiter * self.popsize,
            'final_correlation': 1.0 - best_fun,  # approximate
            'pearson_correlation': 0.0,  # not calculated in GPU version
            'mse_value': best_fun,
            'self_correlation': 0.0,  # not calculated in GPU version
            'intersections_count': 0  # not checked in GPU version
        }

        optimization_result = {
            'well_name': self.well_name,
            'measured_depth': measured_depth,
            'step_type': step_type,
            'optimization_stats': optimization_stats
        }

        logger.info(f"Optimization [{step_type}]: fun={best_fun:.4f}, time={elapsed:.2f}s, "
                    f"device={self.device}")

        self.optimization_logger.log_optimization_result(optimization_result)

    def _copy_output_to_comparison_dir(self, interpretation_data: Dict[str, Any], well_name: str, step_type: str, current_md: float):
        """Copy OUTPUT interpretation JSON to comparison directory for debugging

        GPU Executor version - includes EvoTorch SNES settings in output
        """
        import json
        from pathlib import Path

        comparison_dir = os.getenv('JSON_COMPARISON_DIR')
        if not comparison_dir:
            return

        comparison_path = Path(comparison_dir)
        comparison_path.mkdir(exist_ok=True, parents=True)

        version_tag = os.getenv('VERSION_TAG', 'evotorch_snes')

        # Format: {version}_{wellname}_{step}_{md}_output.json
        filename = f"{version_tag}_{well_name}_{step_type}_{current_md:.1f}_output.json"
        comparison_file = comparison_path / filename

        # Add executor settings to output for traceability
        output_data = {
            'executor': 'GpuAutoGeosteeringExecutor',
            'method': 'EvoTorch_SNES',
            'settings': {
                'n_restarts': self.n_restarts,
                'popsize': self.popsize,
                'maxiter': self.maxiter,
                'device': self.device,
                'segments_count': self.segments_count,
                'lookback_distance': self.lookback_distance
            },
            'interpretation': interpretation_data
        }

        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved EvoTorch SNES output to comparison: {filename}")
