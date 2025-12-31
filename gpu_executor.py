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
from ag_objects.ag_obj_interpretation import create_segments, create_telescope_segments, Segment
from ag_numerical.ag_optimizer_utils import calculate_optimization_bounds, calculate_angle_bounds, angles_to_shifts

# Torch imports
from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
from torch_funcs.batch_objective import TorchObjectiveWrapper, normalized_angles_to_shifts_torch

# EvoTorch
from evotorch import Problem
from evotorch.algorithms import SNES, CMAES, XNES, GeneticAlgorithm
from evotorch.operators import OnePointCrossOver, TwoPointCrossOver, SimulatedBinaryCrossOver, GaussianMutation
import pyswarms as ps
from scipy.optimize import differential_evolution
from gr_utils import apply_gr_smoothing

logger = logging.getLogger(__name__)


def generate_population_centers(K: int, dip_range: float, step: float = 1.0) -> np.ndarray:
    """
    Generate initial centers for multi-population optimization.

    Strategy:
    - Point 1: 0° (from frozen segment)
    - Point K: varies from -dip_range to +dip_range with step
    - Points 2..K-1: bend (symmetric) + asymmetry

    Args:
        K: Number of segments (parameters)
        dip_range: DIP_ANGLE_RANGE from .env
        step: Step size in degrees (default 1.0)

    Returns:
        np.ndarray of shape (N_centers, K) with angles in degrees
    """
    centers = []

    # End angles: -dip_range to +dip_range, step
    end_angles = np.arange(-dip_range, dip_range + step/2, step)

    # Bend values: -dip_range to +dip_range, step
    bends = np.arange(-dip_range, dip_range + step/2, step)

    # Asymmetry: 0, +1/4, -1/4 of dip_range
    asymm_delta = dip_range / 4.0
    asymmetries = [0.0, asymm_delta, -asymm_delta]

    for end_angle in end_angles:
        for bend in bends:
            for asymm in asymmetries:
                # Base line from 0 to end_angle
                center = np.linspace(0, end_angle, K)

                if K >= 4:
                    # Apply bend and asymmetry to middle points
                    # For K=4: points 1,2 are middle (indices 1,2)
                    mid_start = 1
                    mid_end = K - 1
                    n_mid = mid_end - mid_start

                    if n_mid == 2:
                        # K=4: two middle points
                        center[1] += bend + asymm
                        center[2] += bend - asymm
                    elif n_mid > 2:
                        # K>4: distribute bend across middle points
                        for i in range(mid_start, mid_end):
                            # Asymmetry affects first and last middle points
                            if i == mid_start:
                                center[i] += bend + asymm
                            elif i == mid_end - 1:
                                center[i] += bend - asymm
                            else:
                                center[i] += bend

                # Clip to valid range
                center = np.clip(center, -dip_range, dip_range)
                centers.append(center)

    centers = np.array(centers)
    logger.debug(f"Generated {len(centers)} population centers for K={K}, dip_range={dip_range}°, step={step}°")
    return centers


def get_reference_shifts_for_segments(ref_segments, segment_end_mds_m, md_range, first_start_md_m=None):
    """
    Interpolate reference interpretation shifts to optimization segment end points.

    Args:
        ref_segments: List of reference segment dicts (starredInterpretation or referenceInterpretation)
        segment_end_mds_m: List of end MD values for optimization segments (meters)
        md_range: MD range for normalization
        first_start_md_m: MD of first segment start (to get first_start_shift)

    Returns:
        (ref_shifts_m, ref_shifts_norm, first_start_shift_norm) - shifts in meters, normalized, and first start shift
    """
    ref_shifts_m = []
    first_start_shift_m = 0.0

    # Get first_start_shift if first_start_md provided
    if first_start_md_m is not None:
        for i, seg in enumerate(ref_segments):
            seg_start = seg.get('startMd', 0)
            seg_start_shift = seg.get('startShift', 0)
            seg_end_shift = seg.get('endShift', 0)

            if 'endMd' in seg:
                seg_end = seg['endMd']
            elif i + 1 < len(ref_segments):
                seg_end = ref_segments[i + 1].get('startMd', seg_start)
            else:
                seg_end = seg_start + 1000

            if seg_start <= first_start_md_m <= seg_end:
                if seg_end > seg_start:
                    ratio = (first_start_md_m - seg_start) / (seg_end - seg_start)
                    first_start_shift_m = seg_start_shift + ratio * (seg_end_shift - seg_start_shift)
                else:
                    first_start_shift_m = seg_start_shift
                break
            elif first_start_md_m < seg_start:
                first_start_shift_m = seg_start_shift
                break
        else:
            if ref_segments:
                first_start_shift_m = ref_segments[-1].get('endShift', 0)

    for end_md in segment_end_mds_m:
        shift = 0.0
        for i, seg in enumerate(ref_segments):
            seg_start = seg.get('startMd', 0)
            seg_start_shift = seg.get('startShift', 0)
            seg_end_shift = seg.get('endShift', 0)

            # Get end_md
            if 'endMd' in seg:
                seg_end = seg['endMd']
            elif i + 1 < len(ref_segments):
                seg_end = ref_segments[i + 1].get('startMd', seg_start)
            else:
                seg_end = seg_start + 1000

            # Check if end_md is within this segment
            if seg_start <= end_md <= seg_end:
                if seg_end > seg_start:
                    ratio = (end_md - seg_start) / (seg_end - seg_start)
                    shift = seg_start_shift + ratio * (seg_end_shift - seg_start_shift)
                else:
                    shift = seg_start_shift
                break
            elif end_md < seg_start:
                shift = seg_start_shift
                break
        else:
            if ref_segments:
                shift = ref_segments[-1].get('endShift', 0)

        ref_shifts_m.append(shift)

    # Normalize
    ref_shifts_norm = [s / md_range for s in ref_shifts_m]
    first_start_shift_norm = first_start_shift_m / md_range
    return ref_shifts_m, ref_shifts_norm, first_start_shift_norm


def build_reference_segments_torch(ref_segments_json, well, md_start_m, md_end_m, md_range, min_md, device='cpu'):
    """
    Build reference segments with correct geometry for MD range.

    Args:
        ref_segments_json: List of reference segment dicts
        well: Normalized Well object
        md_start_m: Start MD in meters
        md_end_m: End MD in meters
        md_range: MD range for normalization
        min_md: Min MD for normalization
        device: torch device

    Returns:
        segments_torch: (K, 6) tensor with reference segments
    """
    segments = []

    for i, seg in enumerate(ref_segments_json):
        seg_start_md = seg.get('startMd', 0)
        seg_start_shift = seg.get('startShift', 0)
        seg_end_shift = seg.get('endShift', 0)

        if 'endMd' in seg:
            seg_end_md = seg['endMd']
        elif i + 1 < len(ref_segments_json):
            seg_end_md = ref_segments_json[i + 1].get('startMd', seg_start_md)
        else:
            continue

        # Skip if entirely outside range
        if seg_end_md < md_start_m or seg_start_md > md_end_m:
            continue

        # Clip to range
        clipped_start_md = max(seg_start_md, md_start_m)
        clipped_end_md = min(seg_end_md, md_end_m)

        if clipped_end_md - clipped_start_md < 0.1:
            continue

        # Interpolate shifts for clipped boundaries
        seg_len = seg_end_md - seg_start_md
        if seg_len > 0:
            if clipped_start_md > seg_start_md:
                frac = (clipped_start_md - seg_start_md) / seg_len
                clipped_start_shift = seg_start_shift + (seg_end_shift - seg_start_shift) * frac
            else:
                clipped_start_shift = seg_start_shift

            if clipped_end_md < seg_end_md:
                frac = (clipped_end_md - seg_start_md) / seg_len
                clipped_end_shift = seg_start_shift + (seg_end_shift - seg_start_shift) * frac
            else:
                clipped_end_shift = seg_end_shift
        else:
            clipped_start_shift = seg_start_shift
            clipped_end_shift = seg_end_shift

        # Find indices in well
        # well.measured_depth may be normalized (0-1) or absolute depending on well.normalized
        if well.normalized:
            # Convert absolute MD to normalized for search
            search_start = (clipped_start_md - min_md) / md_range
            search_end = (clipped_end_md - min_md) / md_range
        else:
            search_start = clipped_start_md
            search_end = clipped_end_md

        start_idx = int(np.searchsorted(well.measured_depth, search_start))
        end_idx = int(np.searchsorted(well.measured_depth, search_end))

        start_idx = min(start_idx, len(well.measured_depth) - 1)
        end_idx = min(end_idx, len(well.measured_depth) - 1)

        if start_idx >= end_idx:
            continue

        # Get VS from well at indices
        start_vs = well.vs_thl[start_idx]
        end_vs = well.vs_thl[end_idx]

        # Normalize VS and shifts if well is not already normalized
        if well.normalized:
            # VS already normalized
            start_vs_norm = start_vs
            end_vs_norm = end_vs
        else:
            start_vs_norm = (start_vs - well.min_vs) / md_range
            end_vs_norm = (end_vs - well.min_vs) / md_range

        # Shifts always come from reference (absolute meters), need normalization
        start_shift_norm = clipped_start_shift / md_range
        end_shift_norm = clipped_end_shift / md_range

        segments.append([start_idx, end_idx, start_vs_norm, end_vs_norm, start_shift_norm, end_shift_norm])

    if not segments:
        return None

    return torch.tensor(segments, dtype=torch.float64, device=device)


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
        self.mse_power = float(os.getenv('PYTHON_MSE_POWER', '2'))
        self.num_intervals_self_correlation = int(os.getenv('PYTHON_NUM_INTERVALS_SC', '0'))
        self.sc_power = float(os.getenv('PYTHON_SC_POWER', '1.15'))
        self.min_pearson_value = float(os.getenv('PYTHON_MIN_PEARSON_VALUE', '-1.0'))

        # CMA-ES stability parameters (prevent covariance matrix degeneration)
        self.min_angle_diff = float(os.getenv('PYTHON_MIN_ANGLE_DIFF', '0.2'))
        self.min_trend_deviation = float(os.getenv('PYTHON_MIN_TREND_DEVIATION', '0.5'))
        self.trend_power = float(os.getenv('PYTHON_TREND_POWER', '1.0'))

        # GPU-specific parameters
        self.n_restarts = int(os.getenv('GPU_N_RESTARTS', '5'))
        self.popsize = int(os.getenv('GPU_POPSIZE', '100'))
        self.maxiter = int(os.getenv('GPU_MAXITER', '200'))
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.algorithm = os.getenv('GPU_ALGORITHM', 'CMAES').upper()  # CMAES, MONTECARLO, SNES, DE
        self.angle_grid_step = float(os.getenv('GPU_ANGLE_GRID_STEP', '1.0'))  # Angle grid step for center generation (degrees)
        self.cmaes_mode = os.getenv('GPU_CMAES_MODE', 'multi').lower()  # 'single' or 'multi'
        self.single_popsize = int(os.getenv('GPU_SINGLE_POPSIZE', '500'))  # popsize for single mode
        self.montecarlo_samples = int(os.getenv('GPU_MONTECARLO_SAMPLES', '100000'))  # samples per iteration
        self.montecarlo_iterations = int(os.getenv('GPU_MONTECARLO_ITERATIONS', '1'))  # number of iterations
        self.center_lr = float(os.getenv('GPU_CENTER_LR', '0.5'))  # center learning rate for SNES/XNES
        self.stdev_lr = float(os.getenv('GPU_STDEV_LR', '0.1'))  # stdev learning rate for SNES/XNES

        # GA parameters
        self.ga_popsize = int(os.getenv('GPU_GA_POPSIZE', '100'))
        self.ga_mutation_std = float(os.getenv('GPU_GA_MUTATION_STD', '0.1'))
        self.ga_crossover = os.getenv('GPU_GA_CROSSOVER', 'OnePoint')  # OnePoint, TwoPoint, SBX
        self.ga_tournament_size = int(os.getenv('GPU_GA_TOURNAMENT_SIZE', '4'))
        self.ga_elitist = os.getenv('GPU_GA_ELITIST', 'true').lower() == 'true'

        # PSO parameters (pyswarms - CPU)
        self.pso_particles = int(os.getenv('GPU_PSO_PARTICLES', '50'))
        self.pso_iterations = int(os.getenv('GPU_PSO_ITERATIONS', '200'))
        self.pso_c1 = float(os.getenv('GPU_PSO_C1', '2.0'))  # cognitive parameter
        self.pso_c2 = float(os.getenv('GPU_PSO_C2', '2.0'))  # social parameter
        self.pso_w = float(os.getenv('GPU_PSO_W', '0.7'))    # inertia weight

        # PseudoTypeLog support
        self.use_pseudo_typelog = os.getenv('USE_PSEUDOTYPELOG', 'false').lower() == 'true'

        # Telescope mode: skip first segment(s) in Pearson/MSE reward
        # TELESCOPE_REWARD_START_SEGMENT=1 means reward calculated from segment 1 (skipping segment 0)
        self.telescope_reward_start_segment = int(os.getenv('TELESCOPE_REWARD_START_SEGMENT', '0'))

        # Telescope segment creation: one long lever + multiple short work segments
        # TELESCOPE_MODE=true enables telescope segment creation
        self.telescope_mode = os.getenv('TELESCOPE_MODE', 'false').lower() == 'true'
        # TELESCOPE_LEVER_MD = absolute MD where lever ends (e.g. 5944 for ~19500ft)
        telescope_lever_md_str = os.getenv('TELESCOPE_LEVER_MD')
        self.telescope_lever_md = float(telescope_lever_md_str) if telescope_lever_md_str else None
        self.telescope_work_segment_length = float(os.getenv('TELESCOPE_WORK_SEGMENT_LENGTH', '30.0'))  # meters
        self.telescope_work_segments_count = int(os.getenv('TELESCOPE_WORK_SEGMENTS_COUNT', '4'))

        # Angle optimization mode: optimize angles instead of shifts
        # When enabled, CMA-ES optimizes normalized angles (0.1 = 1°), then converts to shifts
        self.angle_optimization_mode = os.getenv('ANGLE_OPTIMIZATION_MODE', 'false').lower() == 'true'
        self.angle_normalize_factor = float(os.getenv('ANGLE_NORMALIZE_FACTOR', '10.0'))

        # Telescope cheat mode: initialize lever angle from reference trajectory
        # CHEATING - uses reference data for initialization
        self.telescope_init_from_reference = os.getenv('TELESCOPE_INIT_FROM_REFERENCE', 'false').lower() == 'true'

        # State management
        self.interpretation = None
        self.ag_well = None
        self.ag_typewell = None
        self.ag_typewell_raw = None  # Unnormalized typewell for re-normalization on update
        self.tvd_to_typewell_shift = None
        self.well_name = None
        self.start_idx = None
        self.start_md_meters = None
        self.reference_segments = None  # For comparison with manual interpretation
        self.last_num_populations = 1  # For logging function evaluations

        # Logging
        self.optimization_logger = OptimizationLogger(results_dir)
        self.interpretation_dir = work_dir
        Path(self.interpretation_dir).mkdir(parents=True, exist_ok=True)

        telescope_info = ""
        if self.telescope_mode:
            telescope_info = f", TELESCOPE: lever_md={self.telescope_lever_md}, work={self.telescope_work_segment_length}m x {self.telescope_work_segments_count}"
        logger.info(f"GPU executor initialized: device={self.device}, algorithm={self.algorithm}, "
                    f"cmaes_mode={self.cmaes_mode}, single_popsize={self.single_popsize}, "
                    f"restarts={self.n_restarts}, popsize={self.popsize}, maxiter={self.maxiter}, "
                    f"use_pseudo={self.use_pseudo_typelog}, telescope_reward_start={self.telescope_reward_start_segment}{telescope_info}")

    def set_algorithm(self, algorithm: str):
        """Override algorithm from CLI parameter"""
        self.algorithm = algorithm.upper()
        logger.info(f"Algorithm overridden to: {self.algorithm}")

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
        """Create TypeWell from well_data.

        Note: TypeWell transformation (pseudo_stitched mode) is now handled
        by TypewellProvider in emulator_processor before data reaches executor.
        """
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

        use_telescope = False
        new_segments = None

        if self.telescope_mode and self.telescope_lever_md is not None:
            # Telescope mode: one long lever + multiple short work segments
            # Calculate lever_length from absolute MD
            # Note: well.measured_depth is normalized, convert back to absolute
            start_md_norm = well.measured_depth[start_idx]
            start_md_abs = start_md_norm * self.fixed_md_range + self.fixed_min_md
            lever_length_m = self.telescope_lever_md - start_md_abs
            if lever_length_m <= 0:
                logger.warning(f"Telescope: lever_md={self.telescope_lever_md} <= start_md={start_md_abs:.1f}, skipping telescope")
            else:
                lever_length_idx = int(lever_length_m / well.horizontal_well_step)
                work_segment_length_idx = int(self.telescope_work_segment_length / well.horizontal_well_step)

                # Get extrapolate_angle
                extrapolate_angle = None
                if self.telescope_init_from_reference:
                    # CHEAT MODE: calculate angle from well trajectory (TVD/VS)
                    lever_end_idx = min(start_idx + lever_length_idx, len(well.true_vertical_depth) - 1)
                    tvd_start = well.true_vertical_depth[start_idx]
                    tvd_end = well.true_vertical_depth[lever_end_idx]
                    vs_start = well.vs_thl[start_idx]
                    vs_end = well.vs_thl[lever_end_idx]
                    delta_tvd = tvd_end - tvd_start
                    delta_vs = vs_end - vs_start
                    if abs(delta_vs) > 0.001:
                        extrapolate_angle = np.degrees(np.arctan(delta_tvd / delta_vs))
                        logger.info(f"Telescope CHEAT: angle from well trajectory = {extrapolate_angle:.2f}° "
                                   f"(dTVD={delta_tvd:.2f}, dVS={delta_vs:.2f})")
                elif self.interpretation and len(self.interpretation) > 0:
                    # Normal: extrapolate from last segment
                    last_segment = self.interpretation[-1]
                    extrapolate_angle = last_segment.angle
                    logger.info(f"Telescope: extrapolating from last segment angle={extrapolate_angle:.2f}°")

                new_segments = create_telescope_segments(
                    well=well,
                    lever_length_idx=lever_length_idx,
                    work_segment_length_idx=work_segment_length_idx,
                    work_segments_count=self.telescope_work_segments_count,
                    start_idx=start_idx,
                    start_shift=start_shift,
                    extrapolate_angle=extrapolate_angle
                )
                if new_segments is not None:
                    use_telescope = True
                    logger.info(f"Telescope mode: lever={lever_length_m:.1f}m + "
                               f"{len(new_segments)-1} work segments @ {self.telescope_work_segment_length}m each")

        # Track current mode for reward calculation
        self._current_use_telescope = use_telescope
        self._current_extrapolate_angle = extrapolate_angle if use_telescope else None

        if not use_telescope:
            # Normal mode: uniform segments
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

    def _create_wrapper_and_bounds(self, segments: List, self_corr_start_idx: int, prev_segment_angle: float = None):
        """Create TorchObjectiveWrapper and bounds for optimization

        Args:
            segments: List of segments to optimize
            self_corr_start_idx: Starting index for self-correlation
            prev_segment_angle: Angle of last frozen segment (degrees), used for angle_sum_penalty
        """
        if self.angle_optimization_mode:
            # Angle mode: bounds for normalized angles
            # Use extrapolate_angle as center if available (cheat mode)
            center_angles = getattr(self, '_current_extrapolate_angle', None)
            bounds = calculate_angle_bounds(segments, self.angle_range, self.angle_normalize_factor, center_angles)
            if center_angles is not None:
                logger.info(f"  ANGLE OPTIMIZATION MODE: bounds centered at {center_angles:.2f}° (cheat)")
            else:
                logger.info(f"  ANGLE OPTIMIZATION MODE: bounds for normalized angles (factor={self.angle_normalize_factor})")
        else:
            # Shift mode: bounds for end_shifts
            bounds = calculate_optimization_bounds(segments, angle_range=self.angle_range, accumulative=True)
        bounds_tensor = torch.tensor(bounds, dtype=torch.float64, device=self.device)

        # Calculate horizontal projection
        self.ag_well.calc_horizontal_projection(self.ag_typewell, segments, self.tvd_to_typewell_shift)

        # Convert to numpy then torch
        well_np = well_to_numpy(self.ag_well)

        # Apply GR smoothing (uses settings from .env via gr_utils)
        well_np['value'] = apply_gr_smoothing(well_np['value'])

        # DUMP after smoothing for comparison
        dump_smoothed_path = os.getenv('DUMP_SMOOTHED_PATH')
        if dump_smoothed_path:
            import json
            dump_data = {
                'gr_smoothed_range': [float(well_np['value'].min()), float(well_np['value'].max())],
                'gr_smoothed_len': len(well_np['value']),
            }
            with open(dump_smoothed_path, 'w') as f:
                json.dump(dump_data, f, indent=2)
            logger.info(f"DUMP smoothed: {dump_data}")

        typewell_np = typewell_to_numpy(self.ag_typewell)
        segments_np = segments_to_numpy(segments, self.ag_well)

        well_torch = numpy_to_torch(well_np, device=self.device)
        typewell_torch = numpy_to_torch(typewell_np, device=self.device)
        segments_torch = segments_numpy_to_torch(segments_np, device=self.device)

        # Determine reward_start_segment_idx: 1 for telescope mode, 0 for normal
        current_reward_start = self.telescope_reward_start_segment if getattr(self, '_current_use_telescope', False) else 0

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
            angle_sum_power=float(os.getenv('PYTHON_ANGLE_SUM_POWER', '2.0')),
            min_pearson_value=self.min_pearson_value,
            tvd_to_typewell_shift=self.tvd_to_typewell_shift,
            prev_segment_angle=prev_segment_angle,
            device=self.device,
            min_angle_diff=self.min_angle_diff,
            min_trend_deviation=self.min_trend_deviation,
            trend_power=self.trend_power,
            reward_start_segment_idx=current_reward_start
        )

        return wrapper, bounds_tensor

    def _optimize_with_snes(self, segments: List, self_corr_start_idx: int, prev_segment_angle: float = None) -> tuple:
        """Run EvoTorch SNES optimization

        Args:
            segments: List of segments to optimize
            self_corr_start_idx: Starting index for self-correlation
            prev_segment_angle: Angle of last frozen segment (degrees)

        Returns:
            (best_fun, best_shifts, elapsed_time, wrapper)
        """
        wrapper, bounds = self._create_wrapper_and_bounds(segments, self_corr_start_idx, prev_segment_angle)

        K = len(segments)
        lb = bounds[:, 0].cpu().numpy()
        ub = bounds[:, 1].cpu().numpy()
        # DEBUG: Log bounds
        if self.angle_optimization_mode:
            # Bounds are normalized angles
            lb_deg = lb * self.angle_normalize_factor
            ub_deg = ub * self.angle_normalize_factor
            logger.info(f"  DEBUG bounds (degrees): lb={[f'{x:.2f}' for x in lb_deg]}, ub={[f'{x:.2f}' for x in ub_deg]}")
        else:
            lb_m = lb * self.fixed_md_range
            ub_m = ub * self.fixed_md_range
            logger.info(f"  DEBUG bounds (meters): lb={[f'{x:.2f}' for x in lb_m]}, ub={[f'{x:.2f}' for x in ub_m]}")
        stdev_init = (ub - lb).mean() / 2.0  # was /4.0 - wider initial distribution

        # Capture for nested class
        angle_mode = self.angle_optimization_mode
        normalize_factor = self.angle_normalize_factor
        segments_torch = wrapper.segments_torch

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
                if angle_mode:
                    # Convert normalized angles to shifts
                    shifts = normalized_angles_to_shifts_torch(x, segments_torch, normalize_factor)
                    fitness = wrapper(shifts)
                else:
                    fitness = wrapper(x)
                solutions.set_evals(fitness)

        start_time = time.time()
        best_overall_fun = float('inf')
        best_overall_x = None

        # Choose algorithm: SNES, DE, or CMAES
        if self.algorithm == 'DE':
            # Scipy DE with uniform distribution
            bounds_list = list(zip(lb, ub))

            def scipy_objective(x):
                x_tensor = torch.tensor(x, dtype=torch.float64, device=self.device).unsqueeze(0)
                if angle_mode:
                    shifts = normalized_angles_to_shifts_torch(x_tensor, segments_torch, normalize_factor)
                    return wrapper(shifts).item()
                return wrapper(x_tensor).item()

            result = differential_evolution(
                scipy_objective,
                bounds_list,
                maxiter=self.maxiter,
                popsize=max(15, self.popsize // 10),  # DE uses smaller popsize
                mutation=(0.5, 1.0),
                recombination=0.7,
                seed=None,
                polish=False,
                workers=1  # GPU wrapper not thread-safe
            )
            best_overall_fun = result.fun
            best_overall_x = torch.tensor(result.x, dtype=torch.float64, device=self.device)

        elif self.algorithm == 'CMAES':
            if self.cmaes_mode == 'single':
                # Single large CMA-ES with uniform initialization
                self.last_num_populations = 1
                total_evals = self.single_popsize * self.maxiter
                logger.info(f"CMAES single: popsize={self.single_popsize}, maxiter={self.maxiter}, total_evals={total_evals}")

                # Create problem with bounds for uniform initialization
                class AGProblemBounded(Problem):
                    def __init__(inner_self):
                        super().__init__(
                            objective_sense="min",
                            solution_length=K,
                            initial_bounds=(lb, ub),  # Uniform init within bounds
                            dtype=torch.float64,
                            device=self.device
                        )

                    def _evaluate_batch(inner_self, solutions):
                        x = solutions.values
                        if angle_mode:
                            shifts = normalized_angles_to_shifts_torch(x, segments_torch, normalize_factor)
                            fitness = wrapper(shifts)
                        else:
                            fitness = wrapper(x)
                        solutions.set_evals(fitness)

                problem = AGProblemBounded()
                # stdev_min prevents covariance matrix degeneration (0 = disabled for debugging)
                stdev_min_val = float(os.getenv('GPU_STDEV_MIN', '0'))
                searcher_kwargs = {
                    'stdev_init': stdev_init,
                    'popsize': self.single_popsize
                }
                if stdev_min_val > 0:
                    searcher_kwargs['stdev_min'] = stdev_min_val
                searcher = CMAES(problem, **searcher_kwargs)

                cmaes_converged_early = False
                try:
                    for iter_num in range(self.maxiter):
                        searcher.step()
                except Exception as e:
                    # Graceful recovery: save current best and continue
                    cmaes_converged_early = True
                    try:
                        stdev = searcher._sigma if hasattr(searcher, '_sigma') else 'N/A'
                        pop = searcher.population
                        if pop is not None and hasattr(pop, 'values') and pop.evals is not None:
                            best_idx = pop.evals.argmin()
                            best_overall_fun = pop.evals[best_idx].item()
                            best_overall_x = pop.values[best_idx].clone()
                            best_shifts_m = best_overall_x.cpu().numpy() * self.fixed_md_range
                            logger.warning(f"CMA-ES degenerated at iter {iter_num}/{self.maxiter}: "
                                          f"stdev={stdev}, fun={best_overall_fun:.6f}, "
                                          f"shifts_m={[f'{s:.2f}' for s in best_shifts_m]}")
                            logger.warning(f"Continuing with current best solution (graceful recovery)")
                        else:
                            raise RuntimeError(f"CMA-ES crashed at iter {iter_num}, no valid population")
                    except Exception as inner_e:
                        raise RuntimeError(f"CMA-ES crashed at iter {iter_num}, recovery failed: {inner_e}")

                # Only extract from population if we didn't recover from degeneration
                if not cmaes_converged_early:
                    pop = searcher.population
                    if pop is not None and hasattr(pop, 'evals') and pop.evals is not None:
                        valid_mask = ~torch.isinf(pop.evals)
                        if valid_mask.any():
                            best_idx = pop.evals.argmin()
                            best_overall_fun = pop.evals[best_idx].item()
                            best_overall_x = pop.values[best_idx].clone()

                if best_overall_x is None:
                    raise RuntimeError("CMA-ES single mode failed: no valid solution found")

            else:
                # CMA-ES with multi-population from generated centers (original approach)
                centers = generate_population_centers(K, self.angle_range, self.angle_grid_step)
                self.last_num_populations = len(centers)
                total_evals = len(centers) * self.popsize * self.maxiter
                logger.info(f"CMAES multi: {len(centers)} populations, popsize={self.popsize}, maxiter={self.maxiter}, total_evals={total_evals}")

                for i, center_angles in enumerate(centers):
                    # Convert angles to shifts
                    center_shifts = center_angles / self.angle_range * (ub - lb) / 2 + (ub + lb) / 2
                    center_shifts = np.clip(center_shifts, lb, ub)
                    center_init = torch.tensor(center_shifts, dtype=torch.float64, device=self.device)

                    problem = AGProblem()
                    stdev_min = float(os.getenv('GPU_STDEV_MIN', '0.001'))
                    searcher = CMAES(
                        problem,
                        stdev_init=stdev_init,
                        stdev_min=stdev_min,
                        popsize=self.popsize,
                        center_init=center_init
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

                if best_overall_x is None:
                    raise RuntimeError("CMA-ES multi mode failed: no valid solution found")

        elif self.algorithm == 'MONTECARLO':
            # Pure random search on GPU - multi-iteration uniform sampling
            samples = self.montecarlo_samples
            iterations = self.montecarlo_iterations
            total_samples = samples * iterations
            logger.info(f"MONTECARLO: {samples} samples x {iterations} iterations = {total_samples} total, bounds={K}D")

            lb_tensor = torch.tensor(lb, dtype=torch.float64, device=self.device)
            ub_tensor = torch.tensor(ub, dtype=torch.float64, device=self.device)

            best_overall_fun = float('inf')
            best_overall_x = None

            for it in range(iterations):
                # Generate uniform random samples within bounds
                random_samples = torch.rand(samples, K, dtype=torch.float64, device=self.device)
                random_samples = random_samples * (ub_tensor - lb_tensor) + lb_tensor

                # Evaluate all samples at once (GPU batch)
                if angle_mode:
                    shifts_samples = normalized_angles_to_shifts_torch(random_samples, segments_torch, normalize_factor)
                    fitness = wrapper(shifts_samples)
                else:
                    fitness = wrapper(random_samples)

                # Find best in this iteration
                best_idx = fitness.argmin()
                iter_best_fun = fitness[best_idx].item()

                if iter_best_fun < best_overall_fun:
                    best_overall_fun = iter_best_fun
                    best_overall_x = random_samples[best_idx].clone()

                # Cleanup GPU memory after each iteration
                del random_samples, fitness
                torch.cuda.empty_cache()

            del lb_tensor, ub_tensor
            self.last_num_populations = iterations

        elif self.algorithm == 'XNES':
            for restart in range(self.n_restarts):
                problem = AGProblem()
                searcher = XNES(
                    problem,
                    popsize=self.popsize,
                    stdev_init=stdev_init,
                    center_learning_rate=self.center_lr,
                    stdev_learning_rate=self.stdev_lr
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

        elif self.algorithm == 'GA':
            for restart in range(self.n_restarts):
                problem = AGProblem()

                # Create operators for problem
                if self.ga_crossover == 'TwoPoint':
                    crossover_op = TwoPointCrossOver(problem, tournament_size=self.ga_tournament_size)
                elif self.ga_crossover == 'SBX':
                    crossover_op = SimulatedBinaryCrossOver(problem, tournament_size=self.ga_tournament_size, eta=20)
                else:
                    crossover_op = OnePointCrossOver(problem, tournament_size=self.ga_tournament_size)
                mutation_op = GaussianMutation(problem, stdev=self.ga_mutation_std)

                searcher = GeneticAlgorithm(
                    problem,
                    popsize=self.ga_popsize,
                    operators=[crossover_op, mutation_op],
                    elitist=self.ga_elitist
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

        elif self.algorithm == 'PSO':
            # PSO using pyswarms (CPU-based)
            # Convert bounds to numpy
            lb_np = lb.cpu().numpy() if isinstance(lb, torch.Tensor) else lb
            ub_np = ub.cpu().numpy() if isinstance(ub, torch.Tensor) else ub
            bounds = (lb_np, ub_np)

            # Objective function for pyswarms (expects shape [n_particles, n_dims])
            def pso_objective(x):
                # x shape: [n_particles, n_dims]
                x_tensor = torch.tensor(x, dtype=torch.float64, device=self.device)
                if angle_mode:
                    shifts = normalized_angles_to_shifts_torch(x_tensor, segments_torch, normalize_factor)
                    costs = wrapper(shifts)
                else:
                    costs = wrapper(x_tensor)
                return costs.cpu().numpy()

            # PSO options
            options = {
                'c1': self.pso_c1,  # cognitive parameter
                'c2': self.pso_c2,  # social parameter
                'w': self.pso_w     # inertia weight
            }

            n_dims = len(lb_np)
            optimizer = ps.single.GlobalBestPSO(
                n_particles=self.pso_particles,
                dimensions=n_dims,
                options=options,
                bounds=bounds
            )

            best_cost, best_pos = optimizer.optimize(
                pso_objective,
                iters=self.pso_iterations,
                verbose=False
            )

            best_overall_fun = best_cost
            best_overall_x = torch.tensor(best_pos, dtype=torch.float64, device=self.device)

        else:  # SNES (default)
            for restart in range(self.n_restarts):
                problem = AGProblem()
                searcher = SNES(
                    problem,
                    popsize=self.popsize,
                    stdev_init=stdev_init,
                    center_learning_rate=self.center_lr,
                    stdev_learning_rate=self.stdev_lr
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

            if best_overall_x is None:
                raise RuntimeError("SNES failed: no valid solution found")

        elapsed = time.time() - start_time

        # Convert angles to shifts if in angle mode
        if angle_mode:
            angles_tensor = best_overall_x.unsqueeze(0)  # (1, K)
            shifts_tensor = normalized_angles_to_shifts_torch(angles_tensor, segments_torch, normalize_factor)
            optimal_shifts = shifts_tensor.squeeze(0).cpu().tolist()
            logger.info(f"  ANGLE MODE: angles={[f'{a * normalize_factor:.2f}°' for a in best_overall_x.cpu().tolist()]}")
        else:
            optimal_shifts = best_overall_x.cpu().tolist()

        return best_overall_fun, optimal_shifts, elapsed, wrapper

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
        self.ag_typewell_raw = self._create_typewell(well_data)  # Store raw for re-normalization
        self.tvd_to_typewell_shift = self._get_tvd_shift(well_data)

        # DUMP for comparison (before normalization)
        dump_path = os.getenv('DUMP_WELL_DATA_PATH')
        if dump_path:
            import json
            dump_data = {
                'well_name': self.well_name,
                'ag_well': {
                    'md_len': len(self.ag_well.measured_depth),
                    'md_range': [float(self.ag_well.min_md), float(self.ag_well.max_md)],
                    'gr_len': len(self.ag_well.value),
                    'gr_range': [float(self.ag_well.value.min()), float(self.ag_well.value.max())],
                    'step': float(self.ag_well.horizontal_well_step),
                },
                'ag_typewell': {
                    'tvd_len': len(self.ag_typewell_raw.tvd),
                    'tvd_range': [float(self.ag_typewell_raw.tvd.min()),
                                  float(self.ag_typewell_raw.tvd.max())],
                    'gr_len': len(self.ag_typewell_raw.value),
                    'gr_range': [float(self.ag_typewell_raw.value.min()),
                                 float(self.ag_typewell_raw.value.max())],
                },
                'tvd_shift': float(self.tvd_to_typewell_shift),
            }
            with open(dump_path, 'w') as f:
                json.dump(dump_data, f, indent=2)
            logger.info(f"DUMP: ag_well and ag_typewell saved to {dump_path}")

        # Get start MD BEFORE normalization
        start_md_meters = well_data['autoGeosteeringParameters']['startMd']
        self.start_idx = self.ag_well.md2idx(start_md_meters)
        self.start_md_meters = start_md_meters

        manual_segments_json = well_data.get('interpretation', {}).get('segments', [])

        # Store reference interpretation for comparison (handle None values)
        ref_interp = well_data.get('referenceInterpretation') or {}
        self.reference_segments = ref_interp.get('segments', [])
        if not self.reference_segments:
            logger.warning("No reference interpretation segments found - quality metrics unavailable")
        else:
            logger.debug(f"Reference interpretation: {len(self.reference_segments)} segments")

        # Calculate fixed planning horizon
        self.fixed_min_md = self.ag_well.min_md
        self.fixed_md_range = (self.ag_well.max_md + self.md_normalization_buffer) - self.fixed_min_md
        logger.info(f"Fixed planning horizon: min_md={self.fixed_min_md:.2f}, "
                    f"fixed_md_range={self.fixed_md_range:.2f}")

        # Normalize with fixed md_range
        # Create normalized copy of typewell (raw is kept for re-normalization on update)
        self.ag_typewell = deepcopy(self.ag_typewell_raw)
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

        # Get prev_segment_angle (last frozen segment before optimization)
        prev_segment_angle = None
        if optimize_indices and optimize_indices[0] > 0:
            prev_segment = self.interpretation[optimize_indices[0] - 1]
            prev_segment_angle = prev_segment.angle
            logger.debug(f"prev_segment_angle={prev_segment_angle:.2f}° from frozen segment")

        best_fun, optimal_shifts, elapsed, wrapper = self._optimize_with_snes(
            segments_to_optimize, self.start_idx, prev_segment_angle
        )

        # DEBUG: Log raw shifts from optimizer
        logger.info(f"  RAW optimal_shifts (normalized): {[f'{s:.6f}' for s in optimal_shifts]}")
        # Denormalize to meters for human readability
        raw_shifts_m = [s * self.fixed_md_range for s in optimal_shifts]
        logger.info(f"  RAW optimal_shifts (meters): {[f'{s:.2f}' for s in raw_shifts_m]}")

        # Compute detailed metrics comparison
        self._log_metrics_comparison(
            wrapper, optimal_shifts, segments_to_optimize, 'init'
        )

        # Apply optimized shifts
        optimized_segments = self._apply_optimized_shifts(segments_to_optimize, optimal_shifts)
        for i, opt_seg in zip(optimize_indices, optimized_segments):
            self.interpretation[i] = opt_seg

        # Log
        self._log_optimization_result('init', self.ag_well.measured_depth[-1] * self.fixed_md_range + self.fixed_min_md,
                                      best_fun, elapsed, wrapper, optimal_shifts, segments_to_optimize)

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
        # Re-normalize typewell with new max_curve_value to keep consistent scaling
        self.ag_typewell = deepcopy(self.ag_typewell_raw)
        max_curve_value = max(updated_well.max_curve, self.ag_typewell.value.max())
        updated_well.normalize(max_curve_value, self.ag_typewell.min_depth, self.fixed_md_range)
        self.ag_typewell.normalize(max_curve_value, updated_well.min_depth, self.fixed_md_range)
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

        # Get prev_segment_angle (last frozen segment before optimization)
        prev_segment_angle = None
        if optimize_indices and optimize_indices[0] > 0:
            prev_segment = self.interpretation[optimize_indices[0] - 1]
            prev_segment_angle = prev_segment.angle
            logger.debug(f"prev_segment_angle={prev_segment_angle:.2f}° from frozen segment")

        best_fun, optimal_shifts, elapsed, wrapper = self._optimize_with_snes(
            segments_to_optimize, lookback_idx, prev_segment_angle
        )

        # DEBUG: Log raw shifts from optimizer
        logger.info(f"  RAW optimal_shifts (normalized): {[f'{s:.6f}' for s in optimal_shifts]}")
        # Denormalize to meters for human readability
        raw_shifts_m = [s * self.fixed_md_range for s in optimal_shifts]
        logger.info(f"  RAW optimal_shifts (meters): {[f'{s:.2f}' for s in raw_shifts_m]}")

        # Compute detailed metrics comparison
        self._log_metrics_comparison(
            wrapper, optimal_shifts, segments_to_optimize, 'step'
        )

        optimized_segments = self._apply_optimized_shifts(segments_to_optimize, optimal_shifts)
        for i, opt_seg in zip(optimize_indices, optimized_segments):
            self.interpretation[i] = opt_seg

        # Log
        self._log_optimization_result('step', current_md * self.fixed_md_range + self.fixed_min_md,
                                      best_fun, elapsed, wrapper, optimal_shifts, segments_to_optimize)

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
                                 best_fun: float, elapsed: float,
                                 wrapper=None, optimal_shifts=None, segments=None):
        """Log optimization result to statistics"""
        import torch

        # Get actual function evaluations from wrapper
        actual_evals = wrapper.eval_count if wrapper else 0

        # Calculate expected evaluations for comparison
        if self.algorithm == 'CMAES':
            if self.cmaes_mode == 'single':
                expected_evals = self.single_popsize * self.maxiter
            else:
                expected_evals = self.last_num_populations * self.popsize * self.maxiter
        else:
            expected_evals = self.n_restarts * self.maxiter * self.popsize

        # Get detailed metrics from wrapper
        detailed = {}
        angles_str = ""
        if wrapper and optimal_shifts:
            shifts_tensor = torch.tensor(optimal_shifts, dtype=torch.float64, device=self.device)
            detailed = wrapper.compute_detailed(shifts_tensor)
            angles_str = ",".join([f"{a:.2f}" for a in detailed.get('angles_deg', [])])

        # Get MD range from segments
        start_md = segments[0].start_md * self.fixed_md_range + self.fixed_min_md if segments else 0
        end_md = segments[-1].end_md * self.fixed_md_range + self.fixed_min_md if segments else measured_depth

        optimization_stats = {
            'method': f'EvoTorch_{self.algorithm}',
            'segments_count': len(segments) if segments else self.segments_count,
            'n_restarts': self.n_restarts,
            'popsize': self.popsize,
            'maxiter': self.maxiter,
            'final_fun': best_fun,
            'elapsed_time': elapsed,
            'device': self.device,
            'success': True,
            # Actual vs expected evaluations
            'n_iterations': self.last_num_populations * self.maxiter if self.algorithm == 'CMAES' else self.n_restarts * self.maxiter,
            'n_function_evaluations': actual_evals,
            'expected_evaluations': expected_evals,
            # Detailed metrics
            'pearson': detailed.get('pearson', 0.0),
            'mse': detailed.get('mse', best_fun),
            'angle_penalty': detailed.get('angle_penalty', 0.0),
            'angle_sum_penalty': detailed.get('angle_sum_penalty', 0.0),
            # Legacy fields
            'final_correlation': detailed.get('pearson', 0.0),
            'pearson_correlation': detailed.get('pearson', 0.0),
            'mse_value': detailed.get('mse', best_fun),
            'self_correlation': 0.0,
            'intersections_count': 0
        }

        optimization_result = {
            'well_name': self.well_name,
            'measured_depth': measured_depth,
            'step_type': step_type,
            'start_md': start_md,
            'end_md': end_md,
            'angles': angles_str,
            'optimization_stats': optimization_stats
        }

        logger.info(f"Optimization [{step_type}]: MD={start_md:.1f}-{end_md:.1f}m, fun={best_fun:.6f}, "
                    f"evals={actual_evals}, pearson={detailed.get('pearson', 0):.4f}, "
                    f"mse={detailed.get('mse', 0):.6f}, time={elapsed:.2f}s")

        self.optimization_logger.log_optimization_result(optimization_result)

    def _log_metrics_comparison(self, wrapper, optimal_shifts: List[float],
                                segments: List, step_type: str):
        """Log detailed metrics comparison: optimized vs reference interpretation"""
        if not self.reference_segments:
            logger.debug("No reference interpretation for comparison")
            return

        # Get segment start and end MDs in meters
        first_start_md_m = segments[0].start_md * self.fixed_md_range + self.fixed_min_md
        last_end_md_m = segments[-1].end_md * self.fixed_md_range + self.fixed_min_md

        # Compute detailed metrics for optimized
        opt_metrics = wrapper.compute_detailed(optimal_shifts)

        # Build reference segments with correct geometry for the same MD range
        ref_segments_torch = build_reference_segments_torch(
            self.reference_segments,
            self.ag_well,
            first_start_md_m,
            last_end_md_m,
            self.fixed_md_range,
            self.fixed_min_md,
            device=self.device
        )

        if ref_segments_torch is None or len(ref_segments_torch) == 0:
            logger.warning("No reference segments in optimization range")
            ref_metrics = {'objective': float('nan'), 'pearson': float('nan'), 'mse': float('nan'),
                          'angle_penalty': 0.0, 'angle_sum_penalty': 0.0, 'angles_deg': []}
        else:
            # Create wrapper with reference segments geometry
            # Reference shifts are already in the segments (columns 4,5)
            ref_shifts = ref_segments_torch[:, 5].tolist()  # end_shifts

            from torch_funcs.batch_objective import compute_detailed_metrics_torch

            # Compute reference metrics using reference geometry
            # Use same reward_start_segment_idx as optimization (single source of truth)
            ref_shifts_tensor = torch.tensor(ref_shifts, dtype=torch.float64, device=self.device)
            ref_metrics = compute_detailed_metrics_torch(
                ref_shifts_tensor,
                wrapper.well_data,
                wrapper.typewell_data,
                ref_segments_torch,
                wrapper.pearson_power,
                wrapper.mse_power,
                wrapper.angle_range,
                wrapper.angle_sum_power,
                self.tvd_to_typewell_shift,
                prev_segment_angle=None,
                reward_start_segment_idx=wrapper.reward_start_segment_idx
            )

        # Log comparison
        logger.info(f"=== Metrics Comparison [{step_type}] ===")
        logger.info(f"  Opt segments: {len(segments)}, Ref segments: {len(ref_segments_torch) if ref_segments_torch is not None else 0}")
        logger.info(f"  {'':12} | {'Optimized':>12} | {'Reference':>12} | {'Diff':>10}")
        logger.info(f"  {'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
        logger.info(f"  {'Objective':12} | {opt_metrics['objective']:>12.6f} | {ref_metrics['objective']:>12.6f} | {opt_metrics['objective'] - ref_metrics['objective']:>+10.6f}")
        logger.info(f"  {'Pearson':12} | {opt_metrics['pearson']:>12.6f} | {ref_metrics['pearson']:>12.6f} | {opt_metrics['pearson'] - ref_metrics['pearson']:>+10.6f}")
        logger.info(f"  {'Pearson(raw)':12} | {opt_metrics.get('pearson_raw', opt_metrics['pearson']):>12.6f} | {ref_metrics.get('pearson_raw', ref_metrics['pearson']):>12.6f} |")
        logger.info(f"  {'MSE':12} | {opt_metrics['mse']:>12.6f} | {ref_metrics['mse']:>12.6f} | {opt_metrics['mse'] - ref_metrics['mse']:>+10.6f}")
        logger.info(f"  {'AnglePenalty':12} | {opt_metrics['angle_penalty']:>12.6f} | {ref_metrics['angle_penalty']:>12.6f} | {opt_metrics['angle_penalty'] - ref_metrics['angle_penalty']:>+10.6f}")
        logger.info(f"  {'AngleSumPen':12} | {opt_metrics['angle_sum_penalty']:>12.6f} | {ref_metrics['angle_sum_penalty']:>12.6f} | {opt_metrics['angle_sum_penalty'] - ref_metrics['angle_sum_penalty']:>+10.6f}")

        # Log angles
        opt_angles = opt_metrics['angles_deg']
        ref_angles = ref_metrics['angles_deg']
        logger.info(f"  Angles (°):  Opt={[f'{a:.2f}' for a in opt_angles[:4]]}...  Ref={[f'{a:.2f}' for a in ref_angles[:4]]}...")

        # Status
        if opt_metrics['objective'] <= ref_metrics['objective']:
            logger.info(f"  ✓ Optimized BETTER than reference")
        else:
            logger.info(f"  ✗ Reference BETTER than optimized (diff={opt_metrics['objective'] - ref_metrics['objective']:.6f})")

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
