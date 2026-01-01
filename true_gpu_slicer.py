"""
True GPU Slicer (RND-809)

Pure Python slicing orchestrator - no StarSteer dependency.
Loads pre-processed dataset and runs GPU optimization incrementally.

Usage:
    python true_gpu_slicer.py --well Well1002_landing~EGFDL
    python true_gpu_slicer.py --all --slice-step 30

Key differences from cpu_baseline/slicer.py:
- Data source: gpu_ag_dataset.pt (not StarSteer JSON files)
- Slicing: Python array truncation (not StarSteer SLICE command)
- Interpretation: In-memory storage (not StarSteer export)
- Typewell: Pre-stitched in dataset (no TypewellProvider needed)
"""

import sys
import os
import torch
import numpy as np
import logging
import argparse
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Peak detectors for auto telescope lever detection
from peak_detectors import OtsuPeakDetector, MADPeakDetector, RollingStdDetector, RegionFinder

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / "cpu_baseline" / ".env")
except ImportError:
    pass  # dotenv is optional

from gpu_executor import GpuAutoGeosteeringExecutor
from cpu_baseline.typewell_provider import extend_pseudo_with_typelog

# Normalization mode: 'OLD' = type×(1/mult), pseudo raw
#                     'NEW' = pseudo×mult, type raw
NORMALIZATION_MODE = os.getenv('NORMALIZATION_MODE', 'OLD')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def tensor_to_raw_well_points(md: torch.Tensor, tvd: torch.Tensor,
                              ns: torch.Tensor, ew: torch.Tensor) -> List[Dict]:
    """
    Convert RAW well trajectory tensors to well points format for Well object.
    No interpolation - direct conversion.
    """
    md_np = md.cpu().numpy()
    tvd_np = tvd.cpu().numpy()
    ns_np = ns.cpu().numpy()
    ew_np = ew.cpu().numpy()

    points = []
    for i in range(len(md_np)):
        points.append({
            'measuredDepth': float(md_np[i]),
            'trueVerticalDepth': float(tvd_np[i]),
            'northSouth': float(ns_np[i]),
            'eastWest': float(ew_np[i])
        })
    return points


def tensor_to_welllog_points(md: torch.Tensor, gr: torch.Tensor) -> List[Dict]:
    """Convert tensors to wellLog points format"""
    md_np = md.cpu().numpy()
    gr_np = gr.cpu().numpy()

    points = []
    for i in range(len(md_np)):
        points.append({
            'measuredDepth': float(md_np[i]),
            'data': float(gr_np[i])
        })
    return points


def tensor_to_typelog_points(tvd: torch.Tensor, gr: torch.Tensor) -> List[Dict]:
    """Convert tensors to typeLog points format (tvdSortedPoints)"""
    tvd_np = tvd.cpu().numpy()
    gr_np = gr.cpu().numpy()

    points = []
    for i in range(len(tvd_np)):
        points.append({
            'trueVerticalDepth': float(tvd_np[i]),
            'data': float(gr_np[i])
        })
    return points


def stitch_typewell_at_runtime(data: Dict[str, Any], mode: str = 'OLD') -> List[Dict]:
    """
    Stitch pseudo + type at runtime with normalization.

    Args:
        data: Well data from dataset (contains pseudo_tvd, pseudo_gr, type_tvd, type_gr, norm_multiplier)
        mode: 'OLD' = type×(1/mult), pseudo raw (stitched)
              'NEW' = pseudo×mult, type raw (stitched)
              'ORIGINAL' = pure typeLog, no pseudo stitching

    Returns:
        Stitched typeLog tvdSortedPoints (or pure typeLog if ORIGINAL)
    """
    norm_multiplier = data.get('norm_multiplier', 1.0)

    # Get raw data
    pseudo_tvd = data['pseudo_tvd'].cpu().numpy()
    pseudo_gr = data['pseudo_gr'].cpu().numpy()
    type_tvd = data['type_tvd'].cpu().numpy()
    type_gr = data['type_gr'].cpu().numpy()

    # ORIGINAL mode: pure typeLog without stitching
    if mode == 'ORIGINAL':
        logger.info(f"Using ORIGINAL typeLog (no pseudo stitching), {len(type_tvd)} points")
        return [{'trueVerticalDepth': float(tvd), 'data': float(gr)}
                for tvd, gr in zip(type_tvd, type_gr)]

    # Build dict format for extend_pseudo_with_typelog
    pseudo_dict = {
        'tvdSortedPoints': [{'trueVerticalDepth': float(tvd), 'data': float(gr)}
                           for tvd, gr in zip(pseudo_tvd, pseudo_gr)]
    }
    type_dict = {
        'tvdSortedPoints': [{'trueVerticalDepth': float(tvd), 'data': float(gr)}
                           for tvd, gr in zip(type_tvd, type_gr)]
    }

    if mode == 'NEW':
        # NEW: pseudo × mult, type raw
        if norm_multiplier != 1.0:
            for p in pseudo_dict['tvdSortedPoints']:
                p['data'] *= norm_multiplier
        stitched = extend_pseudo_with_typelog(pseudo_dict, type_dict, norm_coef=1.0)
    else:
        # OLD: pseudo raw, type × (1/mult)
        norm_coef = 1.0 / norm_multiplier if norm_multiplier != 0 else 1.0
        stitched = extend_pseudo_with_typelog(pseudo_dict, type_dict, norm_coef=norm_coef)

    return stitched.get('tvdSortedPoints', [])


def slice_data_to_md(data: Dict[str, Any], current_md: float) -> Dict[str, Any]:
    """
    Slice RAW well data to current_md (truncate arrays).

    Args:
        data: Full well data from dataset (RAW format)
        current_md: Current MD to slice to (meters)

    Returns:
        Sliced well_data dict for gpu_executor (JSON format)
    """
    # Slice well trajectory points (well_md, well_tvd, well_ns, well_ew)
    well_md = data['well_md'].cpu().numpy()
    well_mask = well_md <= current_md
    well_end_idx = max(1, well_mask.sum())

    well_md_sliced = data['well_md'][:well_end_idx]
    well_tvd_sliced = data['well_tvd'][:well_end_idx]
    well_ns_sliced = data['well_ns'][:well_end_idx]
    well_ew_sliced = data['well_ew'][:well_end_idx]

    # Slice wellLog points (log_md, log_gr)
    log_md = data['log_md'].cpu().numpy()
    log_mask = log_md <= current_md
    log_end_idx = max(1, log_mask.sum())

    log_md_sliced = data['log_md'][:log_end_idx]
    log_gr_sliced = data['log_gr'][:log_end_idx]

    # Apply norm_multiplier to wellLog GR (as StarSteer does)
    norm_multiplier = data.get('norm_multiplier', 1.0)
    if norm_multiplier and norm_multiplier != 1.0:
        log_gr_sliced = log_gr_sliced * norm_multiplier

    # Build well_data for gpu_executor (JSON format - will be processed by Well class)
    well_data = {
        'wellName': data.get('well_name', 'Unknown'),

        # RAW well trajectory points
        'well': {
            'points': tensor_to_raw_well_points(
                well_md_sliced, well_tvd_sliced, well_ns_sliced, well_ew_sliced
            )
        },

        # RAW wellLog points
        'wellLog': {
            'points': tensor_to_welllog_points(log_md_sliced, log_gr_sliced)
        },

        # Stitch pseudo+type at runtime with selected normalization mode
        'typeLog': {
            'tvdSortedPoints': stitch_typewell_at_runtime(data, mode=NORMALIZATION_MODE)
        },

        # tvd_typewell_shift is needed for projection calculation
        # TODO: verify if pseudo_stitched really doesn't need shift
        'tvdTypewellShift': data.get('tvd_typewell_shift', 0.0),

        'autoGeosteeringParameters': {
            'startMd': data.get('detected_start_md') or data.get('start_md', 0.0),
            'lookBackDistance': float(os.getenv('PYTHON_LOOKBACK_DISTANCE', '50.0')),
        },

        'interpretation': {'segments': []},  # Will be filled from work_interpretation

        # Reference interpretation for comparison (full, not sliced)
        'referenceInterpretation': {
            'segments': build_ref_segments(data)
        }
    }

    return well_data


def build_ref_segments(data: Dict[str, Any]) -> List[Dict]:
    """Build reference segments from dataset tensors"""
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_shifts = data['ref_shifts'].cpu().numpy()

    if len(ref_mds) == 0:
        return []

    segments = []
    for i in range(len(ref_mds)):
        seg = {
            'startMd': float(ref_mds[i]),
            'startShift': float(ref_shifts[i - 1]) if i > 0 else 0.0,
            'endShift': float(ref_shifts[i])
        }
        # Add endMd from next segment or estimate
        if i + 1 < len(ref_mds):
            seg['endMd'] = float(ref_mds[i + 1])
        segments.append(seg)

    return segments


def build_manual_interpretation_to_md(data: Dict[str, Any], target_md: float) -> List[Dict]:
    """
    Build manual interpretation segments from ref_shifts up to target_md.

    ref_segment_mds[i] and ref_shifts[i] are the END point of segment i.

    Args:
        data: Well data with ref_segment_mds and ref_shifts
        target_md: MD to build interpretation up to

    Returns:
        List of segment dicts suitable for gpu_executor
    """
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_shifts = data['ref_shifts'].cpu().numpy()

    if len(ref_mds) == 0:
        return []

    # Get start_md for first segment (from data or first ref point)
    first_seg_start = data.get('start_md', ref_mds[0] - 100)

    segments = []
    for i in range(len(ref_mds)):
        # Segment i: starts at previous end, ends at ref_mds[i]
        if i == 0:
            seg_start_md = float(first_seg_start)
            start_shift = 0.0
        else:
            seg_start_md = float(ref_mds[i - 1])
            start_shift = float(ref_shifts[i - 1])

        seg_end_md = float(ref_mds[i])
        end_shift = float(ref_shifts[i])

        # Stop if segment starts after target_md
        if seg_start_md >= target_md:
            break

        # Truncate last segment at target_md if needed
        if seg_end_md > target_md:
            # Interpolate end_shift at target_md
            if seg_end_md > seg_start_md:
                ratio = (target_md - seg_start_md) / (seg_end_md - seg_start_md)
                end_shift = start_shift + ratio * (end_shift - start_shift)
            seg_end_md = target_md

        segments.append({
            'startMd': seg_start_md,
            'endMd': seg_end_md,
            'startShift': start_shift,
            'endShift': end_shift
        })

        # Stop after truncated segment
        if seg_end_md >= target_md:
            break

    return segments


def interpolate_shift_at_md(data: Dict[str, Any], target_md: float) -> float:
    """
    Interpolate shift from ref_shifts at given MD.

    Args:
        data: Well data with ref_segment_mds and ref_shifts
        target_md: MD to interpolate shift at

    Returns:
        Interpolated shift value (meters)
    """
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_shifts = data['ref_shifts'].cpu().numpy()

    if len(ref_mds) == 0:
        return 0.0

    # Find segment containing target_md
    for i in range(len(ref_mds)):
        seg_start = ref_mds[i]
        seg_end = ref_mds[i + 1] if i + 1 < len(ref_mds) else seg_start + 1000

        if seg_start <= target_md <= seg_end:
            # Interpolate within segment
            start_shift = ref_shifts[i - 1] if i > 0 else 0.0
            end_shift = ref_shifts[i]

            if seg_end > seg_start:
                ratio = (target_md - seg_start) / (seg_end - seg_start)
                return start_shift + ratio * (end_shift - start_shift)
            return start_shift

    # target_md before first segment
    if target_md < ref_mds[0]:
        return 0.0

    # target_md after last segment
    return float(ref_shifts[-1])


def run_slicing(data: Dict[str, Any], executor: GpuAutoGeosteeringExecutor,
                slice_step: float = 30.0,
                first_slice_length: float = None) -> Dict[str, Any]:
    """
    Run incremental slicing on a single well.

    Args:
        data: Well data from dataset (with well_name added)
        executor: GPU executor instance
        slice_step: MD step between slices (meters)
        first_slice_length: Length of first slice in meters (default: slice_step)

    Returns:
        Dict with final interpretation and metrics
    """
    if first_slice_length is None:
        first_slice_length = slice_step
    # Use wellLog MD for slicing (last GR point defines max_md)
    log_md_np = data['log_md'].cpu().numpy()
    max_md = float(log_md_np[-1])

    # Get start MD (use detected_start_md if available, else start_md)
    start_md = data.get('detected_start_md') or data.get('start_md', 0.0)
    if start_md is None:
        start_md = 0.0

    # Get initial shift from reference interpretation
    initial_shift = interpolate_shift_at_md(data, start_md)
    logger.info(f"Initial shift at start_md={start_md:.1f}: {initial_shift:.2f}m")

    current_md = start_md + first_slice_length  # First slice length (default=slice_step)

    # Build initial manual interpretation from ref_shifts up to start_md
    work_interpretation = build_manual_interpretation_to_md(data, start_md)
    logger.info(f"Manual interpretation: {len(work_interpretation)} segments up to start_md")

    iteration = 0
    total_time = 0.0

    logger.info(f"Starting slicing: start_md={start_md:.1f}, max_md={max_md:.1f}, step={slice_step}")

    while current_md < max_md:
        iteration += 1

        # Slice data to current_md
        well_data = slice_data_to_md(data, current_md)
        well_data['interpretation'] = {'segments': work_interpretation}

        # Run optimization
        start_time = time.time()

        if iteration == 1:
            result = executor.initialize_well(well_data)
        else:
            result = executor.update_well_data(well_data)

        elapsed = time.time() - start_time
        total_time += elapsed

        # Update work interpretation
        work_interpretation = result['interpretation']['segments']

        logger.info(f"Iteration {iteration}: MD={current_md:.1f}m, "
                    f"segments={len(work_interpretation)}, time={elapsed:.2f}s")

        current_md += slice_step

    # Final slice with all data
    if current_md != max_md:
        iteration += 1
        well_data = slice_data_to_md(data, max_md)
        well_data['interpretation'] = {'segments': work_interpretation}

        start_time = time.time()
        result = executor.update_well_data(well_data)
        elapsed = time.time() - start_time
        total_time += elapsed

        work_interpretation = result['interpretation']['segments']
        logger.info(f"Final iteration {iteration}: MD={max_md:.1f}m, "
                    f"segments={len(work_interpretation)}, time={elapsed:.2f}s")

    return {
        'well_name': data.get('well_name', 'Unknown'),
        'iterations': iteration,
        'total_time': total_time,
        'final_segments': len(work_interpretation),
        'interpretation': work_interpretation
    }


def compare_with_reference(result: Dict[str, Any], data: Dict[str, Any], md_step: float = 1.0) -> Dict[str, Any]:
    """
    Compare slicing result with reference interpretation.

    Uses interpolation to common MD grid for accurate comparison.

    Args:
        result: Slicing result with interpretation
        data: Well data with reference interpretation
        md_step: MD grid step for interpolation (default 1m)

    Returns metrics dict with MAE, RMSE, endpoint_error.
    """
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_shifts = data['ref_shifts'].cpu().numpy()

    if len(ref_shifts) == 0:
        logger.warning("No reference interpretation for comparison")
        return {'error': 'no_reference'}

    # Get final interpretation
    final_segments = result['interpretation']
    if not final_segments:
        return {'error': 'no_result_segments'}

    # Extract our MDs and shifts
    our_mds = np.array([seg['endMd'] for seg in final_segments])
    our_shifts = np.array([seg['endShift'] for seg in final_segments])

    if len(our_mds) == 0:
        return {'error': 'no_result_segments'}

    # Find common MD range
    md_min = max(ref_mds[0], our_mds[0])
    md_max = min(ref_mds[-1], our_mds[-1])

    if md_max <= md_min:
        return {'error': 'no_overlap'}

    # Create common MD grid
    common_mds = np.arange(md_min, md_max + md_step, md_step)

    # Interpolate both to common grid
    ref_interp = np.interp(common_mds, ref_mds, ref_shifts)
    our_interp = np.interp(common_mds, our_mds, our_shifts)

    # Calculate metrics
    diffs = our_interp - ref_interp
    mae = np.mean(np.abs(diffs))
    rmse = np.sqrt(np.mean(diffs ** 2))
    max_error = np.max(np.abs(diffs))
    max_error_md = common_mds[np.argmax(np.abs(diffs))]

    # Endpoint error (at last reference MD)
    endpoint_our = np.interp(ref_mds[-1], our_mds, our_shifts)
    endpoint_error = endpoint_our - ref_shifts[-1]

    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'max_error': float(max_error),
        'max_error_md': float(max_error_md),
        'endpoint_error': float(endpoint_error),
        'endpoint_md': float(ref_mds[-1]),
        'n_points_compared': len(common_mds),
        'md_range': [float(md_min), float(md_max)],
        'n_result_segments': len(our_mds),
        'n_ref_segments': len(ref_mds)
    }


def load_dataset(path: Path) -> Dict[str, Dict]:
    """Load PyTorch dataset"""
    logger.info(f"Loading dataset from {path}")
    dataset = torch.load(path, weights_only=False)
    logger.info(f"Loaded {len(dataset)} wells")
    return dataset


def process_well_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for multiprocessing. Processes a single well.
    Each worker initializes its own GPU executor.
    """
    well_name, well_data, slice_step, work_dir, results_dir, algorithm, first_slice_length, env_config = args

    # Set env vars before importing executor (important for multiprocessing)
    if env_config:
        for key, value in env_config.items():
            if value is not None:
                os.environ[key] = str(value)

    # Import here to avoid issues with multiprocessing
    from gpu_executor import GpuAutoGeosteeringExecutor

    # Each worker creates its own executor
    executor = GpuAutoGeosteeringExecutor(
        work_dir=work_dir,
        results_dir=results_dir
    )
    if algorithm:
        executor.set_algorithm(algorithm)
    executor.start_daemon()

    try:
        well_data['well_name'] = well_name

        # Auto-detect telescope lever MD if requested
        auto_detector = os.getenv('TELESCOPE_AUTO_DETECTOR')
        telescope_mode = os.getenv('TELESCOPE_MODE') == 'true'
        telescope_lever_md = os.getenv('TELESCOPE_LEVER_MD')

        if telescope_mode and telescope_lever_md is None and auto_detector:
            lookback = float(os.getenv('PYTHON_LOOKBACK_DISTANCE', 250))
            detected_md = detect_telescope_lever_md(well_data, auto_detector, lookback)
            os.environ['TELESCOPE_LEVER_MD'] = str(detected_md)
            logger.info(f"[{well_name}] Auto-detected lever MD: {detected_md:.1f}m ({detected_md * 3.28084:.0f}ft) using {auto_detector}")

            # Recalculate first_slice if not specified
            if first_slice_length is None:
                start_md = well_data.get('detected_start_md') or well_data.get('start_md', 0.0) or 0.0
                work_length = float(os.getenv('TELESCOPE_WORK_SEGMENT_LENGTH', 30))
                work_count = int(os.getenv('TELESCOPE_WORK_SEGMENTS_COUNT', 4))
                lever_length = detected_md - start_md
                first_slice_length = lever_length + (work_count - 1) * work_length

        result = run_slicing(well_data, executor, slice_step, first_slice_length)
        metrics = compare_with_reference(result, well_data)
        result['metrics'] = metrics
        result['well_name'] = well_name
        return result
    except Exception as e:
        import traceback
        return {
            'well_name': well_name,
            'error': str(e),
            'traceback': traceback.format_exc()
        }
    finally:
        executor.stop_daemon()


def detect_telescope_lever_md(data: Dict[str, Any], detector_name: str, lookback_m: float = 250.0) -> float:
    """
    Auto-detect best telescope lever position using peak detectors.

    Args:
        data: Well data dict with log_md, log_gr
        detector_name: 'otsu', 'rollingstd', or 'mad'
        lookback_m: Region size in meters (default 250 = lookback distance)

    Returns:
        Best MD position for telescope lever
    """
    md = data['log_md'].cpu().numpy() if hasattr(data['log_md'], 'numpy') else np.array(data['log_md'])
    gr = data['log_gr'].cpu().numpy() if hasattr(data['log_gr'], 'numpy') else np.array(data['log_gr'])

    if detector_name == 'otsu':
        detector = OtsuPeakDetector()
        finder = RegionFinder(detector, search_fraction=0.33)
        result = finder.find_best_region(gr, md, region_length_m=lookback_m)
        return result.best_md

    elif detector_name == 'rollingstd':
        total_length = md[-1] - md[0]
        search_start_md = md[-1] - total_length * 0.33
        search_start_idx = np.searchsorted(md, search_start_md)
        search_end_idx = len(md)
        detector = RollingStdDetector(window_m=100.0)
        best_md, _, _ = detector.find_best_region(gr, md, search_start_idx, search_end_idx, region_length_m=lookback_m)
        return best_md

    elif detector_name == 'mad':
        detector = MADPeakDetector(k=1.5)
        finder = RegionFinder(detector, search_fraction=0.33)
        result = finder.find_best_region(gr, md, region_length_m=lookback_m)
        return result.best_md

    else:
        raise ValueError(f"Unknown detector: {detector_name}")


def main():
    parser = argparse.ArgumentParser(description='True GPU Slicer - pure Python slicing')
    parser.add_argument('--dataset', type=str,
                        default=str(Path(__file__).parent / "dataset" / "gpu_ag_dataset.pt"),
                        help='Path to PyTorch dataset')
    parser.add_argument('--well', type=str,
                        help='Single well to process (e.g., Well1002_landing~EGFDL)')
    parser.add_argument('--all', action='store_true',
                        help='Process all wells in dataset')
    parser.add_argument('--slice-step', type=float, default=30.0,
                        help='MD step between slices (meters)')
    parser.add_argument('--results-dir', type=str,
                        default=str(Path(__file__).parent / "results" / "true_gpu_slicer"),
                        help='Directory for results')
    parser.add_argument('--algorithm', type=str, default=None,
                        help='Override algorithm (DE, CMAES, SNES, etc.)')
    parser.add_argument('--worker', type=int, default=0,
                        help='Worker index for parallel runs (0-based)')
    parser.add_argument('--total-workers', type=int, default=1,
                        help='Total number of parallel workers')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of parallel workers within single process (multiprocessing)')

    # Grid search parameters
    parser.add_argument('--first-slice-length', type=float, default=None,
                        help='Length of first slice in meters (default: slice_step)')
    parser.add_argument('--lookback-distance', type=float, default=None,
                        help='Lookback distance in meters (default: from PYTHON_LOOKBACK_DISTANCE env)')
    parser.add_argument('--max-segment-length', type=float, default=None,
                        help='Max segment length in meters (default: from PYTHON_MAX_SEGMENT_LENGTH env)')
    parser.add_argument('--angle-sum-power', type=float, default=None,
                        help='Angle sum penalty power (default: from PYTHON_ANGLE_SUM_POWER env)')

    # Telescope mode parameters
    parser.add_argument('--telescope', action='store_true',
                        help='Enable telescope mode: one long lever + short work segments')
    parser.add_argument('--telescope-lever-md', type=float, default=None,
                        help='Telescope lever END position as absolute MD in meters (e.g. 5944 for ~19500ft)')
    parser.add_argument('--telescope-work-length', type=float, default=30.0,
                        help='Telescope work segment length in meters (default: 30)')
    parser.add_argument('--telescope-work-count', type=int, default=4,
                        help='Number of work segments after lever (default: 4)')
    parser.add_argument('--telescope-auto-detect', type=str, default=None,
                        choices=['otsu', 'rollingstd', 'mad'],
                        help='Auto-detect telescope lever position using peak detector (otsu/rollingstd/mad)')

    args = parser.parse_args()

    # Fallback to .env for telescope auto-detect
    if args.telescope_auto_detect is None:
        env_detector = os.getenv('TELESCOPE_AUTO_DETECTOR')
        if env_detector and env_detector.lower() in ['otsu', 'rollingstd', 'mad']:
            args.telescope_auto_detect = env_detector.lower()

    dataset_path = Path(args.dataset)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    # Load dataset
    dataset = load_dataset(dataset_path)

    # Select wells to process
    if args.well:
        if args.well not in dataset:
            logger.error(f"Well not found in dataset: {args.well}")
            logger.info(f"Available wells: {list(dataset.keys())[:10]}...")
            sys.exit(1)
        wells_to_process = [args.well]
    elif args.all:
        all_wells = sorted(dataset.keys())  # Sort for consistent ordering across workers
        # Distribute wells across workers: worker i gets wells i, i+n, i+2n, ...
        if args.total_workers > 1:
            wells_to_process = [w for i, w in enumerate(all_wells) if i % args.total_workers == args.worker]
            logger.info(f"Worker {args.worker}/{args.total_workers}: processing {len(wells_to_process)} wells")
        else:
            wells_to_process = all_wells
    else:
        # Default: first well
        wells_to_process = [list(dataset.keys())[0]]
        logger.info(f"No well specified, using first well: {wells_to_process[0]}")

    # Setup directories
    work_dir = str(results_dir / "work")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    # Process wells
    all_results = []

    if args.num_workers > 1:
        # Multiprocessing mode
        logger.info(f"Using {args.num_workers} parallel workers")
        mp.set_start_method('spawn', force=True)

        # Prepare args for workers
        # For telescope mode, first_slice is calculated per-well in run_slicing (needs start_md)
        first_slice = args.first_slice_length  # None if not specified, will be computed in run_slicing
        env_config = {
            'PYTHON_LOOKBACK_DISTANCE': args.lookback_distance,
            'PYTHON_MAX_SEGMENT_LENGTH': args.max_segment_length,
            'PYTHON_ANGLE_SUM_POWER': args.angle_sum_power,
            'TELESCOPE_MODE': 'true' if args.telescope else None,
            'TELESCOPE_LEVER_MD': args.telescope_lever_md if args.telescope else None,
            'TELESCOPE_AUTO_DETECTOR': args.telescope_auto_detect if args.telescope else None,
            'TELESCOPE_WORK_SEGMENT_LENGTH': args.telescope_work_length if args.telescope else None,
            'TELESCOPE_WORK_SEGMENTS_COUNT': args.telescope_work_count if args.telescope else None,
        }
        worker_args = [
            (well_name, dataset[well_name], args.slice_step, work_dir, str(results_dir), args.algorithm, first_slice, env_config)
            for well_name in wells_to_process
        ]

        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(process_well_worker, arg): arg[0] for arg in worker_args}

            for future in as_completed(futures):
                well_name = futures[future]
                try:
                    result = future.result()
                    if 'error' in result:
                        logger.error(f"Error processing {well_name}: {result['error']}")
                    else:
                        all_results.append(result)
                        m = result['metrics']
                        logger.info(f"Completed {well_name}: {result['iterations']} iters, "
                                    f"{result['total_time']:.1f}s, MAE={m.get('mae', 0):.2f}m, "
                                    f"RMSE={m.get('rmse', 0):.2f}m, endpoint={m.get('endpoint_error', 0):+.2f}m")
                except Exception as e:
                    logger.error(f"Worker error for {well_name}: {e}")
    else:
        # Single process mode (original)
        # Set env vars before creating executor
        if args.lookback_distance:
            os.environ['PYTHON_LOOKBACK_DISTANCE'] = str(args.lookback_distance)
        if args.max_segment_length:
            os.environ['PYTHON_MAX_SEGMENT_LENGTH'] = str(args.max_segment_length)
        if args.angle_sum_power:
            os.environ['PYTHON_ANGLE_SUM_POWER'] = str(args.angle_sum_power)

        # Telescope mode
        # NOTE: We do NOT pass TELESCOPE_LEVER_MD to executor - telescope segment creation
        # has issues with bounds/initialization. Instead, we only use lever_md to calculate
        # first_slice length, then executor runs in normal mode with big first_slice.
        if args.telescope:
            os.environ['TELESCOPE_MODE'] = 'true'
            # TELESCOPE_LEVER_MD intentionally NOT set - use normal mode with calculated first_slice
            os.environ['TELESCOPE_WORK_SEGMENT_LENGTH'] = str(args.telescope_work_length)
            os.environ['TELESCOPE_WORK_SEGMENTS_COUNT'] = str(args.telescope_work_count)

        executor = GpuAutoGeosteeringExecutor(
            work_dir=work_dir,
            results_dir=str(results_dir)
        )

        if args.algorithm:
            executor.set_algorithm(args.algorithm)

        executor.start_daemon()

        for well_name in wells_to_process:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {well_name}")
            logger.info(f"{'='*60}")

            data = dataset[well_name]
            data['well_name'] = well_name

            try:
                # Auto-detect telescope lever MD if requested
                telescope_lever_md = args.telescope_lever_md
                if args.telescope and telescope_lever_md is None and args.telescope_auto_detect:
                    lookback = args.lookback_distance or float(os.getenv('PYTHON_LOOKBACK_DISTANCE', 250))
                    telescope_lever_md = detect_telescope_lever_md(data, args.telescope_auto_detect, lookback)
                    # NOTE: telescope mode disabled - lever segments produce wrong shifts
                    # executor.set_telescope_lever_md(telescope_lever_md)
                    logger.info(f"Auto-detected telescope lever MD: {telescope_lever_md:.1f}m ({telescope_lever_md * 3.28084:.0f}ft) using {args.telescope_auto_detect}")

                # Auto-calculate first_slice for telescope mode
                if args.telescope and args.first_slice_length is None and telescope_lever_md:
                    start_md = data.get('detected_start_md') or data.get('start_md', 0.0) or 0.0
                    lever_length = telescope_lever_md - start_md
                    first_slice = lever_length + (args.telescope_work_count - 1) * args.telescope_work_length
                    logger.info(f"Telescope auto first_slice: {first_slice:.1f}m = ({telescope_lever_md:.1f} - {start_md:.1f}) + ({args.telescope_work_count}-1) * {args.telescope_work_length}")
                else:
                    first_slice = args.first_slice_length if args.first_slice_length else args.slice_step
                result = run_slicing(data, executor, args.slice_step, first_slice)
                metrics = compare_with_reference(result, data)
                result['metrics'] = metrics
                all_results.append(result)

                logger.info(f"Completed {well_name}: {result['iterations']} iters, "
                            f"{result['total_time']:.1f}s, MAE={metrics.get('mae', 0):.2f}m, "
                            f"RMSE={metrics.get('rmse', 0):.2f}m, endpoint={metrics.get('endpoint_error', 0):+.2f}m")

            except Exception as e:
                logger.error(f"Error processing {well_name}: {e}")
                import traceback
                traceback.print_exc()

        executor.stop_daemon()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Wells processed: {len(all_results)}")

    if all_results:
        total_iters = sum(r['iterations'] for r in all_results)
        total_time = sum(r['total_time'] for r in all_results)
        print(f"Total iterations: {total_iters}")
        print(f"Total time: {total_time:.1f}s")

        # Collect metrics
        maes = [r['metrics'].get('mae') for r in all_results if r['metrics'].get('mae') is not None]
        rmses = [r['metrics'].get('rmse') for r in all_results if r['metrics'].get('rmse') is not None]
        endpoints = [r['metrics'].get('endpoint_error') for r in all_results if r['metrics'].get('endpoint_error') is not None]

        if maes:
            print(f"MAE: mean={np.mean(maes):.2f}m, min={np.min(maes):.2f}m, max={np.max(maes):.2f}m")
        if rmses:
            overall_rmse = np.sqrt(np.mean(np.array(rmses)**2))
            print(f"RMSE: mean={np.mean(rmses):.2f}m, overall={overall_rmse:.2f}m")
        if endpoints:
            endpoint_rmse = np.sqrt(np.mean(np.array(endpoints)**2))
            print(f"Endpoint errors: mean={np.mean(endpoints):+.2f}m, RMSE={endpoint_rmse:.2f}m")

        # Save results to CSV
        csv_path = results_dir / 'metrics_summary.csv'
        import csv
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['well_name', 'iterations', 'time_s', 'mae_m', 'rmse_m', 'max_error_m',
                             'max_error_md', 'endpoint_error_m', 'endpoint_md', 'n_segments'])
            for r in sorted(all_results, key=lambda x: x['metrics'].get('mae', 0), reverse=True):
                m = r['metrics']
                writer.writerow([
                    r['well_name'],
                    r['iterations'],
                    f"{r['total_time']:.1f}",
                    f"{m.get('mae', 0):.3f}",
                    f"{m.get('rmse', 0):.3f}",
                    f"{m.get('max_error', 0):.3f}",
                    f"{m.get('max_error_md', 0):.1f}",
                    f"{m.get('endpoint_error', 0):.3f}",
                    f"{m.get('endpoint_md', 0):.1f}",
                    m.get('n_result_segments', 0)
                ])
        print(f"\nResults saved to: {csv_path}")


if __name__ == '__main__':
    main()
