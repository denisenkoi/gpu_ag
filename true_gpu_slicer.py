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
from typing import Dict, Any, List, Optional
from copy import deepcopy

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / "cpu_baseline" / ".env")
except ImportError:
    pass  # dotenv is optional

from gpu_executor import GpuAutoGeosteeringExecutor

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

        # Typewell is pre-stitched in dataset - use as-is (StarSteer doesn't multiply typewell)
        'typeLog': {
            'tvdSortedPoints': tensor_to_typelog_points(
                data['typewell_tvd'],
                data['typewell_gr']
            )
        },

        # When USE_PSEUDOTYPELOG=true, tvd_shift is always 0 (as in StarSteer)
        'tvdTypewellShift': 0.0 if os.getenv('USE_PSEUDOTYPELOG', 'true').lower() == 'true' else data.get('tvd_typewell_shift', 0.0),

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

    segments = []
    for i in range(len(ref_mds)):
        seg_start_md = float(ref_mds[i])

        # Stop if segment starts after target_md
        if seg_start_md >= target_md:
            break

        seg_end_md = float(ref_mds[i + 1]) if i + 1 < len(ref_mds) else seg_start_md + 100
        start_shift = float(ref_shifts[i - 1]) if i > 0 else 0.0
        end_shift = float(ref_shifts[i])

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
                slice_step: float = 30.0) -> Dict[str, Any]:
    """
    Run incremental slicing on a single well.

    Args:
        data: Well data from dataset (with well_name added)
        executor: GPU executor instance
        slice_step: MD step between slices (meters)

    Returns:
        Dict with final interpretation and metrics
    """
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

    current_md = start_md + slice_step  # First slice includes some data

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


def compare_with_reference(result: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare slicing result with reference interpretation.

    Returns metrics dict.
    """
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_shifts = data['ref_shifts'].cpu().numpy()

    if len(ref_shifts) == 0:
        logger.warning("No reference interpretation for comparison")
        return {'error': 'no_reference'}

    # Get final interpretation shifts
    final_segments = result['interpretation']
    if not final_segments:
        return {'error': 'no_result_segments'}

    # Compare end shifts
    result_shifts = [seg['endShift'] for seg in final_segments]

    # Simple MAE on overlapping region
    n_compare = min(len(result_shifts), len(ref_shifts))
    if n_compare == 0:
        return {'error': 'no_overlap'}

    mae = np.mean(np.abs(np.array(result_shifts[:n_compare]) - ref_shifts[:n_compare]))

    return {
        'mae_shifts': float(mae),
        'n_result_segments': len(result_shifts),
        'n_ref_segments': len(ref_shifts),
        'n_compared': n_compare
    }


def load_dataset(path: Path) -> Dict[str, Dict]:
    """Load PyTorch dataset"""
    logger.info(f"Loading dataset from {path}")
    dataset = torch.load(path, weights_only=False)
    logger.info(f"Loaded {len(dataset)} wells")
    return dataset


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

    args = parser.parse_args()

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
        wells_to_process = list(dataset.keys())
    else:
        # Default: first well
        wells_to_process = [list(dataset.keys())[0]]
        logger.info(f"No well specified, using first well: {wells_to_process[0]}")

    # Create executor
    work_dir = str(results_dir / "work")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    executor = GpuAutoGeosteeringExecutor(
        work_dir=work_dir,
        results_dir=str(results_dir)
    )

    if args.algorithm:
        executor.set_algorithm(args.algorithm)

    executor.start_daemon()

    # Process wells
    all_results = []

    for well_name in wells_to_process:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {well_name}")
        logger.info(f"{'='*60}")

        data = dataset[well_name]
        data['well_name'] = well_name  # Add name to data

        try:
            result = run_slicing(data, executor, args.slice_step)

            # Compare with reference
            metrics = compare_with_reference(result, data)
            result['metrics'] = metrics

            all_results.append(result)

            logger.info(f"Completed {well_name}: {result['iterations']} iterations, "
                        f"{result['total_time']:.1f}s total, MAE={metrics.get('mae_shifts', 'N/A')}")

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

        # MAE stats
        maes = [r['metrics'].get('mae_shifts') for r in all_results
                if r['metrics'].get('mae_shifts') is not None]
        if maes:
            print(f"MAE shifts: mean={np.mean(maes):.2f}m, "
                  f"min={np.min(maes):.2f}m, max={np.max(maes):.2f}m")


if __name__ == '__main__':
    main()
