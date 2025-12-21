"""
Test GPU DE - correct version based on test_de_quick.py from multidrilling_emulator.

Reads from slicing_well.json and creates segments via create_segments().
Uses stitch_shift from manual interpretation at lookback point.
NO DATA LEAKAGE - we only use manual interpretation BEFORE lookback, not after.

IMPORTANT: interpretation.json is WRITE ONLY - never read from it!
Always read interpretation from slicing_well.json.
"""
import sys
import os
import json
import time
import math
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy
from dotenv import load_dotenv

# Load .env from gpu_ag directory
load_dotenv(Path(__file__).parent / ".env")

sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments, Segment

from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
from torch_funcs.batch_objective import TorchObjectiveWrapper
# EvoTorch for optimization
from evotorch import Problem
from evotorch.algorithms import SNES
from ag_numerical.ag_optimizer_utils import calculate_optimization_bounds


# Slicer parameters (from python_autogeosteering_executor.py)
SEGMENTS_COUNT = 4
MAX_SEGMENT_LENGTH = 50.0  # meters
ANGLE_RANGE = 5.0
LOOKBACK_DISTANCE = 200.0  # meters
STEP_DISTANCE = 30.0  # meters - step between iterations

# Conversion
METERS_TO_FEET = 3.28084


def load_data():
    """Load data from slicing_well.json and normalize.

    If USE_PSEUDOTYPELOG=true in .env:
    - Uses pseudoTypeLog instead of typeLog
    - Sets tvdTypewellShift to 0
    """
    base_dir = Path("/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")

    with open(base_dir / "AG_DATA/InitialData/slicing_well.json", 'r') as f:
        well_data = json.load(f)

    # Check if we should use pseudoTypeLog
    use_pseudo = os.getenv('USE_PSEUDOTYPELOG', 'false').lower() == 'true'

    if use_pseudo and 'pseudoTypeLog' in well_data:
        print("Using pseudoTypeLog instead of typeLog (tvdTypewellShift=0)")
        # Replace typeLog with pseudoTypeLog for TypeWell creation
        well_data_for_typewell = dict(well_data)
        well_data_for_typewell['typeLog'] = well_data['pseudoTypeLog']
        typewell = TypeWell(well_data_for_typewell)
        tvd_shift = 0.0  # No shift for pseudo
    else:
        if use_pseudo:
            print("WARNING: USE_PSEUDOTYPELOG=true but pseudoTypeLog not found, using typeLog")
        typewell = TypeWell(well_data)
        tvd_shift = well_data.get('tvdTypewellShift', 0.0)

    well = Well(well_data)

    # Save original values before normalization
    min_md_original = well.min_md
    md_range = well.md_range

    # IMPORTANT: Normalize well and typewell (like executor does)
    # This normalizes MD, VS, TVD by md_range so angles are not distorted
    max_curve_value = max(well.max_curve, typewell.value.max())
    well.normalize(max_curve_value, typewell.min_depth, md_range)
    typewell.normalize(max_curve_value, well.min_depth, md_range)

    tvd_shift_norm = tvd_shift / md_range

    # Get current_md from autoGeosteeringParameters.startMd
    current_md = well_data.get('autoGeosteeringParameters', {}).get('startMd', min_md_original + 100)

    return well, typewell, well_data, tvd_shift_norm, md_range, min_md_original, current_md


def truncate_interpretation_at_md(interpretation_segments, truncate_md_m, well, md_range, min_md_original):
    """
    Truncate interpretation at given MD, return (truncated_segments, stitch_shift).

    Logic from python_autogeosteering_executor._truncate_interpretation_at_md():
    - Keep segments that END before truncate_md
    - Truncate segment that crosses truncate_md, interpolating its end_shift
    - Return stitch_shift for seamless connection

    Args:
        interpretation_segments: List of segment dicts from JSON
        truncate_md_m: MD value in meters to truncate at
        well: Normalized Well object
        md_range: Original MD range in meters
        min_md_original: Original min MD in meters

    Returns:
        (truncated_segments, stitch_shift) where stitch_shift is normalized
    """
    if not interpretation_segments:
        return [], 0.0

    # Convert to normalized MD for well operations
    truncate_md_norm = (truncate_md_m - min_md_original) / md_range
    truncate_idx = well.md2idx(truncate_md_norm)
    truncate_md_value = well.measured_depth[truncate_idx]  # Normalized
    truncate_md_actual = truncate_md_value * md_range + min_md_original  # Back to meters for comparison

    print(f"  Truncate at MD={truncate_md_m:.1f}m, idx={truncate_idx}, actual={truncate_md_actual:.1f}m")

    truncated = []
    stitch_shift = 0.0

    for i, seg in enumerate(interpretation_segments):
        seg_start_md = seg.get('startMd', 0)
        seg_start_shift = seg.get('startShift', 0.0)
        seg_end_shift = seg.get('endShift', 0.0)

        # Get end_md: either from segment or from next segment's start
        if 'endMd' in seg:
            seg_end_md = seg['endMd']
        elif i + 1 < len(interpretation_segments):
            seg_end_md = interpretation_segments[i + 1].get('startMd', seg_start_md)
        else:
            # Last segment - approximate end
            seg_end_md = seg_start_md + 100

        if seg_end_md <= truncate_md_m:
            # Segment fully before truncate point - keep as is
            truncated.append(dict(seg))
            stitch_shift = seg_end_shift
            print(f"    Seg[{i}] MD={seg_start_md:.1f}-{seg_end_md:.1f}m -> KEEP, stitch_shift={stitch_shift:.4f}m")
        elif seg_start_md < truncate_md_m < seg_end_md:
            # Segment crosses truncate point - truncate and interpolate
            ratio = (truncate_md_m - seg_start_md) / (seg_end_md - seg_start_md)
            interpolated_shift = seg_start_shift + ratio * (seg_end_shift - seg_start_shift)

            truncated_seg = dict(seg)
            truncated_seg['endMd'] = truncate_md_m
            truncated_seg['endShift'] = interpolated_shift
            truncated.append(truncated_seg)
            stitch_shift = interpolated_shift
            print(f"    Seg[{i}] MD={seg_start_md:.1f}-{seg_end_md:.1f}m -> TRUNCATE at {truncate_md_m:.1f}m, "
                  f"ratio={ratio:.3f}, stitch_shift={stitch_shift:.4f}m")
            break
        elif seg_start_md >= truncate_md_m:
            # Segment starts at or after truncate - stop
            print(f"    Seg[{i}] MD={seg_start_md:.1f}m >= truncate -> STOP")
            break

    # Normalize stitch_shift (it's in meters from JSON)
    stitch_shift_norm = stitch_shift / md_range

    print(f"  Truncated: {len(truncated)} segments, stitch_shift={stitch_shift:.4f}m ({stitch_shift_norm:.6f} norm)")
    return truncated, stitch_shift_norm


def create_test_segments(well, lookback_md_m, current_md_m, stitch_shift, md_range, min_md_original,
                         segments_count=SEGMENTS_COUNT, max_segment_length=MAX_SEGMENT_LENGTH):
    """
    Create segments from lookback to current_md with start_shift=stitch_shift.

    Based on _create_and_append_segments from python_autogeosteering_executor.py

    Args:
        well: Normalized Well object
        lookback_md_m: Lookback MD in meters
        current_md_m: Current MD in meters
        stitch_shift: Starting shift (normalized) from truncation point
        md_range: Original MD range in meters
        min_md_original: Original min MD in meters
    """
    # Convert to normalized MD
    lookback_md_norm = (lookback_md_m - min_md_original) / md_range
    current_md_norm = (current_md_m - min_md_original) / md_range

    # Get indices
    lookback_idx = well.md2idx(lookback_md_norm)
    current_idx = well.md2idx(current_md_norm)

    new_length_idx = current_idx - lookback_idx
    new_length_m = new_length_idx * well.horizontal_well_step  # horizontal_well_step is in meters

    # Dynamic segment count (like slicer)
    dynamic_segments = math.ceil(new_length_m / max_segment_length)
    actual_count = min(dynamic_segments, segments_count)
    actual_count = max(actual_count, 1)

    segment_length_idx = new_length_idx // actual_count

    print(f"Creating segments:")
    print(f"  lookback_md={lookback_md_m:.1f}m, current_md={current_md_m:.1f}m")
    print(f"  lookback_idx={lookback_idx}, current_idx={current_idx}")
    print(f"  new_length={new_length_m:.1f}m, segments={actual_count}")
    print(f"  start_shift={stitch_shift:.6f} (normalized) = {stitch_shift * md_range:.4f}m")

    # Create segments starting from stitch_shift
    segments = create_segments(
        well=well,
        segments_count=actual_count,
        segment_len=segment_length_idx,
        start_idx=lookback_idx,
        start_shift=stitch_shift  # From truncation point - NOT zero!
    )

    return segments


def get_reference_shifts_for_segments(ref_segments, segment_end_mds_m, start_md_m, md_range):
    """
    Get reference shifts for optimization segment ends by interpolating starred interpretation.

    Args:
        ref_segments: List of reference segment dicts (starredInterpretation)
        segment_end_mds_m: List of end MD values for optimization segments (meters)
        start_md_m: Start MD of optimization region (meters)
        md_range: MD range for normalization

    Returns:
        List of normalized shifts for each segment end
    """
    ref_shifts_m = []

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
    return ref_shifts_m, ref_shifts_norm


def create_wrapper_and_bounds(well, typewell, segments, tvd_shift, device='cuda'):
    """Create TorchObjectiveWrapper and bounds"""
    bounds = calculate_optimization_bounds(segments, angle_range=ANGLE_RANGE, accumulative=True)
    bounds_tensor = torch.tensor(bounds, dtype=torch.float64, device=device)

    # Calculate horizontal projection
    well.calc_horizontal_projection(typewell, segments, tvd_shift)

    # Convert to numpy
    well_np = well_to_numpy(well)
    typewell_np = typewell_to_numpy(typewell)
    segments_np = segments_to_numpy(segments, well)

    # Convert to torch
    well_torch = numpy_to_torch(well_np, device=device)
    typewell_torch = numpy_to_torch(typewell_np, device=device)
    segments_torch = segments_numpy_to_torch(segments_np, device=device)

    wrapper = TorchObjectiveWrapper(
        well_data=well_torch,
        typewell_data=typewell_torch,
        segments_torch=segments_torch,
        self_corr_start_idx=segments[0].start_idx,
        pearson_power=2.0,
        mse_power=0.001,
        num_intervals_self_correlation=0,
        sc_power=1.15,
        angle_range=ANGLE_RANGE,
        angle_sum_power=2.0,
        min_pearson_value=-1.0,
        tvd_to_typewell_shift=tvd_shift,
        device=device
    )

    return wrapper, bounds_tensor


def export_to_starsteer(truncated_manual, optimized_segments, md_range, min_md_original, output_path):
    """
    Export stitched interpretation to StarSteer.

    Args:
        truncated_manual: List of truncated manual segment dicts (in meters)
        optimized_segments: List of optimized Segment objects (normalized)
        md_range: Original MD range in meters
        min_md_original: Original min MD in meters
        output_path: Path to output JSON file
    """
    all_segments = list(truncated_manual)  # Copy manual segments

    # Convert optimized segments to JSON format
    for seg in optimized_segments:
        # Segment values are NORMALIZED - convert to meters
        start_md_m = seg.start_md * md_range + min_md_original
        end_md_m = seg.end_md * md_range + min_md_original
        start_shift_m = seg.start_shift * md_range
        end_shift_m = seg.end_shift * md_range

        all_segments.append({
            'startMd': start_md_m,
            'endMd': end_md_m,
            'startShift': start_shift_m,
            'endShift': end_shift_m
        })

    print(f"  Total segments: {len(truncated_manual)} manual + {len(optimized_segments)} optimized = {len(all_segments)}")

    # Export
    output_data = {
        'interpretation': {
            'segments': all_segments
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"  Exported to: {output_path}")
    return all_segments


def update_slicing_well(slicing_well_path, interpretation_segments):
    """
    Update slicing_well.json with new interpretation (standalone mode only).

    Args:
        slicing_well_path: Path to slicing_well.json
        interpretation_segments: List of segment dicts (in meters)
    """
    with open(slicing_well_path, 'r') as f:
        data = json.load(f)

    data['interpretation']['segments'] = interpretation_segments

    with open(slicing_well_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"  Updated slicing_well.json with {len(interpretation_segments)} segments")


def optimize_step(current_md_m, interpretation_segments, frozen_md_m,
                  well, typewell, well_data, tvd_shift, md_range, min_md_original,
                  verbose=True):
    """
    Optimize one step: from lookback to current_md.

    Args:
        current_md_m: Current MD position (meters)
        interpretation_segments: Current interpretation (list of segment dicts in meters)
        frozen_md_m: MD before which interpretation is frozen (meters)
        well: Normalized Well object
        typewell: Normalized TypeWell object
        well_data: Original well data dict (for reference interpretation)
        tvd_shift: TVD shift (normalized)
        md_range: Original MD range (meters)
        min_md_original: Original min MD (meters)
        verbose: Print progress

    Returns:
        (updated_interpretation, metrics) where:
        - updated_interpretation: List of segment dicts (in meters)
        - metrics: dict with 'fun', 'time', 'ref_fun', etc.
    """
    lookback_md_m = current_md_m - LOOKBACK_DISTANCE
    lookback_md_m = max(lookback_md_m, min_md_original)

    if verbose:
        print(f"\n--- Step at current_md={current_md_m:.1f}m, lookback={lookback_md_m:.1f}m ---")

    # Truncate interpretation at lookback (but not before frozen_md)
    truncate_at = max(lookback_md_m, frozen_md_m)
    truncated, stitch_shift = truncate_interpretation_at_md(
        interpretation_segments, truncate_at, well, md_range, min_md_original
    )

    # Create segments for optimization
    segments = create_test_segments(
        well, lookback_md_m, current_md_m, stitch_shift, md_range, min_md_original
    )

    if verbose:
        print(f"  Created {len(segments)} segments, stitch_shift={stitch_shift * md_range:.4f}m")

    # Create wrapper and bounds
    wrapper, bounds = create_wrapper_and_bounds(well, typewell, segments, tvd_shift)

    # Get reference shifts for comparison
    segment_end_mds_m = [seg.end_md * md_range + min_md_original for seg in segments]
    ref_segments = well_data.get('starredInterpretation', {}).get('segments', [])
    ref_shifts_m, ref_shifts_norm = get_reference_shifts_for_segments(
        ref_segments, segment_end_mds_m, lookback_md_m, md_range
    )

    # Calculate reference fun
    ref_shifts_tensor = torch.tensor([ref_shifts_norm], dtype=torch.float64, device='cuda')
    ref_fun = wrapper(ref_shifts_tensor).item()

    # Run EvoTorch SNES optimization
    K = len(segments)
    lb = bounds[:, 0].cpu().numpy()
    ub = bounds[:, 1].cpu().numpy()
    stdev_init = (ub - lb).mean() / 4.0

    class AGProblem(Problem):
        def __init__(self):
            super().__init__(
                objective_sense="min",
                solution_length=K,
                initial_bounds=(lb, ub),
                dtype=torch.float64,
                device='cuda'
            )

        def _evaluate_batch(self, solutions):
            x = solutions.values
            fitness = wrapper(x)
            solutions.set_evals(fitness)

    N_RESTARTS = 5
    POPSIZE = 100
    MAXITER = 200

    start_time = time.time()
    best_overall_fun = float('inf')
    best_overall_x = None

    for restart in range(N_RESTARTS):
        problem = AGProblem()
        searcher = SNES(
            problem,
            popsize=POPSIZE,
            stdev_init=stdev_init,
            center_learning_rate=0.5,
            stdev_learning_rate=0.1
        )

        for _ in range(MAXITER):
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

    # Apply optimized shifts to segments
    optimal_shifts = best_overall_x.cpu().tolist()
    optimized_segments = deepcopy(segments)
    for i, shift in enumerate(optimal_shifts):
        optimized_segments[i].end_shift = shift
        if i < len(optimized_segments) - 1:
            optimized_segments[i + 1].start_shift = shift

    # Build updated interpretation: truncated + optimized
    updated_interpretation = list(truncated)
    for seg in optimized_segments:
        start_md_m = seg.start_md * md_range + min_md_original
        end_md_m = seg.end_md * md_range + min_md_original
        start_shift_m = seg.start_shift * md_range
        end_shift_m = seg.end_shift * md_range
        updated_interpretation.append({
            'startMd': start_md_m,
            'endMd': end_md_m,
            'startShift': start_shift_m,
            'endShift': end_shift_m
        })

    metrics = {
        'fun': best_overall_fun,
        'ref_fun': ref_fun,
        'time': elapsed,
        'n_segments': len(segments),
        'current_md': current_md_m,
        'lookback_md': lookback_md_m
    }

    if verbose:
        status = "✓" if best_overall_fun <= ref_fun * 1.1 else "✗"
        print(f"  {status} fun={best_overall_fun:.4f} (ref={ref_fun:.4f}), time={elapsed:.2f}s")

    return updated_interpretation, metrics


def main(n_iterations=1, standalone=True):
    """
    Run full well interpretation.

    Args:
        n_iterations: Number of steps (0 = until end of well)
        standalone: If True, update slicing_well.json after each step (test mode)
                    If False, only write interpretation.json (prod mode with StarSteer)
    """
    print("="*60)
    print("GPU DE Test - Full Well Interpretation")
    print("="*60)

    # Load data
    print("\n--- Loading data ---")
    well, typewell, well_data, tvd_shift, md_range, min_md_original, start_md_m = load_data()

    max_md_m = min_md_original + md_range
    print(f"Well MD: {min_md_original:.1f} - {max_md_m:.1f}m")
    print(f"MD range: {md_range:.1f}m ({md_range * METERS_TO_FEET:.1f}ft)")
    print(f"Points: {len(well.measured_depth)}")
    print(f"Start MD: {start_md_m:.1f}m")
    print(f"Step: {STEP_DISTANCE}m, Lookback: {LOOKBACK_DISTANCE}m")
    print(f"Mode: {'standalone' if standalone else 'with StarSteer'}")

    # Initialize
    current_md_m = start_md_m
    frozen_md_m = start_md_m - LOOKBACK_DISTANCE  # Don't modify interpretation before this
    frozen_md_m = max(frozen_md_m, min_md_original)

    # Get initial interpretation from file
    interpretation = well_data.get('interpretation', {}).get('segments', [])
    print(f"Initial interpretation: {len(interpretation)} segments")

    # Paths
    base_dir = Path("/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")
    output_path = base_dir / "interpretation.json"
    slicing_well_path = base_dir / "AG_DATA/InitialData/slicing_well.json"

    # Determine end condition
    if n_iterations == 0:
        end_md_m = max_md_m
        print(f"Running until end of well ({end_md_m:.1f}m)")
    else:
        end_md_m = current_md_m + n_iterations * STEP_DISTANCE
        end_md_m = min(end_md_m, max_md_m)
        print(f"Running {n_iterations} iterations until {end_md_m:.1f}m")

    # Iterate
    all_metrics = []
    iteration = 0
    total_start = time.time()

    while current_md_m <= end_md_m:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}: current_md={current_md_m:.1f}m")
        print(f"{'='*60}")

        # Optimize this step
        interpretation, metrics = optimize_step(
            current_md_m, interpretation, frozen_md_m,
            well, typewell, well_data, tvd_shift, md_range, min_md_original
        )
        all_metrics.append(metrics)

        # Export to interpretation.json (always)
        output_data = {'interpretation': {'segments': interpretation}}
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"  Exported to: {output_path}")

        # Update slicing_well.json (standalone mode only)
        if standalone:
            update_slicing_well(slicing_well_path, interpretation)

        # Move to next position
        current_md_m += STEP_DISTANCE

    total_elapsed = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total iterations: {len(all_metrics)}")
    print(f"Total time: {total_elapsed:.2f}s ({total_elapsed/len(all_metrics):.2f}s per step)")
    print(f"Final interpretation: {len(interpretation)} segments")

    # Success rate
    successes = sum(1 for m in all_metrics if m['fun'] <= m['ref_fun'] * 1.1)
    print(f"Success rate: {successes}/{len(all_metrics)} ({100*successes/len(all_metrics):.0f}%)")

    # Print per-step results
    print(f"\nPer-step results:")
    print(f"{'Step':>4} {'MD':>8} {'Fun':>8} {'Ref':>8} {'Time':>6} {'Status':>6}")
    print("-" * 50)
    for i, m in enumerate(all_metrics):
        status = "✓" if m['fun'] <= m['ref_fun'] * 1.1 else "✗"
        print(f"{i+1:>4} {m['current_md']:>8.1f} {m['fun']:>8.4f} {m['ref_fun']:>8.4f} {m['time']:>6.2f} {status:>6}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='GPU DE Full Well Interpretation')
    parser.add_argument('-n', '--iterations', type=int, default=1,
                        help='Number of iterations (0 = until end of well)')
    parser.add_argument('--no-standalone', action='store_true',
                        help='Disable standalone mode (do not update slicing_well.json)')
    args = parser.parse_args()

    main(n_iterations=args.iterations, standalone=not args.no_standalone)
