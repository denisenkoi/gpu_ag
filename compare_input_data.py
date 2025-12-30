"""
Compare input data between true_gpu_slicer and StarSteer.

Run 1 iteration and dump well_data for comparison.

Usage:
    python compare_input_data.py --well Well162~EGFDL --dump-json
    python compare_input_data.py --well Well162~EGFDL --compare starsteer_export.json
"""

import sys
import os
import torch
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


def load_dataset(path: Path) -> Dict[str, Any]:
    """Load PyTorch dataset."""
    return torch.load(path, weights_only=False)


def slice_data_to_md(data: Dict[str, Any], current_md: float) -> Dict[str, Any]:
    """
    Slice well data up to current_md and return in well_data format.
    Standalone version without gpu_executor dependency.
    """
    # Extract tensors
    md = data['md'].cpu().numpy()
    tvd = data['tvd'].cpu().numpy()
    vs = data['vs'].cpu().numpy()
    gr = data['gr'].cpu().numpy()
    traj_md = data['traj_md'].cpu().numpy()
    traj_ns = data['traj_ns'].cpu().numpy()
    traj_ew = data['traj_ew'].cpu().numpy()
    typewell_tvd = data['typewell_tvd'].cpu().numpy()
    typewell_gr = data['typewell_gr'].cpu().numpy()

    # Slice well log up to current_md
    log_mask = md <= current_md
    sliced_md = md[log_mask]
    sliced_tvd = tvd[log_mask]
    sliced_vs = vs[log_mask]
    sliced_gr = gr[log_mask]

    # Slice trajectory
    traj_mask = traj_md <= current_md
    sliced_traj_md = traj_md[traj_mask]
    sliced_traj_ns = traj_ns[traj_mask]
    sliced_traj_ew = traj_ew[traj_mask]

    # Interpolate TVD at trajectory points
    sliced_traj_tvd = np.interp(sliced_traj_md, sliced_md, sliced_tvd)

    # Build well_data structure
    well_points = []
    for i in range(len(sliced_traj_md)):
        well_points.append({
            'measuredDepth': float(sliced_traj_md[i]),
            'trueVerticalDepth': float(sliced_traj_tvd[i]),
            'northSouth': float(sliced_traj_ns[i]),
            'eastWest': float(sliced_traj_ew[i]),
        })

    welllog_points = []
    for i in range(len(sliced_md)):
        welllog_points.append({
            'measuredDepth': float(sliced_md[i]),
            'data': float(sliced_gr[i]),
        })

    typelog_points = []
    for i in range(len(typewell_tvd)):
        typelog_points.append({
            'trueVerticalDepth': float(typewell_tvd[i]),
            'data': float(typewell_gr[i]),
        })

    start_md = data.get('detected_start_md') or data.get('start_md', 0.0)

    well_data = {
        'wellName': data.get('well_name', 'Unknown'),
        'well': {'points': well_points},
        'wellLog': {'points': welllog_points},
        'typeLog': {'tvdSortedPoints': typelog_points},
        'interpretation': {'segments': []},
        'tvdTypewellShift': data.get('tvd_typewell_shift', 0.0),
        'autoGeosteeringParameters': {
            'startMd': float(start_md),
            'lookBackDistance': 200.0,
        },
    }

    return well_data


def build_manual_interpretation_to_md(data: Dict[str, Any], target_md: float) -> List[Dict]:
    """Build manual interpretation segments from ref_shifts up to target_md."""
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_shifts = data['ref_shifts'].cpu().numpy()

    segments = []
    for i in range(len(ref_mds)):
        if ref_mds[i] > target_md:
            break
        segments.append({
            'endMeasuredDepth': float(ref_mds[i]),
            'tvdShift': float(ref_shifts[i]),
        })

    return segments


def interpolate_shift_at_md(data: Dict[str, Any], target_md: float) -> float:
    """Interpolate shift from ref_shifts at given MD."""
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_shifts = data['ref_shifts'].cpu().numpy()

    if len(ref_mds) == 0:
        return 0.0

    return float(np.interp(target_md, ref_mds, ref_shifts))


def dump_well_data_for_iteration(data: dict, iteration: int = 1, slice_step: float = 30.0) -> dict:
    """
    Build well_data for given iteration (without running optimization).
    """
    start_md = data.get('detected_start_md') or data.get('start_md', 0.0)
    current_md = start_md + slice_step * iteration

    well_data = slice_data_to_md(data, current_md)

    # Add manual interpretation
    manual_interp = build_manual_interpretation_to_md(data, start_md)
    well_data['interpretation'] = {'segments': manual_interp}

    return well_data, current_md


def compare_arrays(name: str, arr1: np.ndarray, arr2: np.ndarray, tol: float = 1e-6):
    """Compare two arrays and print differences."""
    if len(arr1) != len(arr2):
        print(f"  {name}: LENGTH MISMATCH! {len(arr1)} vs {len(arr2)}")
        min_len = min(len(arr1), len(arr2))
        arr1 = arr1[:min_len]
        arr2 = arr2[:min_len]

    diff = np.abs(arr1 - arr2)
    max_diff = diff.max()
    mean_diff = diff.mean()

    if max_diff > tol:
        print(f"  {name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
        # Show first few differences
        bad_idx = np.where(diff > tol)[0][:5]
        for i in bad_idx:
            print(f"    [{i}]: {arr1[i]:.6f} vs {arr2[i]:.6f} (diff={diff[i]:.6f})")
    else:
        print(f"  {name}: OK (max_diff={max_diff:.2e})")


def compare_well_data(ours: dict, theirs: dict):
    """Compare well_data structures."""
    print("\n=== WELL DATA COMPARISON ===\n")

    # Well points
    our_well = ours['well']['points']
    their_well = theirs.get('well', {}).get('points', [])

    print(f"Well points: ours={len(our_well)}, theirs={len(their_well)}")

    if our_well and their_well:
        our_md = np.array([p['measuredDepth'] for p in our_well])
        their_md = np.array([p['measuredDepth'] for p in their_well])
        compare_arrays("well.MD", our_md, their_md)

        our_tvd = np.array([p['trueVerticalDepth'] for p in our_well])
        their_tvd = np.array([p['trueVerticalDepth'] for p in their_well])
        compare_arrays("well.TVD", our_tvd, their_tvd)

        our_ns = np.array([p['northSouth'] for p in our_well])
        their_ns = np.array([p['northSouth'] for p in their_well])
        compare_arrays("well.NS", our_ns, their_ns)

        our_ew = np.array([p['eastWest'] for p in our_well])
        their_ew = np.array([p['eastWest'] for p in their_well])
        compare_arrays("well.EW", our_ew, their_ew)

    # WellLog points
    our_log = ours['wellLog']['points']
    their_log = theirs.get('wellLog', {}).get('points', [])

    print(f"\nWellLog points: ours={len(our_log)}, theirs={len(their_log)}")

    if our_log and their_log:
        our_md = np.array([p['measuredDepth'] for p in our_log])
        their_md = np.array([p['measuredDepth'] for p in their_log])
        compare_arrays("wellLog.MD", our_md, their_md)

        our_gr = np.array([p['data'] for p in our_log])
        their_gr = np.array([p['data'] for p in their_log])
        compare_arrays("wellLog.GR", our_gr, their_gr)

    # TypeLog points
    our_type = ours['typeLog']['tvdSortedPoints']
    their_type = theirs.get('typeLog', {}).get('tvdSortedPoints', [])

    print(f"\nTypeLog points: ours={len(our_type)}, theirs={len(their_type)}")

    if our_type and their_type:
        our_tvd = np.array([p['trueVerticalDepth'] for p in our_type])
        their_tvd = np.array([p['trueVerticalDepth'] for p in their_type])
        compare_arrays("typeLog.TVD", our_tvd, their_tvd)

        our_gr = np.array([p['data'] for p in our_type])
        their_gr = np.array([p['data'] for p in their_type])
        compare_arrays("typeLog.GR", our_gr, their_gr)

    # Interpretation
    our_interp = ours.get('interpretation', {}).get('segments', [])
    their_interp = theirs.get('interpretation', {}).get('segments', [])

    print(f"\nInterpretation segments: ours={len(our_interp)}, theirs={len(their_interp)}")

    # Parameters
    print("\n=== PARAMETERS ===")
    print(f"tvdTypewellShift: ours={ours.get('tvdTypewellShift')}, theirs={theirs.get('tvdTypewellShift')}")

    our_params = ours.get('autoGeosteeringParameters', {})
    their_params = theirs.get('autoGeosteeringParameters', {})
    print(f"startMd: ours={our_params.get('startMd')}, theirs={their_params.get('startMd')}")


def main():
    parser = argparse.ArgumentParser(description='Compare input data')
    parser.add_argument('--well', type=str, required=True, help='Well name')
    parser.add_argument('--iteration', type=int, default=1, help='Iteration number')
    parser.add_argument('--slice-step', type=float, default=30.0, help='Slice step')
    parser.add_argument('--dump-json', action='store_true', help='Dump well_data to JSON')
    parser.add_argument('--compare', type=str, help='Compare with StarSteer JSON')
    parser.add_argument('--dataset', type=str,
                        default='/mnt/e/Projects/Rogii/gpu_ag/dataset/gpu_ag_dataset.pt')

    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(Path(args.dataset))

    if args.well not in dataset:
        print(f"Well not found: {args.well}")
        print(f"Available: {list(dataset.keys())[:10]}...")
        return

    data = dataset[args.well]
    data['well_name'] = args.well

    # Build well_data for iteration
    well_data, current_md = dump_well_data_for_iteration(
        data, args.iteration, args.slice_step
    )

    print(f"Well: {args.well}")
    print(f"Iteration: {args.iteration}")
    print(f"Current MD: {current_md:.2f}m")
    print(f"Well points: {len(well_data['well']['points'])}")
    print(f"WellLog points: {len(well_data['wellLog']['points'])}")
    print(f"TypeLog points: {len(well_data['typeLog']['tvdSortedPoints'])}")
    print(f"Interpretation segments: {len(well_data['interpretation']['segments'])}")

    if args.dump_json:
        output_file = f"{args.well}_iter{args.iteration}_input.json"
        with open(output_file, 'w') as f:
            json.dump(well_data, f, indent=2)
        print(f"\nDumped to: {output_file}")

    if args.compare:
        print(f"\nComparing with: {args.compare}")
        with open(args.compare) as f:
            their_data = json.load(f)
        compare_well_data(well_data, their_data)


if __name__ == '__main__':
    main()
