#!/usr/bin/env python3
"""
Full well optimization - extends multi-segment approach to entire well.

Strategy:
1. Start from OTSU zone (best GR region in last 1/3)
2. Optimize 5 segments with GPU brute-force
3. Use end_shift of last segment as start_shift for next block
4. Continue until end of well
5. Near end (last ~100m): reduce angle_range if Pearson is low

Based on param_search findings:
- Best: W=200 A=1.5° S=0 MSE^1.0 = 3.30m
- angle_range=1.5° or 1.0° optimal
"""

import sys
import os
import time
import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from torch_funcs.converters import GPU_DTYPE
from smart_segmenter import SmartSegmenter
from peak_detectors import OtsuPeakDetector, RegionFinder
from numpy_funcs.interpretation import interpolate_shift_at_md
from cpu_baseline.typewell_provider import stitch_typewell_from_dataset, compute_overlap_metrics

# Read typewell mode from env
# NEW = correct: TypeLog raw (reference), PseudoTypeLog × mult, Well GR × mult
# OLD = legacy: TypeLog × (1/mult), PseudoTypeLog raw, Well GR × mult
NORMALIZATION_MODE = os.getenv('NORMALIZATION_MODE', 'NEW')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class OptimizedSegment:
    """Result of segment optimization."""
    start_md: float
    end_md: float
    start_shift: float
    end_shift: float
    angle_deg: float
    pearson: float


def optimize_segment_block_gpu(
    segment_indices: List[Tuple[int, int]],
    start_shift: float,
    trajectory_angle: float,
    well_md: np.ndarray,
    well_tvd: np.ndarray,
    well_gr: np.ndarray,
    type_tvd: np.ndarray,
    type_gr: np.ndarray,
    tvd_shift: float,
    angle_range: float = 1.5,
    angle_step: float = 0.2,
    pearson_power: float = 1.0,
    pearson_clamp: float = 0.0,
    device: str = DEVICE,
    chunk_size: int = 250000,
) -> Tuple[float, float, np.ndarray, float]:
    """
    GPU optimization for a block of segments.

    Returns: (best_pearson, best_end_shift, best_angles, best_start_shift)
    """
    n_seg = len(segment_indices)

    # Generate angle candidates
    angle_steps = int(2 * angle_range / angle_step) + 1
    angles = np.linspace(
        trajectory_angle - angle_range,
        trajectory_angle + angle_range,
        angle_steps
    )

    # Fixed start_shift (no search)
    start_shifts_arr = np.array([start_shift])

    # Meshgrid
    grids = np.meshgrid(*[angles]*n_seg, start_shifts_arr, indexing='ij')
    all_angles_np = np.stack([g.ravel() for g in grids[:-1]], axis=1).astype(np.float32)
    all_start_shifts_np = grids[-1].ravel().astype(np.float32)
    n_combos = all_angles_np.shape[0]

    # Segment MD lengths
    seg_md_lens = np.array([well_md[e] - well_md[s] for s, e in segment_indices], dtype=np.float32)

    # Zone data
    start_idx = segment_indices[0][0]
    end_idx = segment_indices[-1][1]
    n_points = end_idx - start_idx

    zone_tvd = torch.tensor(well_tvd[start_idx:end_idx], device=device, dtype=GPU_DTYPE)
    zone_gr = torch.tensor(well_gr[start_idx:end_idx], device=device, dtype=GPU_DTYPE)
    zone_gr_centered = zone_gr - zone_gr.mean()
    zone_gr_ss = (zone_gr_centered**2).sum()

    type_tvd_t = torch.tensor(type_tvd + tvd_shift, device=device, dtype=GPU_DTYPE)
    type_gr_t = torch.tensor(type_gr, device=device, dtype=GPU_DTYPE)

    # Precompute segment data
    seg_data = []
    for seg_i, (s_idx, e_idx) in enumerate(segment_indices):
        local_start = s_idx - start_idx
        local_end = e_idx - start_idx
        seg_n = local_end - local_start
        seg_tvd = torch.tensor(well_tvd[s_idx:e_idx], device=device, dtype=GPU_DTYPE)

        md_start = well_md[s_idx]
        md_end = well_md[e_idx - 1] if e_idx > s_idx else md_start
        if md_end > md_start:
            ratio = torch.tensor((well_md[s_idx:e_idx] - md_start) / (md_end - md_start),
                                device=device, dtype=GPU_DTYPE)
        else:
            ratio = torch.zeros(seg_n, device=device, dtype=GPU_DTYPE)

        seg_data.append((local_start, local_end, seg_n, seg_tvd, ratio))

    best_score = -1e9
    best_idx_global = 0
    best_pearson = 0.0

    # Process in chunks
    for chunk_start in range(0, n_combos, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_combos)
        chunk_angles = torch.tensor(all_angles_np[chunk_start:chunk_end], device=device, dtype=GPU_DTYPE)
        chunk_start_shifts = torch.tensor(all_start_shifts_np[chunk_start:chunk_end], device=device, dtype=GPU_DTYPE)
        chunk_n = chunk_angles.shape[0]

        # Compute shifts
        seg_md_lens_t = torch.tensor(seg_md_lens, device=device, dtype=GPU_DTYPE)
        shift_deltas = torch.tan(torch.deg2rad(chunk_angles)) * seg_md_lens_t
        cumsum = torch.cumsum(shift_deltas, dim=1)
        end_shifts = chunk_start_shifts.unsqueeze(1) + cumsum
        start_shifts = torch.cat([
            chunk_start_shifts.unsqueeze(1),
            end_shifts[:, :-1]
        ], dim=1)

        # Build synthetic
        synthetic = torch.zeros((chunk_n, n_points), device=device, dtype=GPU_DTYPE)

        for seg_i, (local_start, local_end, seg_n, seg_tvd, ratio) in enumerate(seg_data):
            seg_start = start_shifts[:, seg_i:seg_i+1]
            seg_end = end_shifts[:, seg_i:seg_i+1]
            seg_shifts = seg_start + ratio.unsqueeze(0) * (seg_end - seg_start)

            tvt = seg_tvd.unsqueeze(0) - seg_shifts
            tvt_clamped = torch.clamp(tvt, type_tvd_t[0], type_tvd_t[-1])

            indices = torch.searchsorted(type_tvd_t, tvt_clamped.reshape(-1))
            indices = torch.clamp(indices, 1, len(type_tvd_t) - 1)

            tvd_low = type_tvd_t[indices - 1]
            tvd_high = type_tvd_t[indices]
            gr_low = type_gr_t[indices - 1]
            gr_high = type_gr_t[indices]

            t = (tvt_clamped.reshape(-1) - tvd_low) / (tvd_high - tvd_low + 1e-10)
            interp_gr = gr_low + t * (gr_high - gr_low)
            synthetic[:, local_start:local_end] = interp_gr.reshape(chunk_n, seg_n)

        # Pearson
        synthetic_centered = synthetic - synthetic.mean(dim=1, keepdim=True)
        numer = (zone_gr_centered * synthetic_centered).sum(dim=1)
        denom = torch.sqrt(zone_gr_ss * (synthetic_centered**2).sum(dim=1))
        pearsons = torch.where(denom > 1e-10, numer / denom, torch.zeros_like(numer))

        # Score: pearson - 0.1 * MSE_norm (best formula so far)
        mse = ((zone_gr - synthetic)**2).mean(dim=1)
        mse_norm = mse / (zone_gr.var() + 1e-10)
        scores = pearsons - 0.1 * mse_norm

        chunk_best_idx = torch.argmax(scores).item()
        chunk_best_score = scores[chunk_best_idx].item()

        if chunk_best_score > best_score:
            best_score = chunk_best_score
            best_idx_global = chunk_start + chunk_best_idx
            best_pearson = pearsons[chunk_best_idx].item()

    # Get best result
    best_start_shift = float(all_start_shifts_np[best_idx_global])
    best_angles = all_angles_np[best_idx_global]
    best_shift_deltas = np.tan(np.radians(best_angles)) * seg_md_lens
    best_end_shift = best_start_shift + np.sum(best_shift_deltas)

    return best_pearson, best_end_shift, best_angles, best_start_shift


def get_segment_boundaries(
    well_md: np.ndarray,
    well_gr: np.ndarray,
    start_md: float,
    end_md: float,
    max_segments: int = 5,
    settings: dict = None,
) -> List[Tuple[int, int]]:
    """
    Get segment boundaries using SmartSegmenter.
    Falls back to equal segments if SmartSegmenter fails.
    """
    if settings is None:
        settings = {
            'min_segment_length': 30.0,
            'overlap_segments': 0,
            'pelt_penalty': 10.0,
            'min_distance': 20.0,
            'segments_count': max_segments,
        }

    start_idx = int(np.searchsorted(well_md, start_md))
    end_idx = int(np.searchsorted(well_md, end_md))

    if end_idx - start_idx < 20:
        return [(start_idx, end_idx)]

    zone_data = {
        'well_md': torch.tensor(well_md),
        'log_md': torch.tensor(well_md),
        'log_gr': torch.tensor(well_gr),
        'start_md': start_md,
        'detected_start_md': start_md,
    }

    segmenter = SmartSegmenter(zone_data, settings)
    all_boundaries = []
    while not segmenter.is_finished:
        result = segmenter.next_slice()
        for bp in result.boundaries:
            if bp.md <= end_md:
                all_boundaries.append(bp)
        if result.is_final or (result.boundaries and result.boundaries[-1].md >= end_md):
            break

    # Create segment indices
    seg_indices = []
    prev_idx = start_idx
    for bp in all_boundaries[:max_segments-1]:
        bp_idx = int(np.searchsorted(well_md, bp.md))
        if bp_idx > prev_idx:
            seg_indices.append((prev_idx, bp_idx))
            prev_idx = bp_idx
    if prev_idx < end_idx:
        seg_indices.append((prev_idx, end_idx))

    if not seg_indices:
        raise RuntimeError(f"SmartSegmenter returned no segments for MD {start_md:.1f}-{end_md:.1f}")

    return seg_indices


def get_all_segment_boundaries_pelt(
    well_md: np.ndarray,
    well_gr: np.ndarray,
    start_md: float,
    end_md: float,
    settings: dict = None,
) -> List[float]:
    """
    Get ALL segment boundaries from start_md to end_md using SmartSegmenter/PELT.
    Returns list of boundary MDs (not including start_md, including end_md).
    """
    if settings is None:
        settings = {
            'min_segment_length': 30.0,
            'overlap_segments': 0,
            'pelt_penalty': 10.0,
            'min_distance': 20.0,
            'segments_count': 100,  # Large number to get all boundaries
        }

    zone_data = {
        'well_md': torch.tensor(well_md),
        'log_md': torch.tensor(well_md),
        'log_gr': torch.tensor(well_gr),
        'start_md': start_md,
        'detected_start_md': start_md,
    }

    segmenter = SmartSegmenter(zone_data, settings)
    boundaries = []

    while not segmenter.is_finished:
        result = segmenter.next_slice()
        for bp in result.boundaries:
            if start_md < bp.md <= end_md:
                boundaries.append(bp.md)
        if result.is_final:
            break

    # Add end_md if not already there
    if not boundaries or boundaries[-1] < end_md - 1:
        boundaries.append(end_md)

    return sorted(set(boundaries))


def optimize_full_well(
    well_name: str,
    well_data: dict,
    angle_range: float = 1.5,
    angle_step: float = 0.2,
    pearson_power: float = 1.0,
    pearson_clamp: float = 0.0,
    segments_per_block: int = 5,
    end_zone_angle_reduction: float = 0.5,  # Reduce angle_range near end
    start_from_landing: bool = True,  # Start from landing_end_87_200 instead of OTSU
    verbose: bool = False,
) -> List[OptimizedSegment]:
    """
    Optimize entire well from landing point to end using PELT-based segmentation.

    Strategy:
    1. Start from landing_end_87_200 (or OTSU if start_from_landing=False)
    2. Get ALL PELT boundaries from start to well end
    3. Take 5 segments at a time, optimize with GPU brute-force
    4. end_shift of block N = start_shift of block N+1
    5. Near end: reduce angle_range

    Args:
        well_name: Well identifier
        well_data: Dataset entry
        angle_range: Angle search range in degrees
        angle_step: Angle step in degrees
        mse_power: MSE penalty power (0, 0.5, or 1.0)
        segments_per_block: Segments to optimize together (5)
        end_zone_angle_reduction: Factor to reduce angle_range near end
        start_from_landing: If True, start from landing_end_87_200; else use OTSU
        verbose: Print progress

    Returns:
        List of OptimizedSegment from start to well end
    """
    # Extract data
    well_md = well_data['well_md'].numpy()
    well_tvd = well_data['well_tvd'].numpy()
    log_md = well_data['log_md'].numpy()
    log_gr = well_data['log_gr'].numpy()

    # Get typewell (stitched or original based on NORMALIZATION_MODE env)
    type_tvd, type_gr = stitch_typewell_from_dataset(well_data, mode=NORMALIZATION_MODE, apply_smoothing=True)

    # tvd_shift: for stitched mode (pseudo) it's 0, for original it's from dataset
    if NORMALIZATION_MODE == 'ORIGINAL':
        tvd_shift = float(well_data.get('tvd_typewell_shift', 0.0))
    else:
        tvd_shift = 0.0  # Stitched typewell is already in well TVD coordinates

    # Interpolate GR to well_md and apply normalization
    well_gr = np.interp(well_md, log_md, log_gr)
    norm_multiplier = float(well_data.get('norm_multiplier', 1.0))
    if norm_multiplier != 1.0:
        well_gr = well_gr * norm_multiplier

    # Determine start point
    if start_from_landing:
        # Start from landing_end_87_200 (honest baseline point)
        zone_start = float(well_data.get('landing_end_87_200', well_md[len(well_md) // 3]))
        zone_start = max(zone_start, log_md[0])  # Ensure within log range
    else:
        # Find OTSU zone - use it as starting point
        detector = OtsuPeakDetector()
        finder = RegionFinder(detector, search_fraction=0.33)
        result = finder.find_best_region(log_gr, log_md, region_length_m=200.0)
        otsu_center = result.best_md
        zone_start = max(otsu_center - 100.0, log_md[0])

    # Well end
    well_end_md = min(well_md[-1], log_md[-1])

    # Get ALL PELT boundaries from zone_start to well_end
    all_boundaries = get_all_segment_boundaries_pelt(well_md, well_gr, zone_start, well_end_md)

    if verbose:
        print(f"  PELT found {len(all_boundaries)} boundaries from MD {zone_start:.1f} to {well_end_md:.1f}")

    # Initial shift from baseline (TVT=const from 87°+200m) - NO CHEATING!
    baseline_md = float(well_data.get('landing_end_87_200', well_md[len(well_md) // 2]))
    baseline_idx = int(np.searchsorted(well_md, baseline_md))
    tvt_baseline = well_tvd[baseline_idx] - interpolate_shift_at_md(well_data, well_md[baseline_idx])
    zone_start_idx_for_shift = int(np.searchsorted(well_md, zone_start))
    current_shift = well_tvd[zone_start_idx_for_shift] - tvt_baseline

    # Trend angle: from zone_start to well_end (actual trajectory direction)
    zone_start_idx = int(np.searchsorted(well_md, zone_start))
    end_idx = int(np.searchsorted(well_md, well_end_md)) - 1

    delta_tvd = well_tvd[end_idx] - well_tvd[zone_start_idx]
    delta_md = well_md[end_idx] - well_md[zone_start_idx]
    trend_angle = np.degrees(np.arctan(delta_tvd / delta_md)) if delta_md > 10 else 0.0

    if verbose:
        print(f"  Trend angle (zone_start {zone_start:.0f}m → end {well_end_md:.0f}m): {trend_angle:+.2f}°")

    all_segments = []
    current_md = zone_start
    boundary_idx = 0
    block_num = 0

    while boundary_idx < len(all_boundaries):
        block_num += 1

        # Take next segments_per_block boundaries (or remaining)
        block_boundaries = all_boundaries[boundary_idx:boundary_idx + segments_per_block]

        if not block_boundaries:
            break

        block_end = block_boundaries[-1]

        # Check if near end - reduce angle range
        remaining = well_end_md - current_md
        total_length = well_end_md - zone_start
        if remaining < total_length * 0.2:  # Last 20% of well
            effective_angle_range = angle_range * end_zone_angle_reduction
        else:
            effective_angle_range = angle_range

        # Build segment indices from boundaries
        seg_indices = []
        prev_md = current_md
        for bnd_md in block_boundaries:
            s_idx = int(np.searchsorted(well_md, prev_md))
            e_idx = int(np.searchsorted(well_md, bnd_md))
            if e_idx > s_idx:
                seg_indices.append((s_idx, e_idx))
            prev_md = bnd_md

        if not seg_indices:
            boundary_idx += len(block_boundaries)
            current_md = block_end
            continue

        # Use trend angle (from landing to zone_start) instead of local block angle
        traj_angle = trend_angle

        # Optimize block
        pearson, end_shift, best_angles, _ = optimize_segment_block_gpu(
            seg_indices, current_shift, traj_angle,
            well_md, well_tvd, well_gr,
            type_tvd, type_gr, tvd_shift,
            angle_range=effective_angle_range,
            angle_step=angle_step,
            pearson_power=pearson_power,
            pearson_clamp=pearson_clamp,
        )

        # Build segments from result
        seg_md_lens = np.array([well_md[e] - well_md[s] for s, e in seg_indices])
        shift_deltas = np.tan(np.radians(best_angles)) * seg_md_lens

        seg_start_shift = current_shift
        for i, (s_idx, e_idx) in enumerate(seg_indices):
            seg_end_shift = seg_start_shift + shift_deltas[i]

            all_segments.append(OptimizedSegment(
                start_md=well_md[s_idx],
                end_md=well_md[e_idx-1] if e_idx > s_idx else well_md[s_idx],
                start_shift=seg_start_shift,
                end_shift=seg_end_shift,
                angle_deg=best_angles[i],
                pearson=pearson,
            ))

            seg_start_shift = seg_end_shift

        if verbose:
            print(f"  Block {block_num}: MD {current_md:.1f}-{block_end:.1f}m, "
                  f"{len(seg_indices)} seg, Pearson={pearson:.3f}, "
                  f"angle_range=±{effective_angle_range:.1f}°")

        # Move to next block
        boundary_idx += len(block_boundaries)
        current_md = block_end
        current_shift = end_shift

    return all_segments


def test_single_well(well_name: str = "Well1221~EGFDL"):
    """Test on single well."""
    print(f"Testing full well optimization on {well_name}")
    print("=" * 60)

    ds = torch.load('dataset/gpu_ag_dataset.pt', weights_only=False)

    if well_name not in ds:
        print(f"Well {well_name} not found!")
        return

    well_data = ds[well_name]

    # Get reference end shift
    ref_end_md = min(well_data['well_md'].numpy()[-1], well_data['log_md'].numpy()[-1])
    ref_end_shift = interpolate_shift_at_md(well_data, ref_end_md)

    print(f"Reference end shift: {ref_end_shift:.2f}m at MD={ref_end_md:.1f}m")
    print()

    # Test with angle_range=1.5
    print("Optimizing with angle_range=1.5°...")
    t0 = time.time()
    segments = optimize_full_well(
        well_name, well_data,
        angle_range=1.5,
        verbose=True,
    )
    t1 = time.time()

    if segments:
        pred_end_shift = segments[-1].end_shift
        error = pred_end_shift - ref_end_shift
        print(f"\nResult: {len(segments)} segments, end_shift={pred_end_shift:.2f}m")
        print(f"Error: {error:+.2f}m (ref={ref_end_shift:.2f}m)")
        print(f"Time: {t1-t0:.1f}s")

    print()

    # Test with angle_range=1.0
    print("Optimizing with angle_range=1.0°...")
    t0 = time.time()
    segments = optimize_full_well(
        well_name, well_data,
        angle_range=1.0,
        verbose=True,
    )
    t1 = time.time()

    if segments:
        pred_end_shift = segments[-1].end_shift
        error = pred_end_shift - ref_end_shift
        print(f"\nResult: {len(segments)} segments, end_shift={pred_end_shift:.2f}m")
        print(f"Error: {error:+.2f}m (ref={ref_end_shift:.2f}m)")
        print(f"Time: {t1-t0:.1f}s")


def test_all_wells(angle_range: float = 1.5, save_csv: bool = True):
    """Test on all 100 wells with optional CSV export."""
    import csv
    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Testing full well optimization on 100 wells (angle_range=±{angle_range}°)")
    print(f"Run ID: {run_id}")
    print("=" * 70)

    ds = torch.load('dataset/gpu_ag_dataset.pt', weights_only=False)

    # CSV setup
    csv_rows = []
    csv_header = [
        'run_id', 'well_name', 'row_type', 'seg_idx',
        'md_start', 'md_end', 'end_shift', 'end_error',
        'angle_deg', 'pearson', 'baseline_error', 'opt_error',
        'rmse', 'n_segments', 'time_sec'
    ]

    errors = []
    baseline_errors = []
    t_start = time.time()

    for i, (well_name, well_data) in enumerate(ds.items()):
        t_well_start = time.time()

        # Get reference end shift
        well_md = well_data['well_md'].numpy()
        log_md = well_data['log_md'].numpy()
        ref_end_md = min(well_md[-1], log_md[-1])
        ref_end_shift = interpolate_shift_at_md(well_data, ref_end_md)

        # Baseline (TVT=const from 87° + 200m point)
        well_tvd = well_data['well_tvd'].numpy()
        baseline_md = float(well_data.get('landing_end_87_200', well_md[len(well_md)//2]))
        baseline_idx = min(int(np.searchsorted(well_md, baseline_md)), len(well_md) - 1)
        tvt_at_baseline = well_tvd[baseline_idx] - interpolate_shift_at_md(well_data, well_md[baseline_idx])
        baseline_shift = well_tvd[np.searchsorted(well_md, ref_end_md) - 1] - tvt_at_baseline
        baseline_error = baseline_shift - ref_end_shift
        baseline_errors.append(baseline_error)

        # Optimize
        segments = optimize_full_well(
            well_name, well_data,
            angle_range=angle_range,
            verbose=False,
        )

        if not segments:
            raise RuntimeError(f"No segments returned for {well_name}")

        pred_end_shift = segments[-1].end_shift
        opt_error = pred_end_shift - ref_end_shift
        errors.append(opt_error)

        t_well = time.time() - t_well_start

        # Save segment rows
        if save_csv:
            cumulative_shift = segments[0].start_shift
            for seg_idx, seg in enumerate(segments):
                seg_error = seg.end_shift - interpolate_shift_at_md(well_data, seg.end_md)
                csv_rows.append({
                    'run_id': run_id,
                    'well_name': well_name,
                    'row_type': 'segment',
                    'seg_idx': seg_idx,
                    'md_start': f"{seg.start_md:.1f}",
                    'md_end': f"{seg.end_md:.1f}",
                    'end_shift': f"{seg.end_shift:.2f}",
                    'end_error': f"{seg_error:.2f}",
                    'angle_deg': f"{seg.angle_deg:.3f}",
                    'pearson': f"{seg.pearson:.3f}",
                    'baseline_error': '',
                    'opt_error': '',
                    'rmse': '',
                    'n_segments': '',
                    'time_sec': '',
                })

            # Well summary row
            csv_rows.append({
                'run_id': run_id,
                'well_name': well_name,
                'row_type': 'well',
                'seg_idx': '',
                'md_start': f"{segments[0].start_md:.1f}",
                'md_end': f"{segments[-1].end_md:.1f}",
                'end_shift': f"{pred_end_shift:.2f}",
                'end_error': f"{opt_error:.2f}",
                'angle_deg': f"{np.mean([s.angle_deg for s in segments]):.3f}",
                'pearson': f"{np.mean([s.pearson for s in segments]):.3f}",
                'baseline_error': f"{baseline_error:.2f}",
                'opt_error': f"{opt_error:.2f}",
                'rmse': '',
                'n_segments': len(segments),
                'time_sec': f"{t_well:.2f}",
            })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/100 wells processed...")

    t_elapsed = time.time() - t_start

    errors = np.array(errors)
    baseline_errors = np.array(baseline_errors)

    rmse_opt = np.sqrt(np.mean(errors**2))
    rmse_baseline = np.sqrt(np.mean(baseline_errors**2))
    improved = np.sum(np.abs(errors) < np.abs(baseline_errors))

    # Summary row
    if save_csv:
        csv_rows.append({
            'run_id': run_id,
            'well_name': 'ALL_WELLS',
            'row_type': 'summary',
            'seg_idx': '',
            'md_start': '',
            'md_end': '',
            'end_shift': '',
            'end_error': '',
            'angle_deg': f"{angle_range:.1f}",
            'pearson': '',
            'baseline_error': f"{rmse_baseline:.2f}",
            'opt_error': f"{rmse_opt:.2f}",
            'rmse': f"{rmse_opt:.2f}",
            'n_segments': improved,
            'time_sec': f"{t_elapsed:.1f}",
        })

        # Write CSV
        csv_path = Path(__file__).parent / 'results' / 'full_well_stats.csv'
        csv_path.parent.mkdir(exist_ok=True)

        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            if not file_exists:
                writer.writeheader()
            writer.writerows(csv_rows)

        print(f"\nStats saved to: {csv_path}")

    print()
    print(f"Results (angle_range=±{angle_range}°):")
    print(f"  Baseline RMSE: {rmse_baseline:.2f}m")
    print(f"  Optimized RMSE: {rmse_opt:.2f}m")
    print(f"  Improvement: {rmse_baseline - rmse_opt:.2f}m ({100*(rmse_baseline-rmse_opt)/rmse_baseline:.1f}%)")
    print(f"  Wells improved: {improved}/100")
    print(f"  Time: {t_elapsed:.1f}s ({t_elapsed/100:.2f}s/well)")

    return rmse_opt, improved


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        # Test with angle_range=1.5 (best results)
        print("\n" + "="*70)
        test_all_wells(angle_range=1.5)
    else:
        test_single_well()
