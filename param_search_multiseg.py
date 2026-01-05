#!/usr/bin/env python3
"""
Parameter search for multi-segment GPU optimization.

Caches SmartSegmenter boundaries and OTSU zones for speed.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path
from itertools import product
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

from torch_funcs.converters import GPU_DTYPE
from smart_segmenter import SmartSegmenter
from peak_detectors import OtsuPeakDetector, RegionFinder
from numpy_funcs.interpretation import interpolate_shift_at_md

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class CachedWell:
    """Cached data for a single well."""
    well_md: np.ndarray
    well_tvd: np.ndarray
    well_gr: np.ndarray
    log_md: np.ndarray  # Original log MD for OTSU
    log_gr: np.ndarray  # Original log GR for OTSU
    type_tvd: np.ndarray
    type_gr: np.ndarray
    tvd_shift: float
    # Per window_size: {window_size: (zone_start, zone_end, seg_indices, baseline_shift, ref_end_shift, traj_angle)}
    otsu_cache: Dict[int, Tuple] = None

    def __post_init__(self):
        self.otsu_cache = {}


def cache_all_wells(ds: dict) -> Dict[str, CachedWell]:
    """Load and cache all wells data."""
    cached = {}
    for name, well in ds.items():
        well_md = well['well_md'].numpy()
        well_tvd = well['well_tvd'].numpy()
        log_md = well['log_md'].numpy()
        log_gr = well['log_gr'].numpy()
        type_tvd = well['type_tvd'].numpy()
        type_gr = well['type_gr'].numpy()
        tvd_shift = float(well.get('tvd_typewell_shift', 0.0))
        well_gr = np.interp(well_md, log_md, log_gr)

        cached[name] = CachedWell(
            well_md=well_md,
            well_tvd=well_tvd,
            well_gr=well_gr,
            log_md=log_md,
            log_gr=log_gr,
            type_tvd=type_tvd,
            type_gr=type_gr,
            tvd_shift=tvd_shift,
        )
    return cached


def get_otsu_zone_and_segments(
    well_name: str,
    well_data: dict,
    cached: CachedWell,
    window_m: float,
    settings: dict
) -> Optional[Tuple]:
    """
    Get OTSU zone and SmartSegmenter boundaries.
    Uses cache if available.

    Returns: (zone_start, zone_end, seg_indices, baseline_shift, ref_end_shift, traj_angle)
    """
    window_key = int(window_m)

    if window_key in cached.otsu_cache:
        return cached.otsu_cache[window_key]

    well_md = cached.well_md
    well_tvd = cached.well_tvd

    # Find OTSU zone using ORIGINAL log_gr/log_md (not interpolated!)
    try:
        detector = OtsuPeakDetector()
        finder = RegionFinder(detector, search_fraction=0.33)
        result = finder.find_best_region(cached.log_gr, cached.log_md, region_length_m=window_m)
        center_md = result.best_md
        zone_start = max(center_md - window_m / 2, cached.log_md[0])
        zone_end = min(center_md + window_m / 2, cached.log_md[-1])
    except:
        return None

    # Get indices
    start_idx = int(np.searchsorted(well_md, zone_start))
    end_idx = int(np.searchsorted(well_md, zone_end))
    if end_idx - start_idx < 50:
        return None

    # Baseline shift (TVT=const from max_tvd) - NO CHEATING!
    max_tvd_idx = int(np.argmax(well_tvd))
    tvt_baseline = well_tvd[max_tvd_idx] - interpolate_shift_at_md(well_data, well_md[max_tvd_idx])
    baseline_shift = well_tvd[start_idx] - tvt_baseline
    # Ref end shift for comparison
    ref_end_shift = interpolate_shift_at_md(well_data, zone_end)

    # Trajectory angle
    delta_tvd = well_tvd[end_idx] - well_tvd[start_idx]
    delta_md = well_md[end_idx] - well_md[start_idx]
    traj_angle = np.degrees(np.arctan(delta_tvd / delta_md)) if delta_md > 0.1 else 0.0

    # SmartSegmenter for boundaries
    zone_data = {
        'well_md': torch.tensor(well_md),
        'log_md': torch.tensor(well_md),
        'log_gr': torch.tensor(cached.well_gr),
        'start_md': zone_start,
        'detected_start_md': zone_start,
    }

    try:
        segmenter = SmartSegmenter(zone_data, settings)
        all_boundaries = []
        while not segmenter.is_finished:
            result = segmenter.next_slice()
            for bp in result.boundaries:
                if bp.md <= zone_end:
                    all_boundaries.append(bp)
            if result.is_final or (result.boundaries and result.boundaries[-1].md >= zone_end):
                break

        # Create segment indices (max 5)
        seg_indices = []
        prev_idx = start_idx
        for bp in all_boundaries[:4]:
            bp_idx = int(np.searchsorted(well_md, bp.md))
            if bp_idx > prev_idx:
                seg_indices.append((prev_idx, bp_idx))
                prev_idx = bp_idx
        if prev_idx < end_idx:
            seg_indices.append((prev_idx, end_idx))
        if not seg_indices:
            seg_indices = [(start_idx, end_idx)]
    except:
        # Fallback to equal segments
        n_points = end_idx - start_idx
        seg_len = n_points // 5
        if seg_len < 10:
            seg_indices = [(start_idx, end_idx)]
        else:
            seg_indices = [(start_idx + i*seg_len, start_idx + (i+1)*seg_len) for i in range(5)]
            seg_indices[-1] = (seg_indices[-1][0], end_idx)

    result = (zone_start, zone_end, seg_indices, baseline_shift, ref_end_shift, traj_angle)
    cached.otsu_cache[window_key] = result
    return result


def optimize_multiseg_gpu_fast(
    segment_indices: List[Tuple[int, int]],
    baseline_shift: float,
    trajectory_angle: float,
    well_md: np.ndarray,
    well_tvd: np.ndarray,
    well_gr: np.ndarray,
    type_tvd: np.ndarray,
    type_gr: np.ndarray,
    tvd_shift: float,
    angle_range: float = 1.0,
    angle_step: float = 0.2,
    shift_range: float = 0.0,
    shift_steps: int = 1,
    mse_power: float = 0.0,
    device: str = DEVICE,
    chunk_size: int = 250000,  # Reduced for W=400/500
) -> Tuple[float, float, float]:
    """
    GPU optimization with configurable parameters.

    Returns: (best_pearson, best_end_shift, best_start_shift)
    """
    n_seg = len(segment_indices)

    # Generate angle candidates
    angle_steps = int(2 * angle_range / angle_step) + 1
    angles = np.linspace(
        trajectory_angle - angle_range,
        trajectory_angle + angle_range,
        angle_steps
    )

    # Generate start shift values
    if shift_range > 0 and shift_steps > 1:
        start_shifts_arr = np.linspace(
            baseline_shift - shift_range,
            baseline_shift + shift_range,
            shift_steps
        )
    else:
        start_shifts_arr = np.array([baseline_shift])

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

        # MSE penalty if enabled
        if mse_power > 0:
            mse = ((zone_gr - synthetic)**2).mean(dim=1)
            mse_norm = mse / (zone_gr.var() + 1e-10)
            if mse_power == 0.5:
                penalty = torch.sqrt(mse_norm)
            else:
                penalty = mse_norm
            scores = pearsons - 0.1 * penalty
        else:
            scores = pearsons

        chunk_best_idx = torch.argmax(scores).item()
        chunk_best_score = scores[chunk_best_idx].item()

        if chunk_best_score > best_score:
            best_score = chunk_best_score
            best_idx_global = chunk_start + chunk_best_idx

    # Get best result
    best_start_shift = float(all_start_shifts_np[best_idx_global])
    best_angles = all_angles_np[best_idx_global]
    best_shift_deltas = np.tan(np.radians(best_angles)) * seg_md_lens
    best_end_shift = best_start_shift + np.sum(best_shift_deltas)

    return best_score, best_end_shift, best_start_shift


def run_experiment(
    ds: dict,
    cached_wells: Dict[str, CachedWell],
    window_m: float,
    angle_range: float,
    angle_step: float,
    shift_range: float,
    shift_steps: int,
    mse_power: float,
    segmenter_settings: dict,
) -> Tuple[float, float, int, float]:
    """
    Run single experiment configuration.

    Returns: (rmse_1seg, rmse_5seg, improved_count, time_per_well_ms)
    """
    errors_1seg = []
    errors_5seg = []

    t_start = time.time()

    for name, well in ds.items():
        c = cached_wells[name]

        result = get_otsu_zone_and_segments(name, well, c, window_m, segmenter_settings)
        if result is None:
            continue

        zone_start, zone_end, seg_indices, baseline_shift, ref_end_shift, traj_angle = result

        # 1 segment optimization
        seg1 = [(seg_indices[0][0], seg_indices[-1][1])]
        _, end1, _ = optimize_multiseg_gpu_fast(
            seg1, baseline_shift, traj_angle,
            c.well_md, c.well_tvd, c.well_gr,
            c.type_tvd, c.type_gr, c.tvd_shift,
            angle_range=angle_range, angle_step=angle_step,
            shift_range=shift_range, shift_steps=shift_steps,
            mse_power=mse_power,
        )
        errors_1seg.append(end1 - ref_end_shift)

        # Multi-segment optimization
        _, end5, _ = optimize_multiseg_gpu_fast(
            seg_indices, baseline_shift, traj_angle,
            c.well_md, c.well_tvd, c.well_gr,
            c.type_tvd, c.type_gr, c.tvd_shift,
            angle_range=angle_range, angle_step=angle_step,
            shift_range=shift_range, shift_steps=shift_steps,
            mse_power=mse_power,
        )
        errors_5seg.append(end5 - ref_end_shift)

    t_elapsed = time.time() - t_start

    if not errors_1seg:
        return 999.0, 999.0, 0, 0.0

    err1 = np.array(errors_1seg)
    err5 = np.array(errors_5seg)

    rmse1 = np.sqrt(np.mean(err1**2))
    rmse5 = np.sqrt(np.mean(err5**2))
    improved = int(np.sum(np.abs(err5) < np.abs(err1)))
    time_per_well = t_elapsed / len(errors_1seg) * 1000

    return rmse1, rmse5, improved, time_per_well


def main():
    print("=" * 70)
    print("Multi-Segment Parameter Search")
    print("=" * 70)

    ds = torch.load('dataset/gpu_ag_dataset.pt', weights_only=False)
    print(f"Loading {len(ds)} wells...")

    cached_wells = cache_all_wells(ds)
    print(f"Cached well data")

    segmenter_settings = {
        'min_segment_length': 30.0,
        'overlap_segments': 0,
        'pelt_penalty': 10.0,
        'min_distance': 20.0,
        'segments_count': 5,
    }

    # Parameters to search
    # NOTE: angle_step=0.1 removed - too slow (17^5 = 1.4M combos per well)
    windows = [400, 500]  # OTSU window (200, 300 already done)
    angle_steps_list = [0.2]  # 0.1 too slow, 0.2 gives 9^5 = 59k combos
    angle_ranges = [0.8, 1.0, 1.5]
    mse_powers = [0, 0.5, 1.0]
    # Shift configs: (range, step) - only small ranges now
    shift_configs = [(0, 1), (5, 1), (6, 2)]  # S=0, S=5±1, S=6±2

    results = []

    # Calculate total combinations
    total_combos = len(windows) * len(angle_steps_list) * len(angle_ranges) * len(mse_powers) * len(shift_configs)
    print(f"\nTotal experiments: {total_combos}")
    print()

    exp_idx = 0
    for window_m in windows:
        for angle_step in angle_steps_list:
            for angle_range in angle_ranges:
                for mse_power in mse_powers:
                    for shift_range, shift_step in shift_configs:
                        exp_idx += 1
                        # Calculate actual shift_steps from range and step
                        if shift_range > 0:
                            actual_shift_steps = int(2 * shift_range / shift_step) + 1
                        else:
                            actual_shift_steps = 1

                        rmse1, rmse5, improved, time_ms = run_experiment(
                            ds, cached_wells, window_m,
                            angle_range, angle_step,
                            shift_range, actual_shift_steps,
                            mse_power, segmenter_settings
                        )

                        results.append({
                            'window': window_m,
                            'angle_step': angle_step,
                            'angle_range': angle_range,
                            'shift_range': shift_range,
                            'shift_step': shift_step if shift_range > 0 else 0,
                            'mse_power': mse_power,
                            'rmse_1seg': rmse1,
                            'rmse_5seg': rmse5,
                            'improved': improved,
                            'time_ms': time_ms,
                        })

                        delta = rmse5 - rmse1
                        marker = '✓' if delta < 0 else ' '
                        shift_str = f"S={shift_range}±{shift_step}" if shift_range > 0 else "S=0"
                        print(f"[{exp_idx:3d}/{total_combos}] W={window_m:3d} A={angle_range:.1f}±{angle_step:.1f} {shift_str} MSE^{mse_power:.1f}: "
                              f"1seg={rmse1:.2f}m, 5seg={rmse5:.2f}m (Δ={delta:+.2f}m) {improved}/100 {marker} [{time_ms:.0f}ms]")

    # Summary
    print("\n" + "=" * 70)
    print("TOP 10 BY RMSE_5SEG")
    print("=" * 70)

    sorted_results = sorted(results, key=lambda x: x['rmse_5seg'])
    for i, r in enumerate(sorted_results[:10]):
        print(f"{i+1:2d}. W={r['window']:3d} A={r['angle_range']:.1f}±{r['angle_step']:.1f} MSE^{r['mse_power']:.1f}: "
              f"RMSE={r['rmse_5seg']:.2f}m, improved={r['improved']}/100")

    # Save results as CSV
    import csv
    csv_path = 'results/param_search_multiseg.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'window', 'angle_step', 'angle_range', 'shift_range', 'shift_step',
            'mse_power', 'rmse_1seg', 'rmse_5seg', 'improved', 'time_ms'
        ])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {csv_path}")


if __name__ == '__main__':
    main()
