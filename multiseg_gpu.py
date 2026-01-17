#!/usr/bin/env python3
"""
GPU-accelerated multi-segment optimization.
"""

import os
import torch
import numpy as np
from torch_funcs.converters import GPU_DTYPE

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/gpu_ag_dataset.pt')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def optimize_multiseg_gpu(
    segment_indices,
    baseline_shift,
    trajectory_angle,
    well_md, well_tvd, well_gr,
    type_tvd, type_gr, tvd_shift,
    angle_range=1.0,
    angle_steps=11,
    shift_range=20.0,
    shift_steps=21,
    device=DEVICE,
    chunk_size=500000,
):
    """
    GPU brute-force optimization of angles for connected segments.
    Processes in chunks to avoid OOM.

    Returns: (best_pearson, best_angles, best_end_shift, best_start_shift)
    """
    n_seg = len(segment_indices)

    # Generate angle combinations
    angles = np.linspace(
        trajectory_angle - angle_range,
        trajectory_angle + angle_range,
        angle_steps
    )

    # Generate start shift values
    start_shifts_arr = np.linspace(
        baseline_shift - shift_range,
        baseline_shift + shift_range,
        shift_steps
    )

    # Meshgrid: angles for each segment + start_shift
    grids = np.meshgrid(*[angles]*n_seg, start_shifts_arr, indexing='ij')
    # grids[-1] is start_shift, grids[:-1] are angles
    all_angles_np = np.stack([g.ravel() for g in grids[:-1]], axis=1).astype(np.float32)
    all_start_shifts_np = grids[-1].ravel().astype(np.float32)
    n_combos = all_angles_np.shape[0]

    # Segment MD lengths
    seg_md_lens = np.array([well_md[e] - well_md[s] for s, e in segment_indices], dtype=np.float32)

    # Zone indices and data
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

    best_pearson = -1.0
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

        chunk_best_idx = torch.argmax(pearsons).item()
        chunk_best_pearson = pearsons[chunk_best_idx].item()

        if chunk_best_pearson > best_pearson:
            best_pearson = chunk_best_pearson
            best_idx_global = chunk_start + chunk_best_idx

    # Get best result
    best_angles_list = all_angles_np[best_idx_global].tolist()
    best_start_shift = float(all_start_shifts_np[best_idx_global])

    # Recompute best end_shift
    best_shift_deltas = np.tan(np.radians(best_angles_list)) * seg_md_lens
    best_end_shift = best_start_shift + np.sum(best_shift_deltas)

    return best_pearson, best_angles_list, best_end_shift, best_start_shift


def find_otsu_zone(log_gr, log_md, search_fraction=0.33, window_m=200.0):
    """Find best OTSU zone."""
    from peak_detectors import OtsuPeakDetector, RegionFinder
    detector = OtsuPeakDetector()
    finder = RegionFinder(detector, search_fraction=search_fraction)
    result = finder.find_best_region(log_gr, log_md, region_length_m=window_m)
    center_md = result.best_md
    zone_start = max(center_md - window_m / 2, log_md[0])
    zone_end = min(center_md + window_m / 2, log_md[-1])
    return zone_start, zone_end


def process_well_gpu(well_name, well, angle_steps=11, shift_range=20.0, shift_steps=21):
    """Process single well with GPU optimization."""
    from numpy_funcs.interpretation import interpolate_shift_at_md
    from smart_segmenter import SmartSegmenter

    well_md = well['well_md'].numpy()
    well_tvd = well['well_tvd'].numpy()
    log_md = well['log_md'].numpy()
    log_gr = well['log_gr'].numpy()
    type_tvd = well['type_tvd'].numpy()
    type_gr = well['type_gr'].numpy()
    tvd_shift = float(well.get('tvd_typewell_shift', 0.0))
    well_gr = np.interp(well_md, log_md, log_gr)

    try:
        zone_start, zone_end = find_otsu_zone(log_gr, log_md)
    except:
        return None

    baseline_shift = interpolate_shift_at_md(well, zone_start)
    ref_shift_end = interpolate_shift_at_md(well, zone_end)

    start_idx = int(np.searchsorted(well_md, zone_start))
    end_idx = int(np.searchsorted(well_md, zone_end))
    if end_idx - start_idx < 50:
        return None

    # Trajectory angle
    delta_tvd = well_tvd[end_idx] - well_tvd[start_idx]
    delta_md = well_md[end_idx] - well_md[start_idx]
    traj_angle = np.degrees(np.arctan(delta_tvd / delta_md)) if delta_md > 0.1 else 0.0

    # Use SmartSegmenter to find boundaries by peaks
    settings = {
        'min_segment_length': 30.0,
        'overlap_segments': 0,
        'pelt_penalty': 10.0,
        'min_distance': 20.0,
        'segments_count': 5,
    }
    zone_data = {
        'well_md': torch.tensor(well_md),
        'log_md': torch.tensor(log_md),
        'log_gr': torch.tensor(log_gr),
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

        # Create segments from boundaries (max 5)
        seg5_indices = []
        prev_idx = start_idx
        for bp in all_boundaries[:4]:
            bp_idx = int(np.searchsorted(well_md, bp.md))
            if bp_idx > prev_idx:
                seg5_indices.append((prev_idx, bp_idx))
                prev_idx = bp_idx
        if prev_idx < end_idx:
            seg5_indices.append((prev_idx, end_idx))
        if not seg5_indices:
            seg5_indices = [(start_idx, end_idx)]
    except:
        # Fallback to equal segments
        n_points = end_idx - start_idx
        seg_len = n_points // 5
        seg5_indices = [(start_idx + i*seg_len, start_idx + (i+1)*seg_len) for i in range(5)]
        seg5_indices[-1] = (seg5_indices[-1][0], end_idx)

    # 1 segment
    seg1_indices = [(start_idx, end_idx)]

    # Optimize 5 segments
    p5, angles5, end5, start5 = optimize_multiseg_gpu(
        seg5_indices, baseline_shift, traj_angle,
        well_md, well_tvd, well_gr, type_tvd, type_gr, tvd_shift,
        angle_steps=angle_steps, shift_range=shift_range, shift_steps=shift_steps
    )
    err5 = end5 - ref_shift_end

    # Optimize 1 segment
    p1, angles1, end1, start1 = optimize_multiseg_gpu(
        seg1_indices, baseline_shift, traj_angle,
        well_md, well_tvd, well_gr, type_tvd, type_gr, tvd_shift,
        angle_steps=21, shift_range=shift_range, shift_steps=shift_steps
    )
    err1 = end1 - ref_shift_end

    return {
        'well': well_name,
        'ref_end': ref_shift_end,
        'err_1seg': err1,
        'err_5seg': err5,
        'p_1seg': p1,
        'p_5seg': p5,
    }


if __name__ == '__main__':
    import time

    ds = torch.load(DATASET_PATH, weights_only=False)
    wells = list(ds.keys())

    print("=" * 60)
    print("GPU Multi-Segment Optimization (±20m shift, ±1° angle)")
    print("=" * 60)
    print(f"Wells: {len(wells)}")
    print()

    results = []
    t_start = time.time()

    for i, name in enumerate(wells):
        r = process_well_gpu(name, ds[name])
        if r:
            results.append(r)
            improved = abs(r['err_5seg']) < abs(r['err_1seg'])
            marker = '✓' if improved else ' '
            print(f"[{i+1:3d}/{len(wells)}] {name}: 1seg={r['err_1seg']:+.2f}m, 5seg={r['err_5seg']:+.2f}m {marker}")

    t_total = time.time() - t_start

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    err1 = np.array([r['err_1seg'] for r in results])
    err5 = np.array([r['err_5seg'] for r in results])

    rmse1 = np.sqrt(np.mean(err1**2))
    rmse5 = np.sqrt(np.mean(err5**2))
    mae1 = np.mean(np.abs(err1))
    mae5 = np.mean(np.abs(err5))
    improved = np.sum(np.abs(err5) < np.abs(err1))

    print(f"\nProcessed: {len(results)} wells in {t_total:.1f} sec ({t_total/len(results)*1000:.0f} ms/well)")
    print()
    print(f"{'Metric':<15} {'1 Segment':>12} {'5 Segments':>12} {'Δ':>10}")
    print("-" * 50)
    print(f"{'RMSE':<15} {rmse1:>12.2f}m {rmse5:>12.2f}m {rmse5-rmse1:>+10.2f}m")
    print(f"{'MAE':<15} {mae1:>12.2f}m {mae5:>12.2f}m {mae5-mae1:>+10.2f}m")
    print(f"{'Improved':<15} {'-':>12} {improved:>12}/{len(results)}")
