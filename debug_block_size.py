#!/usr/bin/env python
"""
Debug: Compare top candidates at different block sizes (3, 4, 5 segments).
See how metrics change as we add more segments.
"""

import os
import torch
import numpy as np
from itertools import product

os.environ['LANDING_MODE'] = 'dls'

from optimizers import prepare_block_data, compute_score_batch, compute_std_batch, BeamPrefix, GPU_DTYPE
from cpu_baseline.preprocessing import prepare_typelog
from full_well_optimizer import get_segment_boundaries

DATASET_PATH = '/mnt/e/Projects/Rogii/gpu_ag_2/dataset/gpu_ag_dataset.pt'
DEVICE = 'cuda'

def analyze_block_size(well_name: str, n_segments: int, angle_range: float = 2.5, angle_step: float = 0.2):
    """Analyze top candidates for first n_segments."""

    ds = torch.load(DATASET_PATH, weights_only=False)
    well_data = ds[well_name]

    well_md = well_data['well_md'].numpy()
    well_tvd = well_data['well_tvd'].numpy()
    log_md = well_data['log_md'].numpy()
    log_gr = well_data['log_gr'].numpy()
    # Interpolate log_gr to well_md
    well_gr = np.interp(well_md, log_md, log_gr)

    # Prepare typelog
    type_tvd, type_gr, _ = prepare_typelog(well_data)

    # Get landing point (dls mode)
    zone_start = float(well_data.get('landing_end_dls', well_md[len(well_md)//2]))
    zone_start = max(zone_start, well_md[0])
    print(f"Zone start (dls): {zone_start:.1f}")

    # Smart segmentation - returns list of (start_idx, end_idx) tuples
    end_md = well_md[-1]
    segment_indices = get_segment_boundaries(well_md, well_gr, zone_start, end_md, max_segments=50)

    print(f"\n{'='*60}")
    print(f"Block size: {n_segments} segments")
    print(f"Segment MDs: {[(well_md[s], well_md[e-1]) for s, e in segment_indices[:n_segments]]}")

    # Use only first n_segments
    seg_indices = segment_indices[:n_segments]

    # Prepare block data
    block_data = prepare_block_data(
        seg_indices, well_md, well_tvd, well_gr,
        type_tvd, type_gr, DEVICE
    )

    # Initial shift from landing
    start_shift = float(well_data.get('shift_at_landing_dls', 0.0))
    prefix = BeamPrefix.empty(start_shift, DEVICE)

    # Generate angle grid
    angles = np.arange(-angle_range, angle_range + 0.01, angle_step)
    n_angles = len(angles)
    n_combos = n_angles ** n_segments

    print(f"Angle grid: {n_angles} values, {n_combos} combinations")

    # Generate all combinations
    all_angles_list = list(product(angles, repeat=n_segments))
    all_angles_np = np.array(all_angles_list, dtype=np.float32)
    all_angles_gpu = torch.tensor(all_angles_np, device=DEVICE, dtype=GPU_DTYPE)

    # Compute scores in chunks (smaller for more segments to avoid OOM)
    chunk_size = 10000
    all_scores = []
    all_pearsons = []
    all_mses = []
    all_end_shifts = []

    for i in range(0, len(all_angles_gpu), chunk_size):
        chunk = all_angles_gpu[i:i+chunk_size]
        scores, pearsons, mses, end_shifts, _ = compute_score_batch(
            chunk, block_data, prefix,
            trajectory_angle=0.0, angle_range=angle_range, mse_weight=5.0,
            selfcorr_threshold=0.0, selfcorr_weight=0.0
        )
        all_scores.append(scores)
        all_pearsons.append(pearsons)
        all_mses.append(mses)
        all_end_shifts.append(end_shifts)

    all_scores = torch.cat(all_scores).cpu().numpy()
    all_pearsons = torch.cat(all_pearsons).cpu().numpy()
    all_mses = torch.cat(all_mses).cpu().numpy()
    all_end_shifts_full = torch.cat(all_end_shifts)  # (n_combos, n_points)
    all_end_shifts = all_end_shifts_full[:, -1].cpu().numpy()  # Final shift only

    # Compute STD for top candidates
    top_k = 20
    top_indices = np.argsort(all_scores)[-top_k:][::-1].copy()

    top_angles_gpu = all_angles_gpu[torch.tensor(top_indices, device=DEVICE)]
    top_std = compute_std_batch(top_angles_gpu, block_data, prefix).cpu().numpy()

    # Reference angles from BF result
    bf_ref = [1.31, 2.71, 3.31, 1.71, 1.81][:n_segments]
    beam_ref = [1.11, 2.91, 3.31, 1.31, 2.31][:n_segments]

    # Find BF and BEAM references (tolerance = angle_step/2 + margin)
    tol = angle_step / 2 + 0.05
    def find_ref(ref_angles):
        for i, angles in enumerate(all_angles_np):
            if all(abs(angles[j] - ref_angles[j]) < tol for j in range(len(ref_angles))):
                return i
        return None

    bf_idx = find_ref(bf_ref)
    beam_idx = find_ref(beam_ref)

    print(f"\nTop-10 by score:")
    print(f"{'Rank':>4} {'Angles':>35} {'Score':>8} {'Pearson':>8} {'MSE':>8} {'Shift':>8} {'STD':>8}")
    print("-" * 90)

    for rank, idx in enumerate(top_indices[:10]):
        idx = int(idx)  # Convert numpy scalar to int
        angles_str = ','.join(f'{a:.2f}' for a in all_angles_np[idx])
        std_val = float(top_std[rank]) if rank < len(top_std) else float('nan')
        score_v = float(all_scores[idx])
        pearson_v = float(all_pearsons[idx])
        mse_v = float(all_mses[idx])
        shift_v = float(all_end_shifts[idx])
        print(f"{rank+1:>4} [{angles_str:>32}] {score_v:>8.3f} {pearson_v:>8.3f} {mse_v:>8.3f} {shift_v:>8.2f} {std_val:>8.2f}")

    # Show BF and BEAM reference positions
    print("\nReference solutions:")
    if bf_idx is not None:
        bf_rank = int((all_scores > all_scores[bf_idx]).sum()) + 1
        angles_str = ','.join(f'{a:.2f}' for a in all_angles_np[bf_idx])
        print(f"  BF  [{angles_str}]: rank {bf_rank}/{len(all_scores)}, score={float(all_scores[bf_idx]):.3f}, shift={float(all_end_shifts[bf_idx]):.2f}")
    else:
        print(f"  BF  {bf_ref}: NOT FOUND")

    if beam_idx is not None:
        beam_rank = int((all_scores > all_scores[beam_idx]).sum()) + 1
        angles_str = ','.join(f'{a:.2f}' for a in all_angles_np[beam_idx])
        print(f"  BEAM [{angles_str}]: rank {beam_rank}/{len(all_scores)}, score={float(all_scores[beam_idx]):.3f}, shift={float(all_end_shifts[beam_idx]):.2f}")
    else:
        print(f"  BEAM {beam_ref}: NOT FOUND")


if __name__ == '__main__':
    well = 'Well498~EGFDL'

    # Adaptive step prevents OOM
    for n_seg in [3, 4, 5]:
        analyze_block_size(well, n_seg)
