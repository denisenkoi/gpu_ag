#!/usr/bin/env python3
"""
Test to find exact difference between local function and BruteForceOptimizer.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, '/mnt/e/Projects/Rogii/gpu_ag')

from full_well_optimizer import optimize_segment_block_gpu
from optimizers import get_optimizer

DEVICE = 'cuda'
torch.set_printoptions(precision=6)

# Data directory
DATA_DIR = '/mnt/e/Projects/Rogii/gpu_ag/data'


def load_dataset():
    """Load dataset from .pt file."""
    ds = torch.load(f'{DATA_DIR}/../dataset/gpu_ag_dataset.pt', weights_only=False)
    return ds


def test_real_well():
    """Test both implementations on REAL well data."""
    well_name = 'Well1627~EGFDL'
    print(f"Loading {well_name}...")
    ds = load_dataset()

    if well_name not in ds:
        print(f"Well {well_name} not found!")
        return

    well_data = ds[well_name]

    # Get data directly (pseudo=stitched mode)
    well_md = well_data['well_md'].numpy() if isinstance(well_data['well_md'], torch.Tensor) else well_data['well_md']
    well_tvd = well_data['well_tvd'].numpy() if isinstance(well_data['well_tvd'], torch.Tensor) else well_data['well_tvd']

    # Use pseudo (stitched) for both
    type_tvd = well_data['pseudo_tvd'].numpy() if isinstance(well_data['pseudo_tvd'], torch.Tensor) else well_data['pseudo_tvd']
    type_gr = well_data['pseudo_gr'].numpy() if isinstance(well_data['pseudo_gr'], torch.Tensor) else well_data['pseudo_gr']

    # Interpolate log_gr to well_md points
    log_md = well_data['log_md'].numpy() if isinstance(well_data['log_md'], torch.Tensor) else well_data['log_md']
    log_gr = well_data['log_gr'].numpy() if isinstance(well_data['log_gr'], torch.Tensor) else well_data['log_gr']
    well_gr = np.interp(well_md, log_md, log_gr).astype(np.float32)

    # Get trajectory angle (use -2.0 as default)
    traj_angle = -2.0  # default

    # Get segments from ref_segment_mds
    ref_mds = well_data['ref_segment_mds']
    if isinstance(ref_mds, torch.Tensor):
        ref_mds = ref_mds.numpy()
    segments = []
    for i in range(len(ref_mds) - 1):
        start_md, end_md = ref_mds[i], ref_mds[i+1]
        start_idx = np.searchsorted(well_md, start_md)
        end_idx = np.searchsorted(well_md, end_md)
        if end_idx > start_idx:
            segments.append({'start_idx': start_idx, 'end_idx': end_idx})
    print(f"Well has {len(segments)} segments")

    # Test first 8 segments (block 0)
    block_size = 8
    seg_indices = [(s['start_idx'], s['end_idx']) for s in segments[:block_size]]
    print(f"Testing block with {len(seg_indices)} segments")

    # Print segment lengths
    for i, (s, e) in enumerate(seg_indices):
        seg_len = well_md[e] - well_md[s]
        print(f"  Seg {i}: MD {well_md[s]:.1f}-{well_md[e]:.1f}m, len={seg_len:.1f}m")

    start_shift = 0.0
    angle_range = 2.5
    angle_step = 0.2
    mse_weight = 5.0

    print(f"\nParams: range={angle_range}, step={angle_step}, mse={mse_weight}")
    print(f"traj_angle={traj_angle}, start_shift={start_shift}")

    # ===== LOCAL FUNCTION =====
    print("\n=== LOCAL FUNCTION (optimize_segment_block_gpu) ===")
    pearson1, end_shift1, angles1, _ = optimize_segment_block_gpu(
        seg_indices, start_shift, traj_angle,
        well_md, well_tvd, well_gr, type_tvd, type_gr,
        angle_range=angle_range, angle_step=angle_step, mse_weight=mse_weight,
        device=DEVICE, chunk_size=100000
    )
    print(f"Result: pearson={pearson1:.6f}, end_shift={end_shift1:.6f}")
    print(f"Angles: {np.round(angles1, 4)}")

    # ===== BRUTEFORCE OPTIMIZER =====
    print("\n=== BRUTEFORCE OPTIMIZER ===")
    opt = get_optimizer('BRUTEFORCE', device=DEVICE,
                        angle_range=angle_range, angle_step=angle_step,
                        mse_weight=mse_weight, chunk_size=100000)
    result = opt.optimize(seg_indices, start_shift, traj_angle,
                         well_md, well_tvd, well_gr, type_tvd, type_gr,
                         return_result=True)
    print(f"Result: pearson={result.pearson:.6f}, end_shift={result.end_shift:.6f}")
    print(f"Angles: {np.round(result.angles, 4)}")
    print(f"Score: {result.score:.6f}, MSE: {result.mse:.6f}")

    # ===== COMPARE =====
    print("\n=== COMPARISON ===")
    print(f"Pearson diff: {abs(pearson1 - result.pearson):.8f}")
    print(f"End shift diff: {abs(end_shift1 - result.end_shift):.8f}")
    angles_diff = np.abs(angles1 - result.angles)
    print(f"Angles diff: {angles_diff}")
    print(f"Max angle diff: {angles_diff.max():.8f}")

    if angles_diff.max() > 1e-5:
        print("\n!!! ANGLES DIFFER !!!")
        for i, (a1, a2) in enumerate(zip(angles1, result.angles)):
            if abs(a1 - a2) > 1e-5:
                print(f"  Segment {i}: local={a1:.4f}, bf={a2:.4f}, diff={a1-a2:.6f}")
    else:
        print("\n✓ Results IDENTICAL")


if __name__ == '__main__':
    test_real_well()
