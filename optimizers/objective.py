"""
Common objective function for all optimization algorithms.

This module provides unified score/loss computation to avoid code duplication.
All algorithms use the same objective:
    score = pearson - mse_weight * mse_norm  (maximize)
    loss = -score + penalty                   (minimize)
"""

import torch
import numpy as np
from typing import List, Tuple, NamedTuple, Optional
from dataclasses import dataclass


# GPU dtype for consistency
GPU_DTYPE = torch.float32


@dataclass
class BlockData:
    """Precomputed data for a block of segments (held in GPU memory)."""
    # Zone data (whole block)
    zone_tvd: torch.Tensor        # (n_points,)
    zone_gr: torch.Tensor         # (n_points,)
    zone_gr_centered: torch.Tensor
    zone_gr_ss: torch.Tensor      # scalar: sum of squares
    zone_gr_var: torch.Tensor     # scalar: variance

    # TypeLog data
    type_tvd: torch.Tensor        # (n_type,)
    type_gr: torch.Tensor         # (n_type,)

    # Segment metadata
    seg_md_lens: torch.Tensor     # (n_seg,) MD length of each segment
    seg_data: List[Tuple]         # [(local_start, local_end, seg_n, seg_tvd, ratio), ...]

    # Config
    n_points: int
    n_seg: int
    device: str


def prepare_block_data(
    segment_indices: List[Tuple[int, int]],
    well_md: np.ndarray,
    well_tvd: np.ndarray,
    well_gr: np.ndarray,
    type_tvd: np.ndarray,
    type_gr: np.ndarray,
    device: str = 'cuda',
) -> BlockData:
    """
    Prepare and transfer block data to GPU once.

    This data stays in GPU memory and is reused for all evaluations.
    """
    n_seg = len(segment_indices)
    start_idx = segment_indices[0][0]
    end_idx = segment_indices[-1][1]
    n_points = end_idx - start_idx

    # Zone data
    zone_tvd = torch.tensor(well_tvd[start_idx:end_idx], device=device, dtype=GPU_DTYPE)
    zone_gr = torch.tensor(well_gr[start_idx:end_idx], device=device, dtype=GPU_DTYPE)
    zone_gr_centered = zone_gr - zone_gr.mean()
    zone_gr_ss = (zone_gr_centered**2).sum()
    zone_gr_var = zone_gr.var()

    # TypeLog data
    type_tvd_t = torch.tensor(type_tvd, device=device, dtype=GPU_DTYPE)
    type_gr_t = torch.tensor(type_gr, device=device, dtype=GPU_DTYPE)

    # Segment MD lengths
    seg_md_lens = np.array([well_md[e] - well_md[s] for s, e in segment_indices], dtype=np.float32)
    seg_md_lens_t = torch.tensor(seg_md_lens, device=device, dtype=GPU_DTYPE)

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
            ratio = torch.tensor(
                (well_md[s_idx:e_idx] - md_start) / (md_end - md_start),
                device=device, dtype=GPU_DTYPE
            )
        else:
            ratio = torch.zeros(seg_n, device=device, dtype=GPU_DTYPE)

        seg_data.append((local_start, local_end, seg_n, seg_tvd, ratio))

    return BlockData(
        zone_tvd=zone_tvd,
        zone_gr=zone_gr,
        zone_gr_centered=zone_gr_centered,
        zone_gr_ss=zone_gr_ss,
        zone_gr_var=zone_gr_var,
        type_tvd=type_tvd_t,
        type_gr=type_gr_t,
        seg_md_lens=seg_md_lens_t,
        seg_data=seg_data,
        n_points=n_points,
        n_seg=n_seg,
        device=device,
    )


def compute_loss_batch(
    angles_batch: torch.Tensor,
    block_data: BlockData,
    start_shift: float,
    trajectory_angle: float,
    angle_range: float,
    mse_weight: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute loss for a batch of angle combinations.

    This is the UNIFIED objective function for ALL algorithms.

    Args:
        angles_batch: (batch, n_seg) angles in degrees (absolute, not relative)
        block_data: Precomputed block data in GPU memory
        start_shift: Initial TVD shift
        trajectory_angle: Reference angle (for penalty calculation)
        angle_range: Max allowed deviation from trajectory_angle
        mse_weight: Weight for MSE term

    Returns:
        Tuple of:
        - loss: (batch,) loss values (lower is better) - for minimization algorithms
        - pearson: (batch,) Pearson correlations
        - mse_norm: (batch,) normalized MSE values
    """
    device = block_data.device
    batch_size = angles_batch.shape[0]

    # Angle penalty: exponential for violations
    angle_deviation = torch.abs(angles_batch - trajectory_angle)
    angle_excess = angle_deviation - angle_range
    max_excess = torch.clamp(torch.max(angle_excess, dim=1).values, min=0.0)
    angle_penalty = torch.where(
        max_excess > 0,
        1e6 * torch.pow(torch.tensor(100.0, device=device, dtype=GPU_DTYPE), max_excess),
        torch.zeros(batch_size, device=device, dtype=GPU_DTYPE)
    )

    # Compute shifts from angles
    shift_deltas = torch.tan(torch.deg2rad(angles_batch)) * block_data.seg_md_lens
    cumsum = torch.cumsum(shift_deltas, dim=1)
    start_shift_t = torch.tensor(start_shift, device=device, dtype=GPU_DTYPE)
    end_shifts = start_shift_t + cumsum
    start_shifts = torch.cat([
        start_shift_t.expand(batch_size, 1),
        end_shifts[:, :-1]
    ], dim=1)

    # Build synthetic GR
    synthetic = torch.zeros((batch_size, block_data.n_points), device=device, dtype=GPU_DTYPE)

    # Track out-of-range conditions
    MAX_TVT_EXTRAPOLATION = 0.1524  # 0.5 ft
    out_of_range_conditions = []

    for seg_i, (local_start, local_end, seg_n, seg_tvd, ratio) in enumerate(block_data.seg_data):
        seg_start = start_shifts[:, seg_i:seg_i+1]
        seg_end = end_shifts[:, seg_i:seg_i+1]
        seg_shifts = seg_start + ratio.unsqueeze(0) * (seg_end - seg_start)

        tvt = seg_tvd.unsqueeze(0) - seg_shifts

        # Check TypeLog range violation
        below_min = block_data.type_tvd[0] - tvt
        above_max = tvt - block_data.type_tvd[-1]
        max_below = below_min.max(dim=1).values
        max_above = above_max.max(dim=1).values
        out_of_range_conditions.append(
            (max_below > MAX_TVT_EXTRAPOLATION) | (max_above > MAX_TVT_EXTRAPOLATION)
        )

        tvt_clamped = torch.clamp(tvt, block_data.type_tvd[0], block_data.type_tvd[-1])

        # Interpolate TypeLog GR
        indices = torch.searchsorted(block_data.type_tvd, tvt_clamped.reshape(-1))
        indices = torch.clamp(indices, 1, len(block_data.type_tvd) - 1)

        tvd_low = block_data.type_tvd[indices - 1]
        tvd_high = block_data.type_tvd[indices]
        gr_low = block_data.type_gr[indices - 1]
        gr_high = block_data.type_gr[indices]

        t = (tvt_clamped.reshape(-1) - tvd_low) / (tvd_high - tvd_low + 1e-10)
        interp_gr = gr_low + t * (gr_high - gr_low)
        synthetic[:, local_start:local_end] = interp_gr.reshape(batch_size, seg_n)

    # Pearson correlation
    synthetic_centered = synthetic - synthetic.mean(dim=1, keepdim=True)
    numer = (block_data.zone_gr_centered * synthetic_centered).sum(dim=1)
    denom = torch.sqrt(block_data.zone_gr_ss * (synthetic_centered**2).sum(dim=1))
    pearson = torch.where(
        denom > 1e-10,
        numer / denom,
        torch.zeros(batch_size, device=device, dtype=GPU_DTYPE)
    )

    # MSE normalized
    mse = ((block_data.zone_gr - synthetic)**2).mean(dim=1)
    mse_norm = mse / (block_data.zone_gr_var + 1e-10)

    # Score (maximize) -> Loss (minimize)
    # score = pearson - mse_weight * mse_norm
    # loss = -score + angle_penalty
    loss = -pearson + mse_weight * mse_norm + angle_penalty

    # Penalize out-of-range solutions
    if out_of_range_conditions:
        out_of_range_mask = torch.stack(out_of_range_conditions, dim=0).any(dim=0)
        loss = torch.where(
            out_of_range_mask,
            torch.tensor(1e9, device=device, dtype=loss.dtype),
            loss
        )

    return loss, pearson, mse_norm


def compute_score_batch(
    angles_batch: torch.Tensor,
    block_data: BlockData,
    start_shift: float,
    trajectory_angle: float,
    angle_range: float,
    mse_weight: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute score for a batch of angle combinations.

    Same as compute_loss_batch but returns score (higher is better).
    Use this for algorithms that maximize.

    Returns:
        Tuple of:
        - score: (batch,) score values (higher is better)
        - pearson: (batch,) Pearson correlations
        - mse_norm: (batch,) normalized MSE values
    """
    loss, pearson, mse_norm = compute_loss_batch(
        angles_batch, block_data, start_shift, trajectory_angle, angle_range, mse_weight
    )
    score = -loss  # Invert for maximization
    return score, pearson, mse_norm
