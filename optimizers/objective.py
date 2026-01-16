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
from scipy.signal import savgol_filter


# GPU dtype for consistency
GPU_DTYPE = torch.float32


@dataclass
class BeamPrefix:
    """
    Accumulated state from previous segments (for lookback beam search).

    For first block: empty tensors with end_shift=start_shift.
    For subsequent stages: contains synthetic/zone_gr/tvt from previous segments.
    """
    synthetic: torch.Tensor   # (n_prev_points,) or empty - accumulated projection
    zone_gr: torch.Tensor     # (n_prev_points,) or empty - accumulated well GR
    zone_gr_smooth: torch.Tensor  # (n_prev_points,) or empty - for self-corr
    tvt: torch.Tensor         # (n_prev_points,) or empty - for self-corr
    end_shift: float          # accumulated shift (replaces start_shift)

    @staticmethod
    def empty(start_shift: float, device: str = 'cuda') -> 'BeamPrefix':
        """Create empty prefix for first block."""
        return BeamPrefix(
            synthetic=torch.empty(0, device=device, dtype=GPU_DTYPE),
            zone_gr=torch.empty(0, device=device, dtype=GPU_DTYPE),
            zone_gr_smooth=torch.empty(0, device=device, dtype=GPU_DTYPE),
            tvt=torch.empty(0, device=device, dtype=GPU_DTYPE),
            end_shift=start_shift,
        )


@dataclass
class BlockData:
    """Precomputed data for a block of segments (held in GPU memory)."""
    # Zone data (whole block)
    zone_tvd: torch.Tensor        # (n_points,)
    zone_gr: torch.Tensor         # (n_points,)
    zone_gr_smooth: torch.Tensor  # (n_points,) savgol-filtered for selfcorr
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

    # Smoothed GR for selfcorr penalty (savgol window=51, polyorder=3)
    gr_slice = well_gr[start_idx:end_idx]
    if len(gr_slice) >= 51:
        gr_smooth_np = savgol_filter(gr_slice, 51, 3)
    else:
        gr_smooth_np = gr_slice  # Too short for smoothing
    zone_gr_smooth = torch.tensor(gr_smooth_np, device=device, dtype=GPU_DTYPE)

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
        zone_gr_smooth=zone_gr_smooth,
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
    prefix: BeamPrefix,
    trajectory_angle: float,
    angle_range: float,
    mse_weight: float = 5.0,
    selfcorr_threshold: float = 0.0,
    selfcorr_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute loss for a batch of angle combinations with lookback prefix support.

    Args:
        angles_batch: (batch, n_seg) angles in degrees for NEW segments only
        block_data: Precomputed data for NEW segments
        prefix: Accumulated state from previous segments (use BeamPrefix.empty() for first block)
        trajectory_angle: Reference angle (for penalty calculation)
        angle_range: Max allowed deviation from trajectory_angle
        mse_weight: Weight for MSE term

    Returns:
        Tuple of:
        - loss: (batch,) loss values (lower is better)
        - pearson: (batch,) Pearson correlations (computed on prefix + new)
        - mse_norm: (batch,) normalized MSE (computed on new only, prefix MSE is fixed)
        - new_synthetic: (batch, n_new_points) projection for new segments (to save in beam)
        - new_tvt: (batch, n_new_points) TVT for new segments (to save in beam)
    """
    if isinstance(prefix, (int, float)):
        raise TypeError(
            f"compute_loss_batch: start_shift removed, use BeamPrefix. "
            f"Got {type(prefix).__name__}={prefix}. "
            f"Use BeamPrefix.empty(start_shift) for first block."
        )

    device = block_data.device
    batch_size = angles_batch.shape[0]
    n_prefix_points = prefix.synthetic.shape[0]

    # Angle penalty: exponential for violations
    angle_deviation = torch.abs(angles_batch - trajectory_angle)
    angle_excess = angle_deviation - angle_range
    max_excess = torch.clamp(torch.max(angle_excess, dim=1).values, min=0.0)
    angle_penalty = torch.where(
        max_excess > 0,
        1e6 * torch.pow(torch.tensor(100.0, device=device, dtype=GPU_DTYPE), max_excess),
        torch.zeros(batch_size, device=device, dtype=GPU_DTYPE)
    )

    # Compute shifts from angles (starting from prefix.end_shift)
    shift_deltas = torch.tan(torch.deg2rad(angles_batch)) * block_data.seg_md_lens
    cumsum = torch.cumsum(shift_deltas, dim=1)
    start_shift_t = torch.tensor(prefix.end_shift, device=device, dtype=GPU_DTYPE)
    end_shifts = start_shift_t + cumsum
    start_shifts = torch.cat([
        start_shift_t.expand(batch_size, 1),
        end_shifts[:, :-1]
    ], dim=1)

    # Build synthetic GR for NEW segments
    new_synthetic = torch.zeros((batch_size, block_data.n_points), device=device, dtype=GPU_DTYPE)
    new_tvt = torch.zeros((batch_size, block_data.n_points), device=device, dtype=GPU_DTYPE)

    # Track out-of-range conditions
    MAX_TVT_EXTRAPOLATION = 0.1524  # 0.5 ft
    out_of_range_conditions = []

    for seg_i, (local_start, local_end, seg_n, seg_tvd, ratio) in enumerate(block_data.seg_data):
        seg_start = start_shifts[:, seg_i:seg_i+1]
        seg_end = end_shifts[:, seg_i:seg_i+1]
        seg_shifts = seg_start + ratio.unsqueeze(0) * (seg_end - seg_start)

        tvt = seg_tvd.unsqueeze(0) - seg_shifts
        new_tvt[:, local_start:local_end] = tvt

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
        new_synthetic[:, local_start:local_end] = interp_gr.reshape(batch_size, seg_n)

    # Combine prefix + new for Pearson (prefix is same for all batch, expand it)
    if n_prefix_points > 0:
        prefix_synthetic_expanded = prefix.synthetic.unsqueeze(0).expand(batch_size, -1)
        full_synthetic = torch.cat([prefix_synthetic_expanded, new_synthetic], dim=1)
        full_zone_gr = torch.cat([prefix.zone_gr, block_data.zone_gr])
    else:
        full_synthetic = new_synthetic
        full_zone_gr = block_data.zone_gr

    # Pearson correlation on FULL data (prefix + new)
    full_zone_gr_mean = full_zone_gr.mean()
    full_zone_gr_centered = full_zone_gr - full_zone_gr_mean
    full_zone_gr_ss = (full_zone_gr_centered**2).sum()

    synthetic_centered = full_synthetic - full_synthetic.mean(dim=1, keepdim=True)
    numer = (full_zone_gr_centered * synthetic_centered).sum(dim=1)
    denom = torch.sqrt(full_zone_gr_ss * (synthetic_centered**2).sum(dim=1))
    pearson = torch.where(
        denom > 1e-10,
        numer / denom,
        torch.zeros(batch_size, device=device, dtype=GPU_DTYPE)
    )

    # MSE normalized (on NEW segments only - prefix MSE is already fixed)
    mse = ((block_data.zone_gr - new_synthetic)**2).mean(dim=1)
    mse_norm = mse / (block_data.zone_gr_var + 1e-10)

    # Self-correlation penalty on FULL data (prefix + new)
    selfcorr_penalty = torch.zeros(batch_size, device=device, dtype=GPU_DTYPE)
    if selfcorr_threshold > 0 and selfcorr_weight > 0:
        if n_prefix_points > 0:
            prefix_tvt_expanded = prefix.tvt.unsqueeze(0).expand(batch_size, -1)
            full_tvt = torch.cat([prefix_tvt_expanded, new_tvt], dim=1)
            full_zone_gr_smooth = torch.cat([prefix.zone_gr_smooth, block_data.zone_gr_smooth])
        else:
            full_tvt = new_tvt
            full_zone_gr_smooth = block_data.zone_gr_smooth

        bin_size = 0.05  # 5cm bins
        for i in range(batch_size):
            tvt_i = full_tvt[i]
            gr_i = full_zone_gr_smooth

            tvt_min = tvt_i.min()
            bin_idx = ((tvt_i - tvt_min) / bin_size).long()
            n_bins = int(bin_idx.max().item()) + 1

            if n_bins < 5:
                continue

            bin_sums = torch.zeros(n_bins, device=device, dtype=GPU_DTYPE)
            bin_counts = torch.zeros(n_bins, device=device, dtype=GPU_DTYPE)

            bin_sums.scatter_add_(0, bin_idx, gr_i)
            bin_counts.scatter_add_(0, bin_idx, torch.ones_like(gr_i))

            valid_mask = bin_counts > 0
            if valid_mask.sum() < 5:
                continue

            bin_means = bin_sums[valid_mask] / bin_counts[valid_mask]
            std_val = bin_means.std()

            selfcorr_penalty[i] = torch.clamp(std_val - selfcorr_threshold, min=0) * selfcorr_weight

    # Loss = -score + penalties
    loss = -pearson + mse_weight * mse_norm + angle_penalty + selfcorr_penalty

    # Penalize out-of-range solutions
    if out_of_range_conditions:
        out_of_range_mask = torch.stack(out_of_range_conditions, dim=0).any(dim=0)
        loss = torch.where(
            out_of_range_mask,
            torch.tensor(1e9, device=device, dtype=loss.dtype),
            loss
        )

    return loss, pearson, mse_norm, new_synthetic, new_tvt


def compute_score_batch(
    angles_batch: torch.Tensor,
    block_data: BlockData,
    prefix: BeamPrefix,
    trajectory_angle: float,
    angle_range: float,
    mse_weight: float = 5.0,
    selfcorr_threshold: float = 0.0,
    selfcorr_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute score for a batch of angle combinations.

    Same as compute_loss_batch but returns score (higher is better).
    Use this for algorithms that maximize.

    Returns:
        Tuple of:
        - score: (batch,) score values (higher is better)
        - pearson: (batch,) Pearson correlations
        - mse_norm: (batch,) normalized MSE values
        - new_synthetic: (batch, n_new_points) projection for new segments
        - new_tvt: (batch, n_new_points) TVT for new segments
    """
    loss, pearson, mse_norm, new_synthetic, new_tvt = compute_loss_batch(
        angles_batch, block_data, prefix, trajectory_angle, angle_range, mse_weight,
        selfcorr_threshold, selfcorr_weight
    )
    score = -loss  # Invert for maximization
    return score, pearson, mse_norm, new_synthetic, new_tvt


def compute_std_batch(
    angles_batch: torch.Tensor,
    block_data: BlockData,
    prefix: BeamPrefix,
) -> torch.Tensor:
    """
    Compute STD(bin_means) for a batch of angle combinations.

    Used for logging/debugging selfcorr metric for top candidates.

    Args:
        prefix: BeamPrefix (NO backward compat - use BeamPrefix.empty(start_shift) for first block)

    Returns:
        std_values: (batch,) STD of bin means for each candidate
    """
    if isinstance(prefix, (int, float)):
        raise TypeError(
            f"compute_std_batch: start_shift removed, use BeamPrefix. "
            f"Got {type(prefix).__name__}={prefix}. "
            f"Use BeamPrefix.empty(start_shift) for first block."
        )

    device = block_data.device
    batch_size = angles_batch.shape[0]

    # Compute shifts from angles (starting from prefix.end_shift)
    shift_deltas = torch.tan(torch.deg2rad(angles_batch)) * block_data.seg_md_lens
    cumsum = torch.cumsum(shift_deltas, dim=1)
    start_shift_t = torch.tensor(prefix.end_shift, device=device, dtype=GPU_DTYPE)
    end_shifts = start_shift_t + cumsum
    start_shifts = torch.cat([
        start_shift_t.expand(batch_size, 1),
        end_shifts[:, :-1]
    ], dim=1)

    # Build TVT for each point
    tvt_all = torch.zeros((batch_size, block_data.n_points), device=device, dtype=GPU_DTYPE)

    for seg_i, (local_start, local_end, seg_n, seg_tvd, ratio) in enumerate(block_data.seg_data):
        seg_start = start_shifts[:, seg_i:seg_i+1]
        seg_end = end_shifts[:, seg_i:seg_i+1]
        seg_shifts = seg_start + ratio.unsqueeze(0) * (seg_end - seg_start)
        tvt = seg_tvd.unsqueeze(0) - seg_shifts
        tvt_all[:, local_start:local_end] = tvt

    # Compute STD(bin_means) for each candidate
    std_values = torch.zeros(batch_size, device=device, dtype=GPU_DTYPE)
    bin_size = 0.05  # 5cm bins

    for i in range(batch_size):
        tvt_i = tvt_all[i]
        gr_i = block_data.zone_gr_smooth

        tvt_min = tvt_i.min()
        bin_idx = ((tvt_i - tvt_min) / bin_size).long()
        n_bins = int(bin_idx.max().item()) + 1

        if n_bins < 5:
            std_values[i] = float('nan')
            continue

        bin_sums = torch.zeros(n_bins, device=device, dtype=GPU_DTYPE)
        bin_counts = torch.zeros(n_bins, device=device, dtype=GPU_DTYPE)

        bin_sums.scatter_add_(0, bin_idx, gr_i)
        bin_counts.scatter_add_(0, bin_idx, torch.ones_like(gr_i))

        valid_mask = bin_counts > 0
        if valid_mask.sum() < 5:
            std_values[i] = float('nan')
            continue

        bin_means = bin_sums[valid_mask] / bin_counts[valid_mask]
        std_values[i] = bin_means.std()

    return std_values
