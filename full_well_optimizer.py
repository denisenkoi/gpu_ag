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
from cpu_baseline.typelog_preprocessing import prepare_typelog, normalize_well_gr
import pandas as pd

# Load SC baseline from CSV (once at module load)
_SC_BASELINE_PATH = Path(__file__).parent / 'results' / 'self_correlation_baseline.csv'
_SC_BASELINE_CACHE = {}

def get_sc_landing_rmse(well_name: str) -> float:
    """Get sc_landing_rmse for a well from cached CSV."""
    global _SC_BASELINE_CACHE
    if not _SC_BASELINE_CACHE:
        if _SC_BASELINE_PATH.exists():
            df = pd.read_csv(_SC_BASELINE_PATH)
            _SC_BASELINE_CACHE = dict(zip(df['well_name'], df['gr_var_mean']))
    return _SC_BASELINE_CACHE.get(well_name, 4.0)  # Default to median ~4%

# Read typelog mode from env
USE_PSEUDO_TYPELOG = os.getenv('USE_PSEUDO_TYPELOG', 'True').lower() in ('true', '1', 'yes')

# SC (Self-Correlation) penalty - DISABLED for now (expensive, not proven useful)
SC_ENABLED = False

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# GPU configurations
# NOTE: 5090 degrades badly at chunk_size=500k (55s vs 8s)
# Conservative settings for stability
GPU_CONFIGS = {
    '3090': {'chunk_size': 70000, 'min_free_gb': 15},
    '5090': {'chunk_size': 100000, 'min_free_gb': 20},
    'default': {'chunk_size': 70000, 'min_free_gb': 10},
}


def detect_gpu_model() -> str:
    """Detect GPU model (3090 or 5090)."""
    if not torch.cuda.is_available():
        return 'default'
    name = torch.cuda.get_device_name(0).lower()
    if '5090' in name:
        return '5090'
    elif '3090' in name:
        return '3090'
    return 'default'


def get_gpu_free_memory_gb() -> float:
    """Get free GPU memory in GB."""
    if not torch.cuda.is_available():
        return 0.0
    free, total = torch.cuda.mem_get_info(0)
    return free / (1024 ** 3)


def check_gpu_memory(gpu_model: str = None) -> bool:
    """
    Check if GPU has enough free memory for optimization.
    Returns True if OK, raises RuntimeError if not enough memory.
    """
    if gpu_model is None:
        gpu_model = detect_gpu_model()

    config = GPU_CONFIGS.get(gpu_model, GPU_CONFIGS['default'])
    min_free_gb = config['min_free_gb']
    free_gb = get_gpu_free_memory_gb()

    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Not enough GPU memory! GPU={gpu_model}, "
            f"free={free_gb:.1f}GB, required={min_free_gb}GB. "
            f"Check for other processes using the GPU."
        )
    return True


def get_chunk_size(gpu_model: str = None) -> int:
    """Get optimal chunk_size for detected GPU."""
    if gpu_model is None:
        gpu_model = detect_gpu_model()
    return GPU_CONFIGS.get(gpu_model, GPU_CONFIGS['default'])['chunk_size']


@dataclass
class OptimizedSegment:
    """Result of segment optimization."""
    start_md: float
    end_md: float
    start_shift: float
    end_shift: float
    angle_deg: float
    pearson: float


def compute_sc_penalty_gpu(
    tvt: torch.Tensor,  # [n_combos, n_points]
    zone_gr: torch.Tensor,  # [n_points]
    sc_landing_rmse: float,
    bin_size: float = 0.1,  # 10cm bins for SC (larger than 5cm for memory)
    min_bins_threshold: int = 5,  # Minimum overlap bins for reliable SC
    device: str = DEVICE,
) -> torch.Tensor:
    """
    Compute self-correlation penalty on GPU using scatter operations.

    Returns: sc_penalty tensor [n_combos]
    """
    n_combos, n_points = tvt.shape

    # Normalize GR to 0-100% (same as baseline calculation)
    gr_min = zone_gr.min()
    gr_max = zone_gr.max()
    gr_range = gr_max - gr_min
    if gr_range < 1e-6:
        return torch.zeros(n_combos, device=device, dtype=tvt.dtype)
    zone_gr_norm = (zone_gr - gr_min) / gr_range * 100.0

    # Bin indices
    tvt_min = tvt.min()
    tvt_max = tvt.max()
    tvt_range = tvt_max - tvt_min

    if tvt_range < bin_size:
        return torch.zeros(n_combos, device=device, dtype=tvt.dtype)

    n_bins = int(tvt_range / bin_size) + 1

    # Limit bins to avoid OOM (max ~500 bins = 50m range)
    if n_bins > 500:
        bin_size = tvt_range / 500
        n_bins = 500

    bin_idx = ((tvt - tvt_min) / bin_size).long()
    bin_idx = torch.clamp(bin_idx, 0, n_bins - 1)

    # Flat index for scatter: combo_idx * n_bins + bin_idx
    combo_idx = torch.arange(n_combos, device=device).unsqueeze(1)
    flat_bin_idx = (combo_idx * n_bins + bin_idx).reshape(-1)

    # Expand normalized GR for all combos
    zone_gr_expanded = zone_gr_norm.unsqueeze(0).expand(n_combos, -1).reshape(-1)

    # Accumulate sum, sum_sq, count via scatter_add
    total_bins = n_combos * n_bins
    sum_gr = torch.zeros(total_bins, device=device, dtype=tvt.dtype)
    sum_gr_sq = torch.zeros(total_bins, device=device, dtype=tvt.dtype)
    count = torch.zeros(total_bins, device=device, dtype=tvt.dtype)

    sum_gr.scatter_add_(0, flat_bin_idx, zone_gr_expanded)
    sum_gr_sq.scatter_add_(0, flat_bin_idx, zone_gr_expanded ** 2)
    count.scatter_add_(0, flat_bin_idx, torch.ones_like(zone_gr_expanded))

    # Reshape to [n_combos, n_bins]
    sum_gr = sum_gr.reshape(n_combos, n_bins)
    sum_gr_sq = sum_gr_sq.reshape(n_combos, n_bins)
    count = count.reshape(n_combos, n_bins)

    # Variance in bins with count >= 2
    mask = count >= 2
    mean_gr = torch.where(mask, sum_gr / (count + 1e-10), torch.zeros_like(sum_gr))
    var_gr = torch.where(mask, sum_gr_sq / (count + 1e-10) - mean_gr ** 2, torch.zeros_like(sum_gr))
    var_gr = torch.clamp(var_gr, min=0)  # Numerical stability

    # RMSE = sqrt(mean(var)) for bins with count >= 2
    n_valid_bins = mask.sum(dim=1)
    sum_var = (var_gr * mask.float()).sum(dim=1)
    mean_var = torch.where(n_valid_bins > 0, sum_var / n_valid_bins, torch.zeros_like(sum_var))
    sc_rmse = torch.sqrt(mean_var)

    # SC penalty = max(sc_rmse - baseline, 0)
    sc_penalty = torch.clamp(sc_rmse - sc_landing_rmse, min=0)

    # Zero penalty if too few overlap bins
    sc_penalty = torch.where(n_valid_bins >= min_bins_threshold, sc_penalty, torch.zeros_like(sc_penalty))

    return sc_penalty


def optimize_segment_block_gpu(
    segment_indices: List[Tuple[int, int]],
    start_shift: float,
    trajectory_angle: float,
    well_md: np.ndarray,
    well_tvd: np.ndarray,
    well_gr: np.ndarray,
    type_tvd: np.ndarray,
    type_gr: np.ndarray,
    angle_range: float = 1.5,
    angle_step: float = 0.2,
    pearson_power: float = 1.0,
    pearson_clamp: float = 0.0,
    mse_weight: float = 0.1,  # Weight for MSE in score formula
    sc_weight: float = 0.0,  # Weight for SC penalty (0 = disabled)
    sc_landing_rmse: float = 4.0,  # Baseline SC RMSE from landing zone
    device: str = DEVICE,
    chunk_size: int = None,  # Auto-detect based on GPU model
) -> Tuple[float, float, np.ndarray, float]:
    """
    GPU optimization for a block of segments.

    Returns: (best_pearson, best_end_shift, best_angles, best_start_shift)
    """
    if chunk_size is None:
        chunk_size = get_chunk_size()

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

    type_tvd_t = torch.tensor(type_tvd, device=device, dtype=GPU_DTYPE)  # Already in well coordinates
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

        # Build synthetic and TVT
        synthetic = torch.zeros((chunk_n, n_points), device=device, dtype=GPU_DTYPE)
        all_tvt = torch.zeros((chunk_n, n_points), device=device, dtype=GPU_DTYPE) if SC_ENABLED and sc_weight > 0 else None

        for seg_i, (local_start, local_end, seg_n, seg_tvd, ratio) in enumerate(seg_data):
            seg_start = start_shifts[:, seg_i:seg_i+1]
            seg_end = end_shifts[:, seg_i:seg_i+1]
            seg_shifts = seg_start + ratio.unsqueeze(0) * (seg_end - seg_start)

            tvt = seg_tvd.unsqueeze(0) - seg_shifts
            tvt_clamped = torch.clamp(tvt, type_tvd_t[0], type_tvd_t[-1])

            # Store TVT for SC penalty
            if all_tvt is not None:
                all_tvt[:, local_start:local_end] = tvt

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

        # Score: pearson - mse_weight * MSE_norm - sc_weight * sc_penalty
        mse = ((zone_gr - synthetic)**2).mean(dim=1)
        mse_norm = mse / (zone_gr.var() + 1e-10)

        # SC penalty (if enabled)
        if SC_ENABLED and sc_weight > 0 and all_tvt is not None:
            sc_penalty = compute_sc_penalty_gpu(
                all_tvt, zone_gr, sc_landing_rmse,
                bin_size=0.1,  # 10cm bins
                min_bins_threshold=5,
                device=device
            )
            scores = pearsons - mse_weight * mse_norm - sc_weight * sc_penalty
        else:
            scores = pearsons - mse_weight * mse_norm

        chunk_best_idx = torch.argmax(scores).item()
        chunk_best_score = scores[chunk_best_idx].item()

        if chunk_best_score > best_score:
            best_score = chunk_best_score
            best_idx_global = chunk_start + chunk_best_idx
            best_pearson = pearsons[chunk_best_idx].item()

        # Cleanup chunk tensors to prevent VRAM leak
        del chunk_angles, chunk_start_shifts, seg_md_lens_t, shift_deltas
        del cumsum, end_shifts, start_shifts, synthetic, synthetic_centered
        del numer, denom, pearsons, mse, mse_norm, scores
        if all_tvt is not None:
            del all_tvt
        torch.cuda.empty_cache()

    # Get best result
    best_start_shift = float(all_start_shifts_np[best_idx_global])
    best_angles = all_angles_np[best_idx_global]
    best_shift_deltas = np.tan(np.radians(best_angles)) * seg_md_lens
    best_end_shift = best_start_shift + np.sum(best_shift_deltas)

    return best_pearson, best_end_shift, best_angles, best_start_shift


def optimize_segment_block_evolutionary(
    segment_indices: List[Tuple[int, int]],
    start_shift: float,
    trajectory_angle: float,
    well_md: np.ndarray,
    well_tvd: np.ndarray,
    well_gr: np.ndarray,
    type_tvd: np.ndarray,
    type_gr: np.ndarray,
    angle_range: float = 1.5,
    mse_weight: float = 0.1,
    device: str = DEVICE,
    algorithm: str = 'CMAES',  # CMAES or SNES
    popsize: int = 100,
    maxiter: int = 50,
) -> Tuple[float, float, np.ndarray, float]:
    """
    Evolutionary optimization (CMA-ES/SNES) for a block of segments.
    Uses soft exponential penalty for angle constraint instead of hard bounds.

    Returns: (best_pearson, best_end_shift, best_angles, best_start_shift)
    """
    from evotorch import Problem
    from evotorch.algorithms import CMAES, SNES

    n_seg = len(segment_indices)

    # Segment MD lengths
    seg_md_lens = np.array([well_md[e] - well_md[s] for s, e in segment_indices], dtype=np.float32)
    seg_md_lens_t = torch.tensor(seg_md_lens, device=device, dtype=GPU_DTYPE)

    # Zone data
    start_idx = segment_indices[0][0]
    end_idx = segment_indices[-1][1]
    n_points = end_idx - start_idx

    zone_tvd = torch.tensor(well_tvd[start_idx:end_idx], device=device, dtype=GPU_DTYPE)
    zone_gr = torch.tensor(well_gr[start_idx:end_idx], device=device, dtype=GPU_DTYPE)
    zone_gr_centered = zone_gr - zone_gr.mean()
    zone_gr_ss = (zone_gr_centered**2).sum()
    zone_gr_var = zone_gr.var()

    type_tvd_t = torch.tensor(type_tvd, device=device, dtype=GPU_DTYPE)  # Already in well coordinates
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

    def objective_fn(angles_batch: torch.Tensor) -> torch.Tensor:
        """
        Compute objective for a batch of angle combinations.
        angles_batch: (batch, n_seg) or (n_seg,) - angles in degrees, centered at trajectory_angle
        Returns: (batch,) or scalar - loss values (lower is better)
        """
        # Handle single solution (evotorch sometimes passes 1D tensor)
        squeeze_output = False
        if angles_batch.dim() == 1:
            angles_batch = angles_batch.unsqueeze(0)
            squeeze_output = True

        batch_size = angles_batch.shape[0]

        # Angles are relative to trajectory_angle, convert to absolute
        abs_angles = angles_batch + trajectory_angle

        # Exponential penalty for angle violations: 1e6 * 100^max_excess
        angle_excess = torch.abs(angles_batch) - angle_range
        max_excess = torch.clamp(torch.max(angle_excess, dim=1).values, min=0.0)
        angle_penalty = torch.where(
            max_excess > 0,
            1e6 * torch.pow(torch.tensor(100.0, device=device, dtype=GPU_DTYPE), max_excess),
            torch.zeros(batch_size, device=device, dtype=GPU_DTYPE)
        )

        # Compute shifts from angles
        shift_deltas = torch.tan(torch.deg2rad(abs_angles)) * seg_md_lens_t
        cumsum = torch.cumsum(shift_deltas, dim=1)
        start_shift_t = torch.tensor(start_shift, device=device, dtype=GPU_DTYPE)
        end_shifts = start_shift_t + cumsum
        start_shifts = torch.cat([
            start_shift_t.expand(batch_size, 1),
            end_shifts[:, :-1]
        ], dim=1)

        # Build synthetic GR
        synthetic = torch.zeros((batch_size, n_points), device=device, dtype=GPU_DTYPE)

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
            synthetic[:, local_start:local_end] = interp_gr.reshape(batch_size, seg_n)

        # Pearson correlation
        synthetic_centered = synthetic - synthetic.mean(dim=1, keepdim=True)
        numer = (zone_gr_centered * synthetic_centered).sum(dim=1)
        denom = torch.sqrt(zone_gr_ss * (synthetic_centered**2).sum(dim=1))
        pearsons = torch.where(denom > 1e-10, numer / denom, torch.zeros(batch_size, device=device, dtype=GPU_DTYPE))

        # MSE normalized
        mse = ((zone_gr - synthetic)**2).mean(dim=1)
        mse_norm = mse / (zone_gr_var + 1e-10)

        # Score (maximize) -> Loss (minimize)
        # Original: score = pearson - mse_weight * mse_norm
        # Loss = -score + angle_penalty
        loss = -pearsons + mse_weight * mse_norm + angle_penalty

        if squeeze_output:
            return loss.squeeze(0)
        return loss

    # Define EvoTorch Problem
    class BlockOptProblem(Problem):
        def __init__(self):
            super().__init__(
                objective_func=self._evaluate,
                objective_sense='min',
                solution_length=n_seg,
                initial_bounds=(-angle_range, angle_range),
                device=device,
                dtype=GPU_DTYPE,
            )

        def _evaluate(self, solutions):
            x = solutions.values
            fitness = objective_fn(x)
            solutions.set_evals(fitness)

    problem = BlockOptProblem()

    # Select algorithm
    if algorithm.upper() == 'SNES':
        searcher = SNES(problem, popsize=popsize, stdev_init=angle_range / 2)
    else:  # CMAES
        searcher = CMAES(problem, popsize=popsize, stdev_init=angle_range / 2)

    # Run optimization and track best
    best_fun = float('inf')
    best_angles_relative = None

    for _ in range(maxiter):
        searcher.step()
        # Get population and find best
        pop = searcher.population
        if pop is not None and len(pop) > 0:
            best_idx = pop.evals.argmin()
            current_fun = pop.evals[best_idx].item()
            if current_fun < best_fun:
                best_fun = current_fun
                best_angles_relative = pop.values[best_idx].cpu().numpy()

    if best_angles_relative is None:
        # Fallback: use center of distribution
        best_angles_relative = np.zeros(n_seg)

    best_angles = best_angles_relative + trajectory_angle

    # Compute final values
    best_shift_deltas = np.tan(np.radians(best_angles)) * seg_md_lens
    best_end_shift = start_shift + np.sum(best_shift_deltas)

    # Compute pearson for best solution
    with torch.no_grad():
        loss = objective_fn(torch.tensor(best_angles_relative, device=device, dtype=GPU_DTYPE).unsqueeze(0))
        # Approximate pearson from loss (without penalty)
        abs_angles_t = torch.tensor(best_angles, device=device, dtype=GPU_DTYPE).unsqueeze(0)
        shift_deltas_t = torch.tan(torch.deg2rad(abs_angles_t)) * seg_md_lens_t
        cumsum = torch.cumsum(shift_deltas_t, dim=1)
        end_shifts = start_shift + cumsum
        start_shifts_t = torch.cat([
            torch.tensor([[start_shift]], device=device, dtype=GPU_DTYPE),
            end_shifts[:, :-1]
        ], dim=1)

        synthetic = torch.zeros((1, n_points), device=device, dtype=GPU_DTYPE)
        for seg_i, (local_start, local_end, seg_n, seg_tvd, ratio) in enumerate(seg_data):
            seg_start = start_shifts_t[:, seg_i:seg_i+1]
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
            synthetic[:, local_start:local_end] = interp_gr.reshape(1, seg_n)

        synthetic_centered = synthetic - synthetic.mean(dim=1, keepdim=True)
        numer = (zone_gr_centered * synthetic_centered).sum(dim=1)
        denom = torch.sqrt(zone_gr_ss * (synthetic_centered**2).sum(dim=1))
        best_pearson = (numer / (denom + 1e-10)).item()

    return best_pearson, best_end_shift, best_angles, start_shift


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
    mse_weight: float = 0.1,  # Weight for MSE in score formula
    sc_weight: float = 0.0,  # Weight for SC penalty (0 = disabled)
    segments_per_block: int = 5,
    end_zone_angle_reduction: float = 0.5,  # Reduce angle_range near end
    start_from_landing: bool = True,  # Start from landing_end_87_200 instead of OTSU
    verbose: bool = False,
    algorithm: str = 'BRUTEFORCE',  # BRUTEFORCE, CMAES, or SNES
    evo_popsize: int = 100,  # Population size for evolutionary algorithms
    evo_maxiter: int = 50,  # Max iterations for evolutionary algorithms
    chunk_size: int = None,  # Auto-detect based on GPU model
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

    # Get typewell with correct coordinate alignment (tvd_shift applied BEFORE stitching)
    type_tvd, type_gr, typelog_meta = prepare_typelog(
        well_data,
        use_pseudo=USE_PSEUDO_TYPELOG,
        apply_smoothing=True
    )

    # Interpolate GR to well_md and apply same normalization as TypeLog
    well_gr = np.interp(well_md, log_md, log_gr)
    norm_multiplier = float(well_data.get('norm_multiplier', 1.0))
    well_gr = normalize_well_gr(
        well_gr,
        norm_multiplier,
        typelog_meta['gr_min'],
        typelog_meta['gr_max']
    )

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

        # Get SC baseline for this well
        sc_landing_rmse = get_sc_landing_rmse(well_name) if SC_ENABLED and sc_weight > 0 else 4.0

        # Optimize block
        if algorithm.upper() in ('CMAES', 'SNES'):
            # Evolutionary optimization
            pearson, end_shift, best_angles, _ = optimize_segment_block_evolutionary(
                seg_indices, current_shift, traj_angle,
                well_md, well_tvd, well_gr,
                type_tvd, type_gr,
                angle_range=effective_angle_range,
                mse_weight=mse_weight,
                algorithm=algorithm.upper(),
                popsize=evo_popsize,
                maxiter=evo_maxiter,
            )
        else:
            # Brute-force (default)
            pearson, end_shift, best_angles, _ = optimize_segment_block_gpu(
                seg_indices, current_shift, traj_angle,
                well_md, well_tvd, well_gr,
                type_tvd, type_gr,
                angle_range=effective_angle_range,
                angle_step=angle_step,
                pearson_power=pearson_power,
                pearson_clamp=pearson_clamp,
                mse_weight=mse_weight,
                sc_weight=sc_weight,
                sc_landing_rmse=sc_landing_rmse,
                chunk_size=chunk_size,
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
    """Test on all 100 wells with CSV export after each well."""
    import csv
    from datetime import datetime

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_started = datetime.now().isoformat()
    print(f"Testing full well optimization on 100 wells (angle_range=±{angle_range}°)")
    print(f"Run ID: {run_id}")
    print(f"Started: {run_started}")
    print("=" * 70)
    sys.stdout.flush()

    # CSV setup - new file for this run
    csv_path = Path(__file__).parent / 'results' / f'full_well_{run_id}.csv'
    csv_path.parent.mkdir(exist_ok=True)
    csv_header = [
        'run_id', 'well_name', 'row_type', 'seg_idx',
        'md_start', 'md_end', 'end_shift', 'end_error',
        'angle_deg', 'pearson', 'baseline_error', 'opt_error',
        'n_segments', 'prep_ms', 'opt_ms', 'total_ms',
        'started_at', 'finished_at'
    ]

    # Write header
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writeheader()

    print(f"CSV: {csv_path}")
    print("-" * 70)
    sys.stdout.flush()

    ds = torch.load('dataset/gpu_ag_dataset.pt', weights_only=False)

    errors = []
    baseline_errors = []
    t_start = time.time()

    for i, (well_name, well_data) in enumerate(ds.items()):
        well_started = datetime.now().isoformat()
        t_well_start = time.time()

        # === PREP PHASE ===
        t_prep_start = time.time()

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

        prep_ms = int((time.time() - t_prep_start) * 1000)

        # === OPTIMIZATION PHASE ===
        t_opt_start = time.time()

        segments = optimize_full_well(
            well_name, well_data,
            angle_range=angle_range,
            verbose=False,
        )

        opt_ms = int((time.time() - t_opt_start) * 1000)

        if not segments:
            raise RuntimeError(f"No segments returned for {well_name}")

        pred_end_shift = segments[-1].end_shift
        opt_error = pred_end_shift - ref_end_shift
        errors.append(opt_error)

        well_finished = datetime.now().isoformat()
        total_ms = int((time.time() - t_well_start) * 1000)

        # === WRITE TO CSV IMMEDIATELY ===
        csv_rows = []

        # Segment rows
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
                'n_segments': '',
                'prep_ms': '',
                'opt_ms': '',
                'total_ms': '',
                'started_at': '',
                'finished_at': '',
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
            'n_segments': len(segments),
            'prep_ms': prep_ms,
            'opt_ms': opt_ms,
            'total_ms': total_ms,
            'started_at': well_started,
            'finished_at': well_finished,
        })

        # Append to CSV and flush
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_header)
            writer.writerows(csv_rows)

        # Progress output
        status = "✓" if abs(opt_error) < abs(baseline_error) else "✗"
        print(f"{i+1:3d}/100 {well_name:<20} base={baseline_error:+7.2f}m opt={opt_error:+7.2f}m {status} prep={prep_ms}ms opt={opt_ms}ms")
        sys.stdout.flush()

    t_elapsed = time.time() - t_start
    run_finished = datetime.now().isoformat()

    errors = np.array(errors)
    baseline_errors = np.array(baseline_errors)

    rmse_opt = np.sqrt(np.mean(errors**2))
    rmse_baseline = np.sqrt(np.mean(baseline_errors**2))
    improved = np.sum(np.abs(errors) < np.abs(baseline_errors))

    # Summary row
    summary_row = {
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
        'n_segments': improved,
        'prep_ms': '',
        'opt_ms': '',
        'total_ms': int(t_elapsed * 1000),
        'started_at': run_started,
        'finished_at': run_finished,
    }

    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        writer.writerow(summary_row)

    print("-" * 70)
    print(f"Results (angle_range=±{angle_range}°):")
    print(f"  Baseline RMSE: {rmse_baseline:.2f}m")
    print(f"  Optimized RMSE: {rmse_opt:.2f}m")
    print(f"  Improvement: {rmse_baseline - rmse_opt:.2f}m ({100*(rmse_baseline-rmse_opt)/rmse_baseline:.1f}%)")
    print(f"  Wells improved: {improved}/100")
    print(f"  Time: {t_elapsed:.1f}s ({t_elapsed/100:.2f}s/well)")
    print(f"  CSV: {csv_path}")
    print(f"  Finished: {run_finished}")
    sys.stdout.flush()

    return rmse_opt, improved


if __name__ == '__main__':
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Full well GPU optimizer')
    parser.add_argument('--all', action='store_true', help='Test all 100 wells')
    parser.add_argument('--angle-range', type=float, default=1.5, help='Angle range in degrees')
    parser.add_argument('--angle-step', type=float, default=0.2, help='Angle step in degrees')
    parser.add_argument('--chunk-size', type=int, default=None, help='Override chunk size')
    parser.add_argument('--skip-memory-check', action='store_true', help='Skip GPU memory check')
    args = parser.parse_args()

    # Detect GPU and check memory
    gpu_model = detect_gpu_model()
    chunk_size = args.chunk_size or get_chunk_size(gpu_model)
    free_gb = get_gpu_free_memory_gb()

    print(f"GPU: {gpu_model}, Free memory: {free_gb:.1f}GB, chunk_size: {chunk_size}")

    if not args.skip_memory_check:
        try:
            check_gpu_memory(gpu_model)
        except RuntimeError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    if args.all:
        print("\n" + "="*70)
        test_all_wells(angle_range=args.angle_range)
    else:
        test_single_well()
