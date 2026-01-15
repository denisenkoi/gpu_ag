"""
BruteForce optimizer - exhaustive grid search.

Current best algorithm: RMSE 2.99m on 100 wells.
"""

import time
import torch
import numpy as np
from typing import Tuple, List, Optional, Union

from .base import BaseBlockOptimizer, OptimizeResult
from . import register_optimizer, prepare_block_data, compute_score_batch, GPU_DTYPE


def get_chunk_size() -> int:
    """Auto-detect optimal chunk size based on GPU model."""
    if not torch.cuda.is_available():
        return 100000

    gpu_name = torch.cuda.get_device_name(0).lower()
    free_mem = torch.cuda.get_device_properties(0).total_memory / 1e9

    if '5090' in gpu_name or free_mem > 28:
        return 150000
    elif '4090' in gpu_name or free_mem > 20:
        return 200000
    elif '3090' in gpu_name or free_mem > 20:
        return 100000  # 3090 has less effective bandwidth
    else:
        return 50000


@register_optimizer('BRUTEFORCE')
class BruteForceOptimizer(BaseBlockOptimizer):
    """
    Exhaustive grid search over all angle combinations.

    Features:
    - Adaptive step: 0.1° for segments >50m, default for shorter
    - Chunked processing to fit GPU memory
    - Full enumeration guarantees global optimum within grid
    """

    def __init__(
        self,
        device: str = 'cuda',
        angle_range: float = 2.5,
        angle_step: float = 0.2,
        mse_weight: float = 5.0,
        chunk_size: Optional[int] = None,
        adaptive_step: bool = True,
        fine_step: float = 0.1,
        long_segment_threshold: float = 50.0,
    ):
        super().__init__(device, angle_range, angle_step, mse_weight, chunk_size)
        self.adaptive_step = adaptive_step
        self.fine_step = fine_step
        self.long_segment_threshold = long_segment_threshold

        if self.chunk_size is None:
            self.chunk_size = get_chunk_size()

    def optimize(
        self,
        segment_indices: List[Tuple[int, int]],
        start_shift: float,
        trajectory_angle: float,
        well_md: np.ndarray,
        well_tvd: np.ndarray,
        well_gr: np.ndarray,
        type_tvd: np.ndarray,
        type_gr: np.ndarray,
        return_result: bool = False,
        **kwargs
    ) -> Union[Tuple[float, float, np.ndarray, float], OptimizeResult]:
        """
        Optimize by exhaustive enumeration of angle grid.

        Args:
            return_result: If True, return OptimizeResult with full metrics
        """
        t_start = time.perf_counter()
        n_seg = len(segment_indices)

        # Prepare block data (transfers to GPU once)
        block_data = prepare_block_data(
            segment_indices, well_md, well_tvd, well_gr,
            type_tvd, type_gr, self.device
        )

        # Segment MD lengths for adaptive step
        seg_md_lens = np.array([well_md[e] - well_md[s] for s, e in segment_indices], dtype=np.float32)

        # Generate angle grids (adaptive step for long segments)
        angle_grids = []
        for seg_len in seg_md_lens:
            if self.adaptive_step and seg_len > self.long_segment_threshold:
                step = self.fine_step
            else:
                step = self.angle_step

            n_steps = int(2 * self.angle_range / step) + 1
            grid = np.linspace(
                trajectory_angle - self.angle_range,
                trajectory_angle + self.angle_range,
                n_steps
            )
            angle_grids.append(grid)

        # Total combinations
        grid_sizes = [len(g) for g in angle_grids]
        n_combos = 1
        for s in grid_sizes:
            n_combos *= s

        # Transfer grids to GPU
        grids_gpu = [torch.tensor(g, device=self.device, dtype=GPU_DTYPE) for g in angle_grids]

        best_score = -1e9
        best_idx_global = 0
        best_pearson = 0.0
        best_mse = 0.0
        n_evaluations = 0

        # Process in chunks
        for chunk_start in range(0, n_combos, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, n_combos)
            chunk_n = chunk_end - chunk_start
            n_evaluations += chunk_n

            # Generate angle combinations on GPU from flat indices
            indices = torch.arange(chunk_start, chunk_end, device=self.device, dtype=torch.long)
            chunk_angles = torch.zeros((chunk_n, n_seg), device=self.device, dtype=GPU_DTYPE)

            divisor = 1
            for seg in reversed(range(n_seg)):
                seg_indices = (indices // divisor) % grid_sizes[seg]
                chunk_angles[:, seg] = grids_gpu[seg][seg_indices]
                divisor *= grid_sizes[seg]

            # Compute scores
            scores, pearsons, mse_norms = compute_score_batch(
                chunk_angles, block_data, start_shift,
                trajectory_angle, self.angle_range, self.mse_weight
            )

            # Find best in chunk
            chunk_best_idx = scores.argmax().item()
            chunk_best_score = scores[chunk_best_idx].item()

            if chunk_best_score > best_score:
                best_score = chunk_best_score
                best_idx_global = chunk_start + chunk_best_idx
                best_pearson = pearsons[chunk_best_idx].item()
                best_mse = mse_norms[chunk_best_idx].item()

            # Periodic cache clear
            if chunk_start > 0 and chunk_start % (20 * self.chunk_size) == 0:
                torch.cuda.empty_cache()

        # Reconstruct best angles from index
        best_angles = np.zeros(n_seg, dtype=np.float32)
        idx = best_idx_global
        for seg in reversed(range(n_seg)):
            seg_idx = idx % grid_sizes[seg]
            best_angles[seg] = angle_grids[seg][seg_idx]
            idx //= grid_sizes[seg]

        # Compute end shift for best solution
        shift_deltas = np.tan(np.deg2rad(best_angles)) * seg_md_lens
        best_end_shift = start_shift + shift_deltas.sum()

        if return_result:
            opt_ms = int((time.perf_counter() - t_start) * 1000)
            # loss = -score (for minimization, no penalty since within grid)
            best_loss = -best_score
            return OptimizeResult(
                pearson=best_pearson,
                mse=best_mse,
                score=best_score,
                loss=best_loss,
                end_shift=best_end_shift,
                angles=best_angles,
                start_shift=start_shift,
                n_segments=n_seg,
                n_evaluations=n_evaluations,
                opt_ms=opt_ms,
                ref_angle=trajectory_angle,
            )

        return best_pearson, best_end_shift, best_angles, start_shift
