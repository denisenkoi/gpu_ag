"""
Scipy Differential Evolution with GPU batch evaluation.

Key idea: scipy DE generates candidate populations on CPU,
but fitness evaluation is done in batches on GPU.
Data stays in GPU memory, only angles are transferred.

This should be faster than EvoTorch CMA-ES/SNES because:
1. DE doesn't get stuck in local minima as easily
2. Batch evaluation minimizes CPU<->GPU transfers
"""

import time
import torch
import numpy as np
from typing import Tuple, List, Optional, Union
from scipy.optimize import differential_evolution

from .base import BaseBlockOptimizer, OptimizeResult
from . import register_optimizer, prepare_block_data, compute_loss_batch, GPU_DTYPE


@register_optimizer('SCIPY_DE')
class ScipyDEOptimizer(BaseBlockOptimizer):
    """
    Differential Evolution from scipy with GPU batch fitness evaluation.

    The DE algorithm runs on CPU, but each generation's fitness
    is computed in a single GPU batch call.
    """

    def __init__(
        self,
        device: str = 'cuda',
        angle_range: float = 2.5,
        angle_step: float = 0.2,  # Not used, but kept for interface consistency
        mse_weight: float = 5.0,
        chunk_size: Optional[int] = None,
        # DE-specific parameters
        popsize: int = 50,
        maxiter: int = 100,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        strategy: str = 'best1bin',
        tol: float = 1e-6,
        polish: bool = False,
    ):
        super().__init__(device, angle_range, angle_step, mse_weight, chunk_size)
        self.popsize = popsize
        self.maxiter = maxiter
        self.mutation = mutation
        self.recombination = recombination
        self.strategy = strategy
        self.tol = tol
        self.polish = polish

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
        Optimize using scipy Differential Evolution with GPU batch evaluation.

        Args:
            return_result: If True, return OptimizeResult with full metrics.
                          If False (default), return legacy tuple for compatibility.
        """
        n_seg = len(segment_indices)
        start_time = time.perf_counter()

        # Prepare block data (transfers to GPU once, stays there!)
        block_data = prepare_block_data(
            segment_indices, well_md, well_tvd, well_gr,
            type_tvd, type_gr, self.device
        )

        # Segment MD lengths
        seg_md_lens = np.array([well_md[e] - well_md[s] for s, e in segment_indices], dtype=np.float32)

        # Bounds for each angle (absolute angles)
        bounds = [
            (trajectory_angle - self.angle_range, trajectory_angle + self.angle_range)
            for _ in range(n_seg)
        ]

        # Track best result with all metrics
        best_result = {
            'loss': float('inf'),
            'pearson': 0.0,
            'mse': 0.0,
            'score': 0.0,
            'angles': None
        }
        n_evaluations = [0]  # Use list for mutation in closure

        def fitness_vectorized(angles_population: np.ndarray) -> np.ndarray:
            """
            Vectorized fitness function for scipy DE.

            Args:
                angles_population: (n_seg, popsize) array - scipy passes transposed!

            Returns:
                (popsize,) array of fitness values (lower is better)
            """
            # scipy passes (n_seg, popsize), we need (popsize, n_seg)
            if angles_population.ndim == 2 and angles_population.shape[0] == n_seg:
                angles_population = angles_population.T

            n_evaluations[0] += len(angles_population)

            # Transfer angles to GPU (small data, fast)
            angles_tensor = torch.tensor(
                angles_population, device=self.device, dtype=GPU_DTYPE
            )

            # Compute loss in single GPU batch
            loss, pearson, mse = compute_loss_batch(
                angles_tensor, block_data, start_shift,
                trajectory_angle, self.angle_range, self.mse_weight
            )

            # Track best
            loss_np = loss.cpu().numpy()
            pearson_np = pearson.cpu().numpy()
            mse_np = mse.cpu().numpy()

            best_idx = np.argmin(loss_np)
            if loss_np[best_idx] < best_result['loss']:
                best_result['loss'] = float(loss_np[best_idx])
                best_result['pearson'] = float(pearson_np[best_idx])
                best_result['mse'] = float(mse_np[best_idx])
                # score = pearson - mse_weight * mse_norm, loss = -score + penalty
                # Approximate score from loss (may have penalty)
                best_result['score'] = -float(loss_np[best_idx])
                best_result['angles'] = angles_population[best_idx].copy()

            return loss_np

        # Run scipy DE with vectorized evaluation
        result = differential_evolution(
            fitness_vectorized,
            bounds,
            strategy=self.strategy,
            maxiter=self.maxiter,
            popsize=self.popsize,
            mutation=self.mutation,
            recombination=self.recombination,
            tol=self.tol,
            polish=self.polish,
            vectorized=True,  # Enable batch evaluation (scipy 1.9+)
            updating='deferred',  # Required for vectorized=True
            workers=1,  # We batch on GPU, no CPU parallelism needed
        )

        opt_ms = int((time.perf_counter() - start_time) * 1000)

        # Use tracked best (may be better than result.x due to early stopping)
        if best_result['angles'] is not None and best_result['loss'] < result.fun:
            best_angles = best_result['angles']
            best_pearson = best_result['pearson']
            best_mse = best_result['mse']
            best_loss = best_result['loss']
            best_score = best_result['score']
        else:
            best_angles = result.x
            best_loss = float(result.fun)
            # Recompute all metrics for result.x
            angles_tensor = torch.tensor(
                best_angles.reshape(1, -1), device=self.device, dtype=GPU_DTYPE
            )
            loss, pearson, mse = compute_loss_batch(
                angles_tensor, block_data, start_shift,
                trajectory_angle, self.angle_range, self.mse_weight
            )
            best_pearson = float(pearson[0].item())
            best_mse = float(mse[0].item())
            best_score = -best_loss

        # Compute end shift
        shift_deltas = np.tan(np.deg2rad(best_angles)) * seg_md_lens
        best_end_shift = start_shift + shift_deltas.sum()

        if return_result:
            return OptimizeResult(
                pearson=best_pearson,
                mse=best_mse,
                score=best_score,
                loss=best_loss,
                end_shift=best_end_shift,
                angles=best_angles.astype(np.float32),
                start_shift=start_shift,
                n_segments=n_seg,
                n_evaluations=n_evaluations[0],
                opt_ms=opt_ms,
                ref_angle=trajectory_angle,
            )

        return best_pearson, best_end_shift, best_angles.astype(np.float32), start_shift


@register_optimizer('SCIPY_DE_AGGRESSIVE')
class ScipyDEAggressiveOptimizer(ScipyDEOptimizer):
    """
    Aggressive DE settings for better exploration.
    Higher mutation, more iterations.
    """

    def __init__(self, **kwargs):
        defaults = {
            'popsize': 100,
            'maxiter': 200,
            'mutation': (0.8, 1.5),
            'recombination': 0.9,
            'strategy': 'rand1bin',
        }
        defaults.update(kwargs)
        super().__init__(**defaults)
