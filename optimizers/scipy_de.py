"""
Scipy Differential Evolution with GPU batch evaluation.

Key idea: scipy DE generates candidate populations on CPU,
but fitness evaluation is done in batches on GPU.
Data stays in GPU memory, only angles are transferred.

This should be faster than EvoTorch CMA-ES/SNES because:
1. DE doesn't get stuck in local minima as easily
2. Batch evaluation minimizes CPU<->GPU transfers
"""

import torch
import numpy as np
from typing import Tuple, List, Optional
from scipy.optimize import differential_evolution

from .base import BaseBlockOptimizer
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
        **kwargs
    ) -> Tuple[float, float, np.ndarray, float]:
        """
        Optimize using scipy Differential Evolution with GPU batch evaluation.
        """
        n_seg = len(segment_indices)

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

        # Track best result
        best_result = {'loss': float('inf'), 'pearson': 0.0, 'angles': None}

        def fitness_vectorized(angles_population: np.ndarray) -> np.ndarray:
            """
            Vectorized fitness function for scipy DE.

            Args:
                angles_population: (popsize, n_seg) array of angle candidates

            Returns:
                (popsize,) array of fitness values (lower is better)
            """
            # Transfer angles to GPU (small data, fast)
            angles_tensor = torch.tensor(
                angles_population, device=self.device, dtype=GPU_DTYPE
            )

            # Compute loss in single GPU batch
            loss, pearson, _ = compute_loss_batch(
                angles_tensor, block_data, start_shift,
                trajectory_angle, self.angle_range, self.mse_weight
            )

            # Track best
            loss_np = loss.cpu().numpy()
            pearson_np = pearson.cpu().numpy()

            best_idx = np.argmin(loss_np)
            if loss_np[best_idx] < best_result['loss']:
                best_result['loss'] = loss_np[best_idx]
                best_result['pearson'] = pearson_np[best_idx]
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

        # Use tracked best (may be better than result.x due to early stopping)
        if best_result['angles'] is not None and best_result['loss'] < result.fun:
            best_angles = best_result['angles']
            best_pearson = best_result['pearson']
        else:
            best_angles = result.x
            # Recompute pearson for result.x
            angles_tensor = torch.tensor(
                best_angles.reshape(1, -1), device=self.device, dtype=GPU_DTYPE
            )
            _, pearson, _ = compute_loss_batch(
                angles_tensor, block_data, start_shift,
                trajectory_angle, self.angle_range, self.mse_weight
            )
            best_pearson = pearson[0].item()

        # Compute end shift
        shift_deltas = np.tan(np.deg2rad(best_angles)) * seg_md_lens
        best_end_shift = start_shift + shift_deltas.sum()

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
