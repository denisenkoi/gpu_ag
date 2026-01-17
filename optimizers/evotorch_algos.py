"""
EvoTorch evolutionary algorithms: CMA-ES, SNES, XNES.

These run fully on GPU via EvoTorch library.
Historical results: RMSE ~10.53m (worse than BruteForce 2.99m).
"""

import torch
import numpy as np
from typing import Tuple, List, Optional

from .base import BaseBlockOptimizer
from . import register_optimizer, prepare_block_data, compute_loss_batch, GPU_DTYPE


class EvoTorchBaseOptimizer(BaseBlockOptimizer):
    """Base class for EvoTorch-based optimizers."""

    algorithm_class = None  # Override in subclasses

    def __init__(
        self,
        device: str = 'cuda',
        angle_range: float = 2.5,
        angle_step: float = 0.2,
        mse_weight: float = 5.0,
        chunk_size: Optional[int] = None,
        popsize: int = 100,
        maxiter: int = 50,
    ):
        super().__init__(device, angle_range, angle_step, mse_weight, chunk_size)
        self.popsize = popsize
        self.maxiter = maxiter

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
        Optimize using EvoTorch evolutionary algorithm.
        """
        from evotorch import Problem

        n_seg = len(segment_indices)

        # For 1D problems, fall back to bruteforce (CMA-ES doesn't work well)
        if n_seg <= 1:
            from .bruteforce import BruteForceOptimizer
            bf = BruteForceOptimizer(
                device=self.device,
                angle_range=self.angle_range,
                mse_weight=self.mse_weight,
            )
            return bf.optimize(
                segment_indices, start_shift, trajectory_angle,
                well_md, well_tvd, well_gr, type_tvd, type_gr
            )

        # Prepare block data
        block_data = prepare_block_data(
            segment_indices, well_md, well_tvd, well_gr,
            type_tvd, type_gr, self.device
        )

        seg_md_lens = np.array([well_md[e] - well_md[s] for s, e in segment_indices], dtype=np.float32)

        def objective_fn(angles_batch: torch.Tensor) -> torch.Tensor:
            """Objective for EvoTorch (angles relative to trajectory_angle)."""
            if angles_batch.dim() == 1:
                angles_batch = angles_batch.unsqueeze(0)

            # Convert relative angles to absolute
            abs_angles = angles_batch + trajectory_angle

            loss, _, _, _, _, _ = compute_loss_batch(
                abs_angles, block_data, start_shift,
                trajectory_angle, self.angle_range, self.mse_weight
            )

            return loss.squeeze() if angles_batch.shape[0] == 1 else loss

        # Define EvoTorch Problem
        class BlockOptProblem(Problem):
            def __init__(inner_self):
                super().__init__(
                    objective_sense='min',
                    solution_length=n_seg,
                    initial_bounds=(-self.angle_range, self.angle_range),
                    device=self.device,
                    dtype=GPU_DTYPE,
                )

            def _evaluate_batch(inner_self, solutions):
                x = solutions.values
                fitness = objective_fn(x)
                solutions.set_evals(fitness)

        problem = BlockOptProblem()

        # Create searcher
        center_init = torch.zeros(n_seg, device=self.device, dtype=GPU_DTYPE)
        searcher = self.algorithm_class(
            problem,
            popsize=self.popsize,
            stdev_init=self.angle_range,
            center_init=center_init
        )

        # Run optimization
        best_fun = float('inf')
        best_angles_relative = None

        for _ in range(self.maxiter):
            searcher.step()
            pop = searcher.population
            if pop is not None and len(pop) > 0:
                best_idx = pop.evals.argmin()
                current_fun = pop.evals[best_idx].item()
                if current_fun < best_fun:
                    best_fun = current_fun
                    best_angles_relative = pop.values[best_idx].cpu().numpy()

        if best_angles_relative is None:
            # Fallback to center
            best_angles_relative = np.zeros(n_seg, dtype=np.float32)

        # Convert to absolute angles
        best_angles = best_angles_relative + trajectory_angle

        # Recompute pearson
        angles_tensor = torch.tensor(
            best_angles.reshape(1, -1), device=self.device, dtype=GPU_DTYPE
        )
        _, pearson, _, _, _, _ = compute_loss_batch(
            angles_tensor, block_data, start_shift,
            trajectory_angle, self.angle_range, self.mse_weight
        )
        best_pearson = pearson[0].item()

        # Compute end shift
        shift_deltas = np.tan(np.deg2rad(best_angles)) * seg_md_lens
        best_end_shift = start_shift + shift_deltas.sum()

        return best_pearson, best_end_shift, best_angles.astype(np.float32), start_shift


@register_optimizer('CMAES')
class CMAESOptimizer(EvoTorchBaseOptimizer):
    """CMA-ES optimizer via EvoTorch."""

    @property
    def algorithm_class(self):
        from evotorch.algorithms import CMAES
        return CMAES


@register_optimizer('SNES')
class SNESOptimizer(EvoTorchBaseOptimizer):
    """SNES optimizer via EvoTorch."""

    @property
    def algorithm_class(self):
        from evotorch.algorithms import SNES
        return SNES


@register_optimizer('XNES')
class XNESOptimizer(EvoTorchBaseOptimizer):
    """XNES optimizer via EvoTorch."""

    @property
    def algorithm_class(self):
        from evotorch.algorithms import XNES
        return XNES
