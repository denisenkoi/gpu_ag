"""
GreedyBruteForce optimizer - hierarchical beam search.

Instead of optimizing 5 segments at once (51^5 = 345M combos),
we optimize in stages:
1. First N segments -> top-K candidates
2. Extend each candidate with next N segments -> K * 51^N combos
3. Select top-K again
4. Repeat until all segments processed

This allows exploring multiple "branches" in parallel,
reducing chance of getting stuck in local minima.
"""

import time
import torch
import numpy as np
from typing import Tuple, List, Optional, Union

from .base import BaseBlockOptimizer, OptimizeResult
from . import register_optimizer, prepare_block_data, compute_score_batch, compute_std_batch, BeamPrefix, GPU_DTYPE


@register_optimizer('GREEDY_BF')
class GreedyBruteForceOptimizer(BaseBlockOptimizer):
    """
    Hierarchical beam search optimizer.

    Parameters:
        stage_size: Number of segments to optimize per stage (default: 3)
        beam_width: Number of top candidates to keep between stages (default: 100)
        diversity_threshold: Min angle difference between kept candidates (default: 0.0)
        select_by_std: If True, select final winner by lowest STD (default: True)
    """

    def __init__(
        self,
        device: str = 'cuda',
        angle_range: float = 2.5,
        angle_step: float = 0.2,
        mse_weight: float = 5.0,
        chunk_size: Optional[int] = None,
        stage_size: int = 3,
        beam_width: int = 100,
        diversity_threshold: float = 0.0,
        select_by_std: bool = True,
    ):
        super().__init__(device, angle_range, angle_step, mse_weight, chunk_size)
        self.stage_size = stage_size
        self.beam_width = beam_width
        self.diversity_threshold = diversity_threshold
        self.select_by_std = select_by_std

        if self.chunk_size is None:
            self.chunk_size = 200000  # Can be larger since we process fewer combos

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
        Optimize using hierarchical beam search.
        """
        t_start = time.perf_counter()
        n_seg = len(segment_indices)

        # Prepare block data
        block_data = prepare_block_data(
            segment_indices, well_md, well_tvd, well_gr,
            type_tvd, type_gr, self.device
        )

        # Create empty prefix for first block
        prefix = BeamPrefix.empty(start_shift, self.device)

        # Segment MD lengths
        seg_md_lens = np.array([well_md[e] - well_md[s] for s, e in segment_indices], dtype=np.float32)

        # Generate angle grid (same for all segments for simplicity)
        n_steps = int(2 * self.angle_range / self.angle_step) + 1
        angle_grid = np.linspace(
            trajectory_angle - self.angle_range,
            trajectory_angle + self.angle_range,
            n_steps
        )
        grid_gpu = torch.tensor(angle_grid, device=self.device, dtype=GPU_DTYPE)
        grid_size = len(angle_grid)

        n_evaluations = 0

        # Stage 1: First stage_size segments
        stage_start = 0
        stage_end = min(self.stage_size, n_seg)
        stage_n = stage_end - stage_start

        # Generate all combinations for first stage
        n_combos = grid_size ** stage_n
        print(f"    Stage 1: segs {stage_start}-{stage_end-1}, {n_combos} combos")

        # Process in chunks
        all_scores = []
        all_angles = []

        for chunk_start in range(0, n_combos, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, n_combos)
            chunk_n = chunk_end - chunk_start
            n_evaluations += chunk_n

            # Generate angle combinations
            indices = torch.arange(chunk_start, chunk_end, device=self.device, dtype=torch.long)
            chunk_angles = torch.zeros((chunk_n, n_seg), device=self.device, dtype=GPU_DTYPE)
            chunk_angles[:, stage_end:] = trajectory_angle  # Fill rest with trajectory angle

            divisor = 1
            for seg in reversed(range(stage_start, stage_end)):
                seg_indices = (indices // divisor) % grid_size
                chunk_angles[:, seg] = grid_gpu[seg_indices]
                divisor *= grid_size

            # Compute scores
            scores, _, _, _, _, _ = compute_score_batch(
                chunk_angles, block_data, prefix,
                trajectory_angle, self.angle_range, self.mse_weight,
                0.0, 0.0
            )

            all_scores.append(scores)
            all_angles.append(chunk_angles)

        # Concatenate and select top-K
        all_scores = torch.cat(all_scores)
        all_angles = torch.cat(all_angles)

        beam_candidates = self._select_top_k(all_angles, all_scores, self.beam_width)
        print(f"      Selected {len(beam_candidates)} candidates, best score: {all_scores.max().item():.3f}")

        del all_scores, all_angles
        torch.cuda.empty_cache()

        # Subsequent stages
        stage_start = stage_end
        while stage_start < n_seg:
            stage_end = min(stage_start + self.stage_size, n_seg)
            stage_n = stage_end - stage_start

            n_combos_per_candidate = grid_size ** stage_n
            total_combos = len(beam_candidates) * n_combos_per_candidate
            print(f"    Stage: segs {stage_start}-{stage_end-1}, {len(beam_candidates)}x{n_combos_per_candidate}={total_combos} combos")

            all_scores = []
            all_angles = []

            # For each beam candidate, try all combinations of new segments
            for cand_idx, cand_angles in enumerate(beam_candidates):
                for chunk_start in range(0, n_combos_per_candidate, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, n_combos_per_candidate)
                    chunk_n = chunk_end - chunk_start
                    n_evaluations += chunk_n

                    # Generate angle combinations
                    indices = torch.arange(chunk_start, chunk_end, device=self.device, dtype=torch.long)
                    chunk_angles = cand_angles.unsqueeze(0).repeat(chunk_n, 1)

                    divisor = 1
                    for seg in reversed(range(stage_start, stage_end)):
                        seg_indices = (indices // divisor) % grid_size
                        chunk_angles[:, seg] = grid_gpu[seg_indices]
                        divisor *= grid_size

                    # Compute scores
                    scores, _, _, _, _, _ = compute_score_batch(
                        chunk_angles, block_data, prefix,
                        trajectory_angle, self.angle_range, self.mse_weight,
                        0.0, 0.0
                    )

                    all_scores.append(scores)
                    all_angles.append(chunk_angles)

            # Concatenate and select top-K
            all_scores = torch.cat(all_scores)
            all_angles = torch.cat(all_angles)

            beam_candidates = self._select_top_k(all_angles, all_scores, self.beam_width)
            print(f"      Selected {len(beam_candidates)} candidates, best score: {all_scores.max().item():.3f}")

            del all_scores, all_angles
            torch.cuda.empty_cache()

            stage_start = stage_end

        # Final selection from beam candidates
        print(f"    Final selection from {len(beam_candidates)} candidates")

        # Compute final scores and STD for all beam candidates
        final_angles = torch.stack(beam_candidates)
        final_scores, final_pearsons, final_mse, _, _, _ = compute_score_batch(
            final_angles, block_data, prefix,
            trajectory_angle, self.angle_range, self.mse_weight,
            0.0, 0.0
        )

        if self.select_by_std:
            final_std = compute_std_batch(final_angles, block_data, prefix)
            final_std_np = final_std.cpu().numpy()
            best_idx = np.nanargmin(final_std_np)
            print(f"      Selected by STD: {final_std_np[best_idx]:.2f} (range: {final_std_np.min():.2f}-{final_std_np.max():.2f})")
        else:
            best_idx = final_scores.argmax().item()

        best_angles = final_angles[best_idx].cpu().numpy()
        best_score = final_scores[best_idx].item()
        best_pearson = final_pearsons[best_idx].item()
        best_mse = final_mse[best_idx].item()

        # Compute end shift
        shift_deltas = np.tan(np.deg2rad(best_angles)) * seg_md_lens
        best_end_shift = start_shift + shift_deltas.sum()

        print(f"      Winner: Pearson={best_pearson:.3f}, MSE={best_mse:.3f}, Score={best_score:.3f}")
        print(f"      Angles: {[f'{a:.2f}' for a in best_angles[:5]]}...")

        if return_result:
            opt_ms = int((time.perf_counter() - t_start) * 1000)
            return OptimizeResult(
                pearson=best_pearson,
                mse=best_mse,
                score=best_score,
                loss=-best_score,
                end_shift=best_end_shift,
                angles=best_angles,
                start_shift=start_shift,
                n_segments=n_seg,
                n_evaluations=n_evaluations,
                opt_ms=opt_ms,
                ref_angle=trajectory_angle,
            )

        return best_pearson, best_end_shift, best_angles, start_shift

    def _select_top_k(
        self,
        angles: torch.Tensor,
        scores: torch.Tensor,
        k: int
    ) -> List[torch.Tensor]:
        """
        Select top-k candidates, optionally enforcing diversity.
        """
        # Simple top-k selection
        topk_vals, topk_idx = torch.topk(scores, min(k, len(scores)))

        if self.diversity_threshold <= 0:
            return [angles[idx].clone() for idx in topk_idx]

        # Diversity-aware selection
        selected = []
        for idx in topk_idx:
            candidate = angles[idx]

            # Check if sufficiently different from already selected
            is_diverse = True
            for sel in selected:
                diff = torch.abs(candidate - sel).mean().item()
                if diff < self.diversity_threshold:
                    is_diverse = False
                    break

            if is_diverse:
                selected.append(candidate.clone())
                if len(selected) >= k:
                    break

        return selected
