"""
ContinuousBeam optimizer - beam search across entire well.

Unlike GREEDY_BF which resets beam at block boundaries,
this optimizer maintains continuous beam from first to last segment.

Hybrid selection: keeps both best-by-score AND best-by-STD candidates
to maintain diversity and avoid local minima.
"""

import time
import torch
import numpy as np
from typing import Tuple, List, Optional, Union
from dataclasses import dataclass

from .base import BaseBlockOptimizer, OptimizeResult
from . import register_optimizer, prepare_block_data, compute_score_batch, compute_std_batch, BeamPrefix, GPU_DTYPE


@dataclass
class FullWellResult:
    """Result of full-well optimization."""
    end_error: float
    end_shift: float
    angles: np.ndarray
    pearson_mean: float
    opt_ms: int
    n_evaluations: int
    n_segments: int


@register_optimizer('CONTINUOUS_BEAM')
class ContinuousBeamOptimizer:
    """
    Continuous beam search across entire well.

    Parameters:
        stage_size: Segments to add per stage (default: 2)
        beam_width: Total candidates to keep (default: 100)
        score_ratio: Fraction selected by score vs STD (default: 0.5 = 50/50)
        angle_range: Search range around trajectory (default: 2.5)
        angle_step: Grid step (default: 0.1)
        mse_weight: Weight for MSE in score (default: 5.0)
    """

    def __init__(
        self,
        device: str = 'cuda',
        angle_range: float = 2.5,
        angle_step: float = 0.1,
        mse_weight: float = 5.0,
        stage_size: int = 2,
        beam_width: int = 100,
        score_ratio: float = 0.5,
    ):
        self.device = device
        self.angle_range = angle_range
        self.angle_step = angle_step
        self.mse_weight = mse_weight
        self.stage_size = stage_size
        self.beam_width = beam_width
        self.score_ratio = score_ratio

    def optimize_full_well(
        self,
        segment_indices: List[Tuple[int, int]],
        start_shift: float,
        trajectory_angle: float,
        well_md: np.ndarray,
        well_tvd: np.ndarray,
        well_gr: np.ndarray,
        type_tvd: np.ndarray,
        type_gr: np.ndarray,
        ref_end_shift: float = 0.0,
    ) -> FullWellResult:
        """
        Optimize entire well using continuous beam search.

        Key optimization: only compute score for CURRENT stage segments,
        not all accumulated segments. Previous angles are fixed, we just
        track cumulative shift.

        Returns FullWellResult with final angles and metrics.
        """
        t_start = time.perf_counter()
        n_seg = len(segment_indices)

        # Segment MD lengths
        seg_md_lens = np.array([well_md[e] - well_md[s] for s, e in segment_indices], dtype=np.float32)

        # Generate angle grid
        n_steps = int(2 * self.angle_range / self.angle_step) + 1
        angle_grid = np.linspace(
            trajectory_angle - self.angle_range,
            trajectory_angle + self.angle_range,
            n_steps
        )
        grid_gpu = torch.tensor(angle_grid, device=self.device, dtype=GPU_DTYPE)
        grid_size = len(angle_grid)

        n_evaluations = 0
        print(f"  ContinuousBeam: {n_seg} segments, stage={self.stage_size}, beam={self.beam_width}, grid={grid_size}")

        # Beam state: list of (angles_so_far, current_shift)
        # angles_so_far is numpy array, current_shift is float
        beam = [(np.array([], dtype=np.float32), start_shift)]

        current_seg = 0
        stage_num = 0

        while current_seg < n_seg:
            stage_end = min(current_seg + self.stage_size, n_seg)
            stage_n = stage_end - current_seg

            # Prepare block data for ONLY current stage segments
            block_data = prepare_block_data(
                segment_indices[current_seg:stage_end], well_md, well_tvd, well_gr,
                type_tvd, type_gr, self.device
            )

            n_combos_per_beam = grid_size ** stage_n
            total_combos = len(beam) * n_combos_per_beam

            print(f"    Stage {stage_num}: seg {current_seg}-{stage_end-1}, {len(beam)}x{n_combos_per_beam}={total_combos} combos")

            all_candidates = []  # (full_angles, new_shift, score)

            chunk_size = 200000
            for prev_angles, prev_shift in beam:
                # Generate all combinations for new segments
                for chunk_start in range(0, n_combos_per_beam, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_combos_per_beam)
                    chunk_n = chunk_end - chunk_start
                    n_evaluations += chunk_n

                    indices = torch.arange(chunk_start, chunk_end, device=self.device, dtype=torch.long)
                    chunk_angles = torch.zeros((chunk_n, stage_n), device=self.device, dtype=GPU_DTYPE)

                    divisor = 1
                    for seg in reversed(range(stage_n)):
                        seg_indices = (indices // divisor) % grid_size
                        chunk_angles[:, seg] = grid_gpu[seg_indices]
                        divisor *= grid_size

                    # Compute scores for ONLY new segments, starting from prev_shift
                    scores, pearsons, _, _, _, _ = compute_score_batch(
                        chunk_angles, block_data, prev_shift,
                        trajectory_angle, self.angle_range, self.mse_weight,
                        0.0, 0.0
                    )

                    # Compute new shifts for each candidate
                    stage_md_lens = seg_md_lens[current_seg:stage_end]
                    shift_deltas = torch.tan(torch.deg2rad(chunk_angles)) * torch.tensor(stage_md_lens, device=self.device)
                    new_shifts = prev_shift + shift_deltas.sum(dim=1)

                    # Store candidates
                    scores_np = scores.cpu().numpy()
                    angles_np = chunk_angles.cpu().numpy()
                    shifts_np = new_shifts.cpu().numpy()

                    for i in range(chunk_n):
                        full_angles = np.concatenate([prev_angles, angles_np[i]])
                        all_candidates.append((full_angles, shifts_np[i], scores_np[i]))

            # Sort by score and select top beam_width
            all_candidates.sort(key=lambda x: -x[2])  # Descending by score

            # Hybrid selection: half by score, half by STD diversity
            n_by_score = self.beam_width // 2
            selected = all_candidates[:n_by_score]

            # For remaining slots, try to pick diverse candidates
            # (those with different angle patterns)
            remaining = all_candidates[n_by_score:]
            for cand in remaining:
                if len(selected) >= self.beam_width:
                    break
                # Simple diversity check: different from already selected
                is_diverse = True
                for sel in selected[:n_by_score]:  # Compare to score-selected
                    if np.abs(cand[0] - sel[0]).mean() < 0.3:  # Too similar
                        is_diverse = False
                        break
                if is_diverse:
                    selected.append(cand)

            # Fill remaining if needed
            while len(selected) < self.beam_width and len(remaining) > 0:
                selected.append(remaining.pop(0))

            beam = [(s[0], s[1]) for s in selected[:self.beam_width]]
            print(f"      Selected {len(beam)}, best score: {all_candidates[0][2]:.3f}")

            del block_data
            torch.cuda.empty_cache()

            current_seg = stage_end
            stage_num += 1

        # Final selection
        print(f"    Final: {len(beam)} candidates")

        # Pick best by score (or could do final STD check)
        best_angles, best_shift = beam[0]
        end_error = best_shift - ref_end_shift

        opt_ms = int((time.perf_counter() - t_start) * 1000)

        print(f"      Winner shift: {best_shift:.2f}, error: {end_error:.2f}")
        print(f"      First 5 angles: {[f'{a:.2f}' for a in best_angles[:5]]}")

        return FullWellResult(
            end_error=end_error,
            end_shift=best_shift,
            angles=best_angles,
            pearson_mean=0.0,  # Not computed in this version
            opt_ms=opt_ms,
            n_evaluations=n_evaluations,
            n_segments=n_seg,
        )

    def _hybrid_select(
        self,
        angles: torch.Tensor,
        scores: torch.Tensor,
        block_data,
        start_shift: float,
        k: int,
    ) -> List[torch.Tensor]:
        """
        Hybrid selection: top k/2 by score + top k/2 by STD.
        Deduplicates to avoid keeping same candidate twice.
        """
        n_by_score = int(k * self.score_ratio)
        n_by_std = k - n_by_score

        # Top by score
        score_topk_idx = torch.topk(scores, min(n_by_score * 2, len(scores)))[1]

        # Compute STD for all candidates
        std_values = compute_std_batch(angles, block_data, start_shift)

        # Top by STD (lowest)
        valid_std = std_values.clone()
        valid_std[torch.isnan(valid_std)] = float('inf')
        std_topk_idx = torch.topk(-valid_std, min(n_by_std * 2, len(scores)))[1]  # negative for ascending

        # Combine with deduplication
        selected = []
        selected_set = set()

        # First add by score
        for idx in score_topk_idx:
            idx_val = idx.item()
            if idx_val not in selected_set and len(selected) < n_by_score:
                selected.append(angles[idx].clone())
                selected_set.add(idx_val)

        # Then add by STD
        for idx in std_topk_idx:
            idx_val = idx.item()
            if idx_val not in selected_set and len(selected) < k:
                selected.append(angles[idx].clone())
                selected_set.add(idx_val)

        # Fill remaining if needed
        if len(selected) < k:
            for idx in score_topk_idx:
                idx_val = idx.item()
                if idx_val not in selected_set and len(selected) < k:
                    selected.append(angles[idx].clone())
                    selected_set.add(idx_val)

        return selected
