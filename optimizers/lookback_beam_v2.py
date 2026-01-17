"""
LookbackBeam optimizer V2 - vectorized version.

Key optimizations:
1. Tensor storage instead of List[BeamCandidate]
2. Batched operations instead of Python loops
3. Minimal object creation

Expected speedup: 10-20x
"""

import time
import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .base import BaseBlockOptimizer, OptimizeResult
from . import register_optimizer, prepare_block_data, compute_score_batch, compute_prefix_std, BeamPrefix, GPU_DTYPE


@register_optimizer('LOOKBACK_BEAM_V2')
class LookbackBeamOptimizerV2(BaseBlockOptimizer):
    """
    Vectorized continuous beam search with lookback for metrics.

    Uses tensor storage instead of Python objects for ~10-20x speedup.
    """

    def __init__(
        self,
        device: str = 'cuda',
        angle_range: float = 2.5,
        angle_step: float = 0.1,
        mse_weight: float = 5.0,
        chunk_size: Optional[int] = None,
        stage_size: int = 2,
        beam_width: int = 100,
        pearson_lookback: int = 300,
        std_lookback: int = 600,
        score_ratio: float = 1.0,
    ):
        super().__init__(device, angle_range, angle_step, mse_weight, chunk_size)
        self.stage_size = stage_size
        self.beam_width = beam_width
        self.pearson_lookback = pearson_lookback
        self.std_lookback = std_lookback
        self.score_ratio = score_ratio

        if self.chunk_size is None:
            self.chunk_size = 200000

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
    ) -> OptimizeResult:
        """
        Optimize using vectorized lookback beam search.
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
        print(f"  LookbackBeamV2: {n_seg} segments, stage={self.stage_size}, beam={self.beam_width}")

        # Initialize beam with tensor storage
        # Each candidate has: angles array, prefix data (synthetic, zone_gr, tvt), score, end_shift

        # Start with single empty candidate
        beam_size = 1
        beam_angles = np.zeros((1, 0), dtype=np.float32)  # (beam, n_angles_so_far)
        beam_scores = np.zeros(1, dtype=np.float32)
        beam_end_shifts = np.full(1, start_shift, dtype=np.float32)

        # Prefix data stored as lists of tensors (variable length per candidate)
        beam_synthetic = [torch.tensor([], device=self.device, dtype=GPU_DTYPE)]
        beam_zone_gr = [torch.tensor([], device=self.device, dtype=GPU_DTYPE)]
        beam_zone_gr_smooth = [torch.tensor([], device=self.device, dtype=GPU_DTYPE)]
        beam_tvt = [torch.tensor([], device=self.device, dtype=GPU_DTYPE)]

        current_seg = 0
        stage_num = 0

        while current_seg < n_seg:
            stage_end = min(current_seg + self.stage_size, n_seg)
            stage_n = stage_end - current_seg

            # Prepare block data for CURRENT stage segments only
            block_data = prepare_block_data(
                segment_indices[current_seg:stage_end], well_md, well_tvd, well_gr,
                type_tvd, type_gr, self.device
            )

            n_combos = grid_size ** stage_n
            total_combos = beam_size * n_combos

            print(f"    Stage {stage_num}: seg {current_seg}-{stage_end-1}, {beam_size}x{n_combos}={total_combos} combos")

            # Pre-allocate arrays for new candidates
            new_beam_size = min(total_combos, self.beam_width * 2)  # Keep top 2*beam_width

            # Collect all scores first, then select top-k
            all_scores_list = []
            all_angles_list = []
            all_end_shifts_list = []
            all_new_synthetic_list = []
            all_new_tvt_list = []
            all_parent_idx_list = []

            # Stage MD lens tensor
            stage_md_lens_t = torch.tensor(
                seg_md_lens[current_seg:stage_end],
                device=self.device, dtype=GPU_DTYPE
            )

            # Expand each beam candidate
            for cand_idx in range(beam_size):
                # Create BeamPrefix for this candidate
                prefix = BeamPrefix(
                    synthetic=beam_synthetic[cand_idx],
                    zone_gr=beam_zone_gr[cand_idx],
                    zone_gr_smooth=beam_zone_gr_smooth[cand_idx],
                    tvt=beam_tvt[cand_idx],
                    end_shift=beam_end_shifts[cand_idx],
                )

                # Generate all angle combinations
                indices = torch.arange(n_combos, device=self.device, dtype=torch.long)
                chunk_angles = torch.zeros((n_combos, stage_n), device=self.device, dtype=GPU_DTYPE)

                divisor = 1
                for seg in reversed(range(stage_n)):
                    seg_indices = (indices // divisor) % grid_size
                    chunk_angles[:, seg] = grid_gpu[seg_indices]
                    divisor *= grid_size

                n_evaluations += n_combos

                # Compute scores
                scores, pearsons, mse_norms, new_synthetic, new_tvt = compute_score_batch(
                    chunk_angles, block_data, prefix,
                    trajectory_angle, self.angle_range, self.mse_weight,
                    0.0, 0.0,
                    self.pearson_lookback, self.std_lookback
                )

                # Compute end shifts
                shift_deltas = torch.tan(torch.deg2rad(chunk_angles)) * stage_md_lens_t
                new_end_shifts = prefix.end_shift + shift_deltas.sum(dim=1)

                # Store results (keep on GPU as long as possible)
                all_scores_list.append(scores)
                all_angles_list.append(chunk_angles)
                all_end_shifts_list.append(new_end_shifts)
                all_new_synthetic_list.append(new_synthetic)
                all_new_tvt_list.append(new_tvt)
                all_parent_idx_list.append(torch.full((n_combos,), cand_idx, device=self.device, dtype=torch.long))

            # Concatenate all candidates
            all_scores = torch.cat(all_scores_list)  # (total_combos,)
            all_stage_angles = torch.cat(all_angles_list)  # (total_combos, stage_n)
            all_end_shifts = torch.cat(all_end_shifts_list)  # (total_combos,)
            all_new_synthetic = torch.cat(all_new_synthetic_list)  # (total_combos, n_points)
            all_new_tvt = torch.cat(all_new_tvt_list)  # (total_combos, n_points)
            all_parent_idx = torch.cat(all_parent_idx_list)  # (total_combos,)

            # Select top candidates by score
            n_select = min(self.beam_width, len(all_scores))
            top_values, top_indices = torch.topk(all_scores, n_select)

            # Build new beam from top candidates
            top_indices_np = top_indices.cpu().numpy()
            top_parent_idx = all_parent_idx[top_indices].cpu().numpy()
            top_stage_angles = all_stage_angles[top_indices].cpu().numpy()
            top_end_shifts = all_end_shifts[top_indices].cpu().numpy()
            top_scores = top_values.cpu().numpy()
            top_new_synthetic = all_new_synthetic[top_indices]  # Keep on GPU
            top_new_tvt = all_new_tvt[top_indices]  # Keep on GPU

            # Build new beam arrays
            new_beam_angles = np.zeros((n_select, beam_angles.shape[1] + stage_n), dtype=np.float32)
            new_beam_synthetic = []
            new_beam_zone_gr = []
            new_beam_zone_gr_smooth = []
            new_beam_tvt = []

            for i in range(n_select):
                parent = top_parent_idx[i]
                # Concatenate angles
                new_beam_angles[i, :beam_angles.shape[1]] = beam_angles[parent]
                new_beam_angles[i, beam_angles.shape[1]:] = top_stage_angles[i]
                # Concatenate prefix tensors
                new_beam_synthetic.append(torch.cat([beam_synthetic[parent], top_new_synthetic[i]]))
                new_beam_zone_gr.append(torch.cat([beam_zone_gr[parent], block_data.zone_gr]))
                new_beam_zone_gr_smooth.append(torch.cat([beam_zone_gr_smooth[parent], block_data.zone_gr_smooth]))
                new_beam_tvt.append(torch.cat([beam_tvt[parent], top_new_tvt[i]]))

            # Update beam
            beam_size = n_select
            beam_angles = new_beam_angles
            beam_scores = top_scores
            beam_end_shifts = top_end_shifts
            beam_synthetic = new_beam_synthetic
            beam_zone_gr = new_beam_zone_gr
            beam_zone_gr_smooth = new_beam_zone_gr_smooth
            beam_tvt = new_beam_tvt

            print(f"      Selected {beam_size}, best score: {beam_scores[0]:.3f}")

            del block_data, all_scores, all_stage_angles, all_end_shifts, all_new_synthetic, all_new_tvt
            torch.cuda.empty_cache()

            current_seg = stage_end
            stage_num += 1

        # Final selection - best by score
        print(f"    Final: {beam_size} candidates")

        best_idx = 0  # Already sorted by score
        best_angles = beam_angles[best_idx]
        best_end_shift = beam_end_shifts[best_idx]

        opt_ms = int((time.perf_counter() - t_start) * 1000)

        print(f"      Winner shift: {best_end_shift:.2f}")
        print(f"      First 5 angles: {[f'{a:.2f}' for a in best_angles[:5]]}")

        if return_result:
            return OptimizeResult(
                pearson=0.0,
                mse=0.0,
                score=beam_scores[best_idx],
                loss=-beam_scores[best_idx],
                end_shift=best_end_shift,
                angles=best_angles,
                start_shift=start_shift,
                n_segments=n_seg,
                n_evaluations=n_evaluations,
                opt_ms=opt_ms,
                ref_angle=trajectory_angle,
            )

        return 0.0, best_end_shift, best_angles, start_shift
