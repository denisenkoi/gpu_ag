"""
LookbackBeam optimizer - continuous beam search with lookback for metrics.

Unlike GREEDY_BF which resets at block boundaries, this optimizer:
1. Maintains continuous beam of top candidates across entire well
2. Computes Pearson/MSE with lookback window (not full history)
3. Uses hybrid selection: 50% by score + 50% by STD

"Spermatozoid race" - 100 candidates compete for the best path.
"""

import time
import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .base import BaseBlockOptimizer, OptimizeResult
from . import register_optimizer, prepare_block_data, compute_score_batch, compute_std_batch, compute_prefix_std, BeamPrefix, GPU_DTYPE


@dataclass
class BeamCandidate:
    """One candidate in the beam ('spermatozoid')."""
    prefix: BeamPrefix      # accumulated state (synthetic, zone_gr, tvt, end_shift)
    angles: np.ndarray      # all angles from start
    score: float            # current score (for selection)


@register_optimizer('LOOKBACK_BEAM')
class LookbackBeamOptimizer(BaseBlockOptimizer):
    """
    Continuous beam search with lookback for metrics.

    Parameters:
        stage_size: Segments to add per stage (default: 2)
        beam_width: Total candidates to keep (default: 100)
        pearson_lookback: Points from prefix for Pearson (default: 300)
        std_lookback: Points from prefix for STD (default: 600)
        score_ratio: Fraction selected by score vs STD (default: 0.5)
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
        score_ratio: float = 1.0,  # Only score selection by default (faster)
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
        Optimize using lookback beam search.
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
        print(f"  LookbackBeam: {n_seg} segments, stage={self.stage_size}, beam={self.beam_width}")
        print(f"    lookback: pearson={self.pearson_lookback}, std={self.std_lookback}")

        # Initialize beam with single empty candidate
        beam: List[BeamCandidate] = [
            BeamCandidate(
                prefix=BeamPrefix.empty(start_shift, self.device),
                angles=np.array([], dtype=np.float32),
                score=0.0,
            )
        ]

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

            n_combos_per_beam = grid_size ** stage_n
            total_combos = len(beam) * n_combos_per_beam

            print(f"    Stage {stage_num}: seg {current_seg}-{stage_end-1}, {len(beam)}x{n_combos_per_beam}={total_combos} combos")

            all_candidates: List[BeamCandidate] = []

            # Expand each beam candidate with all angle combinations
            for cand in beam:
                for chunk_start in range(0, n_combos_per_beam, self.chunk_size):
                    chunk_end = min(chunk_start + self.chunk_size, n_combos_per_beam)
                    chunk_n = chunk_end - chunk_start
                    n_evaluations += chunk_n

                    # Generate angle combinations for this chunk
                    indices = torch.arange(chunk_start, chunk_end, device=self.device, dtype=torch.long)
                    chunk_angles = torch.zeros((chunk_n, stage_n), device=self.device, dtype=GPU_DTYPE)

                    divisor = 1
                    for seg in reversed(range(stage_n)):
                        seg_indices = (indices // divisor) % grid_size
                        chunk_angles[:, seg] = grid_gpu[seg_indices]
                        divisor *= grid_size

                    # Compute scores with lookback
                    # No selfcorr in scoring - used only in hybrid selection
                    scores, pearsons, mse_norms, new_synthetic, new_tvt = compute_score_batch(
                        chunk_angles, block_data, cand.prefix,
                        trajectory_angle, self.angle_range, self.mse_weight,
                        0.0, 0.0,  # No selfcorr penalty in scoring
                        self.pearson_lookback, self.std_lookback
                    )

                    # Compute end shifts for new segments
                    stage_md_lens_t = torch.tensor(
                        seg_md_lens[current_seg:stage_end],
                        device=self.device, dtype=GPU_DTYPE
                    )
                    shift_deltas = torch.tan(torch.deg2rad(chunk_angles)) * stage_md_lens_t
                    new_end_shifts = cand.prefix.end_shift + shift_deltas.sum(dim=1)

                    # Create new candidates (keep tensors on GPU for next iteration)
                    scores_np = scores.cpu().numpy()
                    angles_np = chunk_angles.cpu().numpy()
                    new_end_shifts_np = new_end_shifts.cpu().numpy()

                    for i in range(chunk_n):
                        # Build new prefix by appending new data (stays on GPU)
                        new_prefix = BeamPrefix(
                            synthetic=torch.cat([cand.prefix.synthetic, new_synthetic[i]]),
                            zone_gr=torch.cat([cand.prefix.zone_gr, block_data.zone_gr]),
                            zone_gr_smooth=torch.cat([cand.prefix.zone_gr_smooth, block_data.zone_gr_smooth]),
                            tvt=torch.cat([cand.prefix.tvt, new_tvt[i]]),
                            end_shift=new_end_shifts_np[i],
                        )

                        all_candidates.append(BeamCandidate(
                            prefix=new_prefix,
                            angles=np.concatenate([cand.angles, angles_np[i]]),
                            score=scores_np[i],
                        ))

            # Hybrid selection: score_ratio by score, rest by STD diversity
            beam = self._hybrid_select(all_candidates, block_data)

            print(f"      Selected {len(beam)}, best score: {beam[0].score:.3f}")

            del block_data
            torch.cuda.empty_cache()

            current_seg = stage_end
            stage_num += 1

        # Final selection from beam
        print(f"    Final: {len(beam)} candidates")

        # Pick best by score
        best = beam[0]
        best_angles = best.angles
        best_end_shift = best.prefix.end_shift

        opt_ms = int((time.perf_counter() - t_start) * 1000)

        print(f"      Winner shift: {best_end_shift:.2f}")
        print(f"      First 5 angles: {[f'{a:.2f}' for a in best_angles[:5]]}")

        if return_result:
            return OptimizeResult(
                pearson=0.0,  # Not tracked separately
                mse=0.0,
                score=best.score,
                loss=-best.score,
                end_shift=best_end_shift,
                angles=best_angles,
                start_shift=start_shift,
                n_segments=n_seg,
                n_evaluations=n_evaluations,
                opt_ms=opt_ms,
                ref_angle=trajectory_angle,
            )

        return 0.0, best_end_shift, best_angles, start_shift

    def _hybrid_select(
        self,
        candidates: List[BeamCandidate],
        block_data,
    ) -> List[BeamCandidate]:
        """
        Hybrid selection: top by score + top by STD diversity.
        """
        # Sort by score descending
        candidates.sort(key=lambda x: -x.score)

        n_by_score = int(self.beam_width * self.score_ratio)
        n_by_std = self.beam_width - n_by_score

        # Take top by score
        selected = candidates[:n_by_score]
        selected_set = set(range(n_by_score))

        # For STD selection, compute STD for top 2*beam_width candidates
        n_for_std = min(2 * self.beam_width, len(candidates))
        std_candidates = candidates[:n_for_std]

        # Compute real STD(bin_means) for these candidates
        if n_by_std > 0 and len(std_candidates) > n_by_score:
            # Use compute_prefix_std for real STD calculation
            std_values = []
            for cand in std_candidates:
                std_val = compute_prefix_std(cand.prefix)
                std_values.append(std_val)

            # Sort by STD (lower is better) and pick top that aren't already selected
            std_order = np.argsort(std_values)
            for idx in std_order:
                if idx not in selected_set and len(selected) < self.beam_width:
                    selected.append(std_candidates[idx])
                    selected_set.add(idx)

        # Fill remaining from score-sorted if needed
        for i, cand in enumerate(candidates):
            if len(selected) >= self.beam_width:
                break
            if i not in selected_set:
                selected.append(cand)

        return selected[:self.beam_width]
