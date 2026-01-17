"""
LookbackBeam optimizer V2 - vectorized version.

Key optimizations:
1. Tensor storage instead of List[BeamCandidate]
2. Batched operations instead of Python loops
3. Minimal object creation

Expected speedup: 10-20x
"""

import os
import time
import torch
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass

from .base import BaseBlockOptimizer, OptimizeResult
from . import register_optimizer, prepare_block_data, compute_score_batch, compute_std_batch, compute_prefix_std, BeamPrefix, GPU_DTYPE


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
        first_block_size: int = 4,  # First stage uses more segments (no prefix yet)
        beam_width: int = 100,
        pearson_lookback: int = 300,
        std_lookback: int = 600,
        score_ratio: float = 1.0,
    ):
        super().__init__(device, angle_range, angle_step, mse_weight, chunk_size)
        self.stage_size = int(os.environ.get('LOOKBACK_STAGE_SIZE', stage_size))
        self.first_block_size = int(os.environ.get('LOOKBACK_FIRST_BLOCK_SIZE', first_block_size))
        self.beam_width = int(os.environ.get('LOOKBACK_BEAM_WIDTH', beam_width))
        self.pearson_lookback = int(os.environ.get('LOOKBACK_PEARSON', pearson_lookback))
        self.std_lookback = int(os.environ.get('LOOKBACK_STD', std_lookback))
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
        beam_sse = np.zeros(1, dtype=np.float32)  # accumulated SSE for incremental MSE
        beam_n_points = np.zeros(1, dtype=np.int32)  # accumulated n_points

        current_seg = 0
        stage_num = 0

        while current_seg < n_seg:
            # First stage uses first_block_size, subsequent stages use stage_size
            current_stage_size = self.first_block_size if stage_num == 0 else self.stage_size
            stage_end = min(current_seg + current_stage_size, n_seg)
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
            all_pearsons_list = []
            all_mse_list = []
            all_angles_list = []
            all_end_shifts_list = []
            all_new_synthetic_list = []
            all_new_tvt_list = []
            all_new_sse_list = []
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
                    sse=float(beam_sse[cand_idx]),
                    n_points=int(beam_n_points[cand_idx]),
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
                scores, pearsons, mse_norms, new_synthetic, new_tvt, new_sse = compute_score_batch(
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
                all_pearsons_list.append(pearsons)
                all_mse_list.append(mse_norms)
                all_angles_list.append(chunk_angles)
                all_end_shifts_list.append(new_end_shifts)
                all_new_synthetic_list.append(new_synthetic)
                all_new_tvt_list.append(new_tvt)
                all_new_sse_list.append(new_sse)
                all_parent_idx_list.append(torch.full((n_combos,), cand_idx, device=self.device, dtype=torch.long))

            # Concatenate all candidates
            all_scores = torch.cat(all_scores_list)  # (total_combos,)
            all_pearsons = torch.cat(all_pearsons_list)  # (total_combos,)
            all_mse = torch.cat(all_mse_list)  # (total_combos,)
            all_stage_angles = torch.cat(all_angles_list)  # (total_combos, stage_n)
            all_end_shifts = torch.cat(all_end_shifts_list)  # (total_combos,)
            all_new_synthetic = torch.cat(all_new_synthetic_list)  # (total_combos, n_points)
            all_new_tvt = torch.cat(all_new_tvt_list)  # (total_combos, n_points)
            all_new_sse = torch.cat(all_new_sse_list)  # (total_combos,)
            all_parent_idx = torch.cat(all_parent_idx_list)  # (total_combos,)

            # Select top candidates by score
            n_select = min(self.beam_width, len(all_scores))
            top_values, top_indices = torch.topk(all_scores, n_select)

            # DEBUG: Every stage analysis
            print(f"      [BEAM Stage {stage_num}] segs {current_seg}-{stage_end-1}, n_combos={len(all_scores)}", flush=True)
            top5_idx = top_indices[:5].cpu().numpy()
            top5_scores = all_scores[top5_idx].cpu().numpy()
            top5_shifts = all_end_shifts[top5_idx].cpu().numpy()
            top5_angles = all_stage_angles[top5_idx].cpu().numpy()
            for i in range(min(5, len(top5_idx))):
                angles_str = ','.join(f'{a:.2f}' for a in top5_angles[i])
                print(f"        Top{i+1}: [{angles_str}] score={top5_scores[i]:.3f}, end_shift={top5_shifts[i]:.2f}", flush=True)

            # DEBUG: Stage 0 analysis with selective STD
            if stage_num == 0:
                # BF angles for first 4 segments of Well498 (old 87_200 mode)
                bf_angles_ref = [1.31, 2.71, 3.31, 1.71]
                # NEW: REF from BF 5-seg winner (dls mode)
                ref_5seg = [2.31, 2.11, 1.51, 3.91]
                all_angles_np = all_stage_angles.cpu().numpy()
                all_scores_np = all_scores.cpu().numpy()

                # Find BF solution (old)
                bf_idx = None
                n_match = min(stage_n, len(bf_angles_ref))
                for i, angles in enumerate(all_angles_np):
                    match = all(abs(angles[j] - bf_angles_ref[j]) < 0.05 for j in range(n_match))
                    if match:
                        bf_idx = i
                        break

                # Find REF[5seg] solution (new)
                ref5_idx = None
                n_match5 = min(stage_n, len(ref_5seg))
                for i, angles in enumerate(all_angles_np):
                    match = all(abs(angles[j] - ref_5seg[j]) < 0.15 for j in range(n_match5))
                    if match:
                        ref5_idx = i
                        break

                # Compute STD only for selected candidates (BF + top-5 + ref5)
                selected_indices = list(top_indices[:5].cpu().numpy())
                if bf_idx is not None and bf_idx not in selected_indices:
                    selected_indices.append(bf_idx)
                if ref5_idx is not None and ref5_idx not in selected_indices:
                    selected_indices.append(ref5_idx)

                selected_angles = all_stage_angles[selected_indices]
                selected_std = compute_std_batch(selected_angles, block_data, prefix).cpu().numpy()

                if bf_idx is not None:
                    bf_pos = selected_indices.index(bf_idx)
                    bf_std = selected_std[bf_pos]
                    score_rank = (all_scores_np > all_scores_np[bf_idx]).sum() + 1
                    print(f"      DEBUG BF [{','.join(f'{a:.2f}' for a in all_angles_np[bf_idx])}]: score={all_scores_np[bf_idx]:.3f} (rank {score_rank}/{len(all_scores_np)}), STD={bf_std:.4f}")
                else:
                    print(f"      DEBUG BF not found in candidates")

                # Print REF[5seg] position
                if ref5_idx is not None:
                    ref5_pos = selected_indices.index(ref5_idx)
                    ref5_std = selected_std[ref5_pos]
                    ref5_score_rank = (all_scores_np > all_scores_np[ref5_idx]).sum() + 1
                    print(f"      DEBUG REF[5seg] [{','.join(f'{a:.2f}' for a in all_angles_np[ref5_idx])}]: score={all_scores_np[ref5_idx]:.3f} (rank {ref5_score_rank}/{len(all_scores_np)}), STD={ref5_std:.4f}")
                else:
                    print(f"      DEBUG REF[5seg] {ref_5seg[:n_match5]} NOT in candidates")

                # Top-5 by score with STD
                print(f"      DEBUG Top-5 by score:")
                for i in range(min(5, len(top_indices))):
                    idx = top_indices[i].item()
                    std_val = selected_std[i]
                    print(f"        {i+1}: [{','.join(f'{a:.2f}' for a in all_angles_np[idx])}] score={all_scores_np[idx]:.3f}, STD={std_val:.4f}")

                # Angle distribution in top-100

            # DEBUG: Stage 1+ analysis - find REF continuation
            if stage_num >= 1:
                ref_5seg_full = [2.31, 2.11, 1.51, 3.91, -0.09]
                all_scores_np = all_scores.cpu().numpy()
                all_pearsons_np = all_pearsons.cpu().numpy()
                all_mse_np = all_mse.cpu().numpy()
                all_stage_angles_np = all_stage_angles.cpu().numpy()
                all_parent_idx_np = all_parent_idx.cpu().numpy()

                # Find parent that matches REF prefix
                ref_parent_idx = None
                for pidx in range(beam_size):
                    n_check = min(beam_angles.shape[1], len(ref_5seg_full))
                    if all(abs(beam_angles[pidx, j] - ref_5seg_full[j]) < 0.15 for j in range(n_check)):
                        ref_parent_idx = pidx
                        break

                if ref_parent_idx is not None:
                    # Find continuation with REF angles
                    ref_continuation_expected = ref_5seg_full[beam_angles.shape[1]:beam_angles.shape[1]+stage_n]
                    ref_cont_idx = None
                    for i in range(len(all_scores_np)):
                        if all_parent_idx_np[i] == ref_parent_idx:
                            if len(ref_continuation_expected) > 0:
                                match = all(abs(all_stage_angles_np[i, j] - ref_continuation_expected[j]) < 0.15
                                           for j in range(min(len(ref_continuation_expected), stage_n)))
                                if match:
                                    ref_cont_idx = i
                                    break

                    if ref_cont_idx is not None:
                        ref_score = all_scores_np[ref_cont_idx]
                        ref_pearson = all_pearsons_np[ref_cont_idx]
                        ref_mse = all_mse_np[ref_cont_idx]
                        ref_rank = int((all_scores_np > ref_score).sum()) + 1
                        ref_angles = all_stage_angles_np[ref_cont_idx]
                        # Also show top-1 for comparison
                        top1_idx = top_indices[0].item()
                        top1_pearson = all_pearsons_np[top1_idx]
                        top1_mse = all_mse_np[top1_idx]
                        print(f"      DEBUG REF continuation [{','.join(f'{a:.2f}' for a in ref_angles)}]: score={ref_score:.3f} P={ref_pearson:.3f} M={ref_mse:.3f} (rank {ref_rank}/{len(all_scores_np)})")
                        print(f"      DEBUG Top-1 comparison: P={top1_pearson:.3f} M={top1_mse:.3f}")
                    else:
                        print(f"      DEBUG REF continuation {ref_continuation_expected} NOT found for parent {ref_parent_idx}")
                else:
                    print(f"      DEBUG REF parent NOT in beam (was lost earlier)")
                top_angles_np = all_angles_np[top_indices.cpu().numpy()[:100]]
                ranges_str = ", ".join([f"seg{i}=[{top_angles_np[:,i].min():.1f},{top_angles_np[:,i].max():.1f}]" for i in range(stage_n)])
                print(f"      DEBUG Top-100 angle ranges: {ranges_str}")

            # Build new beam from top candidates
            top_indices_np = top_indices.cpu().numpy()
            top_parent_idx = all_parent_idx[top_indices].cpu().numpy()
            top_stage_angles = all_stage_angles[top_indices].cpu().numpy()
            top_end_shifts = all_end_shifts[top_indices].cpu().numpy()
            top_scores = top_values.cpu().numpy()
            top_new_synthetic = all_new_synthetic[top_indices]  # Keep on GPU
            top_new_tvt = all_new_tvt[top_indices]  # Keep on GPU
            top_new_sse = all_new_sse[top_indices].cpu().numpy()  # For incremental MSE

            # Build new beam arrays
            new_beam_angles = np.zeros((n_select, beam_angles.shape[1] + stage_n), dtype=np.float32)
            new_beam_synthetic = []
            new_beam_zone_gr = []
            new_beam_zone_gr_smooth = []
            new_beam_tvt = []
            new_beam_sse = np.zeros(n_select, dtype=np.float32)
            new_beam_n_points = np.zeros(n_select, dtype=np.int32)
            new_n_points = block_data.zone_gr.shape[0]

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
                # Accumulate SSE and n_points for incremental MSE
                new_beam_sse[i] = beam_sse[parent] + top_new_sse[i]
                new_beam_n_points[i] = beam_n_points[parent] + new_n_points

            # Update beam
            beam_size = n_select
            beam_angles = new_beam_angles
            beam_scores = top_scores
            beam_end_shifts = top_end_shifts
            beam_synthetic = new_beam_synthetic
            beam_zone_gr = new_beam_zone_gr
            beam_zone_gr_smooth = new_beam_zone_gr_smooth
            beam_tvt = new_beam_tvt
            beam_sse = new_beam_sse
            beam_n_points = new_beam_n_points

            print(f"      Selected {beam_size}, best score: {beam_scores[0]:.3f}")

            # Track REF[5seg] path in beam (for all stages)
            ref_5seg_full = [2.31, 2.11, 1.51, 3.91, -0.09]
            n_match_ref = min(beam_angles.shape[1], len(ref_5seg_full))
            ref_path_found = False
            for i in range(beam_size):
                if all(abs(beam_angles[i, j] - ref_5seg_full[j]) < 0.15 for j in range(n_match_ref)):
                    ref_path_found = True
                    print(f"      >>> REF[5seg] path FOUND at rank {i+1}/{beam_size}, angles={[f'{a:.2f}' for a in beam_angles[i, :n_match_ref]]}, score={beam_scores[i]:.3f}")
                    break
            if not ref_path_found:
                print(f"      >>> REF[5seg] path LOST after Stage {stage_num} (checked first {n_match_ref} angles)")

            del block_data, all_scores, all_pearsons, all_mse, all_stage_angles, all_end_shifts, all_new_synthetic, all_new_tvt, all_new_sse
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
        print(f"      Winner ALL angles: {[f'{a:.2f}' for a in best_angles]}")
        # Show top-5 final candidates
        print(f"      Top-5 final candidates (by score):")
        for i in range(min(5, beam_size)):
            angles_str = ','.join(f'{a:.2f}' for a in beam_angles[i][:6])
            print(f"        {i+1}: [{angles_str}...] shift={beam_end_shifts[i]:.2f}, score={beam_scores[i]:.3f}")

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
