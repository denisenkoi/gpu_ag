"""
Full well interpretation with GPU multi-population DE.

Simulates slicing process: starts from beginning and processes
the well in chunks, optimizing each chunk with N=10 parallel populations.
"""
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments, Segment

from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
from torch_funcs.batch_objective import TorchObjectiveWrapper
from ag_numerical.ag_optimizer_utils import calculate_optimization_bounds
from ag_rewards.ag_func_correlations import calculate_correlation


def load_well_data():
    """Load well data from JSON files."""
    base_dir = Path("/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")

    with open(base_dir / "AG_DATA/InitialData/slicing_well.json", 'r') as f:
        well_data = json.load(f)

    with open(base_dir / "interpretation.json", 'r') as f:
        interp_data = json.load(f)

    return well_data, interp_data


def run_multi_de(wrapper, bounds, n_populations=10, popsize_each=500, maxiter=500, seed=None):
    """
    Run N independent DE populations in parallel (island model).
    Returns best result from all populations.
    """
    device = str(bounds.device)
    dtype = torch.float64
    D = bounds.shape[0]
    total_popsize = n_populations * popsize_each

    if seed is not None:
        torch.manual_seed(seed)

    bounds = bounds.to(device=device, dtype=dtype)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    bound_range = ub - lb

    # Initialize with zeros+noise
    population = torch.zeros(total_popsize, D, device=device, dtype=dtype)
    noise_scale = bound_range * 0.01
    population = population + torch.randn(total_popsize, D, device=device, dtype=dtype) * noise_scale
    population = torch.clamp(population, lb, ub)

    fitness = wrapper(population)

    # Track best per population
    best_fun_per_pop = []
    best_x_per_pop = []
    for p in range(n_populations):
        start_idx = p * popsize_each
        end_idx = (p + 1) * popsize_each
        pop_fitness = fitness[start_idx:end_idx]
        best_idx = torch.argmin(pop_fitness)
        best_fun_per_pop.append(pop_fitness[best_idx].item())
        best_x_per_pop.append(population[start_idx + best_idx].clone())

    F_min, F_max = 0.5, 1.0
    CR = 0.7

    for iteration in range(maxiter):
        trial = torch.zeros_like(population)

        for p in range(n_populations):
            start_idx = p * popsize_each
            end_idx = (p + 1) * popsize_each
            pop = population[start_idx:end_idx]

            indices = torch.arange(popsize_each, device=device)
            r1 = torch.randperm(popsize_each, device=device)
            r2 = torch.randperm(popsize_each, device=device)
            r3 = torch.randperm(popsize_each, device=device)

            mask = (r1 == indices) | (r2 == indices) | (r3 == indices) | (r1 == r2) | (r2 == r3) | (r1 == r3)
            while mask.any():
                r1[mask] = torch.randint(0, popsize_each, (mask.sum(),), device=device)
                r2[mask] = torch.randint(0, popsize_each, (mask.sum(),), device=device)
                r3[mask] = torch.randint(0, popsize_each, (mask.sum(),), device=device)
                mask = (r1 == indices) | (r2 == indices) | (r3 == indices) | (r1 == r2) | (r2 == r3) | (r1 == r3)

            F = F_min + torch.rand(popsize_each, 1, device=device, dtype=dtype) * (F_max - F_min)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = torch.clamp(mutant, lb, ub)

            cross_mask = torch.rand(popsize_each, D, device=device, dtype=dtype) < CR
            jrand = torch.randint(0, D, (popsize_each,), device=device)
            cross_mask[torch.arange(popsize_each, device=device), jrand] = True

            trial[start_idx:end_idx] = torch.where(cross_mask, mutant, pop)

        trial_fitness = wrapper(trial)
        improved = trial_fitness < fitness
        population = torch.where(improved.unsqueeze(1), trial, population)
        fitness = torch.where(improved, trial_fitness, fitness)

        for p in range(n_populations):
            start_idx = p * popsize_each
            end_idx = (p + 1) * popsize_each
            pop_fitness = fitness[start_idx:end_idx]
            best_idx = torch.argmin(pop_fitness)
            current_best = pop_fitness[best_idx].item()
            if current_best < best_fun_per_pop[p]:
                best_fun_per_pop[p] = current_best
                best_x_per_pop[p] = population[start_idx + best_idx].clone()

    overall_best_idx = np.argmin(best_fun_per_pop)
    return {
        'x': best_x_per_pop[overall_best_idx],
        'fun': best_fun_per_pop[overall_best_idx],
        'all_best_fun': best_fun_per_pop
    }


def optimize_segments_gpu(well, typewell, segments, device='cuda', n_populations=10):
    """
    Optimize segments using GPU multi-population DE.
    Returns optimized segments with shifts.
    """
    # Calculate bounds
    bounds = calculate_optimization_bounds(segments, angle_range=10.0, accumulative=True)
    bounds_tensor = torch.tensor(bounds, dtype=torch.float64, device=device)

    # Initial projection
    well.calc_horizontal_projection(typewell, segments, 0.0)

    # Convert to numpy then torch
    well_np = well_to_numpy(well)
    typewell_np = typewell_to_numpy(typewell)
    segments_np = segments_to_numpy(segments, well)

    well_torch = numpy_to_torch(well_np, device=device)
    typewell_torch = numpy_to_torch(typewell_np, device=device)
    segments_torch = segments_numpy_to_torch(segments_np, device=device)

    # Create wrapper
    wrapper = TorchObjectiveWrapper(
        well_data=well_torch,
        typewell_data=typewell_torch,
        segments_torch=segments_torch,
        self_corr_start_idx=0,
        pearson_power=2.0,
        mse_power=0.001,
        num_intervals_self_correlation=0,
        sc_power=1.15,
        angle_range=10.0,
        angle_sum_power=2.0,
        min_pearson_value=-1.0,
        tvd_to_typewell_shift=0.0,
        device=device
    )

    # Run multi-population DE
    result = run_multi_de(
        wrapper, bounds_tensor,
        n_populations=n_populations,
        popsize_each=500,
        maxiter=500
    )

    # Update segments with optimal shifts
    optimal_shifts = result['x'].cpu().tolist()
    optimized_segments = deepcopy(segments)
    for i, shift in enumerate(optimal_shifts):
        optimized_segments[i].end_shift = shift
        if i < len(optimized_segments) - 1:
            optimized_segments[i + 1].start_shift = shift

    return optimized_segments, result['fun']


def main():
    print("="*70)
    print("Full Well Interpretation with GPU Multi-Population DE")
    print("="*70)

    import io
    import contextlib

    # Load data
    well_data, interp_data = load_well_data()

    # Get reference segments from interpretation
    reference_segments = interp_data['interpretation']['segments']
    print(f"Reference interpretation: {len(reference_segments)} segments")
    print(f"MD range: {reference_segments[0]['startMd']:.1f} - {reference_segments[-1].get('endMd', 'end')}")

    # Create Well and TypeWell objects
    well = Well(well_data)
    typewell = TypeWell(well_data)

    # Normalize
    max_curve = max(well.max_curve, typewell.value.max())
    md_range = well.max_md - well.min_md
    well.normalize(max_curve, typewell.min_depth, md_range)
    typewell.normalize(max_curve, well.min_depth, md_range)

    print(f"\nWell normalized: {len(well.measured_depth)} points")
    print(f"TypeWell: {len(typewell.tvd)} points")

    # Configuration for slicing
    SEGMENT_LENGTH_M = 30.0  # meters per segment
    SEGMENTS_PER_CHUNK = 4   # segments to optimize at once
    LOOKBACK_SEGMENTS = 2    # overlap with previous chunk

    step_size_idx = int(SEGMENT_LENGTH_M / well.horizontal_well_step)

    # Starting point (from reference interpretation)
    start_md_m = reference_segments[0]['startMd']
    start_idx = well.md2idx((start_md_m - well.min_md) / md_range)

    # End point
    end_idx = len(well.measured_depth) - 1

    print(f"\nSlicing configuration:")
    print(f"  Segment length: {SEGMENT_LENGTH_M}m ({step_size_idx} indices)")
    print(f"  Segments per chunk: {SEGMENTS_PER_CHUNK}")
    print(f"  Lookback: {LOOKBACK_SEGMENTS} segments")
    print(f"  Start idx: {start_idx}, End idx: {end_idx}")

    # Process well in chunks
    all_segments = []
    total_time = 0
    chunk_count = 0

    current_start_idx = start_idx
    current_shift = 0.0

    print(f"\n{'Chunk':>6} {'Start idx':>10} {'End idx':>10} {'Segments':>10} {'Time':>8} {'Best fun':>12}")
    print("-" * 70)

    while current_start_idx < end_idx - step_size_idx:
        chunk_count += 1

        # Determine chunk end
        chunk_end_idx = min(current_start_idx + SEGMENTS_PER_CHUNK * step_size_idx, end_idx)

        # Create segments for this chunk
        actual_length = chunk_end_idx - current_start_idx
        num_segments = min(SEGMENTS_PER_CHUNK, actual_length // step_size_idx)

        if num_segments < 1:
            break

        segments = create_segments(
            well=well,
            segments_count=num_segments,
            segment_len=step_size_idx,
            start_idx=current_start_idx,
            start_shift=current_shift
        )

        # Optimize this chunk
        start_time = time.time()
        optimized_segments, best_fun = optimize_segments_gpu(
            well, typewell, segments,
            device='cuda',
            n_populations=10
        )
        torch.cuda.synchronize()
        chunk_time = time.time() - start_time
        total_time += chunk_time

        print(f"{chunk_count:>6} {current_start_idx:>10} {chunk_end_idx:>10} {num_segments:>10} {chunk_time:>8.2f}s {best_fun:>12.6f}")

        # Save segments (excluding lookback overlap with next chunk)
        if chunk_count == 1:
            # First chunk - save all
            all_segments.extend(optimized_segments)
        else:
            # Subsequent chunks - skip lookback segments (already saved)
            all_segments.extend(optimized_segments[LOOKBACK_SEGMENTS:])

        # Prepare for next chunk
        # Move forward by (SEGMENTS_PER_CHUNK - LOOKBACK) segments
        advance_segments = SEGMENTS_PER_CHUNK - LOOKBACK_SEGMENTS
        current_start_idx = current_start_idx + advance_segments * step_size_idx

        # Get shift from end of lookback region for continuity
        if len(optimized_segments) > LOOKBACK_SEGMENTS:
            current_shift = optimized_segments[LOOKBACK_SEGMENTS - 1].end_shift
        else:
            current_shift = optimized_segments[-1].end_shift

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Total chunks processed: {chunk_count}")
    print(f"Total segments created: {len(all_segments)}")
    print(f"Total GPU time: {total_time:.2f}s")
    print(f"Average time per chunk: {total_time/chunk_count:.2f}s")

    # Calculate final correlation for full interpretation
    print(f"\n--- Final Interpretation Quality ---")

    # Use last 4 segments for final correlation check
    if len(all_segments) >= 4:
        final_segments = all_segments[-4:]
        well_copy = deepcopy(well)
        well_copy.calc_horizontal_projection(typewell, final_segments, 0.0)

        corr, _, self_corr, _, _, pearson, num_points, mse, _, _ = calculate_correlation(
            well_copy,
            0,  # self_corr_start_idx
            final_segments[0].start_idx,
            final_segments[-1].end_idx,
            float('inf'),
            0, 0,
            2.0, 0.001, 0, 1.15, -1.0
        )

        print(f"Final 4 segments correlation: {corr:.4f}")
        print(f"Pearson: {pearson:.4f}, MSE: {mse:.6f}")
        print(f"Points: {num_points}")

    # Show segment shifts
    print(f"\n--- Segment Shifts (denormalized to meters) ---")
    for i, seg in enumerate(all_segments[-10:]):  # Last 10 segments
        shift_m = seg.end_shift * md_range
        print(f"  Seg[{len(all_segments)-10+i}]: end_shift = {shift_m:.3f}m")


if __name__ == "__main__":
    main()
