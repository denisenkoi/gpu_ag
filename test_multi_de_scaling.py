"""
Test GPU DE scaling: how time grows with N parallel populations.
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
from ag_objects.ag_obj_interpretation import create_segments_from_json

from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
from torch_funcs.batch_objective import TorchObjectiveWrapper
from ag_numerical.ag_optimizer_utils import calculate_optimization_bounds


def load_data():
    base_dir = Path("/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")

    with open(base_dir / "AG_DATA/InitialData/slicing_well.json", 'r') as f:
        well_data = json.load(f)

    well = Well(well_data)
    typewell = TypeWell(well_data)

    with open(base_dir / "interpretation.json", 'r') as f:
        interp_data = json.load(f)

    segments_raw = interp_data["interpretation"]["segments"][-4:]
    segments = create_segments_from_json(segments_raw, well)

    return well, typewell, segments


def create_wrapper_and_bounds(well, typewell, segments, device='cuda'):
    bounds = calculate_optimization_bounds(segments, angle_range=10.0, accumulative=True)
    bounds_tensor = torch.tensor(bounds, dtype=torch.float64, device=device)

    well.calc_horizontal_projection(typewell, segments, 0.0)

    well_np = well_to_numpy(well)
    typewell_np = typewell_to_numpy(typewell)
    segments_np = segments_to_numpy(segments, well)

    well_torch = numpy_to_torch(well_np, device=device)
    typewell_torch = numpy_to_torch(typewell_np, device=device)
    segments_torch = segments_numpy_to_torch(segments_np, device=device)

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

    return wrapper, bounds_tensor


def run_multi_de(wrapper, bounds, n_populations=1, popsize_each=500, maxiter=500, seed=None):
    """
    Run N independent DE populations in parallel (island model).

    Each population evolves independently - mutation only within own population.
    All populations evaluated in single batch for GPU efficiency.
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

    # Initialize N populations with zeros+noise (best init from RND-777)
    # Each population is a contiguous block in the tensor
    population = torch.zeros(total_popsize, D, device=device, dtype=dtype)
    noise_scale = bound_range * 0.01
    population = population + torch.randn(total_popsize, D, device=device, dtype=dtype) * noise_scale
    population = torch.clamp(population, lb, ub)

    # Evaluate all populations in one batch
    fitness = wrapper(population)
    nfev = total_popsize

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
        # Process each population independently for mutation
        # But evaluate all in one batch

        trial = torch.zeros_like(population)

        for p in range(n_populations):
            start_idx = p * popsize_each
            end_idx = (p + 1) * popsize_each

            pop = population[start_idx:end_idx]

            # Generate random indices within this population only
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

        # Evaluate all trials in one batch
        trial_fitness = wrapper(trial)
        nfev += total_popsize

        # Selection for each population
        improved = trial_fitness < fitness
        population = torch.where(improved.unsqueeze(1), trial, population)
        fitness = torch.where(improved, trial_fitness, fitness)

        # Update best per population
        for p in range(n_populations):
            start_idx = p * popsize_each
            end_idx = (p + 1) * popsize_each
            pop_fitness = fitness[start_idx:end_idx]
            best_idx = torch.argmin(pop_fitness)
            current_best = pop_fitness[best_idx].item()
            if current_best < best_fun_per_pop[p]:
                best_fun_per_pop[p] = current_best
                best_x_per_pop[p] = population[start_idx + best_idx].clone()

    # Find overall best
    overall_best_idx = np.argmin(best_fun_per_pop)

    return {
        'x': best_x_per_pop[overall_best_idx],
        'fun': best_fun_per_pop[overall_best_idx],
        'all_best_fun': best_fun_per_pop,
        'nit': maxiter,
        'nfev': nfev,
        'n_populations': n_populations
    }


def main():
    print("="*60)
    print("GPU DE Scaling Test: N parallel populations")
    print("="*60)

    import io
    import contextlib

    # Load data once
    with contextlib.redirect_stdout(io.StringIO()):
        well, typewell, segments = load_data()
        wrapper, bounds = create_wrapper_and_bounds(well, typewell, segments, device='cuda')

    print(f"Segments: {len(segments)}, D={bounds.shape[0]}")
    print(f"Each population: 500 individuals, 500 iterations")
    print(f"Target: fun < 0.20")

    # Test different N values
    test_configs = [1, 2, 3, 5, 10, 15, 20]

    print(f"\n{'N pops':>8} {'Total pop':>10} {'Time (s)':>10} {'Best fun':>12} {'All bests':>40}")
    print("-" * 85)

    results = []

    for n_pop in test_configs:
        # Warmup for first run
        if n_pop == test_configs[0]:
            _ = run_multi_de(wrapper, bounds, n_populations=1, popsize_each=500, maxiter=10, seed=0)
            torch.cuda.synchronize()

        start_time = time.time()
        result = run_multi_de(wrapper, bounds, n_populations=n_pop, popsize_each=500, maxiter=500, seed=42)
        torch.cuda.synchronize()
        elapsed = time.time() - start_time

        all_bests_str = ', '.join([f'{b:.3f}' for b in result['all_best_fun']])
        if len(all_bests_str) > 38:
            all_bests_str = all_bests_str[:35] + '...'

        print(f"{n_pop:>8} {n_pop*500:>10} {elapsed:>10.2f} {result['fun']:>12.6f} {all_bests_str:>40}")

        results.append({
            'n_populations': n_pop,
            'total_popsize': n_pop * 500,
            'time_sec': elapsed,
            'best_fun': result['fun'],
            'all_best_fun': result['all_best_fun']
        })

    # Summary
    print(f"\n{'='*60}")
    print("SCALING ANALYSIS")
    print(f"{'='*60}")

    base_time = results[0]['time_sec']
    print(f"\nBase (N=1): {base_time:.2f} sec")
    print(f"\n{'N':>4} {'Time':>8} {'Ratio':>8} {'Linear would be':>16}")
    print("-" * 40)
    for r in results:
        ratio = r['time_sec'] / base_time
        linear = r['n_populations']
        print(f"{r['n_populations']:>4} {r['time_sec']:>8.2f} {ratio:>8.2f}x {linear:>16}x")


if __name__ == "__main__":
    main()
