"""
Reliability test: 50 runs with N=10 parallel populations.
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

    # Initialize N populations with zeros+noise
    population = torch.zeros(total_popsize, D, device=device, dtype=dtype)
    noise_scale = bound_range * 0.01
    population = population + torch.randn(total_popsize, D, device=device, dtype=dtype) * noise_scale
    population = torch.clamp(population, lb, ub)

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
        nfev += total_popsize

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
        'all_best_fun': best_fun_per_pop,
        'nit': maxiter,
        'nfev': nfev,
        'n_populations': n_populations
    }


def main():
    print("="*70)
    print("GPU DE Reliability Test: 50 runs with N=10 parallel populations")
    print("="*70)

    import io
    import contextlib

    with contextlib.redirect_stdout(io.StringIO()):
        well, typewell, segments = load_data()
        wrapper, bounds = create_wrapper_and_bounds(well, typewell, segments, device='cuda')

    N_RUNS = 50
    N_POPULATIONS = 10
    POPSIZE_EACH = 500
    MAXITER = 500
    TARGET = 0.20

    print(f"Configuration:")
    print(f"  Runs: {N_RUNS}")
    print(f"  Populations per run: {N_POPULATIONS}")
    print(f"  Population size each: {POPSIZE_EACH}")
    print(f"  Total individuals per run: {N_POPULATIONS * POPSIZE_EACH}")
    print(f"  Iterations: {MAXITER}")
    print(f"  Target (success if fun <): {TARGET}")
    print()

    # Warmup
    _ = run_multi_de(wrapper, bounds, n_populations=1, popsize_each=100, maxiter=10, seed=0)
    torch.cuda.synchronize()

    results = []
    successes = 0
    total_time = 0

    print(f"{'Run':>4} {'Time':>8} {'Best fun':>12} {'Status':>10} {'Pops found global':>20}")
    print("-" * 60)

    for run in range(N_RUNS):
        start_time = time.time()
        result = run_multi_de(
            wrapper, bounds,
            n_populations=N_POPULATIONS,
            popsize_each=POPSIZE_EACH,
            maxiter=MAXITER,
            seed=run * 1000  # Different seed each run
        )
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        total_time += elapsed

        # Count how many populations found global minimum
        pops_found_global = sum(1 for f in result['all_best_fun'] if f < TARGET)

        success = result['fun'] < TARGET
        if success:
            successes += 1

        status = "OK" if success else "FAIL"
        print(f"{run+1:>4} {elapsed:>8.2f}s {result['fun']:>12.6f} {status:>10} {pops_found_global:>20}/{N_POPULATIONS}")

        results.append({
            'run': run + 1,
            'time': elapsed,
            'best_fun': result['fun'],
            'all_best_fun': result['all_best_fun'],
            'pops_found_global': pops_found_global
        })

    # Summary
    all_best_funs = [r['best_fun'] for r in results]
    all_times = [r['time'] for r in results]
    all_pops_found = [r['pops_found_global'] for r in results]

    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Success rate: {successes}/{N_RUNS} ({100*successes/N_RUNS:.1f}%)")
    print()
    print(f"Best fun:")
    print(f"  Min:  {min(all_best_funs):.6f}")
    print(f"  Max:  {max(all_best_funs):.6f}")
    print(f"  Mean: {np.mean(all_best_funs):.6f}")
    print(f"  Std:  {np.std(all_best_funs):.6f}")
    print()
    print(f"Time per run:")
    print(f"  Min:  {min(all_times):.2f}s")
    print(f"  Max:  {max(all_times):.2f}s")
    print(f"  Mean: {np.mean(all_times):.2f}s")
    print(f"  Total: {total_time:.1f}s")
    print()
    print(f"Populations finding global (per run):")
    print(f"  Min:  {min(all_pops_found)}/{N_POPULATIONS}")
    print(f"  Max:  {max(all_pops_found)}/{N_POPULATIONS}")
    print(f"  Mean: {np.mean(all_pops_found):.1f}/{N_POPULATIONS}")
    print()

    # Distribution of failures
    failures = [r for r in results if r['best_fun'] >= TARGET]
    if failures:
        print(f"Failed runs: {[r['run'] for r in failures]}")
        failed_vals = [round(r['best_fun'], 4) for r in failures]
        print(f"Failed best_fun values: {failed_vals}")
    else:
        print("No failures - 100% success rate!")


if __name__ == "__main__":
    main()
