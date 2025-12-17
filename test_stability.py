"""
Stability test: 20 runs with LHS init vs zeros init.
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
from torch_funcs.gpu_optimizer import differential_evolution_torch, _latin_hypercube_init
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


def run_gpu_de(wrapper, bounds, popsize=500, maxiter=500, seed=None, init_zeros=False):
    """Run GPU DE with optional zeros initialization."""
    device = str(bounds.device)
    dtype = torch.float64
    D = bounds.shape[0]

    if seed is not None:
        torch.manual_seed(seed)

    bounds = bounds.to(device=device, dtype=dtype)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    bound_range = ub - lb

    # Initialize population
    if init_zeros:
        # Initialize around zeros with small perturbation
        # (pure zeros would kill DE because all individuals are identical)
        center = torch.zeros(D, device=device, dtype=dtype)
        # Add small gaussian noise (1% of bound range)
        noise_scale = bound_range * 0.01
        population = center.unsqueeze(0) + torch.randn(popsize, D, device=device, dtype=dtype) * noise_scale
        # Clip to bounds
        population = torch.clamp(population, lb, ub)
    else:
        # Latin Hypercube Sampling (default)
        population = _latin_hypercube_init(popsize, D, device, dtype) * bound_range + lb

    fitness = wrapper(population)
    nfev = popsize

    best_idx = torch.argmin(fitness)
    best_x = population[best_idx].clone()
    best_fun = fitness[best_idx].item()

    F_min, F_max = 0.5, 1.0
    CR = 0.7

    for iteration in range(maxiter):
        indices = torch.arange(popsize, device=device)
        r1 = torch.randperm(popsize, device=device)
        r2 = torch.randperm(popsize, device=device)
        r3 = torch.randperm(popsize, device=device)

        mask = (r1 == indices) | (r2 == indices) | (r3 == indices) | (r1 == r2) | (r2 == r3) | (r1 == r3)
        while mask.any():
            r1[mask] = torch.randint(0, popsize, (mask.sum(),), device=device)
            r2[mask] = torch.randint(0, popsize, (mask.sum(),), device=device)
            r3[mask] = torch.randint(0, popsize, (mask.sum(),), device=device)
            mask = (r1 == indices) | (r2 == indices) | (r3 == indices) | (r1 == r2) | (r2 == r3) | (r1 == r3)

        F = F_min + torch.rand(popsize, 1, device=device, dtype=dtype) * (F_max - F_min)
        mutant = population[r1] + F * (population[r2] - population[r3])
        mutant = torch.clamp(mutant, lb, ub)

        cross_mask = torch.rand(popsize, D, device=device, dtype=dtype) < CR
        jrand = torch.randint(0, D, (popsize,), device=device)
        cross_mask[torch.arange(popsize, device=device), jrand] = True

        trial = torch.where(cross_mask, mutant, population)
        trial_fitness = wrapper(trial)
        nfev += popsize

        improved = trial_fitness < fitness
        population = torch.where(improved.unsqueeze(1), trial, population)
        fitness = torch.where(improved, trial_fitness, fitness)

        current_best_idx = torch.argmin(fitness)
        current_best_fun = fitness[current_best_idx].item()
        if current_best_fun < best_fun:
            best_fun = current_best_fun
            best_x = population[current_best_idx].clone()

    return {
        'x': best_x,
        'fun': best_fun,
        'nit': maxiter,
        'nfev': nfev
    }


def main():
    print("="*60)
    print("GPU DE Stability Test: 20 runs")
    print("="*60)

    # Suppress segment print messages
    import io
    import contextlib

    # Load data once
    with contextlib.redirect_stdout(io.StringIO()):
        well, typewell, segments = load_data()
        wrapper, bounds = create_wrapper_and_bounds(well, typewell, segments, device='cuda')

    print(f"Segments: {len(segments)}")
    print(f"Bounds: [{bounds[:, 0].min().item():.2f}, {bounds[:, 1].max().item():.2f}]")

    N_RUNS = 20
    TARGET = 0.20  # Считаем успехом если fun < 0.20

    # Test 1: LHS initialization (current approach)
    print(f"\n{'='*60}")
    print("Test 1: Latin Hypercube Sampling initialization")
    print(f"{'='*60}")

    lhs_results = []
    lhs_successes = 0
    start_time = time.time()

    for i in range(N_RUNS):
        result = run_gpu_de(wrapper, bounds, popsize=500, maxiter=500, seed=i*100, init_zeros=False)
        lhs_results.append(result['fun'])
        success = "✓" if result['fun'] < TARGET else "✗"
        if result['fun'] < TARGET:
            lhs_successes += 1
        print(f"  Run {i+1:2d}: fun={result['fun']:.6f} {success}")

    lhs_time = time.time() - start_time

    print(f"\nLHS Summary:")
    print(f"  Success rate: {lhs_successes}/{N_RUNS} ({100*lhs_successes/N_RUNS:.0f}%)")
    print(f"  Best: {min(lhs_results):.6f}")
    print(f"  Worst: {max(lhs_results):.6f}")
    print(f"  Mean: {np.mean(lhs_results):.6f}")
    print(f"  Std: {np.std(lhs_results):.6f}")
    print(f"  Total time: {lhs_time:.1f} sec ({lhs_time/N_RUNS:.2f} sec/run)")

    # Test 2: Zeros initialization
    print(f"\n{'='*60}")
    print("Test 2: Zeros initialization")
    print(f"{'='*60}")

    zeros_results = []
    zeros_successes = 0
    start_time = time.time()

    for i in range(N_RUNS):
        result = run_gpu_de(wrapper, bounds, popsize=500, maxiter=500, seed=i*100, init_zeros=True)
        zeros_results.append(result['fun'])
        success = "✓" if result['fun'] < TARGET else "✗"
        if result['fun'] < TARGET:
            zeros_successes += 1
        print(f"  Run {i+1:2d}: fun={result['fun']:.6f} {success}")

    zeros_time = time.time() - start_time

    print(f"\nZeros Summary:")
    print(f"  Success rate: {zeros_successes}/{N_RUNS} ({100*zeros_successes/N_RUNS:.0f}%)")
    print(f"  Best: {min(zeros_results):.6f}")
    print(f"  Worst: {max(zeros_results):.6f}")
    print(f"  Mean: {np.mean(zeros_results):.6f}")
    print(f"  Std: {np.std(zeros_results):.6f}")
    print(f"  Total time: {zeros_time:.1f} sec ({zeros_time/N_RUNS:.2f} sec/run)")

    # Test 3: LHS with larger popsize (1000) and maxiter (1000)
    print(f"\n{'='*60}")
    print("Test 3: LHS with popsize=1000, maxiter=1000")
    print(f"{'='*60}")

    lhs_big_results = []
    lhs_big_successes = 0
    start_time = time.time()

    for i in range(N_RUNS):
        result = run_gpu_de(wrapper, bounds, popsize=1000, maxiter=1000, seed=i*100, init_zeros=False)
        lhs_big_results.append(result['fun'])
        success = "✓" if result['fun'] < TARGET else "✗"
        if result['fun'] < TARGET:
            lhs_big_successes += 1
        print(f"  Run {i+1:2d}: fun={result['fun']:.6f} {success}")

    lhs_big_time = time.time() - start_time

    print(f"\nLHS (big) Summary:")
    print(f"  Success rate: {lhs_big_successes}/{N_RUNS} ({100*lhs_big_successes/N_RUNS:.0f}%)")
    print(f"  Best: {min(lhs_big_results):.6f}")
    print(f"  Worst: {max(lhs_big_results):.6f}")
    print(f"  Mean: {np.mean(lhs_big_results):.6f}")
    print(f"  Std: {np.std(lhs_big_results):.6f}")
    print(f"  Total time: {lhs_big_time:.1f} sec ({lhs_big_time/N_RUNS:.2f} sec/run)")

    # Final comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"{'Init Method':<20} {'Success':>10} {'Mean fun':>12} {'Std':>10} {'Time/run':>10}")
    print("-"*65)
    print(f"{'LHS (500x500)':<20} {f'{lhs_successes}/{N_RUNS}':>10} {np.mean(lhs_results):>12.6f} {np.std(lhs_results):>10.6f} {lhs_time/N_RUNS:>10.2f}s")
    print(f"{'Zeros+noise (500)':<20} {f'{zeros_successes}/{N_RUNS}':>10} {np.mean(zeros_results):>12.6f} {np.std(zeros_results):>10.6f} {zeros_time/N_RUNS:>10.2f}s")
    print(f"{'LHS (1000x1000)':<20} {f'{lhs_big_successes}/{N_RUNS}':>10} {np.mean(lhs_big_results):>12.6f} {np.std(lhs_big_results):>10.6f} {lhs_big_time/N_RUNS:>10.2f}s")


if __name__ == "__main__":
    main()
