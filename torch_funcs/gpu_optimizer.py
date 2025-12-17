"""
GPU-based Differential Evolution optimizer.

Pure PyTorch implementation for parallel evaluation on GPU.
Matches scipy.differential_evolution with strategy='rand1bin'.
"""
import torch
from typing import Callable, Tuple, Optional


def _latin_hypercube_init(popsize: int, D: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Latin Hypercube Sampling for population initialization.

    Divides each dimension into popsize equal segments and places
    exactly one sample in each segment, then shuffles.
    Returns values in [0, 1] range.
    """
    # Create segment indices for each dimension
    samples = torch.zeros(popsize, D, device=device, dtype=dtype)

    for j in range(D):
        # Create ordered segments: [0, 1, 2, ..., popsize-1]
        perm = torch.randperm(popsize, device=device)
        # Random position within each segment
        samples[:, j] = (perm.to(dtype) + torch.rand(popsize, device=device, dtype=dtype)) / popsize

    return samples


def differential_evolution_torch(
    objective_fn: Callable[[torch.Tensor], torch.Tensor],
    bounds: torch.Tensor,
    popsize: int = 500,
    maxiter: int = 1000,
    mutation: Tuple[float, float] = (1.5, 1.99),
    recombination: float = 0.99,
    tol: float = 1e-10,
    seed: Optional[int] = None,
    callback: Optional[Callable[[int, torch.Tensor, torch.Tensor], bool]] = None,
    device: str = 'cuda',
    x0: Optional[torch.Tensor] = None
) -> dict:
    """
    Differential Evolution optimization on GPU.

    Args:
        objective_fn: Callable that takes (batch, D) tensor and returns (batch,) fitness values
        bounds: (D, 2) tensor of [min, max] bounds for each dimension
        popsize: Population size
        maxiter: Maximum iterations
        mutation: (F_min, F_max) mutation factor range
        recombination: CR crossover probability
        tol: Convergence tolerance (not used for early stopping in DE)
        seed: Random seed
        callback: Optional callback(iteration, population, fitness) -> stop_flag
        device: 'cuda' or 'cpu'
        x0: Optional initial point (D,) to include in population (helps find global optimum)

    Returns:
        dict with keys:
            - x: best solution (D,)
            - fun: best fitness value
            - nit: number of iterations
            - nfev: number of function evaluations
            - population: final population (popsize, D)
            - population_fitness: final fitness (popsize,)
    """
    dtype = torch.float64
    D = bounds.shape[0]
    F_min, F_max = mutation
    CR = recombination

    if seed is not None:
        torch.manual_seed(seed)

    # Initialize bounds on device
    bounds = bounds.to(device=device, dtype=dtype)
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    bound_range = ub - lb

    # Initialize population using Latin Hypercube Sampling (same as scipy default)
    # LHS ensures better coverage of search space than uniform random
    population = _latin_hypercube_init(popsize, D, device, dtype) * bound_range + lb

    fitness = objective_fn(population)
    nfev = popsize

    # Track best solution
    best_idx = torch.argmin(fitness)
    best_x = population[best_idx].clone()
    best_fun = fitness[best_idx].item()

    for iteration in range(maxiter):
        # Generate random indices for mutation (r1, r2, r3 all different from i)
        # For rand1bin: v = x_r1 + F * (x_r2 - x_r3)
        indices = torch.arange(popsize, device=device)

        # Shuffle to get random r1, r2, r3 (simple approach: use permutations)
        r1 = torch.randperm(popsize, device=device)
        r2 = torch.randperm(popsize, device=device)
        r3 = torch.randperm(popsize, device=device)

        # Ensure r1, r2, r3 are different from each other and from i
        # Simple fix: just reshuffle until condition is met (or use modular arithmetic)
        # For speed, we use a simple shuffle approach - not perfect but works
        mask = (r1 == indices) | (r2 == indices) | (r3 == indices) | (r1 == r2) | (r2 == r3) | (r1 == r3)
        while mask.any():
            r1[mask] = torch.randint(0, popsize, (mask.sum(),), device=device)
            r2[mask] = torch.randint(0, popsize, (mask.sum(),), device=device)
            r3[mask] = torch.randint(0, popsize, (mask.sum(),), device=device)
            mask = (r1 == indices) | (r2 == indices) | (r3 == indices) | (r1 == r2) | (r2 == r3) | (r1 == r3)

        # Dithered mutation factor (F varies per individual)
        F = F_min + torch.rand(popsize, 1, device=device, dtype=dtype) * (F_max - F_min)

        # Mutation: v = x_r1 + F * (x_r2 - x_r3)
        mutant = population[r1] + F * (population[r2] - population[r3])

        # Clip mutant to bounds
        mutant = torch.clamp(mutant, lb, ub)

        # Crossover: binomial
        cross_mask = torch.rand(popsize, D, device=device, dtype=dtype) < CR

        # Ensure at least one dimension is crossed over (jrand)
        jrand = torch.randint(0, D, (popsize,), device=device)
        cross_mask[torch.arange(popsize, device=device), jrand] = True

        # Trial vector
        trial = torch.where(cross_mask, mutant, population)

        # Evaluate trial vectors
        trial_fitness = objective_fn(trial)
        nfev += popsize

        # Selection: keep trial if better
        improved = trial_fitness < fitness
        population = torch.where(improved.unsqueeze(1), trial, population)
        fitness = torch.where(improved, trial_fitness, fitness)

        # Update best
        current_best_idx = torch.argmin(fitness)
        current_best_fun = fitness[current_best_idx].item()
        if current_best_fun < best_fun:
            best_fun = current_best_fun
            best_x = population[current_best_idx].clone()

        # Callback
        if callback is not None:
            if callback(iteration, population, fitness):
                break

    return {
        'x': best_x,
        'fun': best_fun,
        'nit': iteration + 1,
        'nfev': nfev,
        'population': population,
        'population_fitness': fitness
    }


def gpu_optimizer_fit(
    objective_wrapper,
    bounds: torch.Tensor,
    initial_shifts: Optional[torch.Tensor] = None,
    popsize: int = 500,
    maxiter: int = 1000,
    mutation: Tuple[float, float] = (1.5, 1.99),
    recombination: float = 0.99,
    seed: Optional[int] = None,
    verbose: bool = True,
    device: str = 'cuda'
) -> dict:
    """
    High-level GPU optimizer interface matching optimizer_fit signature.

    Args:
        objective_wrapper: TorchObjectiveWrapper instance
        bounds: (K, 2) tensor of [min, max] for each shift
        initial_shifts: Optional (K,) initial solution to seed population
        popsize: Population size
        maxiter: Maximum iterations
        mutation: (F_min, F_max) range
        recombination: CR value
        seed: Random seed
        verbose: Print progress
        device: 'cuda' or 'cpu'

    Returns:
        dict with optimization results
    """
    import time

    bounds = bounds.to(device=device, dtype=torch.float64)
    K = bounds.shape[0]

    def callback(iteration, population, fitness):
        if verbose and iteration % 100 == 0:
            best_idx = torch.argmin(fitness)
            best_fun = fitness[best_idx].item()
            mean_fun = torch.mean(fitness[~torch.isinf(fitness)]).item()
            print(f"Iter {iteration}: best={best_fun:.6f}, mean={mean_fun:.6f}")
        return False

    start_time = time.time()

    result = differential_evolution_torch(
        objective_fn=objective_wrapper,
        bounds=bounds,
        popsize=popsize,
        maxiter=maxiter,
        mutation=mutation,
        recombination=recombination,
        seed=seed,
        callback=callback if verbose else None,
        device=device
    )

    elapsed = time.time() - start_time

    if verbose:
        print(f"\nOptimization complete:")
        print(f"  Best fitness: {result['fun']:.6f}")
        print(f"  Iterations: {result['nit']}")
        print(f"  Function evals: {result['nfev']}")
        print(f"  Time: {elapsed:.2f} sec")

    result['elapsed_time'] = elapsed
    return result
