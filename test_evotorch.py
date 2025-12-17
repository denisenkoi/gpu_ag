"""
Test EvoTorch optimization library with AG objective function.

Tests GeneticAlgorithm and other evolutionary algorithms from EvoTorch.
"""
import sys
import json
import time
import torch
import numpy as np
from pathlib import Path

# Add cpu_baseline to path
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json

# GPU imports
from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
from torch_funcs.batch_objective import TorchObjectiveWrapper
from ag_numerical.ag_optimizer_utils import calculate_optimization_bounds


def load_test_data():
    """Load test data from JSON files."""
    base_dir = Path("/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")

    with open(base_dir / "AG_DATA/InitialData/slicing_well.json", 'r') as f:
        well_data = json.load(f)

    well = Well(well_data)
    typewell = TypeWell(well_data)

    # Load interpretation segments
    with open(base_dir / "interpretation.json", 'r') as f:
        interp_data = json.load(f)

    segments_raw = interp_data["interpretation"]["segments"][-4:]
    segments = create_segments_from_json(segments_raw, well)

    return well, typewell, segments


def create_wrapper_and_bounds(well, typewell, segments, device='cuda'):
    """Create objective wrapper and bounds tensor."""
    # Calculate bounds
    bounds = calculate_optimization_bounds(segments, angle_range=10.0, accumulative=True)
    bounds_tensor = torch.tensor(bounds, dtype=torch.float64, device=device)

    # Initial projection
    well.calc_horizontal_projection(typewell, segments, 0.0)

    # Convert to numpy
    well_np = well_to_numpy(well)
    typewell_np = typewell_to_numpy(typewell)
    segments_np = segments_to_numpy(segments, well)

    # Convert to torch
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
        num_intervals_self_correlation=0,  # Disabled
        sc_power=1.15,
        angle_range=10.0,
        angle_sum_power=2.0,
        min_pearson_value=-1.0,
        tvd_to_typewell_shift=0.0,
        device=device
    )

    return wrapper, bounds_tensor


def test_evotorch_ga(wrapper, bounds_tensor, maxiter=100, popsize=100):
    """Test EvoTorch GeneticAlgorithm."""
    from evotorch import Problem
    from evotorch.algorithms import GeneticAlgorithm
    from evotorch.operators import GaussianMutation, SimulatedBinaryCrossOver

    print(f"\n{'='*60}")
    print("Testing EvoTorch GeneticAlgorithm")
    print(f"{'='*60}")

    device = str(bounds_tensor.device)
    K = bounds_tensor.shape[0]
    lb = bounds_tensor[:, 0].cpu().numpy()
    ub = bounds_tensor[:, 1].cpu().numpy()

    # Define problem
    class AGProblem(Problem):
        def __init__(self):
            super().__init__(
                objective_sense="min",
                solution_length=K,
                initial_bounds=(lb, ub),
                dtype=torch.float64,
                device=device
            )

        def _evaluate_batch(self, solutions):
            x = solutions.values  # (batch, K)
            fitness = wrapper(x)  # (batch,)
            solutions.set_evals(fitness)

    problem = AGProblem()

    # Create algorithm
    searcher = GeneticAlgorithm(
        problem,
        popsize=popsize,
        operators=[
            SimulatedBinaryCrossOver(problem, tournament_size=4, eta=20),
            GaussianMutation(problem, stdev=1.0)
        ]
    )

    start_time = time.time()

    # Run optimization
    for i in range(maxiter):
        searcher.step()
        if i % 20 == 0:
            best = searcher.status["best"]
            if best is not None:
                best_fitness = best.evals[0].item() if hasattr(best.evals, '__getitem__') else best.evals.item()
                print(f"  Iter {i}: best_fun={best_fitness:.6f}")

    elapsed = time.time() - start_time

    # Get best solution
    best = searcher.status["best"]
    if best is not None:
        best_x = best.values.cpu().numpy()
        best_fun = best.evals[0].item() if hasattr(best.evals, '__getitem__') else best.evals.item()
    else:
        best_x = None
        best_fun = float('inf')

    print(f"\nGA Results:")
    print(f"  best_fun: {best_fun:.6f}")
    print(f"  best_x: {best_x}")
    print(f"  time: {elapsed:.2f} sec")

    return best_fun, best_x, elapsed


def test_evotorch_snes(wrapper, bounds_tensor, maxiter=100, popsize=100, n_restarts=5):
    """Test EvoTorch SNES with multiple restarts."""
    from evotorch import Problem
    from evotorch.algorithms import SNES

    print(f"\n{'='*60}")
    print(f"Testing EvoTorch SNES ({n_restarts} restarts)")
    print(f"{'='*60}")

    device = str(bounds_tensor.device)
    K = bounds_tensor.shape[0]
    lb = bounds_tensor[:, 0].cpu().numpy()
    ub = bounds_tensor[:, 1].cpu().numpy()

    stdev_init = (ub - lb).mean() / 4.0

    class AGProblem(Problem):
        def __init__(self):
            super().__init__(
                objective_sense="min",
                solution_length=K,
                initial_bounds=(lb, ub),
                dtype=torch.float64,
                device=device
            )

        def _evaluate_batch(self, solutions):
            x = solutions.values
            fitness = wrapper(x)
            solutions.set_evals(fitness)

    start_time = time.time()
    best_overall_fun = float('inf')
    best_overall_x = None

    for restart in range(n_restarts):
        problem = AGProblem()
        searcher = SNES(
            problem,
            popsize=popsize,
            stdev_init=stdev_init,
            center_learning_rate=0.5,
            stdev_learning_rate=0.1
        )

        for i in range(maxiter):
            searcher.step()

        pop = searcher.population
        if pop is not None and hasattr(pop, 'evals') and pop.evals is not None:
            valid_mask = ~torch.isinf(pop.evals)
            if valid_mask.any():
                best_idx = pop.evals.argmin()
                best_fun = pop.evals[best_idx].item()
                best_x = pop.values[best_idx].cpu().numpy()
                print(f"  Restart {restart+1}: best_fun={best_fun:.6f}")
                if best_fun < best_overall_fun:
                    best_overall_fun = best_fun
                    best_overall_x = best_x

    elapsed = time.time() - start_time

    print(f"\nSNES Results (best of {n_restarts} restarts):")
    print(f"  best_fun: {best_overall_fun:.6f}")
    print(f"  best_x: {best_overall_x}")
    print(f"  time: {elapsed:.2f} sec")

    return best_overall_fun, best_overall_x, elapsed


def test_evotorch_xnes(wrapper, bounds_tensor, maxiter=1000, popsize=200):
    """Test EvoTorch XNES (Exponential Natural Evolution Strategy)."""
    from evotorch import Problem
    from evotorch.algorithms import XNES

    print(f"\n{'='*60}")
    print("Testing EvoTorch XNES")
    print(f"{'='*60}")

    device = str(bounds_tensor.device)
    K = bounds_tensor.shape[0]
    lb = bounds_tensor[:, 0].cpu().numpy()
    ub = bounds_tensor[:, 1].cpu().numpy()

    stdev_init = (ub - lb).mean() / 4.0

    class AGProblem(Problem):
        def __init__(self):
            super().__init__(
                objective_sense="min",
                solution_length=K,
                initial_bounds=(lb, ub),
                dtype=torch.float64,
                device=device
            )

        def _evaluate_batch(self, solutions):
            x = solutions.values
            fitness = wrapper(x)
            solutions.set_evals(fitness)

    problem = AGProblem()

    searcher = XNES(
        problem,
        popsize=popsize,
        stdev_init=stdev_init
    )

    start_time = time.time()

    for i in range(maxiter):
        searcher.step()
        if i % 200 == 0:
            pop = searcher.population
            if pop is not None and hasattr(pop, 'evals') and pop.evals is not None:
                valid_mask = ~torch.isinf(pop.evals)
                if valid_mask.any():
                    best_val = pop.evals[valid_mask].min().item()
                    print(f"  Iter {i}: pop_best={best_val:.6f}")

    elapsed = time.time() - start_time

    pop = searcher.population
    if pop is not None and hasattr(pop, 'evals') and pop.evals is not None:
        valid_mask = ~torch.isinf(pop.evals)
        if valid_mask.any():
            best_idx = pop.evals.argmin()
            best_x = pop.values[best_idx].cpu().numpy()
            best_fun = pop.evals[best_idx].item()
        else:
            best_x = None
            best_fun = float('inf')
    else:
        best_x = None
        best_fun = float('inf')

    print(f"\nXNES Results:")
    print(f"  best_fun: {best_fun:.6f}")
    print(f"  best_x: {best_x}")
    print(f"  time: {elapsed:.2f} sec")

    return best_fun, best_x, elapsed


def test_evotorch_cem(wrapper, bounds_tensor, maxiter=1000, popsize=200):
    """Test EvoTorch CEM (Cross-Entropy Method)."""
    from evotorch import Problem
    from evotorch.algorithms import CEM

    print(f"\n{'='*60}")
    print("Testing EvoTorch CEM")
    print(f"{'='*60}")

    device = str(bounds_tensor.device)
    K = bounds_tensor.shape[0]
    lb = bounds_tensor[:, 0].cpu().numpy()
    ub = bounds_tensor[:, 1].cpu().numpy()

    stdev_init = (ub - lb).mean() / 4.0

    class AGProblem(Problem):
        def __init__(self):
            super().__init__(
                objective_sense="min",
                solution_length=K,
                initial_bounds=(lb, ub),
                dtype=torch.float64,
                device=device
            )

        def _evaluate_batch(self, solutions):
            x = solutions.values
            fitness = wrapper(x)
            solutions.set_evals(fitness)

    problem = AGProblem()

    searcher = CEM(
        problem,
        popsize=popsize,
        stdev_init=stdev_init,
        parenthood_ratio=0.5  # Top 50% for elite
    )

    start_time = time.time()

    for i in range(maxiter):
        searcher.step()
        if i % 200 == 0:
            pop = searcher.population
            if pop is not None and hasattr(pop, 'evals') and pop.evals is not None:
                valid_mask = ~torch.isinf(pop.evals)
                if valid_mask.any():
                    best_val = pop.evals[valid_mask].min().item()
                    print(f"  Iter {i}: pop_best={best_val:.6f}")

    elapsed = time.time() - start_time

    pop = searcher.population
    if pop is not None and hasattr(pop, 'evals') and pop.evals is not None:
        valid_mask = ~torch.isinf(pop.evals)
        if valid_mask.any():
            best_idx = pop.evals.argmin()
            best_x = pop.values[best_idx].cpu().numpy()
            best_fun = pop.evals[best_idx].item()
        else:
            best_x = None
            best_fun = float('inf')
    else:
        best_x = None
        best_fun = float('inf')

    print(f"\nCEM Results:")
    print(f"  best_fun: {best_fun:.6f}")
    print(f"  best_x: {best_x}")
    print(f"  time: {elapsed:.2f} sec")

    return best_fun, best_x, elapsed


def test_evotorch_pgpe(wrapper, bounds_tensor, maxiter=1000, popsize=200):
    """Test EvoTorch PGPE (Policy Gradients with Parameter-based Exploration)."""
    from evotorch import Problem
    from evotorch.algorithms import PGPE

    print(f"\n{'='*60}")
    print("Testing EvoTorch PGPE")
    print(f"{'='*60}")

    device = str(bounds_tensor.device)
    K = bounds_tensor.shape[0]
    lb = bounds_tensor[:, 0].cpu().numpy()
    ub = bounds_tensor[:, 1].cpu().numpy()

    stdev_init = (ub - lb).mean() / 4.0

    class AGProblem(Problem):
        def __init__(self):
            super().__init__(
                objective_sense="min",
                solution_length=K,
                initial_bounds=(lb, ub),
                dtype=torch.float64,
                device=device
            )

        def _evaluate_batch(self, solutions):
            x = solutions.values
            fitness = wrapper(x)
            solutions.set_evals(fitness)

    problem = AGProblem()

    searcher = PGPE(
        problem,
        popsize=popsize,
        stdev_init=stdev_init,
        center_learning_rate=0.1,
        stdev_learning_rate=0.05
    )

    start_time = time.time()

    for i in range(maxiter):
        searcher.step()
        if i % 200 == 0:
            pop = searcher.population
            if pop is not None and hasattr(pop, 'evals') and pop.evals is not None:
                valid_mask = ~torch.isinf(pop.evals)
                if valid_mask.any():
                    best_val = pop.evals[valid_mask].min().item()
                    print(f"  Iter {i}: pop_best={best_val:.6f}")

    elapsed = time.time() - start_time

    pop = searcher.population
    if pop is not None and hasattr(pop, 'evals') and pop.evals is not None:
        valid_mask = ~torch.isinf(pop.evals)
        if valid_mask.any():
            best_idx = pop.evals.argmin()
            best_x = pop.values[best_idx].cpu().numpy()
            best_fun = pop.evals[best_idx].item()
        else:
            best_x = None
            best_fun = float('inf')
    else:
        best_x = None
        best_fun = float('inf')

    print(f"\nPGPE Results:")
    print(f"  best_fun: {best_fun:.6f}")
    print(f"  best_x: {best_x}")
    print(f"  time: {elapsed:.2f} sec")

    return best_fun, best_x, elapsed


def test_scipy_de_with_gpu_eval(wrapper, bounds_tensor, maxiter=1000, popsize=50):
    """Test scipy DE but with GPU batch evaluation."""
    from scipy.optimize import differential_evolution
    import numpy as np

    print(f"\n{'='*60}")
    print("Testing scipy DE with GPU evaluation")
    print(f"{'='*60}")

    K = bounds_tensor.shape[0]
    bounds_np = bounds_tensor.cpu().numpy()
    bounds_list = [(bounds_np[i, 0], bounds_np[i, 1]) for i in range(K)]

    eval_count = [0]

    def objective(x):
        # Convert to GPU tensor and evaluate
        x_tensor = torch.tensor(x, dtype=torch.float64, device='cuda').unsqueeze(0)
        result = wrapper(x_tensor)
        eval_count[0] += 1
        return result[0].item()

    start_time = time.time()

    result = differential_evolution(
        objective,
        bounds_list,
        strategy='rand1bin',
        maxiter=maxiter,
        popsize=popsize,
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        disp=False,
        workers=1,  # Single thread for GPU
        updating='deferred'  # Batch mode
    )

    elapsed = time.time() - start_time

    print(f"\nScipy DE + GPU Results:")
    print(f"  best_fun: {result.fun:.6f}")
    print(f"  best_x: {result.x}")
    print(f"  nit: {result.nit}")
    print(f"  nfev: {eval_count[0]}")
    print(f"  time: {elapsed:.2f} sec")

    return result.fun, result.x, elapsed


def main():
    print("="*60)
    print("EvoTorch Optimization Test")
    print("="*60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading test data...")
    well, typewell, segments = load_test_data()
    print(f"  Segments: {len(segments)}")

    # Create wrapper
    print("Creating objective wrapper...")
    wrapper, bounds_tensor = create_wrapper_and_bounds(well, typewell, segments, device='cuda')
    print(f"  Bounds shape: {bounds_tensor.shape}")
    print(f"  Bounds range: [{bounds_tensor[:, 0].min():.2f}, {bounds_tensor[:, 1].max():.2f}]")

    # Test algorithms
    results = {}

    # Skip GA for now - focus on SNES
    # try:
    #     ga_fun, ga_x, ga_time = test_evotorch_ga(wrapper, bounds_tensor, maxiter=100, popsize=100)
    #     results['GA'] = {'fun': ga_fun, 'time': ga_time}
    # except Exception as e:
    #     print(f"GA failed: {e}")
    #     results['GA'] = {'fun': float('inf'), 'error': str(e)}

    # Test our GPU DE implementation
    try:
        from torch_funcs.gpu_optimizer import differential_evolution_torch
        import time as time_module

        print(f"\n{'='*60}")
        print("Testing our GPU DE with popsize=500")
        print(f"{'='*60}")

        start = time_module.time()
        result = differential_evolution_torch(
            objective_fn=wrapper,
            bounds=bounds_tensor,
            popsize=500,
            maxiter=500,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            callback=None,
            device='cuda'
        )
        gpu_de_time = time_module.time() - start

        print(f"\nGPU DE Results:")
        print(f"  best_fun: {result['fun']:.6f}")
        print(f"  best_x: {result['x'].cpu().numpy()}")
        print(f"  nit: {result['nit']}")
        print(f"  time: {gpu_de_time:.2f} sec")

        results['GPU_DE'] = {'fun': result['fun'], 'time': gpu_de_time}
    except Exception as e:
        print(f"GPU DE failed: {e}")
        import traceback
        traceback.print_exc()
        results['GPU_DE'] = {'fun': float('inf'), 'error': str(e)}

    # Skip XNES - worse than SNES
    # try:
    #     xnes_fun, xnes_x, xnes_time = test_evotorch_xnes(wrapper, bounds_tensor, maxiter=1000, popsize=200)
    #     results['XNES'] = {'fun': xnes_fun, 'time': xnes_time}
    # except Exception as e:
    #     print(f"XNES failed: {e}")
    #     results['XNES'] = {'fun': float('inf'), 'error': str(e)}

    # Skip CEM - worse than SNES
    # try:
    #     cem_fun, cem_x, cem_time = test_evotorch_cem(wrapper, bounds_tensor, maxiter=1000, popsize=200)
    #     results['CEM'] = {'fun': cem_fun, 'time': cem_time}
    # except Exception as e:
    #     print(f"CEM failed: {e}")
    #     results['CEM'] = {'fun': float('inf'), 'error': str(e)}

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Algorithm':<10} {'best_fun':>12} {'time (sec)':>12}")
    print("-"*40)
    for algo, res in results.items():
        fun = res.get('fun', float('inf'))
        t = res.get('time', 0)
        err = res.get('error', '')
        if err:
            print(f"{algo:<10} {'ERROR':>12} {err}")
        else:
            print(f"{algo:<10} {fun:>12.6f} {t:>12.2f}")

    print(f"\nTarget: best_fun < 0.2 (CPU scipy achieves ~0.15)")


if __name__ == "__main__":
    main()
