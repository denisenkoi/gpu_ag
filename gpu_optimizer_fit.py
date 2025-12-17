"""
GPU replacement for ag_numerical.ag_func_optimizer.optimizer_fit

Drop-in replacement that uses GPU-accelerated Differential Evolution.
Call signature matches CPU version for easy substitution.
"""
import sys
import numpy as np
import torch
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from ag_rewards.ag_func_correlations import calculate_correlation
from ag_objects.ag_obj_interpretation import create_segments

# GPU imports
from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
from torch_funcs.batch_objective import TorchObjectiveWrapper
from torch_funcs.gpu_optimizer import differential_evolution_torch


def gpu_optimizer_fit(
    well,
    typewell,
    self_corr_start_idx,
    segments,
    angle_range,
    angle_sum_power,
    segm_counts_reg,
    num_iterations,
    pearson_power,
    mse_power,
    num_intervals_self_correlation,
    sc_power,
    optimizer_method,
    min_pearson_value,
    use_accumulative_bounds,
    tvd_to_typewell_shift=0.0,
    multi_threaded=False,
    device='cuda',
    verbose=False
):
    """
    GPU-accelerated optimizer_fit replacement.

    Same interface as cpu_baseline/ag_numerical/ag_func_optimizer.optimizer_fit
    but uses PyTorch GPU for parallel evaluation.

    Args:
        well: Well object
        typewell: TypeWell object
        self_corr_start_idx: Start index for self-correlation
        segments: List of Segment objects to optimize
        angle_range: Max angle constraint
        angle_sum_power: Power for angle sum penalty
        segm_counts_reg: Regional segment counts (not used in DE)
        num_iterations: Max iterations (maps to maxiter)
        pearson_power, mse_power: Metric weights
        num_intervals_self_correlation: Self-correlation intervals
        sc_power: Self-correlation power
        optimizer_method: 'differential_evolution' expected
        min_pearson_value: Minimum pearson threshold
        use_accumulative_bounds: Bound calculation mode
        tvd_to_typewell_shift: Vertical shift
        multi_threaded: Ignored (GPU is parallel by design)
        device: 'cuda' or 'cpu'
        verbose: Print progress

    Returns:
        List of results [(corr, self_corr, pearson, mse, num_points, segments, well), ...]
    """
    print(f"[GPU_OPTIMIZER] gpu_optimizer_fit CALLED! device={device}, num_iterations={num_iterations}, segments={len(segments)}")

    import time
    from ag_numerical.ag_optimizer_utils import calculate_optimization_bounds, collect_optimization_stats

    # NO FALLBACK! GPU is required, fail hard if not available
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU REQUIRED but CUDA is not available! "
                "Install CUDA-enabled PyTorch or check GPU drivers. "
                "NO FALLBACK TO CPU - this is intentional."
            )
        # Check GPU compute capability
        gpu_props = torch.cuda.get_device_properties(0)
        major, minor = gpu_props.major, gpu_props.minor
        # Test actual GPU tensor creation
        try:
            test_tensor = torch.zeros(1, device='cuda')
            del test_tensor
        except Exception as e:
            raise RuntimeError(
                f"GPU {gpu_props.name} (sm_{major}{minor}) detected but tensor creation failed! "
                f"PyTorch may not support this GPU architecture. Error: {e}. "
                "NO FALLBACK TO CPU - upgrade PyTorch or use WSL."
            )

    start_time = time.time()

    # Calculate initial projection
    well.calc_horizontal_projection(typewell, segments, tvd_to_typewell_shift)

    # Calculate bounds (same as CPU)
    bounds = calculate_optimization_bounds(segments, angle_range, use_accumulative_bounds)

    # Initial shifts
    initial_shifts = [segment.end_shift for segment in segments]

    # Convert to numpy
    well_np = well_to_numpy(well)
    typewell_np = typewell_to_numpy(typewell)
    segments_np = segments_to_numpy(segments, well)

    # Convert to torch
    well_torch = numpy_to_torch(well_np, device=device)
    typewell_torch = numpy_to_torch(typewell_np, device=device)
    segments_torch = segments_numpy_to_torch(segments_np, device=device)

    # Create objective wrapper
    wrapper = TorchObjectiveWrapper(
        well_data=well_torch,
        typewell_data=typewell_torch,
        segments_torch=segments_torch,
        self_corr_start_idx=self_corr_start_idx,
        pearson_power=pearson_power,
        mse_power=mse_power,
        num_intervals_self_correlation=0,  # Disabled for speed
        sc_power=sc_power,
        angle_range=angle_range,
        angle_sum_power=angle_sum_power,
        min_pearson_value=min_pearson_value,
        tvd_to_typewell_shift=tvd_to_typewell_shift,
        device=device
    )

    # Convert bounds to tensor
    bounds_tensor = torch.tensor(bounds, dtype=torch.float64, device=device)

    # Callback for progress
    def callback(iteration, population, fitness):
        if verbose and iteration % 100 == 0:
            best_idx = torch.argmin(fitness)
            best_fun = fitness[best_idx].item()
            print(f"GPU DE iter {iteration}: best={best_fun:.6f}")
        return False

    # Run GPU DE with scipy default parameters
    # CRITICAL: mutation=(0.5, 1.0) and recombination=0.7 are scipy defaults
    # Aggressive params (1.5-1.99, 0.99) cause convergence to local minima!
    result = differential_evolution_torch(
        objective_fn=wrapper,
        bounds=bounds_tensor,
        popsize=500,
        maxiter=num_iterations if optimizer_method == 'differential_evolution' else 100,
        mutation=(0.5, 1.0),  # scipy default: dithered F in [0.5, 1.0]
        recombination=0.7,    # scipy default: CR=0.7
        seed=None,
        callback=callback if verbose else None,
        device=device
    )

    elapsed_time = time.time() - start_time

    print(f"[GPU_OPTIMIZER] DE completed in {elapsed_time:.2f}s, best_fun={result['fun']:.6f}, nit={result['nit']}")

    # Extract optimal shifts (avoid .numpy() for compatibility)
    optimal_shifts = result['x'].cpu().tolist()
    print(f"[GPU_OPTIMIZER] Optimal shifts: {optimal_shifts}")

    # Update segments with optimal shifts
    optimal_segments = deepcopy(segments)
    for i, shift in enumerate(optimal_shifts):
        optimal_segments[i].end_shift = shift
        if i < len(optimal_segments) - 1:
            optimal_segments[i + 1].start_shift = shift

    # Update TVT in well
    well.calc_horizontal_projection(typewell, optimal_segments, tvd_to_typewell_shift)

    # Calculate final correlation
    well_copy = deepcopy(well)
    well_copy.calc_horizontal_projection(typewell, optimal_segments, tvd_to_typewell_shift)

    corr, _, self_correlation, _, _, pearson, num_points, mse, _, _ = calculate_correlation(
        well_copy,
        self_corr_start_idx,
        optimal_segments[0].start_idx,
        optimal_segments[-1].end_idx,
        float('inf'),
        0,
        0,
        pearson_power,
        mse_power,
        num_intervals_self_correlation,
        sc_power,
        min_pearson_value
    )

    # Collect stats
    optimization_stats = {
        'method': 'gpu_differential_evolution',
        'final_fun': result['fun'],
        'nit': result['nit'],
        'nfev': result['nfev'],
        'elapsed_time_sec': round(elapsed_time, 2),
        'device': device
    }

    if verbose:
        print(f"GPU optimization: fun={result['fun']:.6f}, time={elapsed_time:.2f}s, "
              f"nfev={result['nfev']}, device={device}")

    # Build results list (same format as CPU)
    results = [(corr, self_correlation, pearson, mse, num_points, optimal_segments, well_copy)]

    # Generate perturbed results (same as CPU for consistency)
    import random
    for _ in range(min(num_iterations - 1, 9)):
        noisy_segments = deepcopy(optimal_segments)

        for i, segment in enumerate(noisy_segments):
            segment_len = segment.end_vs - segment.start_vs
            perturbation_size = segment_len * well.horizontal_well_step * np.tan(np.radians(angle_range * 0.1))
            shift_perturbation = random.uniform(-perturbation_size, perturbation_size)

            segment.end_shift += shift_perturbation
            if i < len(noisy_segments) - 1:
                noisy_segments[i + 1].start_shift = segment.end_shift

        noisy_well = deepcopy(well)
        success = noisy_well.calc_horizontal_projection(typewell, noisy_segments, tvd_to_typewell_shift)
        if success:
            perturbed_corr, _, perturbed_self_correlation, _, _, perturbed_pearson, perturbed_num_points, perturbed_mse, _, _ = calculate_correlation(
                noisy_well,
                self_corr_start_idx,
                noisy_segments[0].start_idx,
                noisy_segments[-1].end_idx,
                float('inf'),
                0,
                0,
                pearson_power,
                mse_power,
                num_intervals_self_correlation,
                sc_power,
                min_pearson_value
            )

            results.append((perturbed_corr, perturbed_self_correlation, perturbed_pearson, perturbed_mse,
                            perturbed_num_points, noisy_segments, noisy_well))

    # Sort by correlation (descending)
    results.sort(key=lambda x: x[0], reverse=True)

    return results
