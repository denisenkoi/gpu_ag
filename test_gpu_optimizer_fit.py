"""
Test gpu_optimizer_fit as drop-in replacement for CPU optimizer_fit.
Compares results and timing with CPU baseline.
"""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))
sys.path.insert(0, str(Path(__file__).parent))

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json

# CPU baseline
from ag_numerical.ag_func_optimizer import optimizer_fit as cpu_optimizer_fit

# GPU replacement
from gpu_optimizer_fit import gpu_optimizer_fit


def main():
    # Load data
    starsteer_dir = Path("/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")
    if not starsteer_dir.exists():
        starsteer_dir = Path("E:/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")

    well_json_path = starsteer_dir / "AG_DATA" / "InitialData" / "slicing_well.json"
    with open(well_json_path, 'r', encoding='utf-8') as f:
        well_data_json = json.load(f)

    checkpoint_path = Path(__file__).parent / "cpu_baseline" / "test_checkpoint_values.json"
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)
    params = checkpoint['parameters']

    # Create objects
    well = Well(well_data_json)
    typewell = TypeWell(well_data_json)

    with open(starsteer_dir / "interpretation.json") as f:
        interp_data = json.load(f)

    segments_raw = interp_data["interpretation"]["segments"][-4:]
    segments = create_segments_from_json(segments_raw, well)

    # Common parameters
    common_params = {
        'well': well,
        'typewell': typewell,
        'self_corr_start_idx': 0,
        'segments': segments,
        'angle_range': params['angle_range'],
        'angle_sum_power': params['angle_sum_power'],
        'segm_counts_reg': [2, 4, 6, 10],
        'num_iterations': 100,  # Reduced for quick comparison
        'pearson_power': params['pearson_power'],
        'mse_power': params['mse_power'],
        'num_intervals_self_correlation': 0,  # Disabled for speed
        'sc_power': params['sc_power'],
        'optimizer_method': 'differential_evolution',
        'min_pearson_value': params['min_pearson_value'],
        'use_accumulative_bounds': True,
        'tvd_to_typewell_shift': 0.0,
    }

    print("=" * 60)
    print("GPU vs CPU OPTIMIZER_FIT COMPARISON")
    print("=" * 60)

    # GPU test
    print("\n--- GPU optimizer_fit (100 iterations) ---")
    # Recreate fresh segments
    segments_gpu = create_segments_from_json(segments_raw, well)
    well_gpu = Well(well_data_json)

    start = time.perf_counter()
    gpu_results = gpu_optimizer_fit(
        well=well_gpu,
        typewell=typewell,
        self_corr_start_idx=0,
        segments=segments_gpu,
        angle_range=params['angle_range'],
        angle_sum_power=params['angle_sum_power'],
        segm_counts_reg=[2, 4, 6, 10],
        num_iterations=100,
        pearson_power=params['pearson_power'],
        mse_power=params['mse_power'],
        num_intervals_self_correlation=0,
        sc_power=params['sc_power'],
        optimizer_method='differential_evolution',
        min_pearson_value=params['min_pearson_value'],
        use_accumulative_bounds=True,
        tvd_to_typewell_shift=0.0,
        verbose=True
    )
    gpu_time = time.perf_counter() - start

    gpu_best = gpu_results[0]
    print(f"\nGPU Results:")
    print(f"  Correlation: {gpu_best[0]:.6f}")
    print(f"  Self-corr: {gpu_best[1]:.6f}")
    print(f"  Pearson: {gpu_best[2]:.6f}")
    print(f"  MSE: {gpu_best[3]:.6f}")
    print(f"  Num points: {gpu_best[4]}")
    print(f"  Time: {gpu_time:.2f} sec")

    gpu_shifts = [seg.end_shift for seg in gpu_best[5]]
    print(f"  Optimal shifts: {gpu_shifts}")

    # CPU test (skip if too slow)
    print("\n--- CPU optimizer_fit (100 iterations) ---")
    print("(This will take ~33 seconds based on 330s for 1000 iter)")

    # Recreate fresh segments
    segments_cpu = create_segments_from_json(segments_raw, well)
    well_cpu = Well(well_data_json)

    start = time.perf_counter()
    cpu_results = cpu_optimizer_fit(
        well=well_cpu,
        typewell=typewell,
        self_corr_start_idx=0,
        segments=segments_cpu,
        angle_range=params['angle_range'],
        angle_sum_power=params['angle_sum_power'],
        segm_counts_reg=[2, 4, 6, 10],
        num_iterations=100,
        pearson_power=params['pearson_power'],
        mse_power=params['mse_power'],
        num_intervals_self_correlation=0,
        sc_power=params['sc_power'],
        optimizer_method='differential_evolution',
        min_pearson_value=params['min_pearson_value'],
        use_accumulative_bounds=True,
        tvd_to_typewell_shift=0.0,
    )
    cpu_time = time.perf_counter() - start

    cpu_best = cpu_results[0]
    print(f"\nCPU Results:")
    print(f"  Correlation: {cpu_best[0]:.6f}")
    print(f"  Self-corr: {cpu_best[1]:.6f}")
    print(f"  Pearson: {cpu_best[2]:.6f}")
    print(f"  MSE: {cpu_best[3]:.6f}")
    print(f"  Num points: {cpu_best[4]}")
    print(f"  Time: {cpu_time:.2f} sec")

    cpu_shifts = [seg.end_shift for seg in cpu_best[5]]
    print(f"  Optimal shifts: {cpu_shifts}")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"GPU time: {gpu_time:.2f} sec")
    print(f"CPU time: {cpu_time:.2f} sec")
    print(f"Speedup: {cpu_time / gpu_time:.1f}x")
    print(f"\nCorrelation diff: {abs(gpu_best[0] - cpu_best[0]):.6f}")
    print(f"Pearson diff: {abs(gpu_best[2] - cpu_best[2]):.6f}")


if __name__ == "__main__":
    main()
