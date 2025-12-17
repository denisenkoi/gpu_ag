"""
Full pipeline test: GPU optimizer in context similar to slicer.

Tests the complete flow: Well -> normalize -> optimize -> denormalize
"""
import sys
import json
import time
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))
sys.path.insert(0, str(Path(__file__).parent))

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json, create_segments
from ag_rewards.ag_func_correlations import calculate_correlation

# GPU optimizer
from gpu_optimizer_fit import gpu_optimizer_fit


def main():
    print("=" * 60)
    print("FULL PIPELINE TEST: GPU Optimizer")
    print("=" * 60)

    # Load data
    starsteer_dir = Path("/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")
    if not starsteer_dir.exists():
        starsteer_dir = Path("E:/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")

    well_json_path = starsteer_dir / "AG_DATA" / "InitialData" / "slicing_well.json"
    with open(well_json_path, 'r', encoding='utf-8') as f:
        well_data_json = json.load(f)

    # Create Well and TypeWell (NOT normalized - GPU optimizer works with raw data)
    well = Well(well_data_json)
    typewell = TypeWell(well_data_json)

    print(f"Well MD range: {well.min_md:.2f} - {well.max_md:.2f}")

    # Load interpretation and create segments
    with open(starsteer_dir / "interpretation.json") as f:
        interp_data = json.load(f)

    segments_raw = interp_data["interpretation"]["segments"][-4:]
    segments = create_segments_from_json(segments_raw, well)

    # NOTE: NOT normalizing - GPU optimizer works with raw (unnormalized) data
    # This matches test_gpu_de.py and test_gpu_optimizer_fit.py which work correctly

    print(f"\nSegments: {len(segments)}")
    for i, seg in enumerate(segments):
        print(f"  [{i}] idx={seg.start_idx}-{seg.end_idx}, "
              f"shift={seg.start_shift:.6f}-{seg.end_shift:.6f}")

    # Parameters (like executor defaults)
    angle_range = 10.0
    pearson_power = 2.0
    mse_power = 0.001
    num_intervals_self_correlation = 0  # Disabled for GPU
    sc_power = 1.15
    min_pearson_value = -1.0
    use_accumulative_bounds = True

    # Calculate initial correlation
    well.calc_horizontal_projection(typewell, segments, 0.0)
    start_idx = segments[0].start_idx
    end_idx = segments[-1].end_idx

    init_corr, _, init_self_corr, _, _, init_pearson, init_points, init_mse, _, _ = calculate_correlation(
        well, start_idx, start_idx, end_idx,
        float('inf'), 0, 0,
        pearson_power, mse_power, num_intervals_self_correlation, sc_power, min_pearson_value
    )
    print(f"\nINITIAL: corr={init_corr:.6f}, pearson={init_pearson:.6f}, mse={init_mse:.6f}")

    # Run GPU optimization
    print("\n" + "=" * 60)
    print("Running GPU optimizer_fit (1000 iterations)...")
    print("=" * 60)

    start_time = time.perf_counter()

    results = gpu_optimizer_fit(
        well=well,
        typewell=typewell,
        self_corr_start_idx=start_idx,
        segments=segments,
        angle_range=angle_range,
        angle_sum_power=2.0,
        segm_counts_reg=[2, 4, 6, 10],
        num_iterations=1000,
        pearson_power=pearson_power,
        mse_power=mse_power,
        num_intervals_self_correlation=num_intervals_self_correlation,
        sc_power=sc_power,
        optimizer_method='differential_evolution',
        min_pearson_value=min_pearson_value,
        use_accumulative_bounds=use_accumulative_bounds,
        tvd_to_typewell_shift=0.0,
        verbose=True
    )

    elapsed = time.perf_counter() - start_time

    # Best result
    best = results[0]
    corr, self_corr, pearson, mse, num_points, opt_segments, _ = best

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Correlation: {corr:.6f}")
    print(f"Pearson: {pearson:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"Points: {num_points}")
    print(f"Time: {elapsed:.2f} sec")

    print("\nOptimized shifts (meters):")
    for i, seg in enumerate(opt_segments):
        print(f"  [{i}] shift={seg.start_shift:.3f}m - {seg.end_shift:.3f}m")

    # Comparison with reference
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"CPU baseline (~330 sec for 1000 iter)")
    print(f"GPU: {elapsed:.2f} sec")
    print(f"Speedup: {330 / elapsed:.1f}x")


if __name__ == "__main__":
    main()
