"""
Benchmark batch objective function speed without self-correlation.
"""
import sys
import json
import time
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))
sys.path.insert(0, str(Path(__file__).parent))

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json

from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
from torch_funcs.batch_objective import batch_objective_function_torch


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data (support both Windows and WSL paths)
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

    well = Well(well_data_json)
    typewell = TypeWell(well_data_json)

    interp_path = starsteer_dir / "interpretation.json"
    with open(interp_path) as f:
        interp_data = json.load(f)

    segments_raw = interp_data["interpretation"]["segments"][-4:]
    segments = create_segments_from_json(segments_raw, well)
    shifts = [seg['endShift'] for seg in segments_raw]

    # Convert
    typewell_np = typewell_to_numpy(typewell)
    segments_np = segments_to_numpy(segments, well)

    typewell_torch = numpy_to_torch(typewell_np, device=device)
    segments_torch = segments_numpy_to_torch(segments_np, device=device)
    shifts_base = torch.tensor(shifts, dtype=torch.float64, device=device)

    print("\n" + "=" * 60)
    print("BENCHMARK: BATCH OBJECTIVE (NO SELF-CORRELATION)")
    print("=" * 60)

    # Create batch with perturbations
    batch_size = 500
    torch.manual_seed(42)
    perturbations = torch.randn(batch_size, len(shifts), dtype=torch.float64, device=device) * 0.1
    shifts_batch = shifts_base.unsqueeze(0) + perturbations
    shifts_batch[0] = shifts_base  # First is exact

    # Warmup
    well_torch = numpy_to_torch(well_to_numpy(well), device=device)
    _ = batch_objective_function_torch(
        shifts_batch[:10],
        well_torch,
        typewell_torch,
        segments_torch,
        self_corr_start_idx=0,
        pearson_power=params['pearson_power'],
        mse_power=params['mse_power'],
        num_intervals_self_correlation=0,  # DISABLED
        sc_power=params['sc_power'],
        angle_range=params['angle_range'],
        angle_sum_power=params['angle_sum_power'],
        min_pearson_value=params['min_pearson_value'],
        tvd_to_typewell_shift=0.0
    )

    # Benchmark
    n_runs = 5
    times = []

    for run in range(n_runs):
        well_torch = numpy_to_torch(well_to_numpy(well), device=device)

        if device == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()

        results = batch_objective_function_torch(
            shifts_batch,
            well_torch,
            typewell_torch,
            segments_torch,
            self_corr_start_idx=0,
            pearson_power=params['pearson_power'],
            mse_power=params['mse_power'],
            num_intervals_self_correlation=0,  # DISABLED
            sc_power=params['sc_power'],
            angle_range=params['angle_range'],
            angle_sum_power=params['angle_sum_power'],
            min_pearson_value=params['min_pearson_value'],
            tvd_to_typewell_shift=0.0
        )

        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"Run {run + 1}: {elapsed:.4f} sec")

    avg_time = sum(times) / len(times)
    min_time = min(times)

    print(f"\nBatch size: {batch_size}")
    print(f"Average time: {avg_time:.4f} sec")
    print(f"Min time: {min_time:.4f} sec")
    print(f"Time per eval: {min_time / batch_size * 1000:.4f} ms")
    print(f"Evals per second: {batch_size / min_time:.1f}")

    # Validate results
    valid_mask = ~torch.isinf(results)
    valid_count = torch.sum(valid_mask).item()
    print(f"\nValid results: {valid_count}/{batch_size}")

    if valid_count > 0:
        valid_results = results[valid_mask]
        print(f"Min: {torch.min(valid_results).item():.6f}")
        print(f"Max: {torch.max(valid_results).item():.6f}")
        print(f"Mean: {torch.mean(valid_results).item():.6f}")

    # Compare with CPU reference
    print("\n" + "=" * 60)
    print("CPU REFERENCE (for comparison)")
    print("=" * 60)
    print("CPU baseline: ~330 sec for 1000 iterations × 500 popsize")
    print("CPU single eval: ~0.66 ms (330 / 500000)")
    print(f"GPU batch eval: {min_time / batch_size * 1000:.4f} ms")
    print(f"Speedup vs CPU single: {0.66 / (min_time / batch_size * 1000):.1f}x")


if __name__ == "__main__":
    main()
