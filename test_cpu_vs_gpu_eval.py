"""
Compare CPU vs GPU objective function evaluation.
"""
import sys
import json
import torch
import numpy as np
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json
from ag_rewards.ag_func_correlations import objective_function_optimizer

from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
from torch_funcs.batch_objective import TorchObjectiveWrapper


def main():
    print("="*60)
    print("CPU vs GPU Objective Function Comparison")
    print("="*60)

    # Load data
    base_dir = Path("/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")

    with open(base_dir / "AG_DATA/InitialData/slicing_well.json", 'r') as f:
        well_data = json.load(f)

    well = Well(well_data)
    typewell = TypeWell(well_data)

    with open(base_dir / "interpretation.json", 'r') as f:
        interp_data = json.load(f)

    segments_raw = interp_data["interpretation"]["segments"][-4:]
    segments = create_segments_from_json(segments_raw, well)

    print(f"Segments: {len(segments)}")
    print(f"Initial shifts: {[s.end_shift for s in segments]}")

    # Test point - the reference interpretation shifts
    test_shifts = np.array([-5.663609951795427, -9.704093997855415, -12.38605218676823])
    print(f"\nTest shifts: {test_shifts}")

    # === CPU evaluation ===
    print("\n--- CPU Evaluation ---")
    well_cpu = deepcopy(well)
    segments_cpu = deepcopy(segments)

    # Apply shifts to segments
    for i, shift in enumerate(test_shifts):
        segments_cpu[i].end_shift = shift
        if i < len(segments_cpu) - 1:
            segments_cpu[i + 1].start_shift = shift

    # Calculate projection
    well_cpu.calc_horizontal_projection(typewell, segments_cpu, 0.0)

    # CPU objective
    cpu_result = objective_function_optimizer(
        shifts=test_shifts.tolist(),
        well=well_cpu,
        typewell=typewell,
        self_corr_start_idx=0,
        segments=segments_cpu,
        angle_range=10.0,
        angle_sum_power=2.0,
        pearson_power=2.0,
        mse_power=0.001,
        num_intervals_self_correlation=0,
        sc_power=1.15,
        min_pearson_value=-1.0,
        tvd_to_typewell_shift=0.0
    )
    print(f"CPU objective: {cpu_result:.10f}")

    # === GPU evaluation ===
    print("\n--- GPU Evaluation ---")
    well_gpu = deepcopy(well)
    segments_gpu = deepcopy(segments)

    # Initial projection
    well_gpu.calc_horizontal_projection(typewell, segments_gpu, 0.0)

    # Convert to numpy/torch
    well_np = well_to_numpy(well_gpu)
    typewell_np = typewell_to_numpy(typewell)
    segments_np = segments_to_numpy(segments_gpu, well_gpu)

    well_torch = numpy_to_torch(well_np, device='cuda')
    typewell_torch = numpy_to_torch(typewell_np, device='cuda')
    segments_torch = segments_numpy_to_torch(segments_np, device='cuda')

    # GPU wrapper
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
        device='cuda'
    )

    # GPU objective
    shifts_tensor = torch.tensor(test_shifts, dtype=torch.float64, device='cuda').unsqueeze(0)
    gpu_result = wrapper(shifts_tensor)[0].item()
    print(f"GPU objective: {gpu_result:.10f}")

    # === Comparison ===
    print("\n--- Comparison ---")
    diff = abs(cpu_result - gpu_result)
    print(f"CPU: {cpu_result:.10f}")
    print(f"GPU: {gpu_result:.10f}")
    print(f"Diff: {diff:.2e}")

    if diff > 1e-6:
        print("\n⚠️ SIGNIFICANT DIFFERENCE! GPU and CPU objectives don't match!")
    else:
        print("\n✓ GPU and CPU objectives match")

    # Test zero shifts
    print("\n--- Test with zero shifts ---")
    zero_shifts = np.array([0.0, 0.0, 0.0])

    # CPU with zero shifts
    well_cpu2 = deepcopy(well)
    segments_cpu2 = deepcopy(segments)
    for i, shift in enumerate(zero_shifts):
        segments_cpu2[i].end_shift = shift
        if i < len(segments_cpu2) - 1:
            segments_cpu2[i + 1].start_shift = shift
    well_cpu2.calc_horizontal_projection(typewell, segments_cpu2, 0.0)
    cpu_zero = objective_function_optimizer(
        shifts=zero_shifts.tolist(),
        well=well_cpu2,
        typewell=typewell,
        self_corr_start_idx=0,
        segments=segments_cpu2,
        angle_range=10.0,
        angle_sum_power=2.0,
        pearson_power=2.0,
        mse_power=0.001,
        num_intervals_self_correlation=0,
        sc_power=1.15,
        min_pearson_value=-1.0,
        tvd_to_typewell_shift=0.0
    )

    # GPU with zero shifts
    zero_tensor = torch.tensor(zero_shifts, dtype=torch.float64, device='cuda').unsqueeze(0)
    gpu_zero = wrapper(zero_tensor)[0].item()

    print(f"CPU (zero): {cpu_zero:.10f}")
    print(f"GPU (zero): {gpu_zero:.10f}")
    print(f"Diff: {abs(cpu_zero - gpu_zero):.2e}")


if __name__ == "__main__":
    main()
