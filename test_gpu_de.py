"""
Test GPU Differential Evolution optimizer.
Runs full DE optimization and compares with CPU baseline.
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
from torch_funcs.batch_objective import TorchObjectiveWrapper
from torch_funcs.gpu_optimizer import gpu_optimizer_fit


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

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

    well = Well(well_data_json)
    typewell = TypeWell(well_data_json)

    interp_path = starsteer_dir / "interpretation.json"
    with open(interp_path) as f:
        interp_data = json.load(f)

    segments_raw = interp_data["interpretation"]["segments"][-4:]
    segments = create_segments_from_json(segments_raw, well)
    shifts = [seg['endShift'] for seg in segments_raw]

    # Convert to numpy
    well_np = well_to_numpy(well)
    typewell_np = typewell_to_numpy(typewell)
    segments_np = segments_to_numpy(segments, well)

    # Convert to torch
    well_torch = numpy_to_torch(well_np, device=device)
    typewell_torch = numpy_to_torch(typewell_np, device=device)
    segments_torch = segments_numpy_to_torch(segments_np, device=device)

    # Calculate bounds (same as CPU: angle_range based)
    angle_range = params['angle_range']
    K = len(segments)
    bounds_list = []
    for i, seg in enumerate(segments):
        segment_len = seg.end_vs - seg.start_vs
        max_shift = segment_len * well.horizontal_well_step * abs(torch.tan(torch.tensor(angle_range * 3.14159 / 180)))
        current_shift = shifts[i]
        bounds_list.append([current_shift - max_shift.item(), current_shift + max_shift.item()])

    bounds = torch.tensor(bounds_list, dtype=torch.float64, device=device)

    print("\n" + "=" * 60)
    print("GPU DIFFERENTIAL EVOLUTION TEST")
    print("=" * 60)
    print(f"Segments: {K}")
    print(f"Initial shifts: {shifts}")
    print(f"Bounds:\n{bounds.cpu().numpy()}")

    # Create objective wrapper
    wrapper = TorchObjectiveWrapper(
        well_data=well_torch,
        typewell_data=typewell_torch,
        segments_torch=segments_torch,
        self_corr_start_idx=0,
        pearson_power=params['pearson_power'],
        mse_power=params['mse_power'],
        num_intervals_self_correlation=0,  # Disabled for speed
        sc_power=params['sc_power'],
        angle_range=params['angle_range'],
        angle_sum_power=params['angle_sum_power'],
        min_pearson_value=params['min_pearson_value'],
        tvd_to_typewell_shift=0.0,
        device=device
    )

    # Test single evaluation first
    print("\nTesting single evaluation...")
    shifts_tensor = torch.tensor(shifts, dtype=torch.float64, device=device)
    single_result = wrapper.evaluate_single(shifts_tensor)
    print(f"Initial fitness: {single_result.item():.6f}")

    # Run GPU DE with reduced iterations for testing
    print("\n" + "=" * 60)
    print("Running GPU DE (100 iterations for quick test)...")
    print("=" * 60)

    result = gpu_optimizer_fit(
        objective_wrapper=wrapper,
        bounds=bounds,
        popsize=500,
        maxiter=100,  # Reduced for testing
        mutation=(1.5, 1.99),
        recombination=0.99,
        seed=42,
        verbose=True,
        device=device
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Best fitness: {result['fun']:.6f}")
    print(f"Best shifts: {result['x'].cpu().numpy()}")
    print(f"Iterations: {result['nit']}")
    print(f"Function evals: {result['nfev']}")
    print(f"Time: {result['elapsed_time']:.2f} sec")

    # Compare with reference
    print("\n" + "=" * 60)
    print("COMPARISON WITH REFERENCE")
    print("=" * 60)
    print(f"CPU baseline (1000 iter): ~330 sec")
    print(f"GPU (100 iter): {result['elapsed_time']:.2f} sec")
    print(f"Estimated GPU (1000 iter): {result['elapsed_time'] * 10:.2f} sec")
    print(f"Estimated speedup: {330 / (result['elapsed_time'] * 10):.1f}x")

    # Reference shifts from AGENT_INSTRUCTIONS.md
    ref_shifts = [-0.00852, -0.00897, -0.00949, -0.01001]
    print(f"\nReference optimal shifts: {ref_shifts}")
    print(f"GPU optimal shifts: {result['x'].cpu().numpy()}")


if __name__ == "__main__":
    main()
