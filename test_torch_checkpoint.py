"""
Test PyTorch implementation against numpy checkpoint.

Validates:
1. Single evaluation matches numpy result
2. Batch evaluation produces consistent results
3. GPU acceleration works (if available)
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))
sys.path.insert(0, str(Path(__file__).parent))

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json

from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
from numpy_funcs.objective import objective_function_numpy

from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
from torch_funcs.projection import calc_horizontal_projection_torch
from torch_funcs.objective import objective_function_torch
from torch_funcs.batch_objective import batch_objective_function_torch, TorchObjectiveWrapper


def load_checkpoint():
    checkpoint_path = Path(__file__).parent / "cpu_baseline" / "test_checkpoint_values.json"
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("PYTORCH IMPLEMENTATION VALIDATION")
    print("=" * 60)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    print(f"\nCUDA available: {cuda_available}")
    print(f"Using device: {device}")

    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"\nCheckpoint reference: {checkpoint['objective_function_result']}")

    # Load data
    starsteer_dir = Path("E:/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")
    well_json_path = starsteer_dir / "AG_DATA" / "InitialData" / "slicing_well.json"

    with open(well_json_path, 'r', encoding='utf-8') as f:
        well_data_json = json.load(f)

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

    # Get numpy result for reference
    well_np_copy = well_to_numpy(well)  # Fresh copy
    params = checkpoint['parameters']

    result_numpy = objective_function_numpy(
        shifts=np.array(shifts),
        well_data=well_np_copy,
        typewell_data=typewell_np,
        segments_data=segments_np,
        self_corr_start_idx=0,
        pearson_power=params['pearson_power'],
        mse_power=params['mse_power'],
        num_intervals_self_correlation=params['num_intervals_self_correlation'],
        sc_power=params['sc_power'],
        angle_range=params['angle_range'],
        angle_sum_power=params['angle_sum_power'],
        min_pearson_value=params['min_pearson_value'],
        tvd_to_typewell_shift=0.0
    )

    print(f"Numpy reference: {result_numpy}")

    # Test Checkpoint 3.1: Torch data structures
    print("\n" + "=" * 60)
    print("CHECKPOINT 3.1: TORCH DATA STRUCTURES")
    print("=" * 60)

    well_torch = numpy_to_torch(well_to_numpy(well), device=device)
    typewell_torch = numpy_to_torch(typewell_np, device=device)
    segments_torch = segments_numpy_to_torch(segments_np, device=device)

    print(f"well_torch['md'] device: {well_torch['md'].device}")
    print(f"well_torch['md'] dtype: {well_torch['md'].dtype}")
    print(f"segments_torch shape: {segments_torch.shape}")
    print("✅ Torch data structures created")

    # Test Checkpoint 3.2: Torch projection
    print("\n" + "=" * 60)
    print("CHECKPOINT 3.2: TORCH PROJECTION")
    print("=" * 60)

    success, well_torch = calc_horizontal_projection_torch(
        well_torch, typewell_torch, segments_torch, tvd_to_typewell_shift=0.0
    )

    print(f"Projection success: {success}")

    # Compare with numpy
    start_idx = checkpoint['segment_indices']['start_idx']
    end_idx = checkpoint['segment_indices']['end_idx']

    tvt_torch_list = well_torch['tvt'][start_idx:end_idx + 1].detach().cpu().tolist()
    tvt_torch = np.array(tvt_torch_list)
    tvt_checkpoint = np.array(checkpoint['intermediate_results']['tvt'])

    tvt_diff = np.max(np.abs(tvt_torch - tvt_checkpoint))
    print(f"TVT max diff vs checkpoint: {tvt_diff:.2e}")

    if tvt_diff < 1e-9:
        print("✅ TORCH PROJECTION VALIDATION PASSED")
    else:
        print(f"❌ TORCH PROJECTION VALIDATION FAILED")

    # Test Checkpoint 3.3: Torch objective function (single)
    print("\n" + "=" * 60)
    print("CHECKPOINT 3.3: TORCH OBJECTIVE FUNCTION (SINGLE)")
    print("=" * 60)

    # Reset well data
    well_torch = numpy_to_torch(well_to_numpy(well), device=device)

    shifts_torch = torch.tensor(shifts, dtype=torch.float64, device=device)

    result_torch = objective_function_torch(
        shifts=shifts_torch,
        well_data=well_torch,
        typewell_data=typewell_torch,
        segments_torch=segments_torch,
        self_corr_start_idx=0,
        pearson_power=params['pearson_power'],
        mse_power=params['mse_power'],
        num_intervals_self_correlation=params['num_intervals_self_correlation'],
        sc_power=params['sc_power'],
        angle_range=params['angle_range'],
        angle_sum_power=params['angle_sum_power'],
        min_pearson_value=params['min_pearson_value'],
        tvd_to_typewell_shift=0.0
    )

    result_torch_val = result_torch.item()
    diff_vs_numpy = abs(result_torch_val - result_numpy)

    print(f"Torch result:  {result_torch_val}")
    print(f"Numpy result:  {result_numpy}")
    print(f"Difference:    {diff_vs_numpy:.2e}")

    if diff_vs_numpy < 1e-6:
        print("✅ TORCH OBJECTIVE (SINGLE) VALIDATION PASSED")
    else:
        print(f"❌ TORCH OBJECTIVE (SINGLE) VALIDATION FAILED")

    # Test Checkpoint 3.4: Batch processing
    print("\n" + "=" * 60)
    print("CHECKPOINT 3.4: BATCH PROCESSING")
    print("=" * 60)

    # Create batch of 500 with small perturbations
    batch_size = 500
    shifts_base = torch.tensor(shifts, dtype=torch.float64, device=device)

    # Add small random perturbations
    torch.manual_seed(42)
    perturbations = torch.randn(batch_size, len(shifts), dtype=torch.float64, device=device) * 0.1
    shifts_batch = shifts_base.unsqueeze(0) + perturbations

    # First row is exact original
    shifts_batch[0] = shifts_base

    # Reset well data
    well_torch = numpy_to_torch(well_to_numpy(well), device=device)

    # Time batch evaluation
    import time
    torch.cuda.synchronize() if cuda_available else None
    start_time = time.time()

    results_batch = batch_objective_function_torch(
        shifts_batch,
        well_torch,
        typewell_torch,
        segments_torch,
        self_corr_start_idx=0,
        pearson_power=params['pearson_power'],
        mse_power=params['mse_power'],
        num_intervals_self_correlation=params['num_intervals_self_correlation'],
        sc_power=params['sc_power'],
        angle_range=params['angle_range'],
        angle_sum_power=params['angle_sum_power'],
        min_pearson_value=params['min_pearson_value'],
        tvd_to_typewell_shift=0.0
    )

    torch.cuda.synchronize() if cuda_available else None
    batch_time = time.time() - start_time

    print(f"Batch size: {batch_size}")
    print(f"Batch time: {batch_time:.4f} sec")
    print(f"Time per evaluation: {batch_time / batch_size * 1000:.4f} ms")

    # Check first result matches single evaluation
    first_result = results_batch[0].item()
    diff_first = abs(first_result - result_torch_val)

    print(f"\nFirst batch result: {first_result}")
    print(f"Single result:      {result_torch_val}")
    print(f"Difference:         {diff_first:.2e}")

    # Statistics
    valid_mask = ~torch.isinf(results_batch)
    valid_count = torch.sum(valid_mask).item()
    print(f"\nValid results: {valid_count}/{batch_size}")

    if valid_count > 0:
        valid_results = results_batch[valid_mask]
        print(f"Min: {torch.min(valid_results).item():.6f}")
        print(f"Max: {torch.max(valid_results).item():.6f}")
        print(f"Mean: {torch.mean(valid_results).item():.6f}")

    if diff_first < 1e-4:  # Looser tolerance for batch due to different code path
        print("\n✅ BATCH PROCESSING VALIDATION PASSED")
    else:
        print("\n❌ BATCH PROCESSING VALIDATION FAILED")

    # Test wrapper class
    print("\n" + "=" * 60)
    print("TESTING WRAPPER CLASS")
    print("=" * 60)

    wrapper = TorchObjectiveWrapper(
        well_data=well_to_numpy(well),
        typewell_data=typewell_np,
        segments_torch=segments_np,
        self_corr_start_idx=0,
        pearson_power=params['pearson_power'],
        mse_power=params['mse_power'],
        num_intervals_self_correlation=params['num_intervals_self_correlation'],
        sc_power=params['sc_power'],
        angle_range=params['angle_range'],
        angle_sum_power=params['angle_sum_power'],
        min_pearson_value=params['min_pearson_value'],
        tvd_to_typewell_shift=0.0,
        device=device
    )

    wrapper_result = wrapper.evaluate_single(shifts)
    print(f"Wrapper single result: {wrapper_result.item()}")
    print("✅ Wrapper class works")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = (
        tvt_diff < 1e-9 and
        diff_vs_numpy < 1e-6 and
        diff_first < 1e-4
    )

    if all_passed:
        print("✅ ALL TORCH CHECKPOINTS PASSED")
    else:
        print("❌ SOME TORCH CHECKPOINTS FAILED")

    print(f"\nDevice: {device}")
    print(f"Batch (500) time: {batch_time:.4f} sec")


if __name__ == "__main__":
    main()
