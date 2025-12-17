"""
Debug batch vs single objective function.
Compare intermediate values to find where they diverge.
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))
sys.path.insert(0, str(Path(__file__).parent))

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json

from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy

from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch, update_segments_with_shifts_torch
from torch_funcs.projection import calc_horizontal_projection_torch, calc_horizontal_projection_batch_torch


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

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

    # Convert to torch
    typewell_torch = numpy_to_torch(typewell_np, device=device)
    segments_torch = segments_numpy_to_torch(segments_np, device=device)
    shifts_torch = torch.tensor(shifts, dtype=torch.float64, device=device)

    print(f"\\nSegments shape: {segments_torch.shape}")
    print(f"Shifts: {shifts}")

    # ===== SINGLE VERSION =====
    print("\\n" + "="*60)
    print("SINGLE VERSION")
    print("="*60)

    well_torch_single = numpy_to_torch(well_to_numpy(well), device=device)

    # Update segments with shifts
    new_segments_single = update_segments_with_shifts_torch(shifts_torch, segments_torch)
    print(f"\\nUpdated segments (single):")
    print(new_segments_single)

    # Run projection
    success_single, well_torch_single = calc_horizontal_projection_torch(
        well_torch_single, typewell_torch, new_segments_single, tvd_to_typewell_shift=0.0
    )

    start_idx = int(new_segments_single[0, 0].item())
    end_idx = int(new_segments_single[-1, 1].item())

    tvt_single = well_torch_single['tvt'][start_idx:end_idx + 1]
    synt_single = well_torch_single['synt_curve'][start_idx:end_idx + 1]

    print(f"\\nSuccess: {success_single}")
    print(f"Indices: {start_idx} to {end_idx}")
    print(f"TVT[0:5]: {tvt_single[:5].tolist()}")
    print(f"SYNT[0:5]: {synt_single[:5].tolist()}")

    # ===== BATCH VERSION (single item) =====
    print("\\n" + "="*60)
    print("BATCH VERSION (single item)")
    print("="*60)

    well_torch_batch = numpy_to_torch(well_to_numpy(well), device=device)

    # Create batch of 1
    shifts_batch = shifts_torch.unsqueeze(0)  # (1, K)

    # Update segments for batch
    new_segments_batch = update_segments_with_shifts_torch(shifts_batch, segments_torch)
    print(f"\\nUpdated segments (batch[0]):")
    print(new_segments_batch[0])

    # Compare segments
    seg_diff = torch.max(torch.abs(new_segments_single - new_segments_batch[0]))
    print(f"\\nSegments difference: {seg_diff.item():.2e}")

    # Run batch projection
    success_batch, tvt_batch, synt_batch, first_idx = calc_horizontal_projection_batch_torch(
        well_torch_batch, typewell_torch, new_segments_batch, tvd_to_typewell_shift=0.0
    )

    print(f"\\nBatch success[0]: {success_batch[0].item()}")
    print(f"First idx: {first_idx}")
    print(f"TVT batch shape: {tvt_batch.shape}")
    print(f"TVT batch[0, 0:5]: {tvt_batch[0, :5].tolist()}")
    print(f"SYNT batch[0, 0:5]: {synt_batch[0, :5].tolist()}")

    # ===== COMPARISON =====
    print("\\n" + "="*60)
    print("COMPARISON")
    print("="*60)

    tvt_diff = torch.max(torch.abs(tvt_single - tvt_batch[0]))
    synt_diff = torch.max(torch.abs(synt_single - synt_batch[0]))

    print(f"TVT max diff: {tvt_diff.item():.2e}")
    print(f"SYNT max diff: {synt_diff.item():.2e}")

    if tvt_diff < 1e-10:
        print("\\n✅ TVT matches!")
    else:
        print("\\n❌ TVT differs!")
        # Find where they differ
        diff_mask = torch.abs(tvt_single - tvt_batch[0]) > 1e-10
        diff_indices = torch.where(diff_mask)[0]
        print(f"Differing indices: {diff_indices[:10].tolist()}")
        for idx in diff_indices[:3]:
            print(f"  idx {idx}: single={tvt_single[idx].item():.6f}, batch={tvt_batch[0, idx].item():.6f}")


def debug_objective():
    """Debug full objective function single vs batch."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
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
    well_np = well_to_numpy(well)
    typewell_np = typewell_to_numpy(typewell)
    segments_np = segments_to_numpy(segments, well)

    typewell_torch = numpy_to_torch(typewell_np, device=device)
    segments_torch = segments_numpy_to_torch(segments_np, device=device)
    shifts_torch = torch.tensor(shifts, dtype=torch.float64, device=device)

    # Import objective functions
    from torch_funcs.objective import objective_function_torch
    from torch_funcs.batch_objective import batch_objective_function_torch
    from torch_funcs.correlations import pearson_torch, mse_torch, pearson_batch_torch, mse_batch_torch

    print("\\n" + "="*60)
    print("DEBUG OBJECTIVE FUNCTION")
    print("="*60)

    # Single version
    well_torch_single = numpy_to_torch(well_to_numpy(well), device=device)

    result_single = objective_function_torch(
        shifts=shifts_torch,
        well_data=well_torch_single,
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

    print(f"Single result: {result_single.item()}")

    # Batch version
    well_torch_batch = numpy_to_torch(well_to_numpy(well), device=device)
    shifts_batch = shifts_torch.unsqueeze(0)

    result_batch = batch_objective_function_torch(
        shifts_batch,
        well_torch_batch,
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

    print(f"Batch result[0]: {result_batch[0].item()}")
    print(f"Difference: {abs(result_single.item() - result_batch[0].item()):.2e}")

    # Manual comparison of intermediate values
    print("\\n--- Manual intermediate check ---")

    # Get projections
    from torch_funcs.converters import update_segments_with_shifts_torch, calc_segment_angles_torch

    new_segments = update_segments_with_shifts_torch(shifts_torch, segments_torch)
    start_idx = int(new_segments[0, 0].item())
    end_idx = int(new_segments[-1, 1].item())

    # Single: get value and synt_curve
    x_single = well_torch_single['value'][start_idx:end_idx + 1]
    y_single = well_torch_single['synt_curve'][start_idx:end_idx + 1]

    mse_single = mse_torch(x_single, y_single)
    pearson_single = pearson_torch(x_single, y_single)

    print(f"Single MSE: {mse_single.item()}")
    print(f"Single Pearson: {pearson_single.item()}")

    # Batch: need to re-run projection to get synt_curve
    well_torch_batch2 = numpy_to_torch(well_to_numpy(well), device=device)
    new_segments_batch = update_segments_with_shifts_torch(shifts_batch, segments_torch)

    success_batch, tvt_batch, synt_batch, first_idx = calc_horizontal_projection_batch_torch(
        well_torch_batch2, typewell_torch, new_segments_batch, tvd_to_typewell_shift=0.0
    )

    value_batch = well_torch_batch2['value'][first_idx:first_idx + synt_batch.shape[1]]

    mse_batch_val = mse_batch_torch(value_batch.unsqueeze(0), synt_batch)
    pearson_batch_val = pearson_batch_torch(value_batch.unsqueeze(0), synt_batch)

    print(f"Batch MSE[0]: {mse_batch_val[0].item()}")
    print(f"Batch Pearson[0]: {pearson_batch_val[0].item()}")

    # Value comparison
    print(f"\\nValue slice [0:5]: {x_single[:5].tolist()}")
    print(f"Value batch [0:5]: {value_batch[:5].tolist()}")
    print(f"Synt single [0:5]: {y_single[:5].tolist()}")
    print(f"Synt batch [0:5]: {synt_batch[0, :5].tolist()}")


if __name__ == "__main__":
    main()
    debug_objective()
