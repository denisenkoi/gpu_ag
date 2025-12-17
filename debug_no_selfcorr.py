"""
Debug batch vs single without self-correlation.
If results match, the problem is in find_intersections_batch_torch.
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

from torch_funcs.converters import numpy_to_torch, segments_numpy_to_torch
from torch_funcs.objective import objective_function_torch
from torch_funcs.batch_objective import batch_objective_function_torch


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

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
    typewell_np = typewell_to_numpy(typewell)
    segments_np = segments_to_numpy(segments, well)

    typewell_torch = numpy_to_torch(typewell_np, device=device)
    segments_torch = segments_numpy_to_torch(segments_np, device=device)
    shifts_torch = torch.tensor(shifts, dtype=torch.float64, device=device)

    # ===== TEST WITHOUT SELF-CORRELATION =====
    print("\n" + "=" * 60)
    print("TEST WITHOUT SELF-CORRELATION (num_intervals=0)")
    print("=" * 60)

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
        num_intervals_self_correlation=0,  # DISABLED!
        sc_power=params['sc_power'],
        angle_range=params['angle_range'],
        angle_sum_power=params['angle_sum_power'],
        min_pearson_value=params['min_pearson_value'],
        tvd_to_typewell_shift=0.0
    )

    print(f"Single result (no self-corr): {result_single.item()}")

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
        num_intervals_self_correlation=0,  # DISABLED!
        sc_power=params['sc_power'],
        angle_range=params['angle_range'],
        angle_sum_power=params['angle_sum_power'],
        min_pearson_value=params['min_pearson_value'],
        tvd_to_typewell_shift=0.0
    )

    print(f"Batch result[0] (no self-corr): {result_batch[0].item()}")

    diff = abs(result_single.item() - result_batch[0].item())
    print(f"Difference: {diff:.2e}")

    if diff < 1e-10:
        print("\n✅ Without self-correlation: MATCH!")
        print("   -> Problem is in find_intersections_batch_torch")
    else:
        print("\n❌ Still differs without self-correlation")
        print("   -> Problem is elsewhere (MSE, Pearson, etc.)")

    # ===== TEST WITH SELF-CORRELATION =====
    print("\n" + "=" * 60)
    print("TEST WITH SELF-CORRELATION (num_intervals=20)")
    print("=" * 60)

    # Single version
    well_torch_single2 = numpy_to_torch(well_to_numpy(well), device=device)

    result_single2 = objective_function_torch(
        shifts=shifts_torch,
        well_data=well_torch_single2,
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

    print(f"Single result (with self-corr): {result_single2.item()}")

    # Batch version
    well_torch_batch2 = numpy_to_torch(well_to_numpy(well), device=device)

    result_batch2 = batch_objective_function_torch(
        shifts_batch,
        well_torch_batch2,
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

    print(f"Batch result[0] (with self-corr): {result_batch2[0].item()}")

    diff2 = abs(result_single2.item() - result_batch2[0].item())
    print(f"Difference: {diff2:.2e}")


if __name__ == "__main__":
    main()
