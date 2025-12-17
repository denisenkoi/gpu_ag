"""
Test numpy implementation against checkpoint values.

Validates:
1. tvt values match checkpoint (124 points)
2. synt_curve values match checkpoint
3. objective_function result matches: 0.00042138593474439547
"""
import sys
import json
import numpy as np
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))
sys.path.insert(0, str(Path(__file__).parent))

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json

from numpy_funcs.converters import well_to_numpy, typewell_to_numpy, segments_to_numpy
from numpy_funcs.projection import calc_horizontal_projection_numpy
from numpy_funcs.objective import objective_function_numpy


def load_checkpoint():
    """Load checkpoint values from JSON"""
    checkpoint_path = Path(__file__).parent / "cpu_baseline" / "test_checkpoint_values.json"
    with open(checkpoint_path, 'r') as f:
        return json.load(f)


def main():
    print("=" * 60)
    print("NUMPY IMPLEMENTATION VALIDATION")
    print("=" * 60)

    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"\nCheckpoint loaded:")
    print(f"  objective_function_result: {checkpoint['objective_function_result']}")
    print(f"  shifts: {checkpoint['shifts']}")
    print(f"  segment_indices: {checkpoint['segment_indices']}")

    # Load data (same as test_checkpoint.py)
    starsteer_dir = Path("E:/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")
    well_json_path = starsteer_dir / "AG_DATA" / "InitialData" / "slicing_well.json"

    print(f"\nLoading data from: {well_json_path}")
    with open(well_json_path, 'r', encoding='utf-8') as f:
        well_data_json = json.load(f)

    # Create original objects
    well = Well(well_data_json)
    typewell = TypeWell(well_data_json)

    print(f"Well: MD range {well.measured_depth.min():.1f} - {well.measured_depth.max():.1f}")
    print(f"TypeWell: TVD range {typewell.tvd.min():.1f} - {typewell.tvd.max():.1f}")

    # Load interpretation
    interp_path = starsteer_dir / "interpretation.json"
    with open(interp_path) as f:
        interp_data = json.load(f)

    segments_raw = interp_data["interpretation"]["segments"][-4:]
    segments = create_segments_from_json(segments_raw, well)

    shifts = [seg['endShift'] for seg in segments_raw]

    # Convert to numpy
    print("\n" + "=" * 60)
    print("CHECKPOINT 2.1: CONVERTERS")
    print("=" * 60)

    well_np = well_to_numpy(well)
    typewell_np = typewell_to_numpy(typewell)
    segments_np = segments_to_numpy(segments, well)

    print(f"well_np keys: {list(well_np.keys())}")
    print(f"typewell_np keys: {list(typewell_np.keys())}")
    print(f"segments_np shape: {segments_np.shape}")
    print(f"segments_np:\n{segments_np}")

    # Test projection
    print("\n" + "=" * 60)
    print("CHECKPOINT 2.2: PROJECTION (tvt, synt_curve)")
    print("=" * 60)

    success, well_np = calc_horizontal_projection_numpy(
        well_np, typewell_np, segments_np, tvd_to_typewell_shift=0.0
    )

    print(f"Projection success: {success}")

    # Compare with checkpoint
    start_idx = checkpoint['segment_indices']['start_idx']
    end_idx = checkpoint['segment_indices']['end_idx']

    tvt_numpy = well_np['tvt'][start_idx:end_idx + 1]
    tvt_checkpoint = np.array(checkpoint['intermediate_results']['tvt'])

    synt_numpy = well_np['synt_curve'][start_idx:end_idx + 1]
    synt_checkpoint = np.array(checkpoint['intermediate_results']['synt_curve'])

    tvt_diff = np.max(np.abs(tvt_numpy - tvt_checkpoint))
    synt_diff = np.max(np.abs(synt_numpy - synt_checkpoint))

    print(f"\nTVT comparison:")
    print(f"  Max difference: {tvt_diff:.2e}")
    print(f"  First 5 numpy:      {tvt_numpy[:5]}")
    print(f"  First 5 checkpoint: {tvt_checkpoint[:5]}")

    print(f"\nSYNT_CURVE comparison:")
    print(f"  Max difference: {synt_diff:.2e}")
    print(f"  First 5 numpy:      {synt_numpy[:5]}")
    print(f"  First 5 checkpoint: {synt_checkpoint[:5]}")

    if tvt_diff < 1e-10:
        print("\n✅ TVT VALIDATION PASSED")
    else:
        print(f"\n❌ TVT VALIDATION FAILED (diff={tvt_diff:.2e})")

    if synt_diff < 1e-10:
        print("✅ SYNT_CURVE VALIDATION PASSED")
    else:
        print(f"❌ SYNT_CURVE VALIDATION FAILED (diff={synt_diff:.2e})")

    # Test objective function
    print("\n" + "=" * 60)
    print("CHECKPOINT 2.3: OBJECTIVE FUNCTION")
    print("=" * 60)

    # Reset well_np arrays for fresh calculation
    well_np['tvt'] = np.empty(len(well_np['md']))
    well_np['tvt'][:] = np.nan
    well_np['synt_curve'] = np.empty(len(well_np['md']))
    well_np['synt_curve'][:] = np.nan

    # Parameters from test_checkpoint.py
    params = checkpoint['parameters']

    result_numpy = objective_function_numpy(
        shifts=np.array(shifts),
        well_data=well_np,
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

    result_checkpoint = checkpoint['objective_function_result']

    print(f"\nObjective function comparison:")
    print(f"  Numpy result:      {result_numpy}")
    print(f"  Checkpoint result: {result_checkpoint}")
    print(f"  Difference:        {abs(result_numpy - result_checkpoint):.2e}")

    if abs(result_numpy - result_checkpoint) < 1e-10:
        print("\n✅ OBJECTIVE FUNCTION VALIDATION PASSED")
    else:
        print(f"\n❌ OBJECTIVE FUNCTION VALIDATION FAILED")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    all_passed = (
        tvt_diff < 1e-10 and
        synt_diff < 1e-10 and
        abs(result_numpy - result_checkpoint) < 1e-10
    )

    if all_passed:
        print("✅ ALL CHECKPOINTS PASSED - Numpy implementation matches CPU baseline")
    else:
        print("❌ SOME CHECKPOINTS FAILED - Review implementation")


if __name__ == "__main__":
    main()
