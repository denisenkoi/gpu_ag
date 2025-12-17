"""
Test checkpoint: calculate reference values for numpy refactoring validation.
Loads data from AG_DATA/InitialData/slicing_well.json (same as emulator).
"""
import sys
import json
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json
from ag_rewards.ag_func_correlations import objective_function_optimizer

def main():
    starsteer_dir = Path("E:/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de")

    # Load well data from AG_DATA (same as emulator)
    well_json_path = starsteer_dir / "AG_DATA" / "InitialData" / "slicing_well.json"

    print(f"Loading well data from: {well_json_path}")
    with open(well_json_path, 'r', encoding='utf-8') as f:
        well_data = json.load(f)

    # Create Well and TypeWell objects (same as emulator.py line 1154-1155)
    well = Well(well_data)
    typewell = TypeWell(well_data)

    print(f"Well: MD range {well.measured_depth.min():.1f} - {well.measured_depth.max():.1f}")
    print(f"TypeWell: TVD range {typewell.tvd.min():.1f} - {typewell.tvd.max():.1f}")

    # Load current interpretation from StarSteer
    interp_path = starsteer_dir / "interpretation.json"
    with open(interp_path) as f:
        interp_data = json.load(f)

    segments_raw = interp_data["interpretation"]["segments"]
    print(f"\nLoaded {len(segments_raw)} segments from interpretation.json")

    # Take last 4 segments for test (like DE optimization)
    test_segments_raw = segments_raw[-4:]
    print(f"\nTest segments (last 4):")
    for i, seg in enumerate(test_segments_raw):
        print(f"  {i}: MD {seg['startMd']:.1f}-{seg['endMd']:.1f}, "
              f"shift {seg['startShift']:.4f}->{seg['endShift']:.4f}")

    # Create Segment objects
    segments = create_segments_from_json(test_segments_raw, well)
    print(f"\nCreated {len(segments)} Segment objects")

    # Extract shifts (endShift values - these are what DE optimizes)
    shifts = [seg['endShift'] for seg in test_segments_raw]
    print(f"Shifts for objective function: {shifts}")

    # Calculate objective function
    print("\n" + "="*60)
    print("CHECKPOINT VALUES FOR NUMPY REFACTORING")
    print("="*60)

    # Parameters from python_autogeosteering_executor.py defaults
    self_corr_start_idx = 0  # Start from beginning for this test
    pearson_power = 2.0
    mse_power = 0.001
    num_intervals_self_correlation = 20
    sc_power = 1.15
    angle_range = 10.0
    angle_sum_power = 2.0
    min_pearson_value = -1

    result = objective_function_optimizer(
        shifts=shifts,
        well=well,
        typewell=typewell,
        self_corr_start_idx=self_corr_start_idx,
        segments=segments,
        pearson_power=pearson_power,
        mse_power=mse_power,
        num_intervals_self_correlation=num_intervals_self_correlation,
        sc_power=sc_power,
        angle_range=angle_range,
        angle_sum_power=angle_sum_power,
        min_pearson_value=min_pearson_value
    )

    print(f"\nobjective_function_optimizer result: {result}")

    # Get intermediate results (tvt and synt_curve are filled by objective_function)
    # We need to find the indices for our segments
    start_idx = segments[0].start_idx
    end_idx = segments[-1].end_idx

    print(f"\nIntermediate results for indices {start_idx} to {end_idx}:")
    print(f"  well.tvt range: {well.tvt[start_idx]:.4f} to {well.tvt[end_idx]:.4f}")
    print(f"  well.synt_curve range: {well.synt_curve[start_idx]:.4f} to {well.synt_curve[end_idx]:.4f}")
    print(f"  well.value range: {well.value[start_idx]:.4f} to {well.value[end_idx]:.4f}")

    # Extract arrays for checkpoint (convert to list for JSON)
    tvt_slice = well.tvt[start_idx:end_idx+1].tolist()
    synt_curve_slice = well.synt_curve[start_idx:end_idx+1].tolist()
    value_slice = well.value[start_idx:end_idx+1].tolist()
    md_slice = well.measured_depth[start_idx:end_idx+1].tolist()

    print(f"  Array length: {len(tvt_slice)} points")

    # Save checkpoint values
    checkpoint = {
        "test_description": "Last 4 segments from current interpretation",
        "shifts": shifts,
        "objective_function_result": float(result),
        "parameters": {
            "pearson_power": pearson_power,
            "mse_power": mse_power,
            "num_intervals_self_correlation": num_intervals_self_correlation,
            "sc_power": sc_power,
            "angle_range": angle_range,
            "angle_sum_power": angle_sum_power,
            "min_pearson_value": min_pearson_value,
        },
        "segments_count": len(segments),
        "segment_indices": {"start_idx": int(start_idx), "end_idx": int(end_idx)},
        "well_md_range": [float(well.measured_depth.min()), float(well.measured_depth.max())],
        "typewell_tvd_range": [float(typewell.tvd.min()), float(typewell.tvd.max())],
        "intermediate_results": {
            "md": md_slice,
            "tvt": tvt_slice,
            "synt_curve": synt_curve_slice,
            "value": value_slice,
        }
    }

    checkpoint_path = Path(__file__).parent / "test_checkpoint_values.json"
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2)

    print(f"\nCheckpoint saved to: {checkpoint_path}")

if __name__ == "__main__":
    main()
