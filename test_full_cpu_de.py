"""
Test full CPU scipy DE with the same data.
"""
import sys
import json
import numpy as np
from pathlib import Path
from copy import deepcopy
from scipy.optimize import differential_evolution

sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json
from ag_rewards.ag_func_correlations import objective_function_optimizer
from ag_numerical.ag_optimizer_utils import calculate_optimization_bounds


def main():
    print("="*60)
    print("Full CPU scipy DE Test")
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

    # Calculate bounds
    bounds = calculate_optimization_bounds(segments, angle_range=10.0, accumulative=True)
    print(f"Bounds: {bounds}")

    # Initial projection
    well.calc_horizontal_projection(typewell, segments, 0.0)

    # Store initial state
    well_base = deepcopy(well)
    segments_base = deepcopy(segments)

    eval_count = [0]

    def cpu_objective(shifts):
        eval_count[0] += 1
        return objective_function_optimizer(
            shifts=list(shifts),
            well=deepcopy(well_base),
            typewell=typewell,
            self_corr_start_idx=0,
            segments=deepcopy(segments_base),
            angle_range=10.0,
            angle_sum_power=2.0,
            pearson_power=2.0,
            mse_power=0.001,
            num_intervals_self_correlation=0,
            sc_power=1.15,
            min_pearson_value=-1.0,
            tvd_to_typewell_shift=0.0
        )

    import time
    start_time = time.time()

    # Run scipy DE with large popsize
    result = differential_evolution(
        cpu_objective,
        bounds,
        strategy='rand1bin',
        maxiter=500,
        popsize=100,  # Much larger
        mutation=(0.5, 1.0),
        recombination=0.7,
        seed=42,
        disp=False,
        workers=1,
        updating='deferred'
    )

    elapsed = time.time() - start_time

    print(f"\n--- Results ---")
    print(f"best_fun: {result.fun:.6f}")
    print(f"best_x: {result.x}")
    print(f"nit: {result.nit}")
    print(f"nfev: {eval_count[0]}")
    print(f"time: {elapsed:.2f} sec")

    # Test initial interpretation shifts
    initial_shifts = np.array([-5.663609951795427, -9.704093997855415, -12.38605218676823])
    initial_fun = cpu_objective(initial_shifts)
    print(f"\nInitial interpretation: fun={initial_fun:.6f}")


if __name__ == "__main__":
    main()
