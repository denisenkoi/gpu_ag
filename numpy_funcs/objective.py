"""
Numpy implementation of objective function for DE optimization.

This is the main function called ~2M times during optimization.
No Python objects, no deepcopy - just numpy arrays.
"""
import numpy as np
from .converters import numpy_to_segments_data, calc_segment_angles
from .projection import calc_horizontal_projection_numpy
from .correlations import pearson_numpy, mse_numpy
from .self_correlation import find_intersections_numpy


def objective_function_numpy(
    shifts,
    well_data,
    typewell_data,
    segments_data,
    self_corr_start_idx,
    pearson_power,
    mse_power,
    num_intervals_self_correlation,
    sc_power,
    angle_range,
    angle_sum_power,
    min_pearson_value,
    tvd_to_typewell_shift=0.0
):
    """
    Objective function for DE optimization (numpy version).

    Args:
        shifts: array of end_shift values (K,) for each segment
        well_data: dict with well arrays (will be modified: tvt, synt_curve)
        typewell_data: dict with typewell arrays
        segments_data: (K, 6) array of segments
        self_corr_start_idx: starting index for self-correlation
        pearson_power, mse_power: metric weights
        num_intervals_self_correlation: number of intervals for intersection counting
        sc_power: self-correlation power
        angle_range: max allowed angle
        angle_sum_power: power for angle sum penalty
        min_pearson_value: minimum pearson value threshold
        tvd_to_typewell_shift: vertical shift between well and typewell

    Returns:
        float: metric value (lower is better)
    """
    # Update segments with new shifts (replaces deepcopy + update)
    new_segments = numpy_to_segments_data(shifts, segments_data)

    # Calculate segment angles
    angles = calc_segment_angles(new_segments)

    # Angle penalty for exceeding angle_range
    angle_excess = np.abs(angles) - angle_range
    angle_penalty = np.sum(np.where(angle_excess > 0, 1000 * angle_excess ** 2, 0))

    # Angle sum penalty (difference between consecutive angles)
    if len(angles) > 1:
        angle_diffs = np.abs(np.diff(angles))
        angle_sum = np.sum(angle_diffs)
        angle_sum_penalty = angle_sum ** angle_sum_power
    else:
        angle_sum_penalty = 0

    # Calculate horizontal projection (fills tvt and synt_curve)
    # Make copies of arrays to avoid modifying originals
    well_data_copy = {
        'md': well_data['md'],
        'vs': well_data['vs'],
        'tvd': well_data['tvd'],
        'value': well_data['value'],
        'tvt': well_data['tvt'].copy(),
        'synt_curve': well_data['synt_curve'].copy(),
        'md_range': well_data['md_range'],
        'min_curve': well_data['min_curve'],
        'max_curve': well_data['max_curve'],
        'horizontal_well_step': well_data['horizontal_well_step'],
        'normalized': well_data['normalized'],
    }

    success, well_data_copy = calc_horizontal_projection_numpy(
        well_data_copy, typewell_data, new_segments, tvd_to_typewell_shift
    )

    if not success:
        return float('inf')

    # Get indices range
    start_idx = int(new_segments[0, 0])
    end_idx = int(new_segments[-1, 1])

    # Check for NaN in synthetic curve
    synt_slice = well_data_copy['synt_curve'][start_idx:end_idx + 1]
    if np.any(np.isnan(synt_slice)):
        return float('inf')

    # Calculate MSE
    x = well_data_copy['value'][start_idx:end_idx + 1]
    y = synt_slice
    mse = mse_numpy(x, y)

    # Calculate Pearson correlation
    const_x = np.all(x == x[0])
    const_y = np.all(y == y[0])

    if not const_x and not const_y:
        pearson = pearson_numpy(x, y)
    else:
        pearson = 0

    pearson = max(pearson, min_pearson_value)

    # Calculate intersections if needed
    if num_intervals_self_correlation > 0:
        tvt_slice = well_data_copy['tvt'][self_corr_start_idx:end_idx + 1]
        value_slice = well_data_copy['value'][self_corr_start_idx:end_idx + 1]

        intersections_count = find_intersections_numpy(
            tvt_slice,
            value_slice,
            well_data_copy['min_curve'],
            well_data_copy['max_curve'],
            well_data_copy['md_range'],
            num_intervals_self_correlation,
            start_idx=start_idx - self_corr_start_idx
        )
        intersections_component = sc_power ** intersections_count
    else:
        intersections_component = 1.0

    # Combine metrics (minimize)
    pearson_component = 1 - pearson

    metric = (
        (pearson_component ** pearson_power) *
        (mse ** mse_power) *
        (1.0 / intersections_component)
    ) * (1 + angle_penalty + angle_sum_penalty)

    return metric
