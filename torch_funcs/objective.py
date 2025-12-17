"""
PyTorch implementation of objective function.
Single evaluation version for validation.
"""
import torch
from .converters import update_segments_with_shifts_torch, calc_segment_angles_torch
from .projection import calc_horizontal_projection_torch
from .correlations import pearson_torch, mse_torch
from .self_correlation import find_intersections_torch


def objective_function_torch(
    shifts,
    well_data,
    typewell_data,
    segments_torch,
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
    Objective function for DE optimization (torch version, single evaluation).

    Args:
        shifts: tensor of end_shift values (K,)
        well_data: dict with well tensors
        typewell_data: dict with typewell tensors
        segments_torch: (K, 6) tensor of segments
        self_corr_start_idx: starting index for self-correlation
        pearson_power, mse_power: metric weights
        num_intervals_self_correlation: number of intervals
        sc_power: self-correlation power
        angle_range: max allowed angle
        angle_sum_power: power for angle sum penalty
        min_pearson_value: minimum pearson threshold
        tvd_to_typewell_shift: vertical shift

    Returns:
        tensor: metric value (lower is better)
    """
    device = shifts.device
    dtype = shifts.dtype

    # Update segments with new shifts
    new_segments = update_segments_with_shifts_torch(shifts, segments_torch)

    # Calculate segment angles
    angles = calc_segment_angles_torch(new_segments)

    # Angle penalty
    angle_excess = torch.abs(angles) - angle_range
    angle_penalty = torch.sum(torch.where(angle_excess > 0, 1000 * angle_excess ** 2, torch.zeros_like(angle_excess)))

    # Angle sum penalty
    if len(angles) > 1:
        angle_diffs = torch.abs(angles[1:] - angles[:-1])
        angle_sum = torch.sum(angle_diffs)
        angle_sum_penalty = angle_sum ** angle_sum_power
    else:
        angle_sum_penalty = torch.tensor(0.0, device=device, dtype=dtype)

    # Create copies for projection (avoid modifying originals)
    well_data_copy = {
        'md': well_data['md'],
        'vs': well_data['vs'],
        'tvd': well_data['tvd'],
        'value': well_data['value'],
        'tvt': well_data['tvt'].clone(),
        'synt_curve': well_data['synt_curve'].clone(),
        'md_range': well_data['md_range'],
        'min_curve': well_data['min_curve'],
        'max_curve': well_data['max_curve'],
        'horizontal_well_step': well_data['horizontal_well_step'],
        'normalized': well_data['normalized'],
    }

    # Calculate projection
    success, well_data_copy = calc_horizontal_projection_torch(
        well_data_copy, typewell_data, new_segments, tvd_to_typewell_shift
    )

    if not success:
        return torch.tensor(float('inf'), device=device, dtype=dtype)

    # Get indices range
    start_idx = int(new_segments[0, 0].item())
    end_idx = int(new_segments[-1, 1].item())

    # Check for NaN in synthetic curve
    synt_slice = well_data_copy['synt_curve'][start_idx:end_idx + 1]
    if torch.any(torch.isnan(synt_slice)):
        return torch.tensor(float('inf'), device=device, dtype=dtype)

    # Calculate MSE
    x = well_data_copy['value'][start_idx:end_idx + 1]
    y = synt_slice
    mse = mse_torch(x, y)

    # Calculate Pearson
    x_const = torch.all(x == x[0])
    y_const = torch.all(y == y[0])

    if not x_const and not y_const:
        pearson = pearson_torch(x, y)
    else:
        pearson = torch.tensor(0.0, device=device, dtype=dtype)

    pearson = torch.max(pearson, torch.tensor(min_pearson_value, device=device, dtype=dtype))

    # Calculate intersections
    if num_intervals_self_correlation > 0:
        tvt_slice = well_data_copy['tvt'][self_corr_start_idx:end_idx + 1]
        value_slice = well_data_copy['value'][self_corr_start_idx:end_idx + 1]

        intersections_count = find_intersections_torch(
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

    # Combine metrics
    pearson_component = 1 - pearson

    metric = (
        (pearson_component ** pearson_power) *
        (mse ** mse_power) *
        (1.0 / intersections_component)
    ) * (1 + angle_penalty + angle_sum_penalty)

    return metric
