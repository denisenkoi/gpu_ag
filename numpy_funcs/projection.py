"""
Numpy implementation of horizontal projection calculations.

Calculates tvt (true vertical thickness) and synt_curve (synthetic curve).
"""
import numpy as np


def linear_interpolation_numpy(x1, y1, x2, y2, x):
    """
    Linear interpolation between two points.
    Vectorized version for numpy arrays.
    """
    # Handle scalar or array inputs
    result = np.where(x1 == x2, y1, y1 + (y2 - y1) * (x - x1) / (x2 - x1))
    return result


def tvt2value_numpy(typewell_data, tvt):
    """
    Convert TVT values to curve values using typewell lookup.
    Vectorized version.

    Args:
        typewell_data: dict with 'tvd', 'value', and step parameters
        tvt: TVT values (scalar or array)

    Returns:
        Interpolated curve values
    """
    if typewell_data['normalized']:
        min_depth = typewell_data['normalized_min_depth']
        step = typewell_data['normalized_typewell_step']
    else:
        min_depth = typewell_data['min_depth']
        step = typewell_data['typewell_step']

    # Calculate indices
    idx_below = ((tvt - min_depth) / step).astype(np.int64)
    idx_above = idx_below + 1

    # Get values at indices
    tvd = typewell_data['tvd']
    value = typewell_data['value']

    # Bounds check
    idx_below = np.clip(idx_below, 0, len(tvd) - 2)
    idx_above = np.clip(idx_above, 1, len(tvd) - 1)

    depth_below = tvd[idx_below]
    depth_above = tvd[idx_above]
    curve_below = value[idx_below]
    curve_above = value[idx_above]

    # Linear interpolation
    result = linear_interpolation_numpy(depth_below, curve_below, depth_above, curve_above, tvt)
    return result


def calc_segment_tvt_numpy(well_data, segment_row, tvd_to_typewell_shift):
    """
    Calculate TVT for a single segment.

    Args:
        well_data: dict with well arrays
        segment_row: [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
        tvd_to_typewell_shift: float shift value

    Returns:
        tvt_values: array of TVT values for segment indices
        segment_indices: array of indices
        success: bool
    """
    start_idx = int(segment_row[0])
    end_idx = int(segment_row[1])
    start_vs = segment_row[2]
    end_vs = segment_row[3]
    start_shift = segment_row[4]
    end_shift = segment_row[5]

    depth_shift = end_shift - start_shift
    segment_indices = np.arange(start_idx, end_idx + 1)

    vs = well_data['vs']
    tvd = well_data['tvd']

    delta_vs = vs[end_idx] - vs[start_idx]
    if delta_vs == 0:
        return None, segment_indices, False

    shifts = start_shift + depth_shift * (vs[segment_indices] - start_vs) / delta_vs
    tvt_values = tvd[segment_indices] - shifts - tvd_to_typewell_shift

    if np.any(np.isnan(tvt_values)):
        return None, segment_indices, False

    return tvt_values, segment_indices, True


def calc_horizontal_projection_numpy(well_data, typewell_data, segments_data, tvd_to_typewell_shift=0.0):
    """
    Calculate horizontal projection: fill tvt and synt_curve arrays.

    Args:
        well_data: dict with well arrays (modified in place: tvt, synt_curve)
        typewell_data: dict with typewell arrays
        segments_data: (K, 6) array of segments
        tvd_to_typewell_shift: float shift value

    Returns:
        success: bool
        well_data: modified with filled tvt and synt_curve
    """
    if segments_data.shape[0] == 0:
        return True, well_data

    tvt = well_data['tvt']
    synt_curve = well_data['synt_curve']

    # Calculate TVT for all segments
    for i in range(segments_data.shape[0]):
        tvt_values, segment_indices, success = calc_segment_tvt_numpy(
            well_data, segments_data[i], tvd_to_typewell_shift
        )
        if not success:
            return False, well_data

        tvt[segment_indices] = tvt_values

    # Get full range of indices
    first_start_idx = int(segments_data[0, 0])
    last_end_idx = int(segments_data[-1, 1])
    all_indices = np.arange(first_start_idx, last_end_idx + 1)

    # Check TVT bounds against typewell
    tvt_min = np.min(typewell_data['tvd'])
    tvt_max = np.max(typewell_data['tvd'])

    if np.any(tvt[all_indices] > tvt_max) or np.any(tvt[all_indices] < tvt_min):
        return False, well_data

    # Calculate synthetic curve
    tvt_slice = tvt[all_indices]

    # Filter valid TVT values (not NaN and within typewell range)
    valid_mask = ~np.isnan(tvt_slice) & (tvt_slice >= tvt_min) & (tvt_slice <= tvt_max)

    # Calculate synt_curve for valid points
    valid_tvt = tvt_slice[valid_mask]
    if len(valid_tvt) > 0:
        valid_synt = tvt2value_numpy(typewell_data, valid_tvt)
        synt_curve[all_indices[valid_mask]] = valid_synt

    well_data['tvt'] = tvt
    well_data['synt_curve'] = synt_curve

    return True, well_data
