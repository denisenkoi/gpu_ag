"""
Converters from Python objects to numpy arrays.

Data structures:

well_data = {
    'md': np.array,           # (N,) measured depth
    'vs': np.array,           # (N,) VS along trajectory horizontal length
    'tvd': np.array,          # (N,) true vertical depth
    'value': np.array,        # (N,) curve values
    'tvt': np.array,          # (N,) true vertical thickness (output, filled by projection)
    'synt_curve': np.array,   # (N,) synthetic curve (output, filled by projection)
    'md_range': float,        # max_md - min_md
    'min_curve': float,
    'max_curve': float,
    'horizontal_well_step': float,
    'normalized': bool,
}

typewell_data = {
    'tvd': np.array,          # (M,) true vertical depth
    'value': np.array,        # (M,) curve values
    'min_depth': float,
    'typewell_step': float,
    'normalized': bool,
    'normalized_min_depth': float,  # only if normalized
    'normalized_typewell_step': float,  # only if normalized
}

segments_data = np.array     # (K, 6) where each row is:
                             # [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
"""
import numpy as np


def well_to_numpy(well):
    """
    Convert Well object to numpy dict.

    Args:
        well: Well object

    Returns:
        dict with numpy arrays
    """
    # Use normalization_md_range if well is normalized, else md_range
    md_range_for_denorm = getattr(well, 'normalization_md_range', well.md_range) if well.normalized else well.md_range
    return {
        'md': well.measured_depth.copy(),
        'vs': well.vs_thl.copy(),
        'tvd': well.true_vertical_depth.copy(),
        'value': well.value.copy(),
        'tvt': well.tvt.copy(),
        'synt_curve': well.synt_curve.copy(),
        'md_range': md_range_for_denorm,
        'min_md': well.min_md,
        'min_curve': well.min_curve,
        'max_curve': well.max_curve,
        'horizontal_well_step': well.horizontal_well_step,
        'normalized': well.normalized,
    }


def typewell_to_numpy(typewell):
    """
    Convert TypeWell object to numpy dict.

    Args:
        typewell: TypeWell object

    Returns:
        dict with numpy arrays
    """
    result = {
        'tvd': typewell.tvd.copy(),
        'value': typewell.value.copy(),
        'min_depth': typewell.min_depth,
        'typewell_step': typewell.typewell_step,
        'normalized': typewell.normalized,
    }

    if typewell.normalized:
        result['normalized_min_depth'] = typewell.normalized_min_depth
        result['normalized_typewell_step'] = typewell.normalized_typewell_step

    return result


def segments_to_numpy(segments, well):
    """
    Convert list of Segment objects to numpy 2D array.

    Args:
        segments: list of Segment objects
        well: Well object (needed for vs values)

    Returns:
        np.array shape (K, 6) where each row is:
        [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
    """
    if not segments:
        return np.empty((0, 6), dtype=np.float64)

    K = len(segments)
    result = np.empty((K, 6), dtype=np.float64)

    for i, seg in enumerate(segments):
        result[i, 0] = seg.start_idx
        result[i, 1] = seg.end_idx
        result[i, 2] = seg.start_vs
        result[i, 3] = seg.end_vs
        result[i, 4] = seg.start_shift
        result[i, 5] = seg.end_shift

    return result


def numpy_to_segments_data(shifts, segments_data):
    """
    Update segments_data with new shift values (like deepcopy + update in original code).

    Args:
        shifts: array of new end_shift values (at least K elements)
        segments_data: original segments array (K, 6)

    Returns:
        new segments_data array (K, 6) with updated shifts

    Columns: [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
             [   0,        1,       2,       3,        4,          5    ]
    """
    # Copy to avoid modifying original
    new_segments = segments_data.copy()

    # Iterate over segments (not shifts) - matches original code behavior
    K = segments_data.shape[0]
    for i in range(K):
        new_segments[i, 5] = shifts[i]  # end_shift
        if i < K - 1:
            new_segments[i + 1, 4] = shifts[i]  # next segment's start_shift

    return new_segments


def calc_segment_angles(segments_data):
    """
    Calculate angles for all segments.

    Args:
        segments_data: (K, 6) array

    Returns:
        angles: (K,) array in degrees

    Columns: [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
    """
    delta_shift = segments_data[:, 5] - segments_data[:, 4]  # end_shift - start_shift
    delta_vs = segments_data[:, 3] - segments_data[:, 2]     # end_vs - start_vs

    angles = np.degrees(np.arctan(delta_shift / delta_vs))
    return angles
