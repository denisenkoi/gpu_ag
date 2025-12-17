"""
Numpy implementation of self-correlation and intersection counting.
"""
import numpy as np


def group_values_numpy(values, threshold):
    """
    Group values that are within threshold of each other.

    Args:
        values: array of values to group
        threshold: maximum distance within a group

    Returns:
        list of groups (each group is a list of values)
    """
    if len(values) == 0:
        return []

    sorted_values = np.sort(values)
    grouped = []
    temp_group = [sorted_values[0]]

    for value in sorted_values[1:]:
        if value - temp_group[0] <= threshold:
            temp_group.append(value)
        else:
            grouped.append(temp_group)
            temp_group = [value]

    if temp_group:
        grouped.append(temp_group)

    return grouped


def find_intersections_numpy(tvt, curve, min_curve, max_curve, md_range, num_intervals, start_idx=0, max_delta_tvt_base=0.15):
    """
    Find curve intersections with target values.

    Args:
        tvt: array of TVT values
        curve: array of curve values
        min_curve: minimum curve value
        max_curve: maximum curve value
        md_range: MD range for normalization
        num_intervals: number of target intervals
        start_idx: starting index in original arrays
        max_delta_tvt_base: base max delta TVT (will be divided by md_range)

    Returns:
        intersections_count: total count of unique intersection groups
    """
    max_delta_tvt = max_delta_tvt_base / md_range

    target_values = np.linspace(min_curve, max_curve, num_intervals)

    # Dictionary for storing intersections
    intersections = {val: [] for val in target_values}

    # Find intersections starting from start_idx
    for i in range(start_idx + 1, len(curve)):
        for target in target_values:
            if (curve[i-1] - target) * (curve[i] - target) <= 0:
                # Interpolate TVT at intersection
                interp_tvt = np.interp(target, [curve[i-1], curve[i]], [tvt[i-1], tvt[i]])
                intersections[target].append(interp_tvt)

    # Group and count unique TVT values for each intersection
    intersections_count = 0
    for target in intersections:
        grouped_tvt = group_values_numpy(intersections[target], max_delta_tvt)
        count = 0 if len(grouped_tvt) < 2 else len(grouped_tvt)
        intersections_count += count

    return intersections_count
