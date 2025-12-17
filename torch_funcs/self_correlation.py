"""
PyTorch implementation of self-correlation and intersection counting.

Note: This function involves dynamic iteration and is harder to vectorize fully.
For batch processing, we use a hybrid approach.
"""
import torch


def find_intersections_torch(tvt, curve, min_curve, max_curve, md_range, num_intervals, start_idx=0, max_delta_tvt_base=0.15):
    """
    Find curve intersections with target values.
    Single evaluation version (mostly sequential due to algorithm nature).

    Args:
        tvt: tensor of TVT values
        curve: tensor of curve values
        min_curve: minimum curve value
        max_curve: maximum curve value
        md_range: MD range for normalization
        num_intervals: number of target intervals
        start_idx: starting index in arrays
        max_delta_tvt_base: base max delta TVT

    Returns:
        intersections_count: total count of unique intersection groups
    """
    device = tvt.device
    max_delta_tvt = max_delta_tvt_base / md_range

    target_values = torch.linspace(min_curve, max_curve, num_intervals, device=device, dtype=tvt.dtype)

    # Initialize intersections dict (use lists for dynamic appending)
    intersections = {i: [] for i in range(num_intervals)}

    # Find intersections starting from start_idx
    for i in range(start_idx + 1, len(curve)):
        curve_prev = curve[i-1]
        curve_curr = curve[i]
        tvt_prev = tvt[i-1]
        tvt_curr = tvt[i]

        for j, target in enumerate(target_values):
            if (curve_prev - target) * (curve_curr - target) <= 0:
                # Interpolate TVT at intersection
                if curve_curr != curve_prev:
                    interp_tvt = tvt_prev + (target - curve_prev) * (tvt_curr - tvt_prev) / (curve_curr - curve_prev)
                else:
                    interp_tvt = tvt_prev
                intersections[j].append(interp_tvt.item())

    # Group and count unique TVT values
    intersections_count = 0
    for j in range(num_intervals):
        tvt_list = intersections[j]
        if len(tvt_list) < 2:
            continue

        # Group values within threshold
        sorted_tvt = sorted(tvt_list)
        groups = []
        temp_group = [sorted_tvt[0]]

        for val in sorted_tvt[1:]:
            if val - temp_group[0] <= max_delta_tvt:
                temp_group.append(val)
            else:
                groups.append(temp_group)
                temp_group = [val]

        if temp_group:
            groups.append(temp_group)

        if len(groups) >= 2:
            intersections_count += len(groups)

    return intersections_count


def find_intersections_batch_torch(tvt_batch, curve_batch, min_curve, max_curve, md_range, num_intervals, start_idx=0, max_delta_tvt_base=0.15):
    """
    Find curve intersections for batch.

    Uses the same algorithm as single version for accuracy.
    Loop over batch elements, but reuses single version logic.

    Args:
        tvt_batch: (batch, N) tensor of TVT values
        curve_batch: (batch, N) tensor of curve values (typically well.value repeated)
        min_curve, max_curve: curve bounds
        md_range: MD range for normalization
        num_intervals: number of target intervals
        start_idx: starting index
        max_delta_tvt_base: base max delta TVT

    Returns:
        intersections_count: (batch,) tensor of intersection counts
    """
    batch_size = tvt_batch.shape[0]
    device = tvt_batch.device

    results = torch.zeros(batch_size, device=device, dtype=torch.long)

    for b in range(batch_size):
        results[b] = find_intersections_torch(
            tvt_batch[b],
            curve_batch[b],
            min_curve,
            max_curve,
            md_range,
            num_intervals,
            start_idx,
            max_delta_tvt_base
        )

    return results
