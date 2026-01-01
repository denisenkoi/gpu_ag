"""
PyTorch implementation of horizontal projection calculations.

Calculates tvt (true vertical thickness) and synt_curve (synthetic curve).
Supports both single evaluation and batch processing.
"""
import os
import torch
from torch.profiler import record_function
from .converters import GPU_DTYPE


def linear_interpolation_torch(x1, y1, x2, y2, x):
    """
    Linear interpolation between two points (vectorized).
    """
    return torch.where(x1 == x2, y1, y1 + (y2 - y1) * (x - x1) / (x2 - x1))


def tvt2value_torch(typewell_data, tvt):
    """
    Convert TVT values to curve values using typewell lookup.
    Fully vectorized torch version.

    Args:
        typewell_data: dict with 'tvd', 'value' tensors and step parameters
        tvt: TVT values tensor (N,) or (batch, N)

    Returns:
        Interpolated curve values tensor
    """
    if typewell_data['normalized']:
        min_depth = typewell_data['normalized_min_depth']
        step = typewell_data['normalized_typewell_step']
    else:
        min_depth = typewell_data['min_depth']
        step = typewell_data['typewell_step']

    tvd = typewell_data['tvd']
    value = typewell_data['value']

    # Calculate indices
    idx_below = ((tvt - min_depth) / step).long()
    idx_above = idx_below + 1

    # Bounds check
    max_idx = len(tvd) - 1
    idx_below = torch.clamp(idx_below, 0, max_idx - 1)
    idx_above = torch.clamp(idx_above, 1, max_idx)

    # Flatten for indexing if batch
    original_shape = tvt.shape
    if tvt.dim() > 1:
        idx_below_flat = idx_below.flatten()
        idx_above_flat = idx_above.flatten()
        tvt_flat = tvt.flatten()
    else:
        idx_below_flat = idx_below
        idx_above_flat = idx_above
        tvt_flat = tvt

    # Get values at indices
    depth_below = tvd[idx_below_flat]
    depth_above = tvd[idx_above_flat]
    curve_below = value[idx_below_flat]
    curve_above = value[idx_above_flat]

    # Linear interpolation
    result = linear_interpolation_torch(depth_below, curve_below, depth_above, curve_above, tvt_flat)

    # Reshape back if batch
    if tvt.dim() > 1:
        result = result.reshape(original_shape)

    return result


def calc_segment_tvt_torch(well_data, segment_row, tvd_to_typewell_shift):
    """
    Calculate TVT for a single segment (single evaluation).

    Args:
        well_data: dict with well tensors
        segment_row: [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
        tvd_to_typewell_shift: float shift value

    Returns:
        tvt_values: tensor of TVT values for segment indices
        segment_indices: tensor of indices
        success: bool
    """
    start_idx = int(segment_row[0].item())
    end_idx = int(segment_row[1].item())
    start_vs = segment_row[2]
    end_vs = segment_row[3]
    start_shift = segment_row[4]
    end_shift = segment_row[5]

    depth_shift = end_shift - start_shift
    device = well_data['vs'].device

    segment_indices = torch.arange(start_idx, end_idx + 1, device=device)

    vs = well_data['vs']
    tvd = well_data['tvd']

    delta_vs = vs[end_idx] - vs[start_idx]
    if delta_vs == 0:
        return None, segment_indices, False

    shifts = start_shift + depth_shift * (vs[segment_indices] - start_vs) / delta_vs
    tvt_values = tvd[segment_indices] - shifts - tvd_to_typewell_shift

    if torch.any(torch.isnan(tvt_values)):
        return None, segment_indices, False

    return tvt_values, segment_indices, True


def calc_horizontal_projection_torch(well_data, typewell_data, segments_torch, tvd_to_typewell_shift=0.0):
    """
    Calculate horizontal projection: fill tvt and synt_curve tensors.
    Single evaluation version.

    Args:
        well_data: dict with well tensors (modified in place: tvt, synt_curve)
        typewell_data: dict with typewell tensors
        segments_torch: (K, 6) tensor of segments
        tvd_to_typewell_shift: float shift value

    Returns:
        success: bool
        well_data: modified with filled tvt and synt_curve
    """
    if segments_torch.shape[0] == 0:
        return True, well_data

    tvt = well_data['tvt']
    synt_curve = well_data['synt_curve']
    device = tvt.device

    # Calculate TVT for all segments
    for i in range(segments_torch.shape[0]):
        tvt_values, segment_indices, success = calc_segment_tvt_torch(
            well_data, segments_torch[i], tvd_to_typewell_shift
        )
        if not success:
            return False, well_data

        tvt[segment_indices] = tvt_values

    # Get full range of indices
    first_start_idx = int(segments_torch[0, 0].item())
    last_end_idx = int(segments_torch[-1, 1].item())
    all_indices = torch.arange(first_start_idx, last_end_idx + 1, device=device)

    # Check TVT bounds against typewell
    tvt_min = torch.min(typewell_data['tvd'])
    tvt_max = torch.max(typewell_data['tvd'])

    if torch.any(tvt[all_indices] > tvt_max) or torch.any(tvt[all_indices] < tvt_min):
        return False, well_data

    # Calculate synthetic curve
    tvt_slice = tvt[all_indices]

    # Filter valid TVT values
    valid_mask = ~torch.isnan(tvt_slice) & (tvt_slice >= tvt_min) & (tvt_slice <= tvt_max)

    # Calculate synt_curve for valid points
    valid_tvt = tvt_slice[valid_mask]
    if len(valid_tvt) > 0:
        valid_synt = tvt2value_torch(typewell_data, valid_tvt)
        synt_curve[all_indices[valid_mask]] = valid_synt

    well_data['tvt'] = tvt
    well_data['synt_curve'] = synt_curve

    return True, well_data


def calc_horizontal_projection_batch_torch(well_data, typewell_data, segments_batch, tvd_to_typewell_shift=0.0):
    """
    Calculate horizontal projection for a batch of segment configurations.
    Batch version for parallel processing.

    Args:
        well_data: dict with well tensors (base data, not modified)
        typewell_data: dict with typewell tensors
        segments_batch: (batch, K, 6) tensor of segments
        tvd_to_typewell_shift: float shift value

    Returns:
        success_mask: (batch,) bool tensor
        tvt_batch: (batch, N_indices) tensor
        synt_curve_batch: (batch, N_indices) tensor
    """
    batch_size = segments_batch.shape[0]
    K = segments_batch.shape[1]
    device = segments_batch.device

    # Get indices range from first batch (same for all)
    first_start_idx = int(segments_batch[0, 0, 0].item())
    last_end_idx = int(segments_batch[0, -1, 1].item())
    N_indices = last_end_idx - first_start_idx + 1

    # Pre-extract well data for the segment range
    vs = well_data['vs'][first_start_idx:last_end_idx + 1]
    tvd_well = well_data['tvd'][first_start_idx:last_end_idx + 1]

    # Initialize output
    tvt_batch = torch.full((batch_size, N_indices), float('nan'), device=device, dtype=GPU_DTYPE)
    synt_curve_batch = torch.full((batch_size, N_indices), float('nan'), device=device, dtype=GPU_DTYPE)
    success_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

    # Calculate TVT for each segment (vectorized across batch)
    with record_function("proj_tvt_loop"):
        for seg_idx in range(K):
            seg = segments_batch[:, seg_idx, :]  # (batch, 6)

            start_idx_rel = int(seg[0, 0].item()) - first_start_idx
            end_idx_rel = int(seg[0, 1].item()) - first_start_idx

            start_vs = seg[:, 2:3]  # (batch, 1)
            end_vs = seg[:, 3:4]    # (batch, 1)
            start_shift = seg[:, 4:5]  # (batch, 1)
            end_shift = seg[:, 5:6]    # (batch, 1)

            depth_shift = end_shift - start_shift

            # VS values for this segment
            vs_seg = vs[start_idx_rel:end_idx_rel + 1].unsqueeze(0)  # (1, seg_len)

            # Calculate delta_vs per batch
            delta_vs = end_vs - start_vs  # (batch, 1)

            # Check for zero delta_vs
            zero_mask = (delta_vs.squeeze() == 0)
            success_mask = success_mask & ~zero_mask

            # Calculate shifts for all batches
            # shifts shape: (batch, seg_len)
            shifts = start_shift + depth_shift * (vs_seg - start_vs) / delta_vs

            # TVD values for this segment
            tvd_seg = tvd_well[start_idx_rel:end_idx_rel + 1].unsqueeze(0)  # (1, seg_len)

            # Calculate TVT
            tvt_seg = tvd_seg - shifts - tvd_to_typewell_shift

            # Store in output
            tvt_batch[:, start_idx_rel:end_idx_rel + 1] = tvt_seg

    # Check TVT bounds
    tvt_min = torch.min(typewell_data['tvd'])
    tvt_max = torch.max(typewell_data['tvd'])

    # Debug TVT bounds
    import os
    if os.getenv('DEBUG_REWARD_RANGE', 'false').lower() == 'true':
        import logging
        logger = logging.getLogger('projection')
        # Handle NaN in min/max
        tvt_valid = tvt_batch[0][~torch.isnan(tvt_batch[0])]
        if len(tvt_valid) > 0:
            tvt_actual_min = tvt_valid.min().item()
            tvt_actual_max = tvt_valid.max().item()
        else:
            tvt_actual_min = float('nan')
            tvt_actual_max = float('nan')
        logger.info(f"DEBUG TVT: batch[0] range=[{tvt_actual_min:.4f}, {tvt_actual_max:.4f}], typewell bounds=[{tvt_min.item():.4f}, {tvt_max.item():.4f}]")

    out_of_bounds = torch.any(tvt_batch > tvt_max, dim=1) | torch.any(tvt_batch < tvt_min, dim=1)
    success_mask = success_mask & ~out_of_bounds

    # Calculate synthetic curve for valid batches
    # For simplicity, calculate for all and mask later
    valid_tvt_mask = ~torch.isnan(tvt_batch) & (tvt_batch >= tvt_min) & (tvt_batch <= tvt_max)

    # Flatten for batch lookup
    tvt_flat = tvt_batch.flatten()
    valid_flat = valid_tvt_mask.flatten()

    synt_flat = torch.full_like(tvt_flat, float('nan'))
    if valid_flat.any():
        with record_function("proj_tvt2value"):
            synt_flat[valid_flat] = tvt2value_torch(typewell_data, tvt_flat[valid_flat])

    synt_curve_batch = synt_flat.reshape(batch_size, N_indices)

    return success_mask, tvt_batch, synt_curve_batch, first_start_idx
