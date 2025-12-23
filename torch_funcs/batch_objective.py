"""
Batch objective function for parallel evaluation of population.

This is the key function for GPU acceleration - evaluates 500 individuals in parallel.
"""
import torch
from .converters import update_segments_with_shifts_torch, calc_segment_angles_torch
from .projection import calc_horizontal_projection_batch_torch
from .correlations import pearson_batch_torch, mse_batch_torch
from .self_correlation import find_intersections_batch_torch


def batch_objective_function_torch(
    shifts_batch,
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
    tvd_to_typewell_shift=0.0,
    prev_segment_angle=None
):
    """
    Batch objective function for DE optimization.
    Evaluates entire population in parallel on GPU.

    Args:
        shifts_batch: tensor (batch, K) of end_shift values for each individual
        well_data: dict with well tensors
        typewell_data: dict with typewell tensors
        segments_torch: (K, 6) base segments tensor
        self_corr_start_idx: starting index for self-correlation
        pearson_power, mse_power: metric weights
        num_intervals_self_correlation: number of intervals
        sc_power: self-correlation power
        angle_range: max allowed angle
        angle_sum_power: power for angle sum penalty
        min_pearson_value: minimum pearson threshold
        tvd_to_typewell_shift: vertical shift

    Returns:
        metrics: tensor (batch,) of metric values (lower is better)
    """
    batch_size = shifts_batch.shape[0]
    device = shifts_batch.device
    dtype = shifts_batch.dtype

    # Initialize output
    metrics = torch.full((batch_size,), float('inf'), device=device, dtype=dtype)

    # Update segments with new shifts for all batches
    # segments_batch: (batch, K, 6)
    segments_batch = update_segments_with_shifts_torch(shifts_batch, segments_torch)

    # Calculate angles for all batches: (batch, K)
    angles = calc_segment_angles_torch(segments_batch)

    # Angle penalty: (batch,)
    angle_excess = torch.abs(angles) - angle_range
    angle_penalty = torch.sum(
        torch.where(angle_excess > 0, 1000 * angle_excess ** 2, torch.zeros_like(angle_excess)),
        dim=1
    )

    # Angle sum penalty: (batch,)
    # Includes angle difference with prev_segment (last frozen) if provided
    K = angles.shape[1]
    if K > 1:
        angle_diffs = torch.abs(angles[:, 1:] - angles[:, :-1])  # (batch, K-1)
        n_diffs_base = angle_diffs.shape[1]  # Original count for normalization
        if prev_segment_angle is not None:
            # Add diff between prev_segment and first optimizing segment
            first_diff = torch.abs(angles[:, 0] - prev_segment_angle)  # (batch,)
            angle_diffs = torch.cat([first_diff.unsqueeze(1), angle_diffs], dim=1)  # (batch, K)
        # Normalize to original scale: sum of original diffs + weighted extra diff
        # This keeps backward compatibility while adding prev_segment influence
        angle_sum = torch.sum(angle_diffs, dim=1)
        if prev_segment_angle is not None:
            # Scale down to match original behavior: new_sum * (K-1)/K
            angle_sum = angle_sum * (n_diffs_base / angle_diffs.shape[1])
        angle_sum_penalty = angle_sum ** angle_sum_power
    elif K == 1 and prev_segment_angle is not None:
        # Single segment but have prev_segment - penalize diff with it
        angle_diffs = torch.abs(angles[:, 0] - prev_segment_angle)
        angle_sum_penalty = angle_diffs ** angle_sum_power
    else:
        angle_sum_penalty = torch.zeros(batch_size, device=device, dtype=dtype)

    # Calculate projection for all batches
    success_mask, tvt_batch, synt_curve_batch, first_start_idx = calc_horizontal_projection_batch_torch(
        well_data, typewell_data, segments_batch, tvd_to_typewell_shift
    )

    # Get value array for the segment range
    last_end_idx = int(segments_torch[-1, 1].item())
    N_indices = last_end_idx - first_start_idx + 1
    value_slice = well_data['value'][first_start_idx:last_end_idx + 1]
    value_batch = value_slice.unsqueeze(0).expand(batch_size, -1)  # (batch, N_indices)

    # Check for NaN in synthetic curve
    has_nan = torch.any(torch.isnan(synt_curve_batch), dim=1)
    success_mask = success_mask & ~has_nan

    # Calculate MSE for successful batches: (batch,)
    mse = mse_batch_torch(value_batch, synt_curve_batch)

    # Calculate Pearson for successful batches: (batch,)
    pearson = pearson_batch_torch(value_batch, synt_curve_batch)

    # Clamp pearson to min value
    pearson = torch.clamp(pearson, min=min_pearson_value)

    # Self-correlation: DISABLED for GPU optimization (sequential bottleneck)
    # TODO: Implement vectorized version (see docs/self_correlation_approaches.md)
    # When num_intervals_self_correlation > 0 and USE_SELF_CORRELATION_BATCH = True,
    # will use vectorized approach for 3-5x speedup target
    USE_SELF_CORRELATION_BATCH = False  # Toggle for development

    if num_intervals_self_correlation > 0 and USE_SELF_CORRELATION_BATCH:
        # For self-correlation, need data from self_corr_start_idx to end_idx
        last_end_idx = int(segments_batch[0, -1, 1].item())
        tvt_full = well_data['tvt'][self_corr_start_idx:last_end_idx + 1]
        value_full = well_data['value'][self_corr_start_idx:last_end_idx + 1]

        full_len = last_end_idx - self_corr_start_idx + 1
        tvt_batch_full = tvt_full.unsqueeze(0).expand(batch_size, -1).clone()
        proj_offset = first_start_idx - self_corr_start_idx
        tvt_batch_full[:, proj_offset:proj_offset + tvt_batch.shape[1]] = tvt_batch
        value_batch_full = value_full.unsqueeze(0).expand(batch_size, -1)

        intersections_count = find_intersections_batch_torch(
            tvt_batch_full,
            value_batch_full,
            well_data['min_curve'],
            well_data['max_curve'],
            well_data['md_range'],
            num_intervals_self_correlation,
            start_idx=proj_offset
        )
        intersections_component = sc_power ** intersections_count.float()
    else:
        intersections_component = torch.ones(batch_size, device=device, dtype=dtype)

    # Combine metrics
    pearson_component = 1 - pearson

    # Calculate full metric
    metric_values = (
        (pearson_component ** pearson_power) *
        (mse ** mse_power) *
        (1.0 / intersections_component)
    ) * (1 + angle_penalty + angle_sum_penalty)

    # Apply success mask - failed evaluations get inf
    metrics = torch.where(success_mask, metric_values, metrics)

    return metrics


def compute_detailed_metrics_torch(
    shifts,
    well_data,
    typewell_data,
    segments_torch,
    pearson_power,
    mse_power,
    angle_range,
    angle_sum_power,
    tvd_to_typewell_shift=0.0,
    prev_segment_angle=None
):
    """
    Compute detailed metrics for a single shift configuration.
    Returns dict with all components: pearson, mse, angles, penalties, objective.
    """
    device = shifts.device
    dtype = shifts.dtype

    # Ensure batch dimension
    if shifts.dim() == 1:
        shifts = shifts.unsqueeze(0)

    from .converters import update_segments_with_shifts_torch, calc_segment_angles_torch
    from .projection import calc_horizontal_projection_batch_torch
    from .correlations import pearson_batch_torch, mse_batch_torch

    # Update segments
    segments_batch = update_segments_with_shifts_torch(shifts, segments_torch)

    # Calculate angles
    angles = calc_segment_angles_torch(segments_batch)
    angles_deg = angles[0].cpu().numpy()  # First (only) batch item

    # Angle penalty
    angle_excess = torch.abs(angles) - angle_range
    angle_penalty = torch.sum(
        torch.where(angle_excess > 0, 1000 * angle_excess ** 2, torch.zeros_like(angle_excess)),
        dim=1
    )

    # Angle sum penalty
    K = angles.shape[1]
    if K > 1:
        angle_diffs = torch.abs(angles[:, 1:] - angles[:, :-1])
        n_diffs_base = angle_diffs.shape[1]
        if prev_segment_angle is not None:
            first_diff = torch.abs(angles[:, 0] - prev_segment_angle)
            angle_diffs = torch.cat([first_diff.unsqueeze(1), angle_diffs], dim=1)
        angle_sum = torch.sum(angle_diffs, dim=1)
        if prev_segment_angle is not None:
            angle_sum = angle_sum * (n_diffs_base / angle_diffs.shape[1])
        angle_sum_penalty = angle_sum ** angle_sum_power
    elif K == 1 and prev_segment_angle is not None:
        angle_diffs = torch.abs(angles[:, 0] - prev_segment_angle)
        angle_sum_penalty = angle_diffs ** angle_sum_power
    else:
        angle_sum_penalty = torch.zeros(1, device=device, dtype=dtype)

    # Projection
    success_mask, tvt_batch, synt_curve_batch, first_start_idx = calc_horizontal_projection_batch_torch(
        well_data, typewell_data, segments_batch, tvd_to_typewell_shift
    )

    # Get value array
    last_end_idx = int(segments_torch[-1, 1].item())
    N_indices = last_end_idx - first_start_idx + 1
    value_slice = well_data['value'][first_start_idx:last_end_idx + 1]
    value_batch = value_slice.unsqueeze(0)

    # MSE and Pearson
    mse = mse_batch_torch(value_batch, synt_curve_batch)
    pearson = pearson_batch_torch(value_batch, synt_curve_batch)

    # Compute objective
    pearson_component = 1 - pearson
    objective = (
        (pearson_component ** pearson_power) *
        (mse ** mse_power)
    ) * (1 + angle_penalty + angle_sum_penalty)

    return {
        'objective': objective[0].item(),
        'pearson': pearson[0].item(),
        'mse': mse[0].item(),
        'pearson_component': pearson_component[0].item(),
        'angle_penalty': angle_penalty[0].item(),
        'angle_sum_penalty': angle_sum_penalty[0].item(),
        'angles_deg': angles_deg.tolist(),
        'success': success_mask[0].item()
    }


class TorchObjectiveWrapper:
    """
    Wrapper class for use with EvoTorch or other optimizers.
    Stores static data and provides callable for batch evaluation.
    """

    def __init__(
        self,
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
        tvd_to_typewell_shift=0.0,
        prev_segment_angle=None,
        device='cuda'
    ):
        """
        Initialize wrapper with static data.

        Args:
            well_data: numpy dict (will be converted to torch)
            typewell_data: numpy dict
            segments_torch: base segments tensor (K, 6)
            ... other parameters
            device: 'cpu' or 'cuda'
        """
        self.device = device

        # Convert numpy to torch if needed
        from .converters import numpy_to_torch
        if not isinstance(well_data.get('md'), torch.Tensor):
            self.well_data = numpy_to_torch(well_data, device=device)
        else:
            self.well_data = well_data

        if not isinstance(typewell_data.get('tvd'), torch.Tensor):
            self.typewell_data = numpy_to_torch(typewell_data, device=device)
        else:
            self.typewell_data = typewell_data

        if not isinstance(segments_torch, torch.Tensor):
            self.segments_torch = torch.tensor(segments_torch, dtype=torch.float64, device=device)
        else:
            self.segments_torch = segments_torch.to(device)

        # Store parameters
        self.self_corr_start_idx = self_corr_start_idx
        self.pearson_power = pearson_power
        self.mse_power = mse_power
        self.num_intervals_self_correlation = num_intervals_self_correlation
        self.sc_power = sc_power
        self.angle_range = angle_range
        self.angle_sum_power = angle_sum_power

        # Evaluation counter
        self.eval_count = 0

        self.min_pearson_value = min_pearson_value
        self.tvd_to_typewell_shift = tvd_to_typewell_shift
        self.prev_segment_angle = prev_segment_angle

    def __call__(self, shifts_batch):
        """
        Evaluate batch of shift configurations.

        Args:
            shifts_batch: tensor (batch, K) or (batch, K+) of shift values

        Returns:
            tensor (batch,) of metric values
        """
        if not isinstance(shifts_batch, torch.Tensor):
            shifts_batch = torch.tensor(shifts_batch, dtype=torch.float64, device=self.device)

        if shifts_batch.device != self.device:
            shifts_batch = shifts_batch.to(self.device)

        # Count actual evaluations
        self.eval_count += shifts_batch.shape[0]

        return batch_objective_function_torch(
            shifts_batch,
            self.well_data,
            self.typewell_data,
            self.segments_torch,
            self.self_corr_start_idx,
            self.pearson_power,
            self.mse_power,
            self.num_intervals_self_correlation,
            self.sc_power,
            self.angle_range,
            self.angle_sum_power,
            self.min_pearson_value,
            self.tvd_to_typewell_shift,
            self.prev_segment_angle
        )

    def evaluate_single(self, shifts):
        """
        Evaluate single shift configuration (for debugging).

        Args:
            shifts: tensor (K,) or array

        Returns:
            scalar metric value
        """
        if not isinstance(shifts, torch.Tensor):
            shifts = torch.tensor(shifts, dtype=torch.float64, device=self.device)

        # Add batch dimension
        shifts_batch = shifts.unsqueeze(0)
        result = self(shifts_batch)
        return result[0]

    def compute_detailed(self, shifts, first_start_shift=None):
        """
        Compute detailed metrics for a single shift configuration.

        Args:
            shifts: tensor (K,) or array of shift values
            first_start_shift: optional override for first segment's start_shift

        Returns:
            dict with objective, pearson, mse, angles, penalties
        """
        if not isinstance(shifts, torch.Tensor):
            shifts = torch.tensor(shifts, dtype=torch.float64, device=self.device)

        # Use modified segments_torch if first_start_shift provided
        segments = self.segments_torch
        if first_start_shift is not None:
            segments = self.segments_torch.clone()
            segments[0, 4] = first_start_shift  # Override first segment's start_shift

        return compute_detailed_metrics_torch(
            shifts,
            self.well_data,
            self.typewell_data,
            segments,
            self.pearson_power,
            self.mse_power,
            self.angle_range,
            self.angle_sum_power,
            self.tvd_to_typewell_shift,
            self.prev_segment_angle
        )
