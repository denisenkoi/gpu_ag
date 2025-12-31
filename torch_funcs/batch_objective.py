"""
Batch objective function for parallel evaluation of population.

This is the key function for GPU acceleration - evaluates 500 individuals in parallel.
"""
import torch
from .converters import update_segments_with_shifts_torch, calc_segment_angles_torch
from .projection import calc_horizontal_projection_batch_torch
from .correlations import pearson_batch_torch, mse_batch_torch
from .self_correlation import find_intersections_batch_torch


def normalized_angles_to_shifts_torch(
    normalized_angles: torch.Tensor,
    segments_torch: torch.Tensor,
    normalize_factor: float = 10.0
) -> torch.Tensor:
    """
    Конвертирует нормализованные углы в end_shift'ы (кумулятивно) - GPU batch версия

    Args:
        normalized_angles: (batch, K) нормализованные углы (0.1 = 1°)
        segments_torch: (K, 6) = [start_idx, end_idx, start_tvd, end_tvd, start_shift, end_shift]
        normalize_factor: множитель (10.0 = 0.1 -> 1°)

    Returns:
        shifts: (batch, K) end_shift для каждого сегмента
    """
    batch_size, K = normalized_angles.shape
    device = normalized_angles.device
    dtype = normalized_angles.dtype

    # segments_torch: start_tvd(2) и end_tvd(3) - это VS (vertical section)
    start_vs = segments_torch[:, 2]  # (K,)
    end_vs = segments_torch[:, 3]    # (K,)
    segment_lens = end_vs - start_vs  # (K,)

    # start_shift первого сегмента - начальная точка
    first_start_shift = segments_torch[0, 4]

    angles_deg = normalized_angles * normalize_factor  # (batch, K)
    angles_rad = torch.deg2rad(angles_deg)
    delta_shifts = segment_lens.unsqueeze(0) * torch.tan(angles_rad)  # (batch, K)

    # Кумулятивная сумма дельт
    cumsum = torch.cumsum(delta_shifts, dim=1)  # (batch, K)
    shifts = first_start_shift + cumsum

    return shifts


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
    prev_segment_angle=None,
    min_angle_diff=0.2,
    min_trend_deviation=0.5,
    trend_power=1.0,
    reward_start_segment_idx=0
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
        reward_start_segment_idx: segment index from which to calculate Pearson/MSE reward.
            0 = all segments (default), 1 = skip first segment (telescope mode), etc.
            Projection is still calculated for ALL segments (geometry needed).

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

    # Exponential penalty for angle violations: 1M * 100^max_excess
    angle_excess = torch.abs(angles) - angle_range
    max_excess = torch.clamp(torch.max(angle_excess, dim=1).values, min=0.0)  # (batch,)
    angle_violation_penalty = torch.where(
        max_excess > 0,
        1_000_000.0 * torch.pow(torch.tensor(100.0, device=device, dtype=dtype), max_excess),
        torch.zeros_like(max_excess)
    )

    # Angle sum penalty: (batch,)
    # Includes angle difference with prev_segment (last frozen) if provided
    # min_angle_diff prevents degenerate CMA-ES covariance matrix
    K = angles.shape[1]
    if K > 1:
        angle_diffs = torch.abs(angles[:, 1:] - angles[:, :-1])  # (batch, K-1)
        # Clamp to minimum to prevent CMA-ES covariance degeneration
        angle_diffs = torch.clamp(angle_diffs, min=min_angle_diff)
        n_diffs_base = angle_diffs.shape[1]  # Original count for normalization
        if prev_segment_angle is not None:
            # Add diff between prev_segment and first optimizing segment
            first_diff = torch.abs(angles[:, 0] - prev_segment_angle)  # (batch,)
            first_diff = torch.clamp(first_diff, min=min_angle_diff)
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
        angle_diffs = torch.clamp(angle_diffs, min=min_angle_diff)
        angle_sum_penalty = angle_diffs ** angle_sum_power
    else:
        angle_sum_penalty = torch.zeros(batch_size, device=device, dtype=dtype)

    # Trend deviation penalty: penalize mean angle deviation from incoming trend
    if prev_segment_angle is not None:
        mean_angle = torch.mean(angles, dim=1)  # (batch,)
        trend_deviation = torch.abs(mean_angle - prev_segment_angle)
        trend_deviation = torch.clamp(trend_deviation, min=min_trend_deviation)
        trend_penalty = trend_deviation ** trend_power
    else:
        trend_penalty = torch.zeros(batch_size, device=device, dtype=dtype)

    # Calculate projection for all batches (always for ALL segments - geometry needed)
    success_mask, tvt_batch, synt_curve_batch, first_start_idx = calc_horizontal_projection_batch_torch(
        well_data, typewell_data, segments_batch, tvd_to_typewell_shift
    )

    # Determine reward range (telescope mode: skip first segment(s) for Pearson/MSE)
    last_end_idx = int(segments_torch[-1, 1].item())
    K = segments_torch.shape[0]
    if reward_start_segment_idx > 0 and reward_start_segment_idx < K:
        reward_start_idx = int(segments_torch[reward_start_segment_idx, 0].item())
    else:
        reward_start_idx = first_start_idx

    # Calculate offset for slicing synthetic curve
    reward_offset = reward_start_idx - first_start_idx

    # Get value array for REWARD range (may skip telescope lever segment)
    value_slice = well_data['value'][reward_start_idx:last_end_idx + 1]
    value_batch = value_slice.unsqueeze(0).expand(batch_size, -1)  # (batch, N_reward)

    # Slice synthetic curve to match reward range
    synt_curve_reward = synt_curve_batch[:, reward_offset:]  # (batch, N_reward)

    # Check for NaN in synthetic curve (check full curve for projection validity)
    has_nan = torch.any(torch.isnan(synt_curve_batch), dim=1)
    success_mask = success_mask & ~has_nan

    # Calculate MSE for reward range (may exclude telescope lever)
    mse = mse_batch_torch(value_batch, synt_curve_reward)

    # Calculate Pearson for reward range (may exclude telescope lever)
    pearson_raw = pearson_batch_torch(value_batch, synt_curve_reward)

    # Debug output for reward range diagnostics
    import os
    if os.getenv('DEBUG_REWARD_RANGE', 'false').lower() == 'true':
        import logging
        logger = logging.getLogger('batch_objective')
        md = well_data.get('md')
        md_range = well_data.get('md_range', 1.0)
        min_md = well_data.get('min_md', 0.0) if 'min_md' in well_data else 0.0
        if md is not None:
            # Denormalize MD for display
            md_start_real = md[reward_start_idx].item() * md_range + min_md if well_data.get('normalized') else md[reward_start_idx].item()
            md_end_real = md[last_end_idx].item() * md_range + min_md if well_data.get('normalized') else md[last_end_idx].item()
            logger.info(f"DEBUG REWARD: idx={reward_start_idx}-{last_end_idx}, MD={md_start_real:.1f}-{md_end_real:.1f}m (normalized={well_data.get('normalized')})")
        logger.info(f"DEBUG REWARD: value_batch shape={value_batch.shape}, synt_curve shape={synt_curve_reward.shape}")
        logger.info(f"DEBUG REWARD: value[0] first5={value_batch[0,:5].tolist()}, last5={value_batch[0,-5:].tolist()}")
        logger.info(f"DEBUG REWARD: synt[0] first5={synt_curve_reward[0,:5].tolist()}, last5={synt_curve_reward[0,-5:].tolist()}")
        logger.info(f"DEBUG REWARD: pearson_raw[0]={pearson_raw[0].item():.4f}")

    # Clamp pearson to min value (no penalty, just floor)
    pearson = torch.clamp(pearson_raw, min=min_pearson_value)

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

    # Calculate full metric (angle_sum_penalty and trend_penalty as soft constraints)
    metric_values = (
        (pearson_component ** pearson_power) *
        (mse ** mse_power) *
        (1.0 / intersections_component)
    ) * (1 + angle_sum_penalty) * (1 + trend_penalty)

    # Apply angle violation penalty (exponential)
    metric_values = metric_values * (1 + angle_violation_penalty)

    # Note: pearson_below_threshold penalty removed (was x100)
    # Now only clamp to min_pearson_value without additional penalty

    # Apply success mask - only projection failure causes inf
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

    # Hard constraint check (for logging)
    angle_excess = torch.abs(angles) - angle_range
    angle_violation = torch.any(angle_excess > 0, dim=1)
    # For logging purposes, compute penalty value
    angle_penalty = torch.sum(
        torch.where(angle_excess > 0, angle_excess ** 2, torch.zeros_like(angle_excess)),
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

    # Clamp pearson to min_pearson_value (passed via angle_range parameter position)
    # Note: compute_detailed doesn't have min_pearson_value param, so we use 0.3 as default
    # TODO: Add min_pearson_value to function signature
    pearson = torch.clamp(pearson, min=0.3)

    # Compute objective (angle violation -> inf)
    pearson_component = 1 - pearson
    base_objective = (
        (pearson_component ** pearson_power) *
        (mse ** mse_power)
    ) * (1 + angle_sum_penalty)
    objective = torch.where(angle_violation, torch.tensor(float('inf'), device=device, dtype=dtype), base_objective)

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
        device='cuda',
        min_angle_diff=0.2,
        min_trend_deviation=0.5,
        trend_power=1.0,
        reward_start_segment_idx=0
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

        # New parameters for CMA-ES stability and trend penalty
        self.min_angle_diff = min_angle_diff
        self.min_trend_deviation = min_trend_deviation
        self.trend_power = trend_power

        # Telescope mode: skip first segment(s) in Pearson/MSE reward
        self.reward_start_segment_idx = reward_start_segment_idx

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
            self.prev_segment_angle,
            self.min_angle_diff,
            self.min_trend_deviation,
            self.trend_power,
            self.reward_start_segment_idx
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
