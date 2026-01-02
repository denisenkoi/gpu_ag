"""
Simple GR metrics calculation.

One function that does everything:
- Takes interpretation segments and MD range
- Finds/clips segments in range
- Does projection
- Computes pearson, mse

No lever, reward_start_segment_idx or other complexity!
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from .converters import GPU_DTYPE
from .projection import calc_horizontal_projection_batch_torch
from .correlations import pearson_batch_torch, mse_batch_torch


def compute_gr_metrics(
    well_data: Dict[str, Any],
    typewell_data: Dict[str, Any],
    interpretation_segments: List[Dict],
    md_start: float,
    md_end: float,
    tvd_to_typewell_shift: float = 0.0,
    pearson_power: float = 1.0,
    mse_power: float = 2.0,
    min_pearson_value: float = 0.3,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compute GR correlation metrics (Pearson, MSE, objective) for interpretation in MD range.

    Simple function - no lever, skip_segments, or other complexity.
    Just: interpretation + MD range -> metrics.

    Clips segments to MD range, interpolates shifts at boundaries.

    Args:
        well_data: Dict with 'md', 'tvd', 'vs', 'value' (GR) tensors/arrays
        typewell_data: Dict with 'tvd', 'value' (GR), 'min_depth', 'typewell_step'
        interpretation_segments: List of segment dicts with 'startMd', 'endMd',
                                 'startShift', 'endShift'
        md_start: Start of MD range (meters)
        md_end: End of MD range (meters)
        tvd_to_typewell_shift: TVD shift for typewell alignment
        pearson_power: Power for pearson in objective (default 1.0)
        mse_power: Power for MSE in objective (default 2.0)
        min_pearson_value: Min pearson for clamping in objective (default 0.3)
        device: torch device ('cpu' or 'cuda')

    Returns:
        Dict with 'pearson', 'pearson_raw', 'mse', 'objective',
                  'n_segments', 'n_points', 'success'
    """
    # Convert well data to numpy if needed
    well_md = _to_numpy(well_data['md'])
    well_tvd = _to_numpy(well_data['tvd'])
    well_vs = _to_numpy(well_data.get('vs'))
    well_gr = _to_numpy(well_data['value'])

    # Calculate VS if not provided
    if well_vs is None:
        well_ns = _to_numpy(well_data.get('ns'))
        well_ew = _to_numpy(well_data.get('ew'))
        if well_ns is not None and well_ew is not None:
            well_vs = _calc_vs(well_ns, well_ew)
        else:
            return {'pearson': float('nan'), 'pearson_raw': float('nan'),
                    'mse': float('nan'), 'objective': float('nan'),
                    'n_segments': 0, 'n_points': 0, 'success': False,
                    'error': 'No VS data and no NS/EW to calculate it'}

    # Build segments in MD range
    segments = _build_segments_in_range(
        interpretation_segments, well_md, well_vs, md_start, md_end
    )

    if len(segments) == 0:
        return {'pearson': float('nan'), 'pearson_raw': float('nan'),
                'mse': float('nan'), 'objective': float('nan'),
                'n_segments': 0, 'n_points': 0, 'success': False,
                'error': 'No segments in MD range'}

    segments_np = np.array(segments, dtype=np.float32)
    start_idx = int(segments_np[0, 0])
    end_idx = int(segments_np[-1, 1])

    # Prepare torch data
    type_tvd = _to_numpy(typewell_data['tvd'])
    type_gr = _to_numpy(typewell_data['value'])
    tw_step = typewell_data.get('typewell_step')
    if tw_step is None:
        tw_step = float(type_tvd[1] - type_tvd[0])
    min_depth = typewell_data.get('min_depth')
    if min_depth is None:
        min_depth = float(type_tvd.min())

    well_torch = {
        'md': torch.tensor(well_md, dtype=GPU_DTYPE, device=device),
        'vs': torch.tensor(well_vs, dtype=GPU_DTYPE, device=device),
        'tvd': torch.tensor(well_tvd, dtype=GPU_DTYPE, device=device),
        'value': torch.tensor(well_gr, dtype=GPU_DTYPE, device=device),
        'normalized': False,
    }
    typewell_torch = {
        'tvd': torch.tensor(type_tvd, dtype=GPU_DTYPE, device=device),
        'value': torch.tensor(type_gr, dtype=GPU_DTYPE, device=device),
        'min_depth': float(min_depth),
        'typewell_step': float(tw_step),
        'normalized': False,
    }
    segments_torch = torch.tensor(segments_np, dtype=GPU_DTYPE, device=device).unsqueeze(0)

    # Compute projection
    success_mask, tvt_batch, synt_batch, first_idx = calc_horizontal_projection_batch_torch(
        well_torch, typewell_torch, segments_torch,
        tvd_to_typewell_shift=tvd_to_typewell_shift
    )

    if not success_mask[0].item():
        return {'pearson': float('nan'), 'pearson_raw': float('nan'),
                'mse': float('nan'), 'objective': float('nan'),
                'n_segments': len(segments), 'n_points': 0, 'success': False,
                'error': 'Projection failed'}

    # Get GR values for range
    n_points = end_idx - start_idx + 1
    well_gr_range = torch.tensor(
        well_gr[start_idx:end_idx + 1], dtype=GPU_DTYPE, device=device
    ).unsqueeze(0)
    synt_range = synt_batch[:, :n_points]

    # Filter valid (non-NaN) points
    valid_mask = ~torch.isnan(synt_range[0])
    valid_count = valid_mask.sum().item()

    if valid_count < 10:
        return {'pearson': float('nan'), 'pearson_raw': float('nan'),
                'mse': float('nan'), 'objective': float('nan'),
                'n_segments': len(segments), 'n_points': int(valid_count),
                'success': False, 'error': f'Too few valid points: {valid_count}'}

    # Compute metrics
    well_gr_valid = well_gr_range[0, valid_mask].unsqueeze(0)
    synt_valid = synt_range[0, valid_mask].unsqueeze(0)

    pearson_raw = pearson_batch_torch(well_gr_valid, synt_valid)[0].item()
    mse_val = mse_batch_torch(well_gr_valid, synt_valid)[0].item()

    # Clamp pearson for objective calculation
    pearson_clamped = max(pearson_raw, min_pearson_value)

    # Compute objective: (1 - pearson)^p * mse^m
    pearson_component = 1.0 - pearson_clamped
    objective = (pearson_component ** pearson_power) * (mse_val ** mse_power)

    return {
        'pearson': float(pearson_clamped),
        'pearson_raw': float(pearson_raw),
        'mse': float(mse_val),
        'objective': float(objective),
        'n_segments': len(segments),
        'n_points': int(valid_count),
        'success': True
    }


def _to_numpy(data) -> Optional[np.ndarray]:
    """Convert tensor/array to numpy, or return None if data is None."""
    if data is None:
        return None
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)


def _calc_vs(ns: np.ndarray, ew: np.ndarray) -> np.ndarray:
    """Calculate VS (vertical section) from NS/EW coordinates."""
    vs = np.zeros(len(ns))
    for i in range(1, len(ns)):
        vs[i] = vs[i - 1] + np.sqrt((ns[i] - ns[i - 1])**2 + (ew[i] - ew[i - 1])**2)
    return vs


def _build_segments_in_range(
    interpretation_segments: List[Dict],
    well_md: np.ndarray,
    well_vs: np.ndarray,
    md_start: float,
    md_end: float
) -> List[List[float]]:
    """
    Build segment tensors for segments that fall within [md_start, md_end].

    Clips segments to range boundaries and interpolates shifts.

    Returns:
        List of [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
    """
    segments = []

    for i, seg in enumerate(interpretation_segments):
        seg_start_md = seg.get('startMd', 0)
        seg_end_md = seg.get('endMd')

        # Get endMd from next segment if not specified
        if seg_end_md is None:
            if i + 1 < len(interpretation_segments):
                seg_end_md = interpretation_segments[i + 1].get('startMd', seg_start_md)
            else:
                continue

        # Skip if entirely outside range
        if seg_end_md < md_start or seg_start_md > md_end:
            continue

        # Clip to range
        clipped_start_md = max(seg_start_md, md_start)
        clipped_end_md = min(seg_end_md, md_end)

        if clipped_end_md - clipped_start_md < 0.1:
            continue

        # Get shifts
        start_shift = seg.get('startShift', 0.0)
        end_shift = seg.get('endShift', 0.0)

        # Interpolate shifts for clipped boundaries
        seg_len = seg_end_md - seg_start_md
        if seg_len > 0:
            if clipped_start_md > seg_start_md:
                frac = (clipped_start_md - seg_start_md) / seg_len
                clipped_start_shift = start_shift + (end_shift - start_shift) * frac
            else:
                clipped_start_shift = start_shift

            if clipped_end_md < seg_end_md:
                frac = (clipped_end_md - seg_start_md) / seg_len
                clipped_end_shift = start_shift + (end_shift - start_shift) * frac
            else:
                clipped_end_shift = end_shift
        else:
            clipped_start_shift = start_shift
            clipped_end_shift = end_shift

        # Find indices in well
        start_idx = int(np.searchsorted(well_md, clipped_start_md))
        end_idx = int(np.searchsorted(well_md, clipped_end_md))

        start_idx = min(start_idx, len(well_md) - 1)
        end_idx = min(end_idx, len(well_md) - 1)

        if start_idx >= end_idx:
            continue

        # Get VS at indices
        start_vs = well_vs[start_idx]
        end_vs = well_vs[end_idx]

        segments.append([
            start_idx, end_idx,
            start_vs, end_vs,
            clipped_start_shift, clipped_end_shift
        ])

    return segments
