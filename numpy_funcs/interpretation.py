"""
Interpretation utilities - работа с сегментами интерпретации.

Формат датасета:
    ref_segment_mds[i] = startMd сегмента i
    ref_start_shifts[i] = startShift сегмента i
    ref_shifts[i] = endShift сегмента i
    endMd сегмента i = ref_mds[i+1] (или конец скважины для последнего)

Формат сегмента (dict):
    {
        'startMd': float,
        'endMd': float,
        'startShift': float,
        'endShift': float
    }

Формат numpy array (K, 6):
    [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple


def build_segments_from_dataset(data: Dict[str, Any], well_md: np.ndarray = None) -> List[Dict]:
    """
    Build segments list from dataset tensors.

    Args:
        data: Dataset dict with ref_segment_mds, ref_start_shifts, ref_shifts
        well_md: Optional well MD array to determine last segment endMd

    Returns:
        List of segment dicts
    """
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_start_shifts = data['ref_start_shifts'].cpu().numpy()
    ref_end_shifts = data['ref_shifts'].cpu().numpy()

    if len(ref_mds) == 0:
        return []

    # Determine end MD for last segment
    if well_md is not None:
        last_end_md = float(well_md[-1])
    elif 'lateral_well_last_md' in data:
        last_end_md = float(data['lateral_well_last_md'])
    else:
        last_end_md = float(ref_mds[-1]) + 300.0  # fallback

    segments = []
    for i in range(len(ref_mds)):
        # endMd is next segment's startMd, or last_end_md for last segment
        if i + 1 < len(ref_mds):
            end_md = float(ref_mds[i + 1])
        else:
            end_md = last_end_md

        segments.append({
            'startMd': float(ref_mds[i]),
            'endMd': end_md,
            'startShift': float(ref_start_shifts[i]),
            'endShift': float(ref_end_shifts[i])
        })

    return segments


def build_segments_to_md(data: Dict[str, Any], target_md: float) -> List[Dict]:
    """
    Build segments from dataset up to target_md.

    Args:
        data: Dataset dict with ref_segment_mds, ref_start_shifts, ref_shifts
        target_md: MD to build interpretation up to

    Returns:
        List of segment dicts, last segment truncated at target_md
    """
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_start_shifts = data['ref_start_shifts'].cpu().numpy()
    ref_end_shifts = data['ref_shifts'].cpu().numpy()

    if len(ref_mds) == 0:
        return []

    segments = []
    for i in range(len(ref_mds)):
        seg_start_md = float(ref_mds[i])
        start_shift = float(ref_start_shifts[i])
        end_shift = float(ref_end_shifts[i])

        # endMd is next segment's startMd, or target_md for last
        if i + 1 < len(ref_mds):
            seg_end_md = float(ref_mds[i + 1])
        else:
            seg_end_md = target_md

        # Stop if segment starts after target_md
        if seg_start_md >= target_md:
            break

        # Truncate at target_md if needed
        if seg_end_md > target_md:
            if seg_end_md > seg_start_md:
                ratio = (target_md - seg_start_md) / (seg_end_md - seg_start_md)
                end_shift = start_shift + ratio * (end_shift - start_shift)
            seg_end_md = target_md

        segments.append({
            'startMd': seg_start_md,
            'endMd': seg_end_md,
            'startShift': start_shift,
            'endShift': end_shift
        })

        if seg_end_md >= target_md:
            break

    return segments


def interpolate_shift_at_md(data: Dict[str, Any], target_md: float) -> float:
    """
    Interpolate shift at given MD.

    Args:
        data: Dataset dict with ref_segment_mds, ref_start_shifts, ref_shifts
        target_md: MD to interpolate shift at

    Returns:
        Interpolated shift value (meters)
    """
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_start_shifts = data['ref_start_shifts'].cpu().numpy()
    ref_end_shifts = data['ref_shifts'].cpu().numpy()

    if len(ref_mds) == 0:
        return 0.0

    # target_md before first segment
    if target_md < ref_mds[0]:
        return float(ref_start_shifts[0])

    # Find segment containing target_md
    for i in range(len(ref_mds)):
        seg_start = ref_mds[i]
        seg_end = ref_mds[i + 1] if i + 1 < len(ref_mds) else float('inf')

        if seg_start <= target_md <= seg_end:
            start_shift = ref_start_shifts[i]
            end_shift = ref_end_shifts[i]

            if seg_end > seg_start and seg_end != float('inf'):
                ratio = (target_md - seg_start) / (seg_end - seg_start)
                return float(start_shift + ratio * (end_shift - start_shift))
            return float(start_shift)

    # target_md after last segment
    return float(ref_end_shifts[-1])


def segments_to_numpy_array(
    segments: List[Dict],
    well_md: np.ndarray,
    well_vs: np.ndarray
) -> np.ndarray:
    """
    Convert segments list to numpy array for projection.

    Args:
        segments: List of segment dicts
        well_md: Well MD array
        well_vs: Well VS array

    Returns:
        numpy array (K, 6): [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
    """
    result = []

    for seg in segments:
        start_md = seg['startMd']
        end_md = seg['endMd']
        start_shift = seg['startShift']
        end_shift = seg['endShift']

        # Skip zero-length segments
        if abs(end_md - start_md) < 0.01:
            continue

        # Find indices
        start_idx = np.searchsorted(well_md, start_md)
        end_idx = np.searchsorted(well_md, end_md)

        if start_idx >= len(well_md):
            start_idx = len(well_md) - 1
        if end_idx >= len(well_md):
            end_idx = len(well_md) - 1

        # Skip if same index
        if start_idx == end_idx:
            continue

        start_vs = well_vs[start_idx]
        end_vs = well_vs[end_idx]

        result.append([start_idx, end_idx, start_vs, end_vs, start_shift, end_shift])

    if not result:
        return np.array([], dtype=np.float64).reshape(0, 6)

    return np.array(result, dtype=np.float64)


def validate_continuity(segments: List[Dict], tolerance: float = 0.01) -> Tuple[bool, List[str]]:
    """
    Validate that segments are continuous (endShift[i] ≈ startShift[i+1]).

    Args:
        segments: List of segment dicts
        tolerance: Max allowed difference in meters

    Returns:
        (is_valid, list of error messages)
    """
    errors = []

    for i in range(len(segments) - 1):
        end_shift = segments[i]['endShift']
        next_start_shift = segments[i + 1]['startShift']
        diff = abs(end_shift - next_start_shift)

        if diff > tolerance:
            errors.append(
                f"Segment {i}: endShift={end_shift:.4f} != startShift[{i+1}]={next_start_shift:.4f} "
                f"(diff={diff:.4f}m)"
            )

    return len(errors) == 0, errors


def calc_vs_from_trajectory(ns: np.ndarray, ew: np.ndarray) -> np.ndarray:
    """
    Calculate VS (cumulative horizontal distance) from NS/EW trajectory.

    Args:
        ns: North-South coordinates
        ew: East-West coordinates

    Returns:
        VS array (same length as input)
    """
    vs = np.zeros(len(ns))
    for i in range(1, len(ns)):
        dx = ns[i] - ns[i - 1]
        dy = ew[i] - ew[i - 1]
        vs[i] = vs[i - 1] + np.sqrt(dx * dx + dy * dy)
    return vs
