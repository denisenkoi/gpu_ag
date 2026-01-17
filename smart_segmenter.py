"""
Smart Segmenter - adaptive segment boundaries based on informative points.

Works on well_md grid (1 ft step) for direct index compatibility with gpu_executor.
Interpolates log_gr onto well_md grid for peak detection.

Settings (from .env):
- SMART_MIN_SEGMENT_LENGTH: minimum segment length (default: 10.0m)
- SMART_OVERLAP_SEGMENTS: how many segments to step back for next slice (default: 1)
- SMART_PELT_PENALTY: penalty for changepoint detection (default: 10.0)
- SMART_MIN_DISTANCE: min distance between info points (default: 10.0m)
- PYTHON_SEGMENTS_COUNT: number of segments per optimization (default: 5)
"""

import os
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import ruptures as rpt
import logging

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/gpu_ag_dataset.pt')

logger = logging.getLogger(__name__)


# ============ SETTINGS ============

def get_settings():
    """Load settings from environment."""
    return {
        'min_segment_length': float(os.getenv('SMART_MIN_SEGMENT_LENGTH', '10.0')),
        'overlap_segments': int(os.getenv('SMART_OVERLAP_SEGMENTS', '1')),
        'pelt_penalty': float(os.getenv('SMART_PELT_PENALTY', '10.0')),
        'min_distance': float(os.getenv('SMART_MIN_DISTANCE', '10.0')),
        'segments_count': int(os.getenv('PYTHON_SEGMENTS_COUNT', '5')),
    }


# ============ DATA CLASSES ============

@dataclass
class InfoPoint:
    """Informative point for segment boundary."""
    idx: int            # Index in well_md array
    md: float           # Absolute MD in meters
    type: str           # PEAK, VALLEY, STEP
    gr: float           # GR value at point
    score: float        # Prominence or |jump|
    detail: str         # Human readable detail


@dataclass
class SliceResult:
    """Result of one slice iteration."""
    boundaries: List[InfoPoint]         # All detected boundaries in this slice
    segment_indices: List[Tuple[int, int]]  # Segments as (start_idx, end_idx)
    segment_mds: List[Tuple[float, float]]  # Segments as (start_md, end_md) for logging
    next_start_idx: int                 # Where to start next slice (index)
    is_final: bool                      # True if reached well end


# ============ DETECTION FUNCTIONS ============

def detect_peaks_valleys(curve: np.ndarray, md: np.ndarray,
                         smooth_window: int = 20) -> Tuple[List[InfoPoint], float]:
    """
    Detect significant peaks and valleys using Otsu threshold.

    Args:
        curve: GR values on well_md grid
        md: well_md array (absolute meters)

    Returns:
        (list of InfoPoints with indices, otsu_threshold)
    """
    if len(curve) < smooth_window * 2:
        return [], 0.0

    smoothed = uniform_filter1d(curve.astype(float), size=smooth_window)

    # Find peaks and valleys
    peaks_up, props_up = find_peaks(smoothed, prominence=0, distance=smooth_window//2)
    peaks_down, props_down = find_peaks(-smoothed, prominence=0, distance=smooth_window//2)

    # Collect all prominences for Otsu
    all_proms = []
    if len(peaks_up) > 0:
        all_proms.extend(props_up['prominences'])
    if len(peaks_down) > 0:
        all_proms.extend(props_down['prominences'])

    if len(all_proms) < 2:
        return [], 0.0

    all_proms = np.array(all_proms)

    # Otsu threshold
    pmin, pmax = all_proms.min(), all_proms.max()
    if pmax - pmin < 1e-10:
        return [], pmin

    normalized = ((all_proms - pmin) / (pmax - pmin) * 255).astype(np.uint8)
    hist, _ = np.histogram(normalized, bins=256, range=(0, 256))
    hist = hist.astype(float) / hist.sum()

    best_t, best_var = 0, 0
    for t in range(1, 256):
        w0, w1 = hist[:t].sum(), hist[t:].sum()
        if w0 == 0 or w1 == 0:
            continue
        mu0 = (np.arange(t) * hist[:t]).sum() / w0
        mu1 = (np.arange(t, 256) * hist[t:]).sum() / w1
        var = w0 * w1 * (mu0 - mu1) ** 2
        if var > best_var:
            best_var, best_t = var, t

    threshold = pmin + (best_t / 255) * (pmax - pmin)

    # Apply threshold multiplier from env (lower = more peaks detected)
    threshold_mult = float(os.getenv('OTSU_THRESHOLD_MULT', '1.0'))
    threshold = threshold * threshold_mult

    # Collect significant points
    points = []

    for local_idx, prom in zip(peaks_up, props_up['prominences']):
        if prom > threshold:
            points.append(InfoPoint(
                idx=local_idx,  # Local index within slice, will be adjusted
                md=md[local_idx],
                type='PEAK',
                gr=curve[local_idx],
                score=prom,
                detail=f'prom={prom:.1f}'
            ))

    for local_idx, prom in zip(peaks_down, props_down['prominences']):
        if prom > threshold:
            points.append(InfoPoint(
                idx=local_idx,
                md=md[local_idx],
                type='VALLEY',
                gr=curve[local_idx],
                score=prom,
                detail=f'prom={prom:.1f}'
            ))

    return points, threshold


def detect_changepoints(curve: np.ndarray, md: np.ndarray,
                        penalty: float = 10.0,
                        jump_window: int = 20) -> List[InfoPoint]:
    """
    Detect changepoints (steps) using ruptures PELT.

    Args:
        curve: GR values on well_md grid
        md: well_md array (absolute meters)
        penalty: PELT penalty (lower = more changepoints)
        jump_window: window size for calculating jump magnitude

    Returns:
        List of InfoPoints for changepoints
    """
    if len(curve) < 50:
        return []

    algo = rpt.Pelt(model="rbf").fit(curve.reshape(-1, 1))
    cps = algo.predict(pen=penalty)
    cps = [cp for cp in cps if cp < len(curve)]

    points = []

    for local_idx in cps:
        # Calculate jump magnitude
        before = curve[max(0, local_idx-jump_window):local_idx].mean()
        after = curve[local_idx:min(len(curve), local_idx+jump_window)].mean()
        jump = after - before

        points.append(InfoPoint(
            idx=local_idx,
            md=md[local_idx],
            type='STEP',
            gr=curve[local_idx],
            score=abs(jump),
            detail=f'jump={jump:+.0f}'
        ))

    return points


def filter_points(points: List[InfoPoint],
                  min_distance: float = 10.0,
                  slice_end_idx: Optional[int] = None,
                  md_array: Optional[np.ndarray] = None) -> List[InfoPoint]:
    """
    Filter points by min_distance with priority rules.

    Args:
        points: list of InfoPoints
        min_distance: minimum distance between points (meters)
        slice_end_idx: if provided, ignore STEP points closer than min_distance to end
        md_array: MD array for distance calculation near end

    Returns:
        Filtered list of InfoPoints
    """
    if len(points) <= 1:
        return points

    # Sort by index (which corresponds to MD order)
    points = sorted(points, key=lambda p: p.idx)

    def priority(p: InfoPoint) -> Tuple[int, float]:
        """Higher priority = kept when conflict. Returns (type_priority, score)."""
        type_priority = 1 if p.type in ('PEAK', 'VALLEY') else 0
        return (type_priority, p.score)

    filtered = []

    for p in points:
        # Skip STEP points too close to slice end
        if slice_end_idx is not None and md_array is not None and p.type == 'STEP':
            if slice_end_idx < len(md_array):
                dist_to_end = md_array[slice_end_idx - 1] - p.md
                if dist_to_end < min_distance:
                    continue

        if not filtered:
            filtered.append(p)
        elif (p.md - filtered[-1].md) < min_distance:
            # Conflict with last accepted - compare priorities
            if priority(p) > priority(filtered[-1]):
                filtered[-1] = p
        else:
            filtered.append(p)

    return filtered


def get_segment_boundaries(curve: np.ndarray, md: np.ndarray,
                           start_idx_offset: int,
                           min_distance: float = 10.0,
                           pelt_penalty: float = 10.0,
                           smooth_window: int = 20,
                           is_final_slice: bool = False) -> List[InfoPoint]:
    """
    Get segment boundaries from informative points.

    Args:
        curve: GR values (slice)
        md: MD values (slice)
        start_idx_offset: offset to add to local indices to get global indices
        min_distance: minimum distance between boundaries (meters)
        pelt_penalty: penalty for PELT changepoint detection
        smooth_window: smoothing window for peak detection
        is_final_slice: if False, ignore STEP points near slice end

    Returns:
        List of InfoPoints as segment boundaries, sorted by index
    """
    # Detect peaks/valleys
    peaks, otsu_thresh = detect_peaks_valleys(curve, md, smooth_window)

    # Detect changepoints
    steps = detect_changepoints(curve, md, pelt_penalty)

    # Combine
    all_points = peaks + steps

    # Adjust indices to global
    for p in all_points:
        p.idx += start_idx_offset

    # Filter
    slice_end_idx = start_idx_offset + len(curve) if not is_final_slice else None
    filtered = filter_points(all_points, min_distance, slice_end_idx, md)

    return filtered


# ============ SMART SEGMENTER CLASS ============

class SmartSegmenter:
    """
    Stateful segmenter that handles iterative slicing with overlap.
    Works on well_md grid (1 ft step) for direct index compatibility.

    Usage:
        segmenter = SmartSegmenter(data)
        while not segmenter.is_finished:
            result = segmenter.next_slice()
            # result.segment_indices has (start_idx, end_idx) for each segment
    """

    def __init__(self, data: Dict[str, Any], settings: dict = None):
        """
        Initialize segmenter from dataset dict.

        Args:
            data: Dataset dict with well_md, log_md, log_gr, start_md
            settings: optional settings dict (uses env if not provided)
        """
        # Extract and convert data
        well_md = data['well_md'].cpu().numpy() if hasattr(data['well_md'], 'cpu') else np.array(data['well_md'])
        log_md = data['log_md'].cpu().numpy() if hasattr(data['log_md'], 'cpu') else np.array(data['log_md'])
        log_gr = data['log_gr'].cpu().numpy() if hasattr(data['log_gr'], 'cpu') else np.array(data['log_gr'])

        self.well_md = well_md
        self.well_step = np.median(np.diff(well_md))  # Should be ~0.3048m (1 ft)

        # Interpolate log_gr onto well_md grid
        # Only interpolate within log_md range, NaN outside
        self.curve = np.full(len(well_md), np.nan)
        log_range_mask = (well_md >= log_md.min()) & (well_md <= log_md.max())
        self.curve[log_range_mask] = np.interp(
            well_md[log_range_mask], log_md, log_gr
        )

        # Start/end indices - use detected_start_md if available (where interpretation starts)
        start_md = data.get('detected_start_md') or data.get('start_md') or well_md[0]
        start_md = float(start_md)
        self.start_idx = int(np.searchsorted(well_md, start_md))
        self.end_idx = len(well_md) - 1

        # Find valid curve end (last non-NaN)
        valid_mask = ~np.isnan(self.curve)
        if valid_mask.any():
            self.end_idx = np.where(valid_mask)[0][-1]

        # Settings
        s = settings or get_settings()
        self.min_segment_length = s['min_segment_length']
        self.overlap_segments = s['overlap_segments']
        self.pelt_penalty = s['pelt_penalty']
        self.min_distance = s['min_distance']
        self.segments_count = s['segments_count']

        # Convert min distances to indices (approximate)
        self.min_segment_idx = int(self.min_segment_length / self.well_step)
        self.min_distance_idx = int(self.min_distance / self.well_step)

        # State
        self.current_start_idx = self.start_idx
        self.all_boundaries: List[InfoPoint] = []
        self.is_finished = False
        self.iteration = 0

        logger.info(f"SmartSegmenter initialized: start_idx={self.start_idx}, end_idx={self.end_idx}, "
                   f"well_step={self.well_step:.4f}m, segments_count={self.segments_count}")

    def next_slice(self) -> SliceResult:
        """
        Process next slice and return segments to optimize.

        Returns:
            SliceResult with segment_indices for direct use in executor
        """
        if self.is_finished:
            return None

        self.iteration += 1

        # Get slice data
        slice_curve = self.curve[self.current_start_idx:self.end_idx + 1]
        slice_md = self.well_md[self.current_start_idx:self.end_idx + 1]

        if len(slice_curve) < 50 or np.isnan(slice_curve).all():
            self.is_finished = True
            return SliceResult(
                boundaries=[],
                segment_indices=[(self.current_start_idx, self.end_idx)],
                segment_mds=[(self.well_md[self.current_start_idx], self.well_md[self.end_idx])],
                next_start_idx=self.end_idx,
                is_final=True
            )

        # Handle NaN values - use valid portion only for detection
        valid_mask = ~np.isnan(slice_curve)
        if not valid_mask.all():
            # Find contiguous valid region
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) < 50:
                self.is_finished = True
                return SliceResult(
                    boundaries=[],
                    segment_indices=[(self.current_start_idx, self.end_idx)],
                    segment_mds=[(self.well_md[self.current_start_idx], self.well_md[self.end_idx])],
                    next_start_idx=self.end_idx,
                    is_final=True
                )
            # Use only valid portion
            valid_start = valid_indices[0]
            valid_end = valid_indices[-1] + 1
            slice_curve = slice_curve[valid_start:valid_end]
            slice_md = slice_md[valid_start:valid_end]
            idx_offset = self.current_start_idx + valid_start
        else:
            idx_offset = self.current_start_idx

        # Detect boundaries (recalculate for each slice)
        boundaries = get_segment_boundaries(
            slice_curve, slice_md,
            start_idx_offset=idx_offset,
            min_distance=self.min_distance,
            pelt_penalty=self.pelt_penalty,
            is_final_slice=True
        )

        # Take first N boundaries
        boundaries_for_slice = boundaries[:self.segments_count]

        # Check if final
        remaining = len(boundaries) - len(boundaries_for_slice)
        last_boundary_idx = boundaries_for_slice[-1].idx if boundaries_for_slice else self.current_start_idx
        gap_to_end = self.end_idx - last_boundary_idx

        is_final = remaining <= 0 or gap_to_end < self.min_segment_idx * 2

        # Create segments
        if is_final:
            slice_end_idx = self.end_idx
        else:
            slice_end_idx = last_boundary_idx

        segment_indices, segment_mds = self._create_segments(
            boundaries_for_slice, self.current_start_idx, slice_end_idx
        )

        # Calculate next start (with overlap)
        if is_final:
            next_start = self.end_idx
            self.is_finished = True
        else:
            if len(boundaries_for_slice) > self.overlap_segments:
                overlap_idx = len(boundaries_for_slice) - 1 - self.overlap_segments
                next_start = boundaries_for_slice[overlap_idx].idx
            else:
                next_start = last_boundary_idx

        # Store boundaries
        self.all_boundaries.extend(boundaries_for_slice)

        # Update state
        self.current_start_idx = next_start

        logger.info(f"Slice {self.iteration}: {len(segment_indices)} segments, "
                   f"next_start_idx={next_start}, is_final={is_final}")

        return SliceResult(
            boundaries=boundaries_for_slice,
            segment_indices=segment_indices,
            segment_mds=segment_mds,
            next_start_idx=next_start,
            is_final=is_final
        )

    def _create_segments(self, boundaries: List[InfoPoint],
                         start_idx: int, end_idx: int
                         ) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]]]:
        """Create segment index ranges from boundary points."""
        if not boundaries:
            return (
                [(start_idx, end_idx)],
                [(self.well_md[start_idx], self.well_md[end_idx])]
            )

        segment_indices = []
        segment_mds = []
        prev_idx = start_idx

        for bp in boundaries:
            if bp.idx > prev_idx:
                segment_indices.append((prev_idx, bp.idx))
                segment_mds.append((self.well_md[prev_idx], self.well_md[bp.idx]))
                prev_idx = bp.idx

        # Last segment to end
        if prev_idx < end_idx:
            gap_idx = end_idx - prev_idx
            if gap_idx >= self.min_segment_idx:
                segment_indices.append((prev_idx, end_idx))
                segment_mds.append((self.well_md[prev_idx], self.well_md[end_idx]))
            elif segment_indices:
                # Extend last segment
                last_start, _ = segment_indices[-1]
                segment_indices[-1] = (last_start, end_idx)
                segment_mds[-1] = (self.well_md[last_start], self.well_md[end_idx])
            else:
                segment_indices.append((prev_idx, end_idx))
                segment_mds.append((self.well_md[prev_idx], self.well_md[end_idx]))

        return segment_indices, segment_mds

    def get_final_segments(self) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]]]:
        """
        Get final N segments to well end for final optimization.
        """
        if not self.all_boundaries:
            return (
                [(self.start_idx, self.end_idx)],
                [(self.well_md[self.start_idx], self.well_md[self.end_idx])]
            )

        if len(self.all_boundaries) >= self.segments_count:
            final_boundaries = self.all_boundaries[-self.segments_count:]
            if len(self.all_boundaries) > self.segments_count:
                start_idx = self.all_boundaries[-self.segments_count - 1].idx
            else:
                start_idx = self.start_idx
        else:
            final_boundaries = self.all_boundaries
            start_idx = self.start_idx

        return self._create_segments(final_boundaries, start_idx, self.end_idx)


# ============ TEST ============
if __name__ == "__main__":
    import torch
    from dotenv import load_dotenv

    load_dotenv()

    # Load dataset
    dataset = torch.load(DATASET_PATH, weights_only=False)
    data = dataset["Well162~EGFDL"]

    print(f"Well162~EGFDL")
    print(f"well_md: {len(data['well_md'])} points, step={np.median(np.diff(data['well_md'].numpy())):.4f}m")
    print(f"log_md: {len(data['log_md'])} points")
    print()

    settings = get_settings()
    print("Settings:")
    for k, v in settings.items():
        print(f"  {k}: {v}")
    print()

    # Test SmartSegmenter
    print("=" * 60)
    print("SMART SEGMENTER TEST")
    print("=" * 60)

    segmenter = SmartSegmenter(data, settings)
    start_md = segmenter.well_md[segmenter.start_idx]

    all_segments = []
    iteration = 0

    while not segmenter.is_finished:
        iteration += 1
        result = segmenter.next_slice()

        print(f"\n--- Iteration {iteration} ---")
        print(f"Boundaries: {len(result.boundaries)}")
        for bp in result.boundaries:
            rel_md = bp.md - start_md
            print(f"  idx={bp.idx:5d}  {rel_md:6.1f}m  {bp.type:>7}  score={bp.score:.1f}")

        print(f"Segments: {len(result.segment_indices)}")
        for i, ((s_idx, e_idx), (s_md, e_md)) in enumerate(zip(result.segment_indices, result.segment_mds)):
            rel_s = s_md - start_md
            rel_e = e_md - start_md
            print(f"  Seg {i+1}: idx[{s_idx:5d}-{e_idx:5d}]  {rel_s:6.1f}-{rel_e:6.1f}m  ({e_md-s_md:.1f}m)")

        print(f"Next start idx: {result.next_start_idx}")
        print(f"Is final: {result.is_final}")

        all_segments.extend(result.segment_indices)

    print()
    print("=" * 60)
    print(f"TOTAL: {iteration} iterations, {len(all_segments)} segments")
    print("=" * 60)
