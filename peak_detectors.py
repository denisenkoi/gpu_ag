"""
Peak detectors for finding informative regions in well curves.

Architecture:
- PeakDetector: base class for different detection strategies
- RegionFinder: uses detector to find best region for telescope lever
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d


@dataclass
class PeakInfo:
    """Information about a detected peak."""
    index: int
    md: float
    prominence: float
    is_significant: bool


@dataclass
class RegionResult:
    """Result of region detection."""
    best_md: float
    best_idx: int
    score: float
    significant_peaks: List[PeakInfo]
    all_peaks: List[PeakInfo]
    threshold: float
    detector_name: str


class PeakDetector(ABC):
    """Base class for peak detectors."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def compute_threshold(self, prominences: np.ndarray) -> float:
        """Compute threshold for significant peaks based on prominence distribution."""
        pass

    def detect(self, curve: np.ndarray, md: np.ndarray,
               start_idx: int, end_idx: int,
               smooth_window: int = 20) -> Tuple[List[PeakInfo], float]:
        """
        Detect peaks AND valleys in curve segment.

        Args:
            curve: value array (e.g., gamma ray)
            md: measured depth array
            start_idx: start index for search
            end_idx: end index for search
            smooth_window: smoothing window size

        Returns:
            (list of PeakInfo, threshold used)
        """
        # Work with segment
        segment = curve[start_idx:end_idx]
        segment_md = md[start_idx:end_idx]

        if len(segment) < smooth_window * 2:
            return [], 0.0

        # Smooth the curve
        smoothed = uniform_filter1d(segment.astype(float), size=smooth_window)

        # Find peaks (local maxima)
        peaks_up, props_up = find_peaks(smoothed, prominence=0, distance=smooth_window//2)

        # Find valleys (local minima) by inverting signal
        peaks_down, props_down = find_peaks(-smoothed, prominence=0, distance=smooth_window//2)

        # Combine all prominences for threshold calculation
        all_prominences = []
        if len(peaks_up) > 0:
            all_prominences.extend(props_up['prominences'])
        if len(peaks_down) > 0:
            all_prominences.extend(props_down['prominences'])

        if len(all_prominences) == 0:
            return [], 0.0

        all_prominences = np.array(all_prominences)

        # Compute threshold using detector-specific method
        threshold = self.compute_threshold(all_prominences)

        # Build peak info list (both peaks and valleys)
        peak_infos = []

        # Add peaks (maxima)
        if len(peaks_up) > 0:
            for peak_idx, prom in zip(peaks_up, props_up['prominences']):
                global_idx = start_idx + peak_idx
                peak_infos.append(PeakInfo(
                    index=global_idx,
                    md=segment_md[peak_idx],
                    prominence=prom,
                    is_significant=(prom > threshold)
                ))

        # Add valleys (minima)
        if len(peaks_down) > 0:
            for peak_idx, prom in zip(peaks_down, props_down['prominences']):
                global_idx = start_idx + peak_idx
                peak_infos.append(PeakInfo(
                    index=global_idx,
                    md=segment_md[peak_idx],
                    prominence=prom,
                    is_significant=(prom > threshold)
                ))

        # Sort by MD
        peak_infos.sort(key=lambda x: x.md)

        return peak_infos, threshold


class OtsuPeakDetector(PeakDetector):
    """
    Otsu's method for automatic threshold.
    Finds optimal threshold that minimizes within-class variance.
    Classic approach from image segmentation.
    """

    @property
    def name(self) -> str:
        return "Otsu"

    def compute_threshold(self, prominences: np.ndarray) -> float:
        if len(prominences) < 2:
            return 0.0

        # Normalize to 0-255 range for Otsu
        pmin, pmax = prominences.min(), prominences.max()
        if pmax - pmin < 1e-10:
            return pmin

        normalized = ((prominences - pmin) / (pmax - pmin) * 255).astype(np.uint8)

        # Otsu's algorithm
        hist, _ = np.histogram(normalized, bins=256, range=(0, 256))
        hist = hist.astype(float) / hist.sum()

        best_threshold = 0
        best_variance = 0

        for t in range(1, 256):
            # Class probabilities
            w0 = hist[:t].sum()
            w1 = hist[t:].sum()

            if w0 == 0 or w1 == 0:
                continue

            # Class means
            mu0 = (np.arange(t) * hist[:t]).sum() / w0
            mu1 = (np.arange(t, 256) * hist[t:]).sum() / w1

            # Between-class variance
            variance = w0 * w1 * (mu0 - mu1) ** 2

            if variance > best_variance:
                best_variance = variance
                best_threshold = t

        # Convert back to original scale
        threshold = pmin + (best_threshold / 255) * (pmax - pmin)
        return threshold


class MADPeakDetector(PeakDetector):
    """
    MAD-based detector using robust statistics.
    Threshold = median + k * MAD (Median Absolute Deviation)
    Resistant to outliers (single huge peak won't break it).
    """

    def __init__(self, k: float = 1.5):
        """
        Args:
            k: multiplier for MAD (1.5 is common, like 1.5*IQR for outliers)
        """
        self.k = k

    @property
    def name(self) -> str:
        return f"MAD(k={self.k})"

    def compute_threshold(self, prominences: np.ndarray) -> float:
        if len(prominences) < 2:
            return 0.0

        median = np.median(prominences)
        mad = np.median(np.abs(prominences - median))

        # Threshold: values above median + k*MAD are significant
        threshold = median + self.k * mad
        return threshold


class RollingStdDetector:
    """
    Detector based on rolling standard deviation.
    Finds regions with maximum amplitude variation (peaks + valleys).
    Does NOT use peak detection - measures overall "activity" of the curve.
    """

    def __init__(self, window_m: float = 100.0):
        """
        Args:
            window_m: window size in meters for rolling std calculation
        """
        self.window_m = window_m

    @property
    def name(self) -> str:
        return f"RollingStd(w={self.window_m}m)"

    def find_best_region(self, curve: np.ndarray, md: np.ndarray,
                         search_start_idx: int, search_end_idx: int,
                         region_length_m: float = 200.0,
                         smooth_window: int = 20) -> Tuple[float, int, float]:
        """
        Find region with maximum rolling std.

        Returns:
            (best_md, best_idx, score)
        """
        segment = curve[search_start_idx:search_end_idx]
        segment_md = md[search_start_idx:search_end_idx]

        if len(segment) < 100:
            mid = (search_start_idx + search_end_idx) // 2
            return md[mid], mid, 0.0

        # Smooth first to remove noise
        segment = uniform_filter1d(segment.astype(float), size=smooth_window)

        # Calculate MD step
        md_step = np.median(np.diff(segment_md))
        window_size = max(10, int(self.window_m / md_step))

        # Calculate rolling std
        # Use pandas-like rolling with numpy
        rolling_std = np.zeros(len(segment))
        for i in range(window_size, len(segment)):
            rolling_std[i] = np.std(segment[i-window_size:i])

        # Find region with maximum average rolling std
        region_size = int(region_length_m / md_step)

        best_score = 0
        best_center_idx = search_start_idx + len(segment) // 2

        for center in range(region_size // 2, len(segment) - region_size // 2):
            region_start = center - region_size // 2
            region_end = center + region_size // 2
            score = np.mean(rolling_std[region_start:region_end])

            # Add position bonus (prefer closer to end)
            position_bonus = center / len(segment)
            score = score * (1 + 0.2 * position_bonus)

            if score > best_score:
                best_score = score
                best_center_idx = search_start_idx + center

        return md[best_center_idx], best_center_idx, best_score


class RegionFinder:
    """
    Finds best region for telescope lever using peak detector.
    """

    def __init__(self, detector: PeakDetector, search_fraction: float = 0.33):
        """
        Args:
            detector: peak detector to use
            search_fraction: fraction of well to search (from end). 0.33 = right 1/3
        """
        self.detector = detector
        self.search_fraction = search_fraction

    def find_best_region(self, curve: np.ndarray, md: np.ndarray,
                         region_length_m: float = 200.0,
                         start_md: Optional[float] = None) -> RegionResult:
        """
        Find best region for telescope lever.

        Args:
            curve: value array
            md: measured depth array
            region_length_m: size of region to evaluate (meters)
            start_md: optional start MD (e.g., end of manual interpretation)

        Returns:
            RegionResult with best region info
        """
        # Determine search range (right fraction of well, or after start_md)
        total_length = md[-1] - md[0]

        if start_md is not None:
            search_start_md = max(start_md, md[-1] - total_length * self.search_fraction)
        else:
            search_start_md = md[-1] - total_length * self.search_fraction

        # Find indices
        search_start_idx = np.searchsorted(md, search_start_md)
        search_end_idx = len(md)

        # Detect peaks
        all_peaks, threshold = self.detector.detect(
            curve, md, search_start_idx, search_end_idx
        )

        significant_peaks = [p for p in all_peaks if p.is_significant]

        if len(significant_peaks) == 0:
            # No significant peaks - return middle of search region
            mid_idx = (search_start_idx + search_end_idx) // 2
            return RegionResult(
                best_md=md[mid_idx],
                best_idx=mid_idx,
                score=0.0,
                significant_peaks=[],
                all_peaks=all_peaks,
                threshold=threshold,
                detector_name=self.detector.name
            )

        # Find region with maximum density of significant peaks
        # Slide window and count significant peaks
        md_step = np.median(np.diff(md))
        window_size_idx = int(region_length_m / md_step)

        best_score = 0
        best_center_idx = search_start_idx

        sig_peak_indices = np.array([p.index for p in significant_peaks])

        for center_idx in range(search_start_idx + window_size_idx // 2,
                                search_end_idx - window_size_idx // 2):
            # Count significant peaks in window
            window_start = center_idx - window_size_idx // 2
            window_end = center_idx + window_size_idx // 2

            peaks_in_window = np.sum(
                (sig_peak_indices >= window_start) & (sig_peak_indices < window_end)
            )

            # Score: number of peaks + bonus for being closer to end
            position_bonus = (center_idx - search_start_idx) / (search_end_idx - search_start_idx)
            score = peaks_in_window + 0.5 * position_bonus

            if score > best_score:
                best_score = score
                best_center_idx = center_idx

        return RegionResult(
            best_md=md[best_center_idx],
            best_idx=best_center_idx,
            score=best_score,
            significant_peaks=significant_peaks,
            all_peaks=all_peaks,
            threshold=threshold,
            detector_name=self.detector.name
        )


def test_detectors_on_well(well_data: dict, well_name: str):
    """
    Test both detectors on a well and print results.

    Args:
        well_data: dict with 'md', 'value' arrays
        well_name: name for logging
    """
    md = well_data['md']
    curve = well_data['value']

    print(f"\n{'='*60}")
    print(f"Testing peak detectors on {well_name}")
    print(f"{'='*60}")
    print(f"MD range: {md[0]:.1f} - {md[-1]:.1f} m")
    print(f"Search region (right 1/3): {md[-1] - (md[-1]-md[0])*0.33:.1f} - {md[-1]:.1f} m")

    # Meters to feet conversion
    M_TO_FT = 3.28084

    # Peak-based detectors
    peak_detectors = [
        OtsuPeakDetector(),
        MADPeakDetector(k=1.5),
    ]

    results = []

    # region_length = lookback from best params (Trial 11)
    lookback_m = 250

    for detector in peak_detectors:
        finder = RegionFinder(detector, search_fraction=0.33)
        result = finder.find_best_region(curve, md, region_length_m=lookback_m)
        results.append(result)

        print(f"\n--- {detector.name} (peaks + valleys) ---")
        print(f"Threshold: {result.threshold:.4f}")
        print(f"All peaks/valleys: {len(result.all_peaks)}")
        print(f"Significant: {len(result.significant_peaks)}")
        print(f"Best region: {result.best_md:.1f} m = {result.best_md * M_TO_FT:.0f} ft")
        print(f"Score: {result.score:.2f}")

        if result.significant_peaks:
            print(f"Significant MDs (first 10): ", end="")
            peak_mds = [f"{p.md*M_TO_FT:.0f}" for p in result.significant_peaks[:10]]
            print(", ".join(peak_mds) + " ft" + ("..." if len(result.significant_peaks) > 10 else ""))

    # RollingStd detector (different interface)
    print(f"\n--- RollingStd (variance-based) ---")
    total_length = md[-1] - md[0]
    search_start_md = md[-1] - total_length * 0.33
    search_start_idx = np.searchsorted(md, search_start_md)
    search_end_idx = len(md)

    rolling_detector = RollingStdDetector(window_m=100.0)
    best_md, best_idx, score = rolling_detector.find_best_region(
        curve, md, search_start_idx, search_end_idx, region_length_m=lookback_m
    )
    print(f"Best region: {best_md:.1f} m = {best_md * M_TO_FT:.0f} ft")
    print(f"Score (avg std): {score:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (Best regions in feet):")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r.detector_name}: {r.best_md * M_TO_FT:.0f} ft")
    print(f"  {rolling_detector.name}: {best_md * M_TO_FT:.0f} ft")
    print(f"\n  Your manual pick: ~19,500-20,000 ft (5944-6096 m)")

    return results


if __name__ == "__main__":
    import torch

    # Load dataset
    dataset_path = "/mnt/e/Projects/Rogii/gpu_ag/dataset/gpu_ag_dataset.pt"
    print(f"Loading dataset from {dataset_path}")
    dataset = torch.load(dataset_path, weights_only=False)

    # Test on Well162~EGFDL
    well_name = "Well162~EGFDL"

    if well_name in dataset:
        w = dataset[well_name]
        # Dataset structure: log_md, log_gr (raw curve)
        md = w['log_md'].numpy() if hasattr(w['log_md'], 'numpy') else np.array(w['log_md'])
        curve = w['log_gr'].numpy() if hasattr(w['log_gr'], 'numpy') else np.array(w['log_gr'])

        well_data = {
            'md': md,
            'value': curve,
            'start_md': w.get('start_md', md[0]),
        }
        print(f"Loaded well: MD {md.min():.1f} - {md.max():.1f} m, {len(md)} points")
        print(f"Start MD (manual interp end): {well_data['start_md']:.1f} m")

        test_detectors_on_well(well_data, well_name)
    else:
        print(f"Well {well_name} not found in dataset")
        print(f"Available wells: {list(dataset.keys())[:10]}...")
