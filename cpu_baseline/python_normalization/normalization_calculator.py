# python_normalization/normalization_calculator.py

import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from dotenv import load_dotenv
from scipy.optimize import minimize
import logging
import matplotlib.pyplot as plt

# Import AG objects for projection calculations
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json, trim_segments_to_range, trim_segments_to_tvt_range

logger = logging.getLogger(__name__)


class NormalizationCalculator:
    """
    Simplified normalization using manual interpretation instead of automatic segment search.
    Uses human expertise from manual interpretation to calculate normalization coefficients.
    """

    def __init__(self, interactive_mode: bool = True):
        load_dotenv()

        # Load parameters from .env
        self.min_normalization_length = float(os.environ['MIN_NORMALIZATION_LENGTH_METERS'])
        self.max_normalization_length = float(os.environ['MAX_NORMALIZATION_LENGTH_METERS'])

        # Store interactive mode setting
        self.interactive_mode = interactive_mode

        # Store figure reference for reusing window
        self.normalization_figure = None

        logger.info(f"NormalizationCalculator initialized: "
                    f"min_length={self.min_normalization_length}m, "
                    f"max_length={self.max_normalization_length}m, "
                    f"interactive_mode={self.interactive_mode}")

    def calculate_normalization_coefficients(self,
                                             well_data: Dict[str, Any],
                                             well: Well,
                                             typewell: TypeWell,
                                             manual_segments: List,
                                             landing_end_md: float) -> Dict[str, Any]:
        """
        Calculate normalization coefficients using manual interpretation.
        Now works with pre-created AG objects and modifies well.value directly.

        Args:
            well_data: Complete well JSON data from emulator (for metadata only)
            well: Pre-created Well object with cleaned data
            typewell: Pre-created TypeWell object with cleaned data
            manual_segments: Pre-created manual interpretation segments
            landing_end_md: End of landing section from LandingDetector

        Returns:
            Dictionary with normalization results and metadata
        """
        well_name = well_data['wellName']
        logger.info(f"Starting normalization calculation for well: {well_name}")

        # STARSTEER-50: Normalization range around perch point [perch-150, perch+50]
        # landing_end_md is now perch_md (landing end point)
        norm_start_md = landing_end_md - 150.0  # 150m before perch
        norm_end_md = landing_end_md + 50.0     # 50m after perch
        logger.info(f"Normalization range: [{norm_start_md:.1f}, {norm_end_md:.1f}]m (perch={landing_end_md:.1f}m)")

        # Filter segments to normalization range [perch-150, perch+50]
        filtered_segments = trim_segments_to_range(
            manual_segments,
            norm_start_md,
            norm_end_md,
            well
        )

        if not filtered_segments:
            return self._create_failure_result(well_name,
                                               "no_segments_in_range",
                                               start_md=norm_start_md,
                                               landing_end_md=norm_end_md)

        # Calculate total normalization length
        normalization_length = filtered_segments[-1].end_md - filtered_segments[0].start_md

        # Check minimum length requirement
        if normalization_length < self.min_normalization_length:
            return self._create_failure_result(
                well_name, "insufficient_length",
                available_length=normalization_length,
                required_length=self.min_normalization_length
            )

        # Apply maximum length constraint if needed
        if normalization_length > self.max_normalization_length:
            # Filter to last max_normalization_length meters before landing_end_md
            normalization_start_md = landing_end_md - self.max_normalization_length
            filtered_segments = trim_segments_to_range(
                filtered_segments,
                normalization_start_md,
                landing_end_md,
                well
            )
            normalization_length = filtered_segments[-1].end_md - filtered_segments[0].start_md
            logger.info(f"Trimmed normalization length to maximum: {normalization_length}m")

        # Get tvd_to_typewell_shift from well_data
        tvd_to_typewell_shift = well_data['tvdTypewellShift']

        for segment in filtered_segments:
            well.calc_segment_tvt(typewell, segment, tvd_to_typewell_shift)

        filtered_segments = trim_segments_to_tvt_range(filtered_segments, np.min(typewell.tvd), np.max(typewell.tvd), well)

        if not filtered_segments:
            return self._create_failure_result(well_name,
                                               "no_segments_in_range",
                                               start_md=start_md,
                                               landing_end_md=landing_end_md)

        # Calculate horizontal projection using manual interpretation
        # Note: This modifies well.tvt and well.synt_curve arrays
        projection_success = well.calc_horizontal_projection(typewell, filtered_segments, tvd_to_typewell_shift)

        # Optional: Export for debugging
        # well.export_to_csv(f"debug_well_{well_name}_projection.csv", use_feet=True)

        if not projection_success:
            return self._create_failure_result(well_name, "projection_calculation_failed")

        # Extract curves for optimization
        measured_curve, reference_curve = self._extract_curves_for_optimization(well, filtered_segments)

        if len(measured_curve) == 0 or len(reference_curve) == 0:
            return self._create_failure_result(well_name, "empty_curves_after_projection")

        # Optimize normalization coefficients
        scale, mask = normalize_gamma(measured_curve, reference_curve,
                                      win_len=150, step=10,
                                      r_thr=0.7, std_thr=4.0, top_k=3)

        optimal_shift = 0  # We only do scaling, no shifting

        # IMPORTANT: Apply normalization directly to well.value array
        # This modifies the Well object in place
        logger.info(f"Applying normalization to well.value: multiplier={scale:.6f}")

        # Store original values for logging
        original_min = well.value.min()
        original_max = well.value.max()
        original_mean = well.value.mean()

        # Apply normalization to the entire well.value array
        well.value = well.value * scale + optimal_shift

        # Update min/max curve values in Well object
        well.min_curve = well.value.min()
        well.max_curve = well.value.max()

        # Log the modification
        logger.info(f"Modified well.value statistics:")
        logger.info(f"  Original: min={original_min:.3f}, max={original_max:.3f}, mean={original_mean:.3f}")
        logger.info(f"  Modified: min={well.min_curve:.3f}, max={well.max_curve:.3f}, mean={well.value.mean():.3f}")

        # Recalculate normalized curve for quality metrics
        # (using the segment range, not the full well)
        normalized_curve = measured_curve * scale

        # Calculate quality metrics
        pearson_correlation = self._calculate_pearson_correlation(
            normalized_curve, reference_curve
        )
        euclidean_distance = self._calculate_euclidean_distance(
            normalized_curve, reference_curve
        )

        # Visualize normalization result only in interactive mode
        if self.interactive_mode:
            self._visualize_normalization_result(
                measured_curve,
                normalized_curve,
                reference_curve,
                well_name,
                scale
            )

        logger.info(f"Normalization successful: multiplier={scale:.6f}, "
                    f"shift={optimal_shift:.6f}, pearson={pearson_correlation:.6f}")

        return {
            'status': 'success',
            'well_name': well_name,
            'landing_end_md': landing_end_md,
            'normalization_length': normalization_length,
            'multiplier': scale,
            'shift': optimal_shift,
            'pearson_correlation': pearson_correlation,
            'euclidean_distance': euclidean_distance,
            'segments_count': len(filtered_segments),
            'well_modified': True  # Flag indicating Well object was modified in place
        }

    def _extract_curves_for_optimization(self, well: Well, segments: List) -> Tuple[np.ndarray, np.ndarray]:
        """Extract measured and reference curves from segments range"""
        if not segments:
            return np.array([]), np.array([])

        # Get start and end indices from first and last segments
        start_idx = segments[0].start_idx
        end_idx = segments[-1].end_idx

        # Extract curves in segments range
        measured_curve = well.value[start_idx:end_idx + 1]
        reference_curve = well.synt_curve[start_idx:end_idx + 1]

        # Remove NaN values
        valid_mask = ~(np.isnan(measured_curve) | np.isnan(reference_curve))
        measured_curve = measured_curve[valid_mask]
        reference_curve = reference_curve[valid_mask]

        return measured_curve, reference_curve

    def _visualize_normalization_result(self, measured_curve: np.ndarray, normalized_curve: np.ndarray,
                                        reference_curve: np.ndarray, well_name: str, multiplier: float):
        """Visualize normalization result with all three curves on one plot"""
        if not self.interactive_mode:
            logger.info("Headless mode - skipping visualization")
            return

        if len(measured_curve) == 0 or len(reference_curve) == 0:
            logger.warning("Cannot visualize empty curves")
            return

        # Get screen dimensions
        import tkinter as tk
        root = tk.Tk()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()

        # Clear previous figure if exists
        if self.normalization_figure is not None:
            plt.close(self.normalization_figure)

        # Left half positioning
        window_width = screen_width // 2 - 100  # Half width minus margins
        window_height = int(screen_height * 0.7)  # 70% of screen height

        # Create new figure and store reference
        self.normalization_figure = plt.figure(figsize=(window_width / 100, window_height / 100))

        # Create point indices for X axis
        points = np.arange(len(measured_curve))

        # Plot all three curves
        plt.plot(points, measured_curve, 'b-', linewidth=2, label='Original Curve (Before Normalization)', alpha=0.8)
        plt.plot(points, normalized_curve, 'r-', linewidth=2, label='Normalized Curve (After Normalization)', alpha=0.8)
        plt.plot(points, reference_curve, 'k-', linewidth=2, label='Reference Curve (Type Well Projection)', alpha=0.8)

        plt.xlabel('Point Index', fontsize=12)
        plt.ylabel('Log Value', fontsize=12)
        plt.title(f'Normalization Result - {well_name}\nMultiplier: {multiplier:.6f}, Points: {len(measured_curve)}',
                  fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)

        # Add statistics as text
        stats_text = f"""Statistics:
    Original: min={np.min(measured_curve):.3f}, max={np.max(measured_curve):.3f}, mean={np.mean(measured_curve):.3f}
    Normalized: min={np.min(normalized_curve):.3f}, max={np.max(normalized_curve):.3f}, mean={np.mean(normalized_curve):.3f}
    Reference: min={np.min(reference_curve):.3f}, max={np.max(reference_curve):.3f}, mean={np.mean(reference_curve):.3f}
    Correlation (normalized vs reference): {np.corrcoef(normalized_curve, reference_curve)[0, 1]:.3f}"""

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                 verticalalignment='top', fontsize=10, fontfamily='monospace')

        plt.tight_layout()

        # Position window on left side - handle different backends
        manager = plt.get_current_fig_manager()
        backend = plt.get_backend().lower()

        if 'qt' in backend:
            # Qt backend
            manager.window.setGeometry(50, 50, window_width, window_height)
        elif 'tk' in backend:
            # Tkinter backend
            manager.window.wm_geometry(f"{window_width}x{window_height}+50+50")
        elif 'agg' in backend:
            # Headless mode - skip window positioning
            logger.info("Running in headless mode - skipping window positioning")
        else:
            # Unknown backend - log and skip
            logger.warning(f"Unknown backend '{backend}' - skipping window positioning")

        logger.info(f"Showing normalization result visualization for {well_name}")
        plt.show(block=False)

    def _calculate_pearson_correlation(self, curve1: np.ndarray, curve2: np.ndarray) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(curve1) == 0 or len(curve2) == 0:
            return 0.0

        correlation_matrix = np.corrcoef(curve1, curve2)
        return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0

    def _calculate_euclidean_distance(self, curve1: np.ndarray, curve2: np.ndarray) -> float:
        """Calculate euclidean distance between curves"""
        if len(curve1) == 0 or len(curve2) == 0:
            return float('inf')

        return float(np.linalg.norm(curve1 - curve2))

    def _create_failure_result(self, well_name: str, issue: str, **kwargs) -> Dict[str, Any]:
        """Create standardized failure result dictionary"""
        result = {
            'status': 'failed',
            'well_name': well_name,
            'issue': issue,
            'landing_end_md': kwargs.get('landing_end_md', ''),
            'normalization_length': kwargs.get('available_length', ''),
            'multiplier': kwargs.get('multiplier', ''),
            'shift': kwargs.get('shift', ''),
            'pearson_correlation': '',
            'euclidean_distance': '',
            'issue_description': self._get_issue_description(issue, **kwargs)
        }

        logger.warning(f"Normalization failed for {well_name}: {result['issue_description']}")
        return result

    def _get_issue_description(self, issue: str, **kwargs) -> str:
        """Generate human-readable issue description"""
        if issue == "no_segments_in_range":
            return f"No manual interpretation segments in range [{kwargs.get('start_md', '?'):.1f}, {kwargs.get('landing_end_md', '?'):.1f}]"
        elif issue == "insufficient_length":
            return f"Insufficient normalization length: {kwargs.get('available_length', '?'):.1f}m < {kwargs.get('required_length', '?'):.1f}m required"
        elif issue == "negative_multiplier":
            return f"Negative multiplier: {kwargs.get('multiplier', '?'):.6f}"
        elif issue == "projection_calculation_failed":
            return "Horizontal projection calculation failed"
        elif issue == "empty_curves_after_projection":
            return "Empty curves after projection calculation"
        else:
            return f"Unknown issue: {issue}"


from scipy.stats import pearsonr
from sklearn.linear_model import HuberRegressor
from scipy.signal import savgol_filter


def find_best_windows(
        measured: np.ndarray,
        reference: np.ndarray,
        *,  # именованные
        win_len: int = 200,  # длина окна, сэмплов
        step: int = 20,  # смещение окна
        r_thr: float = 0.6,
        std_thr: float = 5.0,
        top_k: int = 5
):
    """Возвращает индексы точек, попавших в k лучших окон."""

    n = len(measured)
    candidates = []

    for start in range(0, n - win_len + 1, step):
        end = start + win_len
        m_seg = measured[start:end]
        r_seg = reference[start:end]
        # Pearson r и контраст (std)
        r_val, _ = pearsonr(m_seg, r_seg)
        if abs(r_val) < r_thr or np.std(r_seg) < std_thr:
            continue
        candidates.append((abs(r_val), start, end))

    # сортируем по |r|, берём top_k
    candidates.sort(reverse=True)
    chosen = candidates[:top_k]

    mask = np.zeros(n, dtype=bool)
    for _, s, e in chosen:
        mask[s:e] = True
    return mask


def scale_only_robust(measured_sel, reference_sel):
    """Huber‑регрессия без сдвига (интерсепт=0)"""
    # Check if we have enough data points
    if len(measured_sel) < 1 or len(reference_sel) < 1:
        logger.warning(f"Insufficient data for HuberRegressor: measured={len(measured_sel)}, reference={len(reference_sel)}, returning scale=1.0")
        return 1.0
    
    huber = HuberRegressor(fit_intercept=False, alpha=0.0, epsilon=1.35)
    huber.fit(measured_sel.reshape(-1, 1), reference_sel)
    return float(huber.coef_[0])


def normalize_gamma(
        measured: np.ndarray,
        reference: np.ndarray,
        smooth_window: int = 11,
        smooth_poly: int = 3,
        **win_kwargs  # параметры выбора окна
):
    # 1) сглаживаем, чтобы окно не «увидело» высокочастотный шум
    m_sm = savgol_filter(measured, smooth_window, smooth_poly, mode="nearest")
    r_sm = savgol_filter(reference, smooth_window, smooth_poly, mode="nearest")

    # 2) находим «хорошие» окна
    mask = find_best_windows(m_sm, r_sm, **win_kwargs)

    # 3) оцениваем масштаб robast‑МИНСК без сдвига
    scale = scale_only_robust(m_sm[mask], r_sm[mask])

    return scale, mask