"""
Real-time quality analysis for slicer using composition pattern.

Reuses logic from InterpretationQualityAnalyzer without code duplication.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

from quality_checker import InterpretationQualityAnalyzer, QualityMetrics

logger = logging.getLogger(__name__)


class SlicerQualityAnalyzer:
    """
    Real-time quality analyzer for slicing loop.

    Compares StarSteer (reference) vs Python (computed) interpretations
    on each iteration and saves metrics to CSV.

    Uses composition to reuse InterpretationQualityAnalyzer methods.
    """

    def __init__(self, config: Dict[str, Any], dip_range: float = None, lookback: float = None, smoothness: float = None):
        """
        Initialize real-time quality analyzer.

        Args:
            config: Configuration dict with keys:
                - RESULTS_DIR: Where to save CSV
                - MD_STEP_METERS: Step size
                - QUALITY_STEPS_COUNT: Lookback steps
                - QUALITY_THRESHOLDS_METERS: Comma-separated thresholds
            dip_range: Dip range parameter (for filename)
            lookback: Lookback distance parameter (for filename)
            smoothness: Smoothness parameter (for filename)
        """
        # Create batch analyzer instance to reuse its methods
        self._batch_analyzer = InterpretationQualityAnalyzer()

        # Build CSV filename with parameters
        results_dir = Path(config.get('RESULTS_DIR', 'results'))
        filename_parts = ["slicer_quality_analysis"]

        if dip_range is not None:
            filename_parts.append(f"dip{dip_range:.1f}")
        if lookback is not None:
            filename_parts.append(f"lookback{lookback:.1f}")
        if smoothness is not None:
            filename_parts.append(f"smooth{smoothness:.1f}")

        filename = "_".join(filename_parts) + ".csv"
        self.csv_path = results_dir / filename

        # Initialize CSV file (always recreate to start fresh)
        self._init_csv()

        logger.info(f"SlicerQualityAnalyzer initialized")
        logger.info(f"  Quality distance: {self._batch_analyzer.quality_distance}m")
        logger.info(f"  Thresholds: {self._batch_analyzer.thresholds}m")
        logger.info(f"  Parameters: dip_range={dip_range}, lookback={lookback}, smoothness={smoothness}")
        logger.info(f"  CSV output: {self.csv_path}")

    def _init_csv(self):
        """Initialize CSV file with headers (always recreate to start fresh)"""
        # Ensure results directory exists
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Build fieldnames dynamically based on thresholds
        fieldnames = [
            'well_name', 'step_md', 'rmse_lookback', 'max_deviation',
            'point_count', 'is_failure'
        ]

        # Add threshold-specific columns
        for threshold in self._batch_analyzer.thresholds:
            t_str = f"{threshold:.1f}m"
            fieldnames.extend([
                f'threshold_exceeded_{t_str}',
                f'coverage_percentage_{t_str}'
            ])

        # Always create CSV with headers (overwrite existing)
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

        logger.info(f"Created fresh CSV file: {self.csv_path}")

    def analyze_step(
        self,
        starsteer_segments: List[Dict[str, float]],
        python_segments: List[Dict[str, float]],
        current_md: float,
        well_name: str
    ) -> Optional[QualityMetrics]:
        """
        Analyze quality of Python interpretation vs StarSteer reference.

        Reuses _prepare_comparison_points and _calculate_metrics from
        InterpretationQualityAnalyzer.

        Args:
            starsteer_segments: Reference segments from StarSteer JSON
            python_segments: Computed segments from Python executor
            current_md: Current measured depth
            well_name: Well name for logging

        Returns:
            QualityMetrics object or None if no comparison possible
        """
        # Define quality assessment range: last quality_distance meters
        quality_start_md = current_md - self._batch_analyzer.quality_distance
        quality_end_md = current_md

        # Reuse batch analyzer's method to prepare comparison points
        comparison_points = self._batch_analyzer._prepare_comparison_points(
            starsteer_segments,
            python_segments,
            quality_start_md,
            quality_end_md
        )

        if not comparison_points:
            logger.warning(
                f"No comparison points for {well_name} at MD={current_md:.1f}m"
            )
            return None

        # DEBUG: Log segments info before metrics calculation
        logger.info(f"DEBUG: analyze_step for {well_name} at MD={current_md:.1f}m")
        logger.info(f"DEBUG: Quality range: {quality_start_md:.1f}m - {quality_end_md:.1f}m")
        logger.info(f"DEBUG: StarSteer segments: {len(starsteer_segments)}")
        logger.info(f"DEBUG: Python segments: {len(python_segments)}")
        if starsteer_segments:
            first_seg = starsteer_segments[0]
            md_start = first_seg.get('mdStart', 'None')
            md_end = first_seg.get('mdEnd', 'None')
            logger.info(f"DEBUG: First StarSteer segment: name={first_seg.get('name')}, "
                       f"mdStart={md_start}, mdEnd={md_end}")
        if python_segments:
            first_seg = python_segments[0]
            md_start = first_seg.get('mdStart', 'None')
            md_end = first_seg.get('mdEnd', 'None')
            logger.info(f"DEBUG: First Python segment: name={first_seg.get('name')}, "
                       f"mdStart={md_start}, mdEnd={md_end}")

        # Reuse batch analyzer's method to calculate metrics
        metrics = self._batch_analyzer._calculate_metrics(comparison_points, well_name, current_md)

        logger.debug(
            f"{well_name} MD={current_md:.1f}m: "
            f"RMSE={metrics.rmse:.3f}m, max_dev={metrics.max_deviation:.3f}m, "
            f"points={metrics.point_count}"
        )

        return metrics

    def append_to_csv(
        self,
        metrics: QualityMetrics,
        well_name: str,
        md: float,
        is_failure: bool = False
    ):
        """
        Append metrics to CSV file immediately.

        Args:
            metrics: QualityMetrics object from analyze_step
            well_name: Well name
            md: Measured depth
            is_failure: Whether interpretation failed (no segments)
        """
        # Build row dictionary
        csv_row = {
            'well_name': well_name,
            'step_md': md,
            'rmse_lookback': metrics.rmse if not is_failure else None,
            'max_deviation': metrics.max_deviation if not is_failure else None,
            'point_count': metrics.point_count if not is_failure else None,
            'is_failure': is_failure
        }

        # Add threshold-specific metrics
        for threshold in self._batch_analyzer.thresholds:
            t_str = f"{threshold:.1f}m"
            if is_failure:
                csv_row[f'threshold_exceeded_{t_str}'] = True
                csv_row[f'coverage_percentage_{t_str}'] = None
            else:
                csv_row[f'threshold_exceeded_{t_str}'] = metrics.threshold_exceeded[threshold]
                csv_row[f'coverage_percentage_{t_str}'] = metrics.coverage_percentage[threshold]

        # Append to CSV (using same fieldnames as in _init_csv)
        fieldnames = [
            'well_name', 'step_md', 'rmse_lookback', 'max_deviation',
            'point_count', 'is_failure'
        ]
        for threshold in self._batch_analyzer.thresholds:
            t_str = f"{threshold:.1f}m"
            fieldnames.extend([
                f'threshold_exceeded_{t_str}',
                f'coverage_percentage_{t_str}'
            ])

        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(csv_row)

        logger.debug(f"Appended quality metrics to CSV: {well_name} MD={md:.1f}m")

    def log_metrics(self, metrics: QualityMetrics, well_name: str, md: float):
        """
        Log quality metrics to console.

        Args:
            metrics: QualityMetrics object
            well_name: Well name
            md: Measured depth
        """
        logger.info(
            f"  Quality @ MD={md:.1f}m: "
            f"RMSE={metrics.rmse:.3f}m, "
            f"max_dev={metrics.max_deviation:.3f}m, "
            f"points={metrics.point_count}"
        )

        # Log threshold violations
        for threshold in self._batch_analyzer.thresholds:
            if metrics.threshold_exceeded[threshold]:
                coverage = metrics.coverage_percentage[threshold]
                logger.warning(
                    f"  Threshold {threshold:.1f}m exceeded! "
                    f"Coverage: {coverage:.1f}%"
                )
