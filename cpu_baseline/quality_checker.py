import math
import json
import csv
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class QualityMetrics:
    """Структура для хранения метрик качества интерпретации для множественных порогов"""

    def __init__(self, thresholds: List[float]):
        self.max_deviation: float = 0.0
        self.rmse: float = 0.0
        self.mae: float = 0.0
        self.point_count: int = 0
        
        # Metrics for each threshold
        self.thresholds = thresholds
        self.threshold_exceeded: Dict[float, bool] = {t: False for t in thresholds}
        self.threshold_points: Dict[float, int] = {t: 0 for t in thresholds}
        self.coverage_percentage: Dict[float, float] = {t: 0.0 for t in thresholds}

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь для сериализации"""
        result = {
            "max_deviation": self.max_deviation,
            "rmse": self.rmse,
            "mae": self.mae,
            "point_count": self.point_count
        }
        
        # Add threshold-specific metrics
        for threshold in self.thresholds:
            t_str = f"{threshold:.1f}m"
            result[f"threshold_exceeded_{t_str}"] = self.threshold_exceeded[threshold]
            result[f"threshold_points_{t_str}"] = self.threshold_points[threshold]
            result[f"coverage_percentage_{t_str}"] = self.coverage_percentage[threshold]
        
        return result


class WellState:
    """Состояние скважины для отслеживания интерпретаций"""

    def __init__(self):
        self.interpretation_started: bool = False
        self.total_failures: int = 0


class InterpretationQualityAnalyzer:
    """Анализатор качества интерпретации с отслеживанием failures"""

    def __init__(self):
        # Load configuration from environment
        self.results_dir = os.getenv('RESULTS_DIR')
        self.wells_dir = os.getenv('WELLS_DIR')
        self.md_step = float(os.getenv('MD_STEP_METERS'))
        self.quality_steps = int(os.getenv('QUALITY_STEPS_COUNT'))
        
        # Parse multiple thresholds from comma-separated string
        thresholds_str = os.getenv('QUALITY_THRESHOLDS_METERS')
        if not thresholds_str:
            raise ValueError("QUALITY_THRESHOLDS_METERS not found in .env file")
        self.thresholds = [float(t.strip()) for t in thresholds_str.split(',')]

        # Calculate lookback distance for quality assessment
        self.quality_distance = self.md_step * self.quality_steps

        # Track well states
        self.well_states: Dict[str, WellState] = {}

        logger.info(f"Quality analyzer configuration:")
        logger.info(f"  MD step: {self.md_step} meters")
        logger.info(f"  Quality steps: {self.quality_steps}")
        logger.info(f"  Quality distance: {self.quality_distance} meters")
        logger.info(f"  Thresholds: {self.thresholds} meters")

    def analyze_all_results(self, output_csv: str = "quality_analysis.csv"):
        """Анализирует все результаты и сохраняет в CSV"""

        # Group result files by well name
        well_groups = self._group_result_files()

        if not well_groups:
            logger.warning(f"No result files found in {self.results_dir}")
            return

        # Filter wells with EGFDL in name
        egfdl_wells = {well_name: files for well_name, files in well_groups.items()
                       if 'EGFDL' in well_name}

        if not egfdl_wells:
            logger.warning(f"No wells with EGFDL found in {len(well_groups)} total wells")
            return

        logger.info(f"Found {len(egfdl_wells)} EGFDL wells to analyze (filtered from {len(well_groups)} total wells)")

        # Prepare CSV output
        csv_rows = []
        skipped_wells = []

        for well_name, step_files in egfdl_wells.items():
            logger.info(f"Analyzing well: {well_name} ({len(step_files)} steps)")

            # Initialize well state
            self.well_states[well_name] = WellState()

            # Load reference interpretation
            reference_data = self._load_reference_interpretation(well_name)
            if not reference_data:
                skipped_wells.append(f"{well_name} (no reference)")
                continue

            # Sort step files by MD
            sorted_files = sorted(step_files, key=lambda f: self._extract_md_from_filename(f))

            # Analyze each step in chronological order
            for step_file in sorted_files:
                step_data = self._load_step_data(step_file)
                assert step_data is not None, f"Failed to load step data from {step_file}"

                current_md = step_data['measured_depth']
                well_state = self.well_states[well_name]

                # Check interpretation status
                has_interpretation = self._has_valid_interpretation(step_data)

                if has_interpretation:
                    # Valid interpretation found
                    if not well_state.interpretation_started:
                        well_state.interpretation_started = True
                        logger.debug(f"Well {well_name}: interpretation started at MD {current_md}")

                    # Perform quality analysis
                    metrics = self._analyze_step(reference_data, step_data, well_name)

                    # Add normal row to CSV with multi-threshold data
                    csv_row = {
                        'well_name': well_name,
                        'step_md': current_md,
                        'rmse_lookback': metrics.rmse,
                        'max_deviation': metrics.max_deviation,
                        'point_count': metrics.point_count,
                        'is_failure': False
                    }
                    
                    # Add threshold-specific metrics
                    for threshold in self.thresholds:
                        t_str = f"{threshold:.1f}m"
                        csv_row[f'threshold_exceeded_{t_str}'] = metrics.threshold_exceeded[threshold]
                        csv_row[f'coverage_percentage_{t_str}'] = metrics.coverage_percentage[threshold]
                    
                    csv_rows.append(csv_row)

                else:
                    # No interpretation or invalid interpretation
                    if well_state.interpretation_started:
                        # This is a failure - interpretation was working before
                        well_state.total_failures += 1
                        logger.warning(f"Well {well_name}: interpretation failure at MD {current_md}")

                        # Add failure row to CSV with multi-threshold data
                        csv_row = {
                            'well_name': well_name,
                            'step_md': current_md,
                            'rmse_lookback': None,
                            'max_deviation': None,
                            'point_count': None,
                            'is_failure': True
                        }
                        
                        # Add threshold-specific failure metrics
                        for threshold in self.thresholds:
                            t_str = f"{threshold:.1f}m"
                            csv_row[f'threshold_exceeded_{t_str}'] = True  # All thresholds exceeded on failure
                            csv_row[f'coverage_percentage_{t_str}'] = None
                        
                        csv_rows.append(csv_row)
                    else:
                        # Initial None interpretations - ignore, don't write to CSV
                        logger.debug(f"Well {well_name}: initial None interpretation at MD {current_md}")

        # Save results to CSV
        self._save_csv_results(csv_rows, output_csv)

        # Report statistics
        self._report_statistics(csv_rows, skipped_wells)

        logger.info(f"Analysis complete. Results saved to {output_csv}")

    def _has_valid_interpretation(self, step_data: Dict[str, Any]) -> bool:
        """Проверяет, есть ли валидная интерпретация в данных шага"""

        interpretation = step_data['interpretation']

        # Check if interpretation is None
        if interpretation is None:
            return False

        # Check if interpretation structure exists
        if 'interpretation' not in interpretation:
            return False

        # Check if segments exist
        if 'segments' not in interpretation['interpretation']:
            return False

        segments = interpretation['interpretation']['segments']

        # Assert that segments is not empty if interpretation structure exists
        assert len(
            segments) > 0, f"Interpretation exists but segments is empty in step MD {step_data['measured_depth']}"

        return True

    def _group_result_files(self) -> Dict[str, List[Path]]:
        """Группирует файлы результатов по именам скважин"""
        results_path = Path(self.results_dir)

        if not results_path.exists():
            return {}

        well_groups = {}

        # Find all step files (not init files)
        for file_path in results_path.glob("*_step_*.json"):
            # Parse well name from filename: {well_name}_step_{md}.json
            parts = file_path.stem.split('_step_')
            if len(parts) == 2:
                well_name = parts[0]

                if well_name not in well_groups:
                    well_groups[well_name] = []
                well_groups[well_name].append(file_path)

        return well_groups

    def _extract_md_from_filename(self, file_path: Path) -> float:
        """Извлекает MD из имени файла"""
        # Extract MD from: {well_name}_step_{md}.json
        parts = file_path.stem.split('_step_')
        if len(parts) == 2:
            return float(parts[1])
        return 0.0

    def _load_reference_interpretation(self, well_name: str) -> Optional[Dict[str, Any]]:
        """Загружает эталонную интерпретацию для скважины"""
        reference_file = Path(self.wells_dir) / f"{well_name}.json"

        if not reference_file.exists():
            logger.warning(f"Reference file not found: {reference_file}")
            return None

        with open(reference_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Assert interpretation exists in reference
        assert 'interpretation' in data, f"No interpretation found in reference file: {reference_file}"
        assert 'segments' in data['interpretation'], f"No segments found in reference file: {reference_file}"
        assert len(data['interpretation']['segments']) > 0, f"Empty segments in reference file: {reference_file}"

        return data

    def _load_step_data(self, file_path: Path) -> Dict[str, Any]:
        """Загружает данные шага из файла результата"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Assert required fields exist
        assert 'measured_depth' in data, f"Missing 'measured_depth' in {file_path}"
        assert 'interpretation' in data, f"Missing 'interpretation' in {file_path}"

        return data

    def _analyze_step(
        self,
        reference_data: Dict[str, Any],
        step_data: Dict[str, Any],
        well_name: str
    ) -> QualityMetrics:
        """Анализирует качество одного шага"""

        current_md = step_data['measured_depth']

        # Extract interpretations
        reference_segments = reference_data['interpretation']['segments']
        computed_segments = step_data['interpretation']['interpretation']['segments']

        # Define quality assessment range: last quality_distance meters
        quality_start_md = current_md - self.quality_distance
        quality_end_md = current_md

        # Generate comparison points with 1-meter step
        comparison_points = self._prepare_comparison_points(
            reference_segments, computed_segments, quality_start_md, quality_end_md
        )

        if not comparison_points:
            return QualityMetrics(self.thresholds)

        # Calculate metrics
        return self._calculate_metrics(comparison_points, well_name, current_md)

    def _normalize_segments(self, segments: List[Dict], last_md: float) -> List[Dict]:
        """Normalize segments to ensure all have proper endMd."""
        if not segments:
            return []

        normalized = []
        for i, seg in enumerate(segments):
            norm_seg = seg.copy()

            start_md = seg.get('startMd')
            if start_md is None:
                continue

            end_md = seg.get('endMd')
            if end_md is None:
                if i + 1 < len(segments):
                    next_seg = segments[i + 1]
                    end_md = next_seg.get('startMd')
                else:
                    end_md = last_md
            norm_seg['endMd'] = end_md

            if 'startShift' not in norm_seg:
                norm_seg['startShift'] = seg.get('shift', 0)
            if 'endShift' not in norm_seg:
                norm_seg['endShift'] = seg.get('shift', 0)

            normalized.append(norm_seg)

        return normalized

    def _prepare_comparison_points(self,
                                   reference_segments: List[Dict[str, float]],
                                   computed_segments: List[Dict[str, float]],
                                   start_md: float,
                                   end_md: float) -> List[Tuple[float, float, float]]:
        """Подготавливает точки для сравнения с шагом 1 метр"""

        # Normalize segments to ensure proper endMd
        ref_normalized = self._normalize_segments(reference_segments, end_md)
        comp_normalized = self._normalize_segments(computed_segments, end_md)

        comparison_points = []

        # Generate points every 1 meter in the quality range
        current_md = start_md
        while current_md <= end_md:
            reference_shift = self._interpolate_shift(ref_normalized, current_md)
            computed_shift = self._interpolate_shift(comp_normalized, current_md)

            # Add point only if both values are available
            if reference_shift is not None and computed_shift is not None:
                comparison_points.append((current_md, reference_shift, computed_shift))

            current_md += 1.0  # 1-meter step

        return comparison_points

    def _interpolate_shift(self, segments: List[Dict[str, float]], md: float) -> Optional[float]:
        """Интерполирует значение сдвига для заданной MD"""

        if not segments:
            return None

        # Find segment containing the given MD
        for i, segment in enumerate(segments):
            start_md = segment['startMd']

            # Determine segment end - use endMd from segment, fallback to next segment's startMd
            end_md = segment.get('endMd')
            if end_md is None:
                if i + 1 < len(segments):
                    end_md = segments[i + 1]['startMd']
                else:
                    # Last segment without endMd - this should not happen after normalization
                    raise ValueError(f"Last segment at startMd={start_md} has no endMd. "
                                   f"Segments must be normalized before interpolation.")

            if start_md <= md < end_md:
                # Linear interpolation between startShift and endShift
                start_shift = segment['startShift']
                end_shift = segment['endShift']

                if start_md == end_md:
                    return start_shift

                # Interpolate
                ratio = (md - start_md) / (end_md - start_md)
                return start_shift + ratio * (end_shift - start_shift)

        return None

    def _log_comparison_table(
        self,
        comparison_points: List[Tuple[float, float, float]],
        well_name: str,
        current_md: float,
        sample_every: int = 5
    ):
        """Log comparison points in compact table format"""

        if not comparison_points:
            return

        # Calculate deviations for status symbols
        deviations = [abs(comp - ref) for (_, ref, comp) in comparison_points]
        max_dev = max(deviations)

        # Get thresholds (sorted)
        sorted_thresholds = sorted(self.thresholds)
        t1 = sorted_thresholds[0] if len(sorted_thresholds) >= 1 else 3.0
        t2 = sorted_thresholds[1] if len(sorted_thresholds) >= 2 else 4.0

        # Header
        logger.info("=" * 80)
        logger.info(f"Quality Analysis: {well_name} MD={current_md:.1f}m "
                    f"(Range: {comparison_points[0][0]:.1f} - {comparison_points[-1][0]:.1f}m)")
        logger.info("=" * 80)
        logger.info(f"  #  |    MD    | Ref Shift | Comp Shift |  Delta  | Status")
        logger.info("-" * 5 + "+" + "-" * 10 + "+" + "-" * 11 + "+" + "-" * 12 + "+" + "-" * 9 + "+" + "-" * 8)

        # Data rows (sample every Nth point)
        for i in range(0, len(comparison_points), sample_every):
            md, ref, comp = comparison_points[i]
            delta = abs(comp - ref)

            # Status symbol
            if delta > t2:
                status = "✗"
            elif delta > t1:
                status = "⚠"
            else:
                status = "✓"

            logger.info(f"{i:3d}  | {md:8.2f} | {ref:9.4f} | {comp:10.4f} | {delta:7.4f} | {status}")

        # Always show last point if not already shown
        if (len(comparison_points) - 1) % sample_every != 0:
            i = len(comparison_points) - 1
            md, ref, comp = comparison_points[i]
            delta = abs(comp - ref)
            status = "✗" if delta > t2 else "⚠" if delta > t1 else "✓"
            logger.info(f"{i:3d}  | {md:8.2f} | {ref:9.4f} | {comp:10.4f} | {delta:7.4f} | {status}")

        # Summary
        logger.info("=" * 80)
        import numpy as np
        dev_array = np.array(deviations)
        rmse = np.sqrt(np.mean(dev_array ** 2))
        logger.info(f"Summary: {len(comparison_points)} points, "
                    f"RMSE={rmse:.3f}m, Max={max_dev:.3f}m, Mean={np.mean(dev_array):.3f}m")

        # Threshold violations
        for threshold in self.thresholds:
            violations = sum(1 for d in deviations if d > threshold)
            coverage = ((len(deviations) - violations) / len(deviations)) * 100.0
            logger.info(f"Threshold {threshold:.1f}m: {violations} violations ({coverage:.1f}% coverage)")

        logger.info("=" * 80)

    def _calculate_metrics(
        self,
        comparison_points: List[Tuple[float, float, float]],
        well_name: str,
        current_md: float
    ) -> QualityMetrics:
        """Вычисляет метрики качества по точкам сравнения для всех порогов"""

        metrics = QualityMetrics(self.thresholds)

        if not comparison_points:
            return metrics

        # Extract deviations
        deviations = []
        for md, reference, computed in comparison_points:
            deviation = abs(computed - reference)
            deviations.append(deviation)

        metrics.point_count = len(comparison_points)

        # Log detailed comparison table
        self._log_comparison_table(comparison_points, well_name, current_md, sample_every=5)

        # DEBUG: Log comparison points stats
        logger.info(f"DEBUG: comparison_points count={len(comparison_points)}")
        if comparison_points:
            first_3 = comparison_points[:3]
            logger.info(f"DEBUG: First 3 points (md, ref, computed):")
            for md, ref, comp in first_3:
                logger.info(f"  MD={md:.2f}, ref={ref:.6f}, comp={comp:.6f}, dev={abs(comp-ref):.6f}")

        # DEBUG: Check for NaN/inf in deviations
        import numpy as np
        deviations_array = np.array(deviations)
        nan_count = np.isnan(deviations_array).sum()
        inf_count = np.isinf(deviations_array).sum()
        logger.info(f"DEBUG: Deviations stats:")
        logger.info(f"  count={len(deviations)}, NaN={nan_count}, inf={inf_count}")
        if len(deviations) > 0 and nan_count == 0 and inf_count == 0:
            logger.info(f"  min={np.min(deviations_array):.6f}, max={np.max(deviations_array):.6f}")
            logger.info(f"  mean={np.mean(deviations_array):.6f}, median={np.median(deviations_array):.6f}")
            # Check for extremely large values
            extreme_threshold = 1e10
            extreme_count = (deviations_array > extreme_threshold).sum()
            if extreme_count > 0:
                logger.error(f"DEBUG: Found {extreme_count} extreme deviations > {extreme_threshold}")
                extreme_indices = np.where(deviations_array > extreme_threshold)[0][:5]
                for idx in extreme_indices:
                    md, ref, comp = comparison_points[idx]
                    logger.error(f"  EXTREME at MD={md:.2f}: ref={ref:.6f}, comp={comp:.6f}, dev={deviations[idx]:.6e}")
        elif nan_count > 0 or inf_count > 0:
            logger.error(f"DEBUG: Found NaN or inf in deviations!")

        # Maximum deviation
        metrics.max_deviation = max(deviations)

        # RMSE (Root Mean Square Error)
        squared_deviations = [d ** 2 for d in deviations]
        metrics.rmse = math.sqrt(sum(squared_deviations) / len(squared_deviations))

        # MAE (Mean Absolute Error)
        metrics.mae = sum(deviations) / len(deviations)

        # Calculate metrics for each threshold
        for threshold in self.thresholds:
            threshold_violations = [d for d in deviations if d > threshold]
            metrics.threshold_exceeded[threshold] = len(threshold_violations) > 0
            metrics.threshold_points[threshold] = len(threshold_violations)
            
            # Coverage percentage (points within threshold)
            good_points = metrics.point_count - metrics.threshold_points[threshold]
            metrics.coverage_percentage[threshold] = (good_points / metrics.point_count) * 100.0

        return metrics

    def _save_csv_results(self, csv_rows: List[Dict[str, Any]], output_file: str):
        """Сохраняет результаты в CSV файл"""

        if not csv_rows:
            logger.warning("No data to save to CSV")
            return

        # Build fieldnames dynamically based on thresholds
        fieldnames = ['well_name', 'step_md', 'rmse_lookback', 'max_deviation', 'point_count', 'is_failure']
        
        # Add threshold-specific columns
        for threshold in self.thresholds:
            t_str = f"{threshold:.1f}m"
            fieldnames.extend([f'threshold_exceeded_{t_str}', f'coverage_percentage_{t_str}'])

        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        logger.info(f"Saved {len(csv_rows)} rows to {output_file}")

    def _report_statistics(self, csv_rows: List[Dict[str, Any]], skipped_wells: List[str]):
        """Отчет по статистике анализа"""

        if not csv_rows:
            logger.warning("No data for statistics")
            return

        # Calculate statistics
        total_steps = len(csv_rows)
        failure_steps = len([row for row in csv_rows if row['is_failure']])
        success_steps = total_steps - failure_steps

        wells_with_failures = set()
        for well_name, well_state in self.well_states.items():
            if well_state.total_failures > 0:
                wells_with_failures.add(well_name)

        success_percentage = (success_steps / total_steps) * 100.0 if total_steps > 0 else 0.0

        # Log statistics
        logger.info("=== ANALYSIS STATISTICS ===")
        logger.info(f"Total wells processed: {len(self.well_states)}")
        logger.info(f"Wells with failures: {len(wells_with_failures)}")
        logger.info(f"Total analysis steps: {total_steps}")
        logger.info(f"Successful steps: {success_steps}")
        logger.info(f"Failed steps: {failure_steps}")
        logger.info(f"Success rate: {success_percentage:.1f}%")

        if skipped_wells:
            logger.info(f"Skipped wells: {len(skipped_wells)}")
            for well in skipped_wells:
                logger.info(f"  - {well}")

        # Per-well failure details
        if wells_with_failures:
            logger.info("Wells with failures:")
            for well_name in wells_with_failures:
                failures = self.well_states[well_name].total_failures
                logger.info(f"  - {well_name}: {failures} failures")


def main():
    """Main entry point for quality analysis"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    analyzer = InterpretationQualityAnalyzer()
    analyzer.analyze_all_results()


if __name__ == "__main__":
    main()