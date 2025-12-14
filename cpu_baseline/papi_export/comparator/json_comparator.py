import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

# Import all comparator modules
from .base_comparator import BaseComparator
from .interpolator import DataInterpolator
from .delta_analyzer import DeltaAnalyzer
from .units_converter import UnitsConverter
from .segments_analyzer import SegmentsAnalyzer

logger = logging.getLogger(__name__)


class JSONComparator:
    """Main orchestrator for JSON comparison using specialized modules"""
    
    def __init__(self, tolerance: float = 0.001):
        """Initialize JSON comparator with all modules
        
        Args:
            tolerance: Numerical tolerance for comparison (in FEET after conversion)
        """
        self.tolerance = tolerance
        
        # Initialize all modules
        self.base_comparator = BaseComparator(tolerance)
        self.interpolator = DataInterpolator()
        self.delta_analyzer = DeltaAnalyzer(tolerance)
        self.units_converter = UnitsConverter()
        self.segments_analyzer = SegmentsAnalyzer(tolerance)
        
    def compare_json_files(self, reference_file: str, generated_file: str) -> Dict:
        """Compare two JSON files with conversion to FEET and proper interpolation
        
        Args:
            reference_file: Path to reference JSON file (in METERS)
            generated_file: Path to generated JSON file (in METERS)
            
        Returns:
            Comparison report dictionary (all values in FEET)
        """
        logger.info("="*80)
        logger.info("Starting JSON comparison")
        logger.info(f"  Reference: {reference_file}")
        logger.info(f"  Generated: {generated_file}")
        logger.info("="*80)
        
        # Load JSON files
        with open(reference_file, 'r', encoding='utf-8') as f:
            reference = json.load(f)
        with open(generated_file, 'r', encoding='utf-8') as f:
            generated = json.load(f)
        
        # Step 1: Convert BOTH files from METERS to FEET for StarSteer comparison
        logger.info("Step 1: Converting both files from METERS to FEET...")
        reference_feet, generated_feet = self.units_converter.convert_both_to_feet(
            reference, generated
        )
        
        # Step 2: Interpolate reference to match generated grids (now in FEET)
        logger.info("Step 2: Interpolating reference data to match generated grids (in FEET)...")
        reference_interpolated = self.interpolator.interpolate_to_target_grid(
            reference_feet, generated_feet
        )
        
        # Step 3: Create delta DataFrames for detailed analysis (in FEET)
        logger.info("Step 3: Creating delta analysis DataFrames (in FEET)...")
        delta_dataframes = self.delta_analyzer.create_all_delta_dataframes(
            reference_interpolated, generated_feet
        )
        
        # Log detailed delta analysis
        self.delta_analyzer.log_detailed_analysis()
        
        # Step 4: Analyze segments (in FEET)
        logger.info("Step 4: Analyzing interpretation segments (in FEET)...")
        segments_analysis = self.segments_analyzer.analyze_segments(
            reference_interpolated, generated_feet
        )
        
        # Step 5: Analyze shifts vs segments (if shifts present, in FEET)
        shifts_analysis = {}
        if 'shifts' in generated_feet:
            shifts_analysis = self.segments_analyzer.analyze_shifts_vs_segments(
                reference_interpolated, generated_feet
            )
        
        # Step 6: Perform standard object comparison (in FEET)
        logger.info("Step 6: Performing standard object comparison (in FEET)...")
        self.base_comparator.compare_objects(
            reference_interpolated, generated_feet, "root"
        )
        
        # Step 7: Analyze arrays (in FEET)
        self.base_comparator.analyze_arrays(reference_interpolated, generated_feet)
        
        # Step 8: Generate comprehensive report
        units_info = self.units_converter.get_units_info()
        report = self._generate_report(
            units_info,
            segments_analysis,
            shifts_analysis
        )
        
        # Print summary
        self._print_summary(report)
        
        return report
    
    def _generate_report(
        self,
        units_info: Dict,
        segments_analysis: Dict,
        shifts_analysis: Dict
    ) -> Dict:
        """Generate comprehensive comparison report
        
        Args:
            units_info: Units information from converter
            segments_analysis: Segments analysis results
            shifts_analysis: Shifts analysis results
            
        Returns:
            Complete report dictionary (all values in FEET)
        """
        # Get base report from base comparator
        base_report = self.base_comparator.generate_report()
        
        # Add additional analyses
        base_report['units_info'] = units_info
        base_report['segments_analysis'] = segments_analysis
        base_report['shifts_analysis'] = shifts_analysis
        base_report['interpolation_stats'] = self.interpolator.interpolation_stats
        
        # Add delta DataFrames summary
        delta_summary = {}
        for name, df in self.delta_analyzer.delta_dataframes.items():
            if not df.empty:
                delta_summary[name] = {
                    'count': len(df),
                    'has_significant_deltas': True
                }
            else:
                delta_summary[name] = {
                    'count': 0,
                    'has_significant_deltas': False
                }
        base_report['delta_summary'] = delta_summary
        
        return base_report
    
    def _print_summary(self, report: Dict):
        """Print enhanced summary to console
        
        Args:
            report: Complete comparison report (in FEET)
        """
        print("\n" + "="*80)
        print("ENHANCED JSON COMPARISON REPORT")
        print("="*80)
        
        # Units information
        if report.get('units_info'):
            units = report['units_info']
            print(f"📏 UNITS: Comparison performed in {units['comparison_units']}")
            print(f"  Both files converted from {units['input_units']} to {units['output_units']}")
            print(f"  All values below are in {units['comparison_units']}")
        
        # Interpolation coverage
        if report.get('interpolation_stats'):
            print("\n📊 INTERPOLATION COVERAGE (in FEET):")
            for name, stats in report['interpolation_stats'].items():
                coverage = stats.get('coverage_percent', 0)
                if coverage < 100:
                    print(f"  {name}: {coverage:.1f}% coverage")
                else:
                    print(f"  {name}: Full coverage")
        
        # Delta analysis summary
        if report.get('delta_summary'):
            print("\n🔍 DELTA ANALYSIS (in FEET):")
            for name, summary in report['delta_summary'].items():
                if summary['has_significant_deltas']:
                    print(f"  {name}: {summary['count']} significant deltas found")
                else:
                    print(f"  {name}: ✓ No significant deltas")
        
        # Shifts analysis
        if report.get('shifts_analysis'):
            shifts = report['shifts_analysis']
            print(f"\n🔀 SHIFTS ANALYSIS (in FEET):")
            print(f"  Match quality: {shifts.get('match_quality', 'N/A')}")
            if 'mean_difference' in shifts:
                print(f"  Mean difference: {shifts['mean_difference']:.6f} ft")
                print(f"  Std deviation: {shifts['std_difference']:.6f} ft")
        
        # Segments analysis
        if report.get('segments_analysis'):
            seg = report['segments_analysis']
            print(f"\n📋 SEGMENTS ANALYSIS (in FEET):")
            print(f"  Reference: {seg['reference_count']} segments")
            print(f"  Generated: {seg['generated_count']} segments")
            
            if seg['reference_count'] == seg['generated_count']:
                total = seg['reference_count']
                if total > 0:
                    match_pct = seg['fully_matching_segments'] / total * 100
                    print(f"  Matching: {seg['fully_matching_segments']}/{total} ({match_pct:.1f}%)")
                    
                    if seg['fully_matching_segments'] == total:
                        print("  ✅ ALL SEGMENTS MATCH!")
                    else:
                        mismatched = seg.get('mismatched_segments', [])
                        print(f"  ⚠️ {len(mismatched)} segments with differences > 0.05 ft")
            else:
                print("  ❌ SEGMENT COUNT MISMATCH!")
        
        # Overall result
        print("\n" + "="*60)
        if report['comparison_passed']:
            print("✅ COMPARISON PASSED - Files are identical within tolerance")
        else:
            print(f"❌ COMPARISON FAILED - {report['total_differences']} differences found")
        print(f"Tolerance: {report['tolerance']} ft")
        print("="*80)


def compare_json_files(reference_file: str, generated_file: str, tolerance: float = 0.001) -> Dict:
    """Main entry point for JSON comparison
    
    Args:
        reference_file: Path to reference JSON file (in METERS)
        generated_file: Path to generated JSON file (in METERS)
        tolerance: Numerical tolerance for comparison (in FEET after conversion)
        
    Returns:
        Comparison report dictionary (all values in FEET)
    """
    comparator = JSONComparator(tolerance)
    return comparator.compare_json_files(reference_file, generated_file)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python json_comparator.py <reference_file> <generated_file>")
        sys.exit(1)
    
    reference_file = sys.argv[1]
    generated_file = sys.argv[2]
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comparison
    report = compare_json_files(reference_file, generated_file)
    
    # Save report
    report_file = Path(generated_file).parent / 'comparison_report.json'
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nDetailed report saved to: {report_file}")
    
    # Exit with appropriate code
    sys.exit(0 if report['comparison_passed'] else 1)