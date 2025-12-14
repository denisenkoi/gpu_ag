import math
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseComparator:
    """Base class for JSON comparison with tolerance-based numerical comparison"""
    
    def __init__(self, tolerance: float = 0.001):
        """Initialize base comparator
        
        Args:
            tolerance: Numerical tolerance for float comparison
        """
        self.tolerance = tolerance
        self.differences = []
        self.array_analysis = {}
        
    def compare_objects(self, ref: Any, gen: Any, path: str):
        """Compare two objects recursively
        
        Args:
            ref: Reference object
            gen: Generated object
            path: Current path in object hierarchy
        """
        # Skip special fields from standard comparison
        if path in ["root.shifts", "root._metadata"]:
            return
            
        if type(ref) != type(gen):
            self.differences.append({
                'type': 'type_mismatch',
                'path': path,
                'reference_type': type(ref).__name__,
                'generated_type': type(gen).__name__,
                'reference_value': str(ref)[:100],
                'generated_value': str(gen)[:100]
            })
            return
            
        if isinstance(ref, dict):
            self._compare_dicts(ref, gen, path)
        elif isinstance(ref, list):
            self._compare_arrays(ref, gen, path)
        elif isinstance(ref, (int, float)):
            self._compare_numbers(ref, gen, path)
        elif ref != gen:
            self.differences.append({
                'type': 'value_difference',
                'path': path,
                'reference': ref,
                'generated': gen
            })
            
    def _compare_dicts(self, ref: Dict, gen: Dict, path: str):
        """Compare two dictionaries
        
        Args:
            ref: Reference dictionary
            gen: Generated dictionary
            path: Current path
        """
        # Check for missing keys in generated
        for key in ref:
            if key not in gen:
                # Don't report shifts as missing (it's expected in reference)
                if key != 'shifts':
                    self.differences.append({
                        'type': 'missing_key',
                        'path': f"{path}.{key}",
                        'key': key
                    })
            else:
                self.compare_objects(ref[key], gen[key], f"{path}.{key}")
                
        # Check for extra keys in generated
        for key in gen:
            if key not in ref:
                # Skip metadata and shifts fields
                if key in ['_metadata', 'shifts']:
                    continue
                self.differences.append({
                    'type': 'extra_key',
                    'path': f"{path}.{key}",
                    'key': key
                })
                
    def _compare_arrays(self, ref: List, gen: List, path: str):
        """Compare two arrays
        
        Args:
            ref: Reference array
            gen: Generated array
            path: Current path
        """
        if len(ref) != len(gen):
            self.differences.append({
                'type': 'array_length_mismatch',
                'path': path,
                'reference_length': len(ref),
                'generated_length': len(gen),
                'difference': len(gen) - len(ref)
            })
            
        # Compare elements up to the shorter length
        min_len = min(len(ref), len(gen))
        
        # Compare individual elements
        for i in range(min_len):
            self.compare_objects(ref[i], gen[i], f"{path}[{i}]")
            
    def _compare_numbers(self, ref: float, gen: float, path: str):
        """Compare two numbers with tolerance
        
        Args:
            ref: Reference number
            gen: Generated number
            path: Current path
        """
        # Handle NaN values
        if math.isnan(ref) and math.isnan(gen):
            return
        if math.isnan(ref) or math.isnan(gen):
            self.differences.append({
                'type': 'nan_difference',
                'path': path,
                'reference': ref,
                'generated': gen
            })
            return
            
        # Check numerical difference
        if abs(ref - gen) > self.tolerance:
            self.differences.append({
                'type': 'numerical_difference',
                'path': path,
                'reference': ref,
                'generated': gen,
                'difference': abs(ref - gen),
                'relative_error': abs(ref - gen) / abs(ref) if ref != 0 else float('inf')
            })
            
    def analyze_arrays(self, reference: Dict, generated: Dict):
        """Analyze arrays for range and boundary differences
        
        Args:
            reference: Reference data
            generated: Generated data
        """
        # Key arrays to analyze
        array_paths = [
            'well.points',
            'wellLog.points',
            'wellLog.tvdSortedPoints',
            'typeLog.points', 
            'typeLog.tvdSortedPoints',
            'tops',
            'gridSlice.points',
            'interpretation.segments'
        ]
        
        for path in array_paths:
            ref_array = self._get_nested_value(reference, path.split('.'))
            gen_array = self._get_nested_value(generated, path.split('.'))
            
            if ref_array and gen_array:
                self.array_analysis[path] = self._analyze_single_array(
                    ref_array, gen_array, path
                )
                
    def _analyze_single_array(self, ref_array: List, gen_array: List, path: str) -> Dict:
        """Analyze a single array for detailed comparison
        
        Args:
            ref_array: Reference array
            gen_array: Generated array
            path: Array path
            
        Returns:
            Analysis results dictionary
        """
        analysis = {
            'path': path,
            'reference_length': len(ref_array),
            'generated_length': len(gen_array),
            'length_match': len(ref_array) == len(gen_array),
            'length_difference': len(gen_array) - len(ref_array)
        }
        
        # Add first/last elements info
        if ref_array:
            analysis['reference_first'] = ref_array[0]
            analysis['reference_last'] = ref_array[-1]
            
        if gen_array:
            analysis['generated_first'] = gen_array[0]
            analysis['generated_last'] = gen_array[-1]
            
        return analysis
        
    def _get_nested_value(self, data: Dict, path: List[str]) -> Any:
        """Get nested value from dictionary using path
        
        Args:
            data: Source dictionary
            path: Path as list of keys
            
        Returns:
            Value at path or None
        """
        current = data
        for key in path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current
        
    def generate_report(self) -> Dict:
        """Generate comparison report
        
        Returns:
            Report dictionary with all findings
        """
        # Count differences by type
        diff_counts = {}
        for diff in self.differences:
            diff_type = diff['type']
            diff_counts[diff_type] = diff_counts.get(diff_type, 0) + 1
            
        # Identify critical differences
        critical_diffs = []
        for diff in self.differences:
            if diff['type'] in ['array_length_mismatch', 'missing_key', 'type_mismatch']:
                critical_diffs.append(diff)
                
        # Get numerical differences sample
        numerical_diffs = [d for d in self.differences if d['type'] == 'numerical_difference']
        
        report = {
            'comparison_passed': len(self.differences) == 0,
            'total_differences': len(self.differences),
            'tolerance': self.tolerance,
            'difference_counts': diff_counts,
            'critical_differences': critical_diffs,
            'numerical_differences_sample': numerical_diffs[:10],
            'array_analysis': self.array_analysis,
            'all_differences': self.differences,
            'timestamp': datetime.now().isoformat()
        }
        
        return report
        
    def print_summary(self, report: Dict):
        """Print comparison summary to console
        
        Args:
            report: Comparison report
        """
        print("\n" + "="*80)
        print("JSON COMPARISON REPORT")
        print("="*80)
        
        if report['comparison_passed']:
            print("✅ COMPARISON PASSED - Files are identical")
        else:
            print(f"❌ COMPARISON FAILED - {report['total_differences']} differences found")
            
        print(f"Numerical tolerance: {report['tolerance']}")
        
        # Print difference counts
        if report['difference_counts']:
            print(f"\nDifferences by category:")
            for diff_type, count in report['difference_counts'].items():
                print(f"  {diff_type}: {count} occurrences")
                
        # Array analysis summary
        if report['array_analysis']:
            print("\nArray analysis:")
            for path, analysis in report['array_analysis'].items():
                if not analysis['length_match']:
                    print(f"  {path}: ref={analysis['reference_length']}, "
                          f"gen={analysis['generated_length']} "
                          f"(diff={analysis['length_difference']})")
                          
        print("="*80)