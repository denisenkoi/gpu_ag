import logging
import numpy as np
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SegmentsAnalyzer:
    """Analyzes interpretation segments and shifts data"""
    
    def __init__(self, tolerance: float = 0.001):
        """Initialize segments analyzer
        
        Args:
            tolerance: Tolerance for segment comparison
        """
        self.tolerance = tolerance
        self.segments_analysis = {}
        self.shifts_analysis = {}
        
    def analyze_segments(self, reference: Dict, generated: Dict) -> Dict:
        """Analyze interpretation segments comparison
        
        Args:
            reference: Reference data
            generated: Generated data
            
        Returns:
            Analysis results dictionary
        """
        ref_segments = self._get_nested_value(reference, ['interpretation', 'segments'])
        gen_segments = self._get_nested_value(generated, ['interpretation', 'segments'])
        
        if not ref_segments or not gen_segments:
            logger.warning("Missing segments data for comparison")
            return {}
        
        # Detailed segment-by-segment comparison
        segment_details = []
        matching_count = 0
        md_matching_count = 0
        shift_matching_count = 0
        mismatched_segments = []
        
        max_segments = max(len(ref_segments), len(gen_segments))
        MISMATCH_TOLERANCE = 0.05  # Tolerance for detailed display
        
        for i in range(max_segments):
            detail = {'index': i}
            
            # Get reference segment
            if i < len(ref_segments):
                ref_seg = ref_segments[i]
                detail['ref_startMd'] = ref_seg.get('startMd', 'N/A')
                detail['ref_startShift'] = ref_seg.get('startShift', 'N/A')
                detail['ref_endShift'] = ref_seg.get('endShift', 'N/A')
            else:
                detail['ref_startMd'] = 'MISSING'
                detail['ref_startShift'] = 'MISSING'
                detail['ref_endShift'] = 'MISSING'
            
            # Get generated segment
            if i < len(gen_segments):
                gen_seg = gen_segments[i]
                detail['gen_startMd'] = gen_seg.get('startMd', 'N/A')
                detail['gen_startShift'] = gen_seg.get('startShift', 'N/A')
                detail['gen_endShift'] = gen_seg.get('endShift', 'N/A')
            else:
                detail['gen_startMd'] = 'MISSING'
                detail['gen_startShift'] = 'MISSING'
                detail['gen_endShift'] = 'MISSING'
            
            # Calculate differences if both exist
            if i < len(ref_segments) and i < len(gen_segments):
                ref_seg = ref_segments[i]
                gen_seg = gen_segments[i]
                
                # MD difference
                md_diff = abs(ref_seg['startMd'] - gen_seg['startMd'])
                detail['md_diff'] = md_diff
                detail['md_match'] = md_diff <= self.tolerance
                if detail['md_match']:
                    md_matching_count += 1
                
                # Start shift difference
                start_shift_diff = abs(ref_seg['startShift'] - gen_seg['startShift'])
                detail['startShift_diff'] = start_shift_diff
                detail['startShift_match'] = start_shift_diff <= self.tolerance
                
                # End shift difference
                end_shift_diff = abs(ref_seg['endShift'] - gen_seg['endShift'])
                detail['endShift_diff'] = end_shift_diff
                detail['endShift_match'] = end_shift_diff <= self.tolerance
                
                # Overall match
                detail['fully_matching'] = (detail['md_match'] and 
                                           detail['startShift_match'] and 
                                           detail['endShift_match'])
                if detail['fully_matching']:
                    matching_count += 1
                
                if detail['startShift_match'] and detail['endShift_match']:
                    shift_matching_count += 1
                
                # Check if should be shown in detailed log
                if (md_diff > MISMATCH_TOLERANCE or 
                    start_shift_diff > MISMATCH_TOLERANCE or 
                    end_shift_diff > MISMATCH_TOLERANCE):
                    mismatched_segments.append(detail)
            else:
                # Missing segment - always show
                mismatched_segments.append(detail)
            
            segment_details.append(detail)
        
        # Build analysis results
        self.segments_analysis = {
            'reference_count': len(ref_segments),
            'generated_count': len(gen_segments),
            'count_difference': len(gen_segments) - len(ref_segments),
            'fully_matching_segments': matching_count,
            'md_matching_segments': md_matching_count,
            'shift_matching_segments': shift_matching_count,
            'segment_details': segment_details[:30],  # Store first 30 for analysis
            'mismatched_segments': mismatched_segments
        }
        
        # Log results
        self._log_segments_comparison()
        
        return self.segments_analysis
    
    def analyze_shifts_vs_segments(self, reference: Dict, generated: Dict) -> Dict:
        """Compare generated shifts with reference segments
        
        Args:
            reference: Reference data with segments
            generated: Generated data with shifts
            
        Returns:
            Shifts analysis results
        """
        logger.info("\n" + "="*80)
        logger.info("SHIFTS VS SEGMENTS ANALYSIS")
        logger.info("="*80)
        
        # Get generated shifts
        gen_shifts = generated.get('shifts', [])
        if not gen_shifts:
            logger.warning("No shifts found in generated data")
            return {}
        
        # Get reference segments
        ref_segments = reference.get('interpretation', {}).get('segments', [])
        if not ref_segments:
            logger.warning("No segments found in reference data")
            return {}
        
        logger.info(f"Found {len(gen_shifts)} shifts and {len(ref_segments)} reference segments")
        
        # Build segment ranges for interpolation
        segment_ranges = []
        for i, seg in enumerate(ref_segments):
            start_md = seg['startMd']
            # End MD is start of next segment or last shift MD
            end_md = ref_segments[i+1]['startMd'] if i+1 < len(ref_segments) else gen_shifts[-1]['md']
            
            segment_ranges.append({
                'start_md': start_md,
                'end_md': end_md,
                'start_shift': seg['startShift'],
                'end_shift': seg['endShift']
            })
        
        # Compare shifts at sample MD points
        comparison_results = []
        sample_count = min(50, len(gen_shifts))  # Analyze first 50 points
        
        for shift_point in gen_shifts[:sample_count]:
            md = shift_point['md']
            gen_shift = shift_point['shift']
            
            # Find corresponding segment and interpolate reference shift
            ref_shift = None
            for seg_range in segment_ranges:
                if seg_range['start_md'] <= md <= seg_range['end_md']:
                    # Linear interpolation within segment
                    if seg_range['end_md'] > seg_range['start_md']:
                        ratio = (md - seg_range['start_md']) / (seg_range['end_md'] - seg_range['start_md'])
                        ref_shift = seg_range['start_shift'] + ratio * (seg_range['end_shift'] - seg_range['start_shift'])
                    else:
                        ref_shift = seg_range['start_shift']
                    break
            
            if ref_shift is not None:
                diff = gen_shift - ref_shift
                comparison_results.append({
                    'md': md,
                    'gen_shift': gen_shift,
                    'ref_shift': ref_shift,
                    'difference': diff
                })
        
        if comparison_results:
            # Calculate statistics
            differences = np.array([r['difference'] for r in comparison_results])
            
            self.shifts_analysis = {
                'analyzed_points': len(comparison_results),
                'mean_difference': float(differences.mean()),
                'std_difference': float(differences.std()),
                'max_difference': float(differences.max()),
                'min_difference': float(differences.min()),
                'match_quality': 'good' if differences.std() < 0.1 else 'poor'
            }
            
            logger.info(f"Analyzed {len(comparison_results)} MD points:")
            logger.info(f"  Mean difference: {self.shifts_analysis['mean_difference']:.6f}")
            logger.info(f"  Std deviation: {self.shifts_analysis['std_difference']:.6f}")
            logger.info(f"  Max difference: {self.shifts_analysis['max_difference']:.6f}")
            logger.info(f"  Match quality: {self.shifts_analysis['match_quality']}")
        
        logger.info("="*80)
        
        return self.shifts_analysis
    
    def _log_segments_comparison(self):
        """Log detailed segments comparison"""
        if not self.segments_analysis:
            return
        
        logger.info("\n" + "="*80)
        logger.info("DETAILED SEGMENTS COMPARISON")
        logger.info("="*80)
        
        analysis = self.segments_analysis
        logger.info(f"Total segments: Ref={analysis['reference_count']}, Gen={analysis['generated_count']}")
        
        if analysis['reference_count'] > 0 and analysis['generated_count'] > 0:
            total = min(analysis['reference_count'], analysis['generated_count'])
            logger.info(f"Fully matching: {analysis['fully_matching_segments']}/{total}")
            logger.info(f"MD matching: {analysis['md_matching_segments']}/{total}")
            logger.info(f"Shifts matching: {analysis['shift_matching_segments']}/{total}")
        
        # Show mismatched segments
        mismatched = analysis.get('mismatched_segments', [])
        if mismatched:
            logger.info(f"\nMISMATCHED SEGMENTS (tolerance > 0.05):")
            logger.info("Idx |    Ref MD    Gen MD   Diff  |  Ref Start   Gen Start  Diff  |  "
                       "Ref End    Gen End   Diff  | Match")
            logger.info("-"*110)
            
            for d in mismatched[:10]:  # Show first 10
                i = d['index']
                if (isinstance(d.get('ref_startMd'), (int, float)) and 
                    isinstance(d.get('gen_startMd'), (int, float))):
                    logger.info(
                        f"{i:3d} | "
                        f"{d['ref_startMd']:8.1f} {d['gen_startMd']:8.1f} {d.get('md_diff', 0):6.2f} | "
                        f"{d['ref_startShift']:8.3f} {d['gen_startShift']:8.3f} {d.get('startShift_diff', 0):6.3f} | "
                        f"{d['ref_endShift']:8.3f} {d['gen_endShift']:8.3f} {d.get('endShift_diff', 0):6.3f} | "
                        f"{'✓' if d.get('fully_matching') else '✗'}"
                    )
                else:
                    logger.info(f"{i:3d} | One or both segments missing")
        else:
            logger.info(f"\n✅ ALL SEGMENTS MATCH within tolerance")
        
        logger.info("="*80)
    
    def _get_nested_value(self, data: Dict, path: List[str]) -> Optional[Any]:
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