import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class InterpretationBuilder:
    """Build interpretation segments from horizon shifts (TVT - TVD)"""
    
    # Parameters for segment detection
    STABLE_ANGLE_TOLERANCE = 0.02  # degrees - for stable segments
    FAULT_ANGLE_THRESHOLD = 50.0   # degrees - potential fault indicator
    MIN_SEGMENT_LENGTH = 3.0       # feet - minimum segment length
    
    def __init__(self):
        self.debug_mode = False  # Disable detailed logging
        
    def build_segments_from_shifts(
        self, 
        shifts_data: List[Dict],
        horizon_name: str = None
    ) -> List[Dict]:
        """
        Build interpretation segments from shifts array
        
        Args:
            shifts_data: Array of {'md': value, 'shift': value} from horizons
            horizon_name: Name of horizon for logging
            
        Returns:
            List of segments with startMd, startShift, endShift
        """
        if not shifts_data or len(shifts_data) < 2:
            logger.warning("Not enough shift data to build segments")
            return []
            
        logger.info(f"Building segments from {len(shifts_data)} shift points")
        
        # Convert to numpy arrays for easier processing
        md_array = np.array([p['md'] for p in shifts_data])
        shift_array = np.array([p['shift'] for p in shifts_data])
        
        # ИНВЕРТИРУЕМ ЗНАК для соответствия эталону!
        # В эталоне shifts положительные, а у нас отрицательные
        shift_array = -shift_array
        
        # Remove invalid points (e.g., MD = -99999)
        valid_mask = md_array > -99000
        md_array = md_array[valid_mask]
        shift_array = shift_array[valid_mask]
        
        # Calculate VS (vertical section) - cumulative horizontal distance
        vs_array = self._calculate_vs(shifts_data)
        vs_array = vs_array[valid_mask]
        
        logger.info(f"Valid points after filtering: {len(md_array)}")
        logger.info(f"MD range: {md_array[0]:.1f} to {md_array[-1]:.1f}")
        logger.info(f"Shift range: {shift_array.min():.3f} to {shift_array.max():.3f}")
        
        # Calculate angles between consecutive points
        angles = self._calculate_angles(vs_array, shift_array)
        
        # Detect segment boundaries
        boundaries = self._detect_boundaries(angles, vs_array, shift_array)
        
        # Build segments from boundaries
        segments = self._build_segments(boundaries, md_array, shift_array)
        
        logger.info(f"Built {len(segments)} segments")
        
        return segments
        
    def _calculate_vs(self, shifts_data: List[Dict]) -> np.ndarray:
        """Calculate vertical section (cumulative horizontal distance)"""
        # For now, use MD as proxy for VS (can be refined with actual trajectory)
        # In real implementation, VS should come from trajectory calculations
        vs = np.array([p['md'] for p in shifts_data])
        return vs
        
    def _calculate_angles(
        self, 
        vs_array: np.ndarray, 
        shift_array: np.ndarray
    ) -> List[Dict]:
        """Calculate angles between consecutive points"""
        angles = []
        
        for i in range(len(vs_array) - 1):
            delta_vs = vs_array[i+1] - vs_array[i]
            delta_shift = shift_array[i+1] - shift_array[i]
            
            # Handle vertical segments (delta_vs = 0)
            if abs(delta_vs) < 1e-6:
                if abs(delta_shift) > 1e-6:
                    # Vertical fault
                    angle = 90.0 if delta_shift > 0 else -90.0
                    is_fault = True
                else:
                    # No movement
                    angle = 0.0
                    is_fault = False
            else:
                # Calculate angle in degrees
                angle = np.degrees(np.arctan(delta_shift / delta_vs))
                is_fault = False
                
            angles.append({
                'index': i,
                'angle': angle,
                'delta_vs': delta_vs,
                'delta_shift': delta_shift,
                'is_vertical_fault': is_fault
            })
            
        return angles
        
    def _detect_boundaries(
        self, 
        angles: List[Dict],
        vs_array: np.ndarray,
        shift_array: np.ndarray
    ) -> List[int]:
        """Detect segment boundaries using the patterns"""
        boundaries = [0]  # First point is always a boundary
        
        for i in range(len(angles)):
            # Priority 1: Check for vertical fault (ΔVS = 0, ΔShift ≠ 0)
            if angles[i]['is_vertical_fault']:
                boundaries.append(i + 1)
                continue
                
            # Priority 2: Check for angular fault (angle > 50° with specific pattern)
            if abs(angles[i]['angle']) > self.FAULT_ANGLE_THRESHOLD:
                # Check neighbors
                left_angle = abs(angles[i-1]['angle']) if i > 0 else 0
                right_angle = abs(angles[i+1]['angle']) if i < len(angles)-1 else 0
                current_angle = abs(angles[i]['angle'])
                
                # Fault if both neighbors are smaller than current
                if left_angle < current_angle and right_angle < current_angle:
                    boundaries.append(i + 1)
                    continue
                    
            # Check for smooth transition patterns
            if self._check_smooth_transition(angles, i):
                boundaries.append(i + 1)
                
        # Add last point
        if len(vs_array) - 1 not in boundaries:
            boundaries.append(len(vs_array) - 1)
            
        # Remove duplicates and sort
        boundaries = sorted(list(set(boundaries)))
        
        logger.info(f"Detected {len(boundaries)} boundaries")
        
        return boundaries
        
    def _check_smooth_transition(self, angles: List[Dict], center_idx: int) -> bool:
        """Check for smooth transition pattern at given index"""
        
        # Need at least 2 points on each side
        if center_idx < 2 or center_idx >= len(angles) - 2:
            return False
            
        # Check for stable regions on both sides
        left_stable = self._is_stable_region(angles, center_idx - 2, center_idx)
        right_stable = self._is_stable_region(angles, center_idx + 1, center_idx + 3)
        
        if left_stable and right_stable:
            # Check if center region has changing angles
            center_angles = [angles[center_idx - 1]['angle'], 
                           angles[center_idx]['angle'],
                           angles[center_idx + 1]['angle']]
            
            # Check if angles are changing
            angle_change = max(center_angles) - min(center_angles)
            if angle_change > self.STABLE_ANGLE_TOLERANCE:
                return True
                
        return False
        
    def _is_stable_region(self, angles: List[Dict], start_idx: int, end_idx: int) -> bool:
        """Check if region has stable angles"""
        if start_idx < 0 or end_idx > len(angles):
            return False
            
        region_angles = [angles[i]['angle'] for i in range(start_idx, min(end_idx, len(angles)))]
        
        if len(region_angles) < 2:
            return False
            
        # Check if all angles are similar (within tolerance)
        angle_range = max(region_angles) - min(region_angles)
        return angle_range <= self.STABLE_ANGLE_TOLERANCE
        
    def _find_precise_intersection(
        self,
        md_array: np.ndarray,
        shift_array: np.ndarray,
        boundary_idx: int,
        window: int = 3
    ) -> Tuple[float, float]:
        """
        Find precise intersection point of two segments at boundary
        
        Args:
            boundary_idx: Index of boundary point
            window: Number of points to use for line fitting (default 3)
            
        Returns:
            (md, shift) of precise intersection point
        """
        # Get points for left segment (ignore middle point for better accuracy)
        left_start = max(0, boundary_idx - window)
        left_end = boundary_idx - 1
        
        if left_end > left_start:
            # Use first and last points, skipping middle for better accuracy
            left_md = [md_array[left_start], md_array[left_end]]
            left_shift = [shift_array[left_start], shift_array[left_end]]
            
            # Fit line for left segment
            left_slope = (left_shift[1] - left_shift[0]) / (left_md[1] - left_md[0]) if left_md[1] != left_md[0] else 0
            left_intercept = left_shift[0] - left_slope * left_md[0]
        else:
            # Not enough points, use boundary point
            return float(md_array[boundary_idx]), float(shift_array[boundary_idx])
        
        # Get points for right segment
        right_start = boundary_idx + 1
        right_end = min(len(md_array) - 1, boundary_idx + window)
        
        if right_end > right_start:
            # Use first and last points, skipping middle for better accuracy
            right_md = [md_array[right_start], md_array[right_end]]
            right_shift = [shift_array[right_start], shift_array[right_end]]
            
            # Fit line for right segment  
            right_slope = (right_shift[1] - right_shift[0]) / (right_md[1] - right_md[0]) if right_md[1] != right_md[0] else 0
            right_intercept = right_shift[0] - right_slope * right_md[0]
        else:
            # Not enough points, use boundary point
            return float(md_array[boundary_idx]), float(shift_array[boundary_idx])
        
        # Find intersection point
        if abs(left_slope - right_slope) > 1e-9:
            # Lines intersect
            intersection_md = (right_intercept - left_intercept) / (left_slope - right_slope)
            intersection_shift = left_slope * intersection_md + left_intercept
            
            # Ensure intersection is within reasonable bounds
            min_md = md_array[boundary_idx - 1]
            max_md = md_array[boundary_idx + 1]
            
            if min_md <= intersection_md <= max_md:
                return float(intersection_md), float(intersection_shift)
                
        # If lines are parallel or intersection is out of bounds, use midpoint
        return float(md_array[boundary_idx]), float(shift_array[boundary_idx])
        
    def _build_segments(
        self,
        boundaries: List[int],
        md_array: np.ndarray,
        shift_array: np.ndarray
    ) -> List[Dict]:
        """Build segments from detected boundaries with precise intersections"""
        segments = []
        
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            
            # Get precise intersection points for segment boundaries
            if i == 0:
                # First segment - use actual first point
                start_md = float(md_array[start_idx])
                start_shift = float(shift_array[start_idx])
            else:
                # Find precise intersection with previous segment
                start_md, start_shift = self._find_precise_intersection(
                    md_array, shift_array, start_idx
                )
            
            if i == len(boundaries) - 2:
                # Last segment - use actual last point
                end_md = float(md_array[end_idx])
                end_shift = float(shift_array[end_idx])
            else:
                # Find precise intersection with next segment
                end_md, end_shift = self._find_precise_intersection(
                    md_array, shift_array, end_idx
                )
            
            # Skip if segment is too short
            segment_length = end_md - start_md
            if segment_length < self.MIN_SEGMENT_LENGTH:
                continue
                
            segment = {
                'startMd': start_md,
                'startShift': start_shift,
                'endShift': end_shift
            }
            
            segments.append(segment)
            
        return segments


def build_interpretation_from_shifts(
    shifts_data: List[Dict],
    horizon_name: str = None
) -> Dict:
    """
    Main entry point to build interpretation from shifts
    
    Returns:
        Dictionary with interpretation data including segments
    """
    builder = InterpretationBuilder()
    segments = builder.build_segments_from_shifts(shifts_data, horizon_name)
    
    return {
        'segments': segments,
        'segments_count': len(segments),
        'algorithm': 'shifts_based',
        'horizon_name': horizon_name
    }