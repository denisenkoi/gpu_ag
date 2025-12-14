import math
import logging
import numpy as np
from typing import Dict, List, Optional, Any
from rogii_solo.calculations.trajectory import calculate_trajectory, interpolate_trajectory_point
from rogii_solo.calculations.enums import EMeasureUnits

from ..interpolators.trajectory_interpolator import TrajectoryInterpolator

logger = logging.getLogger(__name__)


class TrajectoryProcessor:
    """Processes raw trajectory data and performs interpolation with proper step size"""
    
    # Interpolation step in feet (will be converted to project units)
    INTERPOLATION_STEP_FEET = 1.0
    MIN_STEP_FEET = 0.1  # Minimum step for adaptive grid
    FEET_TO_METERS = 0.3048
    
    def __init__(self, config):
        """Initialize trajectory processor with configuration"""
        self.config = config
        
    def process_lateral_trajectory(
        self, 
        raw_survey_points: List[Dict], 
        well_metadata: Dict,
        project_measure_unit: str
    ) -> List[Dict]:
        """
        Process lateral raw survey points into interpolated trajectory with 1-foot step.
        
        Args:
            raw_survey_points: Raw PAPI survey points in format:
                [{'md': {'val': float}, 'incl': {'val': float}, 'azim': {'val': float}}]
            well_metadata: Well metadata from PAPI
            project_measure_unit: 'FOOT' or 'METER'
            
        Returns:
            Interpolated trajectory points in PAPI format:
                [{'measuredDepth': float, 'trueVerticalDepth': float, 
                  'inclinationRad': float, 'azimutRad': float, 
                  'northSouth': float, 'eastWest': float}]
        """
        logger.info(f"Processing lateral trajectory with {len(raw_survey_points)} survey points")
        
        # Determine interpolation step and units for Solo SDK
        if project_measure_unit == 'FOOT':
            interpolation_step = self.INTERPOLATION_STEP_FEET
            measure_units = EMeasureUnits.FOOT
        else:
            interpolation_step = self.INTERPOLATION_STEP_FEET * self.FEET_TO_METERS
            measure_units = EMeasureUnits.METER
            
        logger.info(f"Lateral interpolation step: {interpolation_step} {project_measure_unit} (fixed 1 ft)")
        
        # Convert PAPI survey points to Solo SDK format (degrees → radians)
        raw_trajectory = []
        for point in raw_survey_points:
            md = point['md']['val']
            incl = point['incl']['val']  # degrees from PAPI
            azim = point['azim']['val']  # degrees from PAPI
            
            raw_trajectory.append({
                'md': md,
                'incl': math.radians(incl),  # Solo SDK expects radians
                'azim': math.radians(azim)   # Solo SDK expects radians
            })
            
        # Prepare well metadata for Solo SDK
        well_params = {
            'convergence': well_metadata['convergence']['val'],
            'kb': well_metadata['kb']['val'],
            'xsrf': well_metadata['xsrf']['val'],
            'ysrf': well_metadata['ysrf']['val'],
            'tie_in_tvd': well_metadata['tie_in_tvd']['val'],
            'tie_in_ns': well_metadata['tie_in_ns']['val'],
            'tie_in_ew': well_metadata['tie_in_ew']['val'],
            'azimuth': math.radians(well_metadata['azimuth']['val']) if well_metadata['azimuth']['val'] is not None else 0.0
        }
        
        # Calculate trajectory using Solo SDK
        logger.info("Calling Solo SDK calculate_trajectory")
        calculated_trajectory = calculate_trajectory(
            raw_trajectory=raw_trajectory,
            well=well_params,
            measure_units=measure_units
        )
        
        if not calculated_trajectory:
            logger.error("Solo SDK returned empty trajectory")
            return []
        
        logger.info(f"Solo SDK calculated {len(calculated_trajectory)} trajectory points")
        
        # Interpolate with fixed step
        interpolated_trajectory = self._interpolate_trajectory_with_step(
            calculated_trajectory, well_params, interpolation_step, measure_units
        )
        
        logger.info(f"Final interpolated trajectory: {len(interpolated_trajectory)} points")
        return interpolated_trajectory

    def process_complete_well_trajectory_with_log(
        self,
        raw_survey_points: List[Dict],
        raw_log_points: List[Dict], 
        well_metadata: Dict,
        project_measure_unit: str
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Complete well processing: trajectory + log interpolation with adaptive step.
        
        COMPLETE LOGIC:
        1. Calculate optimal step based on log density (0.1 to 1.0 ft)
        2. Determine MD range from BOTH log and survey points (union of ranges)
        3. Handle boundary cases based on survey points count:
           - 0 points: vertical trajectory for entire range
           - 1 point: vertical to point, then ray from point
           - 2+ points: vertical to first, Solo SDK between points, ray from last
        4. Generate uniform MD grid with adaptive step size
        5. Interpolate log data to trajectory MD grid
        6. Return both trajectory and log with TVD
        
        Args:
            raw_survey_points: Raw PAPI survey points in format:
                [{'md': {'val': float}, 'incl': {'val': float}, 'azim': {'val': float}}]
            raw_log_points: Raw PAPI log points in format:
                [{'md': {'val': float}, 'data': {'val': float}}]
            well_metadata: Well metadata from PAPI
            project_measure_unit: 'FOOT' or 'METER'
            
        Returns:
            Tuple of:
            - trajectory_points: Interpolated trajectory points in PAPI format
            - log_with_tvd: Log points with TVD data
        """
        logger.info(f"Processing complete well: {len(raw_survey_points)} survey points, {len(raw_log_points)} log points")
        
        # Calculate optimal interpolation step based on log density
        interpolation_step = self._calculate_optimal_step(raw_log_points, project_measure_unit)
        
        # Determine units for Solo SDK
        if project_measure_unit == 'FOOT':
            measure_units = EMeasureUnits.FOOT
        else:
            measure_units = EMeasureUnits.METER
            
        logger.info(f"Adaptive interpolation step: {interpolation_step:.3f} {project_measure_unit}")
        
        # Filter and sort survey points by MD
        valid_survey_points = []
        for point in raw_survey_points:
            md = point['md']['val']
            incl = point['incl']['val']
            azim = point['azim']['val']
            
            # Assert on invalid data instead of filtering
            assert md is not None, f"Survey point MD is None: {point}"
            assert incl is not None, f"Survey point incl is None: {point}"
            assert azim is not None, f"Survey point azim is None: {point}"
            
            valid_survey_points.append({
                'md': md,
                'incl': incl,  # degrees from PAPI
                'azim': azim   # degrees from PAPI
            })
        
        # Sort by MD
        valid_survey_points.sort(key=lambda p: p['md'])
        
        # Filter and analyze log points to determine MD range
        valid_log_points = []
        for point in raw_log_points:
            # Extract values with flexible format handling
            md = self._extract_value_safe(point.get('md'))
            data = self._extract_value_safe(point.get('data'))
            
            # Only include points with valid data and MD
            if md is not None and data is not None:
                valid_log_points.append({
                    'md': md,
                    'data': data
                })
        
        # Sort log points by MD
        valid_log_points.sort(key=lambda p: p['md'])
        
        # Determine combined MD range (union of log and survey ranges)
        start_md = None
        end_md = None
        
        if valid_log_points:
            log_start = valid_log_points[0]['md']
            log_end = valid_log_points[-1]['md']
            start_md = log_start
            end_md = log_end
            logger.info(f"Log MD range: {log_start:.1f} → {log_end:.1f}")
        
        if valid_survey_points:
            survey_start = valid_survey_points[0]['md']
            survey_end = valid_survey_points[-1]['md']
            
            if start_md is None:
                start_md = survey_start
                end_md = survey_end
            else:
                start_md = min(start_md, survey_start)
                end_md = max(end_md, survey_end)
            
            logger.info(f"Survey points MD range: {survey_start:.1f} → {survey_end:.1f}")
        
        # Handle case where both log and survey points are empty
        if start_md is None or end_md is None:
            logger.warning("No valid log or survey data found")
            return [], []
        
        logger.info(f"Combined MD range: {start_md:.1f} → {end_md:.1f}")
        
        # Generate uniform MD grid with adaptive step
        md_grid = self._generate_md_grid(start_md, end_md, interpolation_step)
        logger.info(f"Generated MD grid: {len(md_grid)} points from {md_grid[0]:.1f} to {md_grid[-1]:.1f}, step: {interpolation_step:.3f}")
        
        # Strategy selection based on survey points count
        if len(valid_survey_points) == 0:
            logger.info("Strategy: No survey points - vertical trajectory")
            trajectory_points = self._create_vertical_trajectory(md_grid, well_metadata)
            
        elif len(valid_survey_points) == 1:
            logger.info("Strategy: One survey point - vertical + ray")
            trajectory_points = self._create_single_point_trajectory(
                md_grid, valid_survey_points[0], well_metadata
            )
            
        else:
            logger.info(f"Strategy: {len(valid_survey_points)} survey points - vertical + SDK + ray")
            trajectory_points = self._create_multi_point_trajectory(
                md_grid, valid_survey_points, well_metadata, measure_units
            )
        
        # Process log data with trajectory
        if valid_log_points and trajectory_points:
            log_with_tvd = self._combine_log_with_trajectory(valid_log_points, trajectory_points)
        else:
            log_with_tvd = []
            
        logger.info(f"Complete well processing finished: {len(trajectory_points)} trajectory points, {len(log_with_tvd)} log points")
        
        return trajectory_points, log_with_tvd
    
    def _calculate_optimal_step(self, raw_log_points: List[Dict], project_measure_unit: str) -> float:
        """
        Calculate optimal interpolation step based on log density.
        
        Finds minimum non-zero MD difference in log to preserve detail.
        Limits result to [0.1, 1.0] ft range (or equivalent in meters).
        
        Args:
            raw_log_points: Raw log points from PAPI
            project_measure_unit: 'FOOT' or 'METER'
            
        Returns:
            Optimal step size in project units
        """
        # Define limits in feet
        MIN_STEP_FEET = 0.1
        MAX_STEP_FEET = 1.0
        ZERO_THRESHOLD_FEET = 0.01  # Consider anything less than this as zero
        
        # Convert limits to project units
        if project_measure_unit == 'METER':
            min_step = MIN_STEP_FEET * self.FEET_TO_METERS
            max_step = MAX_STEP_FEET * self.FEET_TO_METERS
            zero_threshold = ZERO_THRESHOLD_FEET * self.FEET_TO_METERS
        else:
            min_step = MIN_STEP_FEET
            max_step = MAX_STEP_FEET
            zero_threshold = ZERO_THRESHOLD_FEET
        
        # Calculate MD differences between consecutive points
        md_diffs = []
        for i in range(1, min(1000, len(raw_log_points))):  # Analyze first 1000 points for speed
            curr_md = self._extract_value_safe(raw_log_points[i].get('md'))
            prev_md = self._extract_value_safe(raw_log_points[i-1].get('md'))
            
            if curr_md is not None and prev_md is not None:
                diff = curr_md - prev_md
                # Ignore pseudo-zeros (less than threshold)
                if diff > zero_threshold:
                    md_diffs.append(diff)
        
        if not md_diffs:
            logger.warning(f"No valid MD differences found, using default step {max_step:.3f} {project_measure_unit}")
            return max_step
        
        # Find minimum non-zero step (maximum detail preservation)
        found_min_step = min(md_diffs)
        
        # Constrain to allowed range [min_step, max_step]
        optimal_step = max(min_step, min(max_step, found_min_step))
        
        # Log statistics for debugging
        logger.info(f"Log step analysis ({project_measure_unit}): "
                   f"analyzed {len(md_diffs)} valid diffs, "
                   f"found_min={found_min_step:.4f}, "
                   f"found_max={max(md_diffs):.4f}, "
                   f"using={optimal_step:.3f} "
                   f"(limits: [{min_step:.3f}, {max_step:.3f}])")
        
        # Log if we hit the limits
        if optimal_step == min_step and found_min_step < min_step:
            logger.info(f"Step limited by minimum: {found_min_step:.4f} → {min_step:.3f}")
        elif optimal_step == max_step and found_min_step > max_step:
            logger.info(f"Step limited by maximum: {found_min_step:.4f} → {max_step:.3f}")
        
        return optimal_step
    
    def _generate_md_grid(self, start_md: float, end_md: float, step: float) -> List[float]:
        """Generate uniform MD grid with specified step"""
        md_grid = []
        current_md = start_md
        
        while current_md <= end_md:
            md_grid.append(current_md)
            current_md += step
            
        # Ensure end_md is included
        if md_grid and md_grid[-1] < end_md:
            md_grid.append(end_md)
            
        return md_grid
    
    def _create_vertical_trajectory(self, md_grid: List[float], well_metadata: Dict) -> List[Dict]:
        """Create vertical trajectory for entire MD range"""
        logger.info("Creating vertical trajectory")
        
        # Extract well surface coordinates and tie-in
        xsrf = well_metadata['xsrf']['val']
        ysrf = well_metadata['ysrf']['val']
        kb = well_metadata['kb']['val']
        tie_in_tvd = well_metadata['tie_in_tvd']['val']
        tie_in_ns = well_metadata['tie_in_ns']['val']
        tie_in_ew = well_metadata['tie_in_ew']['val']
        
        trajectory_points = []
        
        for md in md_grid:
            # Vertical trajectory: TVD = MD
            tvd = md + tie_in_tvd
            
            # Vertical trajectory: no horizontal displacement
            ns = tie_in_ns
            ew = tie_in_ew
            
            point = {
                'measuredDepth': md,
                'trueVerticalDepth': tvd,
                'inclinationRad': 0.0,  # Vertical
                'azimutRad': 0.0,       # Arbitrary for vertical
                'northSouth': ns,
                'eastWest': ew
            }
            trajectory_points.append(point)
            
        logger.info(f"Created vertical trajectory: {len(trajectory_points)} points")
        return trajectory_points
    
    def _create_single_point_trajectory(
        self, 
        md_grid: List[float], 
        survey_point: Dict, 
        well_metadata: Dict
    ) -> List[Dict]:
        """Create trajectory with one survey point: vertical + ray"""
        logger.info(f"Creating single point trajectory with survey point at MD={survey_point['md']}")
        
        # Extract well parameters
        xsrf = well_metadata['xsrf']['val']
        ysrf = well_metadata['ysrf']['val']
        kb = well_metadata['kb']['val']
        tie_in_tvd = well_metadata['tie_in_tvd']['val']
        tie_in_ns = well_metadata['tie_in_ns']['val']
        tie_in_ew = well_metadata['tie_in_ew']['val']
        
        survey_md = survey_point['md']
        survey_incl_rad = math.radians(survey_point['incl'])
        survey_azim_rad = math.radians(survey_point['azim'])
        
        # Calculate survey point coordinates
        survey_tvd = survey_md * math.cos(survey_incl_rad) + tie_in_tvd
        survey_ns = survey_md * math.sin(survey_incl_rad) * math.cos(survey_azim_rad) + tie_in_ns
        survey_ew = survey_md * math.sin(survey_incl_rad) * math.sin(survey_azim_rad) + tie_in_ew
        
        trajectory_points = []
        
        for md in md_grid:
            if md <= survey_md:
                # Vertical section: from start to survey point
                tvd = md + tie_in_tvd
                ns = tie_in_ns
                ew = tie_in_ew
                incl_rad = 0.0
                azim_rad = 0.0
                
            else:
                # Ray section: from survey point downward
                delta_md = md - survey_md
                
                # Continue with survey point angles
                delta_tvd = delta_md * math.cos(survey_incl_rad)
                delta_ns = delta_md * math.sin(survey_incl_rad) * math.cos(survey_azim_rad)
                delta_ew = delta_md * math.sin(survey_incl_rad) * math.sin(survey_azim_rad)
                
                tvd = survey_tvd + delta_tvd
                ns = survey_ns + delta_ns
                ew = survey_ew + delta_ew
                incl_rad = survey_incl_rad
                azim_rad = survey_azim_rad
            
            point = {
                'measuredDepth': md,
                'trueVerticalDepth': tvd,
                'inclinationRad': incl_rad,
                'azimutRad': azim_rad,
                'northSouth': ns,
                'eastWest': ew
            }
            trajectory_points.append(point)
            
        logger.info(f"Created single point trajectory: {len(trajectory_points)} points")
        return trajectory_points
    
    def _create_multi_point_trajectory(
        self, 
        md_grid: List[float], 
        valid_survey_points: List[Dict], 
        well_metadata: Dict,
        measure_units: EMeasureUnits
    ) -> List[Dict]:
        """Create trajectory with multiple survey points: vertical + Solo SDK + ray"""
        logger.info(f"Creating multi-point trajectory with {len(valid_survey_points)} survey points")
        
        first_survey_md = valid_survey_points[0]['md']
        last_survey_md = valid_survey_points[-1]['md']
        
        # Extract well parameters
        tie_in_tvd = well_metadata['tie_in_tvd']['val']
        tie_in_ns = well_metadata['tie_in_ns']['val']
        tie_in_ew = well_metadata['tie_in_ew']['val']
        
        # Prepare well metadata for Solo SDK
        well_params = {
            'convergence': well_metadata['convergence']['val'],
            'kb': well_metadata['kb']['val'],
            'xsrf': well_metadata['xsrf']['val'],
            'ysrf': well_metadata['ysrf']['val'],
            'tie_in_tvd': tie_in_tvd,
            'tie_in_ns': tie_in_ns,
            'tie_in_ew': tie_in_ew,
            'azimuth': math.radians(well_metadata['azimuth']['val']) if well_metadata['azimuth']['val'] is not None else 0.0
        }
        
        # Convert survey points to Solo SDK format (radians)
        solo_survey_points = []
        for point in valid_survey_points:
            solo_survey_points.append({
                'md': point['md'],
                'incl': math.radians(point['incl']),  # Convert to radians
                'azim': math.radians(point['azim'])   # Convert to radians
            })
        
        # Calculate trajectory using Solo SDK for survey points range
        logger.info(f"Calling Solo SDK for range: {first_survey_md:.1f} → {last_survey_md:.1f}")
        calculated_trajectory = calculate_trajectory(
            raw_trajectory=solo_survey_points,
            well=well_params,
            measure_units=measure_units
        )
        
        assert calculated_trajectory, "Solo SDK returned empty trajectory"
        logger.info(f"Solo SDK calculated {len(calculated_trajectory)} points")
        
        # Get last survey point angles for ray extension
        last_survey = valid_survey_points[-1]
        last_incl_rad = math.radians(last_survey['incl'])
        last_azim_rad = math.radians(last_survey['azim'])
        
        # Find coordinates at last survey point from Solo SDK calculation
        last_calculated_point = None
        for calc_point in calculated_trajectory:
            if abs(calc_point['md'] - last_survey_md) < 0.01:  # Small tolerance
                last_calculated_point = calc_point
                break
        
        assert last_calculated_point, f"Could not find calculated point for MD={last_survey_md}"
        
        trajectory_points = []
        
        for md in md_grid:
            if md < first_survey_md:
                # Vertical section: before first survey point
                tvd = md + tie_in_tvd
                ns = tie_in_ns
                ew = tie_in_ew
                incl_rad = 0.0
                azim_rad = 0.0
                
            elif md <= last_survey_md:
                # Solo SDK section: between survey points
                # Interpolate from calculated trajectory
                interpolated = self._interpolate_from_solo_trajectory(md, calculated_trajectory)
                tvd = interpolated['tvd']
                ns = interpolated['ns']
                ew = interpolated['ew']
                incl_rad = interpolated['incl']
                azim_rad = interpolated['azim']
                
            else:
                # Ray section: beyond last survey point
                delta_md = md - last_survey_md
                
                delta_tvd = delta_md * math.cos(last_incl_rad)
                delta_ns = delta_md * math.sin(last_incl_rad) * math.cos(last_azim_rad)
                delta_ew = delta_md * math.sin(last_incl_rad) * math.sin(last_azim_rad)
                
                tvd = last_calculated_point['tvd'] + delta_tvd
                ns = last_calculated_point['ns'] + delta_ns
                ew = last_calculated_point['ew'] + delta_ew
                incl_rad = last_incl_rad
                azim_rad = last_azim_rad
            
            point = {
                'measuredDepth': md,
                'trueVerticalDepth': tvd,
                'inclinationRad': incl_rad,
                'azimutRad': azim_rad,
                'northSouth': ns,
                'eastWest': ew
            }
            trajectory_points.append(point)
            
        logger.info(f"Created multi-point trajectory: {len(trajectory_points)} points")
        return trajectory_points
    
    def _interpolate_from_solo_trajectory(self, target_md: float, calculated_trajectory: List[Dict]) -> Dict:
        """Interpolate point from Solo SDK calculated trajectory"""
        
        # Find surrounding points
        for i in range(len(calculated_trajectory) - 1):
            left_point = calculated_trajectory[i]
            right_point = calculated_trajectory[i + 1]
            
            if left_point['md'] <= target_md <= right_point['md']:
                if left_point['md'] == right_point['md']:
                    # Same MD - use left point
                    return {
                        'tvd': left_point['tvd'],
                        'ns': left_point['ns'],
                        'ew': left_point['ew'],
                        'incl': left_point['incl'],
                        'azim': left_point['azim']
                    }
                else:
                    # Linear interpolation
                    ratio = (target_md - left_point['md']) / (right_point['md'] - left_point['md'])
                    
                    return {
                        'tvd': left_point['tvd'] + ratio * (right_point['tvd'] - left_point['tvd']),
                        'ns': left_point['ns'] + ratio * (right_point['ns'] - left_point['ns']),
                        'ew': left_point['ew'] + ratio * (right_point['ew'] - left_point['ew']),
                        'incl': left_point['incl'] + ratio * (right_point['incl'] - left_point['incl']),
                        'azim': left_point['azim'] + ratio * (right_point['azim'] - left_point['azim'])
                    }
        
        # Check boundary points
        if target_md == calculated_trajectory[0]['md']:
            point = calculated_trajectory[0]
            return {
                'tvd': point['tvd'],
                'ns': point['ns'],
                'ew': point['ew'],
                'incl': point['incl'],
                'azim': point['azim']
            }
            
        if target_md == calculated_trajectory[-1]['md']:
            point = calculated_trajectory[-1]
            return {
                'tvd': point['tvd'],
                'ns': point['ns'],
                'ew': point['ew'],
                'incl': point['incl'],
                'azim': point['azim']
            }
        
        assert False, f"MD={target_md} not found in Solo SDK trajectory range"
    
    def _combine_log_with_trajectory(
        self, 
        valid_log_points: List[Dict], 
        trajectory_points: List[Dict]
    ) -> List[Dict]:
        """
        Combine log data with trajectory by interpolating to trajectory MD grid.
        
        Args:
            valid_log_points: Already filtered and sorted log points
            trajectory_points: Already processed trajectory points
            
        Returns:
            Log points with trajectory data:
                [{'measuredDepth': float, 'trueVerticalDepth': float, 'data': float}]
        """
        logger.info(f"Combining {len(valid_log_points)} log points with {len(trajectory_points)} trajectory points")
        
        if not valid_log_points or not trajectory_points:
            logger.warning("No valid log or trajectory points for combination")
            return []
            
        # Extract trajectory MD grid
        trajectory_md_grid = [p['measuredDepth'] for p in trajectory_points]
        trajectory_tvd_grid = [p['trueVerticalDepth'] for p in trajectory_points]
        
        logger.info(f"Log MD range: {valid_log_points[0]['md']:.1f}→{valid_log_points[-1]['md']:.1f}")
        logger.info(f"Trajectory MD range: {trajectory_md_grid[0]:.1f}→{trajectory_md_grid[-1]:.1f}")
        
        # Interpolate log data to trajectory MD grid
        interpolated_log_values = self._interpolate_log_to_md_grid(
            valid_log_points, trajectory_md_grid
        )
        
        logger.info(f"Interpolated values count: {len(interpolated_log_values)}")
        logger.debug(f"First 5 interpolated values: {interpolated_log_values[:5]}")
        
        # Combine with trajectory data
        log_with_tvd = []
        for i, (md, tvd, log_val) in enumerate(zip(trajectory_md_grid, trajectory_tvd_grid, interpolated_log_values)):
            if log_val is not None:
                log_with_tvd.append({
                    'measuredDepth': md,
                    'trueVerticalDepth': tvd,
                    'data': log_val
                })
        
        logger.info(f"Combined log with trajectory: {len(log_with_tvd)} valid points")
        return log_with_tvd

    def create_welllog_structure_with_tvd_sorted(
        self,
        raw_log_points: List[Dict],
        trajectory_points: List[Dict]
    ) -> Dict:
        """
        Create complete wellLog structure with both points and tvdSortedPoints.
        
        Returns:
            {
                'points': [...],  # MD-sorted log points with TVD
                'tvdSortedPoints': [...]  # TVD-sorted, monotonic filtered log points
            }
        """
        logger.info("Creating wellLog structure with TVD sorted points")
        
        # Get regular log points with TVD
        if raw_log_points and trajectory_points:
            # Filter valid log points
            valid_log_points = []
            for point in raw_log_points:
                md = self._extract_value_safe(point.get('md'))
                data = self._extract_value_safe(point.get('data'))
                if md is not None and data is not None:
                    valid_log_points.append({'md': md, 'data': data})
            
            log_with_tvd = self._combine_log_with_trajectory(valid_log_points, trajectory_points)
        else:
            log_with_tvd = []
        
        logger.info(f"Points before filtering: {len(log_with_tvd)}")
        # Create TVD sorted points with monotonic filtering
        tvd_sorted_points = self.filter_monotonic_tvd_points(log_with_tvd.copy())
        logger.info(f"Points after filtering: {len(tvd_sorted_points)}")
        
        return {
            'points': log_with_tvd,
            'tvdSortedPoints': tvd_sorted_points
        }
    
    def filter_monotonic_tvd_points(self, tvd_points: List[Dict]) -> List[Dict]:
        """Filter points to keep only those where TVD monotonically increases."""
        if not tvd_points:
            return []
        
        # Фильтруем, оставляя только точки с возрастающим TVD
        # ВАЖНО: НЕ сортируем! Идем по исходному MD порядку
        monotonic = []
        max_tvd = float('-inf')
        
        for point in tvd_points:  # По исходному MD порядку
            current_tvd = point['trueVerticalDepth']
            if current_tvd > max_tvd:  # Только если TVD больше максимального
                monotonic.append(point)
                max_tvd = current_tvd  # Обновляем максимум
            # Иначе пропускаем точку (TVD уменьшился или остался тот же)
        
        return monotonic

    def process_typewell_trajectory(
        self, 
        raw_trajectory_points: List[Dict], 
        typewell_metadata: Dict,
        raw_log_points: List[Dict],
        project_measure_unit: str
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Process typewell trajectory and log using the same logic as lateral wells.
        
        Args:
            raw_trajectory_points: Raw typewell trajectory points
            typewell_metadata: Typewell metadata
            raw_log_points: Raw typewell log points
            project_measure_unit: 'FOOT' or 'METER'
            
        Returns:
            Tuple of (trajectory_points, log_with_tvd)
        """
        logger.info(f"Processing typewell trajectory with {len(raw_trajectory_points)} trajectory points and {len(raw_log_points)} log points")
        
        # Use the same complete processing logic as lateral wells
        trajectory_points, log_with_tvd = self.process_complete_well_trajectory_with_log(
            raw_survey_points=raw_trajectory_points,
            raw_log_points=raw_log_points,
            well_metadata=typewell_metadata,
            project_measure_unit=project_measure_unit
        )
        
        logger.info(f"Processed typewell: {len(trajectory_points)} trajectory points, {len(log_with_tvd)} log points")
        return trajectory_points, log_with_tvd
        
    def process_log_with_trajectory(
        self, 
        raw_log_points: List[Dict], 
        processed_trajectory: List[Dict]
    ) -> List[Dict]:
        """
        Process log data by interpolating to trajectory MD grid and matching with TVD.
        
        CORRECT SEQUENCE:
        1. Interpolate log to uniform MD grid (same as trajectory)
        2. Match log[i] with trajectory[i] by index
        3. No repeated calculations!
        
        Args:
            raw_log_points: Raw log points from PAPI
            processed_trajectory: Already interpolated trajectory points
            
        Returns:
            Log points with trajectory data:
                [{'measuredDepth': float, 'trueVerticalDepth': float, 'data': float}]
        """
        logger.info(f"Processing log with {len(raw_log_points)} raw points and {len(processed_trajectory)} trajectory points")
        
        if not processed_trajectory:
            logger.warning("No trajectory points provided for log processing")
            return []
        
        # Step 1: Filter and sort log points by MD
        valid_log_points = []
        for point in raw_log_points:
            # Extract values with flexible format handling
            data_val = self._extract_value_safe(point.get('data'))
            md_val = self._extract_value_safe(point.get('md'))
            
            # Skip null values - no None filtering for debugging
            if data_val is not None and md_val is not None:
                valid_log_points.append({
                    'md': md_val,
                    'data': data_val
                })
        
        valid_log_points.sort(key=lambda x: x['md'])
        
        if not valid_log_points:
            logger.warning("No valid log points found after filtering")
            return []
            
        logger.info(f"Valid log points: {len(valid_log_points)}")
        logger.debug(f"MD range: {valid_log_points[0]['md']} - {valid_log_points[-1]['md']}")
        
        # Step 2: Extract trajectory MD values (already sorted)
        trajectory_md_values = [p['measuredDepth'] for p in processed_trajectory]
        logger.info(f"Trajectory MD range: {trajectory_md_values[0]:.1f}→{trajectory_md_values[-1]:.1f}")
        
        # Step 3: Interpolate log to trajectory MD grid using correct sequence
        interpolated_log_values = self._interpolate_log_to_md_grid(
            valid_log_points, trajectory_md_values
        )
        
        # Step 3: Combine with trajectory TVD by index
        log_with_tvd = []
        combined_count = 0
        for i, (md, log_value) in enumerate(zip(trajectory_md_values, interpolated_log_values)):
            if log_value is not None:  # Only include points with log data
                tvd = processed_trajectory[i]['trueVerticalDepth']
                log_with_tvd.append({
                    'measuredDepth': md,
                    'trueVerticalDepth': tvd,
                    'data': log_value
                })
                combined_count += 1
                
        logger.info(f"Combined {combined_count} log points with trajectory TVD")
        logger.debug(f"Final log_with_tvd points: {len(log_with_tvd)}")
        return log_with_tvd
        
    def _interpolate_log_to_md_grid(
        self, 
        valid_log_points: List[Dict], 
        target_md_grid: List[float]
    ) -> List[Optional[float]]:
        """
        Interpolate log values to target MD grid with optional extrapolation.
        
        Args:
            valid_log_points: Sorted log points with 'md' and 'data' fields
            target_md_grid: Target MD values for interpolation
            
        Returns:
            List of interpolated values (None for points outside log range when extrapolation disabled)
        """
        if not valid_log_points:
            return [None] * len(target_md_grid)
        
        log_md = [p['md'] for p in valid_log_points]
        log_data = [p['data'] for p in valid_log_points]
        
        min_log_md = log_md[0]
        max_log_md = log_md[-1]
        
        interpolated_values = []
        
        for target_md in target_md_grid:
            if target_md < min_log_md:
                # Before log range
                interpolated_values.append(None)  # No extrapolation
            elif target_md > max_log_md:
                # After log range
                interpolated_values.append(None)  # No extrapolation
                    
            else:
                # Within log range - interpolate
                interp_value = self._linear_interpolate(target_md, log_md, log_data)
                interpolated_values.append(interp_value)
        
        return interpolated_values
    
    def _linear_interpolate(self, target_x: float, x_values: List[float], y_values: List[float]) -> float:
        """Perform linear interpolation for a single point"""
        
        # Find surrounding points
        for i in range(len(x_values) - 1):
            if x_values[i] <= target_x <= x_values[i + 1]:
                if x_values[i] == x_values[i + 1]:
                    return y_values[i]
                else:
                    ratio = (target_x - x_values[i]) / (x_values[i + 1] - x_values[i])
                    return y_values[i] + ratio * (y_values[i + 1] - y_values[i])
        
        # Check exact boundary matches
        if target_x == x_values[0]:
            return y_values[0]
        if target_x == x_values[-1]:
            return y_values[-1]
        
        assert False, f"Target x={target_x} not found in interpolation range [{x_values[0]}, {x_values[-1]}]"

    def _interpolate_trajectory_with_step(
        self, 
        calculated_trajectory: List[Dict], 
        well_params: Dict,
        step: float,
        measure_units: EMeasureUnits
    ) -> List[Dict]:
        """Interpolate trajectory with fixed MD step using Solo SDK"""
        
        if not calculated_trajectory:
            logger.warning("Empty calculated trajectory - cannot interpolate")
            return []
            
        # Get MD range
        first_md = calculated_trajectory[0]['md']
        last_md = calculated_trajectory[-1]['md']
        
        logger.info(f"Interpolating MD range: {first_md:.1f} → {last_md:.1f}, step: {step}")
        
        # Convert to convenient format for interpolation
        trajectory_points = []
        for point in calculated_trajectory:
            trajectory_points.append({
                'md': point['md'],
                'incl': point['incl'],
                'azim': point['azim'],
                'tvd': point['tvd'],
                'ns': point['ns'],
                'ew': point['ew'],
                'x': point['x'],
                'y': point['y'],
                'vs': point['vs'],
                'dls': point['dls'],
                'dog_leg': point['dog_leg']
            })
            
        # Generate interpolated points
        interpolated_points = []
        current_md = first_md
        
        while current_md <= last_md:
            # Find bracketing points
            left_point = None
            right_point = None
            
            for point in trajectory_points:
                if point['md'] <= current_md:
                    left_point = point
                if point['md'] >= current_md and right_point is None:
                    right_point = point
                    break
                    
            # Interpolate or use exact point
            if left_point is None:
                left_point = trajectory_points[0]
            if right_point is None:
                right_point = trajectory_points[-1]
                
            if abs(left_point['md'] - current_md) < 0.001:
                # Use exact point
                interpolated_points.append({
                    'measuredDepth': current_md,
                    'trueVerticalDepth': left_point['tvd'],
                    'inclinationRad': left_point['incl'],  # Already in radians from Solo SDK
                    'azimutRad': left_point['azim'],       # Already in radians from Solo SDK
                    'northSouth': left_point['ns'],
                    'eastWest': left_point['ew']
                })
            elif abs(right_point['md'] - current_md) < 0.001:
                # Use exact point
                interpolated_points.append({
                    'measuredDepth': current_md,
                    'trueVerticalDepth': right_point['tvd'],
                    'inclinationRad': right_point['incl'],  # Already in radians from Solo SDK
                    'azimutRad': right_point['azim'],       # Already in radians from Solo SDK
                    'northSouth': right_point['ns'],
                    'eastWest': right_point['ew']
                })
            else:
                # Use Solo SDK interpolation for best accuracy
                interpolated_point = interpolate_trajectory_point(
                    left_point=left_point,
                    right_point=right_point,
                    md=current_md,
                    well=well_params,
                    measure_units=measure_units
                )
                
                interpolated_points.append({
                    'measuredDepth': current_md,
                    'trueVerticalDepth': interpolated_point['tvd'],
                    'inclinationRad': interpolated_point['incl'],  # Already in radians from Solo SDK
                    'azimutRad': interpolated_point['azim'],       # Already in radians from Solo SDK
                    'northSouth': interpolated_point['ns'],
                    'eastWest': interpolated_point['ew']
                })
                
            current_md += step
            
        logger.info(f"Interpolated trajectory: {len(interpolated_points)} points")
        return interpolated_points
    
    def _find_tvd_for_md(self, md: float, calculated_trajectory: List[Dict]) -> float:
        """Find TVD for given MD using interpolation in Solo SDK calculated trajectory"""
        
        # Find bracketing points
        for i in range(len(calculated_trajectory) - 1):
            left_point = calculated_trajectory[i]
            right_point = calculated_trajectory[i + 1]
            
            if left_point['md'] <= md <= right_point['md']:
                # Found interval for interpolation
                if left_point['md'] == right_point['md']:
                    # Same MD points - use left point TVD
                    return left_point['tvd']
                else:
                    # Linear interpolation between calculated TVD points
                    ratio = (md - left_point['md']) / (right_point['md'] - left_point['md'])
                    tvd = left_point['tvd'] + ratio * (right_point['tvd'] - left_point['tvd'])
                    return tvd
        
        # If we reach here, MD should be exactly at boundary points
        # Check first and last points
        if md == calculated_trajectory[0]['md']:
            return calculated_trajectory[0]['tvd']
        if md == calculated_trajectory[-1]['md']:
            return calculated_trajectory[-1]['tvd']
        
        # Should never reach here if trajectory points cover the MD
        assert False, f"MD={md} not found in Solo SDK calculated trajectory range"
        
    def _extract_value_safe(self, field_data: Any) -> Any:
        """
        Safely extract value from PAPI field data handling different formats.
        
        Handles:
        - Direct values: 123.45 -> 123.45
        - Wrapped values: {'val': 123.45} -> 123.45
        - Undefined values: {'undefined': True} -> None
        - Missing/None values -> None
        
        Args:
            field_data: Field data in various formats
            
        Returns:
            Extracted value or None if not available
        """
        if field_data is None:
            return None
            
        # If direct value (number, string, etc.)
        if isinstance(field_data, (int, float, str)):
            return field_data
            
        # If dict format
        if isinstance(field_data, dict):
            # Check for undefined flag
            if field_data.get('undefined', False):
                return None
            # Extract 'val' field if present
            if 'val' in field_data:
                return field_data['val']
            # If dict without 'val', return None (unknown format)
            return None
            
        # Unknown format
        return None