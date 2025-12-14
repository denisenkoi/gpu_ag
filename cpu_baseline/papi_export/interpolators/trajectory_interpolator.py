import math
import logging
from typing import List, Dict
from rogii_solo.calculations.trajectory import calculate_trajectory
from rogii_solo.calculations.enums import EMeasureUnits

logger = logging.getLogger(__name__)


class TrajectoryInterpolator:
    """Universal TVD interpolator for laterals and typewells using Solo SDK with proper geological logic"""
    
    @staticmethod
    def interpolate_tvd(
        md: float, 
        trajectory_points: List[Dict],
        well_metadata: Dict,
        measure_units: EMeasureUnits = EMeasureUnits.METER
    ) -> float:
        """
        Universal TVD interpolation with geological logic for different trajectory cases.
        
        Geological logic:
        - 0 points: Vertical well (TVD = MD)
        - 1 point: Vertical before point, extrapolate angle after point
        - 2+ points: Vertical before first, Solo SDK between points, extrapolate after last
        
        Args:
            md: Measured depth for calculation
            trajectory_points: List of trajectory points in PAPI format with fields:
                - 'measuredDepth': Measured depth
                - 'inclinationRad': Inclination angle in radians
                - 'azimutRad': Azimuth angle in radians (optional, defaults to 0.0)
            well_metadata: Well metadata dictionary with PAPI format
            measure_units: Units for calculation (METERS or FEET)
                
        Returns:
            Calculated TVD for given MD
        """
        logger.debug(f"Interpolating TVD for MD={md} using {len(trajectory_points)} trajectory points")
        
        # Case 1: Zero points - strictly vertical well
        if len(trajectory_points) == 0:
            logger.debug(f"MD={md}: Zero trajectory points, vertical well (TVD=MD)")
            return md
        
        # Case 2: Single point - vertical before point, extrapolate after
        if len(trajectory_points) == 1:
            point = trajectory_points[0]
            point_md = point['measuredDepth']
            
            if md <= point_md:
                # Before point - vertical
                logger.debug(f"MD={md}: Before single point at {point_md}, vertical (TVD=MD)")
                return md
            else:
                # After point - extrapolate with point angle
                tvd = TrajectoryInterpolator._extrapolate_single_point(md, point, point_md)
                logger.debug(f"MD={md}: After single point at {point_md}, extrapolated TVD={tvd:.2f}")
                return tvd
        
        # Case 3: Multiple points - check MD position
        first_md = trajectory_points[0]['measuredDepth']
        last_md = trajectory_points[-1]['measuredDepth']
        
        if md <= first_md:
            # Before first point - vertical
            logger.debug(f"MD={md}: Before first point at {first_md}, vertical (TVD=MD)")
            return md
        
        elif md >= last_md:
            # After last point - extrapolate with last segment angle
            tvd = TrajectoryInterpolator._extrapolate_last_segment(
                md, trajectory_points, well_metadata, measure_units
            )
            logger.debug(f"MD={md}: After last point at {last_md}, extrapolated TVD={tvd:.2f}")
            return tvd
        
        else:
            # Between points - use Solo SDK interpolation
            tvd = TrajectoryInterpolator._interpolate_with_solo_sdk(
                md, trajectory_points, well_metadata, measure_units
            )
            logger.debug(f"MD={md}: Between trajectory points, Solo SDK TVD={tvd:.2f}")
            return tvd
    
    @staticmethod
    def _extrapolate_single_point(md: float, point: Dict, point_md: float) -> float:
        """Extrapolate TVD using single trajectory point angle"""
        
        # Calculate TVD at the trajectory point (assume vertical up to that point)
        point_tvd = point_md
        
        # Extrapolate beyond point using its inclination
        delta_md = md - point_md
        incl_rad = point['inclinationRad']  # Already in radians
        delta_tvd = delta_md * math.cos(incl_rad)
        
        return point_tvd + delta_tvd
    
    @staticmethod
    def _extrapolate_last_segment(
        md: float, 
        trajectory_points: List[Dict], 
        well_metadata: Dict,
        measure_units: EMeasureUnits
    ) -> float:
        """Extrapolate TVD beyond last trajectory point using last segment angle"""
        
        # Get last trajectory point
        last_point = trajectory_points[-1]
        last_md = last_point['measuredDepth']
        
        # Calculate TVD at last point using Solo SDK up to that point
        tvd_at_last_point = TrajectoryInterpolator._interpolate_with_solo_sdk(
            last_md, trajectory_points, well_metadata, measure_units
        )
        
        # Extrapolate beyond last point
        delta_md = md - last_md
        
        if len(trajectory_points) >= 2:
            # Use last segment average inclination
            prev_point = trajectory_points[-2]
            avg_incl = (last_point['inclinationRad'] + prev_point['inclinationRad']) / 2
        else:
            # Single segment - use point inclination
            avg_incl = last_point['inclinationRad']
        
        delta_tvd = delta_md * math.cos(avg_incl)
        return tvd_at_last_point + delta_tvd
    
    @staticmethod
    def _interpolate_with_solo_sdk(
        md: float, 
        trajectory_points: List[Dict], 
        well_metadata: Dict,
        measure_units: EMeasureUnits
    ) -> float:
        """Interpolate TVD using Solo SDK between trajectory points"""
        
        # Prepare trajectory data for Solo SDK (convert from PAPI format)
        raw_trajectory = []
        for point in trajectory_points:
            raw_trajectory.append({
                'md': point['measuredDepth'],
                'incl': point['inclinationRad'],  # Already in radians
                'azim': point.get('azimutRad', 0.0)  # Default to 0.0 if missing
            })
        
        # Extract well parameters from PAPI metadata format
        well_params = {
            'convergence': well_metadata['convergence']['val'],
            'kb': well_metadata['kb']['val'],
            'xsrf': well_metadata['xsrf']['val'],
            'ysrf': well_metadata['ysrf']['val'],
            'tie_in_tvd': well_metadata['tie_in_tvd']['val'],
            'tie_in_ns': well_metadata['tie_in_ns']['val'],
            'tie_in_ew': well_metadata['tie_in_ew']['val'],
            'azimuth': math.radians(well_metadata.get('azimuth', {}).get('val', 0.0))  # Optional for typewells
        }
        
        # Calculate full trajectory using Solo SDK
        calculated_trajectory = calculate_trajectory(
            raw_trajectory=raw_trajectory,
            well=well_params,
            measure_units=measure_units
        )
        
        assert calculated_trajectory, f"Solo SDK failed to calculate trajectory for MD={md}"
        
        logger.debug(f"Solo SDK calculated {len(calculated_trajectory)} trajectory points")
        
        # Find interpolation interval within calculated trajectory
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