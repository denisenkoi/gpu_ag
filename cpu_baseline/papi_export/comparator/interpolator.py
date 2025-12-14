import logging
import numpy as np
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class DataInterpolator:
    """Interpolates reference data to target grid WITHOUT extrapolation"""
    
    def __init__(self):
        """Initialize interpolator"""
        self.interpolation_stats = {}
        
    def interpolate_to_target_grid(self, reference: Dict, generated: Dict) -> Dict:
        """Main interpolation method - interpolates reference to generated grid
        
        Args:
            reference: Reference data dictionary
            generated: Generated data dictionary with target grids
            
        Returns:
            Interpolated reference data
        """
        logger.info("Interpolating reference data to PAPI grid WITHOUT extrapolation")
        
        # Deep copy to avoid modifying original
        import copy
        ref_interpolated = copy.deepcopy(reference)
        
        # Get target grids from generated data
        gen_trajectory = generated.get('well', {}).get('points', [])
        gen_welllog_points = generated.get('wellLog', {}).get('points', [])
        gen_typewell_points = generated.get('typeLog', {}).get('points', [])
        gen_grid_points = generated.get('gridSlice', {}).get('points', [])
        
        # Log target grids
        if gen_trajectory:
            trajectory_md = [p['measuredDepth'] for p in gen_trajectory]
            logger.info(f"Target lateral trajectory grid: {len(trajectory_md)} points, "
                       f"MD: {trajectory_md[0]:.1f}→{trajectory_md[-1]:.1f}")
        
        if gen_welllog_points:
            welllog_md = [p['measuredDepth'] for p in gen_welllog_points]
            logger.info(f"Target wellLog grid: {len(welllog_md)} points, "
                       f"MD: {welllog_md[0]:.1f}→{welllog_md[-1]:.1f}")
                       
        if gen_typewell_points:
            typewell_md = [p['measuredDepth'] for p in gen_typewell_points]
            logger.info(f"Target typewell grid: {len(typewell_md)} points, "
                       f"MD: {typewell_md[0]:.1f}→{typewell_md[-1]:.1f}")
        
        # Interpolate each data type
        # 1. Trajectory
        if gen_trajectory:
            ref_trajectory = ref_interpolated.get('well', {}).get('points', [])
            if ref_trajectory:
                target_md = np.array([p['measuredDepth'] for p in gen_trajectory])
                ref_interpolated['well']['points'] = self._interpolate_trajectory_points(
                    ref_trajectory, target_md
                )
                self._log_interpolation_result('well.points', ref_trajectory, 
                                              ref_interpolated['well']['points'])
        
        # 2. WellLog points - use wellLog's own MD grid
        if gen_welllog_points:
            ref_welllog = ref_interpolated.get('wellLog', {}).get('points', [])
            if ref_welllog:
                target_md = np.array([p['measuredDepth'] for p in gen_welllog_points])
                ref_interpolated['wellLog']['points'] = self._interpolate_log_points(
                    ref_welllog, target_md
                )
                self._log_interpolation_result('wellLog.points', ref_welllog,
                                              ref_interpolated['wellLog']['points'])
        
        # 3. WellLog tvdSortedPoints - use wellLog's MD grid
        if gen_welllog_points:
            ref_welllog_tvd = ref_interpolated.get('wellLog', {}).get('tvdSortedPoints', [])
            if ref_welllog_tvd:
                target_md = np.array([p['measuredDepth'] for p in gen_welllog_points])
                ref_interpolated['wellLog']['tvdSortedPoints'] = self._interpolate_log_tvd_points(
                    ref_welllog_tvd, target_md
                )
                self._log_interpolation_result('wellLog.tvdSortedPoints', ref_welllog_tvd,
                                              ref_interpolated['wellLog']['tvdSortedPoints'])
        
        # 4. TypeLog points - use typewell's own MD grid
        if gen_typewell_points:
            ref_typewell = ref_interpolated.get('typeLog', {}).get('points', [])
            if ref_typewell:
                target_md = np.array([p['measuredDepth'] for p in gen_typewell_points])
                ref_interpolated['typeLog']['points'] = self._interpolate_log_points(
                    ref_typewell, target_md
                )
                self._log_interpolation_result('typeLog.points', ref_typewell,
                                              ref_interpolated['typeLog']['points'])
        
        # 5. TypeLog tvdSortedPoints
        if gen_typewell_points:
            ref_typewell_tvd = ref_interpolated.get('typeLog', {}).get('tvdSortedPoints', [])
            if ref_typewell_tvd:
                target_md = np.array([p['measuredDepth'] for p in gen_typewell_points])
                ref_interpolated['typeLog']['tvdSortedPoints'] = self._interpolate_log_tvd_points(
                    ref_typewell_tvd, target_md
                )
                self._log_interpolation_result('typeLog.tvdSortedPoints', ref_typewell_tvd,
                                              ref_interpolated['typeLog']['tvdSortedPoints'])
        
        # 6. GridSlice points - use grid's own MD grid
        if gen_grid_points:
            ref_grid = ref_interpolated.get('gridSlice', {}).get('points', [])
            if ref_grid:
                target_md = np.array([p['measuredDepth'] for p in gen_grid_points])
                ref_interpolated['gridSlice']['points'] = self._interpolate_grid_points(
                    ref_grid, target_md
                )
                self._log_interpolation_result('gridSlice.points', ref_grid,
                                              ref_interpolated['gridSlice']['points'])
        
        return ref_interpolated
    
    def _get_valid_range(self, ref_md: np.ndarray, target_md: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Get overlapping MD range between reference and target
        
        Args:
            ref_md: Reference MD array
            target_md: Target MD array
            
        Returns:
            Tuple of (valid_target_md, valid_indices)
        """
        ref_min, ref_max = ref_md[0], ref_md[-1]
        
        # Find target points within reference range (NO EXTRAPOLATION)
        valid_mask = (target_md >= ref_min) & (target_md <= ref_max)
        valid_target = target_md[valid_mask]
        
        return valid_target, valid_mask
    
    def _interpolate_trajectory_points(self, ref_points: List[Dict], target_md: np.ndarray) -> List[Dict]:
        """Interpolate trajectory points WITHOUT extrapolation
        
        Args:
            ref_points: Reference trajectory points
            target_md: Target MD grid
            
        Returns:
            Interpolated points only within reference range
        """
        if not ref_points:
            return []
        
        # Extract reference arrays
        ref_md = np.array([p['measuredDepth'] for p in ref_points])
        ref_tvd = np.array([p['trueVerticalDepth'] for p in ref_points])
        ref_ns = np.array([p['northSouth'] for p in ref_points])
        ref_ew = np.array([p['eastWest'] for p in ref_points])
        
        # Check optional fields
        has_incl = 'inclinationRad' in ref_points[0]
        has_azim = 'azimutRad' in ref_points[0]
        
        if has_incl:
            ref_incl = np.array([p['inclinationRad'] for p in ref_points])
        if has_azim:
            ref_azim = np.array([p['azimutRad'] for p in ref_points])
        
        # Get valid range (NO EXTRAPOLATION)
        valid_md, valid_mask = self._get_valid_range(ref_md, target_md)
        
        if len(valid_md) == 0:
            logger.warning(f"No overlap between reference trajectory MD range "
                         f"[{ref_md[0]:.1f}, {ref_md[-1]:.1f}] and target grid")
            return []
        
        # Interpolate only within valid range
        interp_tvd = np.interp(valid_md, ref_md, ref_tvd)
        interp_ns = np.interp(valid_md, ref_md, ref_ns)
        interp_ew = np.interp(valid_md, ref_md, ref_ew)
        
        if has_incl:
            interp_incl = np.interp(valid_md, ref_md, ref_incl)
        if has_azim:
            interp_azim = np.interp(valid_md, ref_md, ref_azim)
        
        # Build interpolated points
        interpolated_points = []
        for i, md in enumerate(valid_md):
            point = {
                'measuredDepth': float(md),
                'trueVerticalDepth': float(interp_tvd[i]),
                'northSouth': float(interp_ns[i]),
                'eastWest': float(interp_ew[i])
            }
            
            if has_incl:
                point['inclinationRad'] = float(interp_incl[i])
            if has_azim:
                point['azimutRad'] = float(interp_azim[i])
                
            interpolated_points.append(point)
        
        return interpolated_points
    
    def _interpolate_log_points(self, ref_points: List[Dict], target_md: np.ndarray) -> List[Dict]:
        """Interpolate log points (with data field) WITHOUT extrapolation
        
        Args:
            ref_points: Reference log points
            target_md: Target MD grid
            
        Returns:
            Interpolated points only within reference range
        """
        if not ref_points:
            return []
        
        # Extract reference arrays
        ref_md = np.array([p['measuredDepth'] for p in ref_points])
        ref_data = np.array([p['data'] for p in ref_points])
        
        # Get valid range (NO EXTRAPOLATION)
        valid_md, valid_mask = self._get_valid_range(ref_md, target_md)
        
        if len(valid_md) == 0:
            logger.warning(f"No overlap between reference log MD range "
                         f"[{ref_md[0]:.1f}, {ref_md[-1]:.1f}] and target grid")
            return []
        
        # Interpolate only within valid range
        interp_data = np.interp(valid_md, ref_md, ref_data)
        
        # Build interpolated points
        interpolated_points = []
        for i, md in enumerate(valid_md):
            point = {
                'measuredDepth': float(md),
                'data': float(interp_data[i])
            }
            interpolated_points.append(point)
        
        return interpolated_points
    
    def _interpolate_log_tvd_points(self, ref_points: List[Dict], target_md: np.ndarray) -> List[Dict]:
        """Interpolate log TVD points WITHOUT extrapolation
        
        Args:
            ref_points: Reference log TVD points
            target_md: Target MD grid
            
        Returns:
            Interpolated points only within reference range
        """
        if not ref_points:
            return []
        
        # Extract reference arrays
        ref_md = np.array([p['measuredDepth'] for p in ref_points])
        ref_tvd = np.array([p['trueVerticalDepth'] for p in ref_points])
        ref_data = np.array([p['data'] for p in ref_points])
        
        # Get valid range (NO EXTRAPOLATION)
        valid_md, valid_mask = self._get_valid_range(ref_md, target_md)
        
        if len(valid_md) == 0:
            logger.warning(f"No overlap between reference log TVD MD range "
                         f"[{ref_md[0]:.1f}, {ref_md[-1]:.1f}] and target grid")
            return []
        
        # Interpolate only within valid range
        interp_tvd = np.interp(valid_md, ref_md, ref_tvd)
        interp_data = np.interp(valid_md, ref_md, ref_data)
        
        # Build interpolated points
        interpolated_points = []
        for i, md in enumerate(valid_md):
            point = {
                'measuredDepth': float(md),
                'trueVerticalDepth': float(interp_tvd[i]),
                'data': float(interp_data[i])
            }
            interpolated_points.append(point)
        
        return interpolated_points
    
    def _interpolate_grid_points(self, ref_points: List[Dict], target_md: np.ndarray) -> List[Dict]:
        """Interpolate grid slice points WITHOUT extrapolation
        
        Args:
            ref_points: Reference grid points
            target_md: Target MD grid
            
        Returns:
            Interpolated points only within reference range
        """
        if not ref_points:
            return []
        
        # Extract reference arrays
        ref_md = np.array([p['measuredDepth'] for p in ref_points])
        
        # Handle different field names for TVDSS
        if 'trueVerticalDepthSubSea' in ref_points[0]:
            ref_tvdss = np.array([p['trueVerticalDepthSubSea'] for p in ref_points])
        else:
            ref_tvdss = np.array([p.get('trueVerticalDepth', 0.0) for p in ref_points])
        
        ref_ns = np.array([p['northSouth'] for p in ref_points])
        ref_ew = np.array([p['eastWest'] for p in ref_points])
        ref_vs = np.array([p['verticalSection'] for p in ref_points])
        
        # Get valid range (NO EXTRAPOLATION)
        valid_md, valid_mask = self._get_valid_range(ref_md, target_md)
        
        if len(valid_md) == 0:
            logger.warning(f"No overlap between reference grid MD range "
                         f"[{ref_md[0]:.1f}, {ref_md[-1]:.1f}] and target grid")
            return []
        
        # Interpolate only within valid range
        interp_tvdss = np.interp(valid_md, ref_md, ref_tvdss)
        interp_ns = np.interp(valid_md, ref_md, ref_ns)
        interp_ew = np.interp(valid_md, ref_md, ref_ew)
        interp_vs = np.interp(valid_md, ref_md, ref_vs)
        
        # Build interpolated points
        interpolated_points = []
        for i, md in enumerate(valid_md):
            point = {
                'measuredDepth': float(md),
                'trueVerticalDepthSubSea': float(interp_tvdss[i]),
                'northSouth': float(interp_ns[i]),
                'eastWest': float(interp_ew[i]),
                'verticalSection': float(interp_vs[i])
            }
            interpolated_points.append(point)
        
        return interpolated_points
    
    def _log_interpolation_result(self, name: str, original: List, interpolated: List):
        """Log interpolation statistics
        
        Args:
            name: Data type name
            original: Original points list
            interpolated: Interpolated points list
        """
        orig_count = len(original)
        interp_count = len(interpolated)
        
        if orig_count == 0:
            logger.info(f"{name}: No original data")
            return
        
        if interp_count == 0:
            logger.warning(f"{name}: No points after interpolation (no overlap)")
            return
        
        ratio = interp_count / orig_count * 100
        
        if ratio < 100:
            logger.info(f"{name}: {orig_count} → {interp_count} points ({ratio:.1f}% coverage)")
        else:
            logger.info(f"{name}: {orig_count} → {interp_count} points (full coverage)")
        
        # Store stats
        self.interpolation_stats[name] = {
            'original_count': orig_count,
            'interpolated_count': interp_count,
            'coverage_percent': ratio
        }