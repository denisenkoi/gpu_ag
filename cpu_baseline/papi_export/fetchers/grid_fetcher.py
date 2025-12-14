import math
import logging
import numpy as np
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class GridFetcher:
    """Fetcher for grid data with proper bilinear interpolation along well trajectory"""
    
    def __init__(self, config, api_connector):
        self.config = config
        self.api = api_connector
        
    def fetch_grid_slice(
        self, 
        project_uuid: str, 
        trajectory_points: List[Dict],
        x_surface: float,
        y_surface: float
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch grid and calculate slice along well trajectory with proper mathematics
        
        Args:
            project_uuid: Project UUID
            trajectory_points: Trajectory points with 'md', 'ew', 'ns' fields
            x_surface: Well surface X coordinate (absolute)
            y_surface: Well surface Y coordinate (absolute)
            
        Returns:
            Grid slice data with interpolated Z values
        """
        logger.info(f"Fetching grid data for project {project_uuid}")
        logger.info(f"Surface coordinates: X={x_surface:.1f}, Y={y_surface:.1f}")
        
        # Get grids for project
        grids = self.api.get_grids_by_project(project_uuid)
        if not grids:
            raise ValueError("No grids found for project")

        # Find target grid by name
        target_grid = None
        available_grids = []
        for grid in grids:
            grid_name_str = grid.get('name', 'Unknown')
            available_grids.append(grid_name_str)
            if grid_name_str == self.config.grid_name:
                target_grid = grid
                break

        if not target_grid:
            grids_list = ', '.join(available_grids) if available_grids else 'None'
            error_msg = (
                f"Grid '{self.config.grid_name}' not found in project. "
                f"Available grids: {grids_list}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        grid_uuid = target_grid['uuid']
        grid_name = target_grid['name']
        logger.info(f"Using grid '{grid_name}' ({grid_uuid})")
        
        # Get grid data
        grid_data = self.api.get_grid_data(grid_uuid)
        
        # Calculate grid slice with proper mathematics
        grid_slice_points = self._calculate_grid_slice_with_proper_math(
            grid_data, trajectory_points, x_surface, y_surface
        )
        
        grid_slice = {
            'uuid': grid_uuid,
            'points': grid_slice_points
        }
        
        return grid_slice
        
    def _calculate_grid_slice_with_proper_math(
        self,
        grid_data: Dict,
        trajectory_points: List[Dict],
        x_surface: float,
        y_surface: float
    ) -> List[Dict]:
        """Calculate grid values along well trajectory with proper bilinear interpolation"""
        
        metadata = grid_data['metadata']
        data_rows = grid_data['data']
        
        # Extract grid parameters - no .get() fallbacks!
        start_x = float(metadata['start_x'])
        start_y = float(metadata['start_y'])
        step_x = float(metadata['step_x'])
        step_y = float(metadata['step_y'])
        angle = float(metadata['angle'])  # Grid rotation angle in degrees
        n_cols = int(metadata['columns'])
        n_rows = int(metadata['rows'])
        
        logger.info(f"Grid parameters: start=({start_x:.1f}, {start_y:.1f}), "
                   f"step=({step_x:.1f}, {step_y:.1f}), angle={angle:.3f}°, "
                   f"size={n_rows}x{n_cols}")
        
        # Build 2D grid array from data
        grid_array = self._build_grid_array(data_rows, n_rows, n_cols)
        
        # Process trajectory points with uniform 20-unit interpolation (like StarSteer)
        grid_slice_points = []
        cumulative_vs = 0.0  # THL calculation
        prev_ew = None
        prev_ns = None
        self.last_correct_tvd = None  # Track last correct TVD for NaN handling
        
        # Create interpolated trajectory with configurable step size
        step_size = getattr(self.config, 'grid_step_size', 20.0)  # Use config or fallback to 20.0
        interpolated_points = self._create_uniform_step_trajectory(trajectory_points, step_size=step_size)
        
        for point in interpolated_points:
            md = point['md']
            ew = point['ew']  # Relative to surface
            ns = point['ns']  # Relative to surface
            
            # Calculate world coordinates
            x_world = ew + x_surface
            y_world = ns + y_surface
            
            # Calculate THL (cumulative distance) for verticalSection
            if prev_ew is not None and prev_ns is not None:
                dx = ew - prev_ew
                dy = ns - prev_ns
                distance = math.sqrt(dx * dx + dy * dy)
                cumulative_vs += distance
            
            prev_ew = ew
            prev_ns = ns
            
            # Interpolate Z value from grid with proper rotation mathematics
            z_interpolated = self._bilinear_interpolation_with_rotation(
                x_world, y_world, grid_array, 
                start_x, start_y, step_x, step_y, angle
            )
            
            # Handle NaN with last correct TVD instead of 0.0
            if np.isnan(z_interpolated):
                if self.last_correct_tvd is not None:
                    z_interpolated = self.last_correct_tvd
                    logger.debug(f"Using last correct TVD {z_interpolated:.3f} for MD {md:.1f}")
                else:
                    z_interpolated = 0.0  # Fallback if no previous values
                    logger.warning(f"No previous TVD available, using 0.0 for MD {md:.1f}")
            else:
                self.last_correct_tvd = z_interpolated
            
            grid_slice_points.append({
                'measuredDepth': md,
                'trueVerticalDepthSubSea': z_interpolated,
                'northSouth': y_world,  # Absolute world coordinates
                'eastWest': x_world,    # Absolute world coordinates  
                'verticalSection': cumulative_vs
            })
            
        logger.info(f"Calculated {len(grid_slice_points)} grid slice points")
        
        if self.config.save_intermediate:
            self._save_grid_slice(grid_slice_points)
            
        return grid_slice_points

    def _create_uniform_step_trajectory(self, trajectory_points: List[Dict], step_size: float) -> List[Dict]:
        """Create trajectory points with uniform step size like StarSteer"""
        if not trajectory_points or len(trajectory_points) < 2:
            return trajectory_points
            
        # Sort by MD to ensure correct order
        sorted_points = sorted(trajectory_points, key=lambda p: p['md'])
        
        start_md = sorted_points[0]['md']
        end_md = sorted_points[-1]['md']
        
        # Create array of MDs for interpolation  
        mds_original = np.array([p['md'] for p in sorted_points])
        ews_original = np.array([p['ew'] for p in sorted_points])
        nss_original = np.array([p['ns'] for p in sorted_points])
        
        # Generate uniform MD steps
        uniform_mds = []
        current_md = start_md
        while current_md <= end_md:
            uniform_mds.append(current_md)
            current_md += step_size
            
        # Add end point if not exactly on step
        if uniform_mds[-1] < end_md:
            uniform_mds.append(end_md)
            
        uniform_mds = np.array(uniform_mds)
        
        # Interpolate EW and NS coordinates
        ews_interpolated = np.interp(uniform_mds, mds_original, ews_original)
        nss_interpolated = np.interp(uniform_mds, mds_original, nss_original)
        
        # Create interpolated trajectory points
        interpolated_points = []
        for i, md in enumerate(uniform_mds):
            interpolated_points.append({
                'md': md,
                'ew': ews_interpolated[i],
                'ns': nss_interpolated[i]
            })
            
        logger.info(f"Interpolated trajectory: {len(trajectory_points)} → {len(interpolated_points)} points "
                   f"(step size: {step_size})")
        
        return interpolated_points
        
    def _build_grid_array(self, data_rows: List[Dict], n_rows: int, n_cols: int) -> np.ndarray:
        """Build 2D numpy array from grid data rows"""
        
        grid_array = np.full((n_rows, n_cols), np.nan, dtype=float)
        
        for row_data in data_rows:
            row_number = row_data['row_number']  # 1-based indexing
            values = row_data['values']
            
            # Convert to 0-based indexing
            row_index = row_number - 1
            
            assert 0 <= row_index < n_rows, f"Invalid row_number {row_number} for grid size {n_rows}"
            assert len(values) == n_cols, f"Row {row_number} has {len(values)} values, expected {n_cols}"
            
            for j, val in enumerate(values):
                grid_array[row_index, j] = float(val)
                
        logger.info(f"Built grid array: {n_rows}x{n_cols}")
        return grid_array
        
    def _bilinear_interpolation_with_rotation(
        self,
        x: float, y: float,
        grid: np.ndarray,
        start_x: float, start_y: float,
        step_x: float, step_y: float,
        angle_degrees: float
    ) -> float:
        """
        Bilinear interpolation with grid rotation - exact mathematics from C++
        
        Args:
            x, y: World coordinates
            grid: 2D numpy array with Z values
            start_x, start_y: Grid origin coordinates
            step_x, step_y: Grid cell dimensions  
            angle_degrees: Grid rotation angle in degrees
            
        Returns:
            Interpolated Z value
        """
        
        # Convert angle to radians for rotation
        angle_rad = math.radians(angle_degrees)
        
        # Transform world coordinates to grid coordinates with rotation
        dx = x - start_x
        dy = y - start_y
        
        if abs(angle_degrees) > 1e-6:
            # Apply inverse rotation to get grid coordinates
            cos_a = math.cos(-angle_rad)  # Inverse rotation
            sin_a = math.sin(-angle_rad)
            
            grid_x = (dx * cos_a - dy * sin_a) / step_x
            grid_y = (dx * sin_a + dy * cos_a) / step_y
        else:
            # No rotation case
            grid_x = dx / step_x
            grid_y = dy / step_y
            
        # Get integer grid indices
        i = int(math.floor(grid_y))
        j = int(math.floor(grid_x))
        
        n_rows, n_cols = grid.shape
        
        # Check if point is outside grid bounds
        if i < 0 or i >= n_rows - 1 or j < 0 or j >= n_cols - 1:
            # Outside grid - return nearest edge value
            i = max(0, min(i, n_rows - 1))
            j = max(0, min(j, n_cols - 1))
            edge_value = grid[i, j]
            
            if np.isnan(edge_value):
                logger.warning(f"NaN value at grid edge [{i},{j}] for world coords ({x:.1f}, {y:.1f})")
                return np.nan  # Return NaN to be handled by caller
            
            return float(edge_value)
            
        # Get fractional parts for interpolation
        fx = grid_x - j
        fy = grid_y - i
        
        # Get four corner values for bilinear interpolation
        v00 = grid[i, j]         # Bottom-left
        v10 = grid[i, j + 1]     # Bottom-right  
        v01 = grid[i + 1, j]     # Top-left
        v11 = grid[i + 1, j + 1] # Top-right
        
        # Check for NaN values in corners
        if np.isnan(v00) or np.isnan(v10) or np.isnan(v01) or np.isnan(v11):
            logger.warning(f"NaN values in interpolation corners at grid [{i},{j}] "
                          f"for world coords ({x:.1f}, {y:.1f})")
            return np.nan  # Return NaN to be handled by caller
            
        # Perform bilinear interpolation
        v0 = v00 * (1.0 - fx) + v10 * fx  # Interpolate along X at bottom
        v1 = v01 * (1.0 - fx) + v11 * fx  # Interpolate along X at top
        interpolated_value = v0 * (1.0 - fy) + v1 * fy  # Interpolate along Y
        
        return float(interpolated_value)
        
    def _save_grid_slice(self, grid_slice_points: List[Dict]):
        """Save grid slice to intermediate file for debugging"""
        import json
        output_file = self.config.output_dir / 'intermediate' / 'grid_slice.json'
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'points_count': len(grid_slice_points),
                'points': grid_slice_points
            }, f, indent=2, ensure_ascii=False)
            
        logger.debug(f"Saved grid slice to {output_file}")