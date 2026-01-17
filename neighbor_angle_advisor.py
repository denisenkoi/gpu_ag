"""
Neighbor Angle Advisor (RND-817)

Searches for nearest neighbor wells and recommends dip angles from their interpretations.

Usage:
    advisor = NeighborAngleAdvisor('dataset/gpu_ag_dataset.pt')
    neighbors = advisor.find_neighbors('Well1042~EGFDL', md=4000.0)
    for n in neighbors:
        print(f"{n.well_name} at {n.md:.1f}m: dip={n.dip_angle_deg:.2f}°, dist={n.distance_3d:.1f}m")
"""

import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class NeighborResult:
    """Result of neighbor search"""
    well_name: str
    md: float                 # MD on neighbor well
    distance_3d: float        # Weighted 3D distance
    azimuth_diff_deg: float   # Azimuth difference (0-180)
    dip_angle_deg: float      # Dip angle from reference interpretation
    is_opposite: bool = False # True if well goes opposite direction (angle flipped)
    side: str = ""            # "L" = left, "R" = right (looking in azimuth direction)


class NeighborAngleAdvisor:
    """
    Finds nearest neighbor wells and extracts dip angles from their interpretations.

    Uses absolute 3D coordinates:
        X = x_surface + eastWest
        Y = y_surface + northSouth
        Z = trueVerticalDepth

    Weights TVD more heavily (geology changes more vertically than horizontally).
    Filters by azimuth to ensure wells are roughly parallel.
    """

    def __init__(self, dataset_path: str, tvd_weight: float = 2.0):
        """
        Args:
            dataset_path: Path to gpu_ag_dataset.pt
            tvd_weight: Weight for TVD in distance calculation (>1 means TVD matters more)
        """
        self.tvd_weight = tvd_weight
        self.dataset = torch.load(dataset_path, weights_only=False)
        self.well_names = list(self.dataset.keys())

        # Build index of all points with absolute coordinates
        self._build_spatial_index()

        logger.info(f"NeighborAngleAdvisor initialized with {len(self.well_names)} wells")

    def _build_spatial_index(self):
        """Build arrays for fast spatial search (only lateral section, MD > perch_md)"""
        all_points = []  # (well_idx, point_idx, x_abs, y_abs, z, md, azimuth_rad)

        for well_idx, well_name in enumerate(self.well_names):
            data = self.dataset[well_name]

            # Get perch_md (landing end point) - only index points after this
            perch_md = data.get('perch_md', 0.0)

            # Get surface coordinates
            x_surface = data.get('x_surface', 0.0)
            y_surface = data.get('y_surface', 0.0)

            # Get trajectory arrays
            well_md = data['well_md'].numpy()
            well_tvd = data['well_tvd'].numpy()
            well_ns = data['well_ns'].numpy()
            well_ew = data['well_ew'].numpy()

            # Calculate absolute coordinates
            x_abs = x_surface + well_ew
            y_abs = y_surface + well_ns

            # Calculate azimuth from trajectory (approximate from NS/EW changes)
            # azimuth = atan2(dEW, dNS)
            azimuth_rad = np.zeros(len(well_md))
            if len(well_md) > 1:
                dew = np.diff(well_ew)
                dns = np.diff(well_ns)
                azimuth_rad[1:] = np.arctan2(dew, dns)
                azimuth_rad[0] = azimuth_rad[1] if len(azimuth_rad) > 1 else 0

            # Only add points where MD > perch_md (lateral section)
            for i in range(len(well_md)):
                if well_md[i] > perch_md:
                    all_points.append((
                        well_idx,
                        i,
                        x_abs[i],
                        y_abs[i],
                        well_tvd[i],
                        well_md[i],
                        azimuth_rad[i]
                    ))

        # Convert to numpy arrays for fast operations
        self.index_well_idx = np.array([p[0] for p in all_points], dtype=np.int32)
        self.index_point_idx = np.array([p[1] for p in all_points], dtype=np.int32)
        self.index_x = np.array([p[2] for p in all_points], dtype=np.float64)
        self.index_y = np.array([p[3] for p in all_points], dtype=np.float64)
        self.index_z = np.array([p[4] for p in all_points], dtype=np.float64)
        self.index_md = np.array([p[5] for p in all_points], dtype=np.float64)
        self.index_azimuth = np.array([p[6] for p in all_points], dtype=np.float64)

        logger.info(f"Spatial index built with {len(all_points)} points")

    def _calc_azimuth_diff_deg(self, az1_rad: float, az2_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate azimuth difference handling 0-360 wraparound.

        Returns:
            Tuple of (diff_degrees, is_opposite):
            - diff_degrees: difference in degrees (0-180)
            - is_opposite: True if wells are going opposite directions (~180° diff)
        """
        diff_rad = np.abs(az1_rad - az2_rad)
        # Handle wraparound: if diff > pi, use 2*pi - diff
        diff_rad = np.minimum(diff_rad, 2 * np.pi - diff_rad)
        diff_deg = np.degrees(diff_rad)

        # Opposite = diff close to 180°
        is_opposite = diff_deg > 165.0  # within 15° of 180°

        return diff_deg, is_opposite

    def _get_dip_angle_at_md(self, well_name: str, md: float, smoothing_range: float = 100.0) -> Optional[float]:
        """
        Get smoothed dip angle from reference interpretation at given MD.

        Uses trend over ±smoothing_range/2 meters window instead of single segment.
        If data missing on one side, extends window on other side.

        Args:
            well_name: Well name
            md: Target MD
            smoothing_range: Total window size in meters (default 100 = ±50m)

        Returns angle in degrees, or None if no interpretation data.
        """
        data = self.dataset[well_name]
        ref_mds = data['ref_segment_mds'].numpy()
        ref_start_shifts = data['ref_start_shifts'].numpy()
        ref_shifts = data['ref_shifts'].numpy()

        if len(ref_mds) == 0:
            return None

        last_md = data['lateral_well_last_md']
        half_range = smoothing_range / 2

        # Target window
        md_left = md - half_range
        md_right = md + half_range

        # Clamp to data bounds and extend other side if needed
        first_seg_md = ref_mds[0]
        if md_left < first_seg_md:
            extra = first_seg_md - md_left
            md_left = first_seg_md
            md_right = min(md_right + extra, last_md)

        if md_right > last_md:
            extra = md_right - last_md
            md_right = last_md
            md_left = max(md_left - extra, first_seg_md)

        # Interpolate shift at md_left and md_right
        def get_shift_at_md(target_md):
            for i in range(len(ref_mds)):
                seg_start = ref_mds[i]
                seg_end = ref_mds[i + 1] if i + 1 < len(ref_mds) else last_md

                if seg_start <= target_md <= seg_end:
                    # Linear interpolation within segment
                    if seg_end > seg_start:
                        t = (target_md - seg_start) / (seg_end - seg_start)
                        return ref_start_shifts[i] + t * (ref_shifts[i] - ref_start_shifts[i])
                    return ref_start_shifts[i]
            return None

        shift_left = get_shift_at_md(md_left)
        shift_right = get_shift_at_md(md_right)

        if shift_left is None or shift_right is None:
            return None

        dmd = md_right - md_left
        dshift = shift_right - shift_left

        if dmd > 0:
            return np.degrees(np.arctan2(dshift, dmd))

        return None

    def find_neighbors(
        self,
        well_name: str,
        md: float,
        azimuth_threshold_deg: float = 15.0,
        max_neighbors: int = 5,
        max_distance: float = float('inf'),
        include_opposite: bool = True,
        smoothing_range: float = 100.0
    ) -> List[NeighborResult]:
        """
        Find nearest neighbor points on other wells.

        Args:
            well_name: Target well name
            md: Target MD on the well
            azimuth_threshold_deg: Max azimuth difference to consider parallel (0-180)
            max_neighbors: Maximum number of neighbors to return
            max_distance: Maximum 3D distance to consider
            include_opposite: Also search wells going opposite direction (flip dip angle)
            smoothing_range: Window size for dip smoothing (default 100m)

        Returns:
            List of NeighborResult sorted by distance (deduplicated by well)
        """
        # Get target well data
        if well_name not in self.dataset:
            logger.warning(f"Well {well_name} not in dataset")
            return []

        data = self.dataset[well_name]

        # Get target point coordinates
        well_md_arr = data['well_md'].numpy()
        well_tvd = data['well_tvd'].numpy()
        well_ns = data['well_ns'].numpy()
        well_ew = data['well_ew'].numpy()
        x_surface = data.get('x_surface', 0.0)
        y_surface = data.get('y_surface', 0.0)

        # Interpolate to get coordinates at target MD
        if md < well_md_arr[0] or md > well_md_arr[-1]:
            logger.warning(f"MD {md} outside well trajectory range [{well_md_arr[0]:.1f}, {well_md_arr[-1]:.1f}]")
            return []

        target_tvd = np.interp(md, well_md_arr, well_tvd)
        target_ns = np.interp(md, well_md_arr, well_ns)
        target_ew = np.interp(md, well_md_arr, well_ew)

        target_x = x_surface + target_ew
        target_y = y_surface + target_ns
        target_z = target_tvd

        # Calculate target azimuth (from nearby points)
        idx = np.searchsorted(well_md_arr, md)
        if idx > 0 and idx < len(well_md_arr):
            dew = well_ew[idx] - well_ew[idx - 1]
            dns = well_ns[idx] - well_ns[idx - 1]
            target_azimuth_rad = np.arctan2(dew, dns)
        else:
            target_azimuth_rad = 0.0

        # Find target well index to exclude
        target_well_idx = self.well_names.index(well_name)

        # Filter out same well
        mask = self.index_well_idx != target_well_idx

        # Calculate distances
        dx = self.index_x[mask] - target_x
        dy = self.index_y[mask] - target_y
        dz = self.index_z[mask] - target_z

        # Weighted distance: TVD matters more
        distances = np.sqrt(dx**2 + dy**2 + (self.tvd_weight * dz)**2)

        # Filter by azimuth (parallel or opposite)
        azimuth_diffs, is_opposite = self._calc_azimuth_diff_deg(target_azimuth_rad, self.index_azimuth[mask])

        # Parallel: azimuth diff <= threshold
        parallel_mask = azimuth_diffs <= azimuth_threshold_deg

        # Opposite: azimuth diff >= 180 - threshold (i.e. within threshold of 180°)
        opposite_mask = azimuth_diffs >= (180.0 - azimuth_threshold_deg)

        if include_opposite:
            azimuth_mask = parallel_mask | opposite_mask
        else:
            azimuth_mask = parallel_mask

        # Filter by distance
        distance_mask = distances <= max_distance

        # Combined mask
        final_mask = azimuth_mask & distance_mask

        if not np.any(final_mask):
            logger.debug(f"No neighbors found for {well_name} at MD={md}")
            return []

        # Get filtered data
        filtered_well_idx = self.index_well_idx[mask][final_mask]
        filtered_distances = distances[final_mask]
        filtered_azimuth_diffs = azimuth_diffs[final_mask]
        filtered_is_opposite = is_opposite[final_mask]
        filtered_md = self.index_md[mask][final_mask]
        filtered_dx = dx[final_mask]
        filtered_dy = dy[final_mask]

        # Sort by distance
        sort_idx = np.argsort(filtered_distances)

        # Build results with deduplication by well
        results = []
        seen_wells = set()

        # Direction vector for left/right calculation
        # Looking in azimuth direction: cross product determines side
        dir_x = np.sin(target_azimuth_rad)  # EW component
        dir_y = np.cos(target_azimuth_rad)  # NS component

        for i in sort_idx:
            neighbor_well_idx = filtered_well_idx[i]

            # Skip if already have this well
            if neighbor_well_idx in seen_wells:
                continue

            neighbor_well_name = self.well_names[neighbor_well_idx]
            neighbor_md = filtered_md[i]
            neighbor_is_opposite = filtered_is_opposite[i]

            # Get dip angle from reference interpretation (smoothed)
            dip_angle = self._get_dip_angle_at_md(neighbor_well_name, neighbor_md, smoothing_range)
            if dip_angle is None:
                continue  # Skip if no interpretation

            # Flip dip angle for opposite direction wells
            if neighbor_is_opposite:
                dip_angle = -dip_angle

            # Determine left/right side
            # cross = dir_x * dy - dir_y * dx
            # cross > 0 = LEFT, cross < 0 = RIGHT
            cross = dir_x * filtered_dy[i] - dir_y * filtered_dx[i]
            side = "L" if cross > 0 else "R"

            seen_wells.add(neighbor_well_idx)
            results.append(NeighborResult(
                well_name=neighbor_well_name,
                md=float(neighbor_md),
                distance_3d=float(filtered_distances[i]),
                azimuth_diff_deg=float(filtered_azimuth_diffs[i]),
                dip_angle_deg=dip_angle,
                is_opposite=bool(neighbor_is_opposite),
                side=side
            ))

            if len(results) >= max_neighbors:
                break

        return results

    def get_recommended_dip(
        self,
        well_name: str,
        md: float,
        azimuth_threshold_deg: float = 15.0,
        max_neighbors: int = 3,
        max_distance: float = 1000.0,  # Max distance to neighbor in meters
        smoothing_range: float = 100.0  # Smoothing window for dip angle
    ) -> Optional[float]:
        """
        Get recommended dip angle as average of nearest neighbors.

        Args:
            max_distance: Maximum distance to neighbor (default 1000m = 1km)
            smoothing_range: Window size for dip smoothing (default 100m = ±50m)

        Returns:
            Average dip angle in degrees, or None if no neighbors found
        """
        neighbors = self.find_neighbors(
            well_name, md,
            azimuth_threshold_deg=azimuth_threshold_deg,
            max_neighbors=max_neighbors,
            max_distance=max_distance,
            smoothing_range=smoothing_range
        )

        if not neighbors:
            return None

        # Weighted average by inverse distance
        weights = [1.0 / (n.distance_3d + 1.0) for n in neighbors]
        total_weight = sum(weights)

        avg_dip = sum(n.dip_angle_deg * w for n, w in zip(neighbors, weights)) / total_weight

        return avg_dip


def main():
    """Test the advisor"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=os.environ.get('DATASET_PATH', 'dataset/gpu_ag_dataset.pt'))
    parser.add_argument('--well', default='Well1042~EGFDL')
    parser.add_argument('--md', type=float, default=4000.0)
    parser.add_argument('--threshold', type=float, default=15.0)
    parser.add_argument('--max-neighbors', type=int, default=5)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    advisor = NeighborAngleAdvisor(args.dataset)

    neighbors = advisor.find_neighbors(
        args.well, args.md,
        azimuth_threshold_deg=args.threshold,
        max_neighbors=args.max_neighbors
    )

    print(f"\nNeighbors for {args.well} at MD={args.md}ft:")
    print("-" * 90)
    for n in neighbors:
        opp = "[OPP]" if n.is_opposite else "     "
        print(f"  {n.side} {n.well_name:30} MD={n.md:7.1f}ft  dist={n.distance_3d:7.1f}ft  "
              f"az_diff={n.azimuth_diff_deg:5.1f}°  dip={n.dip_angle_deg:+6.2f}° {opp}")

    recommended = advisor.get_recommended_dip(args.well, args.md)
    if recommended is not None:
        print(f"\nRecommended dip: {recommended:+.2f}°")


if __name__ == '__main__':
    main()
