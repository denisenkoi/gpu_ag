#!/usr/bin/env python3
"""
Landing Detector - Standalone Module for Well Trajectory Analysis
Detects landing section in well trajectories using clustering and visualization
"""

import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TrajectoryPoint:
    """Single trajectory point"""
    measured_depth: float
    inclination_deg: float
    true_vertical_depth: float = 0.0
    inclination_derivative: float = 0.0
    cluster_id: int = -1


@dataclass
class LandingBoundaries:
    """Landing detection results"""
    start_md: float
    end_md: float
    vertical_avg_angle: float
    landing_avg_angle: float
    horizontal_avg_angle: float
    optimal_start_md: float
    alternative_start_md: float


class TrajectoryResampler:
    """Resamples trajectory to uniform step intervals"""

    def __init__(self, step_meters: float = 5.0):
        self.step_meters = step_meters

    def resample_trajectory(self, well_data: Dict[str, Any]) -> List[TrajectoryPoint]:
        """Extract and resample trajectory points"""
        try:
            points_data = well_data['well']['points']

            # Extract MD, inclination, and TVD
            md_array = np.array([p['measuredDepth'] for p in points_data])
            incl_rad_array = np.array([p['inclinationRad'] for p in points_data])
            tvd_array = np.array([p['trueVerticalDepth'] for p in points_data])

            # Convert to degrees
            incl_deg_array = np.degrees(incl_rad_array)

            # Create uniform MD grid
            md_start = md_array[0]
            md_end = md_array[-1]
            uniform_md = np.arange(md_start, md_end + self.step_meters, self.step_meters)

            # Interpolate inclination and TVD to uniform grid
            uniform_incl = np.interp(uniform_md, md_array, incl_deg_array)
            uniform_tvd = np.interp(uniform_md, md_array, tvd_array)

            # Compute derivative (degrees per meter)
            derivatives = np.gradient(uniform_incl, self.step_meters)

            # Create trajectory points
            trajectory = []
            for i, (md, incl, tvd, deriv) in enumerate(zip(uniform_md, uniform_incl, uniform_tvd, derivatives)):
                trajectory.append(TrajectoryPoint(
                    measured_depth=md,
                    inclination_deg=incl,
                    true_vertical_depth=tvd,
                    inclination_derivative=deriv
                ))

            logger.info(f"Resampled trajectory: {len(points_data)} -> {len(trajectory)} points")
            return trajectory

        except Exception as e:
            logger.error(f"Failed to resample trajectory: {e}")
            return []


class LandingDetector:
    """Main landing detection logic using 2D clustering"""

    def __init__(self, offset_meters: float, landing_start_angle_deg: float,
                 max_landing_length_meters: float, perch_stability_threshold: float,
                 perch_min_angle_deg: float, perch_stability_window: int,
                 alternative_offset_meters: float):
        self.offset_meters = offset_meters
        self.alternative_offset_meters = alternative_offset_meters
        self.landing_start_angle_deg = landing_start_angle_deg
        self.max_landing_length_meters = max_landing_length_meters
        self.perch_stability_threshold = perch_stability_threshold
        self.perch_min_angle_deg = perch_min_angle_deg
        self.perch_stability_window = perch_stability_window
        self.scaler = StandardScaler()

    def detect_landing(self, trajectory: List[TrajectoryPoint]) -> Optional[LandingBoundaries]:
        """Detect landing using 2D clustering on (angle, derivative)"""
        if len(trajectory) < 10:
            logger.warning("Trajectory too short for landing detection")
            return None

        # Prepare features: [inclination, derivative]
        features = np.array([
            [p.inclination_deg, p.inclination_derivative]
            for p in trajectory
        ])

        # Scale features for clustering
        features_scaled = self.scaler.fit_transform(features)

        # K-means clustering with 3 clusters
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)

        # Assign cluster labels to trajectory points
        for i, point in enumerate(trajectory):
            point.cluster_id = cluster_labels[i]

        # Analyze clusters to identify types
        cluster_stats = self._analyze_clusters(trajectory, cluster_labels)

        # Find landing boundaries
        boundaries = self._find_landing_boundaries(trajectory, cluster_stats)

        return boundaries

    def detect_optimal_start(self, well_data: Dict[str, Any]) -> Tuple[float, float]:
        """Public method for emulator integration.

        Returns:
            Tuple of (alternative_start_md, perch_md):
                - alternative_start_md: for AG start_md calculation (perch + offset)
                - perch_md: landing end point for normalization range
        """
        # Resample trajectory
        resampler = TrajectoryResampler(step_meters=5.0)  # Use fixed step for consistency
        trajectory = resampler.resample_trajectory(well_data)

        if not trajectory:
            raise ValueError("Failed to resample trajectory")

        # Detect landing
        boundaries = self.detect_landing(trajectory)

        if boundaries is None:
            raise ValueError("Landing detection failed")

        logger.info(f"Landing detected: {boundaries.start_md:.1f}m -> {boundaries.end_md:.1f}m, "
                    f"alternative start: {boundaries.alternative_start_md:.1f}m")

        return boundaries.alternative_start_md, boundaries.end_md

    def _analyze_clusters(self, trajectory: List[TrajectoryPoint], labels: np.ndarray) -> Dict[int, Dict[str, float]]:
        """Analyze each cluster to determine its type"""
        cluster_stats = {}

        for cluster_id in range(3):
            mask = labels == cluster_id
            cluster_points = [p for i, p in enumerate(trajectory) if mask[i]]

            if not cluster_points:
                continue

            avg_angle = np.mean([p.inclination_deg for p in cluster_points])
            avg_derivative = np.mean([p.inclination_derivative for p in cluster_points])
            std_derivative = np.std([p.inclination_derivative for p in cluster_points])
            point_count = len(cluster_points)

            cluster_stats[cluster_id] = {
                'avg_angle': avg_angle,
                'avg_derivative': avg_derivative,
                'std_derivative': std_derivative,
                'point_count': point_count,
                'type': self._classify_cluster_type(avg_angle, avg_derivative, std_derivative)
            }

        return cluster_stats

    def _classify_cluster_type(self, avg_angle: float, avg_derivative: float, std_derivative: float) -> str:
        """Classify cluster as vertical, landing, or horizontal"""
        if avg_angle > 70:  # High angle
            if abs(avg_derivative) < 0.5:  # Low derivative
                return 'vertical'
            else:
                return 'landing'
        elif avg_angle < 30:  # Low angle
            return 'horizontal'
        else:  # Medium angle
            if abs(avg_derivative) > 1.0:  # High derivative
                return 'landing'
            else:
                return 'transitional'

    def _find_landing_boundaries(self, trajectory: List[TrajectoryPoint],
                                 cluster_stats: Dict[int, Dict[str, float]]) -> LandingBoundaries:
        """Find start and end of landing section using physical constraints"""

        # Step 1: Find landing start using 60° threshold
        landing_start_idx = self._find_landing_start_60_degrees(trajectory)

        # Step 2: Find landing end using perch point detection within 200m from start
        landing_end_idx = self._find_landing_end_perch_point(trajectory, landing_start_idx)

        # Extract MDs
        start_md = trajectory[landing_start_idx].measured_depth if landing_start_idx < len(trajectory) else trajectory[
            0].measured_depth
        end_md = trajectory[landing_end_idx].measured_depth if landing_end_idx < len(trajectory) else trajectory[
            -1].measured_depth

        # Calculate average angles for each section
        vertical_angles = [p.inclination_deg for p in trajectory[:landing_start_idx]]
        horizontal_angles = [p.inclination_deg for p in trajectory[landing_end_idx:]]
        landing_angles = [p.inclination_deg for p in trajectory[landing_start_idx:landing_end_idx]]

        vertical_avg = np.mean(vertical_angles) if vertical_angles else 90.0
        horizontal_avg = np.mean(horizontal_angles) if horizontal_angles else 10.0
        landing_avg = np.mean(landing_angles) if landing_angles else 45.0

        # Calculate optimal start MD (end of landing + offset)
        optimal_start_md = end_md + self.offset_meters

        # Calculate alternative start MD (end of landing + alternative offset)
        alternative_start_md = end_md + self.alternative_offset_meters

        logger.info(
            f"Landing detection: Start at {start_md:.1f}m ({self.landing_start_angle_deg}° threshold), End at {end_md:.1f}m (perch point)")

        return LandingBoundaries(
            start_md=start_md,
            end_md=end_md,
            vertical_avg_angle=vertical_avg,
            landing_avg_angle=landing_avg,
            horizontal_avg_angle=horizontal_avg,
            optimal_start_md=optimal_start_md,
            alternative_start_md=alternative_start_md
        )

    def _find_landing_start_60_degrees(self, trajectory: List[TrajectoryPoint]) -> int:
        """Find first point where inclination exceeds threshold from vertical"""

        for i, point in enumerate(trajectory):
            # inclination_deg is angle from vertical (0° = vertical, 90° = horizontal)
            # We want first point where angle from vertical > threshold
            if point.inclination_deg > self.landing_start_angle_deg:
                logger.info(
                    f"Landing start detected at MD={point.measured_depth:.1f}m, angle={point.inclination_deg:.1f}°")
                return i

        # No threshold found - this should not happen in real wells
        raise ValueError(f"No landing start found with {self.landing_start_angle_deg}° threshold")

    def _find_landing_end_perch_point(self, trajectory: List[TrajectoryPoint], start_idx: int) -> int:
        """Find landing end point (perch) within max distance from start"""
        start_md = trajectory[start_idx].measured_depth if start_idx < len(trajectory) else 0
        max_search_md = start_md + self.max_landing_length_meters

        # Find the index corresponding to max search distance
        max_search_idx = len(trajectory) - 1
        for i in range(start_idx, len(trajectory)):
            if trajectory[i].measured_depth > max_search_md:
                max_search_idx = i
                break

        # Search for perch point within the allowed range
        best_perch_idx = start_idx
        min_derivative_magnitude = float('inf')

        # Look for point with minimum derivative magnitude (most stable)
        for i in range(start_idx + 5, min(max_search_idx + 1, len(trajectory))):
            # Skip if we're still in steep descent
            if trajectory[i].inclination_deg < self.perch_min_angle_deg:
                continue

            # Look for stability (low derivative)
            derivative_magnitude = abs(trajectory[i].inclination_derivative)
            if derivative_magnitude < min_derivative_magnitude:
                min_derivative_magnitude = derivative_magnitude
                best_perch_idx = i

        # Additional check: look for the point where we've clearly "leveled off"
        # This is where inclination stops changing rapidly
        for i in range(start_idx + 10, min(max_search_idx + 1, len(trajectory))):
            # Check if we have a stable section (low derivative for several points)
            if i + self.perch_stability_window < len(trajectory):
                window_derivatives = [abs(trajectory[j].inclination_derivative)
                                      for j in range(i, i + self.perch_stability_window)]
                avg_derivative = np.mean(window_derivatives)

                if avg_derivative < self.perch_stability_threshold:
                    logger.info(f"Landing end detected at MD={trajectory[i].measured_depth:.1f}m (stable section)")
                    return i

        logger.info(f"Landing end detected at MD={trajectory[best_perch_idx].measured_depth:.1f}m (min derivative)")
        return best_perch_idx


class LandingVisualizer:
    """Visualization of landing detection results"""

    def __init__(self):
        plt.style.use('default')

    def visualize_results(self, trajectory: List[TrajectoryPoint],
                          boundaries: Optional[LandingBoundaries],
                          well_name: str):
        """Create comprehensive visualization"""

        if not trajectory:
            logger.warning("No trajectory data to visualize")
            return

        # Extract data for plotting
        md_array = np.array([p.measured_depth for p in trajectory])
        incl_array = np.array([p.inclination_deg for p in trajectory])
        tvd_array = np.array([p.true_vertical_depth for p in trajectory])
        deriv_array = np.array([p.inclination_derivative for p in trajectory])
        cluster_array = np.array([p.cluster_id for p in trajectory])

        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Landing Detection: {well_name}', fontsize=16, fontweight='bold')

        # Colors for clusters
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        cluster_names = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}

        # Plot 1: MD vs TVD trajectory with clusters (colored line segments)
        for cluster_id in np.unique(cluster_array):
            if cluster_id >= 0:
                mask = cluster_array == cluster_id
                ax1.plot(md_array[mask], tvd_array[mask],
                         color=colors[cluster_id % len(colors)],
                         linewidth=3, alpha=0.8,
                         label=cluster_names.get(cluster_id, f'Cluster {cluster_id}'))

        ax1.set_xlabel('Measured Depth (m)')
        ax1.set_ylabel('True Vertical Depth (m)')
        ax1.set_title('Well Path (MD vs TVD) - Colored by Cluster')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.invert_yaxis()  # TVD increases downward

        # Add boundaries if available
        if boundaries:
            ax1.axvline(boundaries.start_md, color='red', linestyle='--',
                        linewidth=2, alpha=0.7, label='Landing Start')
            ax1.axvline(boundaries.end_md, color='green', linestyle='--',
                        linewidth=2, alpha=0.7, label='Landing End')
            ax1.axvline(boundaries.optimal_start_md, color='purple', linestyle='--',
                        linewidth=2, alpha=0.7, label='Optimal Start')
            ax1.axvline(boundaries.alternative_start_md, color='black', linestyle='--',
                        linewidth=2, alpha=0.7, label='Alternative Start')

        # Plot 2: Trajectory with clusters (MD vs Inclination)
        for cluster_id in np.unique(cluster_array):
            if cluster_id >= 0:
                mask = cluster_array == cluster_id
                ax2.scatter(md_array[mask], incl_array[mask],
                            c=colors[cluster_id % len(colors)],
                            label=cluster_names.get(cluster_id, f'Cluster {cluster_id}'),
                            alpha=0.7, s=20)

        ax2.set_xlabel('Measured Depth (m)')
        ax2.set_ylabel('Inclination (degrees)')
        ax2.set_title('Trajectory (MD vs Inclination) - Colored by Cluster')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Add boundaries if available
        if boundaries:
            ax2.axvline(boundaries.start_md, color='red', linestyle='--',
                        linewidth=2, alpha=0.7)
            ax2.axvline(boundaries.end_md, color='green', linestyle='--',
                        linewidth=2, alpha=0.7)
            ax2.axvline(boundaries.optimal_start_md, color='purple', linestyle='--',
                        linewidth=2, alpha=0.7)
            ax2.axvline(boundaries.alternative_start_md, color='black', linestyle='--',
                        linewidth=2, alpha=0.7)

        # Plot 3: 2D Cluster Scatter
        for cluster_id in np.unique(cluster_array):
            if cluster_id >= 0:
                mask = cluster_array == cluster_id
                ax3.scatter(incl_array[mask], deriv_array[mask],
                            c=colors[cluster_id % len(colors)],
                            label=cluster_names.get(cluster_id, f'Cluster {cluster_id}'),
                            alpha=0.7, s=30)

        ax3.set_xlabel('Inclination (degrees)')
        ax3.set_ylabel('Inclination Derivative (deg/m)')
        ax3.set_title('2D Feature Space (Angle vs Derivative)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Plot 4: Statistics summary
        ax4.axis('off')
        if boundaries:
            stats_text = f"""
Landing Detection Results:

Landing Start MD: {boundaries.start_md:.1f} m
Landing End MD: {boundaries.end_md:.1f} m
Landing Length: {boundaries.end_md - boundaries.start_md:.1f} m

Optimal Start MD: {boundaries.optimal_start_md:.1f} m
Alternative Start MD: {boundaries.alternative_start_md:.1f} m

Average Angles:
• Vertical: {boundaries.vertical_avg_angle:.1f}°
• Landing: {boundaries.landing_avg_angle:.1f}°  
• Horizontal: {boundaries.horizontal_avg_angle:.1f}°

Total Points: {len(trajectory)}

Cluster Analysis:
{self._get_cluster_analysis_text(trajectory)}
            """
        else:
            stats_text = "Landing detection failed"

        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=9,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.show(block=True)  # For PyCharm

        # Log results
        if boundaries:
            logger.info(f"Landing detected: {boundaries.start_md:.1f} -> {boundaries.end_md:.1f} m")
            logger.info(f"Optimal start MD: {boundaries.optimal_start_md:.1f} m")

    def _get_cluster_analysis_text(self, trajectory: List[TrajectoryPoint]) -> str:
        """Generate cluster analysis summary"""
        cluster_summary = {}
        for point in trajectory:
            if point.cluster_id >= 0:
                if point.cluster_id not in cluster_summary:
                    cluster_summary[point.cluster_id] = {
                        'count': 0,
                        'angles': [],
                        'derivatives': []
                    }
                cluster_summary[point.cluster_id]['count'] += 1
                cluster_summary[point.cluster_id]['angles'].append(point.inclination_deg)
                cluster_summary[point.cluster_id]['derivatives'].append(point.inclination_derivative)

        text = ""
        for cluster_id, data in cluster_summary.items():
            avg_angle = np.mean(data['angles'])
            avg_deriv = np.mean(data['derivatives'])
            text += f"Cluster {cluster_id}: {data['count']} pts, "
            text += f"Avg angle: {avg_angle:.1f}°, "
            text += f"Avg deriv: {avg_deriv:.2f}°/m\n"

        return text


def load_well_data(file_path: str) -> Optional[Dict[str, Any]]:
    """Load well data from JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load well data from {file_path}: {e}")
        return None


def main():
    """Main function for standalone testing"""
    # Load configuration
    load_dotenv()

    # Required parameters - no defaults, must be in .env
    try:
        wells_dir = os.environ['WELLS_DIR']
        step_meters = float(os.environ['LANDING_DETECTION_STEP_METERS'])
        offset_meters = float(os.environ['LANDING_OFFSET_METERS'])
        alternative_offset_meters = float(os.environ['LANDING_ALTERNATIVE_OFFSET_METERS'])
        landing_start_angle_deg = float(os.environ['LANDING_START_ANGLE_DEG'])
        max_landing_length_meters = float(os.environ['LANDING_MAX_LENGTH_METERS'])
        perch_stability_threshold = float(os.environ['LANDING_PERCH_STABILITY_THRESHOLD'])
        perch_min_angle_deg = float(os.environ['LANDING_PERCH_MIN_ANGLE_DEG'])
        perch_stability_window = int(os.environ['LANDING_PERCH_STABILITY_WINDOW'])
    except KeyError as e:
        logger.error(f"Missing required environment variable: {e}")
        logger.error("Required .env parameters:")
        logger.error("  WELLS_DIR=./wells_streaming_data")
        logger.error("  LANDING_DETECTION_STEP_METERS=5.0")
        logger.error("  LANDING_OFFSET_METERS=100.0")
        logger.error("  LANDING_ALTERNATIVE_OFFSET_METERS=50.0")
        logger.error("  LANDING_START_ANGLE_DEG=60.0")
        logger.error("  LANDING_MAX_LENGTH_METERS=200.0")
        logger.error("  LANDING_PERCH_STABILITY_THRESHOLD=0.5")
        logger.error("  LANDING_PERCH_MIN_ANGLE_DEG=30.0")
        logger.error("  LANDING_PERCH_STABILITY_WINDOW=5")
        return 1

    logger.info("=== Landing Detector - Standalone Mode ===")
    logger.info(f"Wells directory: {wells_dir}")
    logger.info(f"Resampling step: {step_meters} meters")
    logger.info(f"Start offset: {offset_meters} meters")
    logger.info(f"Alternative offset: {alternative_offset_meters} meters")
    logger.info(f"Landing start angle: {landing_start_angle_deg}°")
    logger.info(f"Max landing length: {max_landing_length_meters} meters")
    logger.info(f"Perch stability threshold: {perch_stability_threshold}°/m")
    logger.info(f"Perch min angle: {perch_min_angle_deg}°")
    logger.info(f"Perch stability window: {perch_stability_window} points")

    # Find well files
    wells_path = Path(wells_dir)
    if not wells_path.exists():
        logger.error(f"Wells directory not found: {wells_dir}")
        return 1

    well_files = list(wells_path.glob('*.json'))
    if not well_files:
        logger.error(f"No JSON files found in {wells_dir}")
        return 1

    logger.info(f"Found {len(well_files)} well files")

    # Initialize components
    resampler = TrajectoryResampler(step_meters=step_meters)
    detector = LandingDetector(
        offset_meters=offset_meters,
        alternative_offset_meters=alternative_offset_meters,
        landing_start_angle_deg=landing_start_angle_deg,
        max_landing_length_meters=max_landing_length_meters,
        perch_stability_threshold=perch_stability_threshold,
        perch_min_angle_deg=perch_min_angle_deg,
        perch_stability_window=perch_stability_window
    )
    visualizer = LandingVisualizer()

    # Process each well
    for i, well_file in enumerate(well_files, 1):
        logger.info(f"\n[{i}/{len(well_files)}] Processing: {well_file.name}")

        # Load well data
        well_data = load_well_data(str(well_file))
        if well_data is None:
            continue

        well_name = well_data.get('wellName', well_file.stem)

        # Resample trajectory
        trajectory = resampler.resample_trajectory(well_data)
        if not trajectory:
            logger.warning(f"Failed to resample trajectory for {well_name}")
            continue

        # Detect landing
        try:
            boundaries = detector.detect_landing(trajectory)
        except ValueError as e:
            logger.error(f"Landing detection failed for {well_name}: {e}")
            continue

        # Visualize results
        visualizer.visualize_results(trajectory, boundaries, well_name)

        # Wait for user input to continue
        input(f"Press Enter to continue to next well...")

    logger.info("Landing detection analysis completed")
    return 0


if __name__ == "__main__":
    main()