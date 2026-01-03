"""
Neighbor Angle Advisor - Production Module (RND-817)

Uses pre-calibrated accuracy config to provide dip predictions
with expected error bounds for well interpretation.
"""

import torch
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class NeighborDipResult:
    """Dip prediction for a single MD point."""
    md: float
    neighbor_config: str  # "L+R", "L_only", "R_only", "none"
    dist_L: float
    dist_R: float

    # Point angle (delta=0)
    dip_0ft: float
    error_0ft_p90: float

    # Trend angles
    dip_100ft: float
    error_100ft_p90: float

    dip_200ft: float
    error_200ft_p90: float

    dip_300ft: float
    error_300ft_p90: float


class NeighborAdvisorProd:
    """
    Production neighbor advisor.

    Loads pre-calibrated config and provides dip predictions
    with error bounds for new wells.
    """

    def __init__(self, dataset_path: str, config_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load dataset (existing wells)
        self.dataset = torch.load(dataset_path, weights_only=False)
        self.well_names = list(self.dataset.keys())

        # Load accuracy config
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Build spatial index
        self._build_index()

        # Build shift curves for dip calculation
        self._build_shift_curves()

        logger.info(f"NeighborAdvisorProd initialized: {len(self.well_names)} wells, device={self.device}")

    def _build_shift_curves(self):
        """Precompute shift curves for all wells."""
        self.shift_curves = {}
        for well_name in self.well_names:
            data = self.dataset[well_name]
            ref_mds = data['ref_segment_mds'].numpy()
            ref_shifts = data['ref_shifts'].numpy()
            last_md = data['lateral_well_last_md']
            perch_md = data.get('perch_md', 0.0)

            if len(ref_mds) > 0:
                mds = np.concatenate([[perch_md], ref_mds, [last_md]])
                shifts = np.concatenate([[0.0, 0.0], ref_shifts])
            else:
                mds = np.array([perch_md, last_md])
                shifts = np.array([0.0, 0.0])

            self.shift_curves[well_name] = (mds, shifts, perch_md, last_md)

    def _build_index(self):
        """Build GPU spatial index from existing wells."""
        all_points = []

        for well_idx, well_name in enumerate(self.well_names):
            data = self.dataset[well_name]
            perch_md = data.get('perch_md', 0.0)
            x_surface = data.get('x_surface', 0.0)
            y_surface = data.get('y_surface', 0.0)

            well_md = data['well_md'].numpy()
            well_tvd = data['well_tvd'].numpy()
            well_ns = data['well_ns'].numpy()
            well_ew = data['well_ew'].numpy()

            x_abs = x_surface + well_ew
            y_abs = y_surface + well_ns

            # Azimuth
            azimuth_rad = np.zeros(len(well_md))
            if len(well_md) > 1:
                dew = np.diff(well_ew)
                dns = np.diff(well_ns)
                azimuth_rad[1:] = np.arctan2(dew, dns)
                azimuth_rad[0] = azimuth_rad[1] if len(azimuth_rad) > 1 else 0

            # Only lateral section (MD > perch_md)
            for i in range(len(well_md)):
                if well_md[i] > perch_md:
                    all_points.append((
                        well_idx, x_abs[i], y_abs[i], well_tvd[i], well_md[i], azimuth_rad[i]
                    ))

        n = len(all_points)
        self.idx_well = torch.tensor([p[0] for p in all_points], dtype=torch.int32, device=self.device)
        self.idx_x = torch.tensor([p[1] for p in all_points], dtype=torch.float32, device=self.device)
        self.idx_y = torch.tensor([p[2] for p in all_points], dtype=torch.float32, device=self.device)
        self.idx_z = torch.tensor([p[3] for p in all_points], dtype=torch.float32, device=self.device)
        self.idx_md = torch.tensor([p[4] for p in all_points], dtype=torch.float32, device=self.device)
        self.idx_az = torch.tensor([p[5] for p in all_points], dtype=torch.float32, device=self.device)

        logger.info(f"Spatial index: {n} points")

    def _get_error_from_config(self, neighbor_config: str, max_dist: float, delta: int) -> float:
        """Lookup expected error from config."""
        matrix = self.config.get('accuracy_matrix', {})

        # Map neighbor_config to matrix key
        if neighbor_config == "L+R":
            key = "L+R_weighted"
        elif neighbor_config == "L_only":
            key = "L_only"
        elif neighbor_config == "R_only":
            key = "R_only"
        else:
            return 999.0  # No neighbors

        entry = matrix.get(key, {})
        by_dist = entry.get('by_distance_ft', {})

        # Find distance bucket
        if max_dist < 100:
            bucket = "0-100"
        elif max_dist < 200:
            bucket = "100-200"
        elif max_dist < 500:
            bucket = "200-500"
        elif max_dist < 1000:
            bucket = "500-1000"
        else:
            bucket = "1000+"

        bucket_data = by_dist.get(bucket, entry.get('all', {}))

        # Adjust for delta (use delta_comparison if available)
        base_p90 = bucket_data.get('p90', 2.0)

        # Scale by delta factor from config
        delta_comp = self.config.get('delta_comparison', {})
        delta_100_p90 = delta_comp.get('100', {}).get('p90', 0.90)
        delta_factor = delta_comp.get(str(delta), {}).get('p90', delta_100_p90) / delta_100_p90

        return base_p90 * delta_factor

    def _get_dip_at_md(self, well_name: str, md: float, delta: float) -> float:
        """Get dip angle (averaged over ±delta, asymmetric if needed)."""
        mds, shifts, perch, last = self.shift_curves[well_name]

        if delta == 0:
            if md < mds[0] or md > mds[-1]:
                return np.nan
            idx = np.searchsorted(mds, md) - 1
            idx = max(0, min(idx, len(mds) - 2))
            dmd = mds[idx + 1] - mds[idx]
            dshift = shifts[idx + 1] - shifts[idx]
            if dmd > 0:
                return np.degrees(np.arctan2(dshift, dmd))
            return np.nan
        else:
            start = max(md - delta, mds[0])
            end = min(md + delta, mds[-1])
            if end <= start:
                return np.nan
            s1 = np.interp(start, mds, shifts)
            s2 = np.interp(end, mds, shifts)
            actual_len = end - start
            if actual_len > 0:
                return np.degrees(np.arctan2(s2 - s1, actual_len))
            return np.nan

    def _find_neighbors_single(
        self,
        target_x: float,
        target_y: float,
        target_z: float,
        target_az: float,
        az_threshold_deg: float = 15.0
    ) -> Tuple[int, float, float, bool, int, float, float, bool]:
        """Find L and R neighbors for single point."""

        # Convert to tensors
        tx = torch.tensor(target_x, dtype=torch.float32, device=self.device)
        ty = torch.tensor(target_y, dtype=torch.float32, device=self.device)
        tz = torch.tensor(target_z, dtype=torch.float32, device=self.device)
        taz = torch.tensor(target_az, dtype=torch.float32, device=self.device)

        # Distances
        dx = self.idx_x - tx
        dy = self.idx_y - ty
        dz = self.idx_z - tz
        distances = torch.sqrt(dx**2 + dy**2 + (2.0 * dz)**2)

        # Azimuth filter
        az_diff = torch.abs(taz - self.idx_az)
        az_diff = torch.minimum(az_diff, 2 * np.pi - az_diff)
        az_diff_deg = torch.rad2deg(az_diff)

        parallel = az_diff_deg <= az_threshold_deg
        opposite = az_diff_deg >= (180.0 - az_threshold_deg)
        az_ok = parallel | opposite

        # Left/Right
        dir_x = torch.sin(taz)
        dir_y = torch.cos(taz)
        cross = dir_x * dy - dir_y * dx
        is_left = cross > 0

        INF = 1e9

        # Find closest L
        dist_L = torch.where(az_ok & is_left, distances, torch.tensor(INF, device=self.device))
        min_dist_L, min_idx_L = torch.min(dist_L, dim=0)

        # Find closest R
        dist_R = torch.where(az_ok & ~is_left, distances, torch.tensor(INF, device=self.device))
        min_dist_R, min_idx_R = torch.min(dist_R, dim=0)

        L_well = int(self.idx_well[min_idx_L].item()) if min_dist_L.item() < INF else -1
        L_md = float(self.idx_md[min_idx_L].item()) if L_well >= 0 else 0.0
        L_dist = float(min_dist_L.item()) if L_well >= 0 else INF
        L_opp = bool(az_diff_deg[min_idx_L].item() >= (180 - az_threshold_deg)) if L_well >= 0 else False

        R_well = int(self.idx_well[min_idx_R].item()) if min_dist_R.item() < INF else -1
        R_md = float(self.idx_md[min_idx_R].item()) if R_well >= 0 else 0.0
        R_dist = float(min_dist_R.item()) if R_well >= 0 else INF
        R_opp = bool(az_diff_deg[min_idx_R].item() >= (180 - az_threshold_deg)) if R_well >= 0 else False

        return L_well, L_md, L_dist, L_opp, R_well, R_md, R_dist, R_opp

    def process_well(
        self,
        well_name: str,
        x_surface: float,
        y_surface: float,
        trajectory: List[Tuple[float, float, float, float]],  # [(md, tvd, ns, ew), ...]
        step: float = 10.0
    ) -> List[NeighborDipResult]:
        """
        Process entire well trajectory and return dip predictions.

        Args:
            well_name: Name for logging
            x_surface: Wellhead X coordinate
            y_surface: Wellhead Y coordinate
            trajectory: List of (md, tvd, ns, ew) tuples
            step: MD step for output (default 10ft)

        Returns:
            List of NeighborDipResult for each MD point
        """
        results = []

        # Convert trajectory to arrays
        traj_arr = np.array(trajectory)
        traj_md = traj_arr[:, 0]
        traj_tvd = traj_arr[:, 1]
        traj_ns = traj_arr[:, 2]
        traj_ew = traj_arr[:, 3]

        # Calculate azimuth along trajectory
        traj_az = np.zeros(len(traj_md))
        if len(traj_md) > 1:
            dew = np.diff(traj_ew)
            dns = np.diff(traj_ns)
            traj_az[1:] = np.arctan2(dew, dns)
            traj_az[0] = traj_az[1]

        # Generate MD points
        md_start = traj_md[0]
        md_end = traj_md[-1]

        md_points = np.arange(md_start + step, md_end, step)

        deltas = [0, 100, 200, 300]

        for md in md_points:
            # Interpolate target position
            target_tvd = np.interp(md, traj_md, traj_tvd)
            target_ns = np.interp(md, traj_md, traj_ns)
            target_ew = np.interp(md, traj_md, traj_ew)
            target_az = np.interp(md, traj_md, traj_az)

            target_x = x_surface + target_ew
            target_y = y_surface + target_ns
            target_z = target_tvd

            # Find neighbors
            L_well, L_md, L_dist, L_opp, R_well, R_md, R_dist, R_opp = \
                self._find_neighbors_single(target_x, target_y, target_z, target_az)

            has_L = L_well >= 0
            has_R = R_well >= 0

            if has_L and has_R:
                neighbor_config = "L+R"
            elif has_L:
                neighbor_config = "L_only"
            elif has_R:
                neighbor_config = "R_only"
            else:
                neighbor_config = "none"

            max_dist = max(L_dist if has_L else 0, R_dist if has_R else 0)

            # Calculate dips for each delta
            dips = {}
            errors = {}

            for delta in deltas:
                dip_L = np.nan
                dip_R = np.nan

                if has_L:
                    dip_L = self._get_dip_at_md(self.well_names[L_well], L_md, delta)
                    if L_opp and not np.isnan(dip_L):
                        dip_L = -dip_L

                if has_R:
                    dip_R = self._get_dip_at_md(self.well_names[R_well], R_md, delta)
                    if R_opp and not np.isnan(dip_R):
                        dip_R = -dip_R

                # Weighted average
                if has_L and has_R and not np.isnan(dip_L) and not np.isnan(dip_R):
                    w_L = 1.0 / max(L_dist, 1.0)
                    w_R = 1.0 / max(R_dist, 1.0)
                    dip = (dip_L * w_L + dip_R * w_R) / (w_L + w_R)
                elif has_L and not np.isnan(dip_L):
                    dip = dip_L
                elif has_R and not np.isnan(dip_R):
                    dip = dip_R
                else:
                    dip = np.nan

                dips[delta] = dip
                errors[delta] = self._get_error_from_config(neighbor_config, max_dist, delta)

            results.append(NeighborDipResult(
                md=float(md),
                neighbor_config=neighbor_config,
                dist_L=L_dist if has_L else -1,
                dist_R=R_dist if has_R else -1,
                dip_0ft=dips[0],
                error_0ft_p90=errors[0],
                dip_100ft=dips[100],
                error_100ft_p90=errors[100],
                dip_200ft=dips[200],
                error_200ft_p90=errors[200],
                dip_300ft=dips[300],
                error_300ft_p90=errors[300],
            ))

        logger.info(f"Processed {well_name}: {len(results)} points")
        return results

    def results_to_json(self, results: List[NeighborDipResult]) -> List[Dict]:
        """Convert results to JSON-serializable format."""
        return [
            {
                "md": r.md,
                "neighbor_config": r.neighbor_config,
                "dist_L": r.dist_L,
                "dist_R": r.dist_R,
                "dip_0ft": r.dip_0ft if not np.isnan(r.dip_0ft) else None,
                "error_0ft_p90": r.error_0ft_p90,
                "dip_100ft": r.dip_100ft if not np.isnan(r.dip_100ft) else None,
                "error_100ft_p90": r.error_100ft_p90,
                "dip_200ft": r.dip_200ft if not np.isnan(r.dip_200ft) else None,
                "error_200ft_p90": r.error_200ft_p90,
                "dip_300ft": r.dip_300ft if not np.isnan(r.dip_300ft) else None,
                "error_300ft_p90": r.error_300ft_p90,
            }
            for r in results
        ]


def main():
    """Test on Well162."""
    logging.basicConfig(level=logging.INFO)

    advisor = NeighborAdvisorProd(
        'dataset/gpu_ag_dataset.pt',
        'neighbor_accuracy_config.json'
    )

    # Load Well162 trajectory
    ds = torch.load('dataset/gpu_ag_dataset.pt', weights_only=False)
    well = ds['Well162~EGFDL']

    x_surface = well.get('x_surface', 0.0)
    y_surface = well.get('y_surface', 0.0)

    well_md = well['well_md'].numpy()
    well_tvd = well['well_tvd'].numpy()
    well_ns = well['well_ns'].numpy()
    well_ew = well['well_ew'].numpy()

    # Start from perch_md
    perch_md = well.get('perch_md', 0.0)
    mask = well_md > perch_md

    trajectory = list(zip(
        well_md[mask],
        well_tvd[mask],
        well_ns[mask],
        well_ew[mask]
    ))

    print(f"Processing Well162~EGFDL ({len(trajectory)} points in lateral)...")
    results = advisor.process_well('Well162~EGFDL', x_surface, y_surface, trajectory)

    # Print sample
    print(f"\nResults ({len(results)} output points):")
    print("-" * 100)
    print(f"{'MD':>7} {'Config':>6} {'distL':>7} {'distR':>7} | {'dip0':>6} {'err0':>5} | {'dip100':>6} {'err100':>5} | {'dip200':>6} {'err200':>5}")
    print("-" * 100)

    for r in results[::25]:  # Every 25th point
        dip0 = f"{r.dip_0ft:+6.2f}" if not np.isnan(r.dip_0ft) else "   nan"
        dip100 = f"{r.dip_100ft:+6.2f}" if not np.isnan(r.dip_100ft) else "   nan"
        dip200 = f"{r.dip_200ft:+6.2f}" if not np.isnan(r.dip_200ft) else "   nan"
        print(f"{r.md:7.0f} {r.neighbor_config:>6} {r.dist_L:7.1f} {r.dist_R:7.1f} | "
              f"{dip0} {r.error_0ft_p90:5.2f} | {dip100} {r.error_100ft_p90:5.2f} | {dip200} {r.error_200ft_p90:5.2f}")

    # Save to JSON
    json_results = advisor.results_to_json(results)
    with open('Well162_neighbor_dips.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nSaved to Well162_neighbor_dips.json")


if __name__ == '__main__':
    main()
