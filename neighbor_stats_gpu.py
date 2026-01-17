"""
GPU-accelerated neighbor statistics collector (RND-817)

Collects angle difference statistics between neighbors and reference.
Supports averaged angles over ±delta windows.
"""

import os
import torch
import numpy as np
from typing import Dict, List, Tuple
import logging
import time

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/gpu_ag_dataset.pt')

logger = logging.getLogger(__name__)


class NeighborStatsGPU:
    """Collect neighbor angle statistics using GPU acceleration."""

    def __init__(self, dataset_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dataset = torch.load(dataset_path, weights_only=False)
        self.well_names = list(self.dataset.keys())
        logger.info(f"Loaded {len(self.well_names)} wells on {self.device}")

        # Build shift curves for all wells (for fast dip interpolation)
        self._build_shift_curves()
        self._build_spatial_index()

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

    def _build_spatial_index(self):
        """Build GPU tensors for spatial search."""
        all_points = []
        tvd_weight = 2.0

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

            # Only lateral section
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

    def get_dip_at_md(self, well_name: str, md: float, delta: float = 0, allow_asymmetric: bool = True) -> float:
        """
        Get dip angle at MD (or averaged over ±delta).

        If allow_asymmetric=True and delta extends beyond well bounds,
        shrink the window to available range (asymmetric averaging).
        """
        mds, shifts, perch, last = self.shift_curves[well_name]

        if delta == 0:
            # Point dip: derivative at md
            if md < mds[0] or md > mds[-1]:
                return np.nan
            # Find segment
            idx = np.searchsorted(mds, md) - 1
            idx = max(0, min(idx, len(mds) - 2))
            dmd = mds[idx + 1] - mds[idx]
            dshift = shifts[idx + 1] - shifts[idx]
            if dmd > 0:
                return np.degrees(np.arctan2(dshift, dmd))
            return np.nan
        else:
            # Averaged dip over [md-delta, md+delta]
            start, end = md - delta, md + delta

            if allow_asymmetric:
                # Shrink to available range
                start = max(start, mds[0])
                end = min(end, mds[-1])
                if end <= start:
                    return np.nan
            else:
                if start < mds[0] or end > mds[-1]:
                    return np.nan

            s1 = np.interp(start, mds, shifts)
            s2 = np.interp(end, mds, shifts)
            actual_delta = end - start
            if actual_delta > 0:
                return np.degrees(np.arctan2(s2 - s1, actual_delta))
            return np.nan

    def find_neighbors_batch(
        self,
        well_name: str,
        mds: np.ndarray,
        az_threshold_deg: float = 15.0,
        include_opposite: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Find L and R neighbors for multiple MDs.

        Returns: (left_wells, left_mds, left_dists, right_wells, right_mds, right_dists)
        Arrays of shape [N], -1 means no neighbor found.
        """
        data = self.dataset[well_name]
        target_well_idx = self.well_names.index(well_name)

        well_md_arr = data['well_md'].numpy()
        well_tvd = data['well_tvd'].numpy()
        well_ns = data['well_ns'].numpy()
        well_ew = data['well_ew'].numpy()
        x_surface = data.get('x_surface', 0.0)
        y_surface = data.get('y_surface', 0.0)

        # Interpolate target coords
        target_tvd = np.interp(mds, well_md_arr, well_tvd)
        target_ns = np.interp(mds, well_md_arr, well_ns)
        target_ew = np.interp(mds, well_md_arr, well_ew)

        target_x = torch.tensor(x_surface + target_ew, dtype=torch.float32, device=self.device)
        target_y = torch.tensor(y_surface + target_ns, dtype=torch.float32, device=self.device)
        target_z = torch.tensor(target_tvd, dtype=torch.float32, device=self.device)

        # Target azimuth
        target_az = np.zeros(len(mds))
        for i, md in enumerate(mds):
            idx = np.searchsorted(well_md_arr, md)
            if 0 < idx < len(well_md_arr):
                dew = well_ew[idx] - well_ew[idx - 1]
                dns = well_ns[idx] - well_ns[idx - 1]
                target_az[i] = np.arctan2(dew, dns)
        target_az_t = torch.tensor(target_az, dtype=torch.float32, device=self.device)

        N = len(mds)
        M = len(self.idx_x)

        # Exclude same well
        other_mask = self.idx_well != target_well_idx

        # Distances [N, M]
        dx = self.idx_x.unsqueeze(0) - target_x.unsqueeze(1)
        dy = self.idx_y.unsqueeze(0) - target_y.unsqueeze(1)
        dz = self.idx_z.unsqueeze(0) - target_z.unsqueeze(1)
        distances = torch.sqrt(dx**2 + dy**2 + (2.0 * dz)**2)

        # Azimuth filter
        az_diff = torch.abs(target_az_t.unsqueeze(1) - self.idx_az.unsqueeze(0))
        az_diff = torch.minimum(az_diff, 2 * np.pi - az_diff)
        az_diff_deg = torch.rad2deg(az_diff)

        parallel = az_diff_deg <= az_threshold_deg
        opposite = az_diff_deg >= (180.0 - az_threshold_deg)
        if include_opposite:
            az_ok = parallel | opposite
        else:
            az_ok = parallel

        valid = az_ok & other_mask.unsqueeze(0)

        # Left/Right: cross product
        dir_x = torch.sin(target_az_t).unsqueeze(1)
        dir_y = torch.cos(target_az_t).unsqueeze(1)
        cross = dir_x * dy - dir_y * dx
        is_left = cross > 0

        # For each target MD, find closest L and R
        INF = 1e9
        dist_L = torch.where(valid & is_left, distances, torch.tensor(INF, device=self.device))
        dist_R = torch.where(valid & ~is_left, distances, torch.tensor(INF, device=self.device))

        min_dist_L, min_idx_L = torch.min(dist_L, dim=1)
        min_dist_R, min_idx_R = torch.min(dist_R, dim=1)

        # Get well indices and MDs
        left_wells = self.idx_well[min_idx_L].cpu().numpy()
        left_mds = self.idx_md[min_idx_L].cpu().numpy()
        left_dists = min_dist_L.cpu().numpy()
        left_opp = (az_diff_deg.gather(1, min_idx_L.unsqueeze(1)).squeeze(1) >= (180 - az_threshold_deg)).cpu().numpy()

        right_wells = self.idx_well[min_idx_R].cpu().numpy()
        right_mds = self.idx_md[min_idx_R].cpu().numpy()
        right_dists = min_dist_R.cpu().numpy()
        right_opp = (az_diff_deg.gather(1, min_idx_R.unsqueeze(1)).squeeze(1) >= (180 - az_threshold_deg)).cpu().numpy()

        # Mark invalid (dist >= INF) as -1
        left_wells[left_dists >= INF] = -1
        right_wells[right_dists >= INF] = -1

        return left_wells, left_mds, left_dists, left_opp, right_wells, right_mds, right_dists, right_opp

    def collect_stats(self, step: float = 10.0, deltas: List[float] = [0, 15, 30, 50, 100, 200, 300]) -> Dict:
        """Collect statistics for all wells."""
        results = {d: {'diffs': [], 'dists': []} for d in deltas}
        coverage = {'both': 0, 'left_only': 0, 'right_only': 0, 'none': 0}
        total_points = 0

        start_time = time.time()

        for wi, well_name in enumerate(self.well_names):
            if wi % 20 == 0:
                logger.info(f"Processing {wi+1}/{len(self.well_names)}...")

            data = self.dataset[well_name]
            perch_md = data.get('perch_md', 0.0)
            last_md = data['lateral_well_last_md']

            if last_md <= perch_md + step:
                continue

            mds = np.arange(perch_md + step, last_md, step)
            if len(mds) == 0:
                continue

            # Find neighbors (GPU)
            L_wells, L_mds, L_dists, L_opp, R_wells, R_mds, R_dists, R_opp = self.find_neighbors_batch(well_name, mds)

            total_points += len(mds)

            # Coverage
            has_L = L_wells >= 0
            has_R = R_wells >= 0
            coverage['both'] += np.sum(has_L & has_R)
            coverage['left_only'] += np.sum(has_L & ~has_R)
            coverage['right_only'] += np.sum(~has_L & has_R)
            coverage['none'] += np.sum(~has_L & ~has_R)

            # For each delta, compute diffs
            for delta in deltas:
                for i, md in enumerate(mds):
                    ref_dip = self.get_dip_at_md(well_name, md, delta)
                    if np.isnan(ref_dip):
                        continue

                    # Left neighbor
                    if L_wells[i] >= 0:
                        n_well = self.well_names[L_wells[i]]
                        n_dip = self.get_dip_at_md(n_well, L_mds[i], delta)
                        if not np.isnan(n_dip):
                            if L_opp[i]:
                                n_dip = -n_dip
                            results[delta]['diffs'].append(n_dip - ref_dip)
                            results[delta]['dists'].append(L_dists[i])

                    # Right neighbor
                    if R_wells[i] >= 0:
                        n_well = self.well_names[R_wells[i]]
                        n_dip = self.get_dip_at_md(n_well, R_mds[i], delta)
                        if not np.isnan(n_dip):
                            if R_opp[i]:
                                n_dip = -n_dip
                            results[delta]['diffs'].append(n_dip - ref_dip)
                            results[delta]['dists'].append(R_dists[i])

        elapsed = time.time() - start_time
        logger.info(f"Processed {total_points} points in {elapsed:.1f}s")

        return results, coverage, total_points


    def collect_stats_v2(self, step: float = 10.0, deltas: List[float] = [0, 50, 100, 200]) -> Dict:
        """
        Collect extended statistics with:
        - Asymmetric averaging (shrink to available range)
        - Weighted average for L+R cases
        - Breakdown by distance buckets and neighbor availability
        """
        # Distance buckets
        dist_buckets = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, float('inf'))]

        # Results structure: [delta][neighbor_type][dist_bucket] -> list of (diff, max_dist)
        # neighbor_type: 'L+R_weighted', 'L+R_left', 'L+R_right', 'L_only', 'R_only'
        results = {}
        for delta in deltas:
            results[delta] = {
                'L+R_weighted': [],  # Weighted average of L and R
                'L+R_left': [],      # Left neighbor when both available
                'L+R_right': [],     # Right neighbor when both available
                'L_only': [],        # Only left available
                'R_only': [],        # Only right available
            }

        total_points = 0
        coverage = {'both': 0, 'left_only': 0, 'right_only': 0, 'none': 0}

        start_time = time.time()

        for wi, well_name in enumerate(self.well_names):
            if wi % 20 == 0:
                logger.info(f"Processing {wi+1}/{len(self.well_names)}...")

            data = self.dataset[well_name]
            perch_md = data.get('perch_md', 0.0)
            last_md = data['lateral_well_last_md']

            if last_md <= perch_md + step:
                continue

            mds = np.arange(perch_md + step, last_md, step)
            if len(mds) == 0:
                continue

            # Find neighbors (GPU)
            L_wells, L_mds, L_dists, L_opp, R_wells, R_mds, R_dists, R_opp = self.find_neighbors_batch(well_name, mds)

            total_points += len(mds)

            has_L = L_wells >= 0
            has_R = R_wells >= 0
            coverage['both'] += np.sum(has_L & has_R)
            coverage['left_only'] += np.sum(has_L & ~has_R)
            coverage['right_only'] += np.sum(~has_L & has_R)
            coverage['none'] += np.sum(~has_L & ~has_R)

            for delta in deltas:
                for i, md in enumerate(mds):
                    ref_dip = self.get_dip_at_md(well_name, md, delta, allow_asymmetric=True)
                    if np.isnan(ref_dip):
                        continue

                    has_left = L_wells[i] >= 0
                    has_right = R_wells[i] >= 0

                    left_dip = np.nan
                    right_dip = np.nan

                    if has_left:
                        n_well = self.well_names[L_wells[i]]
                        left_dip = self.get_dip_at_md(n_well, L_mds[i], delta, allow_asymmetric=True)
                        if L_opp[i] and not np.isnan(left_dip):
                            left_dip = -left_dip

                    if has_right:
                        n_well = self.well_names[R_wells[i]]
                        right_dip = self.get_dip_at_md(n_well, R_mds[i], delta, allow_asymmetric=True)
                        if R_opp[i] and not np.isnan(right_dip):
                            right_dip = -right_dip

                    if has_left and has_right and not np.isnan(left_dip) and not np.isnan(right_dip):
                        # Both neighbors: weighted average (avoid div by zero)
                        d_L = max(L_dists[i], 1.0)
                        d_R = max(R_dists[i], 1.0)
                        w_L = 1.0 / d_L
                        w_R = 1.0 / d_R
                        weighted_dip = (left_dip * w_L + right_dip * w_R) / (w_L + w_R)
                        max_dist = max(L_dists[i], R_dists[i])

                        results[delta]['L+R_weighted'].append((weighted_dip - ref_dip, max_dist, L_dists[i], R_dists[i]))
                        results[delta]['L+R_left'].append((left_dip - ref_dip, L_dists[i]))
                        results[delta]['L+R_right'].append((right_dip - ref_dip, R_dists[i]))

                    elif has_left and not np.isnan(left_dip):
                        results[delta]['L_only'].append((left_dip - ref_dip, L_dists[i]))

                    elif has_right and not np.isnan(right_dip):
                        results[delta]['R_only'].append((right_dip - ref_dip, R_dists[i]))

        elapsed = time.time() - start_time
        logger.info(f"Processed {total_points} points in {elapsed:.1f}s")

        return results, coverage, total_points, dist_buckets


def print_stats_matrix(results, dist_buckets):
    """Print accuracy matrix by delta, neighbor type, and distance."""
    print()
    print("=" * 100)
    print("ACCURACY MATRIX: |diff| percentiles by delta, neighbor config, and distance")
    print("=" * 100)

    for delta in sorted(results.keys()):
        print(f"\n--- Delta = ±{delta}ft ---")
        print(f"{'Type':<15} {'Dist':<12} {'n':>6} {'mean':>7} {'std':>6} {'p50':>5} {'p90':>5} {'p95':>5}")
        print("-" * 70)

        for ntype in ['L+R_weighted', 'L+R_left', 'L+R_right', 'L_only', 'R_only']:
            data = results[delta][ntype]
            if len(data) == 0:
                continue

            # All distances
            if ntype == 'L+R_weighted':
                diffs = np.array([d[0] for d in data])
                dists = np.array([d[1] for d in data])  # max_dist
            else:
                diffs = np.array([d[0] for d in data])
                dists = np.array([d[1] for d in data])

            abs_d = np.abs(diffs)
            print(f"{ntype:<15} {'all':<12} {len(diffs):>6} {np.mean(diffs):>+7.3f} {np.std(diffs):>6.3f} "
                  f"{np.percentile(abs_d, 50):>5.2f} {np.percentile(abs_d, 90):>5.2f} {np.percentile(abs_d, 95):>5.2f}")

            # By distance bucket
            for lo, hi in dist_buckets:
                mask = (dists >= lo) & (dists < hi)
                if np.sum(mask) < 10:
                    continue
                bucket_diffs = diffs[mask]
                abs_bd = np.abs(bucket_diffs)
                hi_str = 'inf' if hi == float('inf') else f'{hi:.0f}'
                print(f"{'':15} {f'{lo:.0f}-{hi_str}':<12} {len(bucket_diffs):>6} {np.mean(bucket_diffs):>+7.3f} "
                      f"{np.std(bucket_diffs):>6.3f} {np.percentile(abs_bd, 50):>5.2f} "
                      f"{np.percentile(abs_bd, 90):>5.2f} {np.percentile(abs_bd, 95):>5.2f}")


def main():
    logging.basicConfig(level=logging.INFO)

    stats = NeighborStatsGPU(DATASET_PATH)

    print("Collecting extended statistics for all 100 wells...")
    results, coverage, total, dist_buckets = stats.collect_stats_v2()

    print()
    print("=" * 80)
    print("COVERAGE (no distance limit):")
    print("=" * 80)
    print(f"  Total points: {total}")
    print(f"  Both L+R:   {coverage['both']:5} ({100*coverage['both']/total:.1f}%)")
    print(f"  Left only:  {coverage['left_only']:5} ({100*coverage['left_only']/total:.1f}%)")
    print(f"  Right only: {coverage['right_only']:5} ({100*coverage['right_only']/total:.1f}%)")
    print(f"  None:       {coverage['none']:5} ({100*coverage['none']/total:.1f}%)")

    print_stats_matrix(results, dist_buckets)


if __name__ == '__main__':
    main()
