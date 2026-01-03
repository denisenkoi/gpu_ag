"""
Neighbor Calibration Module (RND-817)

Runs once per field to collect statistics and generate accuracy config.
Output: neighbor_accuracy_config.json for use in prod module.
"""

import torch
import numpy as np
import json
import time
from typing import Dict, List, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class NeighborCalibrator:
    """
    Calibrates neighbor accuracy for a field.

    Analyzes all wells in dataset, collects statistics,
    and generates JSON config for production use.
    """

    def __init__(self, dataset_path: str, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dataset_path = dataset_path
        self.dataset = torch.load(dataset_path, weights_only=False)
        self.well_names = list(self.dataset.keys())

        logger.info(f"Loaded {len(self.well_names)} wells on {self.device}")

        self._build_shift_curves()
        self._build_spatial_index()

    def _build_shift_curves(self):
        """Precompute shift curves for dip calculation."""
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

            azimuth_rad = np.zeros(len(well_md))
            if len(well_md) > 1:
                dew = np.diff(well_ew)
                dns = np.diff(well_ns)
                azimuth_rad[1:] = np.arctan2(dew, dns)
                azimuth_rad[0] = azimuth_rad[1] if len(azimuth_rad) > 1 else 0

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

    def _get_dip_at_md(self, well_name: str, md: float, delta: float) -> float:
        """Get dip angle with asymmetric averaging."""
        mds, shifts, perch, last = self.shift_curves[well_name]

        if delta == 0:
            if md < mds[0] or md > mds[-1]:
                return np.nan
            idx = np.searchsorted(mds, md) - 1
            idx = max(0, min(idx, len(mds) - 2))
            dmd = mds[idx + 1] - mds[idx]
            dshift = shifts[idx + 1] - shifts[idx]
            return np.degrees(np.arctan2(dshift, dmd)) if dmd > 0 else np.nan
        else:
            start = max(md - delta, mds[0])
            end = min(md + delta, mds[-1])
            if end <= start:
                return np.nan
            s1 = np.interp(start, mds, shifts)
            s2 = np.interp(end, mds, shifts)
            return np.degrees(np.arctan2(s2 - s1, end - start))

    def _find_neighbors_batch(
        self,
        well_name: str,
        mds: np.ndarray,
        az_threshold_deg: float = 15.0
    ) -> Tuple[np.ndarray, ...]:
        """Find L and R neighbors for multiple MDs."""
        data = self.dataset[well_name]
        target_well_idx = self.well_names.index(well_name)

        well_md_arr = data['well_md'].numpy()
        well_tvd = data['well_tvd'].numpy()
        well_ns = data['well_ns'].numpy()
        well_ew = data['well_ew'].numpy()
        x_surface = data.get('x_surface', 0.0)
        y_surface = data.get('y_surface', 0.0)

        target_tvd = np.interp(mds, well_md_arr, well_tvd)
        target_ns = np.interp(mds, well_md_arr, well_ns)
        target_ew = np.interp(mds, well_md_arr, well_ew)

        target_x = torch.tensor(x_surface + target_ew, dtype=torch.float32, device=self.device)
        target_y = torch.tensor(y_surface + target_ns, dtype=torch.float32, device=self.device)
        target_z = torch.tensor(target_tvd, dtype=torch.float32, device=self.device)

        target_az = np.zeros(len(mds))
        for i, md in enumerate(mds):
            idx = np.searchsorted(well_md_arr, md)
            if 0 < idx < len(well_md_arr):
                dew = well_ew[idx] - well_ew[idx - 1]
                dns = well_ns[idx] - well_ns[idx - 1]
                target_az[i] = np.arctan2(dew, dns)
        target_az_t = torch.tensor(target_az, dtype=torch.float32, device=self.device)

        other_mask = self.idx_well != target_well_idx

        dx = self.idx_x.unsqueeze(0) - target_x.unsqueeze(1)
        dy = self.idx_y.unsqueeze(0) - target_y.unsqueeze(1)
        dz = self.idx_z.unsqueeze(0) - target_z.unsqueeze(1)
        distances = torch.sqrt(dx**2 + dy**2 + (2.0 * dz)**2)

        az_diff = torch.abs(target_az_t.unsqueeze(1) - self.idx_az.unsqueeze(0))
        az_diff = torch.minimum(az_diff, 2 * np.pi - az_diff)
        az_diff_deg = torch.rad2deg(az_diff)

        parallel = az_diff_deg <= az_threshold_deg
        opposite = az_diff_deg >= (180.0 - az_threshold_deg)
        az_ok = parallel | opposite

        valid = az_ok & other_mask.unsqueeze(0)

        dir_x = torch.sin(target_az_t).unsqueeze(1)
        dir_y = torch.cos(target_az_t).unsqueeze(1)
        cross = dir_x * dy - dir_y * dx
        is_left = cross > 0

        INF = 1e9
        dist_L = torch.where(valid & is_left, distances, torch.tensor(INF, device=self.device))
        dist_R = torch.where(valid & ~is_left, distances, torch.tensor(INF, device=self.device))

        min_dist_L, min_idx_L = torch.min(dist_L, dim=1)
        min_dist_R, min_idx_R = torch.min(dist_R, dim=1)

        L_wells = self.idx_well[min_idx_L].cpu().numpy()
        L_mds = self.idx_md[min_idx_L].cpu().numpy()
        L_dists = min_dist_L.cpu().numpy()
        L_opp = (az_diff_deg.gather(1, min_idx_L.unsqueeze(1)).squeeze(1) >= (180 - az_threshold_deg)).cpu().numpy()

        R_wells = self.idx_well[min_idx_R].cpu().numpy()
        R_mds = self.idx_md[min_idx_R].cpu().numpy()
        R_dists = min_dist_R.cpu().numpy()
        R_opp = (az_diff_deg.gather(1, min_idx_R.unsqueeze(1)).squeeze(1) >= (180 - az_threshold_deg)).cpu().numpy()

        L_wells[L_dists >= INF] = -1
        R_wells[R_dists >= INF] = -1

        return L_wells, L_mds, L_dists, L_opp, R_wells, R_mds, R_dists, R_opp

    def calibrate(
        self,
        step: float = 10.0,
        deltas: List[int] = [0, 100, 200, 300],
        field_name: str = "Unknown"
    ) -> Dict:
        """
        Run calibration and return config dict.

        Args:
            step: MD step for analysis
            deltas: List of averaging deltas to analyze
            field_name: Name for config

        Returns:
            Config dict ready to save as JSON
        """
        dist_buckets = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, float('inf'))]

        results = {}
        for delta in deltas:
            results[delta] = {
                'L+R_weighted': [],
                'L_only': [],
                'R_only': [],
            }

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

            L_wells, L_mds, L_dists, L_opp, R_wells, R_mds, R_dists, R_opp = \
                self._find_neighbors_batch(well_name, mds)

            total_points += len(mds)

            has_L = L_wells >= 0
            has_R = R_wells >= 0
            coverage['both'] += int(np.sum(has_L & has_R))
            coverage['left_only'] += int(np.sum(has_L & ~has_R))
            coverage['right_only'] += int(np.sum(~has_L & has_R))
            coverage['none'] += int(np.sum(~has_L & ~has_R))

            for delta in deltas:
                for i, md in enumerate(mds):
                    ref_dip = self._get_dip_at_md(well_name, md, delta)
                    if np.isnan(ref_dip):
                        continue

                    left_dip = np.nan
                    right_dip = np.nan

                    if L_wells[i] >= 0:
                        left_dip = self._get_dip_at_md(self.well_names[L_wells[i]], L_mds[i], delta)
                        if L_opp[i] and not np.isnan(left_dip):
                            left_dip = -left_dip

                    if R_wells[i] >= 0:
                        right_dip = self._get_dip_at_md(self.well_names[R_wells[i]], R_mds[i], delta)
                        if R_opp[i] and not np.isnan(right_dip):
                            right_dip = -right_dip

                    if has_L[i] and has_R[i] and not np.isnan(left_dip) and not np.isnan(right_dip):
                        d_L = max(L_dists[i], 1.0)
                        d_R = max(R_dists[i], 1.0)
                        w_L = 1.0 / d_L
                        w_R = 1.0 / d_R
                        weighted_dip = (left_dip * w_L + right_dip * w_R) / (w_L + w_R)
                        max_dist = max(L_dists[i], R_dists[i])
                        results[delta]['L+R_weighted'].append((weighted_dip - ref_dip, max_dist))

                    elif has_L[i] and not np.isnan(left_dip):
                        results[delta]['L_only'].append((left_dip - ref_dip, L_dists[i]))

                    elif has_R[i] and not np.isnan(right_dip):
                        results[delta]['R_only'].append((right_dip - ref_dip, R_dists[i]))

        elapsed = time.time() - start_time
        logger.info(f"Calibration complete: {total_points} points in {elapsed:.1f}s")

        # Build config
        config = {
            "field_name": field_name,
            "dataset": self.dataset_path.split('/')[-1],
            "wells_count": len(self.well_names),
            "points_analyzed": total_points,
            "date_generated": datetime.now().strftime("%Y-%m-%d"),

            "coverage": {
                "both_L_R": round(coverage['both'] / total_points, 3) if total_points > 0 else 0,
                "left_only": round(coverage['left_only'] / total_points, 3) if total_points > 0 else 0,
                "right_only": round(coverage['right_only'] / total_points, 3) if total_points > 0 else 0,
                "none": round(coverage['none'] / total_points, 3) if total_points > 0 else 0
            },

            "recommended_settings": {
                "delta_ft": 100,
                "azimuth_threshold_deg": 15,
                "include_opposite": True,
                "tvd_weight": 2.0
            },

            "accuracy_matrix": {},
            "delta_comparison": {}
        }

        # Build accuracy matrix (using delta=100 as reference)
        for ntype in ['L+R_weighted', 'L_only', 'R_only']:
            if 100 in results and len(results[100][ntype]) > 0:
                data = results[100][ntype]
                diffs = np.array([d[0] for d in data])
                dists = np.array([d[1] for d in data])
                abs_d = np.abs(diffs)

                matrix_entry = {
                    "by_distance_ft": {},
                    "all": {
                        "n": len(diffs),
                        "p50": round(float(np.percentile(abs_d, 50)), 2),
                        "p90": round(float(np.percentile(abs_d, 90)), 2),
                        "p95": round(float(np.percentile(abs_d, 95)), 2)
                    }
                }

                for lo, hi in dist_buckets:
                    mask = (dists >= lo) & (dists < hi)
                    if np.sum(mask) >= 10:
                        bucket_diffs = np.abs(diffs[mask])
                        hi_str = "inf" if hi == float('inf') else str(int(hi))
                        matrix_entry["by_distance_ft"][f"{int(lo)}-{hi_str}"] = {
                            "n": int(np.sum(mask)),
                            "p50": round(float(np.percentile(bucket_diffs, 50)), 2),
                            "p90": round(float(np.percentile(bucket_diffs, 90)), 2),
                            "p95": round(float(np.percentile(bucket_diffs, 95)), 2)
                        }

                config["accuracy_matrix"][ntype] = matrix_entry

        # Build delta comparison
        for delta in deltas:
            all_diffs = []
            for ntype in ['L+R_weighted', 'L_only', 'R_only']:
                all_diffs.extend([d[0] for d in results[delta][ntype]])

            if len(all_diffs) > 0:
                abs_d = np.abs(all_diffs)
                config["delta_comparison"][str(delta)] = {
                    "p50": round(float(np.percentile(abs_d, 50)), 2),
                    "p90": round(float(np.percentile(abs_d, 90)), 2),
                    "p95": round(float(np.percentile(abs_d, 95)), 2)
                }

        return config

    def save_config(self, config: Dict, output_path: str):
        """Save config to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved config to {output_path}")


def main():
    """Run calibration for EGFDL field."""
    logging.basicConfig(level=logging.INFO)

    calibrator = NeighborCalibrator('dataset/gpu_ag_dataset.pt')

    config = calibrator.calibrate(
        step=10.0,
        deltas=[0, 100, 200, 300],
        field_name="EGFDL"
    )

    calibrator.save_config(config, 'neighbor_accuracy_config.json')

    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Field: {config['field_name']}")
    print(f"Wells: {config['wells_count']}")
    print(f"Points: {config['points_analyzed']}")
    print()
    print("Coverage:")
    for k, v in config['coverage'].items():
        print(f"  {k}: {v*100:.1f}%")
    print()
    print("Delta comparison (p90):")
    for d, v in config['delta_comparison'].items():
        print(f"  ±{d}ft: {v['p90']}°")


if __name__ == '__main__':
    main()
