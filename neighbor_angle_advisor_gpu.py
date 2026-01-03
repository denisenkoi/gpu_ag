"""
GPU Neighbor Angle Advisor (RND-817)

Fast parallel search for nearest neighbor wells using CUDA.
Processes all wells and all MDs in parallel.
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class NeighborResult:
    """Result of neighbor search"""
    well_name: str
    md: float
    distance_3d: float
    azimuth_diff_deg: float
    dip_angle_deg: float
    is_opposite: bool = False
    side: str = ""


class NeighborAngleAdvisorGPU:
    """
    GPU-accelerated neighbor search.

    Precomputes dip angles for all index points.
    Uses torch tensors on CUDA for parallel distance calculation.
    """

    def __init__(self, dataset_path: str, tvd_weight: float = 2.0, device: str = 'cuda'):
        self.tvd_weight = tvd_weight
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.dataset = torch.load(dataset_path, weights_only=False)
        self.well_names = list(self.dataset.keys())

        logger.info(f"Using device: {self.device}")

        # Build GPU index with precomputed dip angles
        self._build_gpu_index()

        logger.info(f"NeighborAngleAdvisorGPU initialized with {len(self.well_names)} wells")

    def _get_dip_angle_at_md(self, data: dict, md: float) -> Optional[float]:
        """Get dip angle from reference interpretation at given MD."""
        ref_mds = data['ref_segment_mds'].numpy()
        ref_start_shifts = data['ref_start_shifts'].numpy()
        ref_shifts = data['ref_shifts'].numpy()

        if len(ref_mds) == 0:
            return None

        for i in range(len(ref_mds)):
            start_md = ref_mds[i]
            end_md = ref_mds[i + 1] if i + 1 < len(ref_mds) else data['lateral_well_last_md']

            if start_md <= md <= end_md:
                start_shift = ref_start_shifts[i]
                end_shift = ref_shifts[i]
                dmd = end_md - start_md
                dshift = end_shift - start_shift

                if dmd > 0:
                    return np.degrees(np.arctan2(dshift, dmd))

        return None

    def _build_gpu_index(self):
        """Build GPU tensors for fast parallel search."""
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

            # Only lateral section (MD > perch_md) with precomputed dip
            for i in range(len(well_md)):
                if well_md[i] > perch_md:
                    dip = self._get_dip_angle_at_md(data, well_md[i])
                    if dip is not None:  # Only points with valid interpretation
                        all_points.append((
                            well_idx,
                            x_abs[i],
                            y_abs[i],
                            well_tvd[i],
                            well_md[i],
                            azimuth_rad[i],
                            dip
                        ))

        n = len(all_points)
        logger.info(f"Building GPU index with {n} points (with precomputed dip)")

        # Convert to GPU tensors
        self.idx_well = torch.tensor([p[0] for p in all_points], dtype=torch.int32, device=self.device)
        self.idx_x = torch.tensor([p[1] for p in all_points], dtype=torch.float32, device=self.device)
        self.idx_y = torch.tensor([p[2] for p in all_points], dtype=torch.float32, device=self.device)
        self.idx_z = torch.tensor([p[3] for p in all_points], dtype=torch.float32, device=self.device)
        self.idx_md = torch.tensor([p[4] for p in all_points], dtype=torch.float32, device=self.device)
        self.idx_azimuth = torch.tensor([p[5] for p in all_points], dtype=torch.float32, device=self.device)
        self.idx_dip = torch.tensor([p[6] for p in all_points], dtype=torch.float32, device=self.device)

        logger.info(f"GPU index built: {n} points, {self.idx_x.element_size() * n * 7 / 1024 / 1024:.1f} MB on {self.device}")

    def find_neighbors_batch(
        self,
        well_name: str,
        mds: torch.Tensor,  # [N] target MDs
        azimuth_threshold_deg: float = 15.0,
        max_neighbors: int = 5,
        include_opposite: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Find neighbors for multiple MDs in parallel.

        Args:
            well_name: Target well
            mds: [N] tensor of target MDs
            azimuth_threshold_deg: Threshold for parallel/opposite
            max_neighbors: K nearest to return

        Returns:
            Tuple of [N, K] tensors: (distances, dip_angles, well_indices, sides)
            sides: -1=left, +1=right
        """
        if well_name not in self.dataset:
            raise ValueError(f"Well {well_name} not in dataset")

        data = self.dataset[well_name]
        target_well_idx = self.well_names.index(well_name)

        # Get target well trajectory
        well_md_arr = data['well_md'].numpy()
        well_tvd = data['well_tvd'].numpy()
        well_ns = data['well_ns'].numpy()
        well_ew = data['well_ew'].numpy()
        x_surface = data.get('x_surface', 0.0)
        y_surface = data.get('y_surface', 0.0)

        # Interpolate target coordinates for each MD
        mds_np = mds.cpu().numpy()
        target_tvd = np.interp(mds_np, well_md_arr, well_tvd)
        target_ns = np.interp(mds_np, well_md_arr, well_ns)
        target_ew = np.interp(mds_np, well_md_arr, well_ew)

        target_x = torch.tensor(x_surface + target_ew, dtype=torch.float32, device=self.device)
        target_y = torch.tensor(y_surface + target_ns, dtype=torch.float32, device=self.device)
        target_z = torch.tensor(target_tvd, dtype=torch.float32, device=self.device)

        # Target azimuth (approximate)
        target_azimuth = torch.zeros(len(mds_np), dtype=torch.float32, device=self.device)
        for i, md in enumerate(mds_np):
            idx = np.searchsorted(well_md_arr, md)
            if 0 < idx < len(well_md_arr):
                dew = well_ew[idx] - well_ew[idx - 1]
                dns = well_ns[idx] - well_ns[idx - 1]
                target_azimuth[i] = np.arctan2(dew, dns)

        N = len(mds)
        M = len(self.idx_x)

        # Exclude same well
        other_well_mask = self.idx_well != target_well_idx  # [M]

        # Calculate distances: [N, M]
        dx = self.idx_x.unsqueeze(0) - target_x.unsqueeze(1)  # [N, M]
        dy = self.idx_y.unsqueeze(0) - target_y.unsqueeze(1)
        dz = self.idx_z.unsqueeze(0) - target_z.unsqueeze(1)

        distances = torch.sqrt(dx**2 + dy**2 + (self.tvd_weight * dz)**2)

        # Azimuth diff: [N, M]
        az_diff = torch.abs(target_azimuth.unsqueeze(1) - self.idx_azimuth.unsqueeze(0))
        az_diff = torch.minimum(az_diff, 2 * np.pi - az_diff)
        az_diff_deg = torch.rad2deg(az_diff)

        # Parallel or opposite mask
        threshold_rad = np.radians(azimuth_threshold_deg)
        parallel_mask = az_diff_deg <= azimuth_threshold_deg
        opposite_mask = az_diff_deg >= (180.0 - azimuth_threshold_deg)

        if include_opposite:
            az_mask = parallel_mask | opposite_mask
        else:
            az_mask = parallel_mask

        # Combined mask
        valid_mask = az_mask & other_well_mask.unsqueeze(0)  # [N, M]

        # Set invalid distances to inf
        distances = torch.where(valid_mask, distances, torch.tensor(float('inf'), device=self.device))

        # Top-K nearest
        top_distances, top_indices = torch.topk(distances, k=min(max_neighbors, M), dim=1, largest=False)

        # Gather results
        top_dips = self.idx_dip[top_indices]  # [N, K]
        top_wells = self.idx_well[top_indices]  # [N, K]

        # Flip dip for opposite direction
        is_opposite = az_diff_deg.gather(1, top_indices) >= (180.0 - azimuth_threshold_deg)
        top_dips = torch.where(is_opposite, -top_dips, top_dips)

        # Calculate side (left/right)
        dir_x = torch.sin(target_azimuth).unsqueeze(1)  # [N, 1]
        dir_y = torch.cos(target_azimuth).unsqueeze(1)
        top_dy = dy.gather(1, top_indices)
        top_dx = dx.gather(1, top_indices)
        cross = dir_x * top_dy - dir_y * top_dx
        sides = torch.sign(cross)  # -1=right, +1=left

        return top_distances, top_dips, top_wells, sides


def benchmark():
    """Benchmark GPU vs CPU"""
    import time

    print("Loading dataset...")
    advisor = NeighborAngleAdvisorGPU('dataset/gpu_ag_dataset.pt')

    ds = torch.load('dataset/gpu_ag_dataset.pt', weights_only=False)
    well = ds['Well162~EGFDL']
    perch_md = well['perch_md']
    last_md = well['lateral_well_last_md']

    # Create MD grid
    mds = torch.arange(perch_md + 10, last_md, 10.0, device=advisor.device)
    print(f"Testing {len(mds)} MDs on {advisor.device}")

    # Warmup
    _ = advisor.find_neighbors_batch('Well162~EGFDL', mds[:10])

    # Benchmark
    torch.cuda.synchronize() if advisor.device.type == 'cuda' else None
    start = time.time()

    distances, dips, wells, sides = advisor.find_neighbors_batch('Well162~EGFDL', mds)

    torch.cuda.synchronize() if advisor.device.type == 'cuda' else None
    elapsed = time.time() - start

    print(f"Batch search: {len(mds)} points in {elapsed*1000:.1f}ms ({elapsed/len(mds)*1000:.3f}ms/point)")
    print(f"Speedup vs CPU (29.4ms/point): {29.4 / (elapsed/len(mds)*1000):.0f}x")

    # Sample results
    print("\nSample results (every 50 MDs):")
    for i in range(0, len(mds), 50):
        md = mds[i].item()
        dist = distances[i, 0].item()
        dip = dips[i, 0].item()
        well_idx = wells[i, 0].item()
        side = "L" if sides[i, 0].item() > 0 else "R"
        well_name = advisor.well_names[well_idx]
        print(f"  MD={md:6.0f}: {side} {well_name:30} dist={dist:6.1f} dip={dip:+5.2f}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    benchmark()
