"""
TVT Variance metric.

For each TVT bin, compute variance of GR values.
Good interpretation = low variance (same TVT = same GR).
"""
import os
import numpy as np
import torch
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/gpu_ag_dataset.pt')

# Load dataset lazily
_ds = None

def get_dataset():
    global _ds
    if _ds is None:
        _ds = torch.load(DATASET_PATH, weights_only=False)
    return _ds


def compute_tvt_for_interpretation(
    well_name: str,
    interpretations: List[Dict],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute TVT values for each point based on interpretation.

    Returns: (tvt, gr, md) arrays or (None, None, None) if failed
    """
    ds = get_dataset()
    if well_name not in ds:
        return None, None, None

    well_data = ds[well_name]
    well_md = well_data['well_md'].numpy()
    well_tvd = well_data['well_tvd'].numpy()
    log_md = well_data['log_md'].numpy()
    log_gr = well_data['log_gr'].numpy()
    well_gr = np.interp(well_md, log_md, log_gr)

    tvt = np.full_like(well_tvd, np.nan)

    for interp in interpretations:
        md_start, md_end = interp['md_start'], interp['md_end']
        start_shift, end_shift = interp['start_shift'], interp['end_shift']

        mask = (well_md >= md_start) & (well_md <= md_end)
        if not mask.any():
            continue

        seg_md = well_md[mask]
        seg_tvd = well_tvd[mask]

        if md_end > md_start:
            ratio = (seg_md - md_start) / (md_end - md_start)
        else:
            ratio = np.zeros_like(seg_md)
        seg_shift = start_shift + ratio * (end_shift - start_shift)

        # TVT = TVD - shift
        tvt[mask] = seg_tvd - seg_shift

    # Remove NaN
    valid = ~np.isnan(tvt)
    if valid.sum() < 10:
        return None, None, None

    return tvt[valid], well_gr[valid], well_md[valid]


def compute_tvt_variance(
    well_name: str,
    interpretations: List[Dict],
    tvt_bin_size: float = 0.5
) -> Dict:
    """
    Compute TVT variance metric.

    Args:
        well_name: Well name
        interpretations: List of interpretation dicts with md_start, md_end, start_shift, end_shift
        tvt_bin_size: Size of TVT bins in meters

    Returns:
        Dict with:
        - mean_var: Average variance within TVT bins (lower = better)
        - weighted_var: Variance weighted by number of points per bin
        - n_overlaps: Number of TVT bins with >=2 points
        - n_points: Total number of points
        - score: 1/(1+weighted_var) for comparison (higher = better)
    """
    tvt, gr, md = compute_tvt_for_interpretation(well_name, interpretations)

    if tvt is None:
        return {'error': 'Failed to compute TVT'}

    # Bin by TVT
    tvt_min, tvt_max = tvt.min(), tvt.max()
    n_bins = int((tvt_max - tvt_min) / tvt_bin_size) + 1

    bin_indices = ((tvt - tvt_min) / tvt_bin_size).astype(int)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    # Collect GR values per bin
    bins = defaultdict(list)
    for i, bi in enumerate(bin_indices):
        bins[bi].append(gr[i])

    # Compute metrics
    variances = []
    weights = []
    n_overlaps = 0

    for bi, gr_vals in bins.items():
        if len(gr_vals) >= 2:
            n_overlaps += 1
            var = np.var(gr_vals)
            variances.append(var)
            weights.append(len(gr_vals))

    if not variances:
        return {
            'mean_var': None,
            'weighted_var': None,
            'n_overlaps': 0,
            'n_points': len(tvt),
            'score': None,
            'error': 'No overlaps found'
        }

    mean_var = np.mean(variances)
    weighted_var = np.average(variances, weights=weights)
    score = 1.0 / (1.0 + weighted_var)

    return {
        'mean_var': float(mean_var),
        'weighted_var': float(weighted_var),
        'n_overlaps': n_overlaps,
        'n_points': len(tvt),
        'score': float(score),
        'tvt_range': float(tvt_max - tvt_min),
    }
