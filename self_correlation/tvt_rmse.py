"""
TVT RMSE-based metrics.

More precise than bin-based variance.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from .tvt_variance import compute_tvt_for_interpretation


def compute_tvt_rmse_nearest(
    well_name: str,
    interpretations: List[Dict],
    tvt_max_diff: float = 0.5,
    md_min_diff: float = 30.0,
) -> Dict:
    """
    For each point, find nearest point by TVT (but far by MD).
    Compute RMSE of GR differences.

    Args:
        well_name: Well name
        interpretations: List of interpretation dicts
        tvt_max_diff: Max TVT difference to consider as "same layer"
        md_min_diff: Min MD distance (to avoid adjacent points)

    Returns:
        Dict with rmse, n_pairs, etc.
    """
    tvt, gr, md = compute_tvt_for_interpretation(well_name, interpretations)

    if tvt is None:
        return {'error': 'Failed to compute TVT'}

    n = len(tvt)

    # For each point, find nearest by TVT (but far by MD)
    gr_diffs_sq = []

    for i in range(n):
        best_diff = None
        best_gr_diff = None

        for j in range(n):
            if i == j:
                continue

            md_diff = abs(md[i] - md[j])
            if md_diff < md_min_diff:
                continue

            tvt_diff = abs(tvt[i] - tvt[j])
            if tvt_diff > tvt_max_diff:
                continue

            # This is a valid pair - check if best
            if best_diff is None or tvt_diff < best_diff:
                best_diff = tvt_diff
                best_gr_diff = abs(gr[i] - gr[j])

        if best_gr_diff is not None:
            gr_diffs_sq.append(best_gr_diff ** 2)

    if len(gr_diffs_sq) < 10:
        return {
            'rmse': None,
            'n_matched': len(gr_diffs_sq),
            'coverage': len(gr_diffs_sq) / n if n > 0 else 0,
            'error': 'Not enough matches'
        }

    rmse = np.sqrt(np.mean(gr_diffs_sq))

    return {
        'rmse': float(rmse),
        'n_matched': len(gr_diffs_sq),
        'coverage': len(gr_diffs_sq) / n,
        'mean_diff': float(np.sqrt(np.mean(gr_diffs_sq))),
    }


def compute_tvt_continuity(
    well_name: str,
    interpretations: List[Dict],
    tvt_step: float = 0.1,
) -> Dict:
    """
    Build GR(TVT) curve and measure its smoothness.

    For fine TVT grid, interpolate GR values from nearby points.
    Good interpretation = smooth curve, bad = noisy.

    Args:
        well_name: Well name
        interpretations: List of interpretation dicts
        tvt_step: Step for TVT grid

    Returns:
        Dict with smoothness metrics
    """
    tvt, gr, md = compute_tvt_for_interpretation(well_name, interpretations)

    if tvt is None:
        return {'error': 'Failed to compute TVT'}

    # Sort by TVT
    sort_idx = np.argsort(tvt)
    tvt_sorted = tvt[sort_idx]
    gr_sorted = gr[sort_idx]
    md_sorted = md[sort_idx]

    # Create fine TVT grid
    tvt_min, tvt_max = tvt_sorted.min(), tvt_sorted.max()
    tvt_grid = np.arange(tvt_min, tvt_max, tvt_step)

    if len(tvt_grid) < 10:
        return {'error': 'TVT range too small'}

    # For each grid point, collect all GR values within ±step
    gr_at_grid = []
    gr_std_at_grid = []

    for t in tvt_grid:
        mask = (tvt_sorted >= t - tvt_step) & (tvt_sorted <= t + tvt_step)
        if mask.sum() >= 1:
            gr_at_grid.append(np.mean(gr_sorted[mask]))
            if mask.sum() >= 2:
                gr_std_at_grid.append(np.std(gr_sorted[mask]))

    if len(gr_at_grid) < 10:
        return {'error': 'Not enough grid points'}

    gr_curve = np.array(gr_at_grid)

    # Smoothness = variance of first derivative
    d_gr = np.diff(gr_curve)
    roughness = np.std(d_gr)

    # Also compute mean std at each grid point (measures overlap consistency)
    mean_std = np.mean(gr_std_at_grid) if gr_std_at_grid else None

    return {
        'roughness': float(roughness),
        'mean_local_std': float(mean_std) if mean_std else None,
        'n_grid_points': len(gr_curve),
        'tvt_range': float(tvt_max - tvt_min),
    }


def compute_2d_density(
    well_name: str,
    interpretations: List[Dict],
    n_neighbors: int = 5,
) -> Dict:
    """
    View (TVT, GR) as 2D space.
    Good interpretation = points form a smooth manifold.
    Measure local density consistency.

    Args:
        well_name: Well name
        interpretations: List of interpretation dicts
        n_neighbors: Number of neighbors to consider

    Returns:
        Dict with density metrics
    """
    tvt, gr, md = compute_tvt_for_interpretation(well_name, interpretations)

    if tvt is None:
        return {'error': 'Failed to compute TVT'}

    # Normalize to same scale
    tvt_norm = (tvt - tvt.mean()) / (tvt.std() + 1e-10)
    gr_norm = (gr - gr.mean()) / (gr.std() + 1e-10)

    n = len(tvt)

    # For each point, find k nearest neighbors in (TVT, GR) space
    # Compute mean distance to them
    local_densities = []

    for i in range(min(n, 500)):  # Sample for speed
        # Compute distances to all other points
        dists = np.sqrt((tvt_norm - tvt_norm[i])**2 + (gr_norm - gr_norm[i])**2)
        dists[i] = np.inf  # Exclude self

        # Mean distance to k nearest
        k_nearest = np.partition(dists, n_neighbors)[:n_neighbors]
        mean_dist = np.mean(k_nearest)
        local_densities.append(mean_dist)

    local_densities = np.array(local_densities)

    # Good manifold = consistent local density
    # Bad manifold = some points isolated, some clustered
    density_cv = np.std(local_densities) / (np.mean(local_densities) + 1e-10)

    return {
        'mean_local_dist': float(np.mean(local_densities)),
        'density_cv': float(density_cv),  # Lower = more uniform = better manifold
        'n_points': n,
    }
