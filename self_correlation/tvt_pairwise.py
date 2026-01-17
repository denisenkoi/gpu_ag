"""
TVT Pairwise correlation metric.

For pairs of points with similar TVT but different MD,
compute correlation of their GR values.
"""
import numpy as np
from typing import List, Dict
from .tvt_variance import compute_tvt_for_interpretation


def compute_tvt_pairwise(
    well_name: str,
    interpretations: List[Dict],
    tvt_threshold: float = 0.3,
    md_min_distance: float = 50.0,
    max_samples: int = 500
) -> Dict:
    """
    Compute pairwise TVT correlation.

    For pairs of points with similar TVT (within threshold) but far apart in MD,
    compute mean absolute GR difference.

    Args:
        well_name: Well name
        interpretations: List of interpretation dicts
        tvt_threshold: Max TVT difference to consider as "same layer"
        md_min_distance: Min MD distance to avoid adjacent points
        max_samples: Max points to sample for performance

    Returns:
        Dict with:
        - mean_gr_diff: Average absolute GR difference (lower = better)
        - n_pairs: Number of pairs compared
        - score: 1/(1+mean_gr_diff) for comparison (higher = better)
    """
    tvt, gr, md = compute_tvt_for_interpretation(well_name, interpretations)

    if tvt is None:
        return {'error': 'Failed to compute TVT'}

    n = len(tvt)

    # Sample if too many points
    if n > max_samples:
        idx = np.random.choice(n, max_samples, replace=False)
        tvt = tvt[idx]
        gr = gr[idx]
        md = md[idx]
        n = max_samples

    # Find pairs with similar TVT but far in MD
    gr_diffs = []

    for i in range(n):
        for j in range(i + 1, n):
            tvt_diff = abs(tvt[i] - tvt[j])
            md_diff = abs(md[i] - md[j])

            if tvt_diff < tvt_threshold and md_diff > md_min_distance:
                gr_diffs.append(abs(gr[i] - gr[j]))

    if len(gr_diffs) < 5:
        return {
            'mean_gr_diff': None,
            'n_pairs': len(gr_diffs),
            'score': None,
            'error': 'Not enough pairs'
        }

    mean_gr_diff = np.mean(gr_diffs)
    score = 1.0 / (1.0 + mean_gr_diff)

    return {
        'mean_gr_diff': float(mean_gr_diff),
        'n_pairs': len(gr_diffs),
        'score': float(score),
        'std_gr_diff': float(np.std(gr_diffs)),
    }


def compute_tvt_consistency(
    well_name: str,
    interpretations: List[Dict],
    tvt_bin_size: float = 0.5,
    min_visits: int = 3
) -> Dict:
    """
    Alternative metric: TVT consistency score.

    For bins visited multiple times, compute how consistent the GR is.
    Weight by number of visits to favor more frequent overlaps.

    Args:
        well_name: Well name
        interpretations: List of interpretation dicts
        tvt_bin_size: Size of TVT bins
        min_visits: Minimum visits to a bin to count

    Returns:
        Dict with consistency metrics
    """
    from collections import defaultdict

    tvt, gr, md = compute_tvt_for_interpretation(well_name, interpretations)

    if tvt is None:
        return {'error': 'Failed to compute TVT'}

    tvt_min = tvt.min()
    bin_indices = ((tvt - tvt_min) / tvt_bin_size).astype(int)

    # Collect GR values per bin
    bins = defaultdict(list)
    for i, bi in enumerate(bin_indices):
        bins[bi].append((gr[i], md[i]))

    # Compute consistency for frequently-visited bins
    consistencies = []
    visit_counts = []

    for bi, values in bins.items():
        if len(values) >= min_visits:
            grs = [v[0] for v in values]
            cv = np.std(grs) / (np.mean(grs) + 1e-10)  # Coefficient of variation
            consistencies.append(cv)
            visit_counts.append(len(values))

    if not consistencies:
        return {
            'mean_cv': None,
            'n_bins': 0,
            'score': None,
            'error': 'No bins with enough visits'
        }

    mean_cv = np.average(consistencies, weights=visit_counts)
    score = 1.0 / (1.0 + mean_cv)

    return {
        'mean_cv': float(mean_cv),
        'n_bins': len(consistencies),
        'total_visits': sum(visit_counts),
        'score': float(score),
    }
