"""
Self-Correlation metrics for interpretation quality assessment.

Idea: Project well trajectory onto TVT axis.
Points with same TVT should have similar GR (same geological layer).

Problem: Can find "false positives" - few points matching perfectly
vs many points matching poorly.

Metrics:
1. tvt_variance - bin-based variance (coarse)
2. tvt_pairwise - pairwise GR difference
3. tvt_rmse - RMSE to nearest neighbor by TVT
4. tvt_continuity - smoothness of GR(TVT) curve
5. 2d_density - uniformity in (TVT, GR) space
"""

from .tvt_variance import compute_tvt_variance
from .tvt_pairwise import compute_tvt_pairwise
from .tvt_rmse import compute_tvt_rmse_nearest, compute_tvt_continuity, compute_2d_density
from .compare import compare_algorithms

__all__ = [
    'compute_tvt_variance',
    'compute_tvt_pairwise',
    'compute_tvt_rmse_nearest',
    'compute_tvt_continuity',
    'compute_2d_density',
    'compare_algorithms',
]
