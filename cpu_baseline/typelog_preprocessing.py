"""
TypeLog Preprocessing Module

Prepares TypeLog + PseudoTypeLog data for optimizer with correct coordinate alignment.

CRITICAL: tvd_shift is ALWAYS applied to TypeLog for coordinate alignment.
DO NOT DISABLE THIS SHIFT UNLESS USER EXPLICITLY REQUESTS IT TWICE.
Without tvd_shift, TypeLog and PseudoTypeLog coordinates don't match,
resulting in garbage correlations (e.g., -0.25 instead of +0.74).

Configuration via env:
    USE_PSEUDO_TYPELOG=True  # True = stitch, False = TypeLog only

Author: Auto-generated
Date: 2026-01-07
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Tuple
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# Read params from env
USE_PSEUDO_TYPELOG = os.getenv('USE_PSEUDO_TYPELOG', 'True').lower() in ('true', '1', 'yes')
GR_SMOOTHING_WINDOW = int(os.getenv('GR_SMOOTHING_WINDOW', '0'))
GR_SMOOTHING_ORDER = int(os.getenv('GR_SMOOTHING_ORDER', '2'))


def apply_gr_smoothing(gr: np.ndarray, window: int = None, order: int = None) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to GR data.

    Args:
        gr: GR values array
        window: Window size (default from GR_SMOOTHING_WINDOW env)
        order: Polynomial order (default from GR_SMOOTHING_ORDER env)

    Returns:
        Smoothed GR array (or original if window < 3)
    """
    if window is None:
        window = GR_SMOOTHING_WINDOW
    if order is None:
        order = GR_SMOOTHING_ORDER

    if window < 3 or len(gr) < window:
        return gr

    # Window must be odd
    if window % 2 == 0:
        window += 1

    # Order must be less than window
    order = min(order, window - 1)

    return savgol_filter(gr, window, order)


def normalize_gr_0_100(
    type_gr: np.ndarray,
    pseudo_gr: np.ndarray,
    well_gr: np.ndarray,
    type_tvd: np.ndarray,
    pseudo_tvd: np.ndarray,
    well_md: np.ndarray,
    landing_md: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Normalize ALL GR arrays to 0-100 scale.

    Range (gr_min, gr_max) is computed from WellLog (well_gr)
    from (landing_md - 300m) to the end of the well.

    Args:
        type_gr: TypeLog GR values
        pseudo_gr: PseudoTypeLog GR values
        well_gr: WellLog GR values
        type_tvd: TypeLog TVD (shifted)
        pseudo_tvd: PseudoTypeLog TVD
        well_md: WellLog MD
        landing_md: Landing point MD (start of optimization)

    Returns:
        (type_gr_norm, pseudo_gr_norm, well_gr_norm, gr_min, gr_max)
    """
    # Get GR range from WellLog from (landing_md - 300m) to end
    range_start_md = landing_md - 300.0
    well_mask = well_md >= range_start_md
    if well_mask.any():
        gr_for_range = well_gr[well_mask]
    else:
        gr_for_range = well_gr

    gr_min, gr_max = gr_for_range.min(), gr_for_range.max()

    if gr_max - gr_min < 1e-6:
        # Constant GR - return as-is with dummy range
        return type_gr.copy(), pseudo_gr.copy(), well_gr.copy(), gr_min, gr_max

    # Normalize ALL 3 logs with same range
    type_gr_norm = (type_gr - gr_min) / (gr_max - gr_min) * 100
    pseudo_gr_norm = (pseudo_gr - gr_min) / (gr_max - gr_min) * 100
    well_gr_norm = (well_gr - gr_min) / (gr_max - gr_min) * 100

    return type_gr_norm, pseudo_gr_norm, well_gr_norm, float(gr_min), float(gr_max)


def compute_overlap_metrics(
    type_tvd: np.ndarray, type_gr: np.ndarray,
    pseudo_tvd: np.ndarray, pseudo_gr: np.ndarray
) -> Dict[str, float]:
    """
    Compute Pearson and RMSE in overlap zone between TypeLog and PseudoTypeLog.

    Expects data already normalized to 0-100 scale.

    Args:
        type_tvd, type_gr: TypeLog arrays (shifted, normalized 0-100)
        pseudo_tvd, pseudo_gr: PseudoTypeLog arrays (normalized 0-100)

    Returns:
        Dict with pearson, rmse, overlap_start, overlap_end, overlap_length
    """
    # Find overlap zone
    overlap_start = max(type_tvd.min(), pseudo_tvd.min())
    overlap_end = min(type_tvd.max(), pseudo_tvd.max())

    if overlap_end <= overlap_start:
        return {
            'pearson': np.nan, 'rmse': np.nan,
            'overlap_start': 0, 'overlap_end': 0, 'overlap_length': 0
        }

    # Interpolate both to common grid in overlap zone
    step = 0.1  # 0.1m step
    common_tvd = np.arange(overlap_start, overlap_end, step)

    if len(common_tvd) < 2:
        return {
            'pearson': np.nan, 'rmse': np.nan,
            'overlap_start': float(overlap_start), 'overlap_end': float(overlap_end),
            'overlap_length': float(overlap_end - overlap_start)
        }

    type_interp = np.interp(common_tvd, type_tvd, type_gr)
    pseudo_interp = np.interp(common_tvd, pseudo_tvd, pseudo_gr)

    # RMSE on already normalized 0-100 data
    rmse = np.sqrt(np.mean((type_interp - pseudo_interp) ** 2))

    # Pearson correlation
    type_centered = type_interp - type_interp.mean()
    pseudo_centered = pseudo_interp - pseudo_interp.mean()
    denom = np.sqrt((type_centered ** 2).sum() * (pseudo_centered ** 2).sum())
    pearson = (type_centered * pseudo_centered).sum() / denom if denom > 1e-10 else 0.0

    return {
        'pearson': float(pearson),
        'rmse': float(rmse),
        'overlap_start': float(overlap_start),
        'overlap_end': float(overlap_end),
        'overlap_length': float(overlap_end - overlap_start)
    }


def stitch_typelogs(
    type_tvd: np.ndarray, type_gr: np.ndarray,
    pseudo_tvd: np.ndarray, pseudo_gr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stitch TypeLog and PseudoTypeLog.

    In overlap zone: use PseudoTypeLog (more accurate for this well)
    Outside overlap: use TypeLog

    Both inputs must be in well coordinates (tvd_shift already applied to TypeLog).

    Args:
        type_tvd, type_gr: TypeLog (shifted to well coordinates)
        pseudo_tvd, pseudo_gr: PseudoTypeLog (already in well coordinates)

    Returns:
        (result_tvd, result_gr) - stitched arrays
    """
    # Find overlap zone
    overlap_start = max(type_tvd.min(), pseudo_tvd.min())
    overlap_end = min(type_tvd.max(), pseudo_tvd.max())

    # Pseudo step for interpolation grid
    if len(pseudo_tvd) >= 2:
        pseudo_step = pseudo_tvd[1] - pseudo_tvd[0]
    else:
        pseudo_step = 0.1

    # Build result arrays
    result_tvd_list = []
    result_gr_list = []

    # Part 1: TypeLog BEFORE pseudo - INTERPOLATE to pseudo_step grid
    type_min = type_tvd.min()
    if type_min < overlap_start - pseudo_step:
        # Create fine grid from type_min to overlap_start
        grid_before = np.arange(type_min, overlap_start - pseudo_step / 2, pseudo_step)
        if len(grid_before) > 0:
            gr_before = np.interp(grid_before, type_tvd, type_gr)
            result_tvd_list.append(grid_before)
            result_gr_list.append(gr_before)

    # Part 2: PseudoTypeLog (entire range)
    result_tvd_list.append(pseudo_tvd)
    result_gr_list.append(pseudo_gr)

    # Part 3: TypeLog AFTER pseudo - INTERPOLATE to pseudo_step grid
    type_max = type_tvd.max()
    if type_max > overlap_end + pseudo_step:
        # Create fine grid from overlap_end to type_max
        grid_after = np.arange(overlap_end + pseudo_step, type_max + pseudo_step / 2, pseudo_step)
        if len(grid_after) > 0:
            gr_after = np.interp(grid_after, type_tvd, type_gr)
            result_tvd_list.append(grid_after)
            result_gr_list.append(gr_after)

    result_tvd = np.concatenate(result_tvd_list)
    result_gr = np.concatenate(result_gr_list)

    # Sort by TVD
    sort_idx = np.argsort(result_tvd)
    result_tvd = result_tvd[sort_idx]
    result_gr = result_gr[sort_idx]

    return result_tvd, result_gr


def prepare_typelog(
    data: Dict[str, Any],
    use_pseudo: bool = None,
    apply_smoothing: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Prepare TypeLog for optimizer with correct coordinate alignment.

    Args:
        data: Well data from dataset (tensors)
        use_pseudo: Use PseudoTypeLog stitching (default from USE_PSEUDO_TYPELOG env)
        apply_smoothing: Apply Savitzky-Golay filter

    Returns:
        (tvd, gr, metadata):
            - tvd: TVD array in well coordinates
            - gr: GR array (raw values)
            - metadata: {gr_min, gr_max, tvd_shift, overlap_metrics, ...}
    """
    if use_pseudo is None:
        use_pseudo = USE_PSEUDO_TYPELOG

    # Get raw data from tensors
    pseudo_tvd = data['pseudo_tvd'].cpu().numpy() if hasattr(data['pseudo_tvd'], 'cpu') else np.array(data['pseudo_tvd'])
    pseudo_gr = data['pseudo_gr'].cpu().numpy() if hasattr(data['pseudo_gr'], 'cpu') else np.array(data['pseudo_gr'])
    type_tvd_raw = data['type_tvd'].cpu().numpy() if hasattr(data['type_tvd'], 'cpu') else np.array(data['type_tvd'])
    type_gr = data['type_gr'].cpu().numpy() if hasattr(data['type_gr'], 'cpu') else np.array(data['type_gr'])
    well_md = data['log_md'].cpu().numpy() if hasattr(data['log_md'], 'cpu') else np.array(data['log_md'])
    well_gr = data['log_gr'].cpu().numpy() if hasattr(data['log_gr'], 'cpu') else np.array(data['log_gr'])

    # Get tvd_shift, norm_multiplier, landing_md
    tvd_shift = data.get('tvd_typewell_shift', 0.0)
    if hasattr(tvd_shift, 'item'):
        tvd_shift = tvd_shift.item()

    norm_multiplier = data.get('norm_multiplier', 1.0)
    if hasattr(norm_multiplier, 'item'):
        norm_multiplier = norm_multiplier.item()

    landing_md = data.get('landing_end_87_200', data.get('landing_end_dls', 0.0))
    if hasattr(landing_md, 'item'):
        landing_md = landing_md.item()

    # Step 1: Apply norm_multiplier (pseudo * mult, well * mult, type unchanged)
    pseudo_gr_mult = pseudo_gr * norm_multiplier if norm_multiplier != 1.0 else pseudo_gr.copy()
    well_gr_mult = well_gr * norm_multiplier if norm_multiplier != 1.0 else well_gr.copy()
    type_gr_raw = type_gr.copy()

    # Step 2: ALWAYS apply tvd_shift to type_tvd for coordinate alignment
    # CRITICAL: DO NOT REMOVE THIS SHIFT! Without it, TypeLog and PseudoTypeLog
    # coordinates don't match, resulting in garbage correlations.
    # DO NOT DISABLE UNLESS USER EXPLICITLY REQUESTS IT TWICE.
    type_tvd = type_tvd_raw + tvd_shift

    # Step 3: Normalize ALL data to 0-100 scale
    # Range is computed from WellLog (log_gr) from landing_md to end
    type_gr_norm, pseudo_gr_norm, well_gr_norm, gr_min, gr_max = normalize_gr_0_100(
        type_gr_raw, pseudo_gr_mult, well_gr_mult,
        type_tvd, pseudo_tvd, well_md, landing_md
    )

    # Step 4: Compute overlap metrics (NO transformations - data already normalized)
    overlap_metrics = compute_overlap_metrics(
        type_tvd, type_gr_norm,
        pseudo_tvd, pseudo_gr_norm
    )

    logger.info(
        f"prepare_typelog: tvd_shift={tvd_shift:.1f}, gr_range=[{gr_min:.1f}-{gr_max:.1f}], "
        f"type_tvd=[{type_tvd.min():.0f}-{type_tvd.max():.0f}], "
        f"pseudo_tvd=[{pseudo_tvd.min():.0f}-{pseudo_tvd.max():.0f}], "
        f"overlap={overlap_metrics['overlap_length']:.0f}m, "
        f"pearson={overlap_metrics['pearson']:.3f}, rmse={overlap_metrics['rmse']:.2f}"
    )

    # Step 5: Stitch NORMALIZED data
    if use_pseudo:
        result_tvd, result_gr = stitch_typelogs(
            type_tvd, type_gr_norm,
            pseudo_tvd, pseudo_gr_norm
        )
        logger.debug(f"Stitched TypeLog: {len(type_tvd)} + {len(pseudo_tvd)} -> {len(result_tvd)} points")
    else:
        result_tvd = type_tvd.copy()
        result_gr = type_gr_norm.copy()
        logger.debug(f"TypeLog only: {len(result_tvd)} points")

    # Step 6: Apply smoothing
    if apply_smoothing and GR_SMOOTHING_WINDOW >= 3:
        result_gr = apply_gr_smoothing(result_gr)
        logger.debug(f"Applied smoothing (window={GR_SMOOTHING_WINDOW})")

    metadata = {
        'gr_min': float(gr_min),
        'gr_max': float(gr_max),
        'tvd_shift': float(tvd_shift),  # Always applied
        'norm_multiplier': float(norm_multiplier),
        'use_pseudo': use_pseudo,
        'overlap_metrics': overlap_metrics,
        'n_points': len(result_tvd),
        'well_gr_norm': well_gr_norm,  # Normalized WellLog for optimizer
        'well_md': well_md,
    }

    return result_tvd, result_gr, metadata


def normalize_well_gr(
    well_gr: np.ndarray,
    norm_multiplier: float,
    gr_min: float = None,
    gr_max: float = None
) -> np.ndarray:
    """
    Normalize well GR to match TypeLog (OLD behavior - only apply norm_multiplier).

    Args:
        well_gr: Well GR array
        norm_multiplier: Normalization multiplier from dataset
        gr_min, gr_max: Unused (kept for API compatibility)

    Returns:
        Well GR with norm_multiplier applied (raw scale, no 0-100 normalization)
    """
    # Only apply norm_multiplier (same as OLD)
    result = well_gr * norm_multiplier if norm_multiplier != 1.0 else well_gr.copy()
    return result
