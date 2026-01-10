"""
Preprocessing Module for GPU AG Optimizer

Single entry point for all data preparation:
- TypeLog, PseudoTypeLog, WellLog normalization
- Coordinate alignment (tvd_shift)
- Stitching TypeLog + PseudoTypeLog
- Consistent 0-100 normalization (when enabled)

CRITICAL: All 3 curves (TypeLog, PseudoTypeLog, WellLog) must use
the same gr_min/gr_max for 0-100 normalization!

Configuration via env:
    USE_PSEUDO_TYPELOG=True   # True = stitch, False = TypeLog only
    NORMALIZE_0_100=False     # True = normalize to 0-100 scale
    GR_SMOOTHING_WINDOW=0     # Savitzky-Golay window (0 = disabled)
    GR_SMOOTHING_ORDER=2      # Savitzky-Golay polynomial order

WORKING CONFIGURATION (2026-01-10):
    Dataset: data/wells_limited_pseudo.pt (100 wells, no pseudo leakage)
    NORMALIZE_0_100=True (default) - consistent 0-100 normalization
    USE_PSEUDO_TYPELOG=True: RMSE=4.69m, 64/100 wells improved
    USE_PSEUDO_TYPELOG=False: RMSE=6.22m, 54/100 wells improved

Author: Auto-generated
Date: 2026-01-10
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Tuple, NamedTuple
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

# Configuration from env
USE_PSEUDO_TYPELOG = os.getenv('USE_PSEUDO_TYPELOG', 'True').lower() in ('true', '1', 'yes')
NORMALIZE_0_100 = os.getenv('NORMALIZE_0_100', 'True').lower() in ('true', '1', 'yes')
GR_SMOOTHING_WINDOW = int(os.getenv('GR_SMOOTHING_WINDOW', '0'))
GR_SMOOTHING_ORDER = int(os.getenv('GR_SMOOTHING_ORDER', '2'))


class PreparedData(NamedTuple):
    """All data needed by optimizer."""
    # TypeLog (stitched or not)
    type_tvd: np.ndarray
    type_gr: np.ndarray

    # WellLog interpolated to well_md
    well_md: np.ndarray
    well_tvd: np.ndarray
    well_gr: np.ndarray

    # Metadata
    tvd_shift: float
    gr_min: float
    gr_max: float
    norm_multiplier: float
    use_pseudo: bool
    normalized_0_100: bool
    overlap_metrics: Dict[str, float]


def _to_numpy(tensor) -> np.ndarray:
    """Convert tensor or array to numpy."""
    if hasattr(tensor, 'cpu'):
        return tensor.cpu().numpy()
    return np.array(tensor)


def _get_scalar(value, default=0.0) -> float:
    """Extract scalar from tensor or return default."""
    if value is None:
        return default
    if hasattr(value, 'item'):
        return value.item()
    return float(value)


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
        landing_md: Landing point MD

    Returns:
        (type_gr_norm, pseudo_gr_norm, well_gr_norm, gr_min, gr_max)
    """
    range_start_md = landing_md - 300.0
    well_mask = well_md >= range_start_md
    gr_for_range = well_gr[well_mask] if well_mask.any() else well_gr

    gr_min, gr_max = gr_for_range.min(), gr_for_range.max()

    if gr_max - gr_min < 1e-6:
        return type_gr.copy(), pseudo_gr.copy(), well_gr.copy(), gr_min, gr_max

    scale = 100.0 / (gr_max - gr_min)
    type_gr_norm = (type_gr - gr_min) * scale
    pseudo_gr_norm = (pseudo_gr - gr_min) * scale
    well_gr_norm = (well_gr - gr_min) * scale

    return type_gr_norm, pseudo_gr_norm, well_gr_norm, float(gr_min), float(gr_max)


def compute_overlap_metrics(
    type_tvd: np.ndarray, type_gr: np.ndarray,
    pseudo_tvd: np.ndarray, pseudo_gr: np.ndarray
) -> Dict[str, float]:
    """
    Compute Pearson and RMSE in overlap zone between TypeLog and PseudoTypeLog.

    Args:
        type_tvd, type_gr: TypeLog arrays (shifted)
        pseudo_tvd, pseudo_gr: PseudoTypeLog arrays

    Returns:
        Dict with pearson, rmse, overlap_start, overlap_end, overlap_length
    """
    overlap_start = max(type_tvd.min(), pseudo_tvd.min())
    overlap_end = min(type_tvd.max(), pseudo_tvd.max())

    if overlap_end <= overlap_start:
        return {
            'pearson': np.nan, 'rmse': np.nan,
            'overlap_start': 0, 'overlap_end': 0, 'overlap_length': 0
        }

    step = 0.1
    common_tvd = np.arange(overlap_start, overlap_end, step)

    if len(common_tvd) < 2:
        return {
            'pearson': np.nan, 'rmse': np.nan,
            'overlap_start': float(overlap_start), 'overlap_end': float(overlap_end),
            'overlap_length': float(overlap_end - overlap_start)
        }

    type_interp = np.interp(common_tvd, type_tvd, type_gr)
    pseudo_interp = np.interp(common_tvd, pseudo_tvd, pseudo_gr)

    rmse = np.sqrt(np.mean((type_interp - pseudo_interp) ** 2))

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

    Args:
        type_tvd, type_gr: TypeLog (shifted to well coordinates)
        pseudo_tvd, pseudo_gr: PseudoTypeLog

    Returns:
        (result_tvd, result_gr)
    """
    overlap_start = max(type_tvd.min(), pseudo_tvd.min())
    overlap_end = min(type_tvd.max(), pseudo_tvd.max())

    if len(pseudo_tvd) >= 2:
        pseudo_step = pseudo_tvd[1] - pseudo_tvd[0]
    else:
        pseudo_step = 0.1

    result_tvd_list = []
    result_gr_list = []

    # Part 1: TypeLog BEFORE pseudo
    type_min = type_tvd.min()
    if type_min < overlap_start - pseudo_step:
        grid_before = np.arange(type_min, overlap_start - pseudo_step / 2, pseudo_step)
        if len(grid_before) > 0:
            gr_before = np.interp(grid_before, type_tvd, type_gr)
            result_tvd_list.append(grid_before)
            result_gr_list.append(gr_before)

    # Part 2: PseudoTypeLog (entire range)
    result_tvd_list.append(pseudo_tvd)
    result_gr_list.append(pseudo_gr)

    # Part 3: TypeLog AFTER pseudo
    type_max = type_tvd.max()
    if type_max > overlap_end + pseudo_step:
        grid_after = np.arange(overlap_end + pseudo_step, type_max + pseudo_step / 2, pseudo_step)
        if len(grid_after) > 0:
            gr_after = np.interp(grid_after, type_tvd, type_gr)
            result_tvd_list.append(grid_after)
            result_gr_list.append(gr_after)

    result_tvd = np.concatenate(result_tvd_list)
    result_gr = np.concatenate(result_gr_list)

    sort_idx = np.argsort(result_tvd)
    return result_tvd[sort_idx], result_gr[sort_idx]


def prepare_data(
    well_data: Dict[str, Any],
    use_pseudo: bool = None,
    normalize_0_100: bool = None,
    apply_smoothing: bool = True
) -> PreparedData:
    """
    Prepare all data for optimizer.

    Single entry point - returns TypeLog and WellLog with consistent normalization.

    Args:
        well_data: Well data from dataset (tensors)
        use_pseudo: Use PseudoTypeLog stitching (default from USE_PSEUDO_TYPELOG env)
        normalize_0_100: Normalize to 0-100 scale (default from NORMALIZE_0_100 env)
        apply_smoothing: Apply Savitzky-Golay filter to TypeLog

    Returns:
        PreparedData with all arrays and metadata
    """
    if use_pseudo is None:
        use_pseudo = USE_PSEUDO_TYPELOG
    if normalize_0_100 is None:
        normalize_0_100 = NORMALIZE_0_100

    # === Extract raw data ===
    pseudo_tvd = _to_numpy(well_data['pseudo_tvd'])
    pseudo_gr = _to_numpy(well_data['pseudo_gr'])
    type_tvd_raw = _to_numpy(well_data['type_tvd'])
    type_gr_raw = _to_numpy(well_data['type_gr'])
    log_md = _to_numpy(well_data['log_md'])
    log_gr = _to_numpy(well_data['log_gr'])
    well_md = _to_numpy(well_data['well_md'])
    well_tvd = _to_numpy(well_data['well_tvd'])

    tvd_shift = _get_scalar(well_data.get('tvd_typewell_shift'), 0.0)
    norm_multiplier = _get_scalar(well_data.get('norm_multiplier'), 1.0)
    landing_md = _get_scalar(
        well_data.get('landing_end_87_200', well_data.get('landing_end_dls')),
        0.0
    )

    # === Step 1: Apply norm_multiplier ===
    # TypeLog: unchanged
    # PseudoTypeLog: × mult
    # WellLog: × mult
    type_gr = type_gr_raw.copy()
    pseudo_gr = pseudo_gr * norm_multiplier if norm_multiplier != 1.0 else pseudo_gr.copy()
    well_gr_interp = np.interp(well_md, log_md, log_gr)
    well_gr = well_gr_interp * norm_multiplier if norm_multiplier != 1.0 else well_gr_interp.copy()

    # === Step 2: Apply tvd_shift to TypeLog ===
    type_tvd = type_tvd_raw + tvd_shift

    # === Step 3: Compute gr_min/gr_max from WellLog ===
    range_start_md = landing_md - 300.0
    well_mask = well_md >= range_start_md
    gr_for_range = well_gr[well_mask] if well_mask.any() else well_gr
    gr_min, gr_max = float(gr_for_range.min()), float(gr_for_range.max())

    # === Step 4: Optional 0-100 normalization ===
    if normalize_0_100 and (gr_max - gr_min) > 1e-6:
        scale = 100.0 / (gr_max - gr_min)
        type_gr = (type_gr - gr_min) * scale
        pseudo_gr = (pseudo_gr - gr_min) * scale
        well_gr = (well_gr - gr_min) * scale

    # === Step 5: Compute overlap metrics ===
    overlap_metrics = compute_overlap_metrics(type_tvd, type_gr, pseudo_tvd, pseudo_gr)

    logger.info(
        f"prepare_data: tvd_shift={tvd_shift:.1f}, gr_range=[{gr_min:.1f}-{gr_max:.1f}], "
        f"norm_0_100={normalize_0_100}, use_pseudo={use_pseudo}, "
        f"overlap={overlap_metrics['overlap_length']:.0f}m, "
        f"pearson={overlap_metrics['pearson']:.3f}"
    )

    # === Step 6: Stitch if needed ===
    if use_pseudo:
        result_tvd, result_gr = stitch_typelogs(type_tvd, type_gr, pseudo_tvd, pseudo_gr)
        logger.debug(f"Stitched: {len(type_tvd)} + {len(pseudo_tvd)} -> {len(result_tvd)} points")
    else:
        result_tvd = type_tvd.copy()
        result_gr = type_gr.copy()
        logger.debug(f"TypeLog only: {len(result_tvd)} points")

    # === Step 7: Apply smoothing ===
    if apply_smoothing and GR_SMOOTHING_WINDOW >= 3:
        result_gr = apply_gr_smoothing(result_gr)
        logger.debug(f"Smoothing applied (window={GR_SMOOTHING_WINDOW})")

    return PreparedData(
        type_tvd=result_tvd,
        type_gr=result_gr,
        well_md=well_md,
        well_tvd=well_tvd,
        well_gr=well_gr,
        tvd_shift=tvd_shift,
        gr_min=gr_min,
        gr_max=gr_max,
        norm_multiplier=norm_multiplier,
        use_pseudo=use_pseudo,
        normalized_0_100=normalize_0_100,
        overlap_metrics=overlap_metrics,
    )


# === Backward compatibility ===
# Keep old function names for existing code

def prepare_typelog(
    data: Dict[str, Any],
    use_pseudo: bool = None,
    apply_smoothing: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    DEPRECATED: Use prepare_data() instead.

    Kept for backward compatibility with full_well_optimizer.py
    """
    prepared = prepare_data(data, use_pseudo=use_pseudo, apply_smoothing=apply_smoothing)

    metadata = {
        'gr_min': prepared.gr_min,
        'gr_max': prepared.gr_max,
        'tvd_shift': prepared.tvd_shift,
        'norm_multiplier': prepared.norm_multiplier,
        'use_pseudo': prepared.use_pseudo,
        'overlap_metrics': prepared.overlap_metrics,
        'n_points': len(prepared.type_tvd),
        'well_gr_norm': prepared.well_gr,  # Now properly normalized!
        'well_md': prepared.well_md,
    }

    return prepared.type_tvd, prepared.type_gr, metadata


def normalize_well_gr(
    well_gr: np.ndarray,
    norm_multiplier: float,
    gr_min: float = None,
    gr_max: float = None
) -> np.ndarray:
    """
    DEPRECATED: Use prepare_data() instead.

    Now properly applies 0-100 normalization if gr_min/gr_max provided.
    """
    result = well_gr * norm_multiplier if norm_multiplier != 1.0 else well_gr.copy()

    # Apply 0-100 normalization if range provided
    if gr_min is not None and gr_max is not None and NORMALIZE_0_100:
        if (gr_max - gr_min) > 1e-6:
            result = (result - gr_min) / (gr_max - gr_min) * 100

    return result
