"""
TypeWell Provider Module

Unified interface for obtaining typewell data with different strategies:
- original: Standard typeLog from PAPI/StarSteer
- alternative: Curve-replaced typewell from AlternativeTypewellStorage
- pseudo_stitched: PseudoTypeLog extended with typeLog data

Configuration via .env:
    TYPEWELL_MODE=original  # original | alternative | pseudo_stitched

Author: Auto-generated
Date: 2025-12-24
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


# Read smoothing params from env
GR_SMOOTHING_WINDOW = int(os.getenv('GR_SMOOTHING_WINDOW', '0'))
GR_SMOOTHING_ORDER = int(os.getenv('GR_SMOOTHING_ORDER', '2'))


def extend_pseudo_with_typelog(pseudo_log: Dict, type_log: Dict, norm_coef: float = 1.0) -> Dict:
    """
    Extend pseudoTypeLog with interpolated points from typeLog in BOTH directions.

    Takes points from typeLog where:
    - TVD < min(pseudo.TVD) (extend DOWN)
    - TVD > max(pseudo.TVD) (extend UP)

    Interpolates to pseudo's step size and normalizes data.

    Args:
        pseudo_log: PseudoTypeLog data with 'tvdSortedPoints' and 'points'
        type_log: TypeLog data with 'tvdSortedPoints' and 'points'
        norm_coef: Normalization coefficient for typeLog data (1.0 / multiplier)
                   Applied to typeLog points being added to match normalized well data.

    Returns:
        Extended pseudo log with additional points from typeLog (both directions)
    """
    pseudo_points = pseudo_log.get('tvdSortedPoints', [])
    type_points = type_log.get('tvdSortedPoints', [])

    if not pseudo_points or not type_points:
        logger.warning("Empty pseudo or type log, returning original pseudo")
        return pseudo_log

    # Get TVD ranges
    pseudo_min_tvd = pseudo_points[0]['trueVerticalDepth']
    pseudo_max_tvd = pseudo_points[-1]['trueVerticalDepth']
    type_min_tvd = type_points[0]['trueVerticalDepth']
    type_max_tvd = type_points[-1]['trueVerticalDepth']

    # Calculate pseudo step
    if len(pseudo_points) < 2:
        logger.warning("Not enough pseudo points for step calculation")
        return pseudo_log
    pseudo_step = pseudo_points[1]['trueVerticalDepth'] - pseudo_points[0]['trueVerticalDepth']

    # Build typeLog interpolation arrays (all valid points)
    type_tvd = []
    type_data = []
    for p in type_points:
        tvd = p['trueVerticalDepth']
        data_val = p['data']
        # Skip points with None or non-numeric data
        if data_val is None or not isinstance(data_val, (int, float)):
            continue
        type_tvd.append(tvd)
        type_data.append(float(data_val))

    if len(type_tvd) < 2:
        logger.warning("Not enough typeLog points for interpolation")
        return pseudo_log

    type_tvd = np.array(type_tvd, dtype=np.float64)
    type_data = np.array(type_data, dtype=np.float64)

    # === EXTEND DOWN (typeLog before pseudo) ===
    new_points_down = []
    new_points_md_down = []
    if type_min_tvd < pseudo_min_tvd - pseudo_step:
        # Create TVD grid from type_min to pseudo_min
        down_tvd_grid = np.arange(type_min_tvd, pseudo_min_tvd - pseudo_step / 2, pseudo_step)
        if len(down_tvd_grid) > 0:
            down_data = np.interp(down_tvd_grid, type_tvd, type_data)
            if norm_coef != 1.0:
                down_data = down_data * norm_coef
            for tvd, data in zip(down_tvd_grid, down_data):
                new_points_down.append({
                    'data': float(data),
                    'measuredDepth': float(tvd),
                    'trueVerticalDepth': float(tvd)
                })
                new_points_md_down.append({
                    'data': float(data),
                    'measuredDepth': float(tvd)
                })

    # === EXTEND UP (typeLog after pseudo) ===
    new_points_up = []
    new_points_md_up = []
    if type_max_tvd > pseudo_max_tvd + pseudo_step:
        # Create TVD grid from pseudo_max to type_max
        up_tvd_grid = np.arange(pseudo_max_tvd + pseudo_step, type_max_tvd + pseudo_step / 2, pseudo_step)
        if len(up_tvd_grid) > 0:
            up_data = np.interp(up_tvd_grid, type_tvd, type_data)
            if norm_coef != 1.0:
                up_data = up_data * norm_coef
            for tvd, data in zip(up_tvd_grid, up_data):
                new_points_up.append({
                    'data': float(data),
                    'measuredDepth': float(tvd),
                    'trueVerticalDepth': float(tvd)
                })
                new_points_md_up.append({
                    'data': float(data),
                    'measuredDepth': float(tvd)
                })

    if not new_points_down and not new_points_up:
        logger.info(f"No extension needed: typeLog [{type_min_tvd:.1f}-{type_max_tvd:.1f}] within pseudo [{pseudo_min_tvd:.1f}-{pseudo_max_tvd:.1f}]")
        return pseudo_log

    # Create extended pseudo log: DOWN + pseudo + UP
    pseudo_points_md = pseudo_log.get('points', [])
    extended_pseudo = {
        'points': new_points_md_down + pseudo_points_md + new_points_md_up,
        'tvdSortedPoints': new_points_down + pseudo_points + new_points_up,
        'uuid': pseudo_log.get('uuid', '')
    }

    final_min = new_points_down[0]['trueVerticalDepth'] if new_points_down else pseudo_min_tvd
    final_max = new_points_up[-1]['trueVerticalDepth'] if new_points_up else pseudo_max_tvd
    logger.info(f"Extended pseudoTypeLog: {len(pseudo_points)} -> {len(extended_pseudo['tvdSortedPoints'])} points, "
                f"TVD [{pseudo_min_tvd:.1f}-{pseudo_max_tvd:.1f}] -> [{final_min:.1f}-{final_max:.1f}], "
                f"down={len(new_points_down)}, up={len(new_points_up)}, norm_coef={norm_coef:.6f}")

    return extended_pseudo


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


def stitch_typewell_from_dataset(
    data: Dict[str, Any],
    mode: str = None,
    apply_smoothing: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stitch pseudo + type from dataset tensors with normalization and optional smoothing.

    Unified function for all modules (optimizer, visualizer, slicer).

    Args:
        data: Well data from dataset (contains pseudo_tvd, pseudo_gr, type_tvd, type_gr, norm_multiplier as tensors)
        mode: 'OLD' = type×(1/mult), pseudo raw (stitched)
              'NEW' = pseudo×mult, type raw (stitched)
              'ORIGINAL' = pure typeLog, no pseudo stitching
              None = read from NORMALIZATION_MODE env (default 'OLD')
        apply_smoothing: Apply Savitzky-Golay filter (uses GR_SMOOTHING_WINDOW env)

    Returns:
        Tuple of (tvd_array, gr_array) as numpy arrays
    """
    if mode is None:
        mode = os.getenv('NORMALIZATION_MODE', 'NEW')

    norm_multiplier = data.get('norm_multiplier', 1.0)
    if hasattr(norm_multiplier, 'item'):
        norm_multiplier = norm_multiplier.item()

    # Get raw data from tensors
    pseudo_tvd = data['pseudo_tvd'].cpu().numpy() if hasattr(data['pseudo_tvd'], 'cpu') else np.array(data['pseudo_tvd'])
    pseudo_gr = data['pseudo_gr'].cpu().numpy() if hasattr(data['pseudo_gr'], 'cpu') else np.array(data['pseudo_gr'])
    type_tvd = data['type_tvd'].cpu().numpy() if hasattr(data['type_tvd'], 'cpu') else np.array(data['type_tvd'])
    type_gr = data['type_gr'].cpu().numpy() if hasattr(data['type_gr'], 'cpu') else np.array(data['type_gr'])

    # ORIGINAL mode: pure typeLog without stitching
    if mode == 'ORIGINAL':
        logger.debug(f"Using ORIGINAL typeLog (no pseudo stitching), {len(type_tvd)} points")
        result_tvd = type_tvd
        result_gr = type_gr
    else:
        # Build dict format for extend_pseudo_with_typelog
        pseudo_dict = {
            'tvdSortedPoints': [{'trueVerticalDepth': float(tvd), 'data': float(gr)}
                               for tvd, gr in zip(pseudo_tvd, pseudo_gr)]
        }
        type_dict = {
            'tvdSortedPoints': [{'trueVerticalDepth': float(tvd), 'data': float(gr)}
                               for tvd, gr in zip(type_tvd, type_gr)]
        }

        if mode == 'NEW':
            # NEW: pseudo × mult, type raw
            if norm_multiplier != 1.0:
                for p in pseudo_dict['tvdSortedPoints']:
                    p['data'] *= norm_multiplier
            stitched = extend_pseudo_with_typelog(pseudo_dict, type_dict, norm_coef=1.0)
        else:
            # OLD: pseudo raw, type × (1/mult)
            norm_coef = 1.0 / norm_multiplier if norm_multiplier != 0 else 1.0
            stitched = extend_pseudo_with_typelog(pseudo_dict, type_dict, norm_coef=norm_coef)

        points = stitched.get('tvdSortedPoints', [])
        result_tvd = np.array([p['trueVerticalDepth'] for p in points])
        result_gr = np.array([p['data'] for p in points])

    # Apply smoothing if enabled
    if apply_smoothing and GR_SMOOTHING_WINDOW >= 3:
        result_gr = apply_gr_smoothing(result_gr)
        logger.debug(f"Applied Savitzky-Golay smoothing (window={GR_SMOOTHING_WINDOW}, order={GR_SMOOTHING_ORDER})")

    return result_tvd, result_gr


def compute_overlap_metrics(
    type_tvd: np.ndarray, type_gr: np.ndarray,
    pseudo_tvd: np.ndarray, pseudo_gr: np.ndarray
) -> Dict[str, float]:
    """
    Compute MSE and Pearson correlation in overlap zone (normalized).

    Args:
        type_tvd, type_gr: TypeLog arrays
        pseudo_tvd, pseudo_gr: PseudoTypeLog arrays

    Returns:
        Dict with 'mse' (×10), 'pearson', 'overlap_start', 'overlap_end', 'overlap_length'
    """
    # Find overlap zone
    overlap_start = max(type_tvd.min(), pseudo_tvd.min())
    overlap_end = min(type_tvd.max(), pseudo_tvd.max())

    if overlap_end <= overlap_start:
        return {'mse': np.nan, 'pearson': np.nan, 'overlap_start': 0, 'overlap_end': 0, 'overlap_length': 0}

    # Interpolate both to common grid in overlap zone
    step = 0.1  # 0.1m step
    common_tvd = np.arange(overlap_start, overlap_end, step)

    type_interp = np.interp(common_tvd, type_tvd, type_gr)
    pseudo_interp = np.interp(common_tvd, pseudo_tvd, pseudo_gr)

    # Normalize: find local min/max, scale to 0-100
    all_gr = np.concatenate([type_interp, pseudo_interp])
    gr_min, gr_max = all_gr.min(), all_gr.max()

    if gr_max - gr_min < 1e-6:
        return {'mse': 0.0, 'pearson': 1.0, 'overlap_start': overlap_start, 'overlap_end': overlap_end,
                'overlap_length': overlap_end - overlap_start}

    type_norm = (type_interp - gr_min) / (gr_max - gr_min) * 100
    pseudo_norm = (pseudo_interp - gr_min) / (gr_max - gr_min) * 100

    # MSE × 10
    mse = np.mean((type_norm - pseudo_norm) ** 2) * 10

    # Pearson
    type_centered = type_norm - type_norm.mean()
    pseudo_centered = pseudo_norm - pseudo_norm.mean()
    denom = np.sqrt((type_centered ** 2).sum() * (pseudo_centered ** 2).sum())
    pearson = (type_centered * pseudo_centered).sum() / denom if denom > 1e-10 else 0.0

    return {
        'mse': float(mse),
        'pearson': float(pearson),
        'overlap_start': float(overlap_start),
        'overlap_end': float(overlap_end),
        'overlap_length': float(overlap_end - overlap_start)
    }


def compute_stitch_quality(
    data: Dict[str, Any],
    mode: str = None
) -> Dict[str, float]:
    """
    Compute overlap quality metrics BEFORE stitching.

    Uses same normalization as stitch_typewell_from_dataset but measures
    Pearson and RMSE between TypeLog and PseudoTypeLog in overlap zone.

    Args:
        data: Well data from dataset
        mode: 'OLD', 'NEW', or 'ORIGINAL' (default from env)

    Returns:
        Dict with pearson, rmse, overlap_length, and normalized GR ranges
    """
    if mode is None:
        mode = os.getenv('NORMALIZATION_MODE', 'NEW')

    norm_multiplier = data.get('norm_multiplier', 1.0)
    if hasattr(norm_multiplier, 'item'):
        norm_multiplier = norm_multiplier.item()

    # Get raw data
    pseudo_tvd = data['pseudo_tvd'].cpu().numpy() if hasattr(data['pseudo_tvd'], 'cpu') else np.array(data['pseudo_tvd'])
    pseudo_gr = data['pseudo_gr'].cpu().numpy() if hasattr(data['pseudo_gr'], 'cpu') else np.array(data['pseudo_gr'])
    type_tvd = data['type_tvd'].cpu().numpy() if hasattr(data['type_tvd'], 'cpu') else np.array(data['type_tvd'])
    type_gr = data['type_gr'].cpu().numpy() if hasattr(data['type_gr'], 'cpu') else np.array(data['type_gr'])

    # Apply same normalization as stitch_typewell_from_dataset
    if mode == 'NEW':
        # pseudo × mult, type raw
        pseudo_gr_norm = pseudo_gr * norm_multiplier
        type_gr_norm = type_gr.copy()
    elif mode == 'OLD':
        # pseudo raw, type × (1/mult)
        pseudo_gr_norm = pseudo_gr.copy()
        norm_coef = 1.0 / norm_multiplier if norm_multiplier != 0 else 1.0
        type_gr_norm = type_gr * norm_coef
    else:  # ORIGINAL
        return {'pearson': np.nan, 'rmse': np.nan, 'overlap_length': 0, 'mode': mode}

    # Compute metrics on normalized data
    metrics = compute_overlap_metrics(type_tvd, type_gr_norm, pseudo_tvd, pseudo_gr_norm)

    # Add RMSE (sqrt of MSE/10)
    rmse = np.sqrt(metrics['mse'] / 10) if not np.isnan(metrics['mse']) else np.nan

    return {
        'pearson': metrics['pearson'],
        'rmse': rmse,
        'mse_x10': metrics['mse'],
        'overlap_length': metrics['overlap_length'],
        'overlap_start': metrics['overlap_start'],
        'overlap_end': metrics['overlap_end'],
        'mode': mode,
        'norm_multiplier': norm_multiplier
    }


class TypewellProvider:
    """
    Unified typewell provider with multiple strategies.

    Modes:
        - original: Use typeLog as-is from well_data
        - alternative: Load from AlternativeTypewellStorage (curve replacement)
        - pseudo_stitched: Extend pseudoTypeLog with typeLog data
    """

    # Valid modes
    MODES = ('original', 'alternative', 'pseudo_stitched')

    def __init__(self, mode: Optional[str] = None):
        """
        Initialize provider with specified mode.

        Args:
            mode: Typewell mode. If None, reads from TYPEWELL_MODE env var.
        """
        self.mode = mode or os.getenv('TYPEWELL_MODE', 'original').lower()

        if self.mode not in self.MODES:
            logger.warning(f"Unknown TYPEWELL_MODE '{self.mode}', falling back to 'original'")
            self.mode = 'original'

        # Lazy-load storage for alternative mode
        self._storage = None

        # Normalization coefficient for typeLog (used in pseudo_stitched mode)
        # Default 1.0 = no normalization
        # Set via set_normalization_coef(multiplier) from NormalizationCalculator result
        self._norm_coef = 1.0

        logger.info(f"TypewellProvider initialized with mode: {self.mode}")

    @property
    def storage(self):
        """Lazy-load AlternativeTypewellStorage"""
        if self._storage is None:
            from self_correlation.alternative_typewell_storage import AlternativeTypewellStorage
            self._storage = AlternativeTypewellStorage()
        return self._storage

    @property
    def norm_coef(self) -> float:
        """Current normalization coefficient for typeLog"""
        return self._norm_coef

    def set_normalization_coef(self, multiplier: float):
        """
        Set normalization coefficient from NormalizationCalculator result.

        The multiplier is what NormalizationCalculator applies to well data:
            well.value = well.value * multiplier

        For typeLog we need the inverse to match normalized well:
            typeLog.value = typeLog.value / multiplier

        Args:
            multiplier: Normalization multiplier from NormalizationCalculator
        """
        if multiplier == 0:
            logger.error("Cannot set normalization coef: multiplier is 0")
            return

        self._norm_coef = 1.0 / multiplier
        logger.info(f"TypewellProvider: set norm_coef={self._norm_coef:.6f} (from multiplier={multiplier:.6f})")

    def reset_normalization_coef(self):
        """Reset normalization coefficient to default (1.0 = no normalization)"""
        self._norm_coef = 1.0
        logger.debug("TypewellProvider: reset norm_coef to 1.0")

    def get_typewell(self, well_data: Dict[str, Any], current_md: float = None) -> Dict:
        """
        Get typewell data according to configured mode.

        Args:
            well_data: Full well data dictionary containing typeLog, pseudoTypeLog, etc.
            current_md: Current measured depth (for future per-step updates)

        Returns:
            TypeLog data dictionary ready for use

        Raises:
            ValueError: If required data is missing for selected mode
            FileNotFoundError: If alternative typewell file not found
        """
        well_name = well_data.get('wellName', 'unknown')

        if self.mode == 'original':
            return self._get_original(well_data, well_name)

        elif self.mode == 'alternative':
            return self._get_alternative(well_data, well_name)

        elif self.mode == 'pseudo_stitched':
            return self._get_pseudo_stitched(well_data, well_name, current_md)

        else:
            # Fallback (should not happen due to __init__ validation)
            logger.error(f"Invalid mode '{self.mode}', using original")
            return self._get_original(well_data, well_name)

    def _get_original(self, well_data: Dict, well_name: str) -> Dict:
        """Return original typeLog as-is"""
        type_log = well_data.get('typeLog')
        if not type_log:
            raise ValueError(f"typeLog not found in well_data for '{well_name}'")

        logger.debug(f"[{well_name}] Using original typeLog")
        return type_log

    def _get_alternative(self, well_data: Dict, well_name: str) -> Dict:
        """Load from AlternativeTypewellStorage"""
        # Check if alternative exists
        if not self.storage.exists(well_name):
            logger.warning(f"[{well_name}] Alternative typewell not found, falling back to original")
            return self._get_original(well_data, well_name)

        alternative = self.storage.load(well_name)
        logger.info(f"[{well_name}] Using alternative typewell (curve replacement)")
        return alternative

    def _get_pseudo_stitched(self, well_data: Dict, well_name: str, current_md: float = None) -> Dict:
        """Extend pseudoTypeLog with typeLog"""
        pseudo_log = well_data.get('pseudoTypeLog')
        type_log = well_data.get('typeLog')

        if not pseudo_log:
            logger.warning(f"[{well_name}] pseudoTypeLog not found, falling back to original")
            return self._get_original(well_data, well_name)

        if not type_log:
            logger.warning(f"[{well_name}] typeLog not found for stitching, using pseudo only")
            return pseudo_log

        # Stitch pseudo with typelog, applying normalization coefficient
        stitched = extend_pseudo_with_typelog(pseudo_log, type_log, self._norm_coef)

        if current_md:
            logger.info(f"[{well_name}] Using pseudo_stitched typewell (MD={current_md:.1f}m, norm_coef={self._norm_coef:.6f})")
        else:
            logger.info(f"[{well_name}] Using pseudo_stitched typewell (norm_coef={self._norm_coef:.6f})")

        return stitched

    def apply_to_well_data(self, well_data: Dict[str, Any], current_md: float = None) -> Dict[str, Any]:
        """
        Apply typewell transformation to well_data in-place.

        Convenience method that gets typewell and replaces typeLog in well_data.

        Args:
            well_data: Well data dictionary (will be modified)
            current_md: Current measured depth

        Returns:
            Modified well_data (same object)
        """
        typewell = self.get_typewell(well_data, current_md)
        well_data['typeLog'] = typewell
        return well_data
