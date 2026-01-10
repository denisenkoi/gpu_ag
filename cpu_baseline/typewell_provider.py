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

# Import from same package - try relative first, then absolute
try:
    from .preprocessing import apply_gr_smoothing, compute_overlap_metrics, normalize_gr_0_100
except ImportError:
    from preprocessing import apply_gr_smoothing, compute_overlap_metrics, normalize_gr_0_100

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
    well_md = data['log_md'].cpu().numpy() if hasattr(data['log_md'], 'cpu') else np.array(data['log_md'])
    well_gr = data['log_gr'].cpu().numpy() if hasattr(data['log_gr'], 'cpu') else np.array(data['log_gr'])

    # Get tvd_shift and landing_md
    tvd_shift = data.get('tvd_typewell_shift', 0.0)
    if hasattr(tvd_shift, 'item'):
        tvd_shift = tvd_shift.item()

    landing_md = data.get('landing_end_87_200', data.get('landing_end_dls', 0.0))
    if hasattr(landing_md, 'item'):
        landing_md = landing_md.item()

    # ORIGINAL mode: pure typeLog without stitching
    if mode == 'ORIGINAL':
        logger.debug(f"Using ORIGINAL typeLog (no pseudo stitching), {len(type_tvd)} points")
        result_tvd = type_tvd
        result_gr = type_gr
    else:
        # Apply norm_multiplier to GR values
        if mode == 'NEW':
            # NEW: pseudo × mult, well × mult, type raw
            pseudo_gr_mult = pseudo_gr * norm_multiplier if norm_multiplier != 1.0 else pseudo_gr
            well_gr_mult = well_gr * norm_multiplier if norm_multiplier != 1.0 else well_gr
            type_gr_mult = type_gr.copy()
        else:
            # OLD: pseudo raw, well raw, type × (1/mult)
            pseudo_gr_mult = pseudo_gr.copy()
            well_gr_mult = well_gr.copy()
            norm_coef = 1.0 / norm_multiplier if norm_multiplier != 0 else 1.0
            type_gr_mult = type_gr * norm_coef

        # === NORMALIZE AND COMPUTE METRICS ===
        # Apply tvd_shift to type_tvd for correct coordinate alignment
        type_tvd_shifted = type_tvd + tvd_shift

        # Normalize ALL data to 0-100 scale
        # Range is computed from WellLog (log_gr) from landing_md to end
        type_gr_0_100, pseudo_gr_0_100, well_gr_0_100, gr_min, gr_max = normalize_gr_0_100(
            type_gr_mult, pseudo_gr_mult, well_gr_mult,
            type_tvd_shifted, pseudo_tvd, well_md, landing_md
        )

        # Measure overlap quality (NO transformations - data already normalized)
        metrics = compute_overlap_metrics(type_tvd_shifted, type_gr_0_100, pseudo_tvd, pseudo_gr_0_100)
        logger.info(f"PRE-STITCH metrics: tvd_shift={tvd_shift:.1f}, gr_range=[{gr_min:.1f}-{gr_max:.1f}], "
                    f"type_tvd_shifted=[{type_tvd_shifted.min():.0f}-{type_tvd_shifted.max():.0f}], "
                    f"pseudo_tvd=[{pseudo_tvd.min():.0f}-{pseudo_tvd.max():.0f}], "
                    f"overlap={metrics['overlap_length']:.0f}m, "
                    f"pearson={metrics['pearson']:.3f}, rmse={metrics['rmse']:.2f}")

        # Build dict format for extend_pseudo_with_typelog (NORMALIZED 0-100 data)
        pseudo_dict = {
            'tvdSortedPoints': [{'trueVerticalDepth': float(tvd), 'data': float(gr)}
                               for tvd, gr in zip(pseudo_tvd, pseudo_gr_0_100)]
        }
        type_dict = {
            'tvdSortedPoints': [{'trueVerticalDepth': float(tvd), 'data': float(gr)}
                               for tvd, gr in zip(type_tvd, type_gr_0_100)]
        }

        stitched = extend_pseudo_with_typelog(pseudo_dict, type_dict, norm_coef=1.0)

        points = stitched.get('tvdSortedPoints', [])
        result_tvd = np.array([p['trueVerticalDepth'] for p in points])
        result_gr = np.array([p['data'] for p in points])

    # Apply smoothing if enabled
    if apply_smoothing and GR_SMOOTHING_WINDOW >= 3:
        result_gr = apply_gr_smoothing(result_gr)
        logger.debug(f"Applied Savitzky-Golay smoothing (window={GR_SMOOTHING_WINDOW}, order={GR_SMOOTHING_ORDER})")

    return result_tvd, result_gr


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
