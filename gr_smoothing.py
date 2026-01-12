"""
GR Smoothing Module - Savitzky-Golay filtering for gamma ray signals.

Functions:
    calc_shit_score(): Analyze GR signal quality (quantization, steps, repeats)
    get_adaptive_window(): Select smoothing window (5/11/15) based on quality score
    smooth_gr(): Apply Savitzky-Golay filter with specified or auto window
    apply_gr_smoothing(): Legacy function using .env settings
"""

import os
import numpy as np
from scipy.signal import savgol_filter
from dotenv import load_dotenv

# Load .env on import
load_dotenv()


def apply_gr_smoothing(values: np.ndarray) -> np.ndarray:
    """
    Apply GR smoothing based on .env settings.

    Environment variables:
        GR_SMOOTHING_WINDOW: Savitzky-Golay window size (must be >= 3, odd). 0 = no smoothing.
        GR_SMOOTHING_ORDER: Polynomial order (default: 2)

    Returns:
        Smoothed values array (or original if smoothing disabled)
    """
    window = int(os.getenv('GR_SMOOTHING_WINDOW', '0'))
    order = int(os.getenv('GR_SMOOTHING_ORDER', '2'))

    if window < 3:
        return values

    # Window must be odd for savgol_filter
    if window % 2 == 0:
        window += 1

    # Order must be less than window
    if order >= window:
        order = window - 1

    return savgol_filter(values, window, order)


def get_smoothing_params() -> dict:
    """Return current smoothing parameters for logging."""
    window = int(os.getenv('GR_SMOOTHING_WINDOW', '0'))
    order = int(os.getenv('GR_SMOOTHING_ORDER', '2'))
    return {
        'window': window,
        'order': order,
        'enabled': window >= 3
    }


def calc_shit_score(gr_values: np.ndarray) -> float:
    """
    Calculate GR signal quality score (higher = worse quality = more smoothing needed).

    Components:
    - repeats: ratio of consecutive identical values
    - linear: ratio of points on linear segments (2nd derivative ~0)
    - quant: quantization level (few unique values)
    - steps: ratio of large jumps

    Returns:
        Score from 0 (good quality) to 1 (bad quality / needs smoothing)
    """
    gr = np.array(gr_values)

    # 1. Repeats ratio
    diffs = np.diff(gr)
    repeats_ratio = np.sum(diffs == 0) / len(diffs) if len(diffs) > 0 else 0

    # 2. Linear interpolation (2nd derivative near zero)
    if len(gr) > 2:
        d2 = np.diff(gr, n=2)
        threshold = np.std(d2) * 0.1 if np.std(d2) > 0 else 0.01
        linear_ratio = np.sum(np.abs(d2) < threshold) / len(d2)
    else:
        linear_ratio = 0

    # 3. Quantization - few unique values
    unique_count = len(np.unique(np.round(gr, 1)))
    value_range = gr.max() - gr.min()
    expected_count = max(value_range / 0.1, 1)
    quant_score = max(0, 1 - unique_count / expected_count)

    # 4. Step jumps
    if len(diffs) > 0 and np.std(diffs) > 0:
        big_jumps = np.abs(diffs) > np.std(diffs) * 2
        step_ratio = np.sum(big_jumps) / len(diffs)
    else:
        step_ratio = 0

    return 0.4 * repeats_ratio + 0.3 * linear_ratio + 0.2 * quant_score + 0.1 * step_ratio


def get_adaptive_window(gr_values: np.ndarray) -> int:
    """
    Get adaptive smoothing window based on GR signal quality.

    Thresholds:
    - shit_score > 0.4: window=15 (heavily quantized)
    - shit_score > 0.25: window=11 (moderately quantized)
    - else: window=5 (good quality)

    Returns:
        Recommended Savitzky-Golay window size
    """
    score = calc_shit_score(gr_values)

    if score > 0.4:
        return 15
    elif score > 0.25:
        return 11
    else:
        return 5


def smooth_gr(values: np.ndarray, window: int = None, order: int = 2) -> np.ndarray:
    """
    Apply Savitzky-Golay smoothing to GR values.

    Args:
        values: GR array to smooth
        window: Window size (must be odd, >= 3). If None, uses adaptive window.
        order: Polynomial order (default 2)

    Returns:
        Smoothed GR array
    """
    if window is None:
        window = get_adaptive_window(values)

    if window < 3:
        return values

    # Window must be odd
    if window % 2 == 0:
        window += 1

    # Order must be less than window
    if order >= window:
        order = window - 1

    return savgol_filter(values, window, order)


def apply_gr_smoothing_adaptive(values: np.ndarray, order: int = 2) -> tuple:
    """
    Apply adaptive GR smoothing based on signal quality.

    Returns:
        (smoothed_values, window_used, shit_score)
    """
    score = calc_shit_score(values)
    window = get_adaptive_window(values)
    smoothed = smooth_gr(values, window, order)
    return smoothed, window, score
