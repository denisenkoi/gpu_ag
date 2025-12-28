"""
GR utility functions - shared filtering logic.
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
