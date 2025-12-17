"""
Numpy implementation of correlation metrics.
"""
import numpy as np


def pearson_numpy(x, y):
    """
    Calculate Pearson correlation coefficient.

    Args:
        x, y: 1D arrays of same length

    Returns:
        float: correlation coefficient [-1, 1]
    """
    n = len(x)
    if n < 2:
        return 0.0

    # Check for constant arrays
    x_std = np.std(x)
    y_std = np.std(y)

    if x_std == 0 or y_std == 0:
        return 0.0

    # Calculate Pearson correlation
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean) ** 2) * np.sum((y - y_mean) ** 2))

    if denominator == 0:
        return 0.0

    return numerator / denominator


def mse_numpy(x, y):
    """
    Calculate Mean Squared Error.

    Args:
        x, y: 1D arrays of same length

    Returns:
        float: MSE value
    """
    return np.mean((x - y) ** 2)
