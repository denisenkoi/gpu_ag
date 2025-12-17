"""
PyTorch implementation of correlation metrics.
Supports both single and batch evaluation.
"""
import torch


def pearson_torch(x, y):
    """
    Calculate Pearson correlation coefficient.
    Single evaluation version.

    Args:
        x, y: 1D tensors of same length

    Returns:
        float: correlation coefficient [-1, 1]
    """
    n = x.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    x_std = torch.std(x)
    y_std = torch.std(y)

    if x_std == 0 or y_std == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    numerator = torch.sum((x - x_mean) * (y - y_mean))
    denominator = torch.sqrt(torch.sum((x - x_mean) ** 2) * torch.sum((y - y_mean) ** 2))

    if denominator == 0:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)

    return numerator / denominator


def mse_torch(x, y):
    """
    Calculate Mean Squared Error.
    Single evaluation version.

    Args:
        x, y: 1D tensors of same length

    Returns:
        tensor: MSE value
    """
    return torch.mean((x - y) ** 2)


def pearson_batch_torch(x, y):
    """
    Calculate Pearson correlation for batch.

    Args:
        x, y: 2D tensors (batch, N)

    Returns:
        tensor (batch,): correlation coefficients
    """
    batch_size = x.shape[0]
    device = x.device
    dtype = x.dtype

    # Mean along N dimension
    x_mean = torch.mean(x, dim=1, keepdim=True)
    y_mean = torch.mean(y, dim=1, keepdim=True)

    # Centered values
    x_centered = x - x_mean
    y_centered = y - y_mean

    # Numerator: sum of products
    numerator = torch.sum(x_centered * y_centered, dim=1)

    # Denominator: sqrt of sum of squares product
    x_sq_sum = torch.sum(x_centered ** 2, dim=1)
    y_sq_sum = torch.sum(y_centered ** 2, dim=1)
    denominator = torch.sqrt(x_sq_sum * y_sq_sum)

    # Handle zero denominator
    result = torch.where(
        denominator > 0,
        numerator / denominator,
        torch.zeros(batch_size, device=device, dtype=dtype)
    )

    return result


def mse_batch_torch(x, y):
    """
    Calculate Mean Squared Error for batch.

    Args:
        x, y: 2D tensors (batch, N)

    Returns:
        tensor (batch,): MSE values
    """
    return torch.mean((x - y) ** 2, dim=1)
