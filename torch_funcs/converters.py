"""
Converters between numpy arrays and PyTorch tensors.

All tensors use float64 for numerical precision matching with numpy.
Device can be 'cpu' or 'cuda' for GPU acceleration.
"""
import torch
import numpy as np


def numpy_to_torch(data_dict, device='cuda'):
    """
    Convert numpy data dict to torch tensors dict.

    Args:
        data_dict: dict with numpy arrays (well_data, typewell_data, or segments)
        device: 'cpu' or 'cuda'

    Returns:
        dict with torch tensors on specified device
    """
    result = {}

    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            result[key] = torch.tensor(value, dtype=torch.float64, device=device)
        elif isinstance(value, (int, float, bool)):
            result[key] = value  # Keep scalars as Python types
        else:
            result[key] = value  # Keep other types unchanged

    return result


def torch_to_numpy(data_dict):
    """
    Convert torch tensors dict back to numpy arrays dict.

    Args:
        data_dict: dict with torch tensors

    Returns:
        dict with numpy arrays
    """
    result = {}

    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.cpu().numpy()
        else:
            result[key] = value

    return result


def segments_numpy_to_torch(segments_np, device='cuda'):
    """
    Convert segments numpy array to torch tensor.

    Args:
        segments_np: (K, 6) numpy array
        device: 'cpu' or 'cuda'

    Returns:
        (K, 6) torch tensor
    """
    return torch.tensor(segments_np, dtype=torch.float64, device=device)


def update_segments_with_shifts_torch(shifts, segments_torch):
    """
    Update segments tensor with new shift values.
    Torch version of numpy_to_segments_data.

    Args:
        shifts: tensor of new end_shift values (K,) or (batch, K)
        segments_torch: original segments tensor (K, 6)

    Returns:
        new segments tensor (K, 6) or (batch, K, 6) with updated shifts

    Columns: [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]
    """
    K = segments_torch.shape[0]

    # Handle batch dimension
    if shifts.dim() == 1:
        # Single evaluation
        new_segments = segments_torch.clone()
        new_segments[:, 5] = shifts[:K]  # end_shift
        new_segments[1:, 4] = shifts[:K-1]  # next segment's start_shift
    else:
        # Batch evaluation: shifts is (batch, K+)
        batch_size = shifts.shape[0]
        # Expand segments to batch dimension
        new_segments = segments_torch.unsqueeze(0).expand(batch_size, -1, -1).clone()
        new_segments[:, :, 5] = shifts[:, :K]  # end_shift for all batches
        new_segments[:, 1:, 4] = shifts[:, :K-1]  # next segment's start_shift

    return new_segments


def calc_segment_angles_torch(segments_torch):
    """
    Calculate angles for all segments using torch.

    Args:
        segments_torch: (K, 6) or (batch, K, 6) tensor

    Returns:
        angles: (K,) or (batch, K) tensor in degrees
    """
    if segments_torch.dim() == 2:
        # Single evaluation
        delta_shift = segments_torch[:, 5] - segments_torch[:, 4]
        delta_vs = segments_torch[:, 3] - segments_torch[:, 2]
    else:
        # Batch evaluation
        delta_shift = segments_torch[:, :, 5] - segments_torch[:, :, 4]
        delta_vs = segments_torch[:, :, 3] - segments_torch[:, :, 2]

    angles = torch.rad2deg(torch.atan(delta_shift / delta_vs))
    return angles
