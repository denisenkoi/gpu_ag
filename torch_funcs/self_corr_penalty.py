"""
GPU-friendly self-correlation penalty using TVT bins.

Computes std(bin_means) - standard deviation of mean GR per TVT bin.
Lower value = more "suspicious" (possible false correlation).

Uses scatter_add for GPU-efficient binning.
"""
import torch
from typing import Optional, Tuple


def compute_bin_std_batch(
    tvt_batch: torch.Tensor,  # (batch_size, n_points)
    gr_batch: torch.Tensor,   # (batch_size, n_points)
    bin_size: float = 0.10,   # 10cm bins
    min_bins: int = 5,        # minimum bins for valid std
) -> torch.Tensor:
    """
    Compute std of bin means for batch of solutions.

    Args:
        tvt_batch: TVT values for each solution (batch_size, n_points)
        gr_batch: GR values for each solution (batch_size, n_points)
        bin_size: Size of TVT bins in meters
        min_bins: Minimum number of non-empty bins required

    Returns:
        std_batch: (batch_size,) std of bin means for each solution
                   Returns 0.0 for solutions with insufficient bins
    """
    batch_size, n_points = tvt_batch.shape
    device = tvt_batch.device

    # Find global bin range across all batches
    tvt_min = tvt_batch.min()
    tvt_max = tvt_batch.max()
    n_bins = int((tvt_max - tvt_min) / bin_size) + 1

    # Compute bin indices: (batch_size, n_points)
    bin_idx = ((tvt_batch - tvt_min) / bin_size).long()
    bin_idx = bin_idx.clamp(0, n_bins - 1)

    # Flatten for scatter_add
    # We need to handle each batch separately, so offset bin indices
    batch_offsets = torch.arange(batch_size, device=device).unsqueeze(1) * n_bins
    flat_bin_idx = (bin_idx + batch_offsets).flatten()  # (batch_size * n_points)
    flat_gr = gr_batch.flatten()  # (batch_size * n_points)

    # Allocate bin sums and counts
    total_bins = batch_size * n_bins
    bin_sums = torch.zeros(total_bins, device=device, dtype=gr_batch.dtype)
    bin_counts = torch.zeros(total_bins, device=device, dtype=gr_batch.dtype)

    # Scatter add
    bin_sums.scatter_add_(0, flat_bin_idx, flat_gr)
    bin_counts.scatter_add_(0, flat_bin_idx, torch.ones_like(flat_gr))

    # Reshape back to (batch_size, n_bins)
    bin_sums = bin_sums.view(batch_size, n_bins)
    bin_counts = bin_counts.view(batch_size, n_bins)

    # Compute mean per bin (avoid div by zero)
    bin_means = bin_sums / bin_counts.clamp(min=1)

    # Compute std of means for each batch
    # Only consider non-empty bins
    std_batch = torch.zeros(batch_size, device=device, dtype=gr_batch.dtype)

    for i in range(batch_size):
        valid_mask = bin_counts[i] > 0
        n_valid = valid_mask.sum()
        if n_valid >= min_bins:
            valid_means = bin_means[i][valid_mask]
            std_batch[i] = valid_means.std()

    return std_batch


def compute_bin_std_incremental(
    prev_bin_sums: torch.Tensor,    # (n_bins,)
    prev_bin_counts: torch.Tensor,  # (n_bins,)
    new_tvt: torch.Tensor,          # (n_new_points,)
    new_gr: torch.Tensor,           # (n_new_points,)
    tvt_min: float,
    bin_size: float = 0.10,
    min_bins: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """
    Incrementally update bin statistics and compute std.

    For use when processing segments sequentially - reuse previous bins.

    Args:
        prev_bin_sums: Sum of GR per bin from previous segments
        prev_bin_counts: Count per bin from previous segments
        new_tvt: TVT values for new segment
        new_gr: GR values for new segment
        tvt_min: Minimum TVT (for consistent binning)
        bin_size: Bin size in meters
        min_bins: Minimum bins for valid std

    Returns:
        (updated_sums, updated_counts, std_value)
    """
    device = new_tvt.device
    n_bins = prev_bin_sums.shape[0]

    # Compute bin indices for new points
    bin_idx = ((new_tvt - tvt_min) / bin_size).long()
    bin_idx = bin_idx.clamp(0, n_bins - 1)

    # Update sums and counts
    updated_sums = prev_bin_sums.clone()
    updated_counts = prev_bin_counts.clone()

    updated_sums.scatter_add_(0, bin_idx, new_gr)
    updated_counts.scatter_add_(0, bin_idx, torch.ones_like(new_gr))

    # Compute std of means
    valid_mask = updated_counts > 0
    n_valid = valid_mask.sum()

    if n_valid >= min_bins:
        bin_means = updated_sums[valid_mask] / updated_counts[valid_mask]
        std_value = float(bin_means.std())
    else:
        std_value = 0.0

    return updated_sums, updated_counts, std_value


def self_corr_penalty(
    std_bin_means: torch.Tensor,  # (batch_size,)
    threshold: float = 8.0,       # suspicious if below this
    weight: float = 0.1,          # penalty weight
) -> torch.Tensor:
    """
    Compute penalty for suspiciously low self-correlation std.

    Lower std = more "consistent" TVT->GR mapping = possibly false correlation.

    Args:
        std_bin_means: std of bin means for each solution
        threshold: Values below this are penalized
        weight: Penalty multiplier

    Returns:
        penalty: (batch_size,) penalty to ADD to loss (higher = worse)
    """
    # Penalty increases as std decreases below threshold
    penalty = torch.clamp(threshold - std_bin_means, min=0) * weight
    return penalty


# Test
if __name__ == '__main__':
    print("Testing GPU-friendly bin std computation...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Simulate batch of solutions
    batch_size = 100
    n_points = 5000

    # Random TVT (range ~10m)
    tvt_batch = torch.randn(batch_size, n_points, device=device) * 3 + 50

    # Random GR (range ~100)
    gr_batch = torch.randn(batch_size, n_points, device=device) * 30 + 50

    # Compute std
    import time
    start = time.time()
    std_batch = compute_bin_std_batch(tvt_batch, gr_batch, bin_size=0.10)
    torch.cuda.synchronize() if device == 'cuda' else None
    elapsed = time.time() - start

    print(f"Batch size: {batch_size}, Points: {n_points}")
    print(f"Time: {elapsed*1000:.2f}ms")
    print(f"Std range: {std_batch.min():.2f} - {std_batch.max():.2f}")
    print(f"Std mean: {std_batch.mean():.2f}")

    # Test penalty
    penalty = self_corr_penalty(std_batch, threshold=8.0, weight=0.1)
    print(f"Penalty range: {penalty.min():.3f} - {penalty.max():.3f}")
    print(f"Non-zero penalties: {(penalty > 0).sum()}/{batch_size}")
