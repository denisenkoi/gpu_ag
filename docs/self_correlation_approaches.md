# Self-Correlation: Vectorizable Approaches

## Problem Statement

For different interpretations, each point has different TVT but the same gamma value.
Need to measure how well curves "overlay" on each other along the TVT axis.

**Current algorithm (sequential, slow):**
1. Find intersections of curve with target values
2. Interpolate TVT at intersection points
3. Group TVT values by proximity (threshold-based)
4. Count groups - more groups = worse self-correlation

**Problem:** Dynamic lists, sorting, grouping - hard to vectorize.

**Performance target:** 3-5x faster than current sequential version for batch of 500.

---

## Current Formula

```python
intersections_component = sc_power ** intersections_count  # 1.15^count
metric = (pearson^2 * mse^0.001 / intersections_component) * (1 + penalties)
```

Where:
- `sc_power = 1.15` (default)
- `intersections_count` = number of unique TVT groups where curve crosses target values

Effect:
- count=0 → divide by 1 (no effect)
- count=10 → divide by ~4
- count=20 → divide by ~16

---

## Vectorizable Approaches

### 1. Binning + Histogram Correlation

**Idea:** Discretize TVT axis into bins, accumulate gamma values.

**Algorithm:**
1. Define TVT bins (e.g., 100 bins covering TVT range)
2. For each point, assign to bin based on TVT
3. Accumulate gamma values in each bin using `scatter_add`
4. Compare gamma distributions between different curve sections
5. Metric: variance of gamma within bins (lower = better overlay)

**Pros:**
- Fully vectorizable via `torch.scatter_add`
- O(N) complexity
- Naturally handles batch dimension

**Cons:**
- Bin size affects precision
- May miss fine-grained structure

**Implementation complexity:** Low

---

### 2. KDE (Kernel Density) Overlap

**Idea:** Build 2D density on (TVT, gamma) plane, measure overlap.

**Algorithm:**
1. For each curve section, build 2D KDE on (TVT, gamma) grid
2. Use Gaussian kernel: `K(x) = exp(-x^2 / 2sigma^2)`
3. Convolve point cloud with kernel to get density
4. Measure overlap: `integral(density1 * density2)`

**Pros:**
- Smooth, differentiable metric
- Captures continuous structure

**Cons:**
- Requires grid definition
- Computationally heavier (convolutions)

**Implementation complexity:** Medium

---

### 3. Nearest Neighbor Distance in TVT Space

**Idea:** For points with similar TVT, compare their gamma values.

**Algorithm:**
1. For each point in section A, find nearest point by TVT in section B
2. Compute |gamma_A - gamma_B| for matched pairs
3. Metric: mean absolute gamma difference for TVT-matched pairs

**Pros:**
- Directly measures "overlay" quality
- Intuitive interpretation

**Cons:**
- `cdist` is O(N*M) - expensive for large N
- Need to handle unequal section sizes

**Implementation complexity:** Medium

**Vectorization:**
```python
# Compute all pairwise TVT distances
tvt_dist = torch.cdist(tvt_A.unsqueeze(-1), tvt_B.unsqueeze(-1))  # (N, M)
# Find nearest neighbors
nearest_idx = torch.argmin(tvt_dist, dim=1)  # (N,)
# Get gamma difference
gamma_diff = torch.abs(gamma_A - gamma_B[nearest_idx])
metric = gamma_diff.mean()
```

---

### 4. Sorted Interpolation Correlation

**Idea:** Interpolate one curve onto another's TVT grid, then correlate.

**Algorithm:**
1. Sort both sections by TVT
2. Use `searchsorted` to find interpolation positions
3. Linear interpolate gamma from section B onto TVT grid of section A
4. Compute Pearson correlation between gamma_A and interpolated gamma_B

**Pros:**
- Uses established Pearson metric
- Handles irregular TVT spacing

**Cons:**
- Requires sorting (but O(N log N))
- Interpolation at boundaries needs care

**Implementation complexity:** Medium

**Vectorization:**
```python
# Sort by TVT
idx_A = torch.argsort(tvt_A)
idx_B = torch.argsort(tvt_B)
# Find interpolation positions
positions = torch.searchsorted(tvt_B[idx_B], tvt_A[idx_A])
# Linear interpolation
gamma_interp = torch.lerp(gamma_B[idx_B[positions-1]],
                          gamma_B[idx_B[positions]],
                          weights)
# Pearson correlation
metric = pearson(gamma_A[idx_A], gamma_interp)
```

---

### 5. Fourier in TVT Space

**Idea:** Resample to uniform TVT grid, compare frequency content.

**Algorithm:**
1. Define uniform TVT grid
2. Interpolate gamma onto uniform grid for each section
3. Compute FFT of gamma signal
4. Compare spectra (magnitude and/or phase)
5. Metric: spectral coherence or cross-correlation

**Pros:**
- Captures periodic patterns
- Efficient via FFT

**Cons:**
- Requires uniform resampling
- May lose localized features

**Implementation complexity:** High

---

## Comparison Table

| Approach | Vectorizable | Complexity | Accuracy | Implementation |
|----------|--------------|------------|----------|----------------|
| Binning + Histogram | Yes | O(N) | Medium | Low |
| KDE Overlap | Yes | O(N*G) | High | Medium |
| Nearest Neighbor | Yes | O(N*M) | High | Medium |
| Sorted Interpolation | Yes | O(N log N) | High | Medium |
| Fourier | Yes | O(N log N) | Medium | High |

**Recommended for first attempt:** Binning + Histogram (simplest, fastest)

**Recommended for accuracy:** Nearest Neighbor or Sorted Interpolation

---

## Next Steps

1. Disable self-correlation for initial GPU optimization
2. Benchmark GPU version without self-correlation
3. Implement simplest vectorizable approach (Binning)
4. Validate against CPU results
5. Tune for 3-5x speedup target

---

## Notes

- Current CPU self-correlation takes ~5-10% of total objective function time
- For batch of 500, sequential version takes ~70 sec (main bottleneck)
- Target: <15 sec for batch of 500 with vectorized self-correlation
