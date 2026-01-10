#!/usr/bin/env python3
"""
Analyze TypeLog vs PseudoTypeLog correlation on different segments.

Checks if there's a segment >50m with good correlation (>0.7).
Hypothesis: geologist interpretation may be careless above some point.
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from preprocessing import compute_overlap_metrics


def compute_segment_correlation(
    type_tvd: np.ndarray, type_gr: np.ndarray,
    pseudo_tvd: np.ndarray, pseudo_gr: np.ndarray,
    last_n_meters: float
) -> dict:
    """
    Compute correlation on last N meters of overlap zone.

    Args:
        type_tvd, type_gr: TypeLog data
        pseudo_tvd, pseudo_gr: PseudoTypeLog data
        last_n_meters: Take last N meters from overlap zone end

    Returns:
        Dict with pearson, rmse, segment_length
    """
    # Find overlap zone
    overlap_start = max(type_tvd.min(), pseudo_tvd.min())
    overlap_end = min(type_tvd.max(), pseudo_tvd.max())

    if overlap_end <= overlap_start:
        return {'pearson': np.nan, 'rmse': np.nan, 'segment_length': 0}

    # Take last N meters
    segment_start = max(overlap_start, overlap_end - last_n_meters)
    segment_end = overlap_end
    actual_length = segment_end - segment_start

    if actual_length < 10:  # Need at least 10m
        return {'pearson': np.nan, 'rmse': np.nan, 'segment_length': actual_length}

    # Interpolate to common grid
    step = 0.1
    common_tvd = np.arange(segment_start, segment_end, step)

    if len(common_tvd) < 20:
        return {'pearson': np.nan, 'rmse': np.nan, 'segment_length': actual_length}

    type_interp = np.interp(common_tvd, type_tvd, type_gr)
    pseudo_interp = np.interp(common_tvd, pseudo_tvd, pseudo_gr)

    # RMSE
    rmse = np.sqrt(np.mean((type_interp - pseudo_interp) ** 2))

    # Pearson
    type_centered = type_interp - type_interp.mean()
    pseudo_centered = pseudo_interp - pseudo_interp.mean()
    denom = np.sqrt((type_centered ** 2).sum() * (pseudo_centered ** 2).sum())
    pearson = (type_centered * pseudo_centered).sum() / denom if denom > 1e-10 else 0.0

    return {
        'pearson': float(pearson),
        'rmse': float(rmse),
        'segment_length': float(actual_length),
        'segment_start': float(segment_start),
        'segment_end': float(segment_end)
    }


def analyze_well(data: dict, well_name: str, segments: list) -> dict:
    """Analyze one well on multiple segments."""
    # Get data
    pseudo_tvd = data['pseudo_tvd'].cpu().numpy() if hasattr(data['pseudo_tvd'], 'cpu') else np.array(data['pseudo_tvd'])
    pseudo_gr = data['pseudo_gr'].cpu().numpy() if hasattr(data['pseudo_gr'], 'cpu') else np.array(data['pseudo_gr'])
    type_tvd = data['type_tvd'].cpu().numpy() if hasattr(data['type_tvd'], 'cpu') else np.array(data['type_tvd'])
    type_gr = data['type_gr'].cpu().numpy() if hasattr(data['type_gr'], 'cpu') else np.array(data['type_gr'])

    # Apply norm_multiplier to pseudo
    norm_mult = data.get('norm_multiplier', 1.0)
    if hasattr(norm_mult, 'item'):
        norm_mult = norm_mult.item()
    pseudo_gr = pseudo_gr * norm_mult

    # Full overlap
    full = compute_overlap_metrics(type_tvd, type_gr, pseudo_tvd, pseudo_gr)

    # Segment correlations
    results = {
        'well': well_name,
        'overlap_length': full['overlap_length'],
        'full_pearson': full['pearson'],
        'full_rmse': full['rmse'],
        'segments': {}
    }

    for seg_len in segments:
        seg_result = compute_segment_correlation(
            type_tvd, type_gr, pseudo_tvd, pseudo_gr, seg_len
        )
        results['segments'][seg_len] = seg_result

    return results


def main():
    # Load dataset
    dataset_path = Path('/mnt/e/Projects/Rogii/gpu_ag/data/wells_limited_pseudo.pt')
    if not dataset_path.exists():
        dataset_path = Path('/mnt/e/Projects/Rogii/gpu_ag/data/wells_fresh.pt')

    print(f"Loading: {dataset_path}")
    dataset = torch.load(dataset_path, weights_only=False)

    # Segments to check (from end of overlap)
    segments = [50, 60, 70, 80, 90, 100, 120, 150, 200]

    all_results = []

    for well_name, data in sorted(dataset.items()):
        result = analyze_well(data, well_name, segments)
        all_results.append(result)

    # Print header
    print(f"\n{'Well':<25} {'Overlap':>7} {'Full':>6}", end='')
    for s in segments:
        print(f" {s:>5}m", end='')
    print()
    print("-" * 100)

    # Print results
    good_segments = {s: 0 for s in segments}  # Count wells with pearson > 0.7

    for r in all_results:
        print(f"{r['well']:<25} {r['overlap_length']:>6.0f}m {r['full_pearson']:>6.3f}", end='')
        for s in segments:
            p = r['segments'][s]['pearson']
            if np.isnan(p):
                print("    --", end='')
            else:
                print(f" {p:>5.3f}", end='')
                if p >= 0.7:
                    good_segments[s] += 1
        print()

    # Summary
    n_wells = len(all_results)
    print("\n" + "=" * 100)
    print("SUMMARY: Wells with Pearson >= 0.7 on each segment")
    print("-" * 100)

    # Average Pearson per segment
    print(f"\n{'Segment':<10} {'Count ≥0.7':>12} {'Percent':>10} {'Avg Pearson':>12} {'Best':>8}")
    print("-" * 60)

    for s in segments:
        pearsons = [r['segments'][s]['pearson'] for r in all_results if not np.isnan(r['segments'][s]['pearson'])]
        if pearsons:
            avg_p = np.mean(pearsons)
            best_p = np.max(pearsons)
            count = good_segments[s]
            pct = 100 * count / len(pearsons)
            print(f"{s:>6}m    {count:>6}/{len(pearsons):<5} {pct:>8.1f}%   {avg_p:>10.3f}   {best_p:>7.3f}")

    # Find wells where shorter segment has BETTER correlation than full
    print("\n" + "=" * 100)
    print("Wells where segment correlation > full correlation (by >0.1):")
    print("-" * 100)

    for r in all_results:
        full_p = r['full_pearson']
        if np.isnan(full_p):
            continue

        for s in segments:
            seg_p = r['segments'][s]['pearson']
            if not np.isnan(seg_p) and seg_p > full_p + 0.1:
                print(f"  {r['well']:<25}: {s}m segment {seg_p:.3f} > full {full_p:.3f} (delta +{seg_p - full_p:.3f})")
                break  # Only show first improvement

    # Find best 10 wells by any segment correlation
    print("\n" + "=" * 100)
    print("TOP-10 by best segment Pearson:")
    print("-" * 100)

    best_by_well = []
    for r in all_results:
        best_seg = None
        best_p = r['full_pearson']
        for s in segments:
            p = r['segments'][s]['pearson']
            if not np.isnan(p) and p > best_p:
                best_p = p
                best_seg = s
        best_by_well.append((r['well'], best_p, best_seg, r['full_pearson']))

    best_by_well.sort(key=lambda x: -x[1] if not np.isnan(x[1]) else -999)

    for well, best_p, best_seg, full_p in best_by_well[:10]:
        seg_info = f"({best_seg}m)" if best_seg else "(full)"
        print(f"  {well:<25}: {best_p:.3f} {seg_info:<10} (full={full_p:.3f})")


if __name__ == '__main__':
    main()
