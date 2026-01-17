#!/usr/bin/env python3
"""
Fast vectorized self-correlation metrics.

Usage:
    python -m self_correlation.fast_metrics --run-id 20260115_210742
    python -m self_correlation.fast_metrics --run-id 20260115_210742 --compare-ref
"""
import argparse
import numpy as np
import torch
import psycopg2
from scipy import stats
from scipy.signal import savgol_filter
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Load dataset once
_ds = None
def get_dataset():
    global _ds
    if _ds is None:
        _ds = torch.load('data/wells_limited_pseudo.pt', weights_only=False)
    return _ds


def compute_tvt_fast(
    well_name: str,
    interpretations: List[Dict],
    md_start_filter: Optional[float] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute TVT, GR, MD arrays from interpretation.

    Args:
        well_name: Well name
        interpretations: List of interpretation segments
        md_start_filter: Only use points with MD >= this value (e.g., landing_end_dls)
    """
    ds = get_dataset()
    if well_name not in ds:
        return None, None, None

    well_data = ds[well_name]
    well_md = well_data['well_md'].numpy()
    well_tvd = well_data['well_tvd'].numpy()
    log_md = well_data['log_md'].numpy()
    log_gr = well_data['log_gr'].numpy()
    well_gr = np.interp(well_md, log_md, log_gr)

    tvt = np.full_like(well_tvd, np.nan)

    for interp in interpretations:
        md_start, md_end = interp['md_start'], interp['md_end']
        start_shift, end_shift = interp['start_shift'], interp['end_shift']
        mask = (well_md >= md_start) & (well_md <= md_end)
        if not mask.any():
            continue
        seg_md = well_md[mask]
        seg_tvd = well_tvd[mask]
        if md_end > md_start:
            ratio = (seg_md - md_start) / (md_end - md_start)
        else:
            ratio = np.zeros_like(seg_md)
        seg_shift = start_shift + ratio * (end_shift - start_shift)
        tvt[mask] = seg_tvd - seg_shift

    valid = ~np.isnan(tvt)

    # Apply MD filter if specified
    if md_start_filter is not None:
        valid = valid & (well_md >= md_start_filter)

    if valid.sum() < 10:
        return None, None, None
    return tvt[valid], well_gr[valid], well_md[valid]


def all_metrics_fast(
    well_name: str,
    interpretations: List[Dict],
    md_start_filter: Optional[float] = None
) -> Dict:
    """
    Compute ALL self-correlation metrics in one pass.
    Returns dict with all metrics.
    """
    tvt, gr, md = compute_tvt_fast(well_name, interpretations, md_start_filter)
    if tvt is None:
        return {'error': 'no_data'}

    result = {'n_points': len(tvt), 'md_range': float(md.max() - md.min())}

    # 1. TVT Variance (5cm bins)
    bin_size = 0.05
    tvt_min = tvt.min()
    bin_idx = ((tvt - tvt_min) / bin_size).astype(int)

    bins = defaultdict(list)
    for i, bi in enumerate(bin_idx):
        bins[bi].append(gr[i])

    variances = []
    weights = []
    for bi, gr_vals in bins.items():
        if len(gr_vals) >= 2:
            variances.append(np.var(gr_vals))
            weights.append(len(gr_vals))

    if variances:
        result['variance_5cm'] = float(np.average(variances, weights=weights))
        result['n_overlaps_5cm'] = len(variances)
    else:
        result['variance_5cm'] = None
        result['n_overlaps_5cm'] = 0

    # 2. TVT Variance (10cm bins)
    bin_size = 0.1
    bin_idx = ((tvt - tvt_min) / bin_size).astype(int)
    bins10 = defaultdict(list)
    for i, bi in enumerate(bin_idx):
        bins10[bi].append(gr[i])
    variances = []
    weights = []
    for bi, gr_vals in bins10.items():
        if len(gr_vals) >= 2:
            variances.append(np.var(gr_vals))
            weights.append(len(gr_vals))
    if variances:
        result['variance_10cm'] = float(np.average(variances, weights=weights))
    else:
        result['variance_10cm'] = None

    # 3. RMSE to nearest (vectorized)
    n = len(tvt)
    if n > 1000:
        idx = np.random.choice(n, 1000, replace=False)
        tvt_s, gr_s, md_s = tvt[idx], gr[idx], md[idx]
    else:
        tvt_s, gr_s, md_s = tvt, gr, md

    # Compute pairwise distances
    tvt_diff = np.abs(tvt_s[:, None] - tvt_s[None, :])
    md_diff = np.abs(md_s[:, None] - md_s[None, :])
    gr_diff = np.abs(gr_s[:, None] - gr_s[None, :])

    # Valid pairs: close in TVT, far in MD
    valid_pairs = (tvt_diff < 0.5) & (md_diff > 30)
    np.fill_diagonal(valid_pairs, False)

    if valid_pairs.sum() > 0:
        rmse = np.sqrt(np.mean(gr_diff[valid_pairs] ** 2))
        result['rmse_tvt05'] = float(rmse)
        result['n_pairs'] = int(valid_pairs.sum() // 2)
    else:
        result['rmse_tvt05'] = None
        result['n_pairs'] = 0

    # 4. Roughness (derivative variance)
    sort_idx = np.argsort(tvt)
    tvt_sorted = tvt[sort_idx]
    gr_sorted = gr[sort_idx]

    d_tvt = np.diff(tvt_sorted)
    d_gr = np.diff(gr_sorted)

    valid_d = d_tvt > 0.01
    if valid_d.sum() > 10:
        derivatives = d_gr[valid_d] / d_tvt[valid_d]
        result['roughness'] = float(np.std(derivatives))
        result['mean_deriv'] = float(np.mean(np.abs(derivatives)))
    else:
        result['roughness'] = None
        result['mean_deriv'] = None

    # 5. Spearman correlation
    if len(tvt) > 20:
        r_spearman, _ = stats.spearmanr(tvt, gr)
        result['spearman_tvt_gr'] = float(r_spearman)
    else:
        result['spearman_tvt_gr'] = None

    # 6. Entropy-like metric (10cm bins)
    n_bins = min(50, len(bins10))
    if n_bins > 5:
        bin_means = [np.mean(v) for v in bins10.values() if len(v) >= 1]
        if len(bin_means) > 5:
            result['gr_entropy_10cm'] = float(np.std(bin_means))
        else:
            result['gr_entropy_10cm'] = None
    else:
        result['gr_entropy_10cm'] = None

    # 6b. Entropy 5cm bins
    bin_means_5 = [np.mean(v) for v in bins.values() if len(v) >= 1]
    if len(bin_means_5) > 5:
        result['gr_entropy_5cm'] = float(np.std(bin_means_5))
    else:
        result['gr_entropy_5cm'] = None

    # 7. TVT range (coverage)
    result['tvt_range'] = float(tvt.max() - tvt.min())

    # 8. IQR metrics (new!) - measures "thickness of band" in each bin
    # Grid: bin_size 2,3,5,10 cm x smoothing (raw, savgol51)
    for bin_cm in [2, 3, 5, 10]:
        bin_size_m = bin_cm / 100.0
        for smooth_name, gr_data in [('raw', gr), ('s51', savgol_filter(gr, 51, 3) if len(gr) >= 51 else gr)]:
            bin_idx = ((tvt - tvt_min) / bin_size_m).astype(int)
            bins_iqr = defaultdict(list)
            for i, bi in enumerate(bin_idx):
                bins_iqr[bi].append(gr_data[i])

            iqrs = []
            outlier_counts = []
            total_points = 0

            for bi, gr_vals in bins_iqr.items():
                if len(gr_vals) >= 5:
                    q1, q3 = np.percentile(gr_vals, [25, 75])
                    iqr = q3 - q1
                    iqrs.append(iqr)

                    # Count outliers (beyond 1.5*IQR)
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = sum(1 for v in gr_vals if v < lower or v > upper)
                    outlier_counts.append(outliers)
                    total_points += len(gr_vals)

            key_prefix = f'iqr_{bin_cm}cm_{smooth_name}'
            if iqrs:
                result[f'{key_prefix}_mean'] = float(np.mean(iqrs))
                result[f'{key_prefix}_outliers'] = float(sum(outlier_counts) / total_points * 100) if total_points > 0 else 0.0
            else:
                result[f'{key_prefix}_mean'] = None
                result[f'{key_prefix}_outliers'] = None

    return result


def compute_shift_at_md(well_md: np.ndarray, interpretations: List[Dict]) -> np.ndarray:
    """Compute shift at each MD point from interpretation segments."""
    shift = np.full_like(well_md, np.nan, dtype=float)

    for interp in interpretations:
        md_start, md_end = interp['md_start'], interp['md_end']
        start_shift, end_shift = interp['start_shift'], interp['end_shift']
        mask = (well_md >= md_start) & (well_md <= md_end)
        if not mask.any():
            continue
        seg_md = well_md[mask]
        if md_end > md_start:
            ratio = (seg_md - md_start) / (md_end - md_start)
        else:
            ratio = np.zeros_like(seg_md)
        shift[mask] = start_shift + ratio * (end_shift - start_shift)

    return shift


def compute_shift_rmse(well_name: str, opt_interps: List[Dict], ref_interps: List[Dict],
                       md_step: float = 1.0, md_filter: Optional[float] = None) -> Optional[float]:
    """Compute RMSE of shift difference between OPT and REF at regular MD intervals."""
    ds = get_dataset()
    if well_name not in ds:
        return None

    well_md = ds[well_name]['well_md'].numpy()

    # Create regular MD grid
    md_min = max(well_md.min(), md_filter or 0)
    md_max = well_md.max()
    md_grid = np.arange(md_min, md_max, md_step)

    if len(md_grid) < 10:
        return None

    opt_shift = compute_shift_at_md(md_grid, opt_interps)
    ref_shift = compute_shift_at_md(md_grid, ref_interps)

    # Only compare where both have values
    valid = ~np.isnan(opt_shift) & ~np.isnan(ref_shift)
    if valid.sum() < 10:
        return None

    diff = opt_shift[valid] - ref_shift[valid]
    return float(np.sqrt(np.mean(diff ** 2)))


def run_analysis(run_id: str, compare_ref: bool = False, use_landing_filter: bool = True):
    """Analyze self-correlation metrics for a run."""
    conn = psycopg2.connect(dbname='gpu_ag', user='rogii', password='rogii123', host='localhost')
    cur = conn.cursor()

    # Get run info
    cur.execute("SELECT description, optimized_rmse FROM runs WHERE run_id = %s", (run_id,))
    row = cur.fetchone()
    if not row:
        print(f"Run {run_id} not found!")
        return
    desc, opt_rmse = row

    print("=" * 80)
    print(f"SELF-CORRELATION ANALYSIS")
    print("=" * 80)
    print(f"Run: {run_id}")
    print(f"Description: {desc}")
    print(f"Optimized RMSE: {opt_rmse:.2f}m" if opt_rmse else "Optimized RMSE: N/A")
    print(f"Landing filter: {'ON (after landing_end_dls)' if use_landing_filter else 'OFF'}")
    print()

    # Get wells
    cur.execute("SELECT DISTINCT well_name FROM interpretations WHERE run_id = %s", (run_id,))
    wells = [r[0] for r in cur.fetchall()]
    print(f"Wells: {len(wells)}")

    # Load dataset for landing_end_dls
    ds = get_dataset()

    run_ids = {'OPT': run_id}
    if compare_ref:
        run_ids['REF'] = 'REFERENCE'

    all_results = {algo: [] for algo in run_ids}
    all_interps = {}  # Store interpretations for shift_rmse calculation

    for algo, rid in run_ids.items():
        for well in wells:
            cur.execute("""
                SELECT md_start, md_end, start_shift, end_shift
                FROM interpretations WHERE run_id = %s AND well_name = %s ORDER BY seg_idx
            """, (rid, well))
            interps = [{'md_start': r[0], 'md_end': r[1], 'start_shift': r[2], 'end_shift': r[3]}
                       for r in cur.fetchall()]
            if not interps:
                continue

            # Store for shift_rmse
            if well not in all_interps:
                all_interps[well] = {}
            all_interps[well][algo] = interps

            # Get landing filter
            md_filter = None
            if use_landing_filter and well in ds:
                landing_end = ds[well].get('landing_end_dls')
                if landing_end is not None:
                    md_filter = float(landing_end)

            metrics = all_metrics_fast(well, interps, md_filter)
            if 'error' not in metrics:
                cur.execute("SELECT opt_error FROM well_results WHERE run_id = %s AND well_name = %s",
                           (rid, well))
                row = cur.fetchone()
                metrics['opt_error'] = row[0] if row and row[0] else None
                metrics['abs_error'] = abs(row[0]) if row and row[0] else None
                metrics['well'] = well
                metrics['md_filter'] = md_filter
                all_results[algo].append(metrics)

    # Compute shift_rmse for OPT vs REF
    if compare_ref:
        for r in all_results['OPT']:
            well = r['well']
            if well in all_interps and 'OPT' in all_interps[well] and 'REF' in all_interps[well]:
                shift_rmse = compute_shift_rmse(
                    well, all_interps[well]['OPT'], all_interps[well]['REF'],
                    md_step=1.0, md_filter=r.get('md_filter')
                )
                r['shift_rmse'] = shift_rmse

    conn.close()

    # Summary statistics
    print()
    print("=" * 80)
    print("METRICS SUMMARY (mean values)")
    print("=" * 80)

    metric_names = ['variance_5cm', 'variance_10cm', 'rmse_tvt05', 'roughness',
                    'mean_deriv', 'spearman_tvt_gr', 'gr_entropy_5cm', 'gr_entropy_10cm', 'tvt_range']

    # Add IQR metrics
    iqr_metrics = []
    for bin_cm in [2, 3, 5, 10]:
        for smooth in ['raw', 's51']:
            iqr_metrics.append(f'iqr_{bin_cm}cm_{smooth}_mean')
            iqr_metrics.append(f'iqr_{bin_cm}cm_{smooth}_outliers')
    metric_names.extend(iqr_metrics)

    for metric in metric_names:
        vals = [r[metric] for r in all_results['OPT'] if r.get(metric) is not None]
        if vals:
            print(f"{metric:20s}: mean={np.mean(vals):8.2f}, std={np.std(vals):8.2f}, n={len(vals)}")

    # Correlations with opt_error
    print()
    print("=" * 80)
    print("CORRELATIONS with |opt_error|")
    print("=" * 80)
    print(f"{'Metric':<20s} {'Pearson r':>10s} {'p-value':>10s} {'Useful?':>10s}")
    print("-" * 52)

    opt_results = all_results['OPT']

    for metric in metric_names:
        vals = []
        errs = []
        for r in opt_results:
            if r.get(metric) is not None and r.get('abs_error') is not None:
                vals.append(r[metric])
                errs.append(r['abs_error'])

        if len(vals) >= 20:
            r_val, p_val = stats.pearsonr(vals, errs)
            # Positive r = higher metric → higher error (bad metric predicts bad result)
            # Negative r = higher metric → lower error (good metric predicts good result)
            useful = ""
            if p_val < 0.05:
                if r_val > 0.2:
                    useful = "USEFUL+"
                elif r_val < -0.2:
                    useful = "USEFUL-"
            print(f"{metric:<20s} {r_val:>+10.3f} {p_val:>10.4f} {useful:>10s}")

    # Top 10 worst wells
    print()
    print("=" * 80)
    print("TOP 10 WORST WELLS (by |opt_error|)")
    print("=" * 80)

    sorted_results = sorted(
        [r for r in opt_results if r.get('abs_error') is not None],
        key=lambda x: x['abs_error'],
        reverse=True
    )[:10]

    has_shift = any(r.get('shift_rmse') for r in sorted_results)
    if has_shift:
        print(f"{'Well':<25s} {'opt_err':>8s} {'shft_rms':>8s} {'Entr5cm':>8s} {'TVT_rng':>8s}")
        print("-" * 65)
        for r in sorted_results:
            print(f"{r['well']:<25s} {r['opt_error']:>+8.2f} {r.get('shift_rmse', 0) or 0:>8.2f} "
                  f"{r.get('gr_entropy_5cm', 0) or 0:>8.1f} {r.get('tvt_range', 0) or 0:>8.1f}")
    else:
        print(f"{'Well':<25s} {'opt_err':>8s} {'Entr5cm':>8s} {'Rough':>8s} {'TVT_rng':>8s}")
        print("-" * 60)
        for r in sorted_results:
            print(f"{r['well']:<25s} {r['opt_error']:>+8.2f} {r.get('gr_entropy_5cm', 0) or 0:>8.1f} "
                  f"{r.get('roughness', 0) or 0:>8.1f} {r.get('tvt_range', 0) or 0:>8.1f}")

    # REF comparison if requested
    if compare_ref and 'REF' in all_results:
        print()
        print("=" * 80)
        print("REF vs OPT COMPARISON")
        print("=" * 80)

        for metric in metric_names:
            ref_wins = 0
            opt_wins = 0

            for opt_r in all_results['OPT']:
                well = opt_r['well']
                ref_r = next((r for r in all_results['REF'] if r['well'] == well), None)
                if ref_r and opt_r.get(metric) is not None and ref_r.get(metric) is not None:
                    if ref_r[metric] < opt_r[metric]:
                        ref_wins += 1
                    else:
                        opt_wins += 1

            total = ref_wins + opt_wins
            if total > 0:
                print(f"{metric:20s}: REF wins {ref_wins}/{total} ({100*ref_wins/total:.0f}%)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-correlation metrics analysis')
    parser.add_argument('--run-id', required=True, help='Run ID to analyze')
    parser.add_argument('--compare-ref', action='store_true', help='Compare with REFERENCE')
    parser.add_argument('--no-landing-filter', action='store_true', help='Disable landing_end_dls filter')

    args = parser.parse_args()
    run_analysis(args.run_id, args.compare_ref, not args.no_landing_filter)
