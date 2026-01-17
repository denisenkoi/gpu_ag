#!/usr/bin/env python3
"""
Grid search for optimal entropy parameters (smoothing + bin size).

Goal: Find parameters where REF ALWAYS wins on entropy metric.

Usage:
    python -m self_correlation.entropy_grid_search --run-id 20260115_213032
"""
import argparse
import numpy as np
import torch
import psycopg2
from scipy.signal import savgol_filter
from scipy import stats
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import itertools

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'gpu_ag',
    'user': 'rogii',
    'password': 'rogii123',
}

# Load dataset once
_ds = None
def get_dataset():
    global _ds
    if _ds is None:
        _ds = torch.load('data/wells_limited_pseudo.pt', weights_only=False)
    return _ds


def compute_tvt_gr(
    well_name: str,
    interpretations: List[Dict],
    md_start_filter: Optional[float] = None,
    smooth_window: int = 0,
    smooth_polyorder: int = 2
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute TVT and GR arrays from interpretation.

    Args:
        smooth_window: Savitzky-Golay window (0 = no smoothing, must be odd)
        smooth_polyorder: Polynomial order for Savitzky-Golay
    """
    ds = get_dataset()
    if well_name not in ds:
        return None, None

    well_data = ds[well_name]
    well_md = well_data['well_md'].numpy()
    well_tvd = well_data['well_tvd'].numpy()
    log_md = well_data['log_md'].numpy()
    log_gr = well_data['log_gr'].numpy()
    well_gr = np.interp(well_md, log_md, log_gr)

    # Apply smoothing if requested
    if smooth_window > 0 and smooth_window % 2 == 1 and len(well_gr) > smooth_window:
        well_gr = savgol_filter(well_gr, smooth_window, smooth_polyorder)

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
    if md_start_filter is not None:
        valid = valid & (well_md >= md_start_filter)

    if valid.sum() < 10:
        return None, None

    return tvt[valid], well_gr[valid]


def compute_gr_entropy(tvt: np.ndarray, gr: np.ndarray, bin_size: float = 0.05) -> float:
    """Compute GR 'entropy' in TVT bins.

    Actually computes std(bin_means) - standard deviation of mean GR per bin.
    Lower = more consistent GR across TVT = better self-correlation.
    """
    tvt_min = tvt.min()
    bin_idx = ((tvt - tvt_min) / bin_size).astype(int)

    bins = defaultdict(list)
    for i, bi in enumerate(bin_idx):
        bins[bi].append(gr[i])

    # Compute mean GR for each bin
    bin_means = [np.mean(v) for v in bins.values() if len(v) >= 1]

    if len(bin_means) > 5:
        return float(np.std(bin_means))
    return 0.0


def get_interpretations_from_db(run_id: str, well_name: str) -> List[Dict]:
    """Get interpretation segments from database."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT md_start, md_end, start_shift, end_shift
                FROM interpretations
                WHERE run_id = %s AND well_name = %s AND is_active = true
                ORDER BY seg_idx
            """, (run_id, well_name))
            rows = cur.fetchall()
            return [{'md_start': r[0], 'md_end': r[1], 'start_shift': r[2], 'end_shift': r[3]}
                    for r in rows]
    finally:
        conn.close()


def get_ref_interpretations(well_name: str) -> List[Dict]:
    """Get REFERENCE interpretation from database (ignore is_active flag)."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT md_start, md_end, start_shift, end_shift
                FROM interpretations
                WHERE run_id = 'REFERENCE' AND well_name = %s
                ORDER BY seg_idx
            """, (well_name,))
            rows = cur.fetchall()
            return [{'md_start': r[0], 'md_end': r[1], 'start_shift': r[2], 'end_shift': r[3]}
                    for r in rows]
    finally:
        conn.close()


def get_wells_from_run(run_id: str) -> List[str]:
    """Get list of wells from run."""
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT well_name FROM interpretations
                WHERE run_id = %s AND is_active = true
            """, (run_id,))
            return [r[0] for r in cur.fetchall()]
    finally:
        conn.close()


def get_landing_end_dls(well_name: str) -> Optional[float]:
    """Get landing_end_dls from dataset."""
    ds = get_dataset()
    if well_name not in ds:
        return None
    return ds[well_name].get('landing_end_dls')


def grid_search(run_id: str, smooth_windows: List[int], bin_sizes: List[float]):
    """Run grid search over smoothing and bin size parameters."""
    wells = get_wells_from_run(run_id)
    print(f"Found {len(wells)} wells in run {run_id}")

    # Get opt_error for each well
    conn = psycopg2.connect(**DB_CONFIG)
    well_errors = {}
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT well_name, opt_error FROM well_results
                WHERE run_id = %s AND status = 'done'
            """, (run_id,))
            for r in cur.fetchall():
                well_errors[r[0]] = r[1]
    finally:
        conn.close()

    print(f"\n{'Window':>8} {'Bin(cm)':>8} {'REF wins':>10} {'OPT wins':>10} {'REF%':>8} {'Corr w/err':>12}")
    print("=" * 70)

    best_ref_pct = 0
    best_params = None

    for window, bin_size in itertools.product(smooth_windows, bin_sizes):
        ref_wins = 0
        opt_wins = 0
        entropy_diffs = []  # OPT - REF
        errors = []

        for well_name in wells:
            landing_end = get_landing_end_dls(well_name)

            # Get interpretations
            opt_interps = get_interpretations_from_db(run_id, well_name)
            ref_interps = get_ref_interpretations(well_name)

            if not opt_interps or not ref_interps:
                continue

            # Compute TVT/GR with smoothing
            tvt_opt, gr_opt = compute_tvt_gr(well_name, opt_interps, landing_end, window, 2)
            tvt_ref, gr_ref = compute_tvt_gr(well_name, ref_interps, landing_end, window, 2)

            if tvt_opt is None or tvt_ref is None:
                continue

            # Compute entropy
            ent_opt = compute_gr_entropy(tvt_opt, gr_opt, bin_size)
            ent_ref = compute_gr_entropy(tvt_ref, gr_ref, bin_size)

            # Higher entropy = worse (more disorder)
            if ent_ref > ent_opt:
                opt_wins += 1
            else:
                ref_wins += 1

            # Track for correlation
            if well_name in well_errors:
                entropy_diffs.append(ent_opt - ent_ref)
                errors.append(abs(well_errors[well_name]))

        total = ref_wins + opt_wins
        if total == 0:
            continue

        ref_pct = 100 * ref_wins / total

        # Correlation with error
        corr_str = "N/A"
        if len(entropy_diffs) > 10:
            corr, pval = stats.pearsonr(entropy_diffs, errors)
            corr_str = f"r={corr:.3f}"

        print(f"{window:>8} {bin_size*100:>8.1f} {ref_wins:>10} {opt_wins:>10} {ref_pct:>7.1f}% {corr_str:>12}")

        if ref_pct > best_ref_pct:
            best_ref_pct = ref_pct
            best_params = (window, bin_size)

    print("=" * 70)
    if best_params:
        print(f"Best: window={best_params[0]}, bin={best_params[1]*100:.1f}cm -> REF wins {best_ref_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Grid search for entropy parameters")
    parser.add_argument('--run-id', required=True, help='Run ID to analyze')
    parser.add_argument('--smooth', type=int, nargs='+', default=[0, 15, 25, 51],
                        help='Smoothing windows (0=none, must be odd)')
    parser.add_argument('--bins', type=float, nargs='+', default=[0.02, 0.03, 0.05],
                        help='Bin sizes in meters')
    args = parser.parse_args()

    grid_search(args.run_id, args.smooth, args.bins)


if __name__ == '__main__':
    main()
