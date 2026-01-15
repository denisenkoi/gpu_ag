#!/usr/bin/env python3
"""
Compute full-well score from saved interpretations.
Compares LEGACY vs BRUTEFORCE to find correlation between score and error.
"""
import numpy as np
import torch
import psycopg2
from typing import List, Tuple

# Load dataset
ds = torch.load('dataset/gpu_ag_dataset.pt', weights_only=False)


def compute_well_score(
    well_name: str,
    interpretations: List[dict],
    mse_weight: float = 5.0
) -> Tuple[float, float, float]:
    """
    Compute full-well pearson, mse_norm, score from interpretations.

    Returns: (pearson, mse_norm, score)
    """
    if well_name not in ds:
        return None, None, None

    well_data = ds[well_name]

    # Get data
    well_md = well_data['well_md'].numpy()
    well_tvd = well_data['well_tvd'].numpy()

    # Use pseudo (stitched) typelog
    type_tvd = well_data['pseudo_tvd'].numpy()
    type_gr = well_data['pseudo_gr'].numpy()

    # Interpolate log_gr to well_md
    log_md = well_data['log_md'].numpy()
    log_gr = well_data['log_gr'].numpy()
    well_gr = np.interp(well_md, log_md, log_gr)

    # Build synthetic GR from interpretations
    synthetic = np.zeros_like(well_gr)
    covered = np.zeros_like(well_gr, dtype=bool)

    for interp in interpretations:
        md_start = interp['md_start']
        md_end = interp['md_end']
        start_shift = interp['start_shift']
        end_shift = interp['end_shift']

        # Find indices
        mask = (well_md >= md_start) & (well_md <= md_end)
        if not mask.any():
            continue

        seg_md = well_md[mask]
        seg_tvd = well_tvd[mask]

        # Interpolate shift linearly
        if md_end > md_start:
            ratio = (seg_md - md_start) / (md_end - md_start)
        else:
            ratio = np.zeros_like(seg_md)
        seg_shift = start_shift + ratio * (end_shift - start_shift)

        # TVT = TVD - shift
        tvt = seg_tvd - seg_shift

        # Clamp to typelog range and interpolate
        tvt_clamped = np.clip(tvt, type_tvd[0], type_tvd[-1])
        seg_synthetic = np.interp(tvt_clamped, type_tvd, type_gr)

        synthetic[mask] = seg_synthetic
        covered[mask] = True

    if not covered.any():
        return None, None, None

    # Use only covered region
    zone_gr = well_gr[covered]
    zone_synthetic = synthetic[covered]

    # Pearson correlation
    gr_centered = zone_gr - zone_gr.mean()
    syn_centered = zone_synthetic - zone_synthetic.mean()
    numer = (gr_centered * syn_centered).sum()
    denom = np.sqrt((gr_centered**2).sum() * (syn_centered**2).sum())
    pearson = numer / denom if denom > 1e-10 else 0.0

    # MSE normalized
    mse = ((zone_gr - zone_synthetic)**2).mean()
    mse_norm = mse / (zone_gr.var() + 1e-10)

    # Score
    score = pearson - mse_weight * mse_norm

    return float(pearson), float(mse_norm), float(score)


def main():
    conn = psycopg2.connect(dbname='gpu_ag', user='rogii', password='rogii123', host='localhost')
    cur = conn.cursor()

    # Get wells that have interpretations in both runs
    cur.execute('''
        SELECT DISTINCT well_name FROM interpretations
        WHERE run_id IN ('20260115_023837', '20260115_104553')
    ''')
    wells = [r[0] for r in cur.fetchall()]

    print(f"Computing full-well scores for {len(wells)} wells...")
    print()

    results = []

    for well_name in sorted(wells):
        # Get interpretations for both runs
        for run_id, algo in [('20260115_023837', 'LEGACY'), ('20260115_104553', 'BF')]:
            cur.execute('''
                SELECT md_start, md_end, start_shift, end_shift
                FROM interpretations
                WHERE run_id = %s AND well_name = %s
                ORDER BY seg_idx
            ''', (run_id, well_name))

            interps = [{'md_start': r[0], 'md_end': r[1], 'start_shift': r[2], 'end_shift': r[3]}
                       for r in cur.fetchall()]

            if not interps:
                continue

            # Get opt_error
            cur.execute('''
                SELECT opt_error FROM well_results
                WHERE run_id = %s AND well_name = %s
            ''', (run_id, well_name))
            row = cur.fetchone()
            opt_error = row[0] if row else None

            # Compute score
            pearson, mse_norm, score = compute_well_score(well_name, interps)

            if pearson is not None:
                results.append({
                    'well': well_name,
                    'algo': algo,
                    'pearson': pearson,
                    'mse_norm': mse_norm,
                    'score': score,
                    'opt_error': opt_error,
                    'abs_error': abs(opt_error) if opt_error else None
                })

    conn.close()

    # Analyze correlation
    print(f"{'Well':<25} {'Algo':<8} {'Pearson':>8} {'MSE_n':>8} {'Score':>10} {'Error':>8} {'|Error|':>8}")
    print("-" * 90)

    # Group by well
    from collections import defaultdict
    by_well = defaultdict(list)
    for r in results:
        by_well[r['well']].append(r)

    # Show wells with significant difference
    interesting = []
    for well, data in by_well.items():
        if len(data) == 2:
            d = {d['algo']: d for d in data}
            if 'LEGACY' in d and 'BF' in d:
                score_diff = d['LEGACY']['score'] - d['BF']['score']
                error_diff = d['LEGACY']['abs_error'] - d['BF']['abs_error']
                # Correlation: higher score should mean lower error
                # So if score_diff > 0 (LEGACY higher), error_diff should be < 0 (LEGACY lower)
                if abs(score_diff) > 0.01:
                    interesting.append((well, d, score_diff, error_diff))

    # Sort by score difference
    interesting.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\nWells with score difference (top 15):")
    print(f"{'Well':<25} {'L_score':>10} {'B_score':>10} {'L_err':>8} {'B_err':>8} {'Corr?':>6}")
    print("-" * 75)

    correct = 0
    wrong = 0

    for well, d, score_diff, error_diff in interesting[:15]:
        l, b = d['LEGACY'], d['BF']
        # If L has higher score, L should have lower error
        # score_diff > 0 means L higher, error_diff < 0 means L lower
        # Correlation correct if sign(score_diff) == -sign(error_diff)
        corr_ok = (score_diff > 0 and error_diff < 0) or (score_diff < 0 and error_diff > 0)
        mark = "✓" if corr_ok else "✗"
        print(f"{well:<25} {l['score']:>10.4f} {b['score']:>10.4f} {l['abs_error']:>8.2f} {b['abs_error']:>8.2f} {mark:>6}")
        if corr_ok:
            correct += 1
        else:
            wrong += 1

    print()
    print(f"CORRELATION: {correct} correct, {wrong} wrong out of {len(interesting[:15])}")
    print(f"Accuracy: {100*correct/(correct+wrong):.1f}%")


if __name__ == '__main__':
    main()
