#!/usr/bin/env python3
"""
Load reference interpretations into database as a "REFERENCE" run.
Then compute scores and compare with LEGACY/BRUTEFORCE.
"""
import os
import numpy as np
import torch
import psycopg2
from datetime import datetime

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/gpu_ag_dataset.pt')

# Load dataset
ds = torch.load(DATASET_PATH, weights_only=False)


def compute_well_score(well_name: str, interpretations: list, mse_weight: float = 5.0):
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


def load_reference_to_db():
    """Load reference interpretations into DB."""
    conn = psycopg2.connect(dbname='gpu_ag', user='rogii', password='rogii123', host='localhost')
    cur = conn.cursor()

    run_id = 'REFERENCE'
    now = datetime.now()

    # Check if REFERENCE run exists
    cur.execute("SELECT run_id FROM runs WHERE run_id = %s", (run_id,))
    if cur.fetchone():
        print(f"Run {run_id} already exists. Deleting old data...")
        cur.execute("DELETE FROM interpretations WHERE run_id = %s", (run_id,))
        cur.execute("DELETE FROM well_results WHERE run_id = %s", (run_id,))
        cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))
        conn.commit()

    # Create run
    cur.execute("""
        INSERT INTO runs (run_id, description, started_at, finished_at, algorithm)
        VALUES (%s, %s, %s, %s, %s)
    """, (run_id, 'Manual reference interpretations from StarSteer', now, now, 'REFERENCE'))
    conn.commit()

    print(f"Created run {run_id}")
    print(f"Loading reference interpretations for {len(ds)} wells...")

    results = []

    for well_name in sorted(ds.keys()):
        well_data = ds[well_name]

        # Get reference data
        ref_mds = well_data['ref_segment_mds']
        ref_shifts = well_data['ref_shifts']

        if isinstance(ref_mds, torch.Tensor):
            ref_mds = ref_mds.numpy()
        if isinstance(ref_shifts, torch.Tensor):
            ref_shifts = ref_shifts.numpy()

        # Build interpretations list
        interpretations = []
        for i in range(len(ref_mds) - 1):
            md_start = float(ref_mds[i])
            md_end = float(ref_mds[i + 1])
            start_shift = float(ref_shifts[i])
            end_shift = float(ref_shifts[i + 1])

            interpretations.append({
                'md_start': md_start,
                'md_end': md_end,
                'start_shift': start_shift,
                'end_shift': end_shift
            })

            # Insert into DB
            cur.execute("""
                INSERT INTO interpretations (run_id, well_name, seg_idx, md_start, md_end, start_shift, end_shift)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (run_id, well_name, i, md_start, md_end, start_shift, end_shift))

        # Compute score
        pearson, mse_norm, score = compute_well_score(well_name, interpretations)

        if pearson is not None:
            # Insert well_results
            cur.execute("""
                INSERT INTO well_results (run_id, well_name, status, opt_error, pearson, mse, score)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (run_id, well_name, 'done', 0.0, pearson, mse_norm, score))

            results.append({
                'well': well_name,
                'pearson': pearson,
                'mse_norm': mse_norm,
                'score': score,
                'n_segments': len(interpretations)
            })
            print(f"  {well_name}: {len(interpretations)} segments, pearson={pearson:.4f}, mse={mse_norm:.4f}, score={score:.4f}")

    conn.commit()
    conn.close()

    print(f"\nLoaded {len(results)} wells")
    print(f"Average pearson: {np.mean([r['pearson'] for r in results]):.4f}")
    print(f"Average mse_norm: {np.mean([r['mse_norm'] for r in results]):.4f}")
    print(f"Average score: {np.mean([r['score'] for r in results]):.4f}")


def compare_with_algorithms():
    """Compare REFERENCE with LEGACY and BRUTEFORCE."""
    conn = psycopg2.connect(dbname='gpu_ag', user='rogii', password='rogii123', host='localhost')
    cur = conn.cursor()

    # Get all runs with their metrics
    cur.execute("""
        SELECT run_id, well_name, opt_error, pearson, mse, score
        FROM well_results
        WHERE run_id IN ('REFERENCE', '20260115_023837', '20260115_104553')
        AND status = 'done'
    """)

    # Organize by well
    from collections import defaultdict
    by_well = defaultdict(dict)
    run_names = {
        'REFERENCE': 'REF',
        '20260115_023837': 'LEGACY',
        '20260115_104553': 'BF'
    }

    for run_id, well_name, opt_error, pearson, mse, score in cur.fetchall():
        name = run_names.get(run_id, run_id)
        by_well[well_name][name] = {
            'opt_error': opt_error,
            'pearson': pearson,
            'mse': mse,
            'score': score
        }

    conn.close()

    # Compare
    print("\n" + "="*100)
    print("COMPARISON: REFERENCE vs LEGACY vs BRUTEFORCE")
    print("="*100)

    # Count wins by each metric
    wins = {'pearson': {'REF': 0, 'LEGACY': 0, 'BF': 0},
            'mse': {'REF': 0, 'LEGACY': 0, 'BF': 0},
            'score': {'REF': 0, 'LEGACY': 0, 'BF': 0}}

    wells_with_all = []
    for well, data in by_well.items():
        if 'REF' in data and 'LEGACY' in data and 'BF' in data:
            # Check all have valid metrics
            ref, leg, bf = data['REF'], data['LEGACY'], data['BF']
            if all(x is not None for x in [ref['pearson'], ref['mse'], ref['score'],
                                            leg['pearson'], leg['mse'], leg['score'],
                                            bf['pearson'], bf['mse'], bf['score']]):
                wells_with_all.append((well, data))

    print(f"\nWells with all 3 algorithms: {len(wells_with_all)}")
    print()

    # Table header
    print(f"{'Well':<25} {'REF_p':>7} {'LEG_p':>7} {'BF_p':>7} | {'REF_m':>7} {'LEG_m':>7} {'BF_m':>7} | {'REF_s':>8} {'LEG_s':>8} {'BF_s':>8} | Win")
    print("-"*120)

    for well, data in sorted(wells_with_all)[:30]:  # Show first 30
        ref = data['REF']
        leg = data['LEGACY']
        bf = data['BF']

        # Determine winners (higher pearson = better, lower mse = better, higher score = better)
        p_winner = 'REF' if ref['pearson'] >= max(leg['pearson'], bf['pearson']) else ('LEGACY' if leg['pearson'] > bf['pearson'] else 'BF')
        m_winner = 'REF' if ref['mse'] <= min(leg['mse'], bf['mse']) else ('LEGACY' if leg['mse'] < bf['mse'] else 'BF')
        s_winner = 'REF' if ref['score'] >= max(leg['score'], bf['score']) else ('LEGACY' if leg['score'] > bf['score'] else 'BF')

        wins['pearson'][p_winner] += 1
        wins['mse'][m_winner] += 1
        wins['score'][s_winner] += 1

        # Mark winners
        p_mark = '*' if p_winner == 'REF' else ''
        m_mark = '*' if m_winner == 'REF' else ''
        s_mark = '*' if s_winner == 'REF' else ''

        print(f"{well:<25} {ref['pearson']:>7.4f} {leg['pearson']:>7.4f} {bf['pearson']:>7.4f} | "
              f"{ref['mse']:>7.4f} {leg['mse']:>7.4f} {bf['mse']:>7.4f} | "
              f"{ref['score']:>8.4f} {leg['score']:>8.4f} {bf['score']:>8.4f} | {p_winner[0]}{m_winner[0]}{s_winner[0]}")

    # Summary
    total = len(wells_with_all)
    print()
    print("="*100)
    print("SUMMARY (who wins more often):")
    print("="*100)
    print(f"{'Metric':<10} {'REF':>10} {'LEGACY':>10} {'BF':>10}")
    print("-"*45)
    for metric in ['pearson', 'mse', 'score']:
        r = wins[metric]['REF']
        l = wins[metric]['LEGACY']
        b = wins[metric]['BF']
        print(f"{metric:<10} {r:>10} ({100*r/total:.1f}%) {l:>10} ({100*l/total:.1f}%) {b:>10} ({100*b/total:.1f}%)")

    print()
    print("KEY INSIGHT:")
    print("If REF wins by a metric, that metric is GOOD for selecting better solutions.")
    print("If REF loses, the metric is MISLEADING.")


if __name__ == '__main__':
    load_reference_to_db()
    compare_with_algorithms()
