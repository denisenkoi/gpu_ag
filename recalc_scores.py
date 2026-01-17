#!/usr/bin/env python3
"""
Recalculate mse and score for all runs from saved interpretations.
"""
import os
import numpy as np
import torch
import psycopg2

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/gpu_ag_dataset.pt')

# Load dataset
ds = torch.load(DATASET_PATH, weights_only=False)


def compute_well_score(well_name: str, interpretations: list, mse_weight: float = 5.0):
    """Compute full-well pearson, mse_norm, score from interpretations."""
    if well_name not in ds:
        return None, None, None

    well_data = ds[well_name]
    well_md = well_data['well_md'].numpy()
    well_tvd = well_data['well_tvd'].numpy()
    type_tvd = well_data['pseudo_tvd'].numpy()
    type_gr = well_data['pseudo_gr'].numpy()
    log_md = well_data['log_md'].numpy()
    log_gr = well_data['log_gr'].numpy()
    well_gr = np.interp(well_md, log_md, log_gr)

    synthetic = np.zeros_like(well_gr)
    covered = np.zeros_like(well_gr, dtype=bool)

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
        tvt = seg_tvd - seg_shift
        tvt_clamped = np.clip(tvt, type_tvd[0], type_tvd[-1])
        seg_synthetic = np.interp(tvt_clamped, type_tvd, type_gr)

        synthetic[mask] = seg_synthetic
        covered[mask] = True

    if not covered.any():
        return None, None, None

    zone_gr = well_gr[covered]
    zone_synthetic = synthetic[covered]

    gr_centered = zone_gr - zone_gr.mean()
    syn_centered = zone_synthetic - zone_synthetic.mean()
    numer = (gr_centered * syn_centered).sum()
    denom = np.sqrt((gr_centered**2).sum() * (syn_centered**2).sum())
    pearson = numer / denom if denom > 1e-10 else 0.0

    mse = ((zone_gr - zone_synthetic)**2).mean()
    mse_norm = mse / (zone_gr.var() + 1e-10)
    score = pearson - mse_weight * mse_norm

    return float(pearson), float(mse_norm), float(score)


def recalc_all():
    conn = psycopg2.connect(dbname='gpu_ag', user='rogii', password='rogii123', host='localhost')
    cur = conn.cursor()

    # Get runs that need recalculation
    cur.execute("""
        SELECT DISTINCT run_id FROM well_results
        WHERE mse IS NULL AND run_id != 'REFERENCE'
    """)
    runs = [r[0] for r in cur.fetchall()]
    print(f"Runs to recalculate: {runs}")

    for run_id in runs:
        print(f"\n=== {run_id} ===")

        # Get wells
        cur.execute("SELECT DISTINCT well_name FROM interpretations WHERE run_id = %s", (run_id,))
        wells = [r[0] for r in cur.fetchall()]
        print(f"Wells: {len(wells)}")

        updated = 0
        for well_name in wells:
            # Get interpretations
            cur.execute("""
                SELECT md_start, md_end, start_shift, end_shift
                FROM interpretations
                WHERE run_id = %s AND well_name = %s
                ORDER BY seg_idx
            """, (run_id, well_name))

            interps = [{'md_start': r[0], 'md_end': r[1], 'start_shift': r[2], 'end_shift': r[3]}
                       for r in cur.fetchall()]

            if not interps:
                continue

            pearson, mse_norm, score = compute_well_score(well_name, interps)

            if pearson is not None:
                cur.execute("""
                    UPDATE well_results
                    SET pearson = %s, mse = %s, score = %s
                    WHERE run_id = %s AND well_name = %s
                """, (pearson, mse_norm, score, run_id, well_name))
                updated += 1

        conn.commit()
        print(f"Updated {updated} wells")

    conn.close()
    print("\nDone!")


if __name__ == '__main__':
    recalc_all()
