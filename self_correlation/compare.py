"""
Compare self-correlation metrics across algorithms.
"""
import numpy as np
import psycopg2
from scipy import stats
from typing import List, Dict, Optional
from .tvt_variance import compute_tvt_variance
from .tvt_pairwise import compute_tvt_pairwise, compute_tvt_consistency


def get_interpretations_from_db(run_id: str, well_name: str) -> List[Dict]:
    """Load interpretations from database."""
    conn = psycopg2.connect(dbname='gpu_ag', user='rogii', password='rogii123', host='localhost')
    cur = conn.cursor()

    cur.execute("""
        SELECT md_start, md_end, start_shift, end_shift
        FROM interpretations
        WHERE run_id = %s AND well_name = %s
        ORDER BY seg_idx
    """, (run_id, well_name))

    interps = [{'md_start': r[0], 'md_end': r[1], 'start_shift': r[2], 'end_shift': r[3]}
               for r in cur.fetchall()]

    conn.close()
    return interps


def compare_algorithms(
    wells: Optional[List[str]] = None,
    run_ids: Dict[str, str] = None,
    metrics: List[str] = ['variance', 'pairwise', 'consistency']
) -> Dict:
    """
    Compare self-correlation metrics across algorithms.

    Args:
        wells: List of well names (None = all)
        run_ids: Dict mapping algo name to run_id
        metrics: Which metrics to compute

    Returns:
        Dict with comparison results
    """
    if run_ids is None:
        run_ids = {
            'REF': 'REFERENCE',
            'LEGACY': '20260115_023837',
            'BF': '20260115_104553'
        }

    conn = psycopg2.connect(dbname='gpu_ag', user='rogii', password='rogii123', host='localhost')
    cur = conn.cursor()

    # Get wells if not specified
    if wells is None:
        cur.execute("SELECT DISTINCT well_name FROM interpretations WHERE run_id = 'REFERENCE'")
        wells = [r[0] for r in cur.fetchall()]

    results = {algo: [] for algo in run_ids}

    for well_name in wells:
        for algo, run_id in run_ids.items():
            interps = get_interpretations_from_db(run_id, well_name)
            if not interps:
                continue

            result = {'well': well_name}

            if 'variance' in metrics:
                var_result = compute_tvt_variance(well_name, interps)
                result['var'] = var_result.get('weighted_var')
                result['var_overlaps'] = var_result.get('n_overlaps')

            if 'pairwise' in metrics:
                pw_result = compute_tvt_pairwise(well_name, interps)
                result['pw_diff'] = pw_result.get('mean_gr_diff')
                result['pw_pairs'] = pw_result.get('n_pairs')

            if 'consistency' in metrics:
                cons_result = compute_tvt_consistency(well_name, interps)
                result['cv'] = cons_result.get('mean_cv')
                result['cv_bins'] = cons_result.get('n_bins')

            # Get opt_error
            cur.execute("SELECT opt_error FROM well_results WHERE run_id = %s AND well_name = %s",
                       (run_id, well_name))
            row = cur.fetchone()
            result['opt_error'] = row[0] if row else None

            results[algo].append(result)

    conn.close()

    # Compute summaries
    summary = {}
    for algo, algo_results in results.items():
        if not algo_results:
            continue

        s = {'n_wells': len(algo_results)}

        if 'variance' in metrics:
            vars = [r['var'] for r in algo_results if r.get('var') is not None]
            s['avg_var'] = np.mean(vars) if vars else None

        if 'pairwise' in metrics:
            diffs = [r['pw_diff'] for r in algo_results if r.get('pw_diff') is not None]
            s['avg_pw_diff'] = np.mean(diffs) if diffs else None

        if 'consistency' in metrics:
            cvs = [r['cv'] for r in algo_results if r.get('cv') is not None]
            s['avg_cv'] = np.mean(cvs) if cvs else None

        summary[algo] = s

    # Compute correlations with opt_error for BF
    correlations = {}
    bf_results = results.get('BF', [])
    if bf_results:
        errs = [abs(r['opt_error']) for r in bf_results if r.get('opt_error') is not None]

        for metric_name, key in [('variance', 'var'), ('pairwise', 'pw_diff'), ('consistency', 'cv')]:
            vals = [r[key] for r in bf_results if r.get(key) is not None and r.get('opt_error') is not None]
            if len(vals) >= 10 and len(vals) == len(errs):
                r, p = stats.pearsonr(vals, errs)
                correlations[metric_name] = {'r': r, 'p': p}

    return {
        'results': results,
        'summary': summary,
        'correlations': correlations,
        'n_wells': len(wells)
    }


def print_comparison(comparison: Dict):
    """Pretty print comparison results."""
    print("="*80)
    print("SELF-CORRELATION METRICS COMPARISON")
    print("="*80)
    print()

    summary = comparison['summary']
    for algo, s in summary.items():
        print(f"{algo}:")
        print(f"  Wells: {s['n_wells']}")
        if s.get('avg_var') is not None:
            print(f"  Avg TVT variance: {s['avg_var']:.2f}")
        if s.get('avg_pw_diff') is not None:
            print(f"  Avg pairwise GR diff: {s['avg_pw_diff']:.2f}")
        if s.get('avg_cv') is not None:
            print(f"  Avg CV: {s['avg_cv']:.4f}")
        print()

    if comparison['correlations']:
        print("-"*80)
        print("CORRELATIONS with |opt_error| (BF):")
        print("-"*80)
        for metric, corr in comparison['correlations'].items():
            sign = "+" if corr['r'] > 0 else ""
            useful = "useful!" if corr['r'] > 0 and corr['p'] < 0.05 else "not useful"
            print(f"  {metric}: r={sign}{corr['r']:.3f}, p={corr['p']:.4f} ({useful})")


if __name__ == '__main__':
    comparison = compare_algorithms()
    print_comparison(comparison)
