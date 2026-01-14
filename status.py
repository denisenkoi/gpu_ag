#!/usr/bin/env python3
"""Status monitor for GPU AG optimizer runs."""

import psycopg2
from datetime import datetime, timedelta

DB_CONFIG = {
    'host': 'localhost',
    'database': 'gpu_ag',
    'user': 'rogii',
    'password': 'rogii123',
}


def get_status():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get active runs (have processing/pending wells or recent done wells)
    cur.execute("""
        SELECT r.run_id, r.description, r.angle_range, r.mse_weight, r.block_overlap,
               r.started_at
        FROM runs r
        WHERE EXISTS (
            SELECT 1 FROM well_results w
            WHERE w.run_id = r.run_id
            AND (w.status IN ('processing', 'pending') OR w.finished_at > NOW() - interval '1 hour')
        )
        ORDER BY r.started_at DESC
    """)
    runs = cur.fetchall()

    if not runs:
        print("No active runs found.")
        return

    print("=" * 80)
    print("GPU AG Optimizer Status")
    print("=" * 80)

    for run in runs:
        run_id, desc, angle_range, mse_weight, overlap, started_at = run

        print(f"\nRun: {run_id}")
        print(f"  Desc: {desc or '(no description)'}")
        print(f"  Params: range={angle_range}°, mse_weight={mse_weight}, overlap={overlap or 0}")
        print(f"  Started: {started_at}")

        # Get well counts by status
        cur.execute("""
            SELECT status, COUNT(*) FROM well_results
            WHERE run_id = %s GROUP BY status
        """, (run_id,))
        status_counts = dict(cur.fetchall())

        done = status_counts.get('done', 0)
        processing = status_counts.get('processing', 0)
        failed = status_counts.get('failed', 0)
        pending = status_counts.get('pending', 0)
        total = done + processing + failed + pending

        print(f"  Progress: {done}/{total} done, {processing} processing, {pending} pending, {failed} failed")

        # Check for stale processing (timeout expired)
        cur.execute("""
            SELECT well_name, started_at, locked_until
            FROM well_results
            WHERE run_id = %s AND status = 'processing' AND locked_until < NOW()
        """, (run_id,))
        stale = cur.fetchall()

        if stale:
            print(f"  ⚠️  STALE WELLS (timeout expired, process likely dead):")
            for well, started, locked in stale:
                mins_ago = (datetime.now() - locked).total_seconds() / 60
                print(f"      {well} - locked_until expired {mins_ago:.0f} min ago")

        # Get currently processing (not stale)
        cur.execute("""
            SELECT well_name, started_at, locked_until
            FROM well_results
            WHERE run_id = %s AND status = 'processing' AND locked_until >= NOW()
            ORDER BY started_at
        """, (run_id,))
        active = cur.fetchall()

        if active:
            print(f"  Active processing:")
            for well, started, locked in active:
                elapsed = (datetime.now() - started).total_seconds()
                remaining = (locked - datetime.now()).total_seconds()
                print(f"      {well} - {elapsed:.0f}s elapsed, {remaining:.0f}s until timeout")

        # Calculate avg time and ETA
        cur.execute("""
            SELECT AVG(opt_ms), MIN(opt_ms), MAX(opt_ms)
            FROM well_results
            WHERE run_id = %s AND status = 'done' AND opt_ms IS NOT NULL
        """, (run_id,))
        times = cur.fetchone()

        if times[0]:
            avg_sec = times[0] / 1000
            min_sec = times[1] / 1000
            max_sec = times[2] / 1000
            remaining_wells = total - done
            # Assume 2 GPUs working in parallel
            eta_sec = remaining_wells * avg_sec / 2
            eta_min = eta_sec / 60

            print(f"  Timing: avg={avg_sec:.1f}s, min={min_sec:.1f}s, max={max_sec:.1f}s")
            print(f"  ETA: ~{eta_min:.0f} min ({remaining_wells} wells remaining, 2 GPUs)")

        # Get RMSE if available
        cur.execute("""
            SELECT baseline_error, opt_error
            FROM well_results
            WHERE run_id = %s AND status = 'done'
            AND baseline_error IS NOT NULL AND opt_error IS NOT NULL
        """, (run_id,))
        errors = cur.fetchall()

        if errors:
            import numpy as np
            baseline = np.array([e[0] for e in errors])
            optimized = np.array([e[1] for e in errors])
            rmse_base = np.sqrt(np.mean(baseline**2))
            rmse_opt = np.sqrt(np.mean(optimized**2))
            improved = np.sum(np.abs(optimized) < np.abs(baseline))

            print(f"  RMSE: baseline={rmse_base:.2f}m, optimized={rmse_opt:.2f}m")
            print(f"  Improved: {improved}/{len(errors)} wells")

    print("\n" + "=" * 80)
    conn.close()


if __name__ == '__main__':
    get_status()
