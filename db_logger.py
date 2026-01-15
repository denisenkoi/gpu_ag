"""PostgreSQL logger for GPU AG optimizer runs."""

import os
import json
import subprocess
from datetime import datetime
from typing import Optional, List, Dict, Any

try:
    import psycopg2
    from psycopg2.extras import execute_values
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


# Database connection settings
DB_CONFIG = {
    'host': os.getenv('PGHOST', 'localhost'),
    'port': int(os.getenv('PGPORT', 5432)),
    'database': os.getenv('PGDATABASE', 'gpu_ag'),
    'user': os.getenv('PGUSER', 'rogii'),
    'password': os.getenv('PGPASSWORD', 'rogii123'),
}

# Enable/disable DB logging
DB_LOGGING_ENABLED = os.getenv('DB_LOGGING', '1') == '1'


def get_connection():
    """Get database connection."""
    if not HAS_PSYCOPG2:
        return None
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"DB connection failed: {e}")
        return None


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True, text=True, cwd=os.path.dirname(__file__)
        )
        return result.stdout.strip()[:40] if result.returncode == 0 else None
    except:
        return None


def get_env_settings() -> Dict[str, Any]:
    """Collect relevant ENV settings."""
    env_vars = [
        'NORMALIZE_0_100', 'USE_PSEUDO_TYPELOG', 'NORMALIZATION_MODE',
        'TYPELOG_SMOOTHING_WINDOW', 'WELLLOG_SMOOTHING_WINDOW',
        'SMART_OVERLAP_SEGMENTS', 'OTSU_THRESHOLD_MULT',
    ]
    return {k: os.getenv(k) for k in env_vars if os.getenv(k) is not None}


class RunLogger:
    """Logger for a single optimization run."""

    def __init__(self, run_id: str, batch_id: Optional[str] = None):
        self.run_id = run_id
        self.batch_id = batch_id
        self.conn = get_connection() if DB_LOGGING_ENABLED else None
        self.commit_hash = get_git_commit()
        self.env_settings = get_env_settings()
        self.params = {}
        self.started_at = datetime.now()

    def set_params(self, angle_range: float, angle_step: float = 0.2,
                   mse_weight: float = 0.1, block_overlap: int = 0,
                   center_mode: str = 'trend', chunk_size: int = None,
                   algorithm: str = 'BRUTEFORCE',
                   description: str = None, cli_args: str = None):
        """Set run parameters."""
        self.params = {
            'angle_range': angle_range,
            'angle_step': angle_step,
            'mse_weight': mse_weight,
            'block_overlap': block_overlap,
            'center_mode': center_mode,
            'chunk_size': chunk_size,
            'algorithm': algorithm,
            'description': description,
            'cli_args': cli_args,
        }

    def start_run(self, n_wells: int):
        """Log run start to database."""
        if not self.conn:
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO runs (run_id, batch_id, commit_hash, started_at,
                                     angle_range, angle_step, mse_weight, block_overlap,
                                     center_mode, chunk_size, algorithm, env_settings, n_wells,
                                     description, cli_args)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id) DO UPDATE SET
                        started_at = EXCLUDED.started_at,
                        n_wells = EXCLUDED.n_wells,
                        description = EXCLUDED.description,
                        cli_args = EXCLUDED.cli_args
                """, (
                    self.run_id, self.batch_id, self.commit_hash, self.started_at,
                    self.params.get('angle_range'), self.params.get('angle_step'),
                    self.params.get('mse_weight'), self.params.get('block_overlap'),
                    self.params.get('center_mode'), self.params.get('chunk_size'),
                    self.params.get('algorithm'), json.dumps(self.env_settings), n_wells,
                    self.params.get('description'), self.params.get('cli_args')
                ))
            self.conn.commit()
        except Exception as e:
            print(f"DB error on start_run: {e}")

    def create_pending_wells(self, well_names: List[str]):
        """Create pending entries for all wells at start of run.

        This allows status.py to show total wells count immediately.
        """
        if not self.conn or not well_names:
            return

        try:
            with self.conn.cursor() as cur:
                for well_name in well_names:
                    cur.execute("""
                        INSERT INTO well_results (run_id, well_name, status)
                        VALUES (%s, %s, 'pending')
                        ON CONFLICT (run_id, well_name) DO NOTHING
                    """, (self.run_id, well_name))
            self.conn.commit()
            print(f"Created {len(well_names)} pending wells")
        except Exception as e:
            print(f"DB error on create_pending_wells: {e}")

    def log_well_result(self, well_name: str, baseline_error: float,
                        opt_error: float, n_segments: int, opt_ms: int):
        """Log single well result."""
        if not self.conn:
            return

        try:
            improved = bool(abs(float(opt_error)) < abs(float(baseline_error)))
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO well_results (run_id, well_name, baseline_error,
                                             opt_error, n_segments, opt_ms, improved)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, well_name) DO UPDATE SET
                        opt_error = EXCLUDED.opt_error,
                        opt_ms = EXCLUDED.opt_ms
                """, (self.run_id, well_name, float(baseline_error), float(opt_error),
                      int(n_segments), int(opt_ms), improved))
            self.conn.commit()
        except Exception as e:
            print(f"DB error on log_well_result: {e}")

    def log_block(self, well_name: str, block_idx: int, n_angles: int,
                  n_combos: int, opt_ms: int, best_score: float, best_pearson: float,
                  best_mse: float = None, best_loss: float = None,
                  n_segments: int = None, ref_angle: float = None,
                  angles: list = None, angle_diffs: list = None):
        """Log single optimization block with full metrics.

        Args:
            well_name: Well name
            block_idx: Block index (0, 1, 2...)
            n_angles: Number of angle steps per segment
            n_combos: Total number of combinations evaluated (or estimated)
            opt_ms: Optimization time in milliseconds
            best_score: Best score (pearson - mse_weight * mse_norm)
            best_pearson: Best Pearson correlation
            best_mse: Best MSE value
            best_loss: Best loss value (for minimization)
            n_segments: Number of segments in block
            ref_angle: Reference trajectory angle
            angles: List of optimal angles per segment
            angle_diffs: List of angle differences from ref_angle
        """
        if not self.conn:
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO blocks (run_id, well_name, block_idx,
                                       n_angles, n_combos, opt_ms, best_score, best_pearson,
                                       best_mse, best_loss, n_segments, ref_angle, angles, angle_diffs)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (run_id, well_name, block_idx) DO UPDATE SET
                        n_angles = EXCLUDED.n_angles,
                        n_combos = EXCLUDED.n_combos,
                        opt_ms = EXCLUDED.opt_ms,
                        best_score = EXCLUDED.best_score,
                        best_pearson = EXCLUDED.best_pearson,
                        best_mse = EXCLUDED.best_mse,
                        best_loss = EXCLUDED.best_loss,
                        n_segments = EXCLUDED.n_segments,
                        ref_angle = EXCLUDED.ref_angle,
                        angles = EXCLUDED.angles,
                        angle_diffs = EXCLUDED.angle_diffs
                """, (
                    self.run_id, well_name, int(block_idx), int(n_angles), int(n_combos),
                    int(opt_ms), float(best_score), float(best_pearson),
                    float(best_mse) if best_mse is not None else None,
                    float(best_loss) if best_loss is not None else None,
                    int(n_segments) if n_segments is not None else None,
                    float(ref_angle) if ref_angle is not None else None,
                    json.dumps(angles) if angles is not None else None,
                    json.dumps(angle_diffs) if angle_diffs is not None else None,
                ))
            self.conn.commit()
        except Exception as e:
            print(f"DB error on log_block: {e}")

    def log_block_segments(self, well_name: str, block_idx: int,
                           segments: List[Any], seg_start_idx: int):
        """Log segments from a single block optimization.

        Args:
            well_name: Well name
            block_idx: Block index (0, 1, 2...)
            segments: List of segment objects from this block
            seg_start_idx: Global segment index of first segment in block
        """
        if not self.conn or not segments:
            return

        def _to_float(v):
            return float(v) if v is not None else None

        try:
            rows = []
            for i, seg in enumerate(segments):
                seg_idx = seg_start_idx + i
                rows.append((
                    self.run_id, well_name, int(seg_idx),
                    float(seg.start_md), float(seg.end_md), float(seg.end_md - seg.start_md),
                    float(seg.angle_deg), float(seg.start_shift), float(seg.end_shift),
                    _to_float(getattr(seg, 'end_error', None)),
                    float(seg.pearson),
                    _to_float(getattr(seg, 'mse', None)),
                    _to_float(getattr(seg, 'score', None)),
                    int(block_idx), False  # is_active=False until finalized
                ))

            with self.conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO interpretations (run_id, well_name, seg_idx,
                        md_start, md_end, md_length, angle_deg, start_shift, end_shift,
                        end_error, pearson, mse, score, block_idx, is_active)
                    VALUES %s
                    ON CONFLICT (run_id, well_name, seg_idx) DO UPDATE SET
                        angle_deg = EXCLUDED.angle_deg,
                        start_shift = EXCLUDED.start_shift,
                        end_shift = EXCLUDED.end_shift,
                        pearson = EXCLUDED.pearson,
                        mse = EXCLUDED.mse,
                        score = EXCLUDED.score,
                        block_idx = EXCLUDED.block_idx,
                        is_active = EXCLUDED.is_active
                """, rows)
            self.conn.commit()
        except Exception as e:
            print(f"DB error on log_block_segments: {e}")

    def finalize_well_segments(self, well_name: str):
        """Mark all segments for this well as active (final interpretation)."""
        if not self.conn:
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE interpretations
                    SET is_active = true
                    WHERE run_id = %s AND well_name = %s
                """, (self.run_id, well_name))
            self.conn.commit()
        except Exception as e:
            print(f"DB error on finalize_well_segments: {e}")

    def try_lock_well(self, well_name: str, timeout_sec: int = 600) -> bool:
        """Try to lock a well for processing (atomic).

        Returns True if lock acquired, False if well is done or locked by another.
        """
        if not self.conn:
            return True  # No DB = always proceed

        try:
            with self.conn.cursor() as cur:
                # Atomic upsert: insert if not exists, update if stale or not done
                cur.execute("""
                    INSERT INTO well_results (run_id, well_name, status, started_at, locked_until)
                    VALUES (%s, %s, 'processing', NOW(), NOW() + interval '%s seconds')
                    ON CONFLICT (run_id, well_name) DO UPDATE SET
                        status = 'processing',
                        started_at = NOW(),
                        locked_until = NOW() + interval '%s seconds'
                    WHERE well_results.status != 'done'
                      AND (well_results.status != 'processing' OR well_results.locked_until < NOW())
                    RETURNING well_name
                """, (self.run_id, well_name, timeout_sec, timeout_sec))
                result = cur.fetchone()
            self.conn.commit()
            return result is not None
        except Exception as e:
            print(f"DB error on try_lock_well: {e}")
            return True  # On error, proceed anyway

    def complete_well(self, well_name: str, baseline_error: float, opt_error: float,
                      n_segments: int, opt_ms: int, rmse: float = None,
                      pearson: float = None, mse: float = None, score: float = None):
        """Mark well as completed with all metrics."""
        if not self.conn:
            return

        try:
            improved = bool(abs(float(opt_error)) < abs(float(baseline_error)))
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE well_results SET
                        status = 'done',
                        finished_at = NOW(),
                        baseline_error = %s,
                        opt_error = %s,
                        n_segments = %s,
                        opt_ms = %s,
                        improved = %s,
                        rmse = %s,
                        pearson = %s,
                        mse = %s,
                        score = %s
                    WHERE run_id = %s AND well_name = %s
                """, (
                    float(baseline_error), float(opt_error), int(n_segments), int(opt_ms),
                    improved,
                    float(rmse) if rmse is not None else None,
                    float(pearson) if pearson is not None else None,
                    float(mse) if mse is not None else None,
                    float(score) if score is not None else None,
                    self.run_id, well_name
                ))
            self.conn.commit()
        except Exception as e:
            print(f"DB error on complete_well: {e}")

    def fail_well(self, well_name: str, error_msg: str = None):
        """Mark well as failed."""
        if not self.conn:
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE well_results SET
                        status = 'failed',
                        finished_at = NOW()
                    WHERE run_id = %s AND well_name = %s
                """, (self.run_id, well_name))
            self.conn.commit()
            if error_msg:
                print(f"Well {well_name} failed: {error_msg}")
        except Exception as e:
            print(f"DB error on fail_well: {e}")

    def log_interpretation(self, well_name: str, segments: List[Any]):
        """Log all segments for a well (legacy method, marks all as active)."""
        if not self.conn or not segments:
            return

        def _to_float(v):
            return float(v) if v is not None else None

        try:
            rows = []
            for i, seg in enumerate(segments):
                rows.append((
                    self.run_id, well_name, int(i),
                    float(seg.start_md), float(seg.end_md), float(seg.end_md - seg.start_md),
                    float(seg.angle_deg), float(seg.start_shift), float(seg.end_shift),
                    _to_float(getattr(seg, 'end_error', None)),
                    float(seg.pearson),
                    _to_float(getattr(seg, 'mse', None)),
                    _to_float(getattr(seg, 'score', None)),
                    None, True  # block_idx=None, is_active=True for legacy
                ))

            with self.conn.cursor() as cur:
                execute_values(cur, """
                    INSERT INTO interpretations (run_id, well_name, seg_idx,
                        md_start, md_end, md_length, angle_deg, start_shift, end_shift,
                        end_error, pearson, mse, score, block_idx, is_active)
                    VALUES %s
                    ON CONFLICT (run_id, well_name, seg_idx) DO UPDATE SET
                        angle_deg = EXCLUDED.angle_deg,
                        end_shift = EXCLUDED.end_shift,
                        pearson = EXCLUDED.pearson,
                        block_idx = EXCLUDED.block_idx,
                        is_active = EXCLUDED.is_active
                """, rows)
            self.conn.commit()
        except Exception as e:
            print(f"DB error on log_interpretation: {e}")

    def finish_run(self, baseline_rmse: float, optimized_rmse: float,
                   wells_improved: int, total_time_sec: float):
        """Log run completion."""
        if not self.conn:
            return

        try:
            with self.conn.cursor() as cur:
                cur.execute("""
                    UPDATE runs SET
                        finished_at = %s,
                        baseline_rmse = %s,
                        optimized_rmse = %s,
                        wells_improved = %s,
                        total_time_sec = %s
                    WHERE run_id = %s
                """, (datetime.now(), float(baseline_rmse), float(optimized_rmse),
                      int(wells_improved), float(total_time_sec), self.run_id))
            self.conn.commit()
        except Exception as e:
            print(f"DB error on finish_run: {e}")

    def recalculate_run_stats(self) -> dict:
        """Recalculate run stats from well_results table.

        Returns dict with calculated stats.
        """
        if not self.conn:
            return {}

        try:
            import numpy as np
            with self.conn.cursor() as cur:
                # Get all completed wells
                cur.execute("""
                    SELECT baseline_error, opt_error, opt_ms
                    FROM well_results
                    WHERE run_id = %s AND status = 'done'
                      AND baseline_error IS NOT NULL AND opt_error IS NOT NULL
                """, (self.run_id,))
                rows = cur.fetchall()

                if not rows:
                    return {}

                baseline = np.array([r[0] for r in rows])
                optimized = np.array([r[1] for r in rows])
                times_ms = [r[2] for r in rows if r[2] is not None]

                rmse_base = float(np.sqrt(np.mean(baseline**2)))
                rmse_opt = float(np.sqrt(np.mean(optimized**2)))
                improved = int(np.sum(np.abs(optimized) < np.abs(baseline)))
                n_wells = len(rows)
                total_time = sum(times_ms) / 1000.0 if times_ms else 0.0

                # Update runs table
                cur.execute("""
                    UPDATE runs SET
                        baseline_rmse = %s,
                        optimized_rmse = %s,
                        wells_improved = %s,
                        n_wells = %s,
                        total_time_sec = %s,
                        finished_at = NOW()
                    WHERE run_id = %s
                """, (rmse_base, rmse_opt, improved, n_wells, total_time, self.run_id))

            self.conn.commit()
            return {
                'baseline_rmse': rmse_base,
                'optimized_rmse': rmse_opt,
                'wells_improved': improved,
                'n_wells': n_wells,
                'total_time_sec': total_time,
            }
        except Exception as e:
            print(f"DB error on recalculate_run_stats: {e}")
            return {}

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


# Singleton for current run
_current_logger: Optional[RunLogger] = None


def init_logger(run_id: str, batch_id: Optional[str] = None) -> RunLogger:
    """Initialize logger for a run."""
    global _current_logger
    _current_logger = RunLogger(run_id, batch_id)
    return _current_logger


def get_logger() -> Optional[RunLogger]:
    """Get current logger."""
    return _current_logger
