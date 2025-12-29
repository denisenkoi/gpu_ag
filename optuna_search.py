"""
Optuna hyperparameter search for GPU Slicer.

Optimizes:
- first_slice_length: Length of first slice (120, 150, 170m)
- lookback_distance: Lookback distance (200, 250, 300, 350, 400m)
- max_segment_length: Max segment length (30, 40, 50m)
- angle_sum_power: Angle sum penalty power (1.5, 2.0, 2.5, 3.0)

Usage:
    python optuna_search.py --n-trials 50 --timeout 28800  # 8 hours
"""

import sys
import os
import optuna
import logging
import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base directory
BASE_DIR = Path(__file__).parent
SLICER_SCRIPT = BASE_DIR / "true_gpu_slicer.py"
RESULTS_BASE = BASE_DIR / "results" / "optuna"


def run_experiment(params: dict, trial_dir: Path) -> dict:
    """
    Run a single experiment with given parameters.
    Returns metrics dict or None on error.
    """
    cmd = [
        sys.executable, str(SLICER_SCRIPT),
        "--all",
        "--slice-step", "30",
        "--num-workers", "3",
        "--results-dir", str(trial_dir),
        "--first-slice-length", str(params['first_slice_length']),
        "--lookback-distance", str(params['lookback_distance']),
        "--max-segment-length", str(params['max_segment_length']),
        "--angle-sum-power", str(params['angle_sum_power']),
    ]

    env = os.environ.copy()
    env['NORMALIZATION_MODE'] = 'OLD'

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour max per trial
        )

        # Parse output for metrics
        output = result.stdout + result.stderr

        # Extract RMSE from summary
        rmse_all = None
        rmse_endpoint = None

        for line in output.split('\n'):
            if 'RMSE: mean=' in line and 'overall=' in line:
                # Parse: RMSE: mean=5.82m, overall=6.95m
                try:
                    overall_part = line.split('overall=')[1].split('m')[0]
                    rmse_all = float(overall_part)
                except:
                    pass
            if 'Endpoint errors:' in line and 'RMSE=' in line:
                # Parse: Endpoint errors: mean=+1.36m, RMSE=10.27m
                try:
                    rmse_part = line.split('RMSE=')[1].split('m')[0]
                    rmse_endpoint = float(rmse_part)
                except:
                    pass

        return {
            'rmse_all': rmse_all,
            'rmse_endpoint': rmse_endpoint,
            'success': rmse_all is not None
        }

    except subprocess.TimeoutExpired:
        logger.error("Trial timed out")
        return {'success': False}
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return {'success': False}


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.
    Returns RMSE all points (lower is better).
    """
    # Sample hyperparameters
    params = {
        'first_slice_length': trial.suggest_categorical('first_slice_length', [120, 150, 170]),
        'lookback_distance': trial.suggest_categorical('lookback_distance', [200, 250, 300, 350, 400]),
        'max_segment_length': trial.suggest_categorical('max_segment_length', [30, 40, 50, 70, 100]),
        'angle_sum_power': trial.suggest_float('angle_sum_power', 1.5, 3.0, step=0.5),
    }

    trial_dir = RESULTS_BASE / f"trial_{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Save params
    with open(trial_dir / "params.json", 'w') as f:
        json.dump(params, f, indent=2)

    logger.info(f"Trial {trial.number}: {params}")

    metrics = run_experiment(params, trial_dir)

    if not metrics['success'] or metrics['rmse_all'] is None:
        # Return a high value for failed trials
        return float('inf')

    # Save metrics
    with open(trial_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Trial {trial.number} result: RMSE_all={metrics['rmse_all']:.2f}m, "
                f"RMSE_endpoint={metrics['rmse_endpoint']:.2f}m")

    return metrics['rmse_all']


def main():
    parser = argparse.ArgumentParser(description='Optuna hyperparameter search')
    parser.add_argument('--n-trials', type=int, default=50,
                        help='Number of trials to run')
    parser.add_argument('--timeout', type=int, default=28800,
                        help='Total timeout in seconds (default: 8 hours)')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Study name (for resuming)')
    parser.add_argument('--db-path', type=str, default=None,
                        help='SQLite database path for persistence')

    args = parser.parse_args()

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)

    # Create or load study
    study_name = args.study_name or f"gpu_slicer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if args.db_path:
        storage = f"sqlite:///{args.db_path}"
    else:
        db_file = RESULTS_BASE / f"{study_name}.db"
        storage = f"sqlite:///{db_file}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction='minimize',
        load_if_exists=True
    )

    logger.info(f"Starting Optuna search: {args.n_trials} trials, timeout={args.timeout}s")
    logger.info(f"Study: {study_name}")
    logger.info(f"Storage: {storage}")

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True
    )

    # Print results
    print("\n" + "="*60)
    print("OPTUNA SEARCH COMPLETE")
    print("="*60)

    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best RMSE: {study.best_value:.2f}m ({study.best_value * 3.28084:.2f}ft)")
    print(f"Best params: {study.best_params}")

    # Save best params
    best_file = RESULTS_BASE / "best_params.json"
    with open(best_file, 'w') as f:
        json.dump({
            'trial': study.best_trial.number,
            'rmse': study.best_value,
            'params': study.best_params
        }, f, indent=2)

    print(f"\nBest params saved to: {best_file}")

    # Top 10 trials
    print("\nTop 10 trials:")
    trials_df = study.trials_dataframe()
    if not trials_df.empty:
        trials_df = trials_df.sort_values('value').head(10)
        print(trials_df[['number', 'value', 'params_first_slice_length',
                         'params_lookback_distance', 'params_max_segment_length',
                         'params_angle_sum_power']].to_string())


if __name__ == '__main__':
    main()
