# optimization_logger.py

"""
Logger for optimization statistics and correlation metrics
Similar to NormalizationLogger but for optimization results
"""

import csv
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class OptimizationLogger:
    """Logs optimization statistics to CSV files"""

    def __init__(self, results_dir: str):
        """
        Initialize optimization logger
        
        Args:
            results_dir: Directory to save optimization statistics
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # CSV file for optimization statistics
        self.optimization_csv = self.results_dir / "optimization_statistics.csv"
        self.correlation_csv = self.results_dir / "correlation_metrics.csv"
        
        # Initialize CSV headers if files don't exist
        self._initialize_csv_files()
        
        # In-memory statistics for summary
        self.optimization_stats = []
        self.correlation_stats = []

    def _initialize_csv_files(self):
        """Initialize CSV files with headers if they don't exist"""
        
        # Optimization statistics CSV
        if not self.optimization_csv.exists():
            with open(self.optimization_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'well_name',
                    'step_type',
                    'start_md',
                    'end_md',
                    'segments_count',
                    'fun',
                    'pearson',
                    'mse',
                    'angle_penalty',
                    'angle_sum_penalty',
                    'angles',
                    'actual_evals',
                    'expected_evals',
                    'elapsed_time',
                    'success'
                ])
        
        # Correlation metrics CSV
        if not self.correlation_csv.exists():
            with open(self.correlation_csv, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'well_name',
                    'measured_depth', 
                    'step_type',
                    'correlation_type',  # 'manual', 'initial', 'optimized'
                    'pearson_correlation',
                    'mse_value',
                    'euclidean_distance',
                    'points_count'
                ])

    def log_optimization_result(self, optimization_result: Dict[str, Any]):
        """
        Log optimization result to CSV

        Args:
            optimization_result: Dictionary containing optimization statistics
        """
        timestamp = time.time()

        # Extract data
        well_name = optimization_result['well_name']
        step_type = optimization_result['step_type']
        start_md = optimization_result.get('start_md', 0)
        end_md = optimization_result.get('end_md', optimization_result.get('measured_depth', 0))
        angles = optimization_result.get('angles', '')

        stats = optimization_result['optimization_stats']

        # Write to optimization CSV
        with open(self.optimization_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                well_name,
                step_type,
                f"{start_md:.1f}",
                f"{end_md:.1f}",
                stats['segments_count'],
                f"{stats['final_fun']:.6f}",
                f"{stats.get('pearson', 0):.4f}",
                f"{stats.get('mse', stats['final_fun']):.6f}",
                f"{stats.get('angle_penalty', 0):.4f}",
                f"{stats.get('angle_sum_penalty', 0):.4f}",
                angles,
                stats['n_function_evaluations'],
                stats.get('expected_evaluations', stats['n_function_evaluations']),
                f"{stats['elapsed_time']:.2f}",
                stats['success']
            ])

        # Store in memory for statistics
        self.optimization_stats.append(optimization_result)

    def log_correlation_metrics(self, correlation_result: Dict[str, Any]):
        """
        Log correlation metrics for different interpretation types
        
        Args:
            correlation_result: Dictionary containing correlation metrics
        """
        timestamp = time.time()
        
        # Extract required fields
        well_name = correlation_result['well_name']
        measured_depth = correlation_result['measured_depth']
        step_type = correlation_result['step_type']
        
        # Log each correlation type
        for correlation_type, metrics in correlation_result['correlations'].items():
            with open(self.correlation_csv, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    well_name,
                    measured_depth,
                    step_type,
                    correlation_type,
                    metrics['pearson_correlation'],
                    metrics['mse_value'],
                    metrics['euclidean_distance'],
                    metrics['points_count']
                ])
        
        # Store in memory
        self.correlation_stats.append(correlation_result)
        
        logger.debug(f"Logged correlation metrics: {well_name} MD={measured_depth}")

    def get_optimization_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics for optimization results

        Returns:
            Dictionary with optimization statistics summary
        """
        if not self.optimization_stats:
            return {'total_optimizations': 0}

        # Calculate summary statistics
        total_optimizations = len(self.optimization_stats)
        successful_optimizations = sum(1 for stat in self.optimization_stats
                                     if stat['optimization_stats']['success'])

        fun_values = [stat['optimization_stats']['mse_value']
                     for stat in self.optimization_stats]
        evals = [stat['optimization_stats']['n_function_evaluations']
                for stat in self.optimization_stats]

        return {
            'total_optimizations': total_optimizations,
            'successful_optimizations': successful_optimizations,
            'success_rate': successful_optimizations / total_optimizations,
            'avg_fun': sum(fun_values) / len(fun_values),
            'min_fun': min(fun_values),
            'max_fun': max(fun_values),
            'total_evals': sum(evals),
            'avg_evals': sum(evals) / len(evals)
        }

    def print_statistics(self):
        """Print optimization statistics summary to console"""
        opt_stats = self.get_optimization_statistics()

        if opt_stats['total_optimizations'] == 0:
            logger.info("No optimization statistics available")
            return

        logger.info("=== Optimization Statistics Summary ===")
        logger.info(f"Total optimizations: {opt_stats['total_optimizations']}")
        logger.info(f"Successful: {opt_stats['successful_optimizations']} ({opt_stats['success_rate']:.1%})")
        logger.info(f"Objective (fun): avg={opt_stats['avg_fun']:.6f}, min={opt_stats['min_fun']:.6f}, max={opt_stats['max_fun']:.6f}")
        logger.info(f"Function evals: total={opt_stats['total_evals']}, avg={opt_stats['avg_evals']:.0f}")
        logger.info("=== End Statistics ===")
