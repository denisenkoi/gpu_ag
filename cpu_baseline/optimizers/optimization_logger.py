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
                    'measured_depth',
                    'step_type',
                    'optimizer_method',
                    'segments_count',
                    'iterations',
                    'final_correlation',
                    'pearson_correlation',
                    'mse_value',
                    'self_correlation',
                    'intersections_count',
                    'optimization_success',
                    'function_evaluations'
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
        
        # Extract data with assertions for required fields
        well_name = optimization_result['well_name']
        measured_depth = optimization_result['measured_depth']
        step_type = optimization_result['step_type']
        
        stats = optimization_result['optimization_stats']
        
        # Write to optimization CSV
        with open(self.optimization_csv, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                well_name,
                measured_depth,
                step_type,
                stats['method'],
                stats['segments_count'],
                stats['n_iterations'],
                stats['final_correlation'],
                stats['pearson_correlation'],
                stats['mse_value'],
                stats['self_correlation'],
                stats['intersections_count'],
                stats['success'],
                stats['n_function_evaluations']
            ])
        
        # Store in memory for statistics
        self.optimization_stats.append(optimization_result)
        
        logger.info(f"Logged optimization result: {well_name} MD={measured_depth} "
                   f"correlation={stats['final_correlation']:.6f}")

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
        
        correlations = [stat['optimization_stats']['final_correlation'] 
                       for stat in self.optimization_stats]
        
        return {
            'total_optimizations': total_optimizations,
            'successful_optimizations': successful_optimizations,
            'success_rate': successful_optimizations / total_optimizations,
            'avg_correlation': sum(correlations) / len(correlations),
            'min_correlation': min(correlations),
            'max_correlation': max(correlations)
        }

    def print_statistics(self):
        """Print optimization statistics summary to console"""
        opt_stats = self.get_optimization_statistics()
        
        if opt_stats['total_optimizations'] == 0:
            logger.info("No optimization statistics available")
            return
        
        logger.info("=== Optimization Statistics Summary ===")
        logger.info(f"Total optimizations: {opt_stats['total_optimizations']}")
        logger.info(f"Successful optimizations: {opt_stats['successful_optimizations']}")
        logger.info(f"Success rate: {opt_stats['success_rate']:.1%}")
        logger.info(f"Average correlation: {opt_stats['avg_correlation']:.6f}")
        logger.info(f"Correlation range: {opt_stats['min_correlation']:.6f} - {opt_stats['max_correlation']:.6f}")
        logger.info("=== End Statistics ===")
