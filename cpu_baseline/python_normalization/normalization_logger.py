import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class NormalizationLogger:
    """
    CSV logger for normalization results and failures.
    Logs both successful normalizations and problematic wells for analysis.
    """
    
    def __init__(self, results_dir: str):
        """
        Initialize logger with results directory.
        
        Args:
            results_dir: Directory where CSV file will be created
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.csv_path = self.results_dir / "normalization_results.csv"
        self._ensure_csv_headers()
        
        logger.info(f"NormalizationLogger initialized: {self.csv_path}")
    
    def _ensure_csv_headers(self):
        """Create CSV file with headers if it doesn't exist"""
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'well_name', 
                    'status',
                    'landing_end_md',
                    'normalization_length',
                    'segments_count',
                    'multiplier',
                    'shift',
                    'pearson_correlation',
                    'euclidean_distance',
                    'issue_description'
                ])
            logger.info(f"Created normalization results CSV: {self.csv_path}")
    
    def log_normalization_result(self, result: Dict[str, Any]):
        """
        Log normalization result (success or failure) to CSV.
        
        Args:
            result: Dictionary with normalization results from NormalizationCalculator
        """
        timestamp = datetime.now().isoformat()
        
        # Extract values from result dictionary
        well_name = result['well_name']
        status = result['status']
        landing_end_md = result.get('landing_end_md', '')
        normalization_length = result.get('normalization_length', '')
        segments_count = result.get('segments_count', '')
        multiplier = result.get('multiplier', '')
        shift = result.get('shift', '')
        pearson_correlation = result.get('pearson_correlation', '')
        euclidean_distance = result.get('euclidean_distance', '')
        issue_description = result.get('issue_description', '')
        
        # Format numeric values for CSV
        if isinstance(landing_end_md, (int, float)):
            landing_end_md = f"{landing_end_md:.1f}"
        if isinstance(normalization_length, (int, float)):
            normalization_length = f"{normalization_length:.1f}"
        if isinstance(multiplier, (int, float)):
            multiplier = f"{multiplier:.6f}"
        if isinstance(shift, (int, float)):
            shift = f"{shift:.6f}"
        if isinstance(pearson_correlation, (int, float)):
            pearson_correlation = f"{pearson_correlation:.6f}"
        if isinstance(euclidean_distance, (int, float)):
            euclidean_distance = f"{euclidean_distance:.3f}"
        
        # Write row to CSV
        row = [
            timestamp,
            well_name,
            status,
            landing_end_md,
            normalization_length,
            segments_count,
            multiplier,
            shift,
            pearson_correlation,
            euclidean_distance,
            issue_description
        ]
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        # Log to console based on status
        if status == 'success':
            logger.info(f"✅ {well_name}: multiplier={multiplier}, shift={shift}, "
                       f"pearson={pearson_correlation}")
        else:
            logger.warning(f"❌ {well_name}: {issue_description}")
    
    def log_success(self, well_name: str, landing_end_md: float, 
                   normalization_length: float, segments_count: int,
                   multiplier: float, shift: float,
                   pearson_correlation: float, euclidean_distance: float):
        """
        Convenience method to log successful normalization.
        
        Args:
            well_name: Name of the well
            landing_end_md: End MD of landing section
            normalization_length: Total length used for normalization
            segments_count: Number of segments used
            multiplier: Calculated multiplier coefficient
            shift: Calculated shift coefficient
            pearson_correlation: Pearson correlation of normalized curves
            euclidean_distance: Euclidean distance between normalized curves
        """
        result = {
            'status': 'success',
            'well_name': well_name,
            'landing_end_md': landing_end_md,
            'normalization_length': normalization_length,
            'segments_count': segments_count,
            'multiplier': multiplier,
            'shift': shift,
            'pearson_correlation': pearson_correlation,
            'euclidean_distance': euclidean_distance,
            'issue_description': ''
        }
        
        self.log_normalization_result(result)
    
    def log_failure(self, well_name: str, issue: str, issue_description: str, **kwargs):
        """
        Convenience method to log normalization failure.
        
        Args:
            well_name: Name of the well
            issue: Short issue identifier (e.g., 'negative_multiplier')
            issue_description: Human-readable description of the issue
            **kwargs: Additional metadata (landing_end_md, multiplier, etc.)
        """
        result = {
            'status': 'failed',
            'well_name': well_name,
            'issue': issue,
            'issue_description': issue_description,
            'landing_end_md': kwargs.get('landing_end_md', ''),
            'normalization_length': kwargs.get('normalization_length', ''),
            'segments_count': kwargs.get('segments_count', ''),
            'multiplier': kwargs.get('multiplier', ''),
            'shift': kwargs.get('shift', ''),
            'pearson_correlation': '',
            'euclidean_distance': ''
        }
        
        self.log_normalization_result(result)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics from CSV file for analysis.
        
        Returns:
            Dictionary with normalization statistics
        """
        if not self.csv_path.exists():
            return {
                'total_wells': 0,
                'successful': 0,
                'failed': 0,
                'success_rate': 0.0,
                'issues_count': {}
            }
        
        total_wells = 0
        successful = 0
        failed = 0
        issues_count = {}
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                total_wells += 1
                
                if row['status'] == 'success':
                    successful += 1
                else:
                    failed += 1
                    # Count issues for analysis
                    issue_desc = row['issue_description']
                    if issue_desc:
                        # Extract issue type from description
                        if 'Negative multiplier' in issue_desc:
                            issue_type = 'negative_multiplier'
                        elif 'Insufficient normalization length' in issue_desc:
                            issue_type = 'insufficient_length'
                        elif 'No manual interpretation segments' in issue_desc:
                            issue_type = 'no_segments_in_range'
                        elif 'projection calculation failed' in issue_desc:
                            issue_type = 'projection_failed'
                        elif 'Empty curves' in issue_desc:
                            issue_type = 'empty_curves'
                        else:
                            issue_type = 'other'
                        
                        issues_count[issue_type] = issues_count.get(issue_type, 0) + 1
        
        success_rate = (successful / total_wells * 100) if total_wells > 0 else 0.0
        
        return {
            'total_wells': total_wells,
            'successful': successful,
            'failed': failed,
            'success_rate': success_rate,
            'issues_count': issues_count
        }
    
    def print_statistics(self):
        """Print normalization statistics to console"""
        stats = self.get_statistics()
        
        logger.info("=== Normalization Statistics ===")
        logger.info(f"Total wells processed: {stats['total_wells']}")
        logger.info(f"Successful normalizations: {stats['successful']}")
        logger.info(f"Failed normalizations: {stats['failed']}")
        logger.info(f"Success rate: {stats['success_rate']:.1f}%")
        
        if stats['issues_count']:
            logger.info("Common issues:")
            for issue_type, count in stats['issues_count'].items():
                logger.info(f"  - {issue_type}: {count} wells")
        
        logger.info(f"Detailed results: {self.csv_path}")
    
    def get_problematic_wells(self, issue_type: Optional[str] = None) -> list:
        """
        Get list of wells with specific issues for analysis.
        
        Args:
            issue_type: Filter by specific issue type (optional)
            
        Returns:
            List of dictionaries with problematic well information
        """
        if not self.csv_path.exists():
            return []
        
        problematic_wells = []
        
        with open(self.csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                if row['status'] != 'failed':
                    continue
                
                # Filter by issue type if specified
                if issue_type:
                    if issue_type not in row['issue_description'].lower():
                        continue
                
                problematic_wells.append({
                    'well_name': row['well_name'],
                    'issue_description': row['issue_description'],
                    'multiplier': row['multiplier'],
                    'shift': row['shift'],
                    'landing_end_md': row['landing_end_md'],
                    'normalization_length': row['normalization_length']
                })
        
        return problematic_wells