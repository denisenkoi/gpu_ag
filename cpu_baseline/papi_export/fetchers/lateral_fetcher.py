import logging
from typing import Dict, List, Optional, Any

from ..utils import extract_val

logger = logging.getLogger(__name__)


class LateralFetcher:
    """Fetcher for lateral well raw data - returns unprocessed PAPI data only"""
    
    def __init__(self, config, api_connector):
        self.config = config
        self.api = api_connector
        
    def fetch_raw_lateral_data(self, project_uuid: str, well_name: str) -> Dict[str, Any]:
        """
        Fetch raw lateral well data from PAPI without any processing.
        
        Args:
            project_uuid: Project UUID
            well_name: Well name to fetch
            
        Returns:
            Dictionary with raw PAPI data:
            {
                'well_basic': {...},           # Well metadata
                'well_metadata': {...},       # Full well metadata  
                'survey_trajectory': [...],   # Raw survey points
                'log_data': {...},           # Raw log data
                'project_measure_unit': str  # Project units
            }
        """
        logger.info(f"Fetching raw lateral data for: {well_name}")
        
        # Get well by name
        well = self.api.get_well_by_name(project_uuid, well_name)
        assert well, f"Well '{well_name}' not found in project"
        
        well_uuid = well['uuid']
        
        # Get well metadata (coordinates and parameters)
        well_metadata = self.api.get_well_metadata(well_uuid)
        
        # Get raw trajectory survey points
        survey_trajectory = self.api.get_well_trajectory(well_uuid)
        logger.info(f"Retrieved {len(survey_trajectory)} raw survey points")
        
        # Get raw well log data
        log_data = self._fetch_raw_well_log(well_uuid)
        
        # Get project metadata to determine units
        project_data = self.api.get_project_by_name(self.config.project_name)
        project_measure_unit = project_data.get('measure_unit', 'METER')
        
        raw_data = {
            'well_basic': {
                'uuid': well_uuid,
                'name': well_name,
                'xySurface': {
                    'xSurface': well_metadata.get('xsrf', {}).get('val', 0.0),
                    'ySurface': well_metadata.get('ysrf', {}).get('val', 0.0)
                }
            },
            'well_metadata': well_metadata,
            'survey_trajectory': survey_trajectory,
            'log_data': log_data,
            'project_measure_unit': project_measure_unit
        }
        
        logger.info(f"Fetched raw lateral data: {len(survey_trajectory)} survey points, "
                   f"log: {'present' if log_data else 'missing'}, units: {project_measure_unit}")
        
        return raw_data
        
    def _fetch_raw_well_log(self, well_uuid: str) -> Optional[Dict]:
        """
        Fetch raw GR log data from PAPI without any processing.
        
        Args:
            well_uuid: Well UUID
            
        Returns:
            Raw log data dictionary or None if not found
        """
        logs = self.api.get_logs_by_well(well_uuid)
        
        # Find GR log by name
        target_log = None
        for log in logs:
            if log['name'] == self.config.lateral_log_name:
                target_log = log
                break
                
        if not target_log:
            logger.warning(f"Log '{self.config.lateral_log_name}' not found in lateral well")
            return None
            
        # Get raw log data points
        log_data = self.api.get_log_data(target_log['uuid'])
        log_points = log_data.get('log_points', [])

        logger.info(f"Retrieved {len(log_points)} raw log points for '{self.config.lateral_log_name}'")

        # Check for duplicate MD values in RAW PAPI data
        from collections import Counter
        all_md_values = []
        for i, point in enumerate(log_points):
            md_val = extract_val(point.get('md'))
            if md_val is not None:
                all_md_values.append((i, md_val))

        md_only = [md for idx, md in all_md_values]
        md_counts = Counter(md_only)
        duplicates = {md: count for md, count in md_counts.items() if count > 1}

        if duplicates:
            logger.warning(f"DUPLICATE_CHECK_RAW_PAPI: Found {len(duplicates)} duplicate MD values in RAW PAPI data")
            for dup_md, count in duplicates.items():
                indices = [idx for idx, md in all_md_values if md == dup_md]
                logger.warning(f"  MD={dup_md} appears {count} times at indices: {indices}")

                # Log data values for duplicates to check if they differ
                for idx in indices:
                    data_val = extract_val(log_points[idx].get('data'))
                    logger.warning(f"    [idx={idx}] MD={dup_md}, data={data_val}")

            # Deduplicate: keep last occurrence (more precise from later chunk)
            logger.info(f"DUPLICATE_DEDUP: Removing {len(duplicates)} duplicate MD values (keeping last occurrence)")

            # Create set of indices to remove (all duplicates except last)
            indices_to_remove = set()
            for dup_md in duplicates.keys():
                dup_indices = [idx for idx, md in all_md_values if md == dup_md]
                # Remove all except last
                indices_to_remove.update(dup_indices[:-1])

            # Build deduplicated list
            deduplicated_log_points = [
                point for i, point in enumerate(log_points)
                if i not in indices_to_remove
            ]

            logger.info(f"DUPLICATE_DEDUP: Removed {len(indices_to_remove)} duplicate points, kept {len(deduplicated_log_points)} unique points")
            log_points = deduplicated_log_points
        else:
            logger.info("DUPLICATE_CHECK_RAW_PAPI: No duplicates in RAW PAPI data")

        return {
            'log_uuid': target_log['uuid'],
            'log_name': target_log['name'],
            'log_points': log_points
        }