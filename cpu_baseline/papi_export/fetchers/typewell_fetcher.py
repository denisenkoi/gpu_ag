import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

#from sklearn.externals.array_api_compat.torch import greater_equal

from ..utils import extract_val

logger = logging.getLogger(__name__)


class TypewellFetcher:
    """Fetcher for typewell raw data - returns unprocessed PAPI data with normalized metadata"""
    
    def __init__(self, config, api_connector):
        self.config = config
        self.api = api_connector
        
        # Initialize reports directory
        self.reports_dir = Path('./papi_export_reports')
        self.reports_dir.mkdir(exist_ok=True)
        
    def fetch_raw_typewell_data(self, lateral_uuid: str) -> Optional[Dict[str, Any]]:
        """
        Fetch raw typewell data linked to lateral from PAPI with normalized metadata.
        
        Args:
            lateral_uuid: Lateral UUID to find linked typewell
            
        Returns:
            Dictionary with raw PAPI data and normalized metadata:
            {
                'typewell_basic': {...},      # Basic typewell info
                'typewell_metadata': {...},   # Normalized typewell metadata
                'raw_trajectory': [...],      # Raw trajectory points
                'log_data': {...},           # Raw log data
                'shift_value': float         # Shift to lateral
            }
        """
        logger.info(f"Fetching raw typewell data for lateral: {lateral_uuid}")

        # Get linked typewells
        linked_typewells = self.api.get_linked_typewells(lateral_uuid)
        if not linked_typewells:
            logger.warning("No linked typewell found for lateral")
            return None

        # Get typewell names for all linked typewells
        typewell_list = []
        for tw in linked_typewells:
            tw_meta = self.api.get_typewell_metadata(tw['typewell_id'])
            tw_name = tw_meta.get('name', 'Unknown')
            typewell_list.append({
                'data': tw,
                'name': tw_name,
                'uuid': tw['typewell_id']
            })

        # Select typewell by name if specified in config
        selected_typewell = None
        if self.config.typewell_name:
            # Search by name
            for tw in typewell_list:
                if tw['name'] == self.config.typewell_name:
                    selected_typewell = tw
                    logger.info(f"✅ Selected typewell by name: '{tw['name']}'")
                    break

            if not selected_typewell:
                # Name specified but not found - FAIL
                available_names = [tw['name'] for tw in typewell_list]
                logger.warning(f"🔄 Found {len(typewell_list)} typewells: {available_names}")
                raise ValueError(
                    f"TYPEWELL_NAME='{self.config.typewell_name}' not found in linked typewells. "
                    f"Available typewells: {', '.join(available_names)}"
                )
        else:
            # No name specified - FAIL if multiple, use if single
            if len(typewell_list) > 1:
                available_names = [tw['name'] for tw in typewell_list]
                logger.warning(f"🔄 Found {len(typewell_list)} typewells: {available_names}")
                raise ValueError(
                    f"Multiple typewells found ({len(typewell_list)}): {', '.join(available_names)}. "
                    f"Set TYPEWELL_NAME in .env to choose specific typewell."
                )
            else:
                selected_typewell = typewell_list[0]
                logger.info(f"Using single typewell: '{selected_typewell['name']}'")

        # Extract selected typewell data
        linked_data = selected_typewell['data']
        typewell_uuid = linked_data['typewell_id']
        shift_value = extract_val(linked_data['shift'])

        logger.info(f"Found linked typewell: {typewell_uuid}, name: '{selected_typewell['name']}', shift: {shift_value}")
        
        # Get typewell metadata
        raw_typewell_metadata = self.api.get_typewell_metadata(typewell_uuid)
        typewell_name = raw_typewell_metadata.get('name', 'Unknown')
        
        # Log raw metadata structure for debugging
        logger.debug(f"Raw typewell metadata fields: {list(raw_typewell_metadata.keys())}")
        
        # Normalize typewell metadata to match lateral format
        typewell_metadata = self._normalize_typewell_metadata(raw_typewell_metadata)
        
        # Get raw typewell trajectory
        raw_trajectory = self.api.get_typewell_trajectory(typewell_uuid)
        logger.info(f"Retrieved {len(raw_trajectory)} raw typewell trajectory points")
        
        # Get raw typewell log data
        log_data = self._fetch_raw_typewell_log(typewell_uuid)
        
        raw_data = {
            'typewell_basic': {
                'uuid': typewell_uuid,
                'name': typewell_name,
                'shift_value': shift_value
            },
            'typewell_metadata': typewell_metadata,  # Now normalized
            'raw_trajectory': raw_trajectory,
            'log_data': log_data,
            'shift_value': shift_value
        }
        
        logger.info(f"Fetched raw typewell data: {len(raw_trajectory)} trajectory points, "
                   f"log: {'present' if log_data else 'missing'}, shift: {shift_value}")
        
        return raw_data
        
    def _normalize_typewell_metadata(self, raw_metadata: Dict) -> Dict:
        """
        Normalize typewell metadata to match lateral format.
        
        Typewell differences from lateral:
        - Missing 'azimuth' field (typewells don't have it)
        - Some fields might be undefined or missing
        
        Args:
            raw_metadata: Raw metadata from PAPI
            
        Returns:
            Normalized metadata compatible with trajectory processor
        """
        logger.debug("Normalizing typewell metadata")
        
        # Create normalized metadata
        normalized = {}
        
        # List of expected fields with their default values
        expected_fields = {
            'uuid': None,  # No default for UUID
            'name': 'Unknown',
            'operator': '',
            'api': '',
            'xsrf': 0.0,
            'ysrf': 0.0,
            'kb': 0.0,
            'convergence': 0.0,
            'tie_in_tvd': 0.0,
            'tie_in_ns': 0.0,
            'tie_in_ew': 0.0
        }
        
        # Process each expected field
        for field_name, default_value in expected_fields.items():
            normalized[field_name] = self._extract_field_value(
                raw_metadata, field_name, default_value
            )
        
        # Add missing 'azimuth' field (typewells don't have it in PAPI)
        # This is required for trajectory processor compatibility
        normalized['azimuth'] = {'val': 0.0}
        logger.debug("Added missing 'azimuth' field with default value 0.0")
        
        # Log normalization summary
        added_fields = []
        defaulted_fields = []
        
        if 'azimuth' not in raw_metadata:
            added_fields.append('azimuth')
            
        for field_name in expected_fields:
            if field_name not in raw_metadata:
                defaulted_fields.append(field_name)
            elif isinstance(raw_metadata.get(field_name), dict):
                if raw_metadata[field_name].get('undefined', False):
                    defaulted_fields.append(f"{field_name}(undefined)")
                    
        if added_fields:
            logger.info(f"Added missing fields: {', '.join(added_fields)}")
        if defaulted_fields:
            logger.info(f"Used defaults for: {', '.join(defaulted_fields)}")
            
        return normalized
        
    def _extract_field_value(self, metadata: Dict, field_name: str, default_value: Any) -> Any:
        """
        Extract field value from PAPI metadata handling different formats.
        
        Handles:
        - Missing fields -> use default
        - String fields (name, operator, api) -> return as is
        - Numeric fields with {'val': number} -> return as {'val': number}
        - Fields with {'undefined': true} -> return {'val': default}
        - Direct numeric values -> wrap in {'val': number}
        
        Args:
            metadata: Source metadata dictionary
            field_name: Field name to extract
            default_value: Default value if field missing or undefined
            
        Returns:
            Field value in correct format for trajectory processor
        """
        # Handle string fields (name, operator, api, uuid)
        if field_name in ['name', 'operator', 'api', 'uuid']:
            if field_name in metadata:
                return metadata[field_name]
            return default_value if default_value is not None else ''
            
        # Handle numeric fields (should have {'val': ...} structure)
        if field_name not in metadata:
            logger.debug(f"Field '{field_name}' not found, using default: {default_value}")
            return {'val': default_value}
            
        field_data = metadata[field_name]
        
        # If already in correct format {'val': number}
        if isinstance(field_data, dict):
            if 'undefined' in field_data and field_data['undefined']:
                logger.debug(f"Field '{field_name}' is undefined, using default: {default_value}")
                return {'val': default_value}
            if 'val' in field_data:
                # Ensure it's not None
                val = field_data['val']
                if val is None:
                    logger.debug(f"Field '{field_name}' has null value, using default: {default_value}")
                    return {'val': default_value}
                return field_data  # Return as is
            # Unknown dict format
            logger.warning(f"Unknown dict format for field '{field_name}': {field_data}")
            return {'val': default_value}
            
        # If direct numeric value (shouldn't happen but handle it)
        if isinstance(field_data, (int, float)):
            logger.debug(f"Field '{field_name}' has direct value {field_data}, wrapping in {{'val': ...}}")
            return {'val': float(field_data)}
            
        # Unknown format
        logger.warning(f"Unexpected format for field '{field_name}': {field_data}, using default")
        return {'val': default_value}
        
    def _fetch_raw_typewell_log(self, typewell_uuid: str) -> Optional[Dict]:
        """
        Fetch raw typewell log data from PAPI without any processing.
        
        Args:
            typewell_uuid: Typewell UUID
            
        Returns:
            Raw log data dictionary or None if not found
        """
        logs = self.api.get_logs_by_typewell(typewell_uuid)
        
        # Find specified log by name
        target_log = None
        for log in logs:
            if log['name'] == self.config.typewell_log_name:
                target_log = log
                break

        gamma_name_parts = ["gamma", "GR"]

        gr_logs = []
        for log in logs:
            addlog_to_gr_logs = False
            for gamma_name_part in gamma_name_parts:
                if gamma_name_part.lower() in log['name'].lower():
                    addlog_to_gr_logs = True
            if addlog_to_gr_logs:
                gr_logs.append(log)

                
        if not target_log:
            # Try first available log as fallback
            if gr_logs:
                # Log if multiple logs with gamma_name_parts found and use first one
                if len(gr_logs) > 1:
                    log_names = [log['name'] for log in gr_logs]
                    logger.warning(f"📊 MULTIPLE gr typelogs found! Found {len(gr_logs)} typewell logs: {log_names}. Target '{self.config.typewell_log_name}' not found, using first: {gr_logs[0]['name']}")
                    
                    # Log to file for analysis
                    self._log_multiple_typewell_logs_to_file(typewell_uuid, log_names, self.config.typewell_log_name, gr_logs[0]['name'])
                else:
                    logger.warning(f"📊 Target log '{self.config.typewell_log_name}' not found, using only available: '{gr_logs[0]['name']}'")
                
                target_log = gr_logs[0]
            else:
                logger.warning(f"No logs with any of: {gamma_name_parts} found for typewell")
                return None
                
        # Get raw log data points
        log_data = self.api.get_log_data(target_log['uuid'])
        log_points = log_data.get('log_points', [])

        logger.info(f"Retrieved {len(log_points)} raw log points for '{target_log['name']}'")

        # Check for duplicate MD values and deduplicate (same as lateral)
        from ..utils import extract_val
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
            logger.warning(f"DUPLICATE_CHECK_RAW_PAPI_TYPEWELL: Found {len(duplicates)} duplicate MD values in typewell data")
            for dup_md, count in duplicates.items():
                indices = [idx for idx, md in all_md_values if md == dup_md]
                logger.warning(f"  MD={dup_md} appears {count} times at indices: {indices}")

            # Deduplicate: keep last occurrence
            logger.info(f"DUPLICATE_DEDUP_TYPEWELL: Removing {len(duplicates)} duplicate MD values (keeping last occurrence)")

            indices_to_remove = set()
            for dup_md in duplicates.keys():
                dup_indices = [idx for idx, md in all_md_values if md == dup_md]
                indices_to_remove.update(dup_indices[:-1])

            deduplicated_log_points = [
                point for i, point in enumerate(log_points)
                if i not in indices_to_remove
            ]

            logger.info(f"DUPLICATE_DEDUP_TYPEWELL: Removed {len(indices_to_remove)} duplicate points, kept {len(deduplicated_log_points)} unique points")
            log_points = deduplicated_log_points
        else:
            logger.info("DUPLICATE_CHECK_RAW_PAPI_TYPEWELL: No duplicates in typewell data")

        return {
            'log_uuid': target_log['uuid'],
            'log_name': target_log['name'],
            'log_points': log_points
        }
    
    def _log_multiple_typewells_to_file(self, lateral_uuid: str, typewell_names: List[str]):
        """Log multiple typewells info to file for analysis"""
        timestamp = datetime.now().strftime('%Y%m%d')
        report_file = self.reports_dir / f'multiple_typewells_{timestamp}.txt'
        
        with open(report_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}: LATERAL {lateral_uuid} has {len(typewell_names)} typewells: {', '.join(typewell_names)} (using first)\n")
    
    def _log_multiple_typewell_logs_to_file(self, typewell_uuid: str, log_names: List[str], target_name: str, used_name: str):
        """Log multiple typewell logs info to file for analysis"""
        timestamp = datetime.now().strftime('%Y%m%d')
        report_file = self.reports_dir / f'multiple_typewell_logs_{timestamp}.txt'
        
        with open(report_file, 'a', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}: TYPEWELL {typewell_uuid} has {len(log_names)} logs: {', '.join(log_names)}. Target '{target_name}' not found, using '{used_name}'\n")