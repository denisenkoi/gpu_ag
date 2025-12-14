import os
import math
import time
import random
import logging
from typing import Dict, List, Optional, Any
from urllib3.exceptions import ProtocolError
from requests.exceptions import ConnectionError, ChunkedEncodingError

from ..api.data_fetch_connector import DataFetchConnector

logger = logging.getLogger(__name__)


class InterpretationUploader:
    """Upload interpretation segments to PAPI in real-time with fault tolerance"""
    
    # Retry configuration (copied from papi_loader.py)
    MAX_RETRY_ATTEMPTS = 5
    BASE_DELAY = 2  # Base delay in seconds
    MAX_DELAY = 150  # Maximum delay in seconds
    
    def __init__(self, project_measure_unit: str = None):
        """Initialize uploader with lazy connection

        Args:
            project_measure_unit: Project units ('FOOT' or 'METER'). Required for unit conversion.
        """
        self.api = None
        self.project_name = os.getenv('SOLO_PROJECT_NAME')
        self.base_interpretation_name = os.getenv('PAPI_INTERPRETATION_NAME', 'NightAG')
        self.generate_multiple_interpretations = os.getenv('GENERATE_MULTIPLE_INTERPRETATIONS', 'false').lower() == 'true'
        self.mul = os.getenv('PAPI_INTERPRETATION_NAME', 'NightAG')

        # Store project measure unit - REQUIRED for correct unit conversion
        self.project_measure_unit = project_measure_unit
        if not self.project_measure_unit:
            logger.warning("InterpretationUploader: project_measure_unit not provided. Will assume PAPI expects FEET.")

        # Validate configuration
        assert self.project_name, "SOLO_PROJECT_NAME not set in .env"
        assert self.base_interpretation_name, "PAPI_INTERPRETATION_NAME not set in .env"

        logger.info(f"InterpretationUploader initialized for project: {self.project_name}, base name: {self.base_interpretation_name}, units: {self.project_measure_unit}")
    
    def _ensure_connection(self):
        """Create/recreate PAPI connection"""
        if self.api is None:
            logger.info("Creating PAPI connection...")
            self.api = DataFetchConnector()
            logger.info("PAPI connection established")
    
    def _is_connection_error(self, exception: Exception) -> bool:
        """Check if exception is a temporary connection error that should be retried (copied from papi_loader.py)"""
        error_str = str(exception).lower()
        
        # Check for common connection error patterns
        connection_patterns = [
            'connection aborted',
            'remote end closed connection',
            'connection reset',
            'connection timeout',
            'timeout',
            'network is unreachable',
            'temporary failure in name resolution',
            'read timeout',
            'connection broken'
        ]
        
        # Check for temporary server errors (retryable)
        server_error_patterns = [
            '502', 'bad gateway',
            '503', 'service unavailable', 
            '504', 'gateway timeout', 'gateway time-out'
        ]
        
        # Check exception type
        if isinstance(exception, (ConnectionError, ChunkedEncodingError, ProtocolError)):
            return True
            
        # Check error message patterns
        is_connection = any(pattern in error_str for pattern in connection_patterns)
        is_server_error = any(pattern in error_str for pattern in server_error_patterns)
        
        return is_connection or is_server_error
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay with jitter (copied from papi_loader.py)"""
        # Exponential backoff: 2^attempt * base_delay
        delay = self.BASE_DELAY * (2 ** (attempt - 1))
        
        # Cap at maximum delay
        delay = min(delay, self.MAX_DELAY)
        
        # Add random jitter (±25%)
        jitter = random.uniform(0.75, 1.25)
        final_delay = delay * jitter
        
        return final_delay
    
    def _generate_interpretation_name(self, md_meters: float) -> str:
        """Generate interpretation name with MD (convert to feet if project is in feet)

        Args:
            md_meters: MD value in meters (internal format)

        Returns:
            Interpretation name with MD suffix
        """
        # PAPI expects interpretation names with MD in project units
        if self.project_measure_unit == 'FOOT':
            # Convert meters to feet for display in name
            md_display = md_meters / 0.3048
        else:
            # Project in meters - use as is
            md_display = md_meters

        # Round to integer and pad with leading zeros to 5 digits
        md_str = f"{int(round(md_display)):05d}"
        if self.generate_multiple_interpretations:
            return f"{self.base_interpretation_name}_{md_str}"
        else:
            return self.base_interpretation_name
    
    def upload_interpretation(self, well_data: Dict, interpretation_data: Dict, current_md: float) -> None:
        """Main upload method with retry logic - uses fixed_name approach to preserve interpretation and update segments"""
        well_name = well_data['wellName']

        # Generate interpretation name from current_md
        interpretation_name = self._generate_interpretation_name(current_md)

        for attempt in range(1, self.MAX_RETRY_ATTEMPTS + 1):
            try:
                self._ensure_connection()
                # Use fixed_name method which preserves interpretation and updates segments
                return self.upload_fixed_name_interpretation(well_data, interpretation_data, interpretation_name)

            except Exception as e:
                if self._is_connection_error(e) and attempt < self.MAX_RETRY_ATTEMPTS:
                    # Connection error - recreate connection and retry
                    self.api = None
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(f"PAPI upload failed for {well_name} (attempt {attempt}/{self.MAX_RETRY_ATTEMPTS}): {str(e)}")
                    logger.info(f"Retrying after {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    # Non-retryable error or max attempts reached
                    if attempt >= self.MAX_RETRY_ATTEMPTS:
                        logger.error(f"PAPI upload failed for {well_name} after {self.MAX_RETRY_ATTEMPTS} attempts: {str(e)}")
                    else:
                        logger.error(f"PAPI upload failed for {well_name} (non-retryable error): {str(e)}")
                    raise
    
    def _do_upload(self, well_data: Dict, interpretation_data: Dict, current_md: float) -> None:
        """Perform the actual upload"""
        well_name = well_data['wellName']
        segments = interpretation_data['interpretation']['segments']
        max_md = self._get_max_md(well_data)
        
        # Generate interpretation name with MD in feet
        interpretation_name = self._generate_interpretation_name(current_md)
        
        logger.info(f"Uploading {len(segments)} segments to PAPI for {well_name}, interpretation: {interpretation_name}")
        
        # Step 1: Find project
        project = self.api.get_project_by_name(self.project_name)
        assert project, f"Project '{self.project_name}' not found"
        project_uuid = project['uuid']
        
        # Step 2: Find well
        well = self.api.get_well_by_name(project_uuid, well_name)
        assert well, f"Well '{well_name}' not found"
        well_uuid = well['uuid']
        
        # Step 3: Interpretations already cleaned up at start of well processing
        logger.info(f"Skipping cleanup - interpretations for {well_name} already cleaned at start")

        # Step 3.5: Delete interpretation with exact name if exists
        deleted_count = self.api.delete_interpretations_by_prefix(well_uuid, interpretation_name, exact_name=True)
        if deleted_count > 0:
            logger.info(f"Deleted existing interpretation '{interpretation_name}' before creating new one")

        # Step 4: Create new interpretation
        interpretation_uuid = self.api.create_interpretation(well_uuid, interpretation_name)
        
        # Step 5: Convert and upload segments
        papi_segments = self._convert_segments_to_papi(segments, max_md)
        
        logger.info(f"Starting to add {len(papi_segments)} segments...")
        for i, segment_data in enumerate(papi_segments):
            logger.info(f"Adding segment {i+1}/{len(papi_segments)}: MD={segment_data['md']['val']}")
            segment_uuid = self.api.add_segment(interpretation_uuid, segment_data)
            logger.debug(f"Added segment {i+1}/{len(papi_segments)}: {segment_uuid}")
        
        logger.info(f"Successfully uploaded {len(papi_segments)} segments to PAPI interpretation '{interpretation_name}'")
    
    
    def _extract_md_value(self, md_data):
        """Extract numeric MD value from dict or return as-is if already numeric"""
        if isinstance(md_data, dict) and 'val' in md_data:
            return md_data['val']
        return md_data
    
    def _convert_segments_to_papi(self, segments: List[Dict], max_md: float) -> List[Dict]:
        """Convert emulator segments to PAPI format with feet conversion"""
        papi_segments = []
        
        for i, segment in enumerate(segments):
            # Extract numeric MD values
            start_md = self._extract_md_value(segment['startMd'])
            
            # Calculate endMd
            if i + 1 < len(segments):
                end_md = self._extract_md_value(segments[i + 1]['startMd'])
            else:
                end_md = self._extract_md_value(max_md)
            
            # Calculate throw
            if i == 0:
                throw = segment['startShift']
            else:
                throw = segment['startShift'] - segments[i-1]['endShift']
            
            # Calculate dip using specified formula: arcsin(-delta_shift / delta_md) + 90
            delta_shift = segment['endShift'] - segment['startShift']
            delta_md = end_md - start_md

            # Skip segments that are too short (< 10 meters) to avoid PAPI API errors
            if delta_md < 10.0:
                logger.debug(f"Skipping segment {i+1}: MD length {delta_md:.3f}m is too short (< 10m)")
                continue

            dip_radians = math.asin(-delta_shift / delta_md)
            dip_degrees = math.degrees(dip_radians) + 90.0

            # Convert MD and throw to PAPI units (feet if project is in feet)
            # IMPORTANT: PAPI expects values in PROJECT units, NOT always feet!
            if self.project_measure_unit == 'FOOT':
                # Project in feet → convert internal meters to feet
                md_papi = start_md / 0.3048
                throw_papi = throw / 0.3048
                units_str = "ft"
            else:
                # Project in meters → use as is
                md_papi = start_md
                throw_papi = throw
                units_str = "m"

            # Create PAPI segment
            papi_segment = {
                "md": {"val": md_papi},
                "dip": {"val": dip_degrees},  # Dip stays in degrees
                "throw": {"val": throw_papi}
            }

            papi_segments.append(papi_segment)

            logger.debug(f"Segment {i+1}: MD={md_papi:.1f}{units_str}, dip={dip_degrees:.2f}°, throw={throw_papi:.3f}{units_str}")
        
        return papi_segments
    
    def _get_max_md(self, well_data: Dict) -> float:
        """Extract maximum MD from well data"""
        points = well_data.get('well', {}).get('points', [])
        if points:
            return max(point['measuredDepth'] for point in points)
        return 10000.0  # Fallback
    
    def upload_fixed_name_interpretation(self, well_data: Dict, interpretation_data: Dict, 
                                       interpretation_name: str) -> str:
        """
        Upload interpretation with fixed name - create new or update existing
        
        Args:
            well_data: Well data dictionary
            interpretation_data: Interpretation data with segments
            interpretation_name: Fixed name for interpretation (e.g. 'AI_geosteering')
            
        Returns:
            str: UUID of the interpretation (created or updated)
        """
        well_name = well_data['wellName']
        segments = interpretation_data['interpretation']['segments']
        max_md = self._get_max_md(well_data)
        
        logger.info(f"Updating fixed-name interpretation '{interpretation_name}' with {len(segments)} segments for {well_name}")
        
        for attempt in range(1, self.MAX_RETRY_ATTEMPTS + 1):
            try:
                self._ensure_connection()
                return self._do_fixed_name_upload(well_data, segments, max_md, interpretation_name)
                
            except Exception as e:
                if self._is_connection_error(e) and attempt < self.MAX_RETRY_ATTEMPTS:
                    # Connection error - recreate connection and retry
                    self.api = None
                    delay = self._calculate_retry_delay(attempt)
                    logger.warning(f"Fixed-name upload failed for {well_name} (attempt {attempt}/{self.MAX_RETRY_ATTEMPTS}): {str(e)}")
                    logger.info(f"Retrying after {delay:.1f} seconds...")
                    time.sleep(delay)
                    continue
                else:
                    # Non-retryable error or max attempts reached
                    if attempt >= self.MAX_RETRY_ATTEMPTS:
                        logger.error(f"Fixed-name upload failed for {well_name} after {self.MAX_RETRY_ATTEMPTS} attempts: {str(e)}")
                    else:
                        logger.error(f"Fixed-name upload failed for {well_name} (non-retryable error): {str(e)}")
                    raise
    
    def _do_fixed_name_upload(self, well_data: Dict, segments: List[Dict], max_md: float, 
                            interpretation_name: str) -> str:
        """Perform fixed-name interpretation upload"""
        well_name = well_data['wellName']
        
        # Step 1: Find project
        project = self.api.get_project_by_name(self.project_name)
        assert project, f"Project '{self.project_name}' not found"
        project_uuid = project['uuid']
        
        # Step 2: Find well  
        well = self.api.get_well_by_name(project_uuid, well_name)
        assert well, f"Well '{well_name}' not found"
        well_uuid = well['uuid']
        
        # Step 3: Find existing interpretation by name
        interpretation_uuid = self._find_interpretation_by_name(well_uuid, interpretation_name)
        
        if interpretation_uuid:
            logger.info(f"Found existing interpretation '{interpretation_name}': {interpretation_uuid}")

            # Get existing segments
            existing_segments = self.api.get_segments_by_interpretation(interpretation_uuid)

            if existing_segments:
                # Sort by MD
                segments_sorted = sorted(existing_segments, key=lambda s: s.get('md', {}).get('val', 0))

                # Delete all except first
                for segment in segments_sorted[1:]:
                    try:
                        self.api.delete_segment(interpretation_uuid, segment['uuid'])
                    except Exception:
                        pass

                logger.info(f"Kept first segment for update, deleted {len(segments_sorted)-1} segments")
            else:
                segments_sorted = []
        else:
            logger.info(f"Creating new interpretation '{interpretation_name}'")
            interpretation_uuid = self.api.create_interpretation(well_uuid, interpretation_name)
            logger.info(f"Created new interpretation '{interpretation_name}': {interpretation_uuid}")
            segments_sorted = []

        # Step 4: Convert and upload segments
        papi_segments = self._convert_segments_to_papi(segments, max_md)

        logger.info(f"Uploading {len(papi_segments)} segments to '{interpretation_name}'...")
        for i, segment_data in enumerate(papi_segments):
            if i == 0 and segments_sorted:
                # First segment → UPDATE (PATCH)
                segment_data['uuid'] = segments_sorted[0]['uuid']
                segment_uuid = self.api.update_segment(interpretation_uuid, segment_data)
                logger.debug(f"Updated segment 1/{len(papi_segments)}: MD={segment_data['md']['val']}, UUID={segment_uuid}")
            else:
                # Rest → ADD (POST)
                segment_uuid = self.api.add_segment(interpretation_uuid, segment_data)
                logger.debug(f"Added segment {i+1}/{len(papi_segments)}: MD={segment_data['md']['val']}, UUID={segment_uuid}")

        logger.info(f"Successfully updated interpretation '{interpretation_name}' with {len(papi_segments)} segments")
        return interpretation_uuid
    
    def _find_interpretation_by_name(self, well_uuid: str, interpretation_name: str) -> Optional[str]:
        """Find interpretation by exact name match"""
        interpretations = self.api.get_interpretations_by_well(well_uuid)
        
        for interpretation in interpretations:
            if interpretation['name'] == interpretation_name:
                logger.debug(f"Found interpretation '{interpretation_name}': {interpretation['uuid']}")
                return interpretation['uuid']
        
        logger.debug(f"Interpretation '{interpretation_name}' not found")
        return None