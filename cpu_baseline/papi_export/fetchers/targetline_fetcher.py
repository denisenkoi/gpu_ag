import logging
from typing import Dict, List, Optional, Any
import numpy as np

from ..utils import extract_val

logger = logging.getLogger(__name__)


class TargetLineFetcher:
    """Fetcher for target line data from PAPI API"""
    
    def __init__(self, config, api_connector):
        self.config = config
        self.api = api_connector
        
    def fetch_target_line_data(self, well_uuid: str) -> Optional[Dict[str, Any]]:
        """Fetch starred target line data for well"""
        logger.info(f"Fetching target line for well {well_uuid}")

        # DIAGNOSTIC: Log API connector type
        logger.debug(f"Using API connector: {type(self.api).__name__}")

        # Get starred target line directly using dedicated endpoint
        starred_target_line = self._get_starred_target_line(well_uuid)

        # DIAGNOSTIC: Log response details
        if starred_target_line:
            logger.debug(f"Starred target line response keys: {list(starred_target_line.keys())}")
        else:
            logger.debug("No starred target line received")

        if not starred_target_line or not starred_target_line.get('uuid'):
            logger.warning("No starred target line found for well")
            return None

        target_line_uuid = starred_target_line['uuid']

        # Skip null UUID
        if target_line_uuid == "00000000-0000-0000-0000-000000000000":
            logger.warning("Target line UUID is null")
            return None

        logger.info(f"Using starred target line: {target_line_uuid}")

        # Get target line calculated data
        target_line_data = self._get_target_line_data(target_line_uuid)

        if not target_line_data:
            logger.warning("Could not fetch target line data")
            return None

        # RANGE_DEBUG: Log raw PAPI response before any processing
        logger.info("=" * 60)
        logger.info("RANGE_DEBUG: TargetLineFetcher - Raw PAPI Response")
        logger.info(f"RANGE_DEBUG: Raw target_line_data keys: {list(target_line_data.keys())}")
        if 'content' in target_line_data and target_line_data['content']:
            raw_data = target_line_data['content'][0]
            raw_origin_vs = extract_val(raw_data.get('origin_vs', {}))
            raw_target_vs = extract_val(raw_data.get('target_vs', {}))
            raw_origin_tvd = extract_val(raw_data.get('origin_tvd', {}))
            raw_target_tvd = extract_val(raw_data.get('target_tvd', {}))
            logger.info(f"RANGE_DEBUG: RAW PAPI VS Range: [{raw_origin_vs:.1f}, {raw_target_vs:.1f}] (span: {abs(raw_target_vs - raw_origin_vs):.1f})")
            logger.info(f"RANGE_DEBUG: RAW PAPI TVD Range: [{raw_origin_tvd:.1f}, {raw_target_tvd:.1f}] (span: {abs(raw_target_tvd - raw_origin_tvd):.1f})")
        logger.info("=" * 60)

        # Extract target line parameters
        target_info = self._extract_target_line_info(target_line_data)

        logger.info(f"Target line: {target_info['name']}, length={target_info['length']:.1f}")
        logger.info(f"Origin: VS={target_info['origin_vs']:.1f}, TVD={target_info['origin_tvd']:.1f}")
        logger.info(f"Target: VS={target_info['target_vs']:.1f}, TVD={target_info['target_tvd']:.1f}")

        return target_info

    def _get_starred_target_line(self, well_uuid: str) -> Optional[Dict]:
        """Get starred target line for well using DataFetchConnector method"""
        try:
            return self.api.get_starred_target_line(well_uuid)
        except Exception as e:
            logger.warning(f"Failed to get starred target line: {e}")
            return None

    def _get_target_line_data(self, target_line_uuid: str) -> Optional[Dict]:
        """Get target line calculated data using DataFetchConnector method"""
        try:
            return self.api.get_target_line_data(target_line_uuid)
        except Exception as e:
            logger.warning(f"Failed to get target line data: {e}")
            return None
        
    def _extract_target_line_info(self, target_line_data: Dict) -> Dict[str, Any]:
        """Extract relevant target line information"""

        # DEBUG: Log full target line data structure
        logger.info(f"TARGET_LINE_DEBUG: Full target_line_data keys: {list(target_line_data.keys())}")
        logger.info(f"TARGET_LINE_DEBUG: Full target_line_data content: {target_line_data}")

        # Extract actual target line data from content array
        if 'content' not in target_line_data or not target_line_data['content']:
            raise ValueError("Target line data missing 'content' field or content is empty")

        target_info = target_line_data['content'][0]  # First element contains target line info

        # Target line basic info
        name = target_info['name']
        uuid = target_info['uuid']
        length = extract_val(target_info['length'])
        
        # Origin point data
        origin_vs = extract_val(target_info['origin_vs'])
        origin_tvd = extract_val(target_info['origin_tvd'])
        origin_md = extract_val(target_info['origin_md'])

        # Target point data
        target_vs = extract_val(target_info['target_vs'])
        target_tvd = extract_val(target_info['target_tvd'])
        target_md = extract_val(target_info['target_md'])
        
        # Calculate target line slope (TVD vs VS)
        vs_span = target_vs - origin_vs
        tvd_span = target_tvd - origin_tvd
        
        slope = tvd_span / vs_span if abs(vs_span) > 0.001 else 0.0

        # RANGE_DEBUG: Log extracted ranges before returning
        logger.info("RANGE_DEBUG: _extract_target_line_info - After Processing")
        logger.info(f"RANGE_DEBUG: EXTRACTED VS Range: [{origin_vs:.1f}, {target_vs:.1f}] (span: {abs(vs_span):.1f})")
        logger.info(f"RANGE_DEBUG: EXTRACTED TVD Range: [{origin_tvd:.1f}, {target_tvd:.1f}] (span: {abs(tvd_span):.1f})")
        logger.info(f"RANGE_DEBUG: Length: {length:.1f}, Slope: {slope:.6f}")

        return {
            'uuid': uuid,
            'name': name,
            'length': length,
            'origin_vs': origin_vs,
            'origin_tvd': origin_tvd,
            'origin_md': origin_md,
            'target_vs': target_vs,
            'target_tvd': target_tvd,
            'target_md': target_md,
            'slope': slope,
            'vs_span': vs_span,
            'tvd_span': tvd_span
        }
        
    def interpolate_target_tvd_at_vs(self, target_line_info: Dict, vs_coordinate: float) -> Optional[float]:
        """Interpolate target TVD at given VS coordinate using linear interpolation"""

        origin_vs = target_line_info['origin_vs']
        target_vs = target_line_info['target_vs']
        origin_tvd = target_line_info['origin_tvd']
        target_tvd = target_line_info['target_tvd']

        # RANGE_DEBUG: Log final ranges used in interpolation
        logger.info("RANGE_DEBUG: interpolate_target_tvd_at_vs - Final Ranges for Interpolation")
        logger.info(f"RANGE_DEBUG: INTERPOLATION VS Range: [{origin_vs:.1f}, {target_vs:.1f}] (span: {abs(target_vs - origin_vs):.1f})")
        logger.info(f"RANGE_DEBUG: INTERPOLATION TVD Range: [{origin_tvd:.1f}, {target_tvd:.1f}] (span: {abs(target_tvd - origin_tvd):.1f})")
        logger.info(f"RANGE_DEBUG: VS coordinate to check: {vs_coordinate:.1f}")

        # Check if VS is within target line range
        vs_min = min(origin_vs, target_vs)
        vs_max = max(origin_vs, target_vs)
        
        if vs_coordinate < vs_min or vs_coordinate > vs_max:
            logger.warning(f"VS coordinate {vs_coordinate:.1f} outside target line range [{vs_min:.1f}, {vs_max:.1f}]")
            return None
            
        # Linear interpolation between origin and target
        vs_span = target_vs - origin_vs
        
        if abs(vs_span) < 0.001:
            # Vertical target line - return average TVD
            return (origin_tvd + target_tvd) / 2.0
            
        # Calculate interpolation ratio
        ratio = (vs_coordinate - origin_vs) / vs_span
        
        # Interpolate TVD
        interpolated_tvd = origin_tvd + ratio * (target_tvd - origin_tvd)
        
        logger.debug(f"Target line interpolation: VS={vs_coordinate:.1f} → TVD={interpolated_tvd:.3f}")
        
        return interpolated_tvd
