# File: papi_export/fetchers/tops_fetcher.py

import logging
from typing import Dict, List, Optional, Any

from ..utils import extract_val

logger = logging.getLogger(__name__)


class TopsFetcher:
    """Fetcher for tops (markers) data from topsets"""

    def __init__(self, config, api_connector):
        self.config = config
        self.api = api_connector

    def fetch_tops_data(
            self, well_uuid: str, trajectory_points: List[Dict]
    ) -> List[Dict]:
        """Fetch tops data for well"""
        logger.info(f"Fetching tops for well {well_uuid}")

        # Get topsets for well
        topsets = self.api.get_topsets_by_well(well_uuid)

        if not topsets:
            logger.warning("No topsets found for well")
            return []

        # Use first topset (or could look for specific name)
        topset = topsets[0]
        topset_uuid = topset['uuid']
        topset_name = topset['name']

        logger.info(f"Using topset '{topset_name}' ({topset_uuid})")

        # Get tops from topset
        tops = self.api.get_tops_by_topset(topset_uuid)

        if not tops:
            logger.warning("No tops found in topset")
            return []

        # Process tops - calculate TVD from MD
        processed_tops = []

        for top in tops:
            top_uuid = top['uuid']
            top_name = top['name']

            # Handle PAPI data structure - md is object with 'val' field
            top_md = extract_val(top['md'])
            if top_md is None:
                logger.warning(f"Top '{top_name}' has undefined MD - skipping")
                continue

            # Calculate TVD from trajectory
            tvd = self._interpolate_tvd(top_md, trajectory_points)

            if tvd is None:
                logger.warning(f"Could not calculate TVD for top '{top_name}' at MD {top_md}")
                tvd = 0.0  # Default value

            processed_tops.append({
                'uuid': top_uuid,
                'name': top_name,
                'measuredDepth': top_md,
                'trueVerticalDepth': tvd
            })

        logger.info(f"Processed {len(processed_tops)} tops")
        return processed_tops

    def _interpolate_tvd(self, md: float, trajectory_points: List[Dict]) -> Optional[float]:
        """Interpolate TVD for given MD from trajectory"""

        if not trajectory_points:
            return None

        # Find bracketing points
        left_point = None
        right_point = None

        for point in trajectory_points:
            # trajectory_points from simple_trajectory - already converted to numbers
            point_md = point['md']
            if point_md <= md:
                left_point = point
            if point_md >= md and right_point is None:
                right_point = point
                break

        # Handle edge cases
        if left_point is None:
            return trajectory_points[0]['tvd']
        if right_point is None:
            return trajectory_points[-1]['tvd']

        # If exact match
        if left_point['md'] == md:
            return left_point['tvd']
        if right_point['md'] == md:
            return right_point['tvd']

        # Linear interpolation
        if left_point['md'] < right_point['md']:
            ratio = (md - left_point['md']) / (right_point['md'] - left_point['md'])
            tvd = left_point['tvd'] + ratio * (right_point['tvd'] - left_point['tvd'])
            return tvd

        return left_point['tvd']