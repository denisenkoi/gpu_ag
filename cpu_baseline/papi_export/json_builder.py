import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import numpy as np
from .build_interpretation import build_interpretation_from_shifts

logger = logging.getLogger(__name__)


class JSONBuilder:
    """Build final JSON structure compatible with C++ AutoGeosteering format with units conversion"""
    
    # Conversion factor from feet to meters
    FEET_TO_METERS = 0.3048
    
    def __init__(self, config):
        self.config = config
        
    def build_complete_json(
        self,
        lateral_data: Dict,
        typewell_data: Optional[Dict],
        tops_data: list,
        interpretation_data: Optional[Dict],
        grid_slice_data: Optional[Dict]
    ) -> Dict[str, Any]:
        """Build complete JSON structure as in C++ prepareInitialStageData with units conversion"""
        
        logger.info("Building final JSON structure")
        
        # Start with well data from lateral
        result = {
            'well': lateral_data['well'],
            'wellLog': lateral_data['wellLog']
        }
        
        # Add typeLog if available
        if typewell_data:
            result['typeLog'] = typewell_data['typeLog']
            result['tvdTypewellShift'] = typewell_data['shift']
        else:
            result['typeLog'] = {
                'uuid': '',
                'points': [],
                'tvdSortedPoints': []
            }
            result['tvdTypewellShift'] = 0.0
            
        # Add tops
        result['tops'] = tops_data if tops_data else []
        
        # Add gridSlice
        if grid_slice_data:
            result['gridSlice'] = grid_slice_data
        else:
            result['gridSlice'] = {
                'uuid': '',
                'points': []
            }
            
        # Add autoGeosteeringParameters - from config (NO DEFAULTS - must be in .env)
        result['autoGeosteeringParameters'] = {
            'startMd': self._find_start_md_from_interpretation(lateral_data),
            'lookBackDistance': self.config.lookback_distance,
            'dipAngleRangeDegree': self.config.dip_angle_range_degree,
            'dipStepDegree': self.config.dip_step_degree,
            'dipRegionAvgDegree': self.config.regional_dip_angle,
            'smoothness': self.config.smoothness,
            'isLogNormalizationEnabled': self.config.enable_log_normalization,
            'isOnlyVerticalTrackEnabled': self.config.vertical_track_only
        }
        
        # Add well name and other metadata
        result['wellName'] = lateral_data['well']['name']
        result['startMd'] = result['autoGeosteeringParameters']['startMd']
        
        # Add metadata for reference
        result['_metadata'] = {
            'export_timestamp': datetime.now().isoformat(),
            'project_name': self.config.project_name,
            'well_name': self.config.well_name,
            'horizon_name': self.config.horizon_name,
            'grid_name': self.config.grid_name
        }
        
        # ============================================
        # CALCULATE SHIFTS FROM TVT - TVD
        # ============================================
        shifts_data = []
        
        if interpretation_data and 'horizons_raw_data' in interpretation_data:
            # Determine TVT calculation method
            use_alternative = self.config.use_alternative_tvt_calculation
            calculation_method = "ALTERNATIVE (TVDSS-KB+TVT_api)" if use_alternative else "STANDARD (TVT from API)"
            logger.info(f"Calculating shifts using {calculation_method} method")

            # Get well points for interpolation
            well_points = lateral_data['well']['points']
            well_md = np.array([p['measuredDepth'] for p in well_points])
            well_tvd = np.array([p['trueVerticalDepth'] for p in well_points])

            # Extract horizon data
            horizon_content = interpretation_data['horizons_raw_data']['content']

            # Track TVT values for summary
            tvt_values = []
            first_tvt = None

            # Process all horizon points
            for i, h_point in enumerate(horizon_content):
                md = h_point['md']['val']
                tvt_api = h_point['tvt']['val']

                # Calculate TVT based on method
                if use_alternative:
                    # ALTERNATIVE: Find target horizon and calculate TVT from TVDSS
                    kb = h_point['kb']['val']
                    target_horizon_tvdss = None

                    for horizon in h_point['horizons']:
                        if horizon['name'] == self.config.horizon_name:
                            target_horizon_tvdss = horizon['tvdss']['val']
                            break

                    if target_horizon_tvdss is None:
                        raise ValueError(
                            f"Target horizon '{self.config.horizon_name}' not found at MD={md} (point {i+1}/{len(horizon_content)}). "
                            f"Available horizons: {[h['name'] for h in h_point['horizons']]}"
                        )

                    # Formula: TVT = -[(TVDSS_target - KB) - TVT_api]
                    # Note: subtract TVT_api (which is negative, so adds to absolute value)
                    # then flip sign because TVT convention is positive downward
                    tvt = -((target_horizon_tvdss - kb) - tvt_api)
                else:
                    # STANDARD: Use TVT from API as-is
                    tvt = tvt_api

                # Track for summary
                tvt_values.append(tvt)
                if first_tvt is None:
                    first_tvt = tvt

                # Interpolate well TVD at this MD
                if md <= well_md[0]:
                    well_tvd_at_md = well_tvd[0]
                elif md >= well_md[-1]:
                    well_tvd_at_md = well_tvd[-1]
                else:
                    # Linear interpolation
                    well_tvd_at_md = np.interp(md, well_md, well_tvd)

                # Calculate shift as TVT - TVD
                shift = tvt - well_tvd_at_md

                shifts_data.append({
                    'md': md,
                    'shift': shift
                })

            # Log summary
            logger.info(f"Calculated {len(shifts_data)} shift points using {calculation_method}")
            if tvt_values:
                tvt_array = np.array(tvt_values)
                logger.info(f"TVT range: {tvt_array.min():.3f} to {tvt_array.max():.3f}, first TVT: {first_tvt:.3f}")
            if shifts_data:
                shifts_array = np.array([s['shift'] for s in shifts_data])
                logger.info(f"Shifts range: {shifts_array.min():.3f} to {shifts_array.max():.3f}")
        
        # Add shifts to result
        result['shifts'] = shifts_data
        
        # Build interpretation segments from shifts using new algorithm
        if shifts_data:
            logger.info("Building interpretation segments from shifts")
            interpretation_result = build_interpretation_from_shifts(
                shifts_data, 
                self.config.horizon_name
            )
            
            # Use new segments
            if interpretation_result['segments']:
                result['interpretation'] = {
                    'uuid': interpretation_data.get('interpretation_uuid', '') if interpretation_data else '',
                    'segments': interpretation_result['segments']
                }
                logger.info(f"Built {interpretation_result['segments_count']} segments from shifts")
        else:
            # Empty interpretation
            result['interpretation'] = {
                'uuid': '',
                'segments': []
            }
            
        # Convert units if project is in feet - CRITICAL check!
        project_measure_unit = lateral_data.get('project_measure_unit')
        if not project_measure_unit:
            logger.error("CRITICAL: project_measure_unit missing from lateral_data!")
            raise ValueError("project_measure_unit is required in lateral_data for correct unit conversion")

        if project_measure_unit == 'FOOT':
            logger.info("Converting units from FEET to METERS")
            result = self._convert_units_to_meters(result)
        else:
            logger.info(f"Project units: {project_measure_unit} - no conversion needed")

        return result
        
    def _convert_units_to_meters(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert all distance measurements from feet to meters (except data values like gamma ray)"""
        
        # Convert well.points trajectory
        if 'well' in data and 'points' in data['well']:
            for point in data['well']['points']:
                point['measuredDepth'] = self._convert_value(point['measuredDepth'])
                point['trueVerticalDepth'] = self._convert_value(point['trueVerticalDepth'])
                point['northSouth'] = self._convert_value(point['northSouth'])
                point['eastWest'] = self._convert_value(point['eastWest'])
                
        # Convert well surface coordinates
        if 'well' in data and 'xySurface' in data['well']:
            surface = data['well']['xySurface']
            surface['xSurface'] = self._convert_value(surface['xSurface'])
            surface['ySurface'] = self._convert_value(surface['ySurface'])
            
        # Convert wellLog measured depths and TVDs (NOT data values)
        if 'wellLog' in data:
            self._convert_log_distances(data['wellLog'])
            
        # Convert typeLog measured depths and TVDs (NOT data values)
        if 'typeLog' in data:
            self._convert_log_distances(data['typeLog'])
            
        # Convert typewell shift
        if 'tvdTypewellShift' in data:
            data['tvdTypewellShift'] = self._convert_value(data['tvdTypewellShift'])
            
        # Convert tops
        if 'tops' in data:
            for top in data['tops']:
                top['measuredDepth'] = self._convert_value(top['measuredDepth'])
                top['trueVerticalDepth'] = self._convert_value(top['trueVerticalDepth'])
                
        # Convert interpretation segments
        if 'interpretation' in data and 'segments' in data['interpretation']:
            for segment in data['interpretation']['segments']:
                segment['startMd'] = self._convert_value(segment['startMd'])
                segment['startShift'] = self._convert_value(segment['startShift'])
                segment['endShift'] = self._convert_value(segment['endShift'])
                
        # Convert gridSlice points
        if 'gridSlice' in data and 'points' in data['gridSlice']:
            for point in data['gridSlice']['points']:
                point['measuredDepth'] = self._convert_value(point['measuredDepth'])
                if 'trueVerticalDepthSubSea' in point:
                    point['trueVerticalDepthSubSea'] = self._convert_value(point['trueVerticalDepthSubSea'])
                point['northSouth'] = self._convert_value(point['northSouth'])
                point['eastWest'] = self._convert_value(point['eastWest'])
                point['verticalSection'] = self._convert_value(point['verticalSection'])
                
        # Convert autoGeosteeringParameters
        if 'autoGeosteeringParameters' in data:
            params = data['autoGeosteeringParameters']
            params['startMd'] = self._convert_value(params['startMd'])
            # NOTE: lookBackDistance is NOT converted - it's already in meters from env var
            
        # Convert startMd in root level
        if 'startMd' in data:
            data['startMd'] = self._convert_value(data['startMd'])
            
        # Convert shifts data
        if 'shifts' in data:
            for shift_point in data['shifts']:
                shift_point['md'] = self._convert_value(shift_point['md'])
                shift_point['shift'] = self._convert_value(shift_point['shift'])
            
        logger.info(f"Units conversion completed: FEET → METERS (factor: {self.FEET_TO_METERS})")
        
        return data
        
    def _convert_log_distances(self, log_data: Dict):
        """Convert log distances but NOT data values (gamma ray, etc.)"""

        # Convert points array - only MD, not data
        if 'points' in log_data:
            # DEBUG: Collect data before conversion
            target_md_m = 4206.24  # Target MD in meters
            md_window_m = 90.0  # Wider window to capture duplicates (150ft * 0.3048)
            debug_before = []

            for i, point in enumerate(log_data['points']):
                md_ft = point['measuredDepth']
                md_m_estimated = md_ft * self.FEET_TO_METERS

                # Collect debug data around target MD
                if target_md_m - md_window_m <= md_m_estimated <= target_md_m + md_window_m:
                    debug_before.append({
                        'index': i,
                        'md_ft': md_ft,
                        'md_m_estimated': md_m_estimated,
                        'data': point.get('data')
                    })

                point['measuredDepth'] = self._convert_value(point['measuredDepth'])
                # DO NOT convert 'data' field - it's gamma ray values

            # Log debug data AFTER conversion - DISABLED (too verbose)
            if debug_before:
                logger.debug(f"DEBUG_DUPLICATE: Log conversion - {len(debug_before)} points around MD={target_md_m}m")
                # Log ALL points to see duplicates - COMMENTED OUT (spam)
                # for p in debug_before:
                #     actual_md_m = log_data['points'][p['index']]['measuredDepth']
                #     logger.info(f"  [idx={p['index']}] FEET={p['md_ft']:.10f} -> METERS={actual_md_m:.10f}, data={p['data']}")

                # Check for duplicates in AFTER conversion
                md_after_values = [log_data['points'][p['index']]['measuredDepth'] for p in debug_before]
                unique_after = set(md_after_values)
                if len(md_after_values) != len(unique_after):
                    logger.warning(f"DEBUG_DUPLICATE: DUPLICATES FOUND in AFTER conversion! Total={len(md_after_values)}, Unique={len(unique_after)}")
                    from collections import Counter
                    md_counts_after = Counter(md_after_values)
                    duplicates_after = {md: count for md, count in md_counts_after.items() if count > 1}
                    for dup_md, count in duplicates_after.items():
                        logger.warning(f"  DUPLICATE MD={dup_md:.10f}m appears {count} times AFTER conversion")
                        # Show indices of duplicates
                        dup_indices = [p['index'] for p in debug_before if log_data['points'][p['index']]['measuredDepth'] == dup_md]
                        logger.warning(f"    Duplicate indices: {dup_indices}")
                else:
                    logger.info(f"DEBUG_DUPLICATE: No duplicates in AFTER conversion (all {len(md_after_values)} MD unique)")

        # Convert tvdSortedPoints - only MD and TVD, not data
        if 'tvdSortedPoints' in log_data:
            for point in log_data['tvdSortedPoints']:
                point['measuredDepth'] = self._convert_value(point['measuredDepth'])
                point['trueVerticalDepth'] = self._convert_value(point['trueVerticalDepth'])
                # DO NOT convert 'data' field - it's gamma ray values
                
    def _convert_value(self, value: Union[float, int]) -> float:
        """Convert single numerical value from feet to meters"""
        if isinstance(value, (int, float)):
            return float(value * self.FEET_TO_METERS)
        return value
        
    def get_output_path(self, well_name: str) -> Path:
        """Get output path for a well without saving

        Args:
            well_name: Name of the well

        Returns:
            Path: Expected output file path
        """
        filename = f"{well_name}.json"
        return self.config.output_dir / filename

    def save_json(self, data: Dict[str, Any], filename: Optional[str] = None):
        """Save JSON to file"""

        if not filename:
            # Generate filename from well name without timestamp
            well_name = data.get('wellName', 'unknown')
            filename = f"{well_name}.json"

        output_path = self.config.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        logger.info(f"Saved JSON to {output_path}")
        return output_path
        
    def save_intermediate_raw_data(
        self,
        lateral_data: Dict,
        typewell_data: Optional[Dict],
        tops_data: list,
        interpretation_data: Optional[Dict],
        grid_slice_data: Optional[Dict]
    ):
        """Save intermediate raw data for debugging"""
        
        if not self.config.save_intermediate:
            return
            
        intermediate_dir = self.config.output_dir / 'intermediate'
        
        # Save each component
        components = {
            'lateral': lateral_data,
            'typewell': typewell_data,
            'tops': tops_data,
            'interpretation': interpretation_data,
            'grid_slice': grid_slice_data
        }
        
        for name, data in components.items():
            if data is not None:
                output_file = intermediate_dir / f'{name}_raw.json'
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.debug(f"Saved {name} raw data to {output_file}")
                
    def _find_start_md_from_interpretation(self, lateral_data: Dict) -> float:
        """Find startMd from first interpretation segment"""
        
        # Try to get startMd from first interpretation segment
        interpretation = lateral_data.get('interpretation', {})
        segments = interpretation.get('segments', [])
        
        if segments and isinstance(segments[0], dict):
            first_segment = segments[0]
            
            # Try different possible keys for start MD
            for key in ['startMd', 'md', 'measuredDepth']:
                if key in first_segment:
                    start_md = first_segment[key]
                    logger.info(f"Found startMd from interpretation first segment: {start_md}")
                    return float(start_md)
        
        # Fallback: use 0 if can't find interpretation
        fallback_md = 0.0
        logger.warning(f"Could not find startMd from interpretation, using fallback: {fallback_md}")
        return fallback_md
    
    def _find_start_md(self, lateral_data: Dict) -> float:
        """Find appropriate start MD from well data"""
        
        # Try to find first non-zero MD
        well_points = lateral_data.get('well', {}).get('points', [])
        
        if not well_points:
            return 0.0
            
        # Find first point with significant MD (> 100)
        for point in well_points:
            md = point.get('measuredDepth', 0.0)
            if md > 100.0:
                return md
                
        # Default to first point MD
        return well_points[0].get('measuredDepth', 0.0)