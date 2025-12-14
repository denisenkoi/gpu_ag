import logging
import copy
from typing import Dict, Any, Union, Tuple

logger = logging.getLogger(__name__)


class UnitsConverter:
    """Converts both files from meters to feet for comparison in StarSteer units"""
    
    # Conversion factor
    METERS_TO_FEET = 3.28084  # More precise than 1/0.3048
    
    def __init__(self):
        """Initialize units converter"""
        self.conversion_applied = True  # Always true since we always convert
        
    def convert_both_to_feet(self, reference: Dict, generated: Dict) -> Tuple[Dict, Dict]:
        """Convert both reference and generated files from METERS to FEET
        
        Both files come in METERS (C++ standard) and need to be converted 
        to FEET for comparison in StarSteer display units.
        
        Args:
            reference: Reference data dictionary in METERS
            generated: Generated data dictionary in METERS
            
        Returns:
            Tuple of (reference_in_feet, generated_in_feet)
        """
        logger.info("="*60)
        logger.info("UNITS CONVERSION FOR COMPARISON")
        logger.info("  Input: Both files in METERS (C++ standard)")
        logger.info("  Output: Both files in FEET (StarSteer display)")
        logger.info(f"  Conversion factor: {self.METERS_TO_FEET}")
        logger.info("="*60)
        
        # Convert both files
        reference_feet = self._convert_to_feet(reference)
        generated_feet = self._convert_to_feet(generated)
        
        logger.info("✅ Both files converted to FEET for comparison")
        
        return reference_feet, generated_feet
    
    def _convert_to_feet(self, data: Dict) -> Dict:
        """Convert all distance measurements from meters to feet
        
        Args:
            data: Data dictionary in meters
            
        Returns:
            Data converted to feet
        """
        # Deep copy to avoid modifying original
        data_copy = copy.deepcopy(data)
        
        # Convert well.points trajectory
        if 'well' in data_copy and 'points' in data_copy['well']:
            for point in data_copy['well']['points']:
                point['measuredDepth'] = self._meters_to_feet(point['measuredDepth'])
                point['trueVerticalDepth'] = self._meters_to_feet(point['trueVerticalDepth'])
                point['northSouth'] = self._meters_to_feet(point['northSouth'])
                point['eastWest'] = self._meters_to_feet(point['eastWest'])
        
        # Convert well surface coordinates
        if 'well' in data_copy and 'xySurface' in data_copy['well']:
            surface = data_copy['well']['xySurface']
            surface['xSurface'] = self._meters_to_feet(surface['xSurface'])
            surface['ySurface'] = self._meters_to_feet(surface['ySurface'])
        
        # Convert wellLog
        if 'wellLog' in data_copy:
            self._convert_log_to_feet(data_copy['wellLog'])
        
        # Convert typeLog
        if 'typeLog' in data_copy:
            self._convert_log_to_feet(data_copy['typeLog'])
        
        # Convert typewell shift
        if 'tvdTypewellShift' in data_copy:
            data_copy['tvdTypewellShift'] = self._meters_to_feet(data_copy['tvdTypewellShift'])
        
        # Convert tops
        if 'tops' in data_copy:
            for top in data_copy['tops']:
                top['measuredDepth'] = self._meters_to_feet(top['measuredDepth'])
                top['trueVerticalDepth'] = self._meters_to_feet(top['trueVerticalDepth'])
        
        # Convert interpretation segments
        if 'interpretation' in data_copy and 'segments' in data_copy['interpretation']:
            for segment in data_copy['interpretation']['segments']:
                segment['startMd'] = self._meters_to_feet(segment['startMd'])
                segment['startShift'] = self._meters_to_feet(segment['startShift'])
                segment['endShift'] = self._meters_to_feet(segment['endShift'])
        
        # Convert gridSlice
        if 'gridSlice' in data_copy and 'points' in data_copy['gridSlice']:
            for point in data_copy['gridSlice']['points']:
                point['measuredDepth'] = self._meters_to_feet(point['measuredDepth'])
                if 'trueVerticalDepthSubSea' in point:
                    point['trueVerticalDepthSubSea'] = self._meters_to_feet(point['trueVerticalDepthSubSea'])
                point['northSouth'] = self._meters_to_feet(point['northSouth'])
                point['eastWest'] = self._meters_to_feet(point['eastWest'])
                point['verticalSection'] = self._meters_to_feet(point['verticalSection'])
        
        # Convert autoGeosteeringParameters
        if 'autoGeosteeringParameters' in data_copy:
            params = data_copy['autoGeosteeringParameters']
            params['startMd'] = self._meters_to_feet(params['startMd'])
            params['lookBackDistance'] = self._meters_to_feet(params['lookBackDistance'])
        
        # Convert startMd
        if 'startMd' in data_copy:
            data_copy['startMd'] = self._meters_to_feet(data_copy['startMd'])
        
        # Convert shifts
        if 'shifts' in data_copy:
            for shift_point in data_copy['shifts']:
                shift_point['md'] = self._meters_to_feet(shift_point['md'])
                shift_point['shift'] = self._meters_to_feet(shift_point['shift'])
        
        return data_copy
    
    def _convert_log_to_feet(self, log_data: Dict):
        """Convert log distances to feet (NOT data values like gamma ray)
        
        Args:
            log_data: Log data dictionary
        """
        # Convert points array - only MD, not data values
        if 'points' in log_data:
            for point in log_data['points']:
                point['measuredDepth'] = self._meters_to_feet(point['measuredDepth'])
                # DO NOT convert 'data' field - it's gamma ray or other log values
        
        # Convert tvdSortedPoints - only MD and TVD, not data values
        if 'tvdSortedPoints' in log_data:
            for point in log_data['tvdSortedPoints']:
                point['measuredDepth'] = self._meters_to_feet(point['measuredDepth'])
                point['trueVerticalDepth'] = self._meters_to_feet(point['trueVerticalDepth'])
                # DO NOT convert 'data' field
    
    def _meters_to_feet(self, value: Any) -> Any:
        """Convert single value from meters to feet
        
        Args:
            value: Value in meters
            
        Returns:
            Value in feet
        """
        if isinstance(value, (int, float)):
            return float(value * self.METERS_TO_FEET)
        return value
    
    def get_units_info(self) -> Dict:
        """Get units information for reporting
        
        Returns:
            Dictionary with units information
        """
        return {
            'input_units': 'METERS',
            'output_units': 'FEET',
            'conversion_factor': self.METERS_TO_FEET,
            'conversion_applied': True,
            'comparison_units': 'FEET'
        }