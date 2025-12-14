# File path: self_correlation/typewell_to_json_converter.py
"""
TypeWell to JSON Converter
Path: self_correlation/typewell_to_json_converter.py

Converts modified TypeWell object back to JSON format for saving
"""

import numpy as np
import logging
from typing import Dict, Any, List
from ag_objects.ag_obj_typewell import TypeWell

logger = logging.getLogger(__name__)


def convert_typewell_to_json(typewell: TypeWell, original_well_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert modified TypeWell object back to JSON typeLog format
    Creates both tvdSortedPoints and points arrays from modified TypeWell
    
    Uses MD-TVD mapping from original typeLog (reference well trajectory)
    
    Args:
        typewell: Modified TypeWell object with updated values
        original_well_data: Original well data (contains typeLog with MD-TVD mapping)
        
    Returns:
        Dictionary in typeLog JSON format ready for insertion into well data
    """
    # Get original typeLog for MD-TVD mapping
    original_type_log = original_well_data.get('typeLog', {})
    
    # Extract MD-TVD mapping from original typeLog
    original_md_tvd_map = {}
    
    # Try to get mapping from tvdSortedPoints first (it's more complete)
    if 'tvdSortedPoints' in original_type_log:
        for point in original_type_log['tvdSortedPoints']:
            if 'measuredDepth' in point and 'trueVerticalDepth' in point:
                tvd = point['trueVerticalDepth']
                md = point['measuredDepth']
                original_md_tvd_map[tvd] = md
    
    # If no measuredDepth in tvdSortedPoints, try points array
    if not original_md_tvd_map and 'points' in original_type_log:
        for point in original_type_log['points']:
            if 'measuredDepth' in point and 'trueVerticalDepth' in point:
                tvd = point['trueVerticalDepth']
                md = point['measuredDepth']
                original_md_tvd_map[tvd] = md
    
    # Create sorted arrays for interpolation
    if original_md_tvd_map:
        original_tvds = np.array(sorted(original_md_tvd_map.keys()))
        original_mds = np.array([original_md_tvd_map[tvd] for tvd in original_tvds])
        logger.info(f"📊 Original typeLog MD-TVD mapping loaded: {len(original_tvds)} points")
        logger.info(f"   MD range: {original_mds.min():.2f}m → {original_mds.max():.2f}m")
        logger.info(f"   TVD range: {original_tvds.min():.2f}m → {original_tvds.max():.2f}m")
    else:
        logger.warning("⚠️ No MD-TVD mapping found in original typeLog, using TVD=MD")
        original_tvds = None
        original_mds = None
    
    # Log original typeLog TVD range for comparison
    if 'tvdSortedPoints' in original_type_log:
        original_tvd_points = original_type_log['tvdSortedPoints']
        if original_tvd_points:
            original_min_tvd = min(p['trueVerticalDepth'] for p in original_tvd_points)
            original_max_tvd = max(p['trueVerticalDepth'] for p in original_tvd_points)
            original_count = len(original_tvd_points)
            
            # Find first and last non-null data points
            original_data_points = [p for p in original_tvd_points if p.get('data') is not None]
            if original_data_points:
                original_first_data_tvd = original_data_points[0]['trueVerticalDepth']
                original_last_data_tvd = original_data_points[-1]['trueVerticalDepth']
            else:
                original_first_data_tvd = None
                original_last_data_tvd = None
            
            logger.info(f"📊 ORIGINAL TypeLog TVD range:")
            logger.info(f"   Total TVD range: {original_min_tvd:.2f}m → {original_max_tvd:.2f}m")
            logger.info(f"   Points count: {original_count}")
            if original_first_data_tvd and original_last_data_tvd:
                logger.info(f"   Data (gamma) range: {original_first_data_tvd:.2f}m → {original_last_data_tvd:.2f}m")
                logger.info(f"   Data span: {original_last_data_tvd - original_first_data_tvd:.2f}m")
    
    # Log modified TypeWell TVD range
    logger.info(f"📊 MODIFIED TypeWell TVD range:")
    logger.info(f"   Total TVD range: {typewell.tvd.min():.2f}m → {typewell.tvd.max():.2f}m")
    logger.info(f"   Points count: {len(typewell.tvd)}")
    
    # Find first and last non-NaN data points
    non_nan_mask = ~np.isnan(typewell.value)
    if np.any(non_nan_mask):
        non_nan_indices = np.where(non_nan_mask)[0]
        first_data_idx = non_nan_indices[0]
        last_data_idx = non_nan_indices[-1]
        first_data_tvd = typewell.tvd[first_data_idx]
        last_data_tvd = typewell.tvd[last_data_idx]
        logger.info(f"   Data (gamma) range: {first_data_tvd:.2f}m → {last_data_tvd:.2f}m")
        logger.info(f"   Data span: {last_data_tvd - first_data_tvd:.2f}m")
    
    # Check for TVD shift
    if 'tvdSortedPoints' in original_type_log and original_tvd_points:
        tvd_shift = typewell.tvd.min() - original_min_tvd
        if abs(tvd_shift) > 0.01:
            logger.warning(f"⚠️ TVD SHIFT detected: {tvd_shift:.2f}m (positive = shifted down)")
    
    # Create tvdSortedPoints array from modified TypeWell
    tvd_sorted_points = []
    for i in range(len(typewell.tvd)):
        tvd = typewell.tvd[i]
        value = typewell.value[i]
        
        # Interpolate MD from original typeLog MD-TVD mapping
        if original_tvds is not None and original_mds is not None:
            # Only interpolate if TVD is within original range
            if tvd >= original_tvds[0] and tvd <= original_tvds[-1]:
                md = np.interp(tvd, original_tvds, original_mds)
            else:
                # Extrapolate linearly if outside range
                if tvd < original_tvds[0]:
                    # Use slope from first two points
                    slope = (original_mds[1] - original_mds[0]) / (original_tvds[1] - original_tvds[0])
                    md = original_mds[0] + slope * (tvd - original_tvds[0])
                else:
                    # Use slope from last two points
                    slope = (original_mds[-1] - original_mds[-2]) / (original_tvds[-1] - original_tvds[-2])
                    md = original_mds[-1] + slope * (tvd - original_tvds[-1])
        else:
            # Fallback: MD = TVD if no mapping available
            md = tvd
        
        point = {
            "measuredDepth": float(md),
            "trueVerticalDepth": float(tvd),
            "data": float(value) if not np.isnan(value) else None
        }
        tvd_sorted_points.append(point)
    
    # Create points array - identical to tvdSortedPoints
    points = []
    for point in tvd_sorted_points:
        points.append(point.copy())
    
    logger.info(f"📊 RESULT TypeLog:")
    logger.info(f"   tvdSortedPoints count: {len(tvd_sorted_points)}")
    logger.info(f"   points count: {len(points)}")
    if tvd_sorted_points:
        result_md_min = min(p['measuredDepth'] for p in tvd_sorted_points)
        result_md_max = max(p['measuredDepth'] for p in tvd_sorted_points)
        logger.info(f"   Result MD range: {result_md_min:.2f}m → {result_md_max:.2f}m")
    
    # Create complete typeLog structure
    type_log = {
        "tvdSortedPoints": tvd_sorted_points,
        "points": points
    }
    
    return type_log


def update_well_data_with_modified_typewell(well_data: Dict[str, Any], 
                                           modified_typewell: TypeWell) -> Dict[str, Any]:
    """
    Update well data JSON with modified TypeWell values
    
    Args:
        well_data: Original well data dictionary
        modified_typewell: Modified TypeWell object after curve replacement
        
    Returns:
        Updated well data with modified typeLog
    """
    # Convert TypeWell to JSON format using original typeLog for MD-TVD mapping
    new_type_log = convert_typewell_to_json(modified_typewell, well_data)
    
    # Replace typeLog in well data
    well_data['typeLog'] = new_type_log
    
    return well_data