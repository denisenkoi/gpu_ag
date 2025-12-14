"""
Curve Replacement Processor - FIXED version
Path: self_correlation/curve_replacement_processor.py

- No GAP zone logic
- Correct junction point search with crossing detection
- Search distance from .env
- Works with discrete TypeWell points only (no interpolation)
"""

import os
import logging
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from copy import deepcopy

# AG objects imports
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell

logger = logging.getLogger(__name__)


class CurveReplacementProcessor:
    """Processor for replacing TypeWell curve with horizontal well projection"""

    def __init__(self, config: Dict[str, Any] = None, logger_instance=None):
        # Use passed logger or global
        self.logger = logger_instance or logger

        # If config passed, use it. Otherwise try environment variables
        if config:
            self.enabled = config.get('curve_replacement_enabled', False)
            self.replacement_range_meters = config.get('curve_replacement_range_meters', 500.0)
            self.search_distance_meters = config.get('curve_replacement_search_distance_meters', 2.0)
            self.blend_weight = config.get('curve_replacement_blend_weight', 1.0)
        else:
            # Fallback to environment variables
            self.enabled = os.getenv('CURVE_REPLACEMENT_ENABLED', 'false').lower() == 'true'
            self.replacement_range_meters = float(os.getenv('CURVE_REPLACEMENT_RANGE_METERS', '500.0'))
            self.search_distance_meters = float(os.getenv('CURVE_REPLACEMENT_SEARCH_DISTANCE_METERS', '2.0'))
            self.blend_weight = float(os.getenv('CURVE_REPLACEMENT_BLEND_WEIGHT', '1.0'))

        self.logger.info(f"CurveReplacementProcessor initialized: enabled={self.enabled}, "
                        f"range={self.replacement_range_meters}m, "
                        f"search_distance={self.search_distance_meters}m, "
                        f"weight={self.blend_weight}")

    def process_well_data(self,
                         well_data: Dict[str, Any],
                         well: Well,
                         typewell: TypeWell,
                         manual_segments: List,
                         original_start_md: float,
                         detected_start_md: float) -> Dict[str, Any]:
        """
        Main entry point - process well data with pre-created AG objects
        """
        if not self.enabled:
            self.logger.debug("Curve replacement disabled, skipping")
            return {
                'status': 'disabled',
                'typewell_modified': False
            }

        well_name = well_data['wellName']
        self.logger.info(f"🔄 Processing curve replacement for well: {well_name}")

        # Use rightmost MD as analysis range end
        landing_end_md = max(original_start_md, detected_start_md)

        # Check that we have manual segments
        if not manual_segments:
            self.logger.warning("❌ No manual interpretation segments provided")
            return {
                'status': 'failed',
                'reason': 'no_manual_segments',
                'typewell_modified': False
            }

        # Calculate horizontal projection using manual interpretation
        tvd_to_typewell_shift = well_data['tvdTypewellShift']
        self.logger.info(f"📐 Calculating horizontal projection with TVD shift: {tvd_to_typewell_shift:.3f}m")

        # Check if projection already calculated
        if np.all(np.isnan(well.tvt)) or np.all(np.isnan(well.synt_curve)):
            self.logger.info("Calculating new horizontal projection...")
            success = well.calc_horizontal_projection(typewell, manual_segments, tvd_to_typewell_shift)

            if not success:
                self.logger.error("❌ Failed to calculate horizontal projection")
                return {
                    'status': 'failed',
                    'reason': 'projection_calculation_failed',
                    'typewell_modified': False
                }
        else:
            self.logger.info("Using existing horizontal projection from normalization step")

        # Find maximum penetration point
        max_penetration_result = self._find_maximum_penetration_point(
            well, original_start_md, landing_end_md
        )

        if max_penetration_result is None:
            self.logger.warning("⚠️ No maximum penetration point found - no replacement performed")
            return {
                'status': 'failed',
                'reason': 'no_maximum_penetration',
                'typewell_modified': False
            }

        max_penetration_md, max_penetration_idx, max_penetration_tvt = max_penetration_result

        self.logger.info(f"🎯 Maximum penetration found:")
        self.logger.info(f"   MD: {max_penetration_md:.1f}m")
        self.logger.info(f"   TVT: {max_penetration_tvt:.1f}m")
        self.logger.info(f"   Index: {max_penetration_idx}")

        # Define replacement zone
        replacement_start_tvt = max_penetration_tvt - self.replacement_range_meters
        replacement_end_tvt = max_penetration_tvt

        self.logger.info(f"🔄 Initial replacement zone:")
        self.logger.info(f"   Start TVT: {replacement_start_tvt:.1f}m")
        self.logger.info(f"   End TVT: {replacement_end_tvt:.1f}m")
        self.logger.info(f"   Range: {self.replacement_range_meters:.1f}m")

        # Find seamless junction point - discrete TypeWell points only
        junction_result = self._find_seamless_junction_discrete(
            well, typewell, max_penetration_tvt
        )

        if junction_result is None:
            self.logger.warning(f"⚠️ No seamless junction found, using original replacement start")
            final_replacement_start_tvt = replacement_start_tvt
            junction_tvt = None
            junction_distance = None
        else:
            junction_tvt, junction_distance = junction_result
            self.logger.info(f"✅ Seamless junction found:")
            self.logger.info(f"   Junction TVT: {junction_tvt:.1f}m")
            self.logger.info(f"   Distance from original boundary: {junction_distance:.1f}m")
            final_replacement_start_tvt = junction_tvt

        # Modify typewell.value directly
        replacement_stats = self._modify_typewell_values(
            well, typewell, final_replacement_start_tvt, replacement_end_tvt
        )

        self.logger.info(f"✅ Curve replacement completed successfully for well: {well_name}")

        # Return detailed results
        return {
            'status': 'success',
            'typewell_modified': True,
            'replacement_start_tvt': final_replacement_start_tvt,
            'replacement_end_tvt': replacement_end_tvt,
            'max_penetration_tvt': max_penetration_tvt,
            'max_penetration_md': max_penetration_md,
            'max_penetration_idx': max_penetration_idx,
            'junction_tvt': junction_tvt,
            'junction_distance': junction_distance,
            'points_replaced': replacement_stats['points_replaced'],
            'points_extended': replacement_stats['points_extended']
        }

    def _find_seamless_junction_discrete(self,
                                         well: Well,
                                         typewell: TypeWell,
                                         max_penetration_tvt: float) -> Optional[Tuple[float, float]]:
        """
        Find junction point using discrete TypeWell points only (no interpolation)
        Priority: 1) Crossing point (take point with smaller |diff|), 2) Minimal difference
        If no points in search range, take nearest point above (smaller TVT)
        """
        self.logger.info(f"🔍 Searching for seamless junction point (DISCRETE):")

        # Use search distance from config
        search_distance = self.search_distance_meters
        search_start_tvt = max_penetration_tvt - search_distance
        search_end_tvt = max_penetration_tvt

        self.logger.info(f"   Target TVT: {max_penetration_tvt:.1f}m")
        self.logger.info(f"   Search range: TVT {search_start_tvt:.1f}m → {search_end_tvt:.1f}m")
        self.logger.info(f"   Search distance: ±{search_distance:.1f}m")

        # Find TypeWell points within search range
        typewell_indices_in_range = []
        for i, tvt in enumerate(typewell.tvd):
            if search_start_tvt <= tvt <= search_end_tvt:
                typewell_indices_in_range.append(i)

        # If no points in range, find nearest point above (smaller TVT)
        if not typewell_indices_in_range:
            self.logger.info(f"   No TypeWell points in search range, looking for nearest point above")

            # Find nearest point above target (TVT < target)
            points_above = [(i, tvt) for i, tvt in enumerate(typewell.tvd) if tvt < max_penetration_tvt]

            if not points_above:
                self.logger.error(f"   No TypeWell points above target TVT!")
                return None

            # Take the closest point above (maximum TVT among points above)
            nearest_idx, nearest_tvt = max(points_above, key=lambda x: x[1])

            self.logger.info(f"   Using nearest point above target:")
            self.logger.info(f"      Index: {nearest_idx}, TVT: {nearest_tvt:.1f}m")

            distance_from_target = abs(nearest_tvt - max_penetration_tvt)
            return nearest_tvt, distance_from_target

        self.logger.info(f"   Found {len(typewell_indices_in_range)} TypeWell points in range")

        # Process points in order, looking for crossing
        prev_diff = None
        prev_point = None
        crossing_found = False
        crossing_point = None

        all_points = []  # Store all valid points for fallback

        for idx in typewell_indices_in_range:
            tvt_point = typewell.tvd[idx]
            typewell_value = typewell.value[idx]  # Direct access, no interpolation

            # Get Horizontal well value at this TVT
            horizontal_value = self._get_horizontal_value_by_tvt(well, tvt_point)

            if horizontal_value is None or np.isnan(typewell_value):
                continue

            # Calculate signed difference
            signed_diff = typewell_value - horizontal_value
            abs_diff = abs(signed_diff)

            current_point = {
                'idx': idx,
                'tvt': tvt_point,
                'typewell': typewell_value,
                'horizontal': horizontal_value,
                'signed_diff': signed_diff,
                'abs_diff': abs_diff
            }

            all_points.append(current_point)

            # Check for crossing with previous point
            if prev_diff is not None and not crossing_found:
                # Check for sign change (crossing)
                if prev_diff * signed_diff < 0:  # Different signs = crossing
                    # Select point with smaller absolute difference
                    if abs(prev_diff) < abs_diff:
                        crossing_point = prev_point
                        self.logger.info(f"   🎯 Found CROSSING, selected UPPER point (smaller |diff|):")
                        self.logger.info(
                            f"      Upper: TVT={prev_point['tvt']:.1f}m, diff={prev_diff:.3f}, |diff|={abs(prev_diff):.3f}")
                        self.logger.info(
                            f"      Lower: TVT={tvt_point:.1f}m, diff={signed_diff:.3f}, |diff|={abs_diff:.3f}")
                    else:
                        crossing_point = current_point
                        self.logger.info(f"   🎯 Found CROSSING, selected LOWER point (smaller |diff|):")
                        self.logger.info(
                            f"      Upper: TVT={prev_point['tvt']:.1f}m, diff={prev_diff:.3f}, |diff|={abs(prev_diff):.3f}")
                        self.logger.info(
                            f"      Lower: TVT={tvt_point:.1f}m, diff={signed_diff:.3f}, |diff|={abs_diff:.3f}")

                    crossing_found = True
                    # Don't break - continue to log all points for debugging

            prev_diff = signed_diff
            prev_point = current_point

        # If crossing was found, use it
        if crossing_found:
            best_tvt = crossing_point['tvt']
            distance_from_target = abs(best_tvt - max_penetration_tvt)
            return best_tvt, distance_from_target

        # No crossing found - find minimum absolute difference
        if not all_points:
            self.logger.warning("   No valid comparison points found")
            return None

        min_diff_point = min(all_points, key=lambda x: x['abs_diff'])

        best_match_tvt = min_diff_point['tvt']
        best_match_error = min_diff_point['abs_diff']

        # Log some examples
        self.logger.info("   No crossing found, using minimum difference:")
        for i, d in enumerate(all_points[:5]):
            self.logger.debug(f"      TVT={d['tvt']:.1f}: "
                              f"typewell={d['typewell']:.3f}, "
                              f"horizontal={d['horizontal']:.3f}, "
                              f"diff={d['signed_diff']:.3f}")

        distance_from_target = abs(best_match_tvt - max_penetration_tvt)
        self.logger.info(f"   ✅ Best junction found (min difference):")
        self.logger.info(f"      TVT: {best_match_tvt:.1f}m")
        self.logger.info(f"      Error: {best_match_error:.3f}")
        self.logger.info(f"      Distance from target: {distance_from_target:.1f}m")

        return best_match_tvt, distance_from_target

    def _modify_typewell_values(self,
                               well: Well,
                               typewell: TypeWell,
                               replacement_start_tvt: float,
                               replacement_end_tvt: float) -> Dict[str, int]:
        """
        Modify typewell.value array directly in the replacement zone
        Replaces TypeWell values from replacement_start (junction) up to replacement_end (max penetration)
        """
        self.logger.info(f"🔄 Modifying TypeWell.value array:")
        self.logger.info(f"   Replacement zone: TVT {replacement_start_tvt:.1f}m → {replacement_end_tvt:.1f}m")
        self.logger.info(f"   Blend weight: {self.blend_weight}")
        self.logger.info(f"   TypeWell points: {len(typewell.tvd)}")

        # Store original statistics
        original_min = typewell.value.min()
        original_max = typewell.value.max()
        original_mean = typewell.value.mean()

        replaced_count = 0
        extended_count = 0

        # Log some examples of replacement
        examples_logged = 0

        # Process each point in typewell
        for i, tvt_point in enumerate(typewell.tvd):

            # Check if point is in replacement zone
            if replacement_start_tvt <= tvt_point <= replacement_end_tvt:

                # Find corresponding horizontal well value by TVT
                horizontal_value = self._get_horizontal_value_by_tvt(well, tvt_point)

                if horizontal_value is not None:
                    original_value = typewell.value[i]

                    # Apply weighted blending
                    blended_value = (
                        self.blend_weight * horizontal_value +
                        (1.0 - self.blend_weight) * original_value
                    )

                    # Modify typewell.value directly
                    typewell.value[i] = blended_value
                    replaced_count += 1

                    # Log first 3 replacements as examples
                    if examples_logged < 3:
                        self.logger.debug(f"   Example replacement at TVT={tvt_point:.1f}, index={i}: "
                                        f"{original_value:.3f} → {blended_value:.3f} "
                                        f"(horizontal={horizontal_value:.3f})")
                        examples_logged += 1

            elif tvt_point > replacement_end_tvt:
                # Beyond original typewell range - use horizontal data if available
                horizontal_value = self._get_horizontal_value_by_tvt(well, tvt_point)

                if horizontal_value is not None:
                    typewell.value[i] = horizontal_value
                    extended_count += 1

        # Log modification statistics
        self.logger.info(f"   ✅ TypeWell.value modification completed:")
        self.logger.info(f"      Points replaced: {replaced_count}")
        self.logger.info(f"      Points extended: {extended_count}")
        self.logger.info(f"      Points unchanged: {len(typewell.tvd) - replaced_count - extended_count}")

        # Log value statistics
        self.logger.info(f"   TypeWell.value statistics:")
        self.logger.info(f"      Original: min={original_min:.3f}, max={original_max:.3f}, mean={original_mean:.3f}")
        self.logger.info(f"      Modified: min={typewell.value.min():.3f}, max={typewell.value.max():.3f}, "
                        f"mean={typewell.value.mean():.3f}")

        return {
            'points_replaced': replaced_count,
            'points_extended': extended_count
        }

    def _get_horizontal_value_by_tvt(self, well: Well, target_tvt: float) -> Optional[float]:
        """
        Get horizontal well value by TVT through interpolation
        FIXED: Proper handling of unsorted TVT and duplicates
        """
        # Find valid TVT range in well
        valid_mask = ~np.isnan(well.tvt)
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) < 2:
            return None

        valid_tvt = well.tvt[valid_indices]
        valid_values = well.value[valid_indices]  # Using well.value is CORRECT!

        # Check if target_tvt is within valid range
        if target_tvt < valid_tvt.min() or target_tvt > valid_tvt.max():
            return None

        # Check for NaN values in well.value
        valid_values_mask = ~np.isnan(valid_values)
        if not np.any(valid_values_mask):
            return None

        final_valid_indices = valid_indices[valid_values_mask]
        final_tvt = well.tvt[final_valid_indices]
        final_values = well.value[final_valid_indices]
        
        # Handle unsorted TVT
        if not np.all(np.diff(final_tvt) > 0):
            # Sort arrays by TVT for proper interpolation
            sort_indices = np.argsort(final_tvt)
            final_tvt = final_tvt[sort_indices]
            final_values = final_values[sort_indices]
            
            # Remove duplicates (keep first occurrence)
            unique_mask = np.concatenate([[True], np.diff(final_tvt) > 1e-6])
            final_tvt = final_tvt[unique_mask]
            final_values = final_values[unique_mask]
            
            if len(final_tvt) < 2:
                return None

        # Interpolate to find value at target_tvt
        interpolated_value = np.interp(target_tvt, final_tvt, final_values)

        return float(interpolated_value)

    def _find_maximum_penetration_point(self,
                                       well: Well,
                                       start_md: float,
                                       end_md: float) -> Optional[Tuple[float, int, float]]:
        """
        Find point where TVT stops decreasing and starts increasing (maximum penetration)
        """
        # Find indices for MD range
        start_idx = well.md2idx(start_md)
        end_idx = well.md2idx(end_md)

        self.logger.info(f"🔍 Searching for maximum penetration:")
        self.logger.info(f"   MD range: {start_md:.1f}m → {end_md:.1f}m")
        self.logger.info(f"   Index range: {start_idx} → {end_idx}")

        if start_idx >= end_idx - 1:
            self.logger.warning(f"Invalid MD range for penetration analysis: indices {start_idx} → {end_idx}")
            return None

        # Extract TVT values in range
        tvt_segment = well.tvt[start_idx:end_idx+1]
        md_segment = well.measured_depth[start_idx:end_idx+1]

        # Filter out NaN values for analysis
        valid_mask = ~np.isnan(tvt_segment)
        valid_tvt = tvt_segment[valid_mask]
        valid_md = md_segment[valid_mask]
        valid_indices = np.where(valid_mask)[0]

        if len(valid_tvt) < 3:
            self.logger.warning(f"Not enough valid TVT points for analysis: {len(valid_tvt)}")
            return None

        self.logger.info(f"   Valid TVT points: {len(valid_tvt)} out of {len(tvt_segment)}")
        self.logger.info(f"   TVT range in segment: {valid_tvt.min():.1f}m → {valid_tvt.max():.1f}m")

        # Calculate differences (derivatives)
        tvt_diff = np.diff(valid_tvt)

        # Find transition from negative/zero to positive
        for i in range(len(tvt_diff) - 1):
            current_diff = tvt_diff[i]
            next_diff = tvt_diff[i + 1]

            # Look for sign change from non-positive to positive
            if current_diff < 0 and next_diff < 0:
                # Map back to original indices
                valid_idx_in_segment = valid_indices[i + 1]
                max_penetration_idx = start_idx + valid_idx_in_segment
                max_penetration_md = well.measured_depth[max_penetration_idx]
                max_penetration_tvt = well.tvt[max_penetration_idx]

                self.logger.info(f"   ✅ Found penetration transition at:")
                self.logger.info(f"      MD: {max_penetration_md:.1f}m")
                self.logger.info(f"      TVT: {max_penetration_tvt:.1f}m")

                return max_penetration_md, max_penetration_idx, max_penetration_tvt

        # Check for valley (minimum) point
        for i in range(1, len(valid_tvt) - 1):
            tvt_prev = valid_tvt[i-1]
            tvt_curr = valid_tvt[i]
            tvt_next = valid_tvt[i+1]

            # Check if we have a minimum (valley) point
            if tvt_prev > tvt_curr and tvt_curr < tvt_next:
                valid_idx_in_segment = valid_indices[i]
                max_penetration_idx = start_idx + valid_idx_in_segment
                max_penetration_md = well.measured_depth[max_penetration_idx]
                max_penetration_tvt = well.tvt[max_penetration_idx]

                self.logger.info(f"   ✅ Found penetration valley at:")
                self.logger.info(f"      MD: {max_penetration_md:.1f}m")
                self.logger.info(f"      TVT: {max_penetration_tvt:.1f}m")

                return max_penetration_md, max_penetration_idx, max_penetration_tvt

        self.logger.warning("   ❌ No maximum penetration point found")
        return None