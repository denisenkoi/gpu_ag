import logging
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from .notifier import NotificationManager

logger = logging.getLogger(__name__)


class AlertAnalyzer:
    """Alert analyzer with integrated SMS notifications for Night Guard system"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize analyzer with configuration"""
        self.base_tvd = float(config['BASE_TVD_METERS'])
        self.threshold_feet = float(config.get('ALERT_THRESHOLD_FEET', '7.5'))
        self.alerts_dir = Path(config.get('ALERTS_DIR', 'alerts/'))

        # Initialize notification manager
        self.notifier = NotificationManager(config)
        
        # Create alerts directory
        self.alerts_dir.mkdir(exist_ok=True)
        
        logger.info(f"AlertAnalyzer initialized: threshold={self.threshold_feet}ft, base_tvd={self.base_tvd}m")
        
    def analyze_deviation(
        self,
        interpretation_data: Dict[str, Any],
        current_md: float,
        target_tvd: float,
        vs_coordinate: float,
        well_name: str,
        target_line_info: Optional[Dict[str, Any]] = None,
        project_measure_unit: str = None,
        base_tvd_override: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyze deviation between interpretation and target line at current MD

        Args:
            interpretation_data: Interpretation segments from emulator
            current_md: Current maximum MD from PAPI data
            target_tvd: Ready-calculated target TVD from PAPI Loader
            vs_coordinate: VS coordinate for current MD from PAPI Loader
            well_name: Well name for logging and alerts
            target_line_info: Additional target line info for alert context
            project_measure_unit: Project units from PAPI ('FOOT' or 'METER'). REQUIRED!
            base_tvd_override: Override base_tvd (for StarSteer mode - use horizon TVD from JSON)
        """

        # CRITICAL: project_measure_unit MUST be provided
        if not project_measure_unit:
            raise ValueError("project_measure_unit is required for correct deviation calculation")

        logger.debug(f"Analyzing deviation: well={well_name}, MD={current_md:.1f}, target_tvd={target_tvd:.1f}, units={project_measure_unit}")

        # Get interpretation segments
        segments = interpretation_data['interpretation']['segments']
        if not segments:
            logger.warning("No interpretation segments found")
            return None

        # Interpolate shift at current MD
        shift_at_md = self._interpolate_shift_from_segments(segments, current_md)
        if shift_at_md is None:
            logger.warning(f"Could not interpolate shift for MD={current_md:.1f}")
            return None

        # Calculate interpretation TVD
        interp_tvd = self.base_tvd + shift_at_md

        # Calculate deviation - values are in METERS (internal format)
        deviation_meters = abs(interp_tvd - target_tvd)

        # Convert to feet ONLY if project is in feet
        if project_measure_unit == 'FOOT':
            deviation_feet = deviation_meters / 0.3048
        else:
            # Project in meters - deviation is already in meters
            # But threshold is in feet, so convert deviation to feet for comparison
            deviation_feet = deviation_meters / 0.3048

        # Check if alert should be triggered
        alert_triggered = deviation_feet > self.threshold_feet

        # ========== DETAILED LOGGING OF ALL CALCULATION COMPONENTS ==========
        logger.info("=" * 80)
        logger.info(f"DEVIATION CALCULATION DETAILS for {well_name} at MD={current_md:.1f}m:")
        logger.info(f"  1. base_tvd (from config) = {self.base_tvd:.3f} m")
        logger.info(f"  2. shift_at_md (interpolated from segments) = {shift_at_md:.3f} m")
        logger.info(f"  3. interp_tvd = base_tvd + shift_at_md = {self.base_tvd:.3f} + ({shift_at_md:.3f}) = {interp_tvd:.3f} m")
        logger.info(f"  4. target_tvd (from well.points) = {target_tvd:.3f} m")
        logger.info(f"  5. deviation_meters = |interp_tvd - target_tvd| = |{interp_tvd:.3f} - {target_tvd:.3f}| = {deviation_meters:.3f} m")
        logger.info(f"  6. deviation_feet = deviation_meters / 0.3048 = {deviation_meters:.3f} / 0.3048 = {deviation_feet:.2f} ft")
        logger.info(f"  7. threshold_feet = {self.threshold_feet:.2f} ft")
        logger.info(f"  8. alert_triggered = {alert_triggered} ({deviation_feet:.2f} ft {'>' if alert_triggered else '<='} {self.threshold_feet:.2f} ft)")
        logger.info(f"  9. project_measure_unit = {project_measure_unit}")
        logger.info("=" * 80)
        
        # Create alert data
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'well_name': well_name,
            'measured_depth': current_md,
            'vs_coordinate': vs_coordinate,
            'interpretation_tvd': interp_tvd,
            'target_tvd': target_tvd,
            'shift_at_md': shift_at_md,
            'base_tvd': self.base_tvd,
            'deviation_meters': deviation_meters,
            'deviation_feet': deviation_feet,
            'threshold_feet': self.threshold_feet,
            'alert': alert_triggered,
            'alert_level': 'WARNING' if alert_triggered else 'OK',
            'project_measure_unit': project_measure_unit
        }
        
        # Add target line info if provided
        if target_line_info:
            alert_data['target_line_info'] = {
                'name': target_line_info.get('name', 'Unknown'),
                'uuid': target_line_info.get('uuid', ''),
                'origin_vs': target_line_info.get('origin_vs'),
                'target_vs': target_line_info.get('target_vs'),
                'origin_tvd': target_line_info.get('origin_tvd'),
                'target_tvd': target_line_info.get('target_tvd')
            }
        
        # Save alert data to file
        self._save_alert_data(alert_data)

        # Send notifications if alert triggered
        if alert_triggered:
            notification_results = self.notifier.send_alert(alert_data)
            logger.info(f"Notification results: {notification_results}")
            
        # Log result
        if alert_triggered:
            logger.warning(f"🚨 ALERT: {well_name} MD={current_md:.1f} deviation={deviation_feet:.2f}ft > {self.threshold_feet}ft")
        else:
            logger.info(f"✅ OK: {well_name} MD={current_md:.1f} deviation={deviation_feet:.2f}ft <= {self.threshold_feet}ft")
            
        return alert_data
        
    def _interpolate_shift_from_segments(self, segments: List[Dict], target_md: float) -> Optional[float]:
        """Interpolate shift value from segments for given MD"""
        
        if not segments:
            return None
            
        # Sort segments by startMd
        sorted_segments = sorted(segments, key=lambda s: s['startMd'])
        
        # Find segment containing target MD
        for i, segment in enumerate(sorted_segments):
            start_md = segment['startMd']
            
            # Determine end MD (start of next segment or assume large value)
            if i + 1 < len(sorted_segments):
                end_md = sorted_segments[i + 1]['startMd']
            else:
                end_md = start_md + 10000.0  # Large value for last segment
                
            if start_md <= target_md <= end_md:
                # Linear interpolation within segment
                if abs(end_md - start_md) < 0.001:
                    return segment['startShift']
                    
                ratio = (target_md - start_md) / (end_md - start_md)
                interpolated_shift = segment['startShift'] + ratio * (segment['endShift'] - segment['startShift'])
                
                logger.debug(f"Interpolated shift: MD={target_md:.1f} → shift={interpolated_shift:.3f}")
                return interpolated_shift
                
        # If not in any segment, use closest segment
        closest_segment = min(sorted_segments, key=lambda s: abs(s['startMd'] - target_md))
        logger.warning(f"MD {target_md:.1f} not in any segment, using closest at MD {closest_segment['startMd']:.1f}")
        return closest_segment['startShift']
        
    def _save_alert_data(self, alert_data: Dict[str, Any]):
        """Save alert data to JSON file"""
        
        well_name = alert_data['well_name']
        md = alert_data['measured_depth']
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"alert_{well_name}_{md:.1f}_{timestamp}.json"
        filepath = self.alerts_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(alert_data, f, indent=2, ensure_ascii=False)
            
        logger.debug(f"Alert data saved to: {filepath}")
        
    def _should_send_sms(self) -> bool:
        """Check if enough time has passed for next SMS (rate limiting)"""
        
        if self.last_sms_time is None:
            return True
            
        current_time = time.time()
        time_since_last_sms = current_time - self.last_sms_time
        
        return time_since_last_sms >= (self.rate_limit_minutes * 60)
        
    def _send_alert_sms(self, alert_data: Dict[str, Any]):
        """Send SMS alert via Twilio"""
        
        if not self.twilio_client:
            logger.error("Twilio client not initialized")
            return
            
        # Format SMS message
        message_body = self._format_alert_message(alert_data)
        
        # Send SMS
        message = self.twilio_client.messages.create(
            body=message_body,
            from_=self.from_number,
            to=self.to_number
        )
        
        # Update last SMS time
        self.last_sms_time = time.time()
        
        logger.info(f"SMS alert sent: SID={message.sid}")
        
    def _format_alert_message(self, alert_data: Dict[str, Any]) -> str:
        """Format alert data into SMS message"""

        well_name = alert_data['well_name']
        md = alert_data['measured_depth']
        deviation_feet = alert_data['deviation_feet']
        threshold_feet = alert_data['threshold_feet']
        target_tvd = alert_data['target_tvd']
        interp_tvd = alert_data['interpretation_tvd']
        timestamp = datetime.now().strftime("%H:%M:%S")

        message = f"""🚨 DRILLING ALERT
Well: {well_name}
MD: {md:.1f}m
Deviation: {deviation_feet:.1f}ft > {threshold_feet:.1f}ft
Target TVD: {target_tvd:.1f}m
Actual TVD: {interp_tvd:.1f}m
Time: {timestamp}"""

        return message

    def check_horizon_breach(
        self,
        point_data: Dict[str, Any],
        horizons: Dict[str, Any],
        interpretation_segments: Optional[List[Dict]] = None,
        tvd_typewell_shift: float = 0.0
    ) -> Optional[Dict[str, Any]]:
        """Check if corridor breaches horizon boundaries

        Args:
            point_data: Well point with targetLineTVD, corridorTop, corridorBase
            horizons: Dict with bottomHorizon and topHorizon (each has trueVerticalDepth in METERS)
            interpretation_segments: Segments for shift interpolation (optional)
            tvd_typewell_shift: TVD to typewell shift value (optional, default 0.0)

        Returns:
            Dict with alert info if breach detected, None otherwise:
            {
                'has_alert': bool,
                'breach_type': 'upper' | 'lower',
                'message': str,
                'details': {
                    'md': float,
                    'target_tvd': float,
                    'upper_corridor': float,
                    'lower_corridor': float,
                    'top_horizon': float,
                    'bottom_horizon': float
                }
            }
        """

        # Validate input data
        if not point_data or not horizons:
            logger.debug("check_horizon_breach: missing point_data or horizons")
            return None

        # Extract required fields
        target_tvd = point_data.get('targetLineTVD')
        corridor_top = point_data.get('corridorTop')
        corridor_base = point_data.get('corridorBase')
        md = point_data.get('measuredDepth')

        if target_tvd is None or corridor_top is None or corridor_base is None:
            logger.debug(f"check_horizon_breach: missing corridor data at MD={md}")
            return None

        # Extract base horizons TVD (in METERS from StarSteer JSON)
        bottom_horizon_data = horizons.get('bottomHorizon')
        top_horizon_data = horizons.get('topHorizon')

        if not bottom_horizon_data or not top_horizon_data:
            logger.warning("check_horizon_breach: missing horizon data")
            return None

        base_bottom_tvd = bottom_horizon_data.get('trueVerticalDepth')
        base_top_tvd = top_horizon_data.get('trueVerticalDepth')

        if base_bottom_tvd is None or base_top_tvd is None:
            logger.warning("check_horizon_breach: horizon TVD values missing")
            return None

        # Apply geological shift to horizons
        shift_at_md = 0.0
        if interpretation_segments:
            shift_at_md = self._interpolate_shift_from_segments(interpretation_segments, md)
            if shift_at_md is None:
                logger.warning(f"check_horizon_breach: Could not interpolate shift at MD={md:.1f}, using 0.0")
                shift_at_md = 0.0

        # Calculate shifted horizon TVDs
        # Formula: shifted_tvd = base_tvd + shift_at_md + tvd_typewell_shift
        shifted_bottom_tvd = base_bottom_tvd + shift_at_md + tvd_typewell_shift
        shifted_top_tvd = base_top_tvd + shift_at_md + tvd_typewell_shift

        # Corridor boundaries (corridor values already in METERS from StarSteer)
        corridor_top_m = corridor_top
        corridor_base_m = corridor_base

        upper_corridor = target_tvd - corridor_top_m
        lower_corridor = target_tvd + corridor_base_m

        # ========== DETAILED LOGGING OF HORIZON BREACH CALCULATION ==========
        logger.info("=" * 80)
        logger.info(f"HORIZON BREACH CALCULATION DETAILS at MD={md:.1f}m:")
        logger.info(f"  1. BASE HORIZONS (from JSON):")
        logger.info(f"     - base_bottom_tvd ({bottom_horizon_data.get('name', 'Unknown')}) = {base_bottom_tvd:.3f} m")
        logger.info(f"     - base_top_tvd ({top_horizon_data.get('name', 'Unknown')}) = {base_top_tvd:.3f} m")
        logger.info(f"  2. SHIFT VALUES:")
        logger.info(f"     - shift_at_md (from interpretation segments) = {shift_at_md:.3f} m")
        logger.info(f"     - tvd_typewell_shift (from JSON) = {tvd_typewell_shift:.3f} m")
        logger.info(f"  3. SHIFTED HORIZONS:")
        logger.info(f"     - shifted_bottom_tvd = base_bottom + shift + tvd_shift = {base_bottom_tvd:.3f} + ({shift_at_md:.3f}) + {tvd_typewell_shift:.3f} = {shifted_bottom_tvd:.3f} m")
        logger.info(f"     - shifted_top_tvd = base_top + shift + tvd_shift = {base_top_tvd:.3f} + ({shift_at_md:.3f}) + {tvd_typewell_shift:.3f} = {shifted_top_tvd:.3f} m")
        logger.info(f"  4. TARGET AND CORRIDOR:")
        logger.info(f"     - target_tvd (from well.points) = {target_tvd:.3f} m")
        logger.info(f"     - corridor_top = {corridor_top_m:.3f} m")
        logger.info(f"     - corridor_base = {corridor_base_m:.3f} m")
        logger.info(f"  5. CORRIDOR BOUNDARIES:")
        logger.info(f"     - upper_corridor = target_tvd - corridor_top = {target_tvd:.3f} - {corridor_top_m:.3f} = {upper_corridor:.3f} m")
        logger.info(f"     - lower_corridor = target_tvd + corridor_base = {target_tvd:.3f} + {corridor_base_m:.3f} = {lower_corridor:.3f} m")
        # Check breach conditions
        lower_breach = lower_corridor > shifted_bottom_tvd
        upper_breach = upper_corridor < shifted_top_tvd

        logger.info(f"  6. BREACH CHECK:")
        logger.info(f"     - Is lower_corridor > shifted_bottom_tvd? {lower_corridor:.3f} > {shifted_bottom_tvd:.3f} = {'BREACH!' if lower_breach else 'OK'}")
        logger.info(f"     - Is upper_corridor < shifted_top_tvd? {upper_corridor:.3f} < {shifted_top_tvd:.3f} = {'BREACH!' if upper_breach else 'OK'}")
        logger.info("=" * 80)

        # Check for breaches using SHIFTED horizons
        # TVD logic: higher TVD = deeper, lower TVD = shallower
        breach_type = None
        breach_message = None

        # Lower corridor breaches bottom horizon (goes BELOW/deeper than allowed)
        if lower_corridor > shifted_bottom_tvd:
            breach_type = 'lower'
            breach_distance_m = lower_corridor - shifted_bottom_tvd
            breach_distance_ft = breach_distance_m / 0.3048
            breach_message = (
                f"Lower corridor breaches bottom horizon by {breach_distance_ft:.2f}ft "
                f"(corridor={lower_corridor:.2f}m > shifted_horizon={shifted_bottom_tvd:.2f}m)"
            )

        # Upper corridor breaches top horizon (goes ABOVE/shallower than allowed)
        elif upper_corridor < shifted_top_tvd:
            breach_type = 'upper'
            breach_distance_m = shifted_top_tvd - upper_corridor
            breach_distance_ft = breach_distance_m / 0.3048
            breach_message = (
                f"Upper corridor breaches top horizon by {breach_distance_ft:.2f}ft "
                f"(corridor={upper_corridor:.2f}m < shifted_horizon={shifted_top_tvd:.2f}m)"
            )

        # No breach detected
        if not breach_type:
            # Calculate clearances
            lower_clearance_m = shifted_bottom_tvd - lower_corridor
            lower_clearance_ft = lower_clearance_m / 0.3048
            upper_clearance_m = upper_corridor - shifted_top_tvd
            upper_clearance_ft = upper_clearance_m / 0.3048

            # Format OK message with numbers
            ok_message = (
                f"Corridor within limits: "
                f"Lower OK by {lower_clearance_ft:.2f}ft ({lower_corridor:.2f}m < {shifted_bottom_tvd:.2f}m), "
                f"Upper OK by {upper_clearance_ft:.2f}ft ({upper_corridor:.2f}m > {shifted_top_tvd:.2f}m)"
            )

            logger.info(f"✅ HORIZON OK at MD={md:.1f}: {ok_message}")
            return {
                'has_alert': False,
                'breach_type': None,
                'message': ok_message,
                'details': {
                    'md': md,
                    'target_tvd': target_tvd,
                    'upper_corridor': upper_corridor,
                    'lower_corridor': lower_corridor,
                    'top_horizon': shifted_top_tvd,
                    'bottom_horizon': shifted_bottom_tvd,
                    'lower_clearance': lower_clearance_m,
                    'upper_clearance': upper_clearance_m
                }
            }

        # Breach detected - create alert
        alert_result = {
            'has_alert': True,
            'breach_type': breach_type,
            'message': breach_message,
            'details': {
                'md': md,
                'target_tvd': target_tvd,
                'upper_corridor': upper_corridor,
                'lower_corridor': lower_corridor,
                'top_horizon': shifted_top_tvd,
                'bottom_horizon': shifted_bottom_tvd,
                'top_horizon_name': top_horizon_data.get('name', 'Unknown'),
                'bottom_horizon_name': bottom_horizon_data.get('name', 'Unknown')
            }
        }

        logger.warning(f"🚨 HORIZON BREACH: {breach_message} at MD={md:.1f}")

        return alert_result
