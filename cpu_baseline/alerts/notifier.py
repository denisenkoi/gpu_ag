import logging
import os
import html
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class NotificationManager:
    """Manages SMS (Twilio) and Telegram notifications for alerts"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize notification channels

        Args:
            config: Configuration dictionary with notification settings
        """
        # Telegram configuration
        self.telegram_enabled = config.get('TELEGRAM_ENABLED', 'false').lower() == 'true'
        self.telegram_bot_token = config.get('TG_BOT_KEY', '')
        self.telegram_chat_id = config.get('TG_CHAT_ID', '')

        # Twilio SMS configuration
        self.sms_enabled = config.get('SMS_ENABLED', 'false').lower() == 'true'
        self.sms_key = config.get('SMS_KEY', '')
        self.twilio_account_sid = config.get('TWILIO_ACCOUNT_SID', '')
        self.twilio_auth_token = config.get('TWILIO_AUTH_TOKEN', '')
        self.twilio_from_number = config.get('TWILIO_FROM_NUMBER', '')
        self.twilio_to_numbers = config.get('TWILIO_TO_NUMBERS', '').split(',')

        # Rate limiting
        self.rate_limit_minutes = int(config.get('NOTIFICATION_RATE_LIMIT_MINUTES', '10'))
        self.last_notification_time = {}  # Track by channel

        # Alert formatting
        self.alert_units_format = config.get('ALERT_UNITS_FORMAT', 'both').lower()
        if self.alert_units_format not in ['meters', 'feet', 'both']:
            logger.warning(f"Invalid ALERT_UNITS_FORMAT='{self.alert_units_format}', using 'both'")
            self.alert_units_format = 'both'

        # Initialize clients
        self._init_telegram()
        self._init_twilio()

        logger.info(f"NotificationManager initialized: Telegram={self.telegram_enabled}, SMS={self.sms_enabled}, units={self.alert_units_format}")

    def _init_telegram(self):
        """Initialize Telegram bot with proper connection pool settings"""
        if self.telegram_enabled:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                logger.warning("Telegram enabled but missing TG_BOT_KEY or TG_CHAT_ID")
                self.telegram_enabled = False
                return

            try:
                import telegram
                from telegram.request import HTTPXRequest

                # Configure HTTPXRequest with larger connection pool and timeouts
                request = HTTPXRequest(
                    connection_pool_size=8,  # Increase from default 1 to handle frequent requests
                    connect_timeout=10.0,
                    read_timeout=10.0,
                    pool_timeout=10.0  # Wait up to 10s for connection from pool
                )

                self.telegram_bot = telegram.Bot(token=self.telegram_bot_token, request=request)
                logger.info("Telegram bot initialized with connection pool size=8")
            except ImportError:
                logger.error("python-telegram-bot not installed. Install: pip install python-telegram-bot")
                self.telegram_enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Telegram bot: {e}")
                self.telegram_enabled = False
        else:
            self.telegram_bot = None

    def _init_twilio(self):
        """Initialize Twilio SMS client"""
        if self.sms_enabled:
            if not self.twilio_account_sid or not self.twilio_auth_token:
                logger.warning("SMS enabled but missing TWILIO_ACCOUNT_SID or TWILIO_AUTH_TOKEN")
                self.sms_enabled = False
                return

            try:
                from twilio.rest import Client
                self.twilio_client = Client(self.twilio_account_sid, self.twilio_auth_token)
                logger.info("Twilio SMS client initialized")
            except ImportError:
                logger.error("twilio not installed. Install: pip install twilio")
                self.sms_enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize Twilio client: {e}")
                self.sms_enabled = False
        else:
            self.twilio_client = None

    def _check_rate_limit(self, channel: str) -> bool:
        """Check if notification is rate-limited

        Args:
            channel: Notification channel (telegram/sms)

        Returns:
            True if allowed, False if rate-limited
        """
        if channel not in self.last_notification_time:
            return True

        elapsed = datetime.now() - self.last_notification_time[channel]
        if elapsed < timedelta(minutes=self.rate_limit_minutes):
            remaining = timedelta(minutes=self.rate_limit_minutes) - elapsed
            logger.debug(f"{channel} rate-limited: {remaining.seconds}s remaining")
            return False

        return True

    def _update_rate_limit(self, channel: str):
        """Update last notification time for channel"""
        self.last_notification_time[channel] = datetime.now()

    def send_telegram(self, message: str, force: bool = False) -> bool:
        """Send Telegram message

        Args:
            message: Message text
            force: Skip rate limiting

        Returns:
            True if sent successfully
        """
        if not self.telegram_enabled:
            logger.debug("Telegram disabled, skipping")
            return False

        # Check rate limit
        if not force and not self._check_rate_limit('telegram'):
            logger.warning("Telegram notification rate-limited")
            return False

        try:
            import asyncio

            # Run async send_message in sync context
            async def send():
                await self.telegram_bot.send_message(
                    chat_id=self.telegram_chat_id,
                    text=message,
                    parse_mode='HTML'
                )

            # Execute async function
            asyncio.run(send())

            self._update_rate_limit('telegram')
            logger.info("Telegram notification sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False

    def send_sms(self, message: str, force: bool = False) -> bool:
        """Send SMS via Twilio

        Args:
            message: SMS text
            force: Skip rate limiting

        Returns:
            True if sent successfully
        """
        if not self.sms_enabled:
            logger.debug("SMS disabled, skipping")
            return False

        # Check rate limit
        if not force and not self._check_rate_limit('sms'):
            logger.warning("SMS notification rate-limited")
            return False

        success_count = 0
        for to_number in self.twilio_to_numbers:
            to_number = to_number.strip()
            if not to_number:
                continue

            try:
                self.twilio_client.messages.create(
                    body=message,
                    from_=self.twilio_from_number,
                    to=to_number
                )
                logger.info(f"SMS sent to {to_number}")
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to send SMS to {to_number}: {e}")

        if success_count > 0:
            self._update_rate_limit('sms')
            return True

        return False

    def send_alert(self, alert_data: Dict[str, Any], force: bool = False) -> Dict[str, bool]:
        """Send alert via all enabled channels

        Args:
            alert_data: Alert data dictionary
            force: Skip rate limiting

        Returns:
            Dict with status for each channel
        """
        # Format message
        message = self._format_alert_message(alert_data)

        results = {}

        # Send Telegram
        if self.telegram_enabled:
            results['telegram'] = self.send_telegram(message, force=force)

        # Send SMS
        if self.sms_enabled:
            results['sms'] = self.send_sms(message, force=force)

        return results

    def _format_alert_message(self, alert_data: Dict[str, Any]) -> str:
        """Format alert data into notification message

        Args:
            alert_data: Alert data dictionary

        Returns:
            Formatted message string
        """
        well_name = alert_data.get('well_name', 'Unknown')
        md = alert_data.get('measured_depth', 0)
        target_tvd = alert_data.get('target_tvd', 0)
        interp_tvd = alert_data.get('interpretation_tvd', 0)
        deviation_m = alert_data.get('deviation_meters', 0)
        deviation_ft = alert_data.get('deviation_feet', 0)
        threshold_ft = alert_data.get('threshold_feet', 0)
        project_measure_unit = alert_data.get('project_measure_unit')
        timestamp = alert_data.get('timestamp', datetime.now().isoformat())

        # CRITICAL: project_measure_unit MUST be in alert_data
        if not project_measure_unit:
            logger.error("CRITICAL: project_measure_unit missing from alert_data!")
            raise ValueError("project_measure_unit is required in alert_data for correct message formatting")

        # Determine alert type
        alert_level = alert_data.get('alert_level', 'WARNING')

        # Helper function to format values with units
        def format_value(value_m: float, label: str) -> str:
            """Format value according to ALERT_UNITS_FORMAT setting"""
            value_ft = value_m * 3.28084

            if self.alert_units_format == 'meters':
                return f"{label}: {value_m:.1f} m"
            elif self.alert_units_format == 'feet':
                return f"{label}: {value_ft:.1f} ft"
            else:  # both
                return f"{label}: {value_m:.1f} m / {value_ft:.1f} ft"

        # Format message based on alert type
        if alert_level in ['HORIZON_BREACH', 'HORIZON_OK']:
            # Horizon breach alert - use ready message from alert_data
            message_text = alert_data.get('message', 'No details')

            # Escape HTML entities (< > &) for Telegram HTML mode
            message_text_escaped = html.escape(message_text)

            if alert_level == 'HORIZON_BREACH':
                icon = "🚨"
                header = "HORIZON BREACH ALERT"
            else:
                icon = "✅"
                header = "HORIZON OK"

            # Fix timestamp - if None, use current time
            timestamp_display = timestamp if timestamp else datetime.now().isoformat()

            # Ready message already contains all details with correct numbers
            message = (
                f"{icon} {header}\n\n"
                f"Well: {well_name}\n"
                f"{format_value(md, 'MD')}\n"
                f"{message_text_escaped}\n"
                f"Time: {timestamp_display}\n"
            )
        else:
            # Deviation alert (original format)
            threshold_ft = alert_data.get('threshold_feet', 0)
            threshold_m = threshold_ft / 3.28084

            message = (
                f"🚨 NIGHT GUARD ALERT\n\n"
                f"Well: {well_name}\n"
                f"{format_value(md, 'MD')}\n"
                f"{format_value(target_tvd, 'Target TVD')}\n"
                f"{format_value(interp_tvd, 'Current TVD')}\n"
                f"{format_value(deviation_m, 'Deviation')}\n"
                f"{format_value(threshold_m, 'Threshold')}\n"
                f"Time: {timestamp}\n"
            )

        return message
