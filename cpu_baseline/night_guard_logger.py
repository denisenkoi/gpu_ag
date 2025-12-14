"""
Night Guard Debug Logger - Disk-based logging for diagnostics
"""
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

class NightGuardLogger:
    """Minimal disk-based logger for Night Guard diagnostics"""

    def __init__(self, log_dir: str = "night_guard_logs"):
        self.log_dir = log_dir
        self.log_file = os.path.join(log_dir, "night_guard_debug.log")
        self.status_file = os.path.join(log_dir, "night_guard_status.json")

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # Initialize status tracking
        self.status = {
            "session_start": datetime.now().isoformat(),
            "initialization": {},
            "checkpoints": {},
            "last_activity": None,
            "alerts_count": 0,
            "errors_count": 0
        }
        self._save_status()

    def log_checkpoint(self, checkpoint: str, data: Optional[Dict] = None, success: bool = True):
        """Log a key checkpoint with minimal data"""
        timestamp = datetime.now().isoformat()

        # Update status
        self.status["checkpoints"][checkpoint] = {
            "timestamp": timestamp,
            "success": success,
            "data": data or {}
        }
        self.status["last_activity"] = timestamp

        # Write to log file
        log_entry = f"[{timestamp}] CHECKPOINT: {checkpoint}"
        if not success:
            log_entry += " ❌ FAILED"
            self.status["errors_count"] += 1
        else:
            log_entry += " ✅"

        if data:
            log_entry += f" | Data: {json.dumps(data, default=str)}"

        self._write_log(log_entry)
        self._save_status()

    def log_initialization(self, component: str, enabled: bool, details: Optional[Dict] = None):
        """Log component initialization"""
        self.status["initialization"][component] = {
            "enabled": enabled,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }

        status_text = "ENABLED" if enabled else "DISABLED"
        log_entry = f"[{datetime.now().isoformat()}] INIT: {component} = {status_text}"
        if details:
            log_entry += f" | {json.dumps(details, default=str)}"

        self._write_log(log_entry)
        self._save_status()

    def log_alert(self, well_name: str, md: float, deviation: float, alert_triggered: bool):
        """Log alert analysis result"""
        if alert_triggered:
            self.status["alerts_count"] += 1

        data = {
            "well_name": well_name,
            "md": md,
            "deviation": deviation,
            "alert_triggered": alert_triggered
        }

        checkpoint_name = "alert_triggered" if alert_triggered else "alert_checked"
        self.log_checkpoint(checkpoint_name, data, True)

    def log_error(self, component: str, error: str, details: Optional[Dict] = None):
        """Log error with details"""
        self.status["errors_count"] += 1

        data = {"error": error}
        if details:
            data.update(details)

        self.log_checkpoint(f"error_{component}", data, False)

    def _write_log(self, message: str):
        """Write message to log file"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(message + "\n")
        except Exception:
            pass  # Silent fail for logging

    def _save_status(self):
        """Save current status to JSON file"""
        try:
            with open(self.status_file, "w", encoding="utf-8") as f:
                json.dump(self.status, f, indent=2, default=str)
        except Exception:
            pass  # Silent fail for logging

    def get_status_summary(self) -> str:
        """Get quick status summary"""
        try:
            init_status = []
            for component, data in self.status["initialization"].items():
                status = "✅" if data["enabled"] else "❌"
                init_status.append(f"{component}: {status}")

            checkpoints_count = len(self.status["checkpoints"])
            last_activity = self.status.get("last_activity", "Never")

            return (f"Night Guard Status:\n"
                   f"  Initialization: {', '.join(init_status)}\n"
                   f"  Checkpoints: {checkpoints_count}\n"
                   f"  Alerts: {self.status['alerts_count']}\n"
                   f"  Errors: {self.status['errors_count']}\n"
                   f"  Last Activity: {last_activity}")
        except Exception:
            return "Status unavailable"