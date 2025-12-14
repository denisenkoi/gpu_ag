import json
import os
import time
import logging
import uuid
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class StarSteerTargetLineInterface:
    """
    Interface для получения target line TVD от StarSteer через JSON файловый обмен
    Используется в Night Guard Alert System как замена PAPI target line fetcher
    """

    def __init__(self, config: Dict[str, Any], ng_logger=None):
        """Initialize with configuration"""

        # StarSteer file exchange paths
        self.ss_directory = Path(config.get('STARSTEER_DIRECTORY', 'SS_slicer'))
        self.request_file = self.ss_directory / "target_line_request.json"
        self.response_file = self.ss_directory / "target_line_response.json"

        # Request configuration
        self.timeout_seconds = int(config.get('STARSTEER_TIMEOUT_SECONDS', '60'))
        self.poll_interval = float(config.get('STARSTEER_POLL_INTERVAL', '2.0'))  # секунды
        self.max_retries = int(config.get('STARSTEER_MAX_RETRIES', '3'))

        # Target line configuration
        self.target_well_name = config.get('NIGHTGUARD_WELL_NAME')
        if not self.target_well_name:
            logger.error("CRITICAL: NIGHTGUARD_WELL_NAME not provided in config!")
            raise ValueError("NIGHTGUARD_WELL_NAME is required for StarSteer target line interface")

        self.target_line_name = config.get('TARGET_LINE_NAME')
        if not self.target_line_name:
            logger.error("CRITICAL: TARGET_LINE_NAME not provided in config!")
            raise ValueError("TARGET_LINE_NAME is required for StarSteer target line interface")

        # Night Guard logger for file-based logging
        self.ng_logger = ng_logger

        # Project units for VS/TVD conversion - REQUIRED!
        self.project_measure_unit = config.get('PROJECT_MEASURE_UNIT')
        if not self.project_measure_unit:
            logger.error("CRITICAL: PROJECT_MEASURE_UNIT not provided in config!")
            raise ValueError("PROJECT_MEASURE_UNIT is required for VS/TVD conversion")

        # Ensure SS directory exists
        self.ss_directory.mkdir(exist_ok=True)

        logger.info(f"StarSteer Interface initialized: well={self.target_well_name}, timeout={self.timeout_seconds}s, units={self.project_measure_unit}")

        # Log initialization to ng_logger
        if self.ng_logger:
            self.ng_logger.log_checkpoint("starsteer_interface_init", {
                "ss_directory": str(self.ss_directory),
                "target_well_name": self.target_well_name,
                "timeout_seconds": self.timeout_seconds,
                "poll_interval": self.poll_interval,
                "max_retries": self.max_retries
            })

    def get_target_tvd_at_vs(self, vs_coordinate: float) -> Optional[float]:
        """
        Получить target TVD по VS координате через StarSteer

        Args:
            vs_coordinate: VS coordinate в МЕТРАХ (внутренние единицы Guard)

        Returns:
            TVD value в МЕТРАХ или None если ошибка
        """

        # Конвертируем VS в единицы проекта для StarSteer API
        vs_for_starsteer = vs_coordinate
        if self.project_measure_unit == 'FOOT':
            # Проект в футах - конвертируем из метров в футы
            METERS_TO_FEET = 1.0 / 0.3048
            vs_for_starsteer = vs_coordinate * METERS_TO_FEET
            logger.debug(f"Converting VS for StarSteer: {vs_coordinate:.2f}m → {vs_for_starsteer:.2f}ft")

        logger.debug(f"Requesting target TVD @ VS={vs_for_starsteer:.2f} ({self.project_measure_unit})")

        # Log request start to ng_logger
        if self.ng_logger:
            self.ng_logger.log_checkpoint("starsteer_json_request_start", {
                "vs_coordinate": vs_coordinate,
                "target_well": self.target_well_name,
                "max_retries": self.max_retries,
                "timeout_seconds": self.timeout_seconds
            })

        for attempt in range(self.max_retries):
            try:
                # Создаем уникальный request (с VS в единицах проекта)
                request_data = self._create_request(vs_for_starsteer)

                # Log request creation to ng_logger
                if self.ng_logger:
                    self.ng_logger.log_checkpoint("starsteer_json_request_created", {
                        "request_id": request_data['request_id'],
                        "vs_coordinate": vs_coordinate,
                        "attempt": attempt + 1
                    })

                # Записываем request
                success = self._write_request(request_data)
                if not success:
                    logger.warning(f"Failed to write request, attempt {attempt + 1}/{self.max_retries}")

                    # Log write failure to ng_logger
                    if self.ng_logger:
                        self.ng_logger.log_error("starsteer_json_write",
                                               f"Failed to write request file, attempt {attempt + 1}", {
                                                   "request_id": request_data['request_id'],
                                                   "attempt": attempt + 1
                                               })
                    continue

                # Log request written to ng_logger
                if self.ng_logger:
                    self.ng_logger.log_checkpoint("starsteer_json_request_written", {
                        "request_id": request_data['request_id'],
                        "request_file": str(self.request_file),
                        "attempt": attempt + 1
                    })

                # Ждем response
                response_data = self._wait_for_response(request_data['request_id'])
                if response_data is None:
                    logger.warning(f"No response received, attempt {attempt + 1}/{self.max_retries}")

                    # Log timeout to ng_logger
                    if self.ng_logger:
                        self.ng_logger.log_error("starsteer_json_timeout",
                                               f"No response received within {self.timeout_seconds}s, attempt {attempt + 1}", {
                                                   "request_id": request_data['request_id'],
                                                   "timeout_seconds": self.timeout_seconds,
                                                   "attempt": attempt + 1
                                               })
                    continue

                # Log response received to ng_logger
                if self.ng_logger:
                    self.ng_logger.log_checkpoint("starsteer_json_response_received", {
                        "request_id": request_data['request_id'],
                        "response_status": response_data.get('status'),
                        "attempt": attempt + 1
                    })

                # Парсим response
                tvd_from_starsteer = self._parse_response(response_data)
                if tvd_from_starsteer is not None:
                    # Конвертируем TVD обратно в метры если проект в футах
                    tvd_meters = tvd_from_starsteer
                    if self.project_measure_unit == 'FOOT':
                        FEET_TO_METERS = 0.3048
                        tvd_meters = tvd_from_starsteer * FEET_TO_METERS
                        logger.debug(f"Converting TVD from StarSteer: {tvd_from_starsteer:.2f}ft → {tvd_meters:.2f}m")

                    logger.info(f"StarSteer TVD @ VS={vs_coordinate:.2f}m: {tvd_meters:.3f}m")

                    # Log successful result to ng_logger
                    if self.ng_logger:
                        self.ng_logger.log_checkpoint("starsteer_json_success", {
                            "request_id": request_data['request_id'],
                            "vs_coordinate": vs_coordinate,
                            "target_tvd": tvd_meters,
                            "attempt": attempt + 1,
                            "total_attempts": attempt + 1
                        })
                    return tvd_meters
                else:
                    logger.warning(f"Invalid response, attempt {attempt + 1}/{self.max_retries}")

                    # Log invalid response to ng_logger
                    if self.ng_logger:
                        self.ng_logger.log_error("starsteer_json_invalid_response",
                                               f"Invalid response parsing, attempt {attempt + 1}", {
                                                   "request_id": request_data['request_id'],
                                                   "response_status": response_data.get('status'),
                                                   "attempt": attempt + 1
                                               })

            except Exception as e:
                logger.error(f"StarSteer request failed, attempt {attempt + 1}/{self.max_retries}: {e}")

                # Log exception to ng_logger
                if self.ng_logger:
                    self.ng_logger.log_error("starsteer_json_exception",
                                           f"Request failed with exception, attempt {attempt + 1}: {str(e)}", {
                                               "vs_coordinate": vs_coordinate,
                                               "attempt": attempt + 1,
                                               "exception_type": type(e).__name__
                                           })

            # Delay before retry
            if attempt < self.max_retries - 1:
                time.sleep(1.0 * (attempt + 1))  # Exponential backoff

        logger.error(f"All {self.max_retries} attempts failed to get TVD @ VS={vs_coordinate:.2f}")

        # Log final failure to ng_logger
        if self.ng_logger:
            self.ng_logger.log_error("starsteer_json_final_failure",
                                   f"All {self.max_retries} attempts failed to get TVD", {
                                       "vs_coordinate": vs_coordinate,
                                       "total_attempts": self.max_retries,
                                       "timeout_seconds": self.timeout_seconds
                                   })
        return None

    def _create_request(self, vs_coordinate: float) -> Dict[str, Any]:
        """Создать JSON request для StarSteer"""

        request_id = f"ng_req_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

        return {
            "request_id": request_id,
            "target_well": self.target_well_name,
            "vs_coordinate": vs_coordinate,
            "target_line_name": self.target_line_name,
            "timestamp": datetime.now().isoformat(),
            "timeout_seconds": self.timeout_seconds
        }

    def _write_request(self, request_data: Dict[str, Any]) -> bool:
        """Записать request в JSON файл (atomic operation)"""

        try:
            temp_file = self.request_file.with_suffix('.tmp')

            # Atomic write
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(request_data, f, indent=2, ensure_ascii=False)

            # Atomic replace (works on both Windows and Linux)
            temp_file.replace(self.request_file)

            logger.debug(f"Request written: {request_data['request_id']}")
            return True

        except Exception as e:
            logger.error(f"Failed to write request: {e}")
            return False

    def _wait_for_response(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Ждать response файл с exponential backoff"""

        start_time = time.time()
        poll_interval = self.poll_interval

        while time.time() - start_time < self.timeout_seconds:
            try:
                # Проверяем существование response файла
                if self.response_file.exists():
                    with open(self.response_file, 'r', encoding='utf-8') as f:
                        response_data = json.load(f)

                    # Проверяем request_id
                    if response_data.get('request_id') == request_id:
                        # Удаляем response файл после прочтения
                        try:
                            self.response_file.unlink()
                        except Exception as e:
                            logger.warning(f"Failed to remove response file: {e}")

                        logger.debug(f"Response received: {request_id}")
                        return response_data
                    else:
                        logger.debug(f"Response for different request: {response_data.get('request_id')} vs {request_id}")

                # Exponential backoff
                time.sleep(poll_interval)
                poll_interval = min(poll_interval * 1.5, 10.0)  # Max 10 seconds

            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in response file: {e}")
                time.sleep(poll_interval)
            except Exception as e:
                logger.error(f"Error reading response: {e}")
                time.sleep(poll_interval)

        logger.warning(f"Timeout waiting for response: {request_id}")
        return None

    def _parse_response(self, response_data: Dict[str, Any]) -> Optional[float]:
        """Парсить response и извлечь TVD"""

        try:
            status = response_data.get('status')

            if status == 'success':
                tvd = response_data.get('tvd')
                if tvd is not None:
                    return float(tvd)
                else:
                    logger.error("Success response missing TVD field")
                    return None

            elif status == 'error':
                error_type = response_data.get('error', 'unknown')
                error_message = response_data.get('error_message', 'No error message')
                logger.error(f"StarSteer error: {error_type} - {error_message}")
                return None

            else:
                logger.error(f"Unknown response status: {status}")
                return None

        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None

    def get_target_line_info(self) -> Optional[Dict[str, Any]]:
        """
        Получить дополнительную информацию о target line
        Возвращает cached данные из последнего successful response
        """

        # Для упрощения возвращаем базовую информацию
        return {
            'name': self.target_line_name,
            'well_name': self.target_well_name,
            'source': 'StarSteer',
            'interface_version': '1.0'
        }

    def is_available(self) -> bool:
        """Проверить доступность StarSteer интерфейса"""

        try:
            # Проверяем, что SS directory доступен
            if not self.ss_directory.exists():
                return False

            # Проверяем write permissions
            test_file = self.ss_directory / ".test_write"
            test_file.write_text("test")
            test_file.unlink()

            return True

        except Exception as e:
            logger.warning(f"StarSteer interface not available: {e}")
            return False

    def cleanup_stale_files(self, max_age_seconds: int = 300):
        """Очистить старые request/response файлы"""

        current_time = time.time()

        for file_path in [self.request_file, self.response_file]:
            if file_path.exists():
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        logger.info(f"Removed stale file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove stale file {file_path}: {e}")