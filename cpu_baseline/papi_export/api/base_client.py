import os
import base64
import hashlib
import uuid
import logging
import time
from typing import Dict, Optional, Any
from urllib.parse import urljoin

from oauthlib.oauth2 import BackendApplicationClient, LegacyApplicationClient
from requests import codes as status_codes
from requests.auth import HTTPBasicAuth
from requests_oauthlib import OAuth2Session

logger = logging.getLogger(__name__)


class PAPIException(Exception):
    """Base exception for PAPI errors"""
    pass


class AccessTokenFailureException(PAPIException):
    """Failed to get access token"""
    pass


class BasePapiClient:
    """Base PAPI client with OAuth2 authentication - taken from papi_connection_compl.py"""
    
    DEFAULT_OFFSET = 0
    DEFAULT_LIMIT = 100
    LIMIT_MAX = 200
    DATETIME_FORMAT = '%Y-%m-%dT%H:%M:%SZ'

    def __init__(
        self,
        papi_url: str,
        papi_auth_url: str,
        papi_client_id: str,
        papi_client_secret: str,
        solo_username: str = None,
        solo_password: str = None,
        headers: Optional[Dict] = None,
        proxies: Optional[Dict] = None,
    ):
        self.papi_url = papi_url
        self.token_url = f'{papi_auth_url}/token'
        self.papi_client_id = papi_client_id
        self.papi_client_secret = papi_client_secret
        self.solo_username = solo_username
        self.solo_password = solo_password
        self.headers = headers or {}
        self.proxies = proxies or {}
        self._session = None

    @property
    def session(self):
        if not self._session:
            self._session = self._get_session()
        return self._session

    def _get_session(self):
        """Create OAuth2 session with auto-refresh"""
        token_params = {
            'token_url': self.token_url,
            'client_id': self.papi_client_id,
            'client_secret': self.papi_client_secret,
            'auth': HTTPBasicAuth(self.papi_client_id, self.papi_client_secret),
            'headers': self.headers,
        }

        if self.solo_username and self.solo_password:
            # Password grant type for user credentials
            client = LegacyApplicationClient(client_id=self.papi_client_id)
            token_params['username'] = self.solo_username
            token_params['password'] = self.solo_password
        else:
            # Client credentials grant type
            client = BackendApplicationClient(client_id=self.papi_client_id)

        try:
            auth_session = OAuth2Session(client=client)
            auth_session.proxies.update(self.proxies)
            
            logger.info(f"Connecting to PAPI...")
            token_data = auth_session.fetch_token(**token_params)
            logger.info("PAPI connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to PAPI: {e}")
            raise AccessTokenFailureException(
                'Failed to get access token. Please check that your auth settings are correct.'
            )

        session = OAuth2Session(
            client=client, 
            token=token_data, 
            auto_refresh_url=self.token_url, 
            token_updater=self._update_token_data
        )

        # Client credentials grant type does not support token refreshing
        if isinstance(session._client, BackendApplicationClient):
            session.refresh_token = lambda *args, **kwargs: session.fetch_token(**token_params)

        session.headers.update({'Authorization': f"Bearer {token_data['access_token']}"})
        session.headers.update(self.headers)
        session.proxies.update(self.proxies)

        # Add basic auth to refresh_token method
        session.refresh_token = self._wrap_with_auth(session.refresh_token)

        # Session hook for retry/reconnect
        session.hooks['response'].append(self._retry_on_failure)

        return session

    def _wrap_with_auth(self, func):
        """Wrap function with Basic auth"""
        def wrapper(*args, **kwargs):
            kwargs['auth'] = HTTPBasicAuth(self.papi_client_id, self.papi_client_secret)
            return func(*args, **kwargs)
        return wrapper

    def _update_token_data(self, token_data):
        """Update session headers with new token"""
        self.session.headers.update({'Authorization': f"Bearer {token_data['access_token']}"})

    def _retry_on_failure(self, response, *args, **kwargs):
        """Auto retry with reconnect on timeout/401/ConnectionError using session hooks"""
        import time
        import os
        from requests.exceptions import ConnectionError, Timeout, RequestException

        # Night Guard Logger integration
        try:
            from night_guard_logger import NightGuardLogger
            ng_logger = NightGuardLogger()
        except ImportError:
            ng_logger = None

        # Конфигурация из .env без defaults
        max_retries = int(os.getenv('PAPI_MAX_RETRY_COUNT'))
        simple_retry_count = int(os.getenv('PAPI_SIMPLE_RETRY_COUNT'))
        backoff_factor = float(os.getenv('PAPI_BACKOFF_FACTOR'))

        # Получаем текущий retry count из request
        retry_count = getattr(response.request, '_retry_count', 0)

        # Проверяем нужен ли retry для HTTP статусов
        need_retry = (
            response.status_code in [401, 403, 429, 502, 503, 504] or
            # Также обрабатываем случаи когда response на самом деле Exception
            isinstance(response, (ConnectionError, Timeout, RequestException))
        )

        if need_retry and retry_count < max_retries:
            # Логируем конкретную причину retry
            if hasattr(response, 'status_code'):
                error_msg = f"PAPI HTTP {response.status_code} error on {response.url} (attempt {retry_count + 1}/{max_retries})"
                logger.warning(error_msg)
                if ng_logger:
                    ng_logger.log_checkpoint("papi_http_error", {
                        "status_code": response.status_code,
                        "url": str(response.url),
                        "attempt": retry_count + 1,
                        "max_retries": max_retries
                    }, False)
            else:
                error_msg = f"PAPI connection error: {type(response).__name__} on {response.request.url} (attempt {retry_count + 1}/{max_retries})"
                logger.warning(error_msg)
                if ng_logger:
                    ng_logger.log_checkpoint("papi_connection_error", {
                        "error_type": type(response).__name__,
                        "url": str(response.request.url),
                        "attempt": retry_count + 1,
                        "max_retries": max_retries
                    }, False)

            # Решаем: простой retry или OAuth переподключение
            if retry_count < simple_retry_count and response.status_code in [503, 502, 504]:
                # Простой retry БЕЗ переподключения для временных серверных проблем
                logger.info(f"PAPI simple retry (attempt {retry_count + 1}/{simple_retry_count})")
                if ng_logger:
                    ng_logger.log_checkpoint("papi_simple_retry", {
                        "attempt": retry_count + 1,
                        "max_retries": max_retries
                    }, True)
            else:
                # Полный OAuth переподключение для 401/403 или после исчерпания простых retry
                self._session = None
                logger.info(f"PAPI OAuth reconnecting (attempt {retry_count + 1}/{max_retries})")
                if ng_logger:
                    ng_logger.log_checkpoint("papi_oauth_reconnect", {
                        "attempt": retry_count + 1,
                        "max_retries": max_retries
                    }, True)

            # Повторяем запрос с новой сессией
            retry_count += 1
            time.sleep(backoff_factor * (2 ** retry_count))

            # Клонируем оригинальный запрос
            new_request = response.request.copy()
            new_request._retry_count = retry_count

            try:
                # Отправляем через новую сессию
                new_response = self.session.send(new_request)
                if ng_logger:
                    ng_logger.log_checkpoint("papi_retry_success", {
                        "attempt": retry_count,
                        "total_attempts": retry_count,
                        "url": str(new_request.url)
                    }, True)
                return new_response
            except Exception as e:
                logger.error(f"PAPI retry attempt {retry_count} failed: {e}")
                if ng_logger:
                    ng_logger.log_checkpoint("papi_retry_failed", {
                        "attempt": retry_count,
                        "error": str(e)
                    }, False)
                # Возвращаем исходную ошибку если retry тоже не удался
                return response

        elif retry_count >= max_retries:
            if hasattr(response, 'status_code'):
                error_msg = f"PAPI permanently failed on {response.url} after {max_retries} attempts (HTTP {response.status_code})"
                logger.error(error_msg)
                if ng_logger:
                    ng_logger.log_checkpoint("papi_permanent_failure", {
                        "url": str(response.url),
                        "status_code": response.status_code,
                        "max_retries": max_retries
                    }, False)
            else:
                error_msg = f"PAPI permanently failed after {max_retries} attempts: {type(response).__name__}"
                logger.error(error_msg)
                if ng_logger:
                    ng_logger.log_checkpoint("papi_permanent_failure", {
                        "error_type": type(response).__name__,
                        "max_retries": max_retries
                    }, False)

        return response

    def _send_request(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None):
        """Send GET request"""
        full_url = f'{self.papi_url}/{url}'
        
        # Add timeout: (connect_timeout, read_timeout)
        timeout = (int(os.getenv('PAPI_CONNECT_TIMEOUT')), int(os.getenv('PAPI_READ_TIMEOUT_GET')))
        response = self.session.get(full_url, params=params, headers=headers, timeout=timeout)

        if response.status_code != status_codes.ok:
            error_msg = f"API request failed: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise PAPIException(error_msg)

        if response.text:
            return response.json()

        return response

    def _send_post_request(
        self, url: str, request_data: Dict[str, Any],
        params: Optional[Dict] = None, headers: Optional[Dict] = None
    ):
        """Send POST request"""
        full_url = f'{self.papi_url}/{url}'

        # Add timeout: (connect_timeout, read_timeout)
        timeout = (int(os.getenv('PAPI_CONNECT_TIMEOUT')), int(os.getenv('PAPI_READ_TIMEOUT_POST')))
        response = self.session.post(full_url, params=params, json=request_data, headers=headers, timeout=timeout)

        if response.status_code != status_codes.ok:
            error_msg = f"API request failed: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise PAPIException(error_msg)

        # Rate limiting for interpretation requests
        if 'interpretations' in url or 'segments' in url:
            delay = int(os.getenv('PAPI_REQUEST_DELAY_SECONDS'))
            time.sleep(delay)
            logger.debug(f"Rate limiting: waited {delay}s after interpretation/segment POST request")

        if response.text:
            return response.json()

        return response

    def _send_put_request(self, url: str, request_data: Dict[str, Any], headers: Optional[Dict] = None):
        """Send PUT request"""
        full_url = f'{self.papi_url}/{url}'

        # Add timeout: (connect_timeout, read_timeout)
        timeout = (int(os.getenv('PAPI_CONNECT_TIMEOUT')), int(os.getenv('PAPI_READ_TIMEOUT_POST')))
        response = self.session.put(full_url, json=request_data, headers=headers, timeout=timeout)

        if response.status_code != status_codes.ok:
            error_msg = f"API request failed: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise PAPIException(error_msg)

        if response.text:
            return response.json()

        return response

    def _send_patch_request(self, url: str, request_data: Dict[str, Any], headers: Optional[Dict] = None):
        """Send PATCH request"""
        full_url = f'{self.papi_url}/{url}'

        # Add timeout: (connect_timeout, read_timeout)
        timeout = (int(os.getenv('PAPI_CONNECT_TIMEOUT')), int(os.getenv('PAPI_READ_TIMEOUT_POST')))
        response = self.session.patch(full_url, json=request_data, headers=headers, timeout=timeout)

        if response.status_code != status_codes.ok:
            error_msg = f"API request failed: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise PAPIException(error_msg)

        if response.text:
            return response.json()

        return response

    def _send_delete_request(self, url: str, headers: Optional[Dict] = None):
        """Send DELETE request"""
        full_url = f'{self.papi_url}/{url}'

        # Add timeout: (connect_timeout, read_timeout)
        timeout = (int(os.getenv('PAPI_CONNECT_TIMEOUT')), int(os.getenv('PAPI_READ_TIMEOUT_DELETE')))
        response = self.session.delete(full_url, headers=headers, timeout=timeout)

        if response.status_code != status_codes.ok:
            error_msg = f"API request failed: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise PAPIException(error_msg)

        if response.text:
            return response.json()

        return response