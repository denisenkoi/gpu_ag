import os
import base64
import hashlib
import uuid
import logging
from typing import Dict, Optional, NamedTuple
from urllib.parse import urljoin
from dotenv import load_dotenv

from .base_client import BasePapiClient

logger = logging.getLogger(__name__)

# Load environment variables from root .env
load_dotenv()

# Constants matching Good version
PYTHON_SDK_APP_ID = 'pythonsdk'
SOLO_PAPI_URL = 'public/api/v1'
SOLO_OPEN_AUTH_SERVICE_URL = 'oauth'
SDK_VERSION = '1.0.0'


class SettingsAuth(NamedTuple):
    """Settings for authentication - matching Good version"""
    client_id: str
    client_secret: str
    papi_domain_name: str
    proxies: Optional[Dict]


class PapiClient(BasePapiClient):
    """PAPI client exactly like Good version"""
    
    def __init__(self, settings_auth: SettingsAuth):
        """Initialize with settings_auth like Good version"""
        
        # Create headers exactly like Good version
        app_id = base64.standard_b64encode(PYTHON_SDK_APP_ID.encode()).decode()
        fingerprint = hashlib.sha256(str(uuid.uuid4()).encode()).hexdigest()
        
        headers = {
            'User-Agent': f'PythonSDK/{SDK_VERSION}',
            'X-Solo-Hid': f'{fingerprint}:{app_id}',
        }
        
        # Build URLs exactly like Good version
        papi_url = urljoin(settings_auth.papi_domain_name, SOLO_PAPI_URL)
        papi_auth_url = urljoin(settings_auth.papi_domain_name, SOLO_OPEN_AUTH_SERVICE_URL)
        
        logger.info(f"PapiClient URLs:")
        logger.info(f"  Domain: {settings_auth.papi_domain_name}")
        logger.info(f"  PAPI URL: {papi_url}")
        logger.info(f"  Auth URL: {papi_auth_url}")
        logger.info(f"  Client ID: {settings_auth.client_id}")
        
        # Initialize base client exactly like Good version
        super().__init__(
            papi_url=papi_url,
            papi_auth_url=papi_auth_url,
            papi_client_id=settings_auth.client_id,
            papi_client_secret=settings_auth.client_secret,
            headers=headers,
            proxies=self._get_proxies(settings_auth.proxies),
        )

    def _get_proxies(self, proxies_data: Optional[Dict]) -> Dict:
        """Get proxies - copied from Good version"""
        proxies = {}
        
        if not proxies_data:
            return proxies

        for scheme, url in proxies_data.items():
            if self._is_correct_proxy_url(url):
                proxies[scheme] = url

        return proxies

    def _is_correct_proxy_url(self, url: str) -> bool:
        """Validate proxy URL - copied from Good version"""
        from urllib.parse import urlparse
        
        parsed_url = urlparse(url)

        if parsed_url.scheme not in ['https', 'http']:
            return False

        if not isinstance(parsed_url.port, int):
            return False

        return True


def create_papi_client() -> PapiClient:
    """Factory function to create PapiClient from environment variables"""
    
    # Get credentials from .env exactly like Good version
    client_id = os.getenv('SOLO_SDK_CLIENT_ID')
    client_secret = os.getenv('SOLO_SDK_CLIENT_SECRET')
    papi_domain = os.getenv('SOLO_DOMAIN')
    
    # Validate required credentials
    assert client_id, "SOLO_SDK_CLIENT_ID not set in .env"
    assert client_secret, "SOLO_SDK_CLIENT_SECRET not set in .env"
    assert papi_domain, "SOLO_DOMAIN not set in .env"
    
    logger.info(f"Creating PapiClient from environment:")
    logger.info(f"  Client ID: {client_id}")
    logger.info(f"  Domain: {papi_domain}")
    
    # Create SettingsAuth exactly like Good version
    settings_auth = SettingsAuth(
        client_id=client_id,
        client_secret=client_secret,
        papi_domain_name=papi_domain,
        proxies=None
    )
    
    return PapiClient(settings_auth)