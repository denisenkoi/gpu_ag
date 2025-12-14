import base64
import json
import logging
import time
from typing import Dict, List, Optional, Any
import requests

logger = logging.getLogger(__name__)


class PAPIConnector:
    """PAPI connector for Solo Cloud API - based on working papi_upload_utils.py"""
    
    def __init__(self, config):
        """Initialize PAPI connector with configuration"""
        self.config = config
        self.base_url = config.base_url
        self.access_token = None
        self.refresh_token = None
        self.token_expiry = 0
        self.session = requests.Session()
        
        # Wells cache for fast lookup
        self.wells_cache = {}
        self.wells_by_name = {}
        
    def authenticate(self) -> bool:
        """Authenticate with OAuth2 and get access token"""
        logger.info(f"Authenticating with PAPI at {self.base_url}")
        
        # Prepare Basic auth header
        credentials = f"{self.config.client_id}:{self.config.client_secret}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        
        headers = {
            'Authorization': f'Basic {encoded_credentials}',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        data = {
            'grant_type': 'password',
            'username': self.config.username,
            'password': self.config.password
        }
        
        url = f"{self.base_url}/oauth/token"
        
        response = self.session.post(url, headers=headers, data=data)
        
        if response.status_code != 200:
            logger.error(f"Authentication failed: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
        token_data = response.json()
        self.access_token = token_data['access_token']
        self.refresh_token = token_data.get('refresh_token')
        expires_in = token_data.get('expires_in', 3600)
        self.token_expiry = time.time() + expires_in - 60  # Refresh 1 minute early
        
        # Set default authorization header
        self.session.headers['Authorization'] = f'Bearer {self.access_token}'
        
        logger.info("Successfully authenticated with PAPI")
        return True
        
    def ensure_authenticated(self):
        """Ensure we have a valid token"""
        if not self.access_token or time.time() >= self.token_expiry:
            assert self.authenticate(), "Failed to authenticate with PAPI"
            
    def get_project_by_name(self, project_name: str) -> Optional[Dict]:
        """Get project by name"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/projects"
        params = {'filter': project_name, 'limit': 100}
        
        response = self.session.get(url, params=params)
        assert response.status_code == 200, f"Failed to get projects: {response.status_code}"
        
        data = response.json()
        projects = data.get('content', [])

        print(projects)

        for project in projects:
            if project['name'] == project_name:
                logger.info(f"Found project: {project_name} (UUID: {project['uuid']})")
                return project
                
        return None
        
    def get_well_by_name(self, project_uuid: str, well_name: str) -> Optional[Dict]:
        """Get well by name from project"""
        self.ensure_authenticated()
        
        # Check cache first
        if well_name in self.wells_by_name:
            return self.wells_by_name[well_name]
            
        url = f"{self.base_url}/public/api/v1/projects/{project_uuid}/wells"
        params = {'filter': well_name, 'limit': 100}
        
        response = self.session.get(url, params=params)
        assert response.status_code == 200, f"Failed to get wells: {response.status_code}"
        
        data = response.json()
        wells = data.get('content', [])
        
        for well in wells:
            if well['name'] == well_name:
                logger.info(f"Found well: {well_name} (UUID: {well['uuid']})")
                # Cache it
                self.wells_cache[well['uuid']] = well
                self.wells_by_name[well_name] = well
                return well
                
        return None
        
    def get_well_trajectory(self, well_uuid: str) -> List[Dict]:
        """Get well trajectory"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/wells/{well_uuid}/trajectory"
        
        response = self.session.get(url)
        assert response.status_code == 200, f"Failed to get trajectory: {response.status_code}"
        
        data = response.json()
        return data.get('content', [])
        
    def get_well_metadata(self, well_uuid: str) -> Dict:
        """Get well metadata including coordinates"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/wells/{well_uuid}"
        
        response = self.session.get(url)
        assert response.status_code == 200, f"Failed to get well metadata: {response.status_code}"
        
        return response.json()
        
    def get_linked_typewell(self, well_uuid: str) -> Optional[Dict]:
        """Get linked typewell for a lateral"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/wells/{well_uuid}/linked"
        params = {'offset': 0, 'limit': 10}
        
        response = self.session.get(url, params=params)
        assert response.status_code == 200, f"Failed to get linked typewells: {response.status_code}"
        
        data = response.json()
        typewells = data.get('content', [])
        
        if typewells:
            # Return first linked typewell with shift
            return typewells[0]
            
        return None
        
    def get_typewell_metadata(self, typewell_uuid: str) -> Dict:
        """Get typewell metadata"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/typewells/{typewell_uuid}"
        
        response = self.session.get(url)
        assert response.status_code == 200, f"Failed to get typewell metadata: {response.status_code}"
        
        return response.json()
        
    def get_typewell_trajectory(self, typewell_uuid: str) -> List[Dict]:
        """Get typewell trajectory"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/typewells/{typewell_uuid}/trajectory"
        
        response = self.session.get(url)
        assert response.status_code == 200, f"Failed to get typewell trajectory: {response.status_code}"
        
        data = response.json()
        return data.get('content', [])
        
    def get_logs_by_well(self, well_uuid: str) -> List[Dict]:
        """Get logs for a lateral well"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/wells/{well_uuid}/logs"
        params = {'offset': 0, 'limit': 100}
        
        response = self.session.get(url, params=params)
        assert response.status_code == 200, f"Failed to get logs: {response.status_code}"
        
        data = response.json()
        return data.get('content', [])
        
    def get_logs_by_typewell(self, typewell_uuid: str) -> List[Dict]:
        """Get logs for a typewell"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/typewells/{typewell_uuid}/logs"
        params = {'offset': 0, 'limit': 100}
        
        response = self.session.get(url, params=params)
        assert response.status_code == 200, f"Failed to get typewell logs: {response.status_code}"
        
        data = response.json()
        return data.get('content', [])
        
    def get_log_data(self, log_uuid: str) -> Dict:
        """Get log data points"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/logs/{log_uuid}/data"
        
        response = self.session.get(url)
        assert response.status_code == 200, f"Failed to get log data: {response.status_code}"
        
        return response.json()
        
    def get_topsets_by_well(self, well_uuid: str) -> List[Dict]:
        """Get topsets for a well"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/wells/{well_uuid}/topsets"
        params = {'offset': 0, 'limit': 100}
        
        response = self.session.get(url, params=params)
        assert response.status_code == 200, f"Failed to get topsets: {response.status_code}"
        
        data = response.json()
        return data.get('content', [])
        
    def get_tops_by_topset(self, topset_uuid: str) -> List[Dict]:
        """Get tops from topset"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/topsets/{topset_uuid}/tops"
        params = {'offset': 0, 'limit': 100}
        
        response = self.session.get(url, params=params)
        assert response.status_code == 200, f"Failed to get tops: {response.status_code}"
        
        data = response.json()
        return data.get('content', [])
        
    def get_interpretations_by_well(self, well_uuid: str) -> List[Dict]:
        """Get interpretations for a well"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/wells/{well_uuid}/interpretations"
        params = {'offset': 0, 'limit': 100}
        
        response = self.session.get(url, params=params)
        assert response.status_code == 200, f"Failed to get interpretations: {response.status_code}"
        
        data = response.json()
        return data.get('content', [])
        
    def get_horizons_data(self, interpretation_uuid: str, spacing: float) -> Dict:
        """Get horizons data with specified spacing"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/interpretations/{interpretation_uuid}/horizons/data/spacing/{spacing}"
        
        response = self.session.get(url)
        assert response.status_code == 200, f"Failed to get horizons data: {response.status_code}"
        
        return response.json()
        
    def get_grids_by_project(self, project_uuid: str) -> List[Dict]:
        """Get grids for a project"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/projects/{project_uuid}/grids"
        params = {'offset': 0, 'limit': 100}
        
        response = self.session.get(url, params=params)
        assert response.status_code == 200, f"Failed to get grids: {response.status_code}"
        
        data = response.json()
        return data.get('content', [])
        
    def get_grid_data(self, grid_uuid: str) -> Dict:
        """Get grid data with metadata"""
        self.ensure_authenticated()
        
        url = f"{self.base_url}/public/api/v1/grids/{grid_uuid}/data"
        
        response = self.session.get(url)
        assert response.status_code == 200, f"Failed to get grid data: {response.status_code}"
        
        return response.json()