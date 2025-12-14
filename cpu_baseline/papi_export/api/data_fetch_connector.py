# whole module 

import logging
from typing import Dict, List, Optional

from .papi_client import create_papi_client

logger = logging.getLogger(__name__)


class DataFetchConnector:
    """Connector for fetching data from PAPI - minimal wrapper over PapiClient"""
    
    def __init__(self):
        """Initialize with corrected PapiClient"""
        logger.info("Initializing PAPI connection...")
        self.client = create_papi_client()
        logger.info("PAPI connection ready")
        
    def get_project_by_name(self, project_name: str) -> Optional[Dict]:
        """Get project by name from both virtual and global projects"""
        params = {'filter': project_name, 'limit': 100, 'offset': 0}

        # 1. Try virtual projects first (StarSteer)
        logger.info(f"Searching in virtual projects: {project_name}")
        response = self.client._send_request('projects/virtual', params=params)
        projects = response.get('content', [])

        for project in projects:
            if project.get('name', '') == project_name:
                logger.info(f"Found VIRTUAL project: {project_name}")
                return project

        # 2. Try global projects
        logger.info(f"Searching in global projects: {project_name}")
        response = self.client._send_request('projects', params=params)
        projects = response.get('content', [])

        for project in projects:
            if project.get('name', '') == project_name:
                logger.info(f"Found GLOBAL project: {project_name}")
                return project

        # 3. Not found in either
        logger.error(f"Project '{project_name}' not found in virtual or global projects")
        return None
        
    def get_wells_by_project(self, project_uuid: str, filter_name: str = None) -> List[Dict]:
        """Get wells from project"""
        url = f'projects/{project_uuid}/wells'
        params = {'limit': 100, 'offset': 0}
        
        if filter_name:
            params['filter'] = filter_name
            
        response = self.client._send_request(url, params=params)
        wells = response.get('content', [])
        return wells
        
    def get_well_by_name(self, project_uuid: str, well_name: str) -> Optional[Dict]:
        """Get specific well by name"""
        wells = self.get_wells_by_project(project_uuid, filter_name=well_name)
        
        for well in wells:
            if well.get('name', '') == well_name:
                logger.info(f"Found well: {well_name}")
                return well
                
        logger.error(f"Well '{well_name}' not found")
        return None
        
    def get_well_metadata(self, well_uuid: str) -> Dict:
        """Get well metadata including coordinates"""
        return self.client._send_request(f'wells/{well_uuid}')
        
    def get_well_trajectory(self, well_uuid: str) -> List[Dict]:
        """Get well trajectory"""
        response = self.client._send_request(f'wells/{well_uuid}/trajectory')
        trajectory = response.get('content', [])
        logger.info(f"Retrieved trajectory with {len(trajectory)} points")
        return trajectory
        
    def get_linked_typewells(self, well_uuid: str) -> List[Dict]:
        """Get linked typewells for a lateral"""
        response = self.client._send_request(f'wells/{well_uuid}/linked', 
                                           params={'offset': 0, 'limit': 10})
        return response.get('content', [])
        
    def get_typewell_metadata(self, typewell_uuid: str) -> Dict:
        """Get typewell metadata"""
        return self.client._send_request(f'typewells/{typewell_uuid}')
        
    def get_typewell_trajectory(self, typewell_uuid: str) -> List[Dict]:
        """Get typewell trajectory"""
        response = self.client._send_request(f'typewells/{typewell_uuid}/trajectory')
        return response.get('content', [])
        
    def get_logs_by_well(self, well_uuid: str) -> List[Dict]:
        """Get logs for a lateral well"""
        response = self.client._send_request(f'wells/{well_uuid}/logs',
                                           params={'offset': 0, 'limit': 100})
        return response.get('content', [])
        
    def get_logs_by_typewell(self, typewell_uuid: str) -> List[Dict]:
        """Get logs for a typewell"""
        response = self.client._send_request(f'typewells/{typewell_uuid}/logs',
                                           params={'offset': 0, 'limit': 100})
        return response.get('content', [])
        
    def get_log_data(self, log_uuid: str) -> Dict:
        """Get log data points"""
        response = self.client._send_request(f'logs/{log_uuid}/data')
        log_points = response.get('log_points', [])
        logger.info(f"Retrieved log data with {len(log_points)} points")
        return response
        
    def get_topsets_by_well(self, well_uuid: str) -> List[Dict]:
        """Get topsets for a well"""
        response = self.client._send_request(f'wells/{well_uuid}/topsets',
                                           params={'offset': 0, 'limit': 100})
        return response.get('content', [])
        
    def get_tops_by_topset(self, topset_uuid: str) -> List[Dict]:
        """Get tops from topset"""
        response = self.client._send_request(f'topsets/{topset_uuid}/tops',
                                           params={'offset': 0, 'limit': 100})
        return response.get('content', [])
        
    def get_interpretations_by_well(self, well_uuid: str) -> List[Dict]:
        """Get interpretations for a well"""
        response = self.client._send_request(f'wells/{well_uuid}/interpretations',
                                           params={'offset': 0, 'limit': 100})
        return response.get('content', [])
        
    def get_horizons_data(self, interpretation_uuid: str, spacing: float) -> Dict:
        """Get horizons data with specified spacing"""
        url = f'interpretations/{interpretation_uuid}/horizons/data/spacing/{spacing}'
        response = self.client._send_request(url)
        content = response.get('content', [])
        logger.info(f"Retrieved horizons data with {len(content)} points")
        return response
        
    def get_grids_by_project(self, project_uuid: str) -> List[Dict]:
        """Get grids for a project"""
        response = self.client._send_request(f'projects/{project_uuid}/grids',
                                           params={'offset': 0, 'limit': 100})
        return response.get('content', [])
        
    def get_grid_data(self, grid_uuid: str) -> Dict:
        """Get grid data with metadata"""
        response = self.client._send_request(f'grids/{grid_uuid}/data')
        data_rows = response.get('data', [])
        logger.info(f"Retrieved grid data with {len(data_rows)} rows")
        return response
    
    def create_interpretation(self, well_uuid: str, name: str) -> str:
        """Create interpretation, return UUID"""
        request_data = {
            "name": name,
            "description": f"Auto-generated interpretation by NightAG"
        }
        response = self.client._send_post_request(f'wells/{well_uuid}/interpretations', request_data)
        interpretation_uuid = response.get('uuid')
        logger.info(f"Created interpretation '{name}': {interpretation_uuid}")
        return interpretation_uuid
    
    def get_segments_by_interpretation(self, interpretation_uuid: str) -> List[Dict]:
        """Get segments for an interpretation"""
        url = f'interpretations/{interpretation_uuid}/segments'
        params = {'offset': 0, 'limit': 1000, 'horizontal_scale': 'MD'}
        response = self.client._send_request(url, params=params)
        segments = response.get('content', [])
        return segments
    
    def delete_segment(self, interpretation_uuid: str, segment_uuid: str) -> None:
        """Delete a segment"""
        try:
            url = f'interpretations/{interpretation_uuid}/segments/{segment_uuid}'
            self.client._send_delete_request(url)
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                pass  # Skip 404 silently
            else:
                raise

    def delete_all_segments(self, interpretation_uuid: str) -> None:
        """Delete all segments from interpretation"""
        segments = self.get_segments_by_interpretation(interpretation_uuid)

        # Sort segments by MD descending (delete from last to first)
        segments_sorted = sorted(segments, key=lambda s: s.get('md', {}).get('val', 0), reverse=True)

        for segment in segments_sorted:
            try:
                self.delete_segment(interpretation_uuid, segment['uuid'])
            except Exception:
                pass  # Skip errors silently
    
    def delete_interpretation(self, interpretation_uuid: str) -> None:
        """Delete interpretation"""
        try:
            self.client._send_delete_request(f'interpretations/{interpretation_uuid}')
            logger.info(f"Deleted interpretation: {interpretation_uuid}")
        except Exception as e:
            if "404" in str(e) or "Not Found" in str(e):
                pass  # Skip 404 silently
            else:
                raise
    
    def delete_interpretations_by_prefix(self, well_uuid: str, name_prefix: str, exact_name: bool = False) -> None:
        """Delete interpretations by prefix or exact name

        Args:
            well_uuid: UUID of the well
            name_prefix: Name prefix to match (or exact name if exact_name=True)
            exact_name: If True, match exact name instead of prefix
        """
        interpretations = self.get_interpretations_by_well(well_uuid)
        deleted_count = 0

        for interpretation in interpretations:
            interp_name = interpretation.get('name', '')
            matches = (interp_name == name_prefix) if exact_name else interp_name.startswith(name_prefix)

            if matches:
                try:
                    self.delete_interpretation(interpretation['uuid'])
                    deleted_count += 1
                    logger.info(f"Deleted interpretation: {interpretation['name']}")
                except Exception as e:
                    logger.warning(f"Failed to delete interpretation {interpretation['name']}: {str(e)}")

        match_type = "with exact name" if exact_name else "with prefix"
        logger.info(f"Deleted {deleted_count} interpretations {match_type} '{name_prefix}'")
        return deleted_count
    
    def add_segment(self, interpretation_uuid: str, segment_data: Dict) -> str:
        """Add segment, return UUID"""
        # Use the working combination: URL query parameter only
        url = f'interpretations/{interpretation_uuid}/segments?horizontal_scale=MD'
        
        # Искусственно подменим MD=0 на MD=1 для избежания дублирования
        test_data = segment_data.copy()
        if test_data.get('md', {}).get('val') == 0.0:
            test_data['md']['val'] = 1.0
            logger.info(f"Changed MD from 0.0 to 1.0 to avoid duplication")
        
        try:
            response = self.client._send_post_request(url, test_data, params=None)
            
            # Handle different response types
            if isinstance(response, dict):
                # JSON response with data
                segment_uuid = response.get('uuid', 'created_no_uuid')
            elif hasattr(response, 'status_code'):
                # Response object - likely empty successful response
                segment_uuid = "created_success"
            else:
                # Unexpected response type
                segment_uuid = "created_unknown"
            
            logger.info(f"Added segment successfully. UUID: {segment_uuid}")
            return segment_uuid
            
        except Exception as e:
            logger.error(f"Failed to add segment: {str(e)}")
            raise
    
    def update_segment(self, interpretation_uuid: str, segment_data: Dict) -> str:
        """Update segment using PATCH, return UUID"""
        url = f'interpretations/{interpretation_uuid}/segments?horizontal_scale=MD'

        if 'uuid' not in segment_data:
            raise ValueError("segment_data must contain 'uuid' field for update")

        try:
            response = self.client._send_patch_request(url, segment_data)

            if isinstance(response, dict):
                segment_uuid = response.get('uuid', segment_data['uuid'])
            elif hasattr(response, 'status_code'):
                segment_uuid = segment_data['uuid']
            else:
                segment_uuid = segment_data['uuid']

            logger.info(f"Updated segment successfully. UUID: {segment_uuid}")
            return segment_uuid

        except Exception as e:
            logger.error(f"Failed to update segment: {str(e)}")
            raise
    
    def get_starred_interpretation(self, well_uuid: str) -> Optional[Dict]:
        """
        Получить starred interpretation для скважины
        Использовать endpoint: GET /wells/{uuid}/starred
        Вернуть interpretation UUID из ответа
        """
        starred_response = self.client._send_request(f'wells/{well_uuid}/starred')
        assert starred_response, f"No starred data returned for well {well_uuid}"
        logger.info(f"Starred data received: {starred_response}")
        
        interpretation_uuid = starred_response['interpretation']
        assert interpretation_uuid, f"No starred interpretation found for well {well_uuid}, starred data: {starred_response}"
        logger.info(f"Found starred interpretation: {interpretation_uuid}")
        
        # Возвращаем starred response с UUID - direct interpretation endpoint не работает
        return {'uuid': interpretation_uuid, 'starred_data': starred_response}
    
    def get_interpretation_by_uuid(self, well_uuid: str, interpretation_uuid: str) -> Optional[Dict]:
        """
        Получить интерпретацию по UUID через правильный endpoint
        API /interpretations/{uuid} не существует - используем /wells/{uuid}/interpretations
        """
        interpretations = self.get_interpretations_by_well(well_uuid)
        for interp in interpretations:
            if interp.get('uuid') == interpretation_uuid:
                logger.info(f"Found interpretation by UUID: {interpretation_uuid}, name: {interp.get('name')}")
                return interp
        logger.warning(f"Interpretation UUID {interpretation_uuid} not found in well {well_uuid}")
        return None
    
    def get_starred_target_line(self, well_uuid: str) -> Optional[Dict]:
        """
        Получить starred target line для скважины
        Использовать endpoint: GET /wells/{uuid}/targetlines/starred
        Может вернуть None если starred target line не настроен
        """
        try:
            logger.info(f"PAPI REQUEST: GET wells/{well_uuid}/targetlines/starred")
            response = self.client._send_request(f'wells/{well_uuid}/targetlines/starred')

            # RAW RESPONSE LOGGING
            logger.info("=" * 80)
            logger.info("RAW_RESPONSE: DataFetchConnector.get_starred_target_line() - PAPI Raw Response")
            logger.info(f"RAW_RESPONSE: Endpoint: GET wells/{well_uuid}/targetlines/starred")
            logger.info(f"RAW_RESPONSE: Response Type: {type(response)}")
            if response:
                logger.info(f"RAW_RESPONSE: Response Keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
                logger.info(f"RAW_RESPONSE: Full Response: {response}")
            else:
                logger.info("RAW_RESPONSE: Response is None/Empty")
            logger.info("=" * 80)

            if response and response.get('uuid'):
                logger.info(f"Found starred target line: {response['uuid']}")
                return response
            else:
                logger.info(f"No starred target line configured for well {well_uuid}")
                return None
        except Exception as e:
            logger.info(f"No starred target line found for well {well_uuid}: {str(e)}")
            return None

    def get_target_line_data(self, target_line_uuid: str) -> Optional[Dict]:
        """
        Получить данные target line по UUID
        Использовать endpoint: GET /targetlines/{target_line_uuid}/data
        Возвращает координаты и точки target line
        """
        try:
            logger.info(f"PAPI REQUEST: GET targetlines/{target_line_uuid}/data")
            response = self.client._send_request(f'targetlines/{target_line_uuid}/data')

            # RAW RESPONSE LOGGING - логируем сырой ответ сразу после получения
            logger.info("=" * 80)
            logger.info("RAW_RESPONSE: DataFetchConnector.get_target_line_data() - PAPI Raw Response")
            logger.info(f"RAW_RESPONSE: Endpoint: GET targetlines/{target_line_uuid}/data")
            logger.info(f"RAW_RESPONSE: Response Type: {type(response)}")
            if response:
                logger.info(f"RAW_RESPONSE: Response Keys: {list(response.keys()) if isinstance(response, dict) else 'Not a dict'}")
                logger.info(f"RAW_RESPONSE: Full Response: {response}")
            else:
                logger.info("RAW_RESPONSE: Response is None/Empty")
            logger.info("=" * 80)

            if response:
                logger.info(f"Retrieved target line data for {target_line_uuid}")
                return response
            else:
                logger.warning(f"No target line data found for {target_line_uuid}")
                return None
        except Exception as e:
            logger.warning(f"Failed to get target line data for {target_line_uuid}: {e}")
            return None

    def get_interpretation_segments_as_horizons(self, interpretation_uuid: str) -> List[Dict]:
        """
        Получить сегменты интерпретации через horizons/data/spacing/1
        Использовать endpoint: GET /interpretations/{uuid}/horizons/data/spacing/1
        """
        # Используем integer spacing вместо float
        response = self.get_horizons_data(interpretation_uuid, spacing=1)
        assert response, f"No horizons data returned for interpretation {interpretation_uuid}"
        
        content = response['content']
        assert content is not None, f"No content in horizons data for interpretation {interpretation_uuid}, response: {response}"
        
        logger.info(f"Retrieved {len(content)} horizon segments for interpretation {interpretation_uuid}")
        return content