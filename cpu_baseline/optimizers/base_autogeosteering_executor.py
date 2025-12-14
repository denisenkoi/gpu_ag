# base_autogeosteering_executor.py

"""
Abstract base class for AutoGeosteering Executors
Defines common interface for both C++ and Python implementations
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseAutoGeosteeringExecutor(ABC):
    """Abstract base executor for AutoGeosteering implementations"""

    def __init__(self, work_dir: str):
        """
        Initialize base executor

        Args:
            work_dir: Working directory for executor operations
        """
        self.work_dir = work_dir

    @abstractmethod
    def start_daemon(self):
        """Start the executor daemon/service"""
        pass

    @abstractmethod
    def stop_daemon(self):
        """Stop the executor daemon/service"""
        pass

    @abstractmethod
    def initialize_well(self, well_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize well with first dataset

        Args:
            well_data: Well JSON data for initialization

        Returns:
            Dict containing interpretation result
        """
        pass

    @abstractmethod
    def update_well_data(self, well_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update well with new data

        Args:
            well_data: Updated well JSON data

        Returns:
            Dict containing updated interpretation result
        """
        pass

    @abstractmethod
    def get_interpretation_from_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract interpretation from executor result

        Args:
            result: Result from initialize_well() or update_well_data()

        Returns:
            Dict containing interpretation data
        """
        pass

    def __enter__(self):
        """Context manager entry"""
        self.start_daemon()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_daemon()