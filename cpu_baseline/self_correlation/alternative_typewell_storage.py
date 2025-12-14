"""
Alternative TypeWell Storage Module

Manages saving and loading alternative typewell data to/from JSON files with versioning.

Author: Auto-generated
Date: 2025-10-03
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Version for alternative typewell JSON format
ALTERNATIVE_TYPEWELL_VERSION = "1.0"


class AlternativeTypewellStorage:
    """
    Manages storage and retrieval of alternative typewell data.

    Stores typewell data in JSON files with versioning and metadata.
    Uses fail-fast approach - raises exceptions on errors instead of silent fallback.
    """

    def __init__(self, storage_dir: str = "alternative_typewells"):
        """
        Initialize storage manager.

        Args:
            storage_dir: Directory for storing alternative typewell JSON files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"AlternativeTypewellStorage initialized with directory: {self.storage_dir}")

    def _get_file_path(self, well_name: str) -> Path:
        """
        Get file path for a given well name.

        Args:
            well_name: Name of the well

        Returns:
            Path to the JSON file
        """
        filename = f"{well_name}_alt_typewell.json"
        return self.storage_dir / filename

    def exists(self, well_name: str) -> bool:
        """
        Check if alternative typewell file exists for a given well.

        Args:
            well_name: Name of the well

        Returns:
            True if file exists, False otherwise
        """
        file_path = self._get_file_path(well_name)
        exists = file_path.exists()
        logger.debug(f"Alternative typewell file for '{well_name}': {'EXISTS' if exists else 'NOT FOUND'}")
        return exists

    def save(self, well_name: str, typewell_data: Dict, interpretation_used: str) -> Path:
        """
        Save alternative typewell data to JSON file.

        Args:
            well_name: Name of the well
            typewell_data: TypeWell data in PAPI format (dict)
            interpretation_used: Name of interpretation used for normalization/replacement

        Returns:
            Path to the saved file

        Raises:
            ValueError: If typewell_data is None or empty
            IOError: If file write fails
        """
        if not typewell_data:
            raise ValueError(f"Cannot save alternative typewell: typewell_data is empty for well '{well_name}'")

        file_path = self._get_file_path(well_name)

        # Create JSON structure with metadata
        json_data = {
            "version": ALTERNATIVE_TYPEWELL_VERSION,
            "created_at": datetime.now().isoformat(),
            "well_name": well_name,
            "interpretation_used": interpretation_used,
            "typewell_data": typewell_data
        }

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)

            logger.info("="*60)
            logger.info("🔄 ALTERNATIVE TYPEWELL CREATED WITH CURVE REPLACEMENT")
            logger.info(f"   Well: {well_name}")
            logger.info(f"   Interpretation used: {interpretation_used}")
            logger.info(f"   File: {file_path}")
            logger.info(f"   Created at: {json_data['created_at']}")
            logger.info("   This typewell contains modified curves from vertical section")
            logger.info("="*60)

            return file_path

        except Exception as e:
            raise IOError(f"Failed to save alternative typewell for '{well_name}': {str(e)}") from e

    def load(self, well_name: str) -> Dict:
        """
        Load alternative typewell data from JSON file.

        Args:
            well_name: Name of the well

        Returns:
            TypeWell data in PAPI format (dict)

        Raises:
            FileNotFoundError: If file doesn't exist (fail-fast)
            ValueError: If version mismatch or invalid JSON structure
            IOError: If file read fails
        """
        file_path = self._get_file_path(well_name)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Alternative typewell file not found for well '{well_name}'\n"
                f"Expected path: {file_path}\n"
                f"Hint: Set CURVE_REPLACEMENT_ENABLED=true to auto-create on first run"
            )

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Validate version
            file_version = json_data.get("version")
            if file_version != ALTERNATIVE_TYPEWELL_VERSION:
                raise ValueError(
                    f"Version mismatch for alternative typewell '{well_name}'\n"
                    f"File version: {file_version}\n"
                    f"Expected version: {ALTERNATIVE_TYPEWELL_VERSION}\n"
                    f"Action required: Delete {file_path} to regenerate"
                )

            # Validate structure
            if "typewell_data" not in json_data:
                raise ValueError(
                    f"Invalid alternative typewell file structure for '{well_name}'\n"
                    f"Missing 'typewell_data' key in {file_path}"
                )

            typewell_data = json_data["typewell_data"]

            logger.info("="*60)
            logger.info("✅ LOADING CACHED ALTERNATIVE TYPEWELL")
            logger.info(f"   Well: {well_name}")
            logger.info(f"   Interpretation used: {json_data.get('interpretation_used', 'N/A')}")
            logger.info(f"   File: {file_path}")
            logger.info(f"   Created at: {json_data.get('created_at', 'N/A')}")
            logger.info("   Using cached typewell with modified curves instead of PAPI typewell")
            logger.info("="*60)

            return typewell_data

        except FileNotFoundError:
            raise  # Re-raise as-is
        except ValueError:
            raise  # Re-raise as-is
        except Exception as e:
            raise IOError(f"Failed to load alternative typewell for '{well_name}': {str(e)}") from e
