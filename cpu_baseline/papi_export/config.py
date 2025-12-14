import os
import json
from pathlib import Path
from dotenv import load_dotenv
import logging
from typing import List

logger = logging.getLogger(__name__)

# Load .env from project root
root_dir = Path(__file__).parent.parent
env_path = root_dir / '.env'
load_dotenv(env_path)


class PAPIConfig:
    """Configuration for PAPI export functionality - reads well settings from JSON, .env for other params"""
    
    def __init__(self, wells_config_file: str = None, well_name: str = None, config_dict: dict = None):
        # Load configuration from file or dict
        if config_dict:
            # Direct dict configuration (for nightguard)
            config_data = config_dict
            # Set defaults for global settings
            self.output_format = 'json'
            self.verbose = False
            self.save_intermediate = False
        elif wells_config_file:
            # Load from JSON file (existing logic)
            config_data = self._load_config_from_file(wells_config_file)
            # Global settings from JSON
            self.output_format = config_data['output_format']
            self.verbose = config_data['log_level'].upper() == 'DEBUG'
            self.save_intermediate = config_data['keep_debug_files']
        else:
            raise ValueError("Either config_dict or wells_config_file must be provided")

        # Common configuration loading
        self._load_from_config_data(config_data, well_name)

        # Load remaining parameters from .env
        self._load_env_parameters()

    def _load_config_from_file(self, wells_config_file: str) -> dict:
        """Load configuration data from JSON file"""
        wells_config_path = Path(wells_config_file)
        if not wells_config_path.exists():
            raise FileNotFoundError(f"Wells config file not found: {wells_config_file}")

        with open(wells_config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_from_config_data(self, config_data: dict, well_name: str):
        """Load configuration from config data (from file or dict)"""
        if 'wells' in config_data:
            # File format: find well in wells array
            well_config = None
            for well in config_data['wells']:
                if well['well_name'] == well_name:
                    well_config = well
                    break

            if not well_config:
                raise ValueError(f"Well '{well_name}' not found in wells config")

            # Set from JSON file format
            self.project_name = config_data['project_name']
            self.well_name = well_config['well_name']
            self.lateral_log_name = well_config['lateral_log_name']
            self.typewell_log_name = well_config['typewell_log_name']
            self.typewell_name = well_config.get('typewell_name') or os.getenv('TYPEWELL_NAME')
            self.horizon_name = well_config['horizon_name']
            self.grid_name = well_config['grid_name']
            self.enable_log_normalization = well_config['enable_log_normalization']

            # Source interpretation name - OPTIONAL (None = use starred)
            self.source_interpretation_name = well_config.get('source_interpretation_name', None)
        else:
            # Dict format: direct parameters for single well
            self.project_name = config_data['project_name']
            self.well_name = well_name
            self.lateral_log_name = config_data['lateral_log_name']
            self.typewell_log_name = config_data['typewell_log_name']
            self.typewell_name = config_data.get('typewell_name') or os.getenv('TYPEWELL_NAME')
            self.horizon_name = config_data['horizon_name']
            self.grid_name = config_data['grid_name']
            self.enable_log_normalization = config_data.get('enable_log_normalization', True)

            # Source interpretation name - OPTIONAL (None = use starred)
            self.source_interpretation_name = config_data.get('source_interpretation_name', None)
        
    def _load_env_parameters(self):
        """Load parameters that come from .env file"""
        # Processing Parameters - REQUIRED, horizon_spacing must be INTEGER for PAPI
        spacing_str = self._get_required_env('SOLO_HORIZON_SPACING')
        assert spacing_str.isdigit(), f"SOLO_HORIZON_SPACING must be integer, got: '{spacing_str}'"
        self.horizon_spacing = int(spacing_str)
        assert 1 <= self.horizon_spacing <= 100000, f"SOLO_HORIZON_SPACING must be 1-100000, got: {self.horizon_spacing}"
        
        tolerance_str = self._get_required_env('SOLO_SEGMENT_TOLERANCE')
        self.segment_tolerance = float(tolerance_str)
        
        grid_step_str = self._get_required_env('SOLO_GRID_STEP_SIZE')
        self.grid_step_size = float(grid_step_str)
        
        # AutoGeosteering Parameters - REQUIRED
        self.lookback_distance = float(self._get_required_env('LOOKBACK_DISTANCE'))
        self.dip_angle_range_degree = float(self._get_required_env('DIP_ANGLE_RANGE_DEGREE'))
        self.dip_step_degree = float(self._get_required_env('DIP_STEP_DEGREE'))
        self.regional_dip_angle = float(self._get_required_env('REGIONAL_DIP_ANGLE'))
        self.smoothness = float(self._get_required_env('SMOOTHNESS'))

        norm_str = self._get_required_env('ENABLE_LOG_NORMALIZATION')
        assert norm_str.lower() in ['true', 'false'], f"ENABLE_LOG_NORMALIZATION must be 'true' or 'false', got: '{norm_str}'"
        self.enable_log_normalization = norm_str.lower() == 'true'

        vertical_str = self._get_required_env('VERTICAL_TRACK_ONLY')
        assert vertical_str.lower() in ['true', 'false'], f"VERTICAL_TRACK_ONLY must be 'true' or 'false', got: '{vertical_str}'"
        self.vertical_track_only = vertical_str.lower() == 'true'

        # Output directory - WELLS_DIR required only for emulator mode, optional for nightguard
        # Check EMULATOR_MODE to determine if WELLS_DIR is required
        emulator_mode = os.getenv('EMULATOR_MODE', 'emulator').lower()
        wells_dir = os.getenv('WELLS_DIR')

        if emulator_mode == 'emulator' and not wells_dir:
            # Required for emulator mode
            self._print_required_env_vars()
            raise ValueError("WELLS_DIR is required when EMULATOR_MODE=emulator")
        elif wells_dir:
            # Use WELLS_DIR if provided
            self.output_dir = Path(wells_dir)
        else:
            # Nightguard mode without WELLS_DIR - use STARSTEER_DIR + WELLS_DATA_SUBDIR
            starsteer_dir = os.getenv('STARSTEER_DIR')
            wells_subdir = os.getenv('WELLS_DATA_SUBDIR', 'AG_DATA/InitialData')
            if starsteer_dir:
                self.output_dir = Path(starsteer_dir) / wells_subdir
            else:
                # Fallback to results directory
                self.output_dir = Path('./results')

        # Alternative TVT calculation flag
        use_alt_tvt_str = os.getenv('USE_ALTERNATIVE_TVT_CALCULATION', 'false')
        assert use_alt_tvt_str.lower() in ['true', 'false'], f"USE_ALTERNATIVE_TVT_CALCULATION must be 'true' or 'false', got: '{use_alt_tvt_str}'"
        self.use_alternative_tvt_calculation = use_alt_tvt_str.lower() == 'true'

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if self.save_intermediate:
            (self.output_dir / 'intermediate').mkdir(exist_ok=True)
            
    def _get_required_env(self, var_name: str) -> str:
        """Get required environment variable or fail with helpful message"""
        value = os.getenv(var_name)
        if value is None:
            self._print_required_env_vars()
            raise ValueError(f"Required environment variable '{var_name}' not found in .env file")
        return value
        
    def _print_required_env_vars(self):
        """Print all required environment variables"""
        required_vars = [
            "# Authentication (already in .env)",
            "SOLO_SDK_CLIENT_ID=your_client_id",
            "SOLO_SDK_CLIENT_SECRET=your_client_secret", 
            "SOLO_DOMAIN=https://solo.cloud",
            "",
            "# Project and data selection (already in .env)",
            "SOLO_PROJECT_NAME=your_project_name",
            "SOLO_WELL_NAME=your_well_name",
            "",
            "# Log names to extract (already in .env)",
            "SOLO_LATERAL_LOG_NAME=GR",
            "SOLO_TYPEWELL_LOG_NAME=Hyde 1PH GR",
            "",
            "# Horizon and grid names (already in .env)",
            "SOLO_HORIZON_NAME=EGFDL",
            "SOLO_GRID_NAME=Grid_001",
            "",
            "# Processing parameters (already in .env)",
            "SOLO_HORIZON_SPACING=1",
            "SOLO_SEGMENT_TOLERANCE=0.01",
            "SOLO_GRID_STEP_SIZE=20.0",
            "",
            "# AutoGeosteering parameters (already in .env)",
            "LOOKBACK_DISTANCE=100.0",
            "REGIONAL_DIP_ANGLE=0",
            "ENABLE_LOG_NORMALIZATION=true",
            "VERTICAL_TRACK_ONLY=false",
            "",
            "# Output configuration (already in .env)",
            "SOLO_OUTPUT_DIR=./papi_export_results",
            "SOLO_OUTPUT_FORMAT=json",
            "",
            "# Debug settings (already in .env)",
            "LOG_LEVEL=INFO",
            "KEEP_DEBUG_FILES=false"
        ]
        
        print("\n" + "="*60)
        print("MISSING REQUIRED ENVIRONMENT VARIABLES!")
        print("="*60)
        print("Please check the following variables in your .env file:")
        print("")
        for var in required_vars:
            print(var)
        print("="*60)
            
    def validate(self):
        """Validate configuration - all validation now happens in __init__"""
        logger.info(f"Configuration loaded successfully:")
        logger.info(f"  Project: {self.project_name}")
        logger.info(f"  Well: {self.well_name}")
        logger.info(f"  Horizon: {self.horizon_name} (spacing: {self.horizon_spacing})")
        logger.info(f"  Grid: {self.grid_name}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Debug mode: {self.verbose}")
        logger.info(f"  Save intermediate: {self.save_intermediate}")
        
    def to_dict(self):
        """Convert config to dictionary for passing to other modules"""
        return {
            'project_name': self.project_name,
            'well_name': self.well_name,
            'lateral_log_name': self.lateral_log_name,
            'typewell_log_name': self.typewell_log_name,
            'horizon_name': self.horizon_name,
            'grid_name': self.grid_name,
            'horizon_spacing': self.horizon_spacing,
            'segment_tolerance': self.segment_tolerance,
            'lookback_distance': self.lookback_distance,
            'regional_dip_angle': self.regional_dip_angle,
            'enable_log_normalization': self.enable_log_normalization,
            'vertical_track_only': self.vertical_track_only,
            'output_dir': str(self.output_dir),
            'verbose': self.verbose
        }
    
    @staticmethod
    def load_wells_list(wells_config_file: str) -> List[str]:
        """Load list of well names from JSON config file, filtering only wells with 'process': true"""
        wells_config_path = Path(wells_config_file)
        if not wells_config_path.exists():
            raise FileNotFoundError(f"Wells config file not found: {wells_config_file}")
            
        with open(wells_config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # Filter wells with 'process': true
        all_wells = config_data['wells']
        processed_wells = [well for well in all_wells if well.get('process', False) is True]
        
        logger.info(f"📋 Found {len(all_wells)} total wells, {len(processed_wells)} marked for processing")
        if len(processed_wells) == 0:
            logger.warning("⚠️  No wells marked with 'process': true found!")
        else:
            well_names = [well['well_name'] for well in processed_wells]
            logger.info(f"🎯 Wells to process: {', '.join(well_names)}")
            
        return [well['well_name'] for well in processed_wells]