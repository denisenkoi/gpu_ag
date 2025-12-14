#!/usr/bin/env python3
"""
AutoGeosteering Drilling Emulator - Main Entry Point
Thin orchestrator that delegates work to DrillingEmulator
"""

import os
import sys
import logging
import time
from pathlib import Path
from dotenv import load_dotenv

from emulator import DrillingEmulator
# from quality_checker import InterpretationQualityChecker


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('emulator.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_configuration() -> dict:
    """Load configuration from environment"""
    # Load .env file
    env_file = '.env'
    if not os.path.exists(env_file):
        raise FileNotFoundError(f"Configuration file {env_file} not found")

    load_dotenv(env_file)
    logging.info(f"Loaded configuration from: {env_file}")

    # Extract configuration
    config = {
        'exe_path': os.getenv('EXE_PATH'),
        'wells_dir': os.getenv('WELLS_DIR'),
        'work_dir': os.getenv('WORK_DIR'),
        'results_dir': os.getenv('RESULTS_DIR'),
        'md_step_meters': float(os.getenv('MD_STEP_METERS')),
        'quality_thresholds_meters': os.getenv('QUALITY_THRESHOLDS_METERS', '3.0,4.0'),
        'daemon_timeout_seconds': int(os.getenv('DAEMON_TIMEOUT_SECONDS')),
        'log_level': os.getenv('LOG_LEVEL'),

        # Continue mode configuration
        'continue_mode': os.getenv('CONTINUE_MODE', 'false').lower() == 'true',

        # Landing detection configuration
        'landing_detection_enabled': os.getenv('LANDING_DETECTION_ENABLED', 'false').lower() == 'true',
        'landing_offset_meters': float(os.getenv('LANDING_OFFSET_METERS', '100.0')),
        'landing_alternative_offset_meters': float(os.getenv('LANDING_ALTERNATIVE_OFFSET_METERS', '50.0')),
        'landing_start_angle_deg': float(os.getenv('LANDING_START_ANGLE_DEG', '60.0')),
        'landing_max_length_meters': float(os.getenv('LANDING_MAX_LENGTH_METERS', '200.0')),
        'landing_perch_stability_threshold': float(os.getenv('LANDING_PERCH_STABILITY_THRESHOLD', '0.5')),
        'landing_perch_min_angle_deg': float(os.getenv('LANDING_PERCH_MIN_ANGLE_DEG', '30.0')),
        'landing_perch_stability_window': int(os.getenv('LANDING_PERCH_STABILITY_WINDOW', '5')),

        # StarSteer directory for interpretation export
        'starsteer_dir': os.getenv('STARSTEER_DIR', ''),
    }

    return config


def validate_environment(config: dict):
    """Validate that all required files and directories exist"""
    logger = logging.getLogger(__name__)

    # Log configuration
    logger.info("=== AutoGeosteering Drilling Emulator Configuration ===")
    for key, value in config.items():
        logger.info(f"  {key.upper()}: {value}")
    logger.info("=======================================================")

    # Check executable
    exe_path = Path(config['exe_path'])
    if not exe_path.exists():
        raise FileNotFoundError(f"AutoGeosteering executable not found: {exe_path}")

    # Wells directory validation - only for emulator mode
    # In nightguard mode, wells_dir is None (data path built from STARSTEER_DIR)
    mode = os.getenv('EMULATOR_MODE', 'emulator').lower()

    if mode == 'emulator' and config.get('wells_dir'):
        # Wait for wells directory to exist
        wells_dir = Path(config['wells_dir'])
        if not wells_dir.exists():
            logger.info(f"Wells directory not found: {wells_dir}")
            logger.info("Waiting for PAPI Loader to create directory and files...")

        while not wells_dir.exists():
            logger.info("⏳ Waiting for WELLS_DIR to be created...")
            time.sleep(5)

        logger.info(f"✅ Wells directory found: {wells_dir}")

        # Wait for well files to appear
        well_files = list(wells_dir.glob('*.json'))
        if not well_files:
            logger.info("📂 Wells directory exists but no JSON files found yet...")
            logger.info("Waiting for PAPI Loader to generate well files...")

        while not well_files:
            logger.info("⏳ Waiting for well files to appear...")
            time.sleep(5)
            well_files = list(wells_dir.glob('*.json'))

        logger.info(f"✅ Found {len(well_files)} well files ready for processing")
    else:
        logger.info("⏩ Skipping wells_dir validation (nightguard mode uses STARSTEER_DIR)")

    # Create work and results directories
    Path(config['work_dir']).mkdir(exist_ok=True)
    Path(config['results_dir']).mkdir(exist_ok=True)

    logger.info("Environment validation passed:")
    logger.info(f"  - Executable: {exe_path.name}")
    if mode == 'emulator' and config.get('wells_dir'):
        logger.info(f"  - Wells directory: {wells_dir.name}")
        logger.info(f"  - Found {len(well_files)} well files")
    else:
        logger.info(f"  - Mode: nightguard (wells loaded from STARSTEER_DIR)")


def main():
    """Main entry point for the AutoGeosteering Drilling Emulator"""
    # Load configuration
    config = load_configuration()
    
    # Setup logging
    setup_logging(config['log_level'])
    logger = logging.getLogger(__name__)
    
    # Validate environment
    validate_environment(config)
    
    # Create and run emulator
    logger.info("Starting DrillingEmulator...")
    emulator = DrillingEmulator(config)
    
    # Run continuous processing (unified method for both modes)
    emulator.run_continuous_processing()
    
    logger.info("Emulation completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
