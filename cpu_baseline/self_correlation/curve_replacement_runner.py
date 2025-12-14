# self_correlation/curve_replacement_runner.py

"""
Curve Replacement Runner - standalone batch processor for TypeWell curve replacement
Final version with proper AG objects pipeline
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, List
from copy import deepcopy

# AG objects imports
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_interpretation import create_segments_from_json

# Import curve replacement processor
from self_correlation.curve_replacement_processor import CurveReplacementProcessor

# Import curve replacement visualizer
from self_correlation.curve_replacement_visualizer import CurveReplacementVisualizer

# Import landing detection (same as emulator)
import sys

sys.path.append('../')

from landing_detector import LandingDetector

# Import normalization modules
from python_normalization.normalization_calculator import NormalizationCalculator
from python_normalization.normalization_logger import NormalizationLogger

from self_correlation.typewell_to_json_converter import update_well_data_with_modified_typewell


class CurveReplacementRunner:
    """Standalone runner for batch curve replacement processing"""

    def __init__(self, config: Dict[str, Any], logger=None):
        # Use passed logger or create own
        self.logger = logger or logging.getLogger(__name__)

        # Store config for access in other methods
        self.config = config

        self.input_dir = Path(config['input_wells_dir'])
        self.output_dir = Path(config['output_wells_dir'])

        # Create output directory
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Landing detection configuration
        self.landing_detection_enabled = config['landing_detection_enabled']

        if self.landing_detection_enabled:
            self.landing_detector = LandingDetector(
                offset_meters=config['landing_offset_meters'],
                alternative_offset_meters=config['landing_alternative_offset_meters'],
                landing_start_angle_deg=config['landing_start_angle_deg'],
                max_landing_length_meters=config['landing_max_length_meters'],
                perch_stability_threshold=config['landing_perch_stability_threshold'],
                perch_min_angle_deg=config['landing_perch_min_angle_deg'],
                perch_stability_window=config['landing_perch_stability_window']
            )
            self.logger.info("Landing detection enabled")
        else:
            self.landing_detector = None
            self.logger.info("Landing detection disabled")

        # Initialize normalization
        self.normalization_calculator = NormalizationCalculator(interactive_mode=False)
        self.normalization_logger = NormalizationLogger(config.get('results_dir', 'results'))

        # Initialize curve replacement processor with config and logger
        self.curve_replacement_processor = CurveReplacementProcessor(config=config, logger_instance=self.logger)

        # Initialize visualizer if needed
        if config['curve_replacement_save_plots']:
            self.visualizer = CurveReplacementVisualizer(config, logger_instance=self.logger)
        else:
            self.visualizer = None

        self.logger.info(f"CurveReplacementRunner initialized:")
        self.logger.info(f"  Input dir: {self.input_dir}")
        self.logger.info(f"  Output dir: {self.output_dir}")
        self.logger.info(f"  Landing detection enabled: {self.landing_detection_enabled}")
        self.logger.info(f"  Normalization available: True")
        self.logger.info(f"  Visualization enabled: {config['curve_replacement_save_plots']}")

    def run_batch_processing(self):
        """Process all wells in input directory"""
        self.logger.info("Starting batch curve replacement processing")

        # Find all well files
        assert self.input_dir.exists(), f"Input directory not found: {self.input_dir}"

        well_files = list(self.input_dir.glob('*.json'))
        assert well_files, f"No JSON files found in {self.input_dir}"

        # Apply processing limit if configured
        max_wells = self.config['max_wells_to_process']
        if max_wells > 0:
            original_count = len(well_files)
            well_files = well_files[:max_wells]
            self.logger.info(
                f"Processing limit applied: {original_count} wells found, processing first {len(well_files)}")
        else:
            self.logger.info(f"No processing limit - processing all {len(well_files)} wells")

        self.logger.info(f"Found {len(well_files)} well files to process")

        processed_count = 0
        success_count = 0

        for i, well_file in enumerate(well_files):
            self.logger.info(f"[{i + 1}/{len(well_files)}] Processing: {well_file.name}")

            success = self.process_single_well(well_file)
            processed_count += 1
            if success:
                success_count += 1

        self.logger.info(f"Batch processing completed: {success_count}/{processed_count} wells successful")

    def process_single_well(self, well_file: Path) -> bool:
        """
        Process single well file with full AG objects pipeline

        Returns:
            True if processing was successful
        """
        # Load well data
        well_data = self.load_well_data(well_file)
        well_name = well_data['wellName']

        self.logger.info(f"Processing well: {well_name}")

        # Get original start MD
        original_start_md = self.get_start_md(well_data)
        if original_start_md is None:
            self.logger.warning(f"Cannot determine start MD for {well_name}, skipping")
            return False

        # Apply landing detection if enabled
        start_md = original_start_md
        detected_start_md = original_start_md
        if self.landing_detection_enabled and self.landing_detector:
            self.logger.info(f"Applying landing detection for {well_name}")
            detected_start_md = self.landing_detector.detect_optimal_start(well_data)
            # Use the rightmost (maximum) MD to ensure we don't go "left"
            start_md = max(original_start_md, detected_start_md)

            if detected_start_md > original_start_md:
                self.logger.info(
                    f"Landing detector adjusted start MD: {original_start_md:.1f}m -> {detected_start_md:.1f}m")
            else:
                self.logger.info(
                    f"Landing detector: keeping original start MD {original_start_md:.1f}m (detected: {detected_start_md:.1f}m)")

        self.logger.info(f"MD Analysis Range:")
        self.logger.info(f"  Original Start MD: {original_start_md:.1f}m")
        self.logger.info(f"  Detected Start MD: {detected_start_md:.1f}m")
        self.logger.info(f"  Final Start MD: {start_md:.1f}m")
        self.logger.info(f"  Range Size: {abs(detected_start_md - original_start_md):.1f}m")

        # ========== STEP 1: Create AG Objects ==========
        self.logger.info("📊 Creating AG objects from JSON...")

        # Create Well object (cleans data automatically)
        well = Well(well_data)
        self.logger.info(f"  Well created: {len(well.measured_depth)} points after cleaning")

        # Create TypeWell object (cleans data automatically)
        typewell = TypeWell(well_data)
        self.logger.info(f"  TypeWell created: {len(typewell.tvd)} points after cleaning")

        # Create manual interpretation segments
        if 'interpretation' not in well_data or not well_data['interpretation']['segments']:
            self.logger.warning(f"No manual interpretation found for {well_name}, skipping")
            return False

        manual_segments = create_segments_from_json(
            well_data['interpretation']['segments'],
            well,
            well.measured_depth[-1]
        )
        self.logger.info(f"  Manual segments created: {len(manual_segments)} segments")

        # ========== STEP 2: Apply Normalization (modifies well.value) ==========
        self.logger.info("📐 Applying normalization...")
        landing_end_md = max(original_start_md, detected_start_md)

        normalization_result = self.normalization_calculator.calculate_normalization_coefficients(
            well_data, well, typewell, manual_segments, landing_end_md
        )

        # Log normalization result
        self.normalization_logger.log_normalization_result(normalization_result)

        if normalization_result['status'] == 'success':
            # well.value already modified in place by normalization
            self.logger.info(f"  Normalization applied: multiplier={normalization_result['multiplier']:.6f}")
        else:
            self.logger.warning(f"  Normalization failed: {normalization_result.get('issue_description', 'Unknown')}")

        # ========== STEP 3: Create backup of TypeWell for visualization ==========
        self.logger.info("💾 Creating TypeWell backup for visualization...")
        original_typewell = deepcopy(typewell)
        self.logger.info(f"  TypeWell backup created")

        # ========== STEP 4: Apply Curve Replacement (modifies typewell.value) ==========
        self.logger.info("🔄 Applying curve replacement...")

        replacement_result = self.curve_replacement_processor.process_well_data(
            well_data, well, typewell, manual_segments,
            original_start_md, detected_start_md
        )

        if replacement_result['status'] == 'success':
            self.logger.info(f"  ✅ Curve replacement successful:")
            self.logger.info(f"     Points replaced: {replacement_result.get('points_replaced', 0)}")
            self.logger.info(f"     Points extended: {replacement_result.get('points_extended', 0)}")
        else:
            self.logger.warning(f"  ⚠️ Curve replacement failed: {replacement_result.get('reason', 'Unknown')}")

        # ========== STEP 5: Generate Visualization ==========
        if self.visualizer and replacement_result['status'] == 'success':
            self.logger.info("🎨 Generating visualization...")

            # Prepare full replacement info for visualization
            viz_info = {
                **replacement_result,  # All info from processor
                'original_start_md': original_start_md,
                'detected_start_md': detected_start_md,
                'blend_weight': self.config['curve_replacement_blend_weight'],
                'replacement_range_meters': self.config['curve_replacement_range_meters']
            }

            plot_path = self.visualizer.create_replacement_plot(
                well_name, well, original_typewell, typewell, viz_info
            )

            if plot_path:
                self.logger.info(f"  ✅ Visualization saved: {plot_path}")
            else:
                self.logger.warning("  ⚠️ Visualization failed")

        # ========== STEP 6: Convert modified objects back to JSON ==========
        self.logger.info("📝 Converting modified objects back to JSON...")

        # TODO: This is where we need the conversion function
        # For now, we'll save the original JSON with a note
        modified_well_data = deepcopy(well_data)

        # Add metadata about processing
        modified_well_data['_processing_metadata'] = {
            'curve_replacement_applied': replacement_result['status'] == 'success',
            'normalization_applied': True,
            'original_start_md': original_start_md,
            'detected_start_md': detected_start_md,
            'processing_timestamp': time.time()
        }

        if replacement_result['status'] == 'success':
            # Обновляем JSON с модифицированным TypeWell
            modified_well_data = update_well_data_with_modified_typewell(
                well_data,
                typewell  # это уже модифицированный объект
            )

        # ========== STEP 7: Save modified well data ==========
        output_file = self.output_dir / well_file.name
        self.save_well_data(modified_well_data, output_file)

        # Save processing log
        self.save_processing_log(well_name, original_start_md, detected_start_md,
                                 well_file, output_file, replacement_result)

        self.logger.info(f"✅ Completed processing: {well_name}")
        return replacement_result['status'] == 'success'

    def load_well_data(self, file_path: Path) -> Dict[str, Any]:
        """Load well data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            well_data = json.load(f)

        # Basic validation
        assert 'wellName' in well_data, "Missing wellName in well data"
        assert 'well' in well_data, "Missing well trajectory data"
        assert 'wellLog' in well_data, "Missing wellLog data"
        assert 'typeLog' in well_data, "Missing typeLog data"

        return well_data

    def get_start_md(self, well_data: Dict[str, Any]) -> float:
        """Get start MD from well data"""
        if 'autoGeosteeringParameters' in well_data:
            start_md = well_data['autoGeosteeringParameters']['startMd']
            if start_md is not None:
                return float(start_md)

        self.logger.warning("autoGeosteeringParameters.startMd not found")
        return None

    def save_well_data(self, well_data: Dict[str, Any], output_file: Path):
        """Save modified well data to JSON file"""
        # Remove internal metadata before saving (if needed)
        if '_curve_replacement_info' in well_data:
            del well_data['_curve_replacement_info']
        if '_replacement_stats' in well_data:
            del well_data['_replacement_stats']

        with open(output_file, 'w') as f:
            json.dump(well_data, f, indent=2)

    def save_processing_log(self,
                            well_name: str,
                            original_start_md: float,
                            detected_start_md: float,
                            input_file: Path,
                            output_file: Path,
                            replacement_result: Dict[str, Any]):
        """Save processing log for tracking"""
        self.logger.info(f"Processing summary for {well_name}:")
        self.logger.info(f"  Original Start MD: {original_start_md:.1f}m")
        self.logger.info(f"  Detected Start MD: {detected_start_md:.1f}m")
        self.logger.info(f"  Input: {input_file.name}")
        self.logger.info(f"  Output: {output_file.name}")
        self.logger.info(f"  Replacement status: {replacement_result['status']}")
        if replacement_result['status'] == 'success':
            self.logger.info(f"    Points replaced: {replacement_result.get('points_replaced', 0)}")
            self.logger.info(f"    Points extended: {replacement_result.get('points_extended', 0)}")


def load_configuration() -> Dict[str, Any]:
    """Load configuration from environment"""
    config = {
        'input_wells_dir': os.getenv('WELLS_DIR'),
        'output_wells_dir': os.getenv('WELLS_MODIFIED_DIR'),
        'results_dir': os.getenv('RESULTS_DIR', 'results'),
        'log_level': os.getenv('LOG_LEVEL', 'INFO'),

        # Curve replacement settings
        'curve_replacement_enabled': os.getenv('CURVE_REPLACEMENT_ENABLED', 'false').lower() == 'true',
        'curve_replacement_range_meters': float(os.getenv('CURVE_REPLACEMENT_RANGE_METERS', '500.0')),
        'curve_replacement_blend_weight': float(os.getenv('CURVE_REPLACEMENT_BLEND_WEIGHT', '1.0')),
        'curve_replacement_search_distance_meters': float(os.getenv('CURVE_REPLACEMENT_SEARCH_DISTANCE_METERS', '3.0')),
        'curve_replacement_match_tolerance': float(os.getenv('CURVE_REPLACEMENT_MATCH_TOLERANCE', '0.01')),

        # Visualization settings
        'curve_replacement_save_plots': os.getenv('CURVE_REPLACEMENT_SAVE_PLOTS', 'false').lower() == 'true',
        'curve_replacement_plots_dir': os.getenv('CURVE_REPLACEMENT_PLOTS_DIR', 'self_correlation_plots'),

        # Landing detection configuration
        'landing_detection_enabled': os.getenv('LANDING_DETECTION_ENABLED', 'false').lower() == 'true',
        'landing_offset_meters': float(os.getenv('LANDING_OFFSET_METERS', '100.0')),
        'landing_alternative_offset_meters': float(os.getenv('LANDING_ALTERNATIVE_OFFSET_METERS', '50.0')),
        'landing_start_angle_deg': float(os.getenv('LANDING_START_ANGLE_DEG', '60.0')),
        'landing_max_length_meters': float(os.getenv('LANDING_MAX_LENGTH_METERS', '200.0')),
        'landing_perch_stability_threshold': float(os.getenv('LANDING_PERCH_STABILITY_THRESHOLD', '0.5')),
        'landing_perch_min_angle_deg': float(os.getenv('LANDING_PERCH_MIN_ANGLE_DEG', '30.0')),
        'landing_perch_stability_window': int(os.getenv('LANDING_PERCH_STABILITY_WINDOW', '5')),

        # Processing limit
        'max_wells_to_process': int(os.getenv('CURVE_REPLACEMENT_MAX_WELLS', '3'))
    }

    return config


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('curve_replacement.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main entry point for standalone curve replacement"""
    # Load environment from root
    from dotenv import load_dotenv
    load_dotenv('.env')

    config = load_configuration()
    setup_logging(config['log_level'])

    logger = logging.getLogger(__name__)

    logger.info("=== Curve Replacement Runner ===")
    logger.info(f"  Enabled: {config['curve_replacement_enabled']}")
    logger.info(f"  Input: {config['input_wells_dir']}")
    logger.info(f"  Output: {config['output_wells_dir']}")
    logger.info("================================")

    # Validate required directories
    input_dir = Path(config['input_wells_dir'])
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1

    # Create and run processor
    runner = CurveReplacementRunner(config, logger=logger)
    runner.run_batch_processing()

    logger.info("Curve replacement runner completed")
    return 0


if __name__ == "__main__":
    import sys
    import time

    sys.exit(main())