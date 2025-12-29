"""
JSON to PyTorch Dataset Converter (RND-809)

Orchestrator for converting AG_DATA/InitialData/*.json files to a single PyTorch dataset.

Usage:
    python json_to_torch.py [--source PATH] [--output PATH] [--device DEVICE]

Output format:
    {
        'Well1002_landing~EGFDL': {
            # RAW well trajectory (from well.points) - NO Well class processing
            'well_md': tensor,      # (N,) trajectory MD
            'well_tvd': tensor,     # (N,) trajectory TVD
            'well_ns': tensor,      # (N,) trajectory NorthSouth
            'well_ew': tensor,      # (N,) trajectory EastWest

            # RAW wellLog (from wellLog.points) - NO interpolation
            'log_md': tensor,       # (L,) log measured depth
            'log_gr': tensor,       # (L,) gamma ray values

            # Stitched typewell
            'typewell_tvd': tensor, # (M,) stitched pseudo+type TVD
            'typewell_gr': tensor,  # (M,) stitched pseudo+type GR (normalized with 1/multiplier)

            'ref_shifts': tensor,   # (K,) reference interpretation end_shifts
            'ref_segment_mds': tensor,  # (K,) segment start MDs
            'norm_multiplier': float,   # normalization multiplier
            'perch_md': float,          # landing end point
        },
        ...
    }
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
import logging

# Add gpu_ag and cpu_baseline to path for AG modules
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "cpu_baseline"))

from ag_objects.ag_obj_typewell import TypeWell
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_interpretation import create_segments_from_json
from numpy_funcs.converters import typewell_to_numpy
from landing_detector import LandingDetector
from python_normalization.normalization_calculator import NormalizationCalculator
from typewell_provider import extend_pseudo_with_typelog

from dotenv import load_dotenv
import os

# Load .env from cpu_baseline
load_dotenv(Path(__file__).parent.parent / "cpu_baseline" / ".env")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_landing_detector() -> LandingDetector:
    """Create LandingDetector with params from .env"""
    return LandingDetector(
        offset_meters=float(os.getenv('LANDING_OFFSET_METERS', '100.0')),
        alternative_offset_meters=float(os.getenv('LANDING_ALTERNATIVE_OFFSET_METERS', '50.0')),
        landing_start_angle_deg=float(os.getenv('LANDING_START_ANGLE_DEG', '60.0')),
        max_landing_length_meters=float(os.getenv('LANDING_MAX_LENGTH_METERS', '200.0')),
        perch_stability_threshold=float(os.getenv('LANDING_PERCH_STABILITY_THRESHOLD', '0.5')),
        perch_min_angle_deg=float(os.getenv('LANDING_PERCH_MIN_ANGLE_DEG', '30.0')),
        perch_stability_window=int(os.getenv('LANDING_PERCH_STABILITY_WINDOW', '5'))
    )


def calc_landing_and_normalization(json_data: dict, landing_detector: LandingDetector,
                                    norm_calculator: NormalizationCalculator) -> dict:
    """
    Calculate landing point and normalization coefficients.

    Returns:
        dict with perch_md, norm_multiplier, norm_shift
    """
    result = {
        'perch_md': None,
        'detected_start_md': None,
        'norm_multiplier': 1.0,
        'norm_shift': 0.0,
        'norm_status': 'skipped'
    }

    try:
        # Landing detection
        detected_start_md, perch_md = landing_detector.detect_optimal_start(json_data)
        result['detected_start_md'] = detected_start_md
        result['perch_md'] = perch_md

        # Normalization calculation
        well = Well(json_data)
        typewell = TypeWell(json_data)

        # Get manual interpretation segments
        interp_segments = json_data.get('interpretation', {}).get('segments', [])
        if not interp_segments:
            result['norm_status'] = 'no_interpretation'
            return result

        manual_segments = create_segments_from_json(
            interp_segments,
            well,
            well.measured_depth[-1]
        )

        # Calculate normalization
        norm_result = norm_calculator.calculate_normalization_coefficients(
            well_data=json_data,
            well=well,
            typewell=typewell,
            manual_segments=manual_segments,
            landing_end_md=perch_md
        )

        if norm_result['status'] == 'success':
            result['norm_multiplier'] = norm_result['multiplier']
            result['norm_shift'] = norm_result['shift']
            result['norm_status'] = 'success'
        else:
            result['norm_status'] = norm_result.get('issue_description', 'failed')

    except Exception as e:
        logger.warning(f"Landing/normalization failed: {e}")
        result['norm_status'] = f'error: {e}'

    return result


def extract_ref_shifts(json_data: dict) -> tuple:
    """
    Extract reference interpretation shifts from starredInterpretation.

    Returns:
        (start_mds, end_shifts) - arrays of segment start MDs and end shifts
    """
    starred = json_data.get('starredInterpretation', {})
    segments = starred.get('segments', [])

    if not segments:
        return np.array([]), np.array([])

    start_mds = np.array([s['startMd'] for s in segments], dtype=np.float64)
    end_shifts = np.array([s['endShift'] for s in segments], dtype=np.float64)

    return start_mds, end_shifts


def extract_trajectory(json_data: dict) -> tuple:
    """Extract NS/EW from well trajectory"""
    well_points = json_data.get('well', {}).get('points', [])
    if not well_points:
        return np.array([]), np.array([]), np.array([])

    md = np.array([p['measuredDepth'] for p in well_points], dtype=np.float64)
    ns = np.array([p['northSouth'] for p in well_points], dtype=np.float64)
    ew = np.array([p['eastWest'] for p in well_points], dtype=np.float64)
    return md, ns, ew


def process_single_json(json_path: Path, device: str = 'cpu',
                        landing_detector: LandingDetector = None,
                        norm_calculator: NormalizationCalculator = None) -> Optional[Dict[str, Any]]:
    """
    Process a single JSON file and convert to torch tensors.

    Returns:
        dict with torch tensors or None if processing fails
    """
    try:
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        # Calculate landing and normalization if detectors provided
        landing_norm = None
        if landing_detector and norm_calculator:
            landing_norm = calc_landing_and_normalization(
                json_data, landing_detector, norm_calculator
            )

        # Extract RAW well data directly from JSON (no Well class processing)
        # Well class will be used only in gpu_executor
        well_points = json_data.get('well', {}).get('points', [])
        welllog_points = json_data.get('wellLog', {}).get('points', [])

        # Extract raw arrays from well points
        raw_well_md = np.array([p['measuredDepth'] for p in well_points])
        raw_well_tvd = np.array([p['trueVerticalDepth'] for p in well_points])
        raw_well_ns = np.array([p.get('northSouth', 0.0) for p in well_points])
        raw_well_ew = np.array([p.get('eastWest', 0.0) for p in well_points])

        # Extract raw arrays from wellLog points (filter None values)
        raw_log_md = np.array([p['measuredDepth'] for p in welllog_points if p.get('data') is not None])
        raw_log_gr = np.array([p['data'] for p in welllog_points if p.get('data') is not None])

        # Get norm_multiplier for stitching (if available)
        norm_multiplier = 1.0
        if landing_norm and landing_norm.get('norm_multiplier'):
            norm_multiplier = landing_norm['norm_multiplier']

        # Stitch pseudoTypeLog + typeLog with norm_coef = 1/multiplier
        pseudo_typelog = json_data.get('pseudoTypeLog', None)
        type_log = json_data.get('typeLog', None)

        if pseudo_typelog and type_log and norm_multiplier != 0:
            norm_coef = 1.0 / norm_multiplier
            stitched_typelog = extend_pseudo_with_typelog(pseudo_typelog, type_log, norm_coef)
        elif pseudo_typelog:
            stitched_typelog = pseudo_typelog
        else:
            stitched_typelog = type_log

        # Create TypeWell from stitched data
        typewell = TypeWell({'typeLog': stitched_typelog})
        typewell_np = typewell_to_numpy(typewell)

        # Extract reference interpretation
        ref_mds, ref_shifts = extract_ref_shifts(json_data)

        # Extract trajectory for future analysis
        traj_md, traj_ns, traj_ew = extract_trajectory(json_data)

        # Get params from autoGeosteeringParameters
        ag_params = json_data.get('autoGeosteeringParameters', {})
        start_md = ag_params.get('startMd', 0.0)
        is_log_normalization_enabled = ag_params.get('isLogNormalizationEnabled', True)

        # Get lateral well max MD
        lateral_well_last_md = json_data.get('lateralWellLastMD', 0.0)

        # Get tvdTypewellShift
        tvd_typewell_shift = json_data.get('tvdTypewellShift', 0.0)

        # Convert to torch tensors
        result = {
            # RAW well trajectory data (from well.points)
            'well_md': torch.tensor(raw_well_md, dtype=torch.float64, device=device),
            'well_tvd': torch.tensor(raw_well_tvd, dtype=torch.float64, device=device),
            'well_ns': torch.tensor(raw_well_ns, dtype=torch.float64, device=device),
            'well_ew': torch.tensor(raw_well_ew, dtype=torch.float64, device=device),

            # RAW wellLog data (from wellLog.points)
            'log_md': torch.tensor(raw_log_md, dtype=torch.float64, device=device),
            'log_gr': torch.tensor(raw_log_gr, dtype=torch.float64, device=device),

            # Stitched typewell (pseudo + type with norm_coef = 1/multiplier)
            'typewell_tvd': torch.tensor(typewell_np['tvd'], dtype=torch.float64, device=device),
            'typewell_gr': torch.tensor(typewell_np['value'], dtype=torch.float64, device=device),

            # Reference interpretation
            'ref_segment_mds': torch.tensor(ref_mds, dtype=torch.float64, device=device),
            'ref_shifts': torch.tensor(ref_shifts, dtype=torch.float64, device=device),

            # Metadata
            'start_md': start_md,
            'lateral_well_last_md': lateral_well_last_md,
            'is_log_normalization_enabled': is_log_normalization_enabled,
            'tvd_typewell_shift': tvd_typewell_shift,
            'typewell_step': typewell_np['typewell_step'],
            'typewell_min_depth': typewell_np['min_depth'],
        }

        # Add landing and normalization if calculated
        if landing_norm:
            result['perch_md'] = landing_norm['perch_md']
            result['detected_start_md'] = landing_norm['detected_start_md']
            result['norm_multiplier'] = landing_norm['norm_multiplier']
            result['norm_shift'] = landing_norm['norm_shift']
            result['norm_status'] = landing_norm['norm_status']

        return result

    except Exception as e:
        logger.error(f"Error processing {json_path.name}: {e}")
        return None


def load_welllist(config_path: Path) -> set:
    """Load well names from wells_config.json"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    wells = config.get('wells', [])
    return {w['well_name'] for w in wells if w.get('process', True)}


def convert_all_jsons(source_dir: Path, output_path: Path, device: str = 'cpu',
                      config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Convert all JSON files in source directory to a single PyTorch dataset.

    Args:
        source_dir: Directory with JSON files
        output_path: Output .pt file
        device: Device for tensors
        config_path: Optional wells_config.json to filter wells

    Returns:
        dict mapping well names to their data dicts
    """
    json_files = sorted(source_dir.glob("*.json"))
    logger.info(f"Found {len(json_files)} JSON files in {source_dir}")

    # Filter by welllist if config provided
    if config_path and config_path.exists():
        welllist = load_welllist(config_path)
        logger.info(f"Filtering by welllist: {len(welllist)} wells")
        json_files = [f for f in json_files if f.stem in welllist]
        logger.info(f"After filter: {len(json_files)} files")

    # Create detectors for landing and normalization
    landing_detector = create_landing_detector()
    norm_calculator = NormalizationCalculator(interactive_mode=False)
    logger.info("Created landing detector and normalization calculator")

    dataset = {}
    success_count = 0
    error_count = 0
    norm_success = 0

    for json_path in json_files:
        well_name = json_path.stem
        logger.info(f"Processing: {well_name}")

        result = process_single_json(json_path, device, landing_detector, norm_calculator)

        if result is not None:
            dataset[well_name] = result
            success_count += 1
            if result.get('norm_status') == 'success':
                norm_success += 1
        else:
            error_count += 1

    logger.info(f"Conversion complete: {success_count} success, {error_count} errors, {norm_success} normalized")

    # Save dataset
    torch.save(dataset, output_path)
    logger.info(f"Dataset saved to {output_path}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Convert JSON files to PyTorch dataset')
    parser.add_argument('--source', type=str,
                        default='/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de/AG_DATA/InitialData/',
                        help='Source directory with JSON files')
    parser.add_argument('--output', type=str,
                        default='/mnt/e/Projects/Rogii/gpu_ag/dataset/gpu_ag_dataset.pt',
                        help='Output .pt file path')
    parser.add_argument('--config', type=str,
                        default='/mnt/e/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de/wells_config.json',
                        help='wells_config.json for filtering (optional)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device for tensors (cpu or cuda)')

    args = parser.parse_args()

    source_dir = Path(args.source)
    output_path = Path(args.output)
    config_path = Path(args.config) if args.config else None

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = convert_all_jsons(source_dir, output_path, args.device, config_path)

    # Print summary
    print(f"\n=== Dataset Summary ===")
    print(f"Wells: {len(dataset)}")

    # Count normalization success
    norm_ok = sum(1 for d in dataset.values() if d.get('norm_status') == 'success')
    print(f"Normalization success: {norm_ok}/{len(dataset)}")

    if dataset:
        sample_name = next(iter(dataset))
        sample = dataset[sample_name]
        print(f"\nSample well: {sample_name}")
        print(f"  Well trajectory points: {len(sample['well_md'])}")
        print(f"  WellLog points: {len(sample['log_md'])}")
        print(f"  Typewell points: {len(sample['typewell_tvd'])}")
        print(f"  Reference segments: {len(sample['ref_shifts'])}")
        print(f"  start_md: {sample['start_md']:.2f}")
        print(f"  perch_md: {sample.get('perch_md', 'N/A')}")
        print(f"  norm_multiplier: {sample.get('norm_multiplier', 'N/A')}")
        print(f"  norm_shift: {sample.get('norm_shift', 'N/A')}")
        print(f"  norm_status: {sample.get('norm_status', 'N/A')}")


if __name__ == '__main__':
    main()
