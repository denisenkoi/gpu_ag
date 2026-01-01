#!/usr/bin/env python3
"""
Plot well gamma vs projected gamma (synt_curve) for visual correlation check.
Uses numpy_funcs/projection.py directly.
"""

import sys
import logging
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))

from cpu_baseline.typewell_provider import extend_pseudo_with_typelog
from numpy_funcs.projection import calc_horizontal_projection_numpy

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

METERS_TO_FEET = 3.28084


def load_dataset():
    """Load full dataset."""
    dataset_path = Path(__file__).parent / "dataset" / "gpu_ag_dataset.pt"
    return torch.load(dataset_path, map_location='cpu', weights_only=False)


def calc_vs_from_trajectory(ns, ew):
    """Calculate VS (cumulative horizontal distance) from NS/EW."""
    vs = np.zeros(len(ns))
    for i in range(1, len(ns)):
        dx = ns[i] - ns[i-1]
        dy = ew[i] - ew[i-1]
        vs[i] = vs[i-1] + np.sqrt(dx*dx + dy*dy)
    return vs


def stitch_typewell(data, mode='OLD'):
    """Stitch pseudo + type typewell."""
    norm_multiplier = data.get('norm_multiplier', 1.0)
    if isinstance(norm_multiplier, torch.Tensor):
        norm_multiplier = norm_multiplier.item()

    pseudo_tvd = data['pseudo_tvd'].cpu().numpy()
    pseudo_gr = data['pseudo_gr'].cpu().numpy()
    type_tvd = data['type_tvd'].cpu().numpy()
    type_gr = data['type_gr'].cpu().numpy()

    if mode == 'ORIGINAL':
        return type_tvd, type_gr

    pseudo_dict = {
        'tvdSortedPoints': [{'trueVerticalDepth': float(tvd), 'data': float(gr)}
                           for tvd, gr in zip(pseudo_tvd, pseudo_gr)]
    }
    type_dict = {
        'tvdSortedPoints': [{'trueVerticalDepth': float(tvd), 'data': float(gr)}
                           for tvd, gr in zip(type_tvd, type_gr)]
    }

    if mode == 'NEW':
        if norm_multiplier != 1.0:
            for p in pseudo_dict['tvdSortedPoints']:
                p['data'] *= norm_multiplier
        stitched = extend_pseudo_with_typelog(pseudo_dict, type_dict, norm_coef=1.0)
    else:  # OLD
        norm_coef = 1.0 / norm_multiplier if norm_multiplier != 0 else 1.0
        stitched = extend_pseudo_with_typelog(pseudo_dict, type_dict, norm_coef=norm_coef)

    points = stitched.get('tvdSortedPoints', [])
    tw_tvd = np.array([p['trueVerticalDepth'] for p in points])
    tw_gr = np.array([p['data'] for p in points])
    return tw_tvd, tw_gr


def build_segments_numpy(ref_mds, ref_shifts, well_md, well_vs):
    """
    Build segments_data numpy array.
    Format: (K, 6) - [start_idx, end_idx, start_vs, end_vs, start_shift, end_shift]

    ref_mds[i] = end MD of segment i
    ref_shifts[i] = end shift of segment i
    """
    segments = []

    for i in range(len(ref_mds) - 1):
        start_md = ref_mds[i]
        end_md = ref_mds[i + 1]

        # Skip zero-length segments
        if abs(end_md - start_md) < 0.01:
            continue

        # startShift = previous segment's end shift (or 0 for first)
        start_shift = ref_shifts[i - 1] if i > 0 else 0.0
        # endShift = this segment's end shift
        end_shift = ref_shifts[i]

        # Find indices
        start_idx = np.searchsorted(well_md, start_md)
        end_idx = np.searchsorted(well_md, end_md)

        if start_idx >= len(well_md):
            start_idx = len(well_md) - 1
        if end_idx >= len(well_md):
            end_idx = len(well_md) - 1

        # Skip if same index (zero VS delta)
        if start_idx == end_idx:
            continue

        start_vs = well_vs[start_idx]
        end_vs = well_vs[end_idx]

        segments.append([start_idx, end_idx, start_vs, end_vs, start_shift, end_shift])

    return np.array(segments, dtype=np.float64)


def main():
    # === HARDCODED CONFIG ===
    WELL_NAME = "Well162~EGFDL"
    MD_START_FT = 19600  # None for full well
    MD_END_FT = 20000    # None for full well
    MODE = 'ORIGINAL'    # ТОЛЬКО ORIGINAL!
    # ========================

    logger.info(f"Loading well: {WELL_NAME}")
    dataset = load_dataset()
    data = dataset[WELL_NAME]

    # Get arrays
    well_md = data['well_md'].cpu().numpy()
    well_tvd = data['well_tvd'].cpu().numpy()
    well_ns = data['well_ns'].cpu().numpy()
    well_ew = data['well_ew'].cpu().numpy()
    well_vs = calc_vs_from_trajectory(well_ns, well_ew)

    log_md = data['log_md'].cpu().numpy()
    log_gr = data['log_gr'].cpu().numpy()

    # NOTE: In ORIGINAL mode, typewell is raw, so log_gr should also be raw (no scaling)
    # In OLD mode: typewell scaled by 1/mult, log_gr should be raw
    # In NEW mode: typewell raw, log_gr should be scaled by mult
    # For simplicity, keep log_gr raw here

    # Interpolate log to well trajectory grid
    well_gr = np.interp(well_md, log_md, log_gr)

    # Reference interpretation
    ref_mds = data['ref_segment_mds'].cpu().numpy()
    ref_shifts = data['ref_shifts'].cpu().numpy()

    # TVD shift from dataset (critical for projection!)
    tvd_typewell_shift = data.get('tvd_typewell_shift', 0.0)
    if isinstance(tvd_typewell_shift, torch.Tensor):
        tvd_typewell_shift = tvd_typewell_shift.item()
    logger.info(f"tvd_typewell_shift: {tvd_typewell_shift:.4f}m")

    logger.info(f"Well MD: {well_md.min():.1f}-{well_md.max():.1f}m, points={len(well_md)}")
    logger.info(f"Reference: {len(ref_mds)} points, MD {ref_mds[0]:.1f}-{ref_mds[-1]:.1f}m")

    # Build typewell
    tw_tvd, tw_gr = stitch_typewell(data, mode=MODE)
    tw_step = tw_tvd[1] - tw_tvd[0] if len(tw_tvd) > 1 else 0.3048

    typewell_data = {
        'tvd': tw_tvd,
        'value': tw_gr,
        'min_depth': tw_tvd.min(),
        'typewell_step': tw_step,
        'normalized': False,
    }
    logger.info(f"Typewell: TVD {tw_tvd.min():.1f}-{tw_tvd.max():.1f}m, points={len(tw_tvd)}")

    # Build well_data
    well_data = {
        'md': well_md,
        'vs': well_vs,
        'tvd': well_tvd,
        'value': well_gr,
        'tvt': np.full_like(well_md, np.nan),
        'synt_curve': np.full_like(well_md, np.nan),
        'normalized': False,
    }

    # Build segments
    segments_data = build_segments_numpy(ref_mds, ref_shifts, well_md, well_vs)
    logger.info(f"Segments: {len(segments_data)}")

    # Calculate projection
    success, well_data = calc_horizontal_projection_numpy(
        well_data, typewell_data, segments_data, tvd_to_typewell_shift=tvd_typewell_shift
    )
    logger.info(f"Projection success: {success}")

    synt_curve = well_data['synt_curve']
    valid_mask = ~np.isnan(synt_curve)
    logger.info(f"Valid synt_curve: {valid_mask.sum()} / {len(synt_curve)}")

    # Calculate Pearson
    if valid_mask.sum() > 10:
        pearson = np.corrcoef(well_gr[valid_mask], synt_curve[valid_mask])[0, 1]
        logger.info(f"Overall Pearson: {pearson:.4f}")

    # Convert to feet
    md_ft = well_md * METERS_TO_FEET

    # Plot
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(md_ft, well_gr, 'b-', label='Well GR', linewidth=0.8, alpha=0.8)
    ax.plot(md_ft, synt_curve, 'r-', label='Synt GR (projected)', linewidth=0.8, alpha=0.8)

    ax.set_xlabel('MD (ft)')
    ax.set_ylabel('GR')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    if MD_START_FT and MD_END_FT:
        ax.set_xlim(MD_START_FT, MD_END_FT)

    def update_title(xmin, xmax):
        md_min = xmin / METERS_TO_FEET
        md_max = xmax / METERS_TO_FEET
        mask = (well_md >= md_min) & (well_md <= md_max) & valid_mask
        if mask.sum() > 10:
            p = np.corrcoef(well_gr[mask], synt_curve[mask])[0, 1]
            ax.set_title(f'{WELL_NAME} - MD {xmin:.0f}-{xmax:.0f} ft - Pearson: {p:.4f}')
        fig.canvas.draw_idle()

    ax.callbacks.connect('xlim_changed', lambda ev: update_title(*ev.get_xlim()))
    update_title(*ax.get_xlim())

    def on_scroll(event):
        if event.inaxes != ax:
            return
        factor = 1.2 if event.button == 'down' else 1/1.2
        cur = ax.get_xlim()
        width = (cur[1] - cur[0]) * factor
        rel = (event.xdata - cur[0]) / (cur[1] - cur[0])
        ax.set_xlim(event.xdata - width * rel, event.xdata + width * (1 - rel))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)

    print("\n=== Scroll to zoom MD ===")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
