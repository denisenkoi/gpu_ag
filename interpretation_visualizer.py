#!/usr/bin/env python3
"""
Interpretation Visualizer - generates comparison images for REF vs OPT interpretations.

Layout:
  ┌─────────────────┬─────────────────┬──────────┬──────────┐
  │ GR + Synth REF  │ GR + Synth OPT  │   REF    │   OPT    │
  ├─────────────────┴─────────────────┤ self-    │ self-    │
  │      Траектория + интерпретации   │ corr     │ corr     │
  └───────────────────────────────────┴──────────┴──────────┘

Usage:
    python interpretation_visualizer.py --well Well239~EGFDL --run-id 20260116_122821
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import psycopg2
from scipy.signal import savgol_filter

sys.path.insert(0, str(Path(__file__).parent))
from cpu_baseline.preprocessing import prepare_typelog

METERS_TO_FEET = 3.28084


def load_well_data(well_name: str) -> dict:
    """Load well data from dataset."""
    dataset_path = Path(__file__).parent / "data" / "wells_limited_pseudo.pt"
    dataset = torch.load(dataset_path, weights_only=False)
    return dataset[well_name]


def load_opt_interpretation(run_id: str, well_name: str) -> list:
    """Load OPT interpretation from database."""
    conn = psycopg2.connect(
        host='localhost', database='gpu_ag', user='rogii', password='rogii123'
    )
    cur = conn.cursor()
    cur.execute('''
        SELECT md_start, md_end, start_shift, end_shift
        FROM interpretations
        WHERE run_id = %s AND well_name = %s
        ORDER BY md_start
    ''', (run_id, well_name))
    segments = cur.fetchall()
    conn.close()
    return [{'startMd': s[0], 'endMd': s[1], 'startShift': s[2], 'endShift': s[3]}
            for s in segments]


def compute_shift_at_md(well_md: np.ndarray, segments: list) -> np.ndarray:
    """Compute interpolated shift at each MD point."""
    n_pts = len(well_md)
    shifts = np.full(n_pts, np.nan)

    for seg in segments:
        start_md = seg['startMd']
        end_md = seg['endMd']
        start_shift = seg['startShift']
        end_shift = seg['endShift']

        mask = (well_md >= start_md) & (well_md < end_md)
        if not np.any(mask):
            continue

        md_pts = well_md[mask]
        if end_md > start_md:
            ratio = (md_pts - start_md) / (end_md - start_md)
        else:
            ratio = np.zeros_like(md_pts)
        shifts[mask] = start_shift + ratio * (end_shift - start_shift)

    return shifts


def compute_tvt_from_shifts(well_tvd: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    """Compute TVT = TVD - shift."""
    return well_tvd - shifts


def compute_visual_interp_tvd(
    well_tvd: np.ndarray,
    shifts: np.ndarray,
    viz_start_idx: int,
    offset_above: float = 10.0
) -> np.ndarray:
    """
    Compute visual TVD for interpretation line.

    Formula: visual_tvd[i] = tvd_well[viz_start] - offset_above - shift[viz_start] + shift[i]

    Args:
        well_tvd: Well TVD array
        shifts: Interpretation shifts array
        viz_start_idx: Index of first point in visualization range (same for REF and OPT)
        offset_above: Offset above trajectory at start point (meters)

    In starting point: interpretation is offset_above meters above trajectory.
    Then it moves according to shift changes.
    """
    if np.isnan(shifts[viz_start_idx]):
        # Fallback: find first valid point
        valid = ~np.isnan(shifts)
        if not valid.any():
            return np.full_like(well_tvd, np.nan)
        viz_start_idx = np.argmax(valid)

    tvd_first = well_tvd[viz_start_idx]
    shift_first = shifts[viz_start_idx]

    # Constant base + current shift
    base = tvd_first - offset_above - shift_first
    return base + shifts


def compute_synthetic_gr(
    well_md: np.ndarray,
    well_tvd: np.ndarray,
    shifts: np.ndarray,
    type_tvd: np.ndarray,
    type_gr: np.ndarray
) -> np.ndarray:
    """Compute synthetic GR by projecting TypeLog using interpretation shifts."""
    tvt = compute_tvt_from_shifts(well_tvd, shifts)

    synthetic = np.full_like(well_md, np.nan)
    valid = ~np.isnan(tvt)

    if valid.sum() > 0:
        tvt_clamped = np.clip(tvt[valid], type_tvd.min(), type_tvd.max())
        synthetic[valid] = np.interp(tvt_clamped, type_tvd, type_gr)

    return synthetic


def get_ref_segments(well_data: dict) -> list:
    """Extract REF interpretation segments from well data."""
    ref_mds = np.asarray(well_data['ref_segment_mds'])
    ref_start_shifts = np.asarray(well_data['ref_start_shifts'])
    ref_shifts = np.asarray(well_data['ref_shifts'])

    segments = []
    n_segs = len(ref_mds)
    for i in range(n_segs):
        start_md = ref_mds[i]
        end_md = ref_mds[i + 1] if i < n_segs - 1 else start_md + 1000
        segments.append({
            'startMd': float(start_md),
            'endMd': float(end_md),
            'startShift': float(ref_start_shifts[i]),
            'endShift': float(ref_shifts[i])
        })
    return segments


def compute_std_bin_means(tvt: np.ndarray, gr: np.ndarray, bin_size: float = 0.05) -> float:
    """Compute STD of bin means for self-correlation metric."""
    valid = ~np.isnan(tvt)
    if valid.sum() < 10:
        return np.nan

    tvt_valid = tvt[valid]
    gr_valid = gr[valid]

    if len(gr_valid) >= 51:
        gr_smooth = savgol_filter(gr_valid, 51, 3)
    else:
        gr_smooth = gr_valid

    tvt_min = tvt_valid.min()
    bin_idx = ((tvt_valid - tvt_min) / bin_size).astype(int)
    n_bins = bin_idx.max() + 1

    if n_bins < 5:
        return np.nan

    bin_sums = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    np.add.at(bin_sums, bin_idx, gr_smooth)
    np.add.at(bin_counts, bin_idx, 1)

    valid_bins = bin_counts > 0
    if valid_bins.sum() < 5:
        return np.nan

    bin_means = bin_sums[valid_bins] / bin_counts[valid_bins]
    return np.std(bin_means)


def plot_interpretation(
    well_name: str,
    well_data: dict,
    opt_segments: list,
    run_id: str,
    output_path: str = None,
    show: bool = True
):
    """Generate comparison plot for REF vs OPT interpretation."""

    # Prepare data EXACTLY as optimizer does (same ENV settings)
    type_tvd, type_gr, meta = prepare_typelog(well_data)
    well_md = meta['well_md']
    well_gr = meta['well_gr_norm']  # Normalized and smoothed same as optimizer
    well_tvd = np.asarray(well_data['well_tvd'])

    # Landing point - start visualization from here - 600ft
    landing_end_md = float(well_data.get('landing_end_dls', well_md[len(well_md)//2]))
    viz_start_md = landing_end_md - 600 / METERS_TO_FEET

    # Get REF segments
    ref_segments = get_ref_segments(well_data)

    # Mask for visualization range
    viz_mask = well_md >= viz_start_md

    # Compute shifts for both interpretations
    shifts_ref = compute_shift_at_md(well_md, ref_segments)
    shifts_opt = compute_shift_at_md(well_md, opt_segments)

    # Find OPT start point (where optimization begins, shift taken from REF)
    opt_start_md = opt_segments[0]['startMd'] if opt_segments else viz_start_md
    opt_start_idx = np.searchsorted(well_md, opt_start_md)
    opt_start_idx = min(opt_start_idx, len(well_md) - 1)


    # Compute TVT for metrics
    tvt_ref = compute_tvt_from_shifts(well_tvd, shifts_ref)
    tvt_opt = compute_tvt_from_shifts(well_tvd, shifts_opt)

    # Compute synthetic GR
    synth_ref = compute_synthetic_gr(well_md, well_tvd, shifts_ref, type_tvd, type_gr)
    synth_opt = compute_synthetic_gr(well_md, well_tvd, shifts_opt, type_tvd, type_gr)

    # Compute visual interpretation lines (10m above trajectory at OPT start)
    # Both aligned at OPT start point where they should match
    visual_tvd_ref = compute_visual_interp_tvd(well_tvd, shifts_ref, opt_start_idx, offset_above=10.0)
    visual_tvd_opt = compute_visual_interp_tvd(well_tvd, shifts_opt, opt_start_idx, offset_above=10.0)

    # Post-landing mask for metrics
    post_mask = well_md > landing_end_md

    # Compute STD metrics (5cm bins)
    std_ref = compute_std_bin_means(tvt_ref[post_mask], well_gr[post_mask], 0.05)
    std_opt = compute_std_bin_means(tvt_opt[post_mask], well_gr[post_mask], 0.05)

    # TVT ranges
    tvt_ref_post = tvt_ref[post_mask]
    tvt_opt_post = tvt_opt[post_mask]
    valid_ref = ~np.isnan(tvt_ref_post)
    valid_opt = ~np.isnan(tvt_opt_post)

    tvt_range_ref = (np.nanmax(tvt_ref_post) - np.nanmin(tvt_ref_post)) * METERS_TO_FEET if valid_ref.sum() > 0 else 0
    tvt_range_opt = (np.nanmax(tvt_opt_post) - np.nanmin(tvt_opt_post)) * METERS_TO_FEET if valid_opt.sum() > 0 else 0

    # === Create figure with GridSpec ===
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, width_ratios=[3, 0.8, 0.8], height_ratios=[1, 1, 1.5],
                  hspace=0.15, wspace=0.15)

    # Subplots - left column stacked vertically
    ax_gr_ref = fig.add_subplot(gs[0, 0])      # GR + Synth REF
    ax_gr_opt = fig.add_subplot(gs[1, 0], sharex=ax_gr_ref, sharey=ax_gr_ref)  # GR + Synth OPT
    ax_traj = fig.add_subplot(gs[2, 0], sharex=ax_gr_ref)  # Trajectory
    ax_sc_ref = fig.add_subplot(gs[:, 1])      # REF self-correlation (spans all rows)
    ax_sc_opt = fig.add_subplot(gs[:, 2], sharey=ax_sc_ref)  # OPT self-correlation

    fig.suptitle(f'{well_name} | Run: {run_id}\n'
                 f'STD: REF={std_ref:.2f}, OPT={std_opt:.2f} | '
                 f'TVT range: REF={tvt_range_ref:.1f}ft, OPT={tvt_range_opt:.1f}ft',
                 fontsize=12, fontweight='bold')

    # Data for visualization
    md_viz = well_md[viz_mask] * METERS_TO_FEET
    gr_viz = well_gr[viz_mask]
    synth_ref_viz = synth_ref[viz_mask]
    synth_opt_viz = synth_opt[viz_mask]
    tvd_viz = well_tvd[viz_mask] * METERS_TO_FEET
    visual_tvd_ref_viz = visual_tvd_ref[viz_mask] * METERS_TO_FEET
    visual_tvd_opt_viz = visual_tvd_opt[viz_mask] * METERS_TO_FEET

    # === ROW 0: GR + Synth REF ===
    ax_gr_ref.plot(md_viz, gr_viz, 'k-', linewidth=0.8, label='Well GR', alpha=0.7)
    ax_gr_ref.plot(md_viz, synth_ref_viz, 'g-', linewidth=1.5, label='Synth REF', alpha=0.9)
    ax_gr_ref.axvline(x=landing_end_md * METERS_TO_FEET, color='orange', linestyle='--', alpha=0.5)
    ax_gr_ref.set_ylabel('GR', fontsize=10)
    ax_gr_ref.legend(loc='upper right', fontsize=8)
    ax_gr_ref.set_title('REF: GR & Synthetic', fontsize=10)
    ax_gr_ref.grid(True, alpha=0.3)
    plt.setp(ax_gr_ref.get_xticklabels(), visible=False)

    # === ROW 1: GR + Synth OPT ===
    ax_gr_opt.plot(md_viz, gr_viz, 'k-', linewidth=0.8, label='Well GR', alpha=0.7)
    ax_gr_opt.plot(md_viz, synth_opt_viz, 'r-', linewidth=1.5, label='Synth OPT', alpha=0.9)
    ax_gr_opt.axvline(x=landing_end_md * METERS_TO_FEET, color='orange', linestyle='--', alpha=0.5)
    ax_gr_opt.set_ylabel('GR', fontsize=10)
    ax_gr_opt.legend(loc='upper right', fontsize=8)
    ax_gr_opt.set_title('OPT: GR & Synthetic', fontsize=10)
    ax_gr_opt.grid(True, alpha=0.3)
    plt.setp(ax_gr_opt.get_xticklabels(), visible=False)

    # === BOTTOM-LEFT: Trajectory + Interpretations ===
    ax_traj.plot(md_viz, tvd_viz, 'k-', linewidth=2, label='Trajectory')

    # REF interpretation - thick gray (background)
    valid_ref_viz = ~np.isnan(visual_tvd_ref_viz)
    ax_traj.plot(md_viz[valid_ref_viz], visual_tvd_ref_viz[valid_ref_viz],
                 color='gray', linewidth=4, label='REF interp', alpha=0.6)

    # OPT interpretation - thin blue (on top)
    valid_opt_viz = ~np.isnan(visual_tvd_opt_viz)
    ax_traj.plot(md_viz[valid_opt_viz], visual_tvd_opt_viz[valid_opt_viz],
                 color='blue', linewidth=2, label='OPT interp', alpha=0.9)

    ax_traj.axvline(x=landing_end_md * METERS_TO_FEET, color='orange', linestyle='--',
                    alpha=0.5, label='Landing end')
    ax_traj.axvline(x=opt_start_md * METERS_TO_FEET, color='purple', linestyle=':',
                    alpha=0.7, label='OPT start')

    ax_traj.set_xlabel('MD (ft)', fontsize=10)
    ax_traj.set_ylabel('TVD (ft)', fontsize=10)
    ax_traj.legend(loc='lower right', fontsize=9)
    ax_traj.set_title('Trajectory & Interpretations', fontsize=10)
    ax_traj.invert_yaxis()
    ax_traj.grid(True, alpha=0.3)

    # === RIGHT: Self-correlation plots ===
    # Use viz_mask data for self-correlation (same range as left plots)
    tvt_ref_viz = tvt_ref[viz_mask]
    tvt_opt_viz = tvt_opt[viz_mask]
    gr_viz_sc = well_gr[viz_mask]
    valid_ref_sc = ~np.isnan(tvt_ref_viz)
    valid_opt_sc = ~np.isnan(tvt_opt_viz)

    # Determine TVT range from viz data only
    all_tvt_viz = np.concatenate([tvt_ref_viz[valid_ref_sc], tvt_opt_viz[valid_opt_sc]])
    tvt_min = np.nanmin(all_tvt_viz) * METERS_TO_FEET
    tvt_max = np.nanmax(all_tvt_viz) * METERS_TO_FEET
    tvt_margin = (tvt_max - tvt_min) * 0.05
    tvt_ylim = (tvt_max + tvt_margin, tvt_min - tvt_margin)

    # REF self-correlation
    ax_sc_ref.plot(gr_viz_sc[valid_ref_sc], tvt_ref_viz[valid_ref_sc] * METERS_TO_FEET,
                   'g-', linewidth=0.5, alpha=0.7, label='Well GR')
    ax_sc_ref.plot(type_gr, type_tvd * METERS_TO_FEET,
                   'b-', linewidth=2.5, alpha=0.8, label='TypeLog')
    ax_sc_ref.set_xlabel('GR', fontsize=10)
    ax_sc_ref.set_ylabel('TVT (ft)', fontsize=10)
    ax_sc_ref.set_title(f'REF Self-Corr\nSTD={std_ref:.2f}', fontsize=10)
    ax_sc_ref.legend(loc='upper right', fontsize=8)
    ax_sc_ref.set_ylim(tvt_ylim)
    ax_sc_ref.grid(True, alpha=0.3)

    # OPT self-correlation
    ax_sc_opt.plot(gr_viz_sc[valid_opt_sc], tvt_opt_viz[valid_opt_sc] * METERS_TO_FEET,
                   'r-', linewidth=0.5, alpha=0.7, label='Well GR')
    ax_sc_opt.plot(type_gr, type_tvd * METERS_TO_FEET,
                   'b-', linewidth=2.5, alpha=0.8, label='TypeLog')
    ax_sc_opt.set_xlabel('GR', fontsize=10)
    ax_sc_opt.set_title(f'OPT Self-Corr\nSTD={std_opt:.2f}', fontsize=10)
    ax_sc_opt.legend(loc='upper right', fontsize=8)
    ax_sc_opt.set_ylim(tvt_ylim)
    ax_sc_opt.grid(True, alpha=0.3)
    plt.setp(ax_sc_opt.get_yticklabels(), visible=False)

    plt.tight_layout()

    # Save
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return {
        'std_ref': std_ref,
        'std_opt': std_opt,
        'tvt_range_ref': tvt_range_ref,
        'tvt_range_opt': tvt_range_opt,
    }


def main():
    parser = argparse.ArgumentParser(description='Interpretation Visualizer')
    parser.add_argument('--well', required=True, help='Well name (e.g., Well239~EGFDL)')
    parser.add_argument('--run-id', required=True, help='Run ID for OPT interpretation')
    parser.add_argument('--output', '-o', help='Output file path (PNG)')
    parser.add_argument('--no-show', action='store_true', help='Do not display plot')
    args = parser.parse_args()

    print(f"Loading well: {args.well}")
    well_data = load_well_data(args.well)

    print(f"Loading OPT interpretation: {args.run_id}")
    opt_segments = load_opt_interpretation(args.run_id, args.well)
    print(f"Loaded {len(opt_segments)} OPT segments")

    output_path = args.output
    if not output_path:
        output_dir = Path(__file__).parent / "visualizations"
        output_path = output_dir / f"viz_{args.well}_{args.run_id}.png"

    metrics = plot_interpretation(
        args.well, well_data, opt_segments, args.run_id,
        output_path=str(output_path), show=not args.no_show
    )

    print(f"\nMetrics:")
    print(f"  REF STD: {metrics['std_ref']:.2f}, TVT range: {metrics['tvt_range_ref']:.1f}ft")
    print(f"  OPT STD: {metrics['std_opt']:.2f}, TVT range: {metrics['tvt_range_opt']:.1f}ft")


if __name__ == '__main__':
    main()
