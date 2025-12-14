# self_correlation/curve_replacement_visualizer.py

"""
Curve Replacement Visualizer - generates PNG plots for curve replacement analysis
Path: self_correlation/curve_replacement_visualizer.py

Creates detailed plots showing the original TypeWell curve, horizontal well projection,
and the modified TypeWell result. Works in headless mode (no GUI display).
"""

import os
import logging
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Force headless backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional

# AG objects imports
from ag_objects.ag_obj_well import Well
from ag_objects.ag_obj_typewell import TypeWell

logger = logging.getLogger(__name__)


class CurveReplacementVisualizer:
    """Headless visualizer for curve replacement analysis"""

    def __init__(self, config: Dict[str, Any], logger_instance=None):
        # Use passed logger or global
        self.logger = logger_instance or logger

        # Load settings from config instead of environment
        self.save_plots = config.get('curve_replacement_save_plots', False)
        self.plots_dir = config.get('curve_replacement_plots_dir', 'self_correlation_plots')

        # Ensure plots directory exists (relative to project root)
        if self.save_plots:
            plots_path = Path(self.plots_dir)
            plots_path.mkdir(exist_ok=True, parents=True)
            self.logger.info(f"Curve replacement plots will be saved to: {plots_path.absolute()}")

        self.logger.info(f"CurveReplacementVisualizer initialized: save_plots={self.save_plots}")

    def create_replacement_plot(self,
                                well_name: str,
                                well: Well,
                                original_typewell: TypeWell,
                                modified_typewell: TypeWell,
                                replacement_info: Dict[str, Any]) -> Optional[Path]:
        """
        Create and save curve replacement visualization plot using AG objects directly

        Args:
            well_name: Name of the well for plot title
            well: Well object with horizontal projection calculated
            original_typewell: Original TypeWell object before replacement
            modified_typewell: Modified TypeWell object after replacement
            replacement_info: Dictionary with replacement parameters and results

        Returns:
            Path to saved plot file or None if plotting disabled
        """
        if not self.save_plots:
            self.logger.debug("Plot saving disabled, skipping visualization")
            return None

        self.logger.info(f"Creating curve replacement plot for: {well_name}")

        try:
            # Create high-resolution figure - VERY TALL for detailed scrolling
            fig, ax = plt.subplots(1, 1, figsize=(5, 210))  # 20 inches wide, 100 inches tall
            fig.suptitle(f'Curve Replacement Analysis: {well_name}', fontsize=24, fontweight='bold')

            # Plot all curves on single axis
            self._plot_all_curves_combined(ax, well, original_typewell, modified_typewell, replacement_info)

            # Add replacement info text
            self._add_replacement_info_text(fig, replacement_info)

            # Save plot with VERY HIGH DPI
            plot_filename = f"{well_name}_curve_replacement.png"
            plot_path = Path(self.plots_dir) / plot_filename

            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')  # High DPI for quality
            plt.close(fig)

            self.logger.info(f"Saved curve replacement plot: {plot_path}")
            return plot_path

        except Exception as e:
            self.logger.error(f"Failed to create plot for {well_name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            plt.close('all')  # Clean up any open figures
            return None

    def _plot_all_curves_combined(self, ax, well: Well, original_typewell: TypeWell,
                                  modified_typewell: TypeWell, replacement_info: Dict[str, Any]):
        """Plot all curves on single axis with contrasting styles"""

        # 1. Original TypeWell - ТОЛЩЕ, БЛЕДНЕЕ, ПЕРВАЯ
        original_line = ax.plot(original_typewell.value, original_typewell.tvd,
                                color='#0066bb', linewidth=4, label='Original TypeWell',
                                alpha=0.6, zorder=1)
        self.logger.info(f"✅ Plotted Original TypeWell: {len(original_typewell.value)} points")

        # 2. Modified TypeWell - ЖИРНЕЕ, КОНТРАСТНЕЕ, ВТОРАЯ
        modified_line = ax.plot(modified_typewell.value, modified_typewell.tvd,
                                color='red', linewidth=2, label='Modified TypeWell (After Replacement)',
                                alpha=0.9, zorder=2)
        self.logger.info(f"✅ Plotted Modified TypeWell: {len(modified_typewell.value)} points")

        # 3. Horizontal Projection - КОНТРАСТНО, ДРУГИМ ЦВЕТОМ, ТОЧКАМИ
        valid_mask = ~np.isnan(well.tvt) & ~np.isnan(well.value)
        valid_count = np.sum(valid_mask)

        if np.any(valid_mask):
            valid_tvt = well.tvt[valid_mask]
            valid_value = well.value[valid_mask]

            # Plot with smaller markers for better visibility
            projection_line = ax.plot(valid_value, valid_tvt,
                                      color='green', marker='o', markersize=3, linewidth=1,
                                      label='Horizontal Projection', alpha=0.7, zorder=3,
                                      linestyle='-', markevery=5)  # Show every 5th marker
            self.logger.info(f"✅ Plotted Horizontal Projection: {len(valid_value)} points")
        else:
            self.logger.error("❌ NO VALID horizontal projection data to plot!")
            # Add text to plot indicating no data
            ax.text(0.5, 0.5, 'NO VALID HORIZONTAL PROJECTION DATA',
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=20, color='red', weight='bold')

        # Mark replacement zone if available
        if 'replacement_start_tvt' in replacement_info and 'replacement_end_tvt' in replacement_info:
            start_tvt = replacement_info['replacement_start_tvt']
            end_tvt = replacement_info['replacement_end_tvt']

            if start_tvt is not None and end_tvt is not None:
                ax.axhspan(start_tvt, end_tvt, alpha=0.15, color='yellow',
                           label='Replacement Zone', zorder=0)
                ax.axhline(start_tvt, color='orange', linestyle='--', alpha=0.7, linewidth=1.5,
                           label=f'Replacement Start (TVT={start_tvt:.1f}m)')
                ax.axhline(end_tvt, color='darkred', linestyle=':', alpha=0.7, linewidth=1.5,
                           label=f'Max Penetration (TVT={end_tvt:.1f}m)')
                self.logger.info(f"✅ Marked replacement zone: {start_tvt:.1f} → {end_tvt:.1f}")

        # REMOVED: Large red dot for maximum penetration point
        # We still log the information but don't plot the big marker
        if ('max_penetration_tvt' in replacement_info and
                'max_penetration_idx' in replacement_info and
                replacement_info['max_penetration_tvt'] is not None and
                replacement_info['max_penetration_idx'] is not None):
            max_pen_tvt = replacement_info['max_penetration_tvt']
            max_pen_idx = replacement_info['max_penetration_idx']

            self.logger.info(f"Max Penetration Point (not plotted): TVT={max_pen_tvt:.1f}, Index={max_pen_idx}")

        # Mark junction point if found
        if 'junction_tvt' in replacement_info and replacement_info['junction_tvt'] is not None:
            junction_tvt = replacement_info['junction_tvt']
            ax.axhline(junction_tvt, color='purple', linestyle='-.', alpha=0.8, linewidth=1.5,
                       label=f'Junction Point (TVT={junction_tvt:.1f}m)')
            self.logger.info(f"✅ Marked junction point at TVT={junction_tvt:.1f}")

        # Configure axis
        ax.set_xlabel('Curve Value', fontsize=20)
        ax.set_ylabel('True Vertical Depth (TVD), m', fontsize=20)
        ax.set_title('Curve Replacement Analysis: TypeWell vs Horizontal Projection', fontsize=22)
        ax.legend(loc='best', fontsize=14, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Geological convention - depth increases downward

        # Increase tick label sizes for high resolution
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=12)

        # Add grid for better readability
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.1)
        ax.minorticks_on()

    def _add_replacement_info_text(self, fig, replacement_info: Dict[str, Any]):
        """Add text box with replacement parameters and results"""

        info_lines = []

        # Processing status
        if 'status' in replacement_info:
            status = replacement_info['status']
            info_lines.append(f"Status: {status}")
            if status != 'success':
                if 'reason' in replacement_info:
                    info_lines.append(f"Reason: {replacement_info['reason']}")

        # Replacement parameters
        if 'blend_weight' in replacement_info:
            info_lines.append(f"Blend Weight: {replacement_info['blend_weight']}")

        if 'replacement_range_meters' in replacement_info:
            info_lines.append(f"Replacement Range: {replacement_info['replacement_range_meters']}m")

        # Results
        if 'points_replaced' in replacement_info:
            info_lines.append(f"Points Replaced: {replacement_info['points_replaced']}")

        if 'points_extended' in replacement_info:
            info_lines.append(f"Points Extended: {replacement_info['points_extended']}")

        # MD range
        if 'original_start_md' in replacement_info and 'detected_start_md' in replacement_info:
            orig_md = replacement_info['original_start_md']
            det_md = replacement_info['detected_start_md']
            if orig_md is not None and det_md is not None:
                info_lines.append(f"Start MD: {orig_md:.1f}m → {det_md:.1f}m")

        # Junction info
        if 'junction_distance' in replacement_info and replacement_info['junction_distance'] is not None:
            dist = replacement_info['junction_distance']
            info_lines.append(f"Junction Distance: {dist:.1f}m")

        # TypeWell modification status
        if 'typewell_modified' in replacement_info:
            info_lines.append(f"TypeWell Modified: {replacement_info['typewell_modified']}")

        # Create text box with larger font for high resolution
        if info_lines:
            info_text = '\n'.join(info_lines)
            fig.text(0.02, 0.02, info_text, transform=fig.transFigure,
                     fontsize=16, verticalalignment='bottom',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))