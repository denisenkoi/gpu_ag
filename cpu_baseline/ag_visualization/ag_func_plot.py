import numpy as np
import matplotlib.pyplot as plt
from ag_rewards.ag_func_correlations import calculate_correlation


def show_well_curve_with_synth_well_data(well_data,
                                         segments,
                                         type_well,
                                         well_data_manual_interpretation=None):
    # Creating a GridSpec layout
    grid = plt.GridSpec(2, 2, width_ratios=[1, 0.5], wspace=0.5, hspace=0.5)

    fig = plt.figure(figsize=(10, 8))

    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0])
    ax3 = fig.add_subplot(grid[:, 1])

    # Interpolation of start_shift and end_shift for each segment
    interpolated_shifts = np.zeros_like(well_data.measured_depth)
    for segment in segments:
        interpolated_shifts[segment.start_idx:segment.interpretation_len + 1] = np.linspace(
            segment.start_shift, segment.end_shift, segment.interpretation_len - segment.start_idx + 1)


    return fig, [ax1, ax2, ax3]

def print_interpretation_info(
        well_data,
        well_calc_params,
        self_corr_start_idx,
        start_idx = 0,
        end_idx = 0
):
    well_data.first_valid_tvt_index()

    corr, \
    best_corr, \
    self_correlation, \
    mean_self_correlation_intersect, \
    num_try_with_intersection, \
    pearson, \
    num_points, \
    mse, \
    intersections_mult, \
    intersections_count = \
        calculate_correlation(well_data,
                              self_corr_start_idx,
                              start_idx=well_data.first_valid_tvt_index() if start_idx == 0 else start_idx,
                              end_idx=len(well_data.tvt) if end_idx == 0 else end_idx,
                              mean_self_correlation_intersect=1,
                              num_try_with_intersection=1,
                              best_corr=1,
                              pearson_power=well_calc_params['pearson_power'],
                              mse_power=well_calc_params['mse_power'],
                              num_intervals_self_correlation=well_calc_params['num_intervals_sc'],
                              sc_power=well_calc_params['sc_power'],
                              min_pearson_value=well_calc_params['min_pearson_value']
                              )

    print(f'corr = {corr}',
          f'pearson = {pearson}',
          f'mse = {mse}',
          f'self_correlation = {self_correlation}',
          f'intersections_mult = {intersections_mult}',
          f'intersections_count = {intersections_count}',
          f'num_points = {num_points}'),

