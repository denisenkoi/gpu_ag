#!/usr/bin/env python3
"""
TVT Visualizer - интерактивное сравнение TypeLog, PseudoTypeLog и Well GR на оси TVT.

Dash приложение:
- Dropdown скважин (сортировка по RMSE, худшие первые)
- 2 вертикальных графика (ось Y = TVT)
- Фильтры Савицкого-Голея для PseudoTypeLog
- Подсветка зоны нормализации

Запуск: python tvt_visualizer.py
"""

import sys
import os
from pathlib import Path

# Load .env before other imports
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / "cpu_baseline" / ".env")
except ImportError:
    pass

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "cpu_baseline"))
from numpy_funcs.interpretation import interpolate_shift_at_md
from cpu_baseline.typewell_provider import (
    stitch_typewell_from_dataset,
    GR_SMOOTHING_WINDOW,
    GR_SMOOTHING_ORDER
)
from cpu_baseline.typelog_preprocessing import (
    prepare_typelog,
    apply_gr_smoothing,
    compute_overlap_metrics
)

# Load data once at startup
print("Loading dataset...")
DATASET = torch.load('dataset/gpu_ag_dataset.pt', weights_only=False)

# Try to load baseline errors, fallback to simple well list
ERRORS = None
try:
    ERRORS = torch.load('dataset/baseline_errors.pt', weights_only=False)
    print(f"Loaded {len(DATASET)} wells with baseline errors")
except FileNotFoundError:
    print(f"Loaded {len(DATASET)} wells (no baseline_errors.pt)")

# Build dropdown options
if ERRORS:
    WELL_OPTIONS = [
        {'label': f"{e['well']} (RMSE={e['rmse']:.1f}m, end={e['endpoint_error']:+.1f}m)", 'value': e['well']}
        for e in ERRORS
    ]
    DEFAULT_WELL = ERRORS[0]['well']
else:
    WELL_OPTIONS = [{'label': w, 'value': w} for w in sorted(DATASET.keys())]
    DEFAULT_WELL = WELL_OPTIONS[0]['value'] if WELL_OPTIONS else None


def get_well_tvt_data(well_name):
    """Get all TVT data for a well."""
    data = DATASET[well_name]

    well_md = data['well_md'].numpy()
    well_tvd = data['well_tvd'].numpy()
    log_md = data['log_md'].numpy()
    log_gr = data['log_gr'].numpy()

    type_tvd = data['type_tvd'].numpy()
    type_gr = data['type_gr'].numpy()
    pseudo_tvd = data['pseudo_tvd'].numpy()
    pseudo_gr = data['pseudo_gr'].numpy()

    tvd_shift = float(data.get('tvd_typewell_shift', 0.0))
    norm_mult = float(data.get('norm_multiplier', 1.0))

    # TypeLog TVT = type_tvd + tvd_shift (to align with well)
    type_tvt = type_tvd + tvd_shift

    # PseudoTypeLog is already in TVT coordinates
    pseudo_tvt = pseudo_tvd

    # OLD: Stitched typewell (extend_pseudo_with_typelog, tvd_shift applied in optimizer)
    stitched_tvt_old, stitched_gr_old = stitch_typewell_from_dataset(data, apply_smoothing=True)

    # NEW: prepare_typelog (stitch_typelogs, tvd_shift applied before stitch)
    try:
        stitched_tvt_new, stitched_gr_new, new_meta = prepare_typelog(data, use_pseudo=True, apply_smoothing=True)
    except Exception as e:
        print(f"Warning: prepare_typelog failed for {well_name}: {e}")
        stitched_tvt_new, stitched_gr_new = stitched_tvt_old, stitched_gr_old

    # Overlap metrics (normalized MSE×10 and Pearson)
    overlap_metrics = compute_overlap_metrics(type_tvd, type_gr, pseudo_tvd, pseudo_gr)

    # Well GR interpolated to well_md grid and normalized
    well_gr = np.interp(well_md, log_md, log_gr)
    if norm_mult != 1.0:
        well_gr = well_gr * norm_mult

    # === REF projection: TVT from ref interpretation ===
    well_tvt_ref = np.full_like(well_md, np.nan)
    for i, md in enumerate(well_md):
        shift = interpolate_shift_at_md(data, md)
        well_tvt_ref[i] = well_tvd[i] - shift

    # === Normalization zone (perch-150 to perch+50) ===
    perch_md = float(data.get('perch_md', well_md[-1]))
    norm_start_md = perch_md - 150.0
    norm_end_md = perch_md + 50.0

    # Convert to TVT using ref interpretation
    norm_start_shift = interpolate_shift_at_md(data, norm_start_md)
    norm_end_shift = interpolate_shift_at_md(data, norm_end_md)
    perch_shift = interpolate_shift_at_md(data, perch_md)

    # Interpolate TVD at these MDs
    norm_start_tvd = np.interp(norm_start_md, well_md, well_tvd)
    norm_end_tvd = np.interp(norm_end_md, well_md, well_tvd)
    perch_tvd = np.interp(perch_md, well_md, well_tvd)

    norm_start_tvt = norm_start_tvd - norm_start_shift
    norm_end_tvt = norm_end_tvd - norm_end_shift
    perch_tvt = perch_tvd - perch_shift

    # Landing point
    landing_md = float(data.get('landing_end_87_200', well_md[len(well_md)//2]))
    landing_idx = int(np.searchsorted(well_md, landing_md))
    if landing_idx >= len(well_md):
        landing_idx = len(well_md) - 1
    ref_shift_at_landing = interpolate_shift_at_md(data, well_md[landing_idx])
    landing_tvt = well_tvd[landing_idx] - ref_shift_at_landing

    return {
        'type_tvt': type_tvt,
        'type_gr': type_gr,
        'pseudo_tvt': pseudo_tvt,
        'pseudo_gr': pseudo_gr,
        # OLD (master) stitched typelog
        'stitched_tvt_old': stitched_tvt_old,
        'stitched_gr_old': stitched_gr_old,
        # NEW (fixed) stitched typelog
        'stitched_tvt_new': stitched_tvt_new,
        'stitched_gr_new': stitched_gr_new,
        'well_tvt_ref': well_tvt_ref,
        'well_gr': well_gr,
        'well_md': well_md,
        'tvd_shift': tvd_shift,
        'norm_mult': norm_mult,
        'landing_tvt': landing_tvt,
        'landing_md': landing_md,
        # Normalization zone
        'norm_start_tvt': norm_start_tvt,
        'norm_end_tvt': norm_end_tvt,
        'perch_tvt': perch_tvt,
        # Overlap metrics
        'overlap_mse': overlap_metrics['mse'],
        'overlap_pearson': overlap_metrics['pearson'],
        'overlap_length': overlap_metrics['overlap_length'],
        # Counts for info
        'n_points_old': len(stitched_tvt_old),
        'n_points_new': len(stitched_tvt_new),
    }


def apply_filter(data, filter_type, window, polyorder):
    """Apply filter to data."""
    if filter_type == 'none' or window < 3:
        return data

    # Window must be odd for savgol
    if window % 2 == 0:
        window += 1

    if filter_type == 'savgol':
        # Ensure polyorder < window
        polyorder = min(polyorder, window - 1)
        return savgol_filter(data, window, polyorder)
    elif filter_type == 'moving_avg':
        return uniform_filter1d(data, size=window)
    else:
        return data


# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2('TVT Visualizer: TypeLog vs PseudoTypeLog vs Well GR', style={'textAlign': 'center'}),

    # Well selector
    html.Div([
        html.Label('Скважина (сортировка по RMSE):'),
        dcc.Dropdown(
            id='well-dropdown',
            options=WELL_OPTIONS,
            value=DEFAULT_WELL,
            style={'width': '100%'}
        ),
    ], style={'width': '50%', 'margin': '10px auto'}),

    html.Div(id='well-info', style={'textAlign': 'center', 'fontSize': '14px', 'color': 'gray', 'marginBottom': '10px'}),

    # Filters row
    html.Div([
        html.Div([
            html.Label('Filter:'),
            dcc.Dropdown(
                id='filter-type',
                options=[
                    {'label': 'None', 'value': 'none'},
                    {'label': 'Savitzky-Golay', 'value': 'savgol'},
                    {'label': 'Moving Average', 'value': 'moving_avg'},
                ],
                value='none',
                style={'width': '150px'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '20px'}),

        html.Div([
            html.Label('Window:'),
            dcc.Slider(
                id='filter-window',
                min=3, max=51, step=2, value=5,
                marks={3: '3', 11: '11', 21: '21', 31: '31', 51: '51'},
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
        ], style={'display': 'inline-block', 'width': '200px', 'marginRight': '20px'}),

        html.Div([
            html.Label('Poly order (S-G):'),
            dcc.Slider(
                id='filter-polyorder',
                min=1, max=5, step=1, value=2,
                marks={1: '1', 2: '2', 3: '3', 4: '4', 5: '5'},
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
        ], style={'display': 'inline-block', 'width': '150px'}),
    ], style={'textAlign': 'center', 'marginBottom': '10px'}),

    # TypeLog mode selector
    html.Div([
        html.Div([
            html.Label('TypeLog mode:'),
            dcc.RadioItems(
                id='typelog-mode',
                options=[
                    {'label': 'OLD (master, extend_pseudo)', 'value': 'old'},
                    {'label': 'NEW (fixed, stitch_typelogs)', 'value': 'new'},
                ],
                value='old',
                inline=True,
                style={'marginLeft': '10px'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '30px'}),

        html.Div([
            dcc.Checklist(
                id='show-both-typelogs',
                options=[{'label': 'Show both OLD and NEW', 'value': 'show'}],
                value=[],
                inline=True
            ),
        ], style={'display': 'inline-block'}),
    ], style={'textAlign': 'center', 'marginBottom': '10px', 'padding': '10px', 'backgroundColor': '#f0f0f0'}),

    dcc.Graph(
        id='tvt-graph',
        style={'height': '80vh'},
        config={'scrollZoom': True}
    ),
])


@app.callback(
    [Output('tvt-graph', 'figure'),
     Output('well-info', 'children')],
    [Input('well-dropdown', 'value'),
     Input('filter-type', 'value'),
     Input('filter-window', 'value'),
     Input('filter-polyorder', 'value'),
     Input('typelog-mode', 'value'),
     Input('show-both-typelogs', 'value')]
)
def update_graph(well_name, filter_type, filter_window, filter_polyorder, typelog_mode, show_both):
    """Update graph when inputs change."""
    if not well_name:
        return go.Figure(), "Select a well"

    data = get_well_tvt_data(well_name)

    # Apply filter to pseudo
    pseudo_gr_filtered = apply_filter(data['pseudo_gr'], filter_type, filter_window, filter_polyorder)

    # Create subplots: 2 columns, shared Y axis
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=['REF interpretation', 'AUTO (baseline)']
    )

    # Colors
    COLOR_TYPE = 'blue'
    COLOR_PSEUDO = 'green'
    COLOR_WELL = 'red'

    # === Normalization zone highlight (both plots) ===
    norm_start = data['norm_start_tvt']
    norm_end = data['norm_end_tvt']

    for col in [1, 2]:
        fig.add_hrect(
            y0=norm_start, y1=norm_end,
            fillcolor='yellow', opacity=0.15,
            line_width=0,
            row=1, col=col
        )
        # Norm zone boundaries
        fig.add_hline(y=norm_start, line_dash='dot', line_color='goldenrod', opacity=0.6, row=1, col=col)
        fig.add_hline(y=norm_end, line_dash='dot', line_color='goldenrod', opacity=0.6, row=1, col=col)

    # === LEFT: REF interpretation ===
    # TypeLog
    fig.add_trace(
        go.Scatter(
            x=data['type_gr'], y=data['type_tvt'],
            mode='lines', name='TypeLog',
            line=dict(color=COLOR_TYPE, width=1.5),
            legendgroup='type', showlegend=True,
            hovertemplate='GR: %{x:.1f}<br>TVT: %{y:.2f}m<extra>TypeLog</extra>'
        ),
        row=1, col=1
    )

    # PseudoTypeLog (filtered)
    fig.add_trace(
        go.Scatter(
            x=pseudo_gr_filtered, y=data['pseudo_tvt'],
            mode='lines', name='PseudoTypeLog',
            line=dict(color=COLOR_PSEUDO, width=1.5),
            legendgroup='pseudo', showlegend=True,
            hovertemplate='GR: %{x:.1f}<br>TVT: %{y:.2f}m<extra>PseudoTypeLog</extra>'
        ),
        row=1, col=1
    )

    # Well GR (REF projection)
    valid_ref = ~np.isnan(data['well_tvt_ref'])
    fig.add_trace(
        go.Scatter(
            x=data['well_gr'][valid_ref], y=data['well_tvt_ref'][valid_ref],
            mode='lines', name='Well GR (REF)',
            line=dict(color=COLOR_WELL, width=1), opacity=0.7,
            legendgroup='well_ref', showlegend=True,
            hovertemplate='GR: %{x:.1f}<br>TVT: %{y:.2f}m<extra>Well GR REF</extra>'
        ),
        row=1, col=1
    )

    # === RIGHT: AUTO (stitched typewell) ===
    # Choose which stitched typelog to show based on mode
    COLOR_OLD = 'purple'
    COLOR_NEW = 'green'

    # Primary stitched typelog (based on mode)
    if typelog_mode == 'new':
        primary_tvt = data['stitched_tvt_new']
        primary_gr = data['stitched_gr_new']
        primary_color = COLOR_NEW
        primary_name = f"Stitched NEW ({data['n_points_new']}pts)"
    else:
        primary_tvt = data['stitched_tvt_old']
        primary_gr = data['stitched_gr_old']
        primary_color = COLOR_OLD
        primary_name = f"Stitched OLD ({data['n_points_old']}pts)"

    fig.add_trace(
        go.Scatter(
            x=primary_gr, y=primary_tvt,
            mode='lines', name=primary_name,
            line=dict(color=primary_color, width=2),
            legendgroup='stitched_primary', showlegend=True,
            hovertemplate='GR: %{x:.1f}<br>TVT: %{y:.2f}m<extra>' + primary_name + '</extra>'
        ),
        row=1, col=2
    )

    # Show both if checkbox is checked
    if show_both and 'show' in show_both:
        if typelog_mode == 'new':
            secondary_tvt = data['stitched_tvt_old']
            secondary_gr = data['stitched_gr_old']
            secondary_color = COLOR_OLD
            secondary_name = f"Stitched OLD ({data['n_points_old']}pts)"
        else:
            secondary_tvt = data['stitched_tvt_new']
            secondary_gr = data['stitched_gr_new']
            secondary_color = COLOR_NEW
            secondary_name = f"Stitched NEW ({data['n_points_new']}pts)"

        fig.add_trace(
            go.Scatter(
                x=secondary_gr, y=secondary_tvt,
                mode='lines', name=secondary_name,
                line=dict(color=secondary_color, width=1, dash='dash'),
                legendgroup='stitched_secondary', showlegend=True, opacity=0.7,
                hovertemplate='GR: %{x:.1f}<br>TVT: %{y:.2f}m<extra>' + secondary_name + '</extra>'
            ),
            row=1, col=2
        )

    # TypeLog (faded, for comparison)
    fig.add_trace(
        go.Scatter(
            x=data['type_gr'], y=data['type_tvt'],
            mode='lines', name='TypeLog (orig)',
            line=dict(color=COLOR_TYPE, width=1, dash='dot'),
            legendgroup='type', showlegend=False, opacity=0.4,
            hovertemplate='GR: %{x:.1f}<br>TVT: %{y:.2f}m<extra>TypeLog</extra>'
        ),
        row=1, col=2
    )

    # PseudoTypeLog (faded, for comparison)
    fig.add_trace(
        go.Scatter(
            x=pseudo_gr_filtered, y=data['pseudo_tvt'],
            mode='lines', name='PseudoTypeLog (orig)',
            line=dict(color=COLOR_PSEUDO, width=1, dash='dot'),
            legendgroup='pseudo', showlegend=False, opacity=0.4,
            hovertemplate='GR: %{x:.1f}<br>TVT: %{y:.2f}m<extra>PseudoTypeLog</extra>'
        ),
        row=1, col=2
    )

    # Well GR (AUTO) - same as REF for now, will add auto-interp later
    fig.add_trace(
        go.Scatter(
            x=data['well_gr'][valid_ref], y=data['well_tvt_ref'][valid_ref],
            mode='lines', name='Well GR (AUTO)',
            line=dict(color=COLOR_WELL, width=1, dash='dot'), opacity=0.7,
            legendgroup='well_auto', showlegend=True,
            hovertemplate='GR: %{x:.1f}<br>TVT: %{y:.2f}m<extra>Well GR AUTO</extra>'
        ),
        row=1, col=2
    )

    # Landing point marker
    landing_tvt = data['landing_tvt']
    fig.add_hline(y=landing_tvt, line_dash='dash', line_color='orange', opacity=0.5,
                  annotation_text=f'Landing (87°+200m)', row=1, col=1)
    fig.add_hline(y=landing_tvt, line_dash='dash', line_color='orange', opacity=0.5, row=1, col=2)

    # Layout
    fig.update_layout(
        title=f'<b>{well_name}</b>',
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
        margin=dict(l=60, r=20, t=80, b=40),
        hovermode='closest',
    )

    # Y axis: TVT (inverted)
    fig.update_yaxes(title_text='TVT (m)', autorange='reversed', row=1, col=1)
    fig.update_yaxes(autorange='reversed', row=1, col=2)

    # X axes - fixed range
    fig.update_xaxes(title_text='GR', fixedrange=True, row=1, col=1)
    fig.update_xaxes(title_text='GR', fixedrange=True, row=1, col=2)

    # Info text
    error_info = next((e for e in ERRORS if e['well'] == well_name), None)
    filter_info = f"Filter: {filter_type}" if filter_type != 'none' else "No filter"
    if filter_type == 'savgol':
        filter_info += f" (w={filter_window}, p={filter_polyorder})"
    elif filter_type == 'moving_avg':
        filter_info += f" (w={filter_window})"

    # Overlap metrics
    overlap_info = f"Overlap: MSE={data['overlap_mse']:.1f}, r={data['overlap_pearson']:.3f}, {data['overlap_length']:.0f}m"

    # TypeLog mode info
    mode_info = f"TypeLog: {typelog_mode.upper()} (OLD:{data['n_points_old']}pts, NEW:{data['n_points_new']}pts)"

    if error_info:
        info = (f"RMSE={error_info['rmse']:.2f}m | Endpoint={error_info['endpoint_error']:+.2f}m | "
                f"norm_mult={data['norm_mult']:.4f} | {overlap_info} | {mode_info} | {filter_info}")
    else:
        info = f"norm_mult={data['norm_mult']:.4f} | {overlap_info} | {mode_info} | {filter_info}"

    return fig, info


if __name__ == '__main__':
    import subprocess
    ip = subprocess.check_output("hostname -I | awk '{print $1}'", shell=True).decode().strip()
    print(f"\nStarting TVT Visualizer at http://{ip}:8051")
    print("Press Ctrl+C to stop\n")
    app.run(debug=False, host='0.0.0.0', port=8051)
