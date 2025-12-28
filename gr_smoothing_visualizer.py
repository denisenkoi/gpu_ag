#!/usr/bin/env python3
"""
GR Smoothing Visualizer - Multi-Filter

Визуализатор для сравнения разных фильтров сглаживания GR.
Фильтры: Gaussian, Savitzky-Golay, Moving Average, Median, Butterworth

Запуск: python gr_smoothing_visualizer.py
"""

import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter, butter, filtfilt
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import load_dotenv

load_dotenv()

STARSTEER_DIR = os.getenv('STARSTEER_DIR', 'E:/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de')
WELLS_DATA_SUBDIR = os.getenv('WELLS_DATA_SUBDIR', 'AG_DATA/InitialData')
SLICING_WELL_PATH = Path(STARSTEER_DIR) / WELLS_DATA_SUBDIR / 'slicing_well.json'


def load_data():
    """Load wellLog from slicing_well.json"""
    with open(SLICING_WELL_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    points = data['wellLog']['tvdSortedPoints']
    valid_points = [p for p in points if p['data'] is not None]

    md = np.array([p['measuredDepth'] for p in valid_points])
    gr = np.array([p['data'] for p in valid_points])

    sort_idx = np.argsort(md)
    return md[sort_idx], gr[sort_idx]


def apply_filter(gr, filter_type, param1, param2):
    """Apply selected filter to GR data"""
    n = len(gr)

    if filter_type == 'none':
        return gr.copy()

    elif filter_type == 'gaussian':
        sigma = param1
        if sigma <= 0:
            return gr.copy()
        return gaussian_filter1d(gr, sigma=sigma)

    elif filter_type == 'savgol':
        window = int(param1)
        order = int(param2)
        # Window must be odd and > order
        if window % 2 == 0:
            window += 1
        window = max(window, order + 2)
        window = min(window, n - 1)
        if window % 2 == 0:
            window -= 1
        return savgol_filter(gr, window, order)

    elif filter_type == 'moving_avg':
        window = int(param1)
        if window <= 1:
            return gr.copy()
        kernel = np.ones(window) / window
        # Pad to avoid edge effects
        padded = np.pad(gr, (window//2, window//2), mode='edge')
        smoothed = np.convolve(padded, kernel, mode='valid')
        return smoothed[:n]

    elif filter_type == 'median':
        window = int(param1)
        if window <= 1:
            return gr.copy()
        if window % 2 == 0:
            window += 1
        return median_filter(gr, size=window)

    elif filter_type == 'butterworth':
        cutoff = param1  # 0.01 - 0.5
        order = int(param2)
        if cutoff <= 0 or cutoff >= 0.5:
            return gr.copy()
        b, a = butter(order, cutoff, btype='low')
        return filtfilt(b, a, gr)

    return gr.copy()


MD, GR = load_data()

print(f"Loaded: {len(MD)} points, MD: {MD.min():.1f} - {MD.max():.1f} m")

app = dash.Dash(__name__)

FILTERS = {
    'none': 'No filter',
    'gaussian': 'Gaussian (sigma)',
    'savgol': 'Savitzky-Golay (window, order)',
    'moving_avg': 'Moving Average (window)',
    'median': 'Median (window)',
    'butterworth': 'Butterworth Lowpass (cutoff, order)',
}

app.layout = html.Div([
    html.H2('GR Smoothing Visualizer'),

    html.Div([
        html.Div([
            html.Label('Filter: ', style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='filter-dropdown',
                options=[{'label': v, 'value': k} for k, v in FILTERS.items()],
                value='savgol',
                style={'width': '300px'}
            ),
        ], style={'display': 'inline-block', 'marginRight': '30px'}),

        html.Div([
            html.Label('Param 1: ', style={'fontWeight': 'bold'}),
            html.Span(id='param1-label'),
        ], style={'display': 'inline-block', 'marginRight': '10px'}),

        html.Div([
            html.Label('Param 2: ', style={'fontWeight': 'bold'}),
            html.Span(id='param2-label'),
        ], style={'display': 'inline-block'}),
    ], style={'marginBottom': '10px'}),

    html.Div([
        html.Div([
            html.Label('Param 1:'),
            dcc.Slider(
                id='param1-slider',
                min=0.1,
                max=50,
                step=0.1,
                value=11,
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
        ], style={'width': '45%', 'display': 'inline-block', 'marginRight': '5%'}),

        html.Div([
            html.Label('Param 2:'),
            dcc.Slider(
                id='param2-slider',
                min=1,
                max=10,
                step=1,
                value=3,
                tooltip={'placement': 'bottom', 'always_visible': True}
            ),
        ], style={'width': '45%', 'display': 'inline-block'}),
    ]),

    dcc.Graph(id='combined-graph', style={'height': '40vh'}, config={'scrollZoom': True}),
    dcc.Graph(id='separate-graphs', style={'height': '45vh'}, config={'scrollZoom': True}),

    html.Div(id='stats-display', style={'fontSize': '12px', 'color': 'gray', 'marginTop': '10px'})
])


@app.callback(
    [Output('param1-slider', 'min'),
     Output('param1-slider', 'max'),
     Output('param1-slider', 'step'),
     Output('param1-slider', 'value'),
     Output('param2-slider', 'min'),
     Output('param2-slider', 'max'),
     Output('param2-slider', 'step'),
     Output('param2-slider', 'value'),
     Output('param2-slider', 'disabled')],
    [Input('filter-dropdown', 'value')]
)
def update_sliders(filter_type):
    """Update slider ranges based on filter type"""
    if filter_type == 'none':
        return 0, 1, 1, 0, 1, 1, 1, 1, True
    elif filter_type == 'gaussian':
        return 0.1, 20, 0.1, 2, 1, 1, 1, 1, True
    elif filter_type == 'savgol':
        return 3, 101, 2, 11, 1, 7, 1, 3, False
    elif filter_type == 'moving_avg':
        return 2, 100, 1, 11, 1, 1, 1, 1, True
    elif filter_type == 'median':
        return 3, 101, 2, 11, 1, 1, 1, 1, True
    elif filter_type == 'butterworth':
        return 0.01, 0.49, 0.01, 0.1, 1, 8, 1, 4, False
    return 0, 1, 1, 0, 1, 1, 1, 1, True


@app.callback(
    [Output('combined-graph', 'figure'),
     Output('separate-graphs', 'figure'),
     Output('param1-label', 'children'),
     Output('param2-label', 'children'),
     Output('stats-display', 'children')],
    [Input('filter-dropdown', 'value'),
     Input('param1-slider', 'value'),
     Input('param2-slider', 'value')]
)
def update_graphs(filter_type, param1, param2):
    """Update graphs when parameters change"""

    gr_smoothed = apply_filter(GR, filter_type, param1, param2)

    # Parameter labels
    param_labels = {
        'none': ('—', '—'),
        'gaussian': (f'σ = {param1}', '—'),
        'savgol': (f'window = {int(param1)}', f'order = {int(param2)}'),
        'moving_avg': (f'window = {int(param1)}', '—'),
        'median': (f'window = {int(param1)}', '—'),
        'butterworth': (f'cutoff = {param1:.2f}', f'order = {int(param2)}'),
    }
    p1_label, p2_label = param_labels.get(filter_type, ('', ''))

    filter_name = FILTERS.get(filter_type, filter_type)

    # Combined graph
    fig_combined = go.Figure()

    fig_combined.add_trace(go.Scatter(
        x=MD, y=GR,
        mode='lines', name='Original',
        line=dict(color='blue', width=1), opacity=0.6
    ))

    fig_combined.add_trace(go.Scatter(
        x=MD, y=gr_smoothed,
        mode='lines', name=f'Filtered',
        line=dict(color='red', width=2)
    ))

    fig_combined.update_layout(
        title=f'{filter_name}: {p1_label} {p2_label}'.strip(),
        xaxis_title='MD (m)', yaxis_title='GR',
        hovermode='x unified',
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=20, t=50, b=40)
    )

    # Separate graphs
    fig_separate = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                                  subplot_titles=('Original GR', f'Filtered: {filter_name}'))

    fig_separate.add_trace(go.Scatter(x=MD, y=GR, mode='lines', line=dict(color='blue', width=1)), row=1, col=1)
    fig_separate.add_trace(go.Scatter(x=MD, y=gr_smoothed, mode='lines', line=dict(color='red', width=1.5)), row=2, col=1)

    y_min = min(GR.min(), gr_smoothed.min())
    y_max = max(GR.max(), gr_smoothed.max())
    y_margin = (y_max - y_min) * 0.05

    fig_separate.update_yaxes(range=[y_min - y_margin, y_max + y_margin])
    fig_separate.update_xaxes(title_text='MD (m)', row=2, col=1)
    fig_separate.update_layout(hovermode='x unified', showlegend=False, margin=dict(l=60, r=20, t=40, b=40))

    # Stats
    diff = np.abs(GR - gr_smoothed)
    stats = f"Points: {len(MD)} | Diff: mean={diff.mean():.2f}, max={diff.max():.2f} | Smoothed range: {gr_smoothed.min():.1f} - {gr_smoothed.max():.1f}"

    return fig_combined, fig_separate, p1_label, p2_label, stats


if __name__ == '__main__':
    print(f"\nGR Smoothing Visualizer: http://127.0.0.1:8051")
    print("Filters: Gaussian, Savitzky-Golay, Moving Average, Median, Butterworth")
    print("Press Ctrl+C to stop\n")
    app.run(debug=False, port=8051)
