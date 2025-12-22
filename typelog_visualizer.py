#!/usr/bin/env python3
"""
TypeLog vs PseudoTypeLog Visualizer

Визуализатор для сравнения typeLog и pseudoTypeLog с нормализацией.
- Ось Y: TVD (trueVerticalDepth)
- Ось X: data (GR показания)
- Слайдер: множитель для data typeLog (0.3-3.0)

Запуск: python typelog_visualizer.py
Откроется браузер с интерактивным графиком.
"""

import json
import os
from pathlib import Path

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Path to slicing_well.json
STARSTEER_DIR = os.getenv('STARSTEER_DIR', 'E:/Projects/Rogii/ss/2025_3_release_dynamic_CUSTOM_instances/slicer_de')
WELLS_DATA_SUBDIR = os.getenv('WELLS_DATA_SUBDIR', 'AG_DATA/InitialData')
SLICING_WELL_PATH = Path(STARSTEER_DIR) / WELLS_DATA_SUBDIR / 'slicing_well.json'

# Normalization coefficient from MDE for Well1798~EGFDL
# In MDE: well_GR × 1.446594 → normalized
# Here: typeLog_GR / 1.446594 to match normalized well
MDE_MULTIPLIER = 1.446594
DEFAULT_NORM_COEF = 1.0 / MDE_MULTIPLIER  # ≈ 0.691


def load_data():
    """Load typeLog and pseudoTypeLog from slicing_well.json"""
    with open(SLICING_WELL_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Direct access - no fallbacks, will raise KeyError if structure is wrong
    tvd_shift = data['tvdTypewellShift']
    type_log_points = data['typeLog']['tvdSortedPoints']
    pseudo_log_points = data['pseudoTypeLog']['tvdSortedPoints']

    return {
        'typeLog': type_log_points,
        'pseudoTypeLog': pseudo_log_points,
        'tvdTypewellShift': tvd_shift
    }


# Load data once at startup
DATA = load_data()

print(f"Loaded from: {SLICING_WELL_PATH}")
print(f"typeLog: {len(DATA['typeLog'])} points")
print(f"pseudoTypeLog: {len(DATA['pseudoTypeLog'])} points")
print(f"tvdTypewellShift: {DATA['tvdTypewellShift']}")

if DATA['typeLog']:
    tvd_range_type = [p['trueVerticalDepth'] for p in DATA['typeLog']]
    print(f"typeLog TVD range: {min(tvd_range_type):.1f} - {max(tvd_range_type):.1f}")

if DATA['pseudoTypeLog']:
    tvd_range_pseudo = [p['trueVerticalDepth'] for p in DATA['pseudoTypeLog']]
    print(f"pseudoTypeLog TVD range: {min(tvd_range_pseudo):.1f} - {max(tvd_range_pseudo):.1f}")

# Create Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2('TypeLog vs PseudoTypeLog Visualizer'),
    html.Div([
        html.Label('Коэффициент нормализации typeLog: '),
        html.Span(id='coef-display', style={'fontWeight': 'bold'}),
    ]),
    dcc.Slider(
        id='norm-slider',
        min=0.3,
        max=3.0,
        step=0.01,
        value=DEFAULT_NORM_COEF,
        marks={0.3: '0.3', round(DEFAULT_NORM_COEF, 2): f'{DEFAULT_NORM_COEF:.2f} (MDE)', 1.0: '1.0', 2.0: '2.0', 3.0: '3.0'},
        tooltip={'placement': 'bottom', 'always_visible': True}
    ),
    dcc.Graph(
        id='main-graph',
        style={'height': '85vh'},
        config={'scrollZoom': True}
    ),
    html.Div([
        html.P(f"typeLog: {len(DATA['typeLog'])} points, tvdTypewellShift: {DATA['tvdTypewellShift']:.2f}m"),
        html.P(f"pseudoTypeLog: {len(DATA['pseudoTypeLog'])} points"),
    ], style={'fontSize': '12px', 'color': 'gray'})
])


@app.callback(
    [Output('main-graph', 'figure'),
     Output('coef-display', 'children')],
    [Input('norm-slider', 'value')]
)
def update_graph(norm_coef):
    """Update graph when slider changes"""

    fig = go.Figure()

    # pseudoTypeLog (reference, no modification)
    if DATA['pseudoTypeLog']:
        pseudo_tvd = [p['trueVerticalDepth'] for p in DATA['pseudoTypeLog']]
        pseudo_data = [p['data'] for p in DATA['pseudoTypeLog']]

        fig.add_trace(go.Scatter(
            x=pseudo_data,
            y=pseudo_tvd,
            mode='lines',
            name='pseudoTypeLog',
            line=dict(color='blue', width=1.5),
            hovertemplate='TVD: %{y:.1f}m<br>GR: %{x:.1f}<extra>pseudo</extra>'
        ))

    # typeLog (with TVD shift and data normalization)
    if DATA['typeLog']:
        tvd_shift = DATA['tvdTypewellShift']
        type_tvd = [p['trueVerticalDepth'] + tvd_shift for p in DATA['typeLog']]
        type_data = [p['data'] * norm_coef for p in DATA['typeLog']]

        fig.add_trace(go.Scatter(
            x=type_data,
            y=type_tvd,
            mode='lines',
            name=f'typeLog (×{norm_coef:.2f}, shift +{tvd_shift:.1f}m)',
            line=dict(color='red', width=1.5),
            hovertemplate='TVD: %{y:.1f}m<br>GR: %{x:.1f}<extra>type</extra>'
        ))

    fig.update_layout(
        title=f'TypeLog vs PseudoTypeLog (коэф. = {norm_coef:.2f})',
        xaxis_title='GR (data)',
        yaxis_title='TVD (m)',
        xaxis=dict(fixedrange=True),  # No zoom on X axis
        yaxis=dict(autorange='reversed'),  # TVD increases downward
        hovermode='closest',
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=60, r=20, t=50, b=40)
    )

    return fig, f'{norm_coef:.2f}'


if __name__ == '__main__':
    print("\nStarting visualizer at http://127.0.0.1:8050")
    print("Press Ctrl+C to stop\n")
    app.run(debug=False, port=8050)
