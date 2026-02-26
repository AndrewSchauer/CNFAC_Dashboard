# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25

@author: andrew schauer/CNFAC
"""
import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json
import copy
import numpy as np

app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.DARKLY,
    "https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&display=swap"
])
app.title = "CMAH Dashboard"
server = app.server

# Inject CSS to style danger cell dropdowns
app.index_string = app.index_string.replace(
    "</head>",
    """<style>
/* Remove default Dash dropdown chrome on danger cells */
.danger-cell-dropdown .Select-control {
    background-color: inherit !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 0 !important;
    height: 38px !important;
    min-height: 38px !important;
    cursor: pointer !important;
}
.danger-cell-dropdown .Select-value-label,
.danger-cell-dropdown .Select-placeholder {
    color: inherit !important;
    font-family: "Barlow Condensed", sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    line-height: 38px !important;
    padding: 0 4px !important;
}
.danger-cell-dropdown .Select-arrow-zone { display: none !important; }
.danger-cell-dropdown .Select-clear-zone { display: none !important; }
.danger-cell-dropdown .Select-input { display: none !important; }
.danger-cell-dropdown .Select--single > .Select-control .Select-value { padding: 0 4px !important; }
/* Dropdown menu panel */
.danger-cell-dropdown .Select-menu-outer {
    background-color: #0d1b2a !important;
    border: 1px solid #1e3a4a !important;
    border-radius: 0 !important;
    z-index: 9999 !important;
    width: 160px !important;
    min-width: 160px !important;
}
.danger-cell-dropdown .Select-option {
    background-color: #0d1b2a !important;
    color: #ccc !important;
    font-family: "Barlow Condensed", sans-serif !important;
    font-size: 12px !important;
    padding: 6px 10px !important;
}
.danger-cell-dropdown .Select-option:hover,
.danger-cell-dropdown .Select-option.is-focused {
    background-color: #1e3a4a !important;
}
.danger-cell-dropdown .VirtualizedSelectFocusedOption {
    background-color: #1e3a4a !important;
}
/* Hide the filled range bar on the sensitivity and distribution point sliders */
#sens-slider .dash-slider-range,
#dist-slider .dash-slider-range {
    background: transparent !important;
}
#sens-slider .dash-slider-track,
#dist-slider .dash-slider-track {
    background-color: #1e3a4a !important;
}
/* Responsive: ensure graphs fill container on all screen sizes */
.js-plotly-plot, .plotly, .plot-container {
    width: 100% !important;
}
/* Mobile padding adjustment */
@media (max-width: 768px) {
    .dash-graph { width: 100% !important; }
    .card-body { padding: 10px !important; }
}
</style>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>"""
)

# ─── Constants ────────────────────────────────────────────────────────────────

# Full labels (indices 0,2,4,6) and half-step labels (indices 1,3,5)
SENSITIVITY_LABELS  = ["Unreactive", "Stubborn", "Reactive", "Touchy"]
SENSITIVITY_SLIDER_LABELS = [
    "Unreactive",
    "Unr.–Stub.",
    "Stubborn",
    "Stub.–React.",
    "Reactive",
    "React.–Touchy",
    "Touchy",
]
# Map slider index → effective sensitivity index (0-3) for matrix lookup
def slider_to_sens(v):
    return v / 2.0  # 0,0.5,1,1.5,2,2.5,3
DISTRIBUTION_LABELS = ["Isolated", "Specific", "Widespread"]
DISTRIBUTION_SLIDER_LABELS = [
    "Isolated",
    "Isol.–Specific",
    "Specific",
    "Spec.–Widespread",
    "Widespread",
]
def slider_to_dist(v):
    return v / 2.0  # 0,0.5,1,1.5,2

# 10 size steps
SIZE_LABELS = ["1", "1.5", "2", "2.5", "3", "3.5", "4", "4.5", "5"]

# 10 likelihood steps (5 named + 4 intermediates + 1 extreme)
LIKELIHOOD_LABELS = [
    "Unlikely",
    "Unlikely–Possible",
    "Possible",
    "Possible–Likely",
    "Likely",
    "Likely–Very Likely",
    "Very Likely",
    "Very Likely–Almost Certain",
    "Almost Certain",
]

# Likelihood matrix from the uploaded image:
# rows = Distribution (0=Isolated, 1=Specific, 2=Widespread)
# cols = Sensitivity  (0=Unreactive, 1=Stubborn, 2=Reactive, 3=Touchy)
# Values map to indices in LIKELIHOOD_LABELS (using steps 0,2,4,6,8 = the named levels)
LIKELIHOOD_MATRIX = [
    [0, 0, 2, 4],   # Isolated:    Unlikely, Unlikely, Possible, Likely
    [0, 2, 4, 6],   # Specific:    Unlikely, Possible, Likely,   Very Likely
    [0, 2, 6, 8],   # Widespread:  Unlikely, Possible, Very Likely, Almost Certain
]

# Official avalanche danger colors
DANGER_COLORS = {
    "No Rating":    "#444444",
    "Low":          "#50B848",
    "Moderate":     "#FFF200",
    "Considerable": "#F7941E",
    "High":         "#ED1C24",
    "Extreme":      "#231F20",
}
DANGER_TEXT = {
    "No Rating":    "#cccccc",
    "Low":          "#111111",
    "Moderate":     "#111111",
    "Considerable": "#111111",
    "High":         "#ffffff",
    "Extreme":      "#ffffff",
}
DANGER_LEVELS = ["No Rating", "Low", "Moderate", "Considerable", "High", "Extreme"]

DANGER_ABBREV = {
    "No Rating": "—", "Low": "Low", "Moderate": "Mod",
    "Considerable": "Con", "High": "High", "Extreme": "Ext"
}


def _build_default_grid():
    """9×9 danger grid: rows=Likelihood (0-8), cols=Size (0-8)."""
    return [
        ["Low","Low","Low","Low","Low","Low","Low","Low","Low"],
        ["Low","Low","Low","Low","Moderate","Moderate","Considerable","Considerable","Considerable"],
        ["Low","Low","Moderate","Moderate","Moderate","Considerable","Considerable","High","High"],
        ["Low","Low","Moderate","Moderate","Considerable","Considerable","High","High","Extreme"],
        ["Low","Moderate","Moderate","Considerable","Considerable","High","High","Extreme","Extreme"],
        ["Low","Moderate","Considerable","Considerable","High","High","Extreme","Extreme","Extreme"],
        ["Low","Moderate","Considerable","High","High","Extreme","Extreme","Extreme","Extreme"],
        ["Low","Moderate","High","High","Extreme","Extreme","Extreme","Extreme","Extreme"],
        ["Low","Moderate","High","High","Extreme","Extreme","Extreme","Extreme","Extreme"],
    ]

DEFAULT_DANGER_GRID = _build_default_grid()


# ─── Figure builders ──────────────────────────────────────────────────────────

def rounded_rect_path(x0, y0, x1, y1, r=0.12):
    """Return an SVG path string for a rounded rectangle in data coordinates."""
    # Clamp r so it doesn't exceed half the box size
    r = min(r, abs(x1 - x0) / 2.0, abs(y1 - y0) / 2.0)
    p = (
        f"M {x0+r},{y0} "
        f"L {x1-r},{y0} "
        f"Q {x1},{y0} {x1},{y0+r} "
        f"L {x1},{y1-r} "
        f"Q {x1},{y1} {x1-r},{y1} "
        f"L {x0+r},{y1} "
        f"Q {x0},{y1} {x0},{y1-r} "
        f"L {x0},{y0+r} "
        f"Q {x0},{y0} {x0+r},{y0} Z"
    )
    return p


def build_likelihood_figure(sf, df):
    """
    sf = sensitivity float 0.0-3.0, df = distribution float 0.0-2.0.
    Numeric axes so add_shape works for any position including half-steps.
    """
    z = np.array(LIKELIHOOD_MATRIX, dtype=float)
    colorscale = [
        [0.00, "#2a2a2a"],
        [0.33, "#5a5a5a"],
        [0.66, "#9a9a9a"],
        [1.00, "#d8d8d8"],
    ]
    # Numeric positions for each cell centre
    x_vals = list(range(len(SENSITIVITY_LABELS)))   # [0,1,2,3]
    y_vals = list(range(len(DISTRIBUTION_LABELS)))  # [0,1,2]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x=x_vals, y=y_vals,
        colorscale=colorscale, zmin=0, zmax=8,
        showscale=False, hoverinfo="skip",
    ))

    cell_text = [
        ["Unlikely",  "Unlikely", "Possible",    "Likely"],
        ["Unlikely",  "Possible", "Likely",       "Very Likely"],
        ["Unlikely",  "Possible", "Very Likely",  "Almost Certain"],
    ]
    # Use dark text on lighter cells (higher likelihood values), light on dark
    for ri in range(3):
        for ci in range(4):
            val = LIKELIHOOD_MATRIX[ri][ci]  # 0-8
            text_color = "#111111" if val >= 6 else "#ffffff"
            fig.add_annotation(
                x=ci, y=ri,
                text=cell_text[ri][ci], showarrow=False,
                font=dict(size=12, color=text_color, family="Barlow Condensed"),
            )

    if sf is not None and df is not None:
        import math
        # Draw a box covering the cell(s) touched by this point.
        # On a half-step, the point sits on a boundary so we highlight both neighbours.
        s_lo = math.floor(sf); s_hi = math.ceil(sf)
        d_lo = math.floor(df); d_hi = math.ceil(df)
        fig.add_shape(
            type="path",
            path=rounded_rect_path(s_lo - 0.45, d_lo - 0.45, s_hi + 0.45, d_hi + 0.45, r=0.15),
            line=dict(color="#00e5ff", width=3),
            fillcolor="rgba(0, 229, 255, 0.18)",
        )
        # Always draw a crosshair at the exact point
        fig.add_trace(go.Scatter(
            x=[sf], y=[df],
            mode="markers",
            marker=dict(symbol="cross", size=16, color="#00e5ff",
                        line=dict(width=3, color="#00e5ff")),
            hoverinfo="skip", showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        margin=dict(l=10, r=10, t=10, b=40),
        dragmode="drawrect",
        newshape=dict(
            line=dict(color="#00e5ff", width=3),
            fillcolor="rgba(0, 229, 255, 0.18)",
            opacity=1,
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=x_vals,
            ticktext=SENSITIVITY_LABELS,
            tickfont=dict(family="Barlow Condensed", color="#bbb", size=12),
            title=dict(text="Sensitivity to Triggers",
                       font=dict(family="Barlow Condensed", color="#888", size=11)),
            range=[-0.5, len(SENSITIVITY_LABELS) - 0.5],
            gridcolor="#1e2d3d", showline=False, zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=y_vals,
            ticktext=DISTRIBUTION_LABELS,
            tickfont=dict(family="Barlow Condensed", color="#bbb", size=12),
            title=dict(text="Spatial Distribution",
                       font=dict(family="Barlow Condensed", color="#888", size=11)),
            range=[-0.5, len(DISTRIBUTION_LABELS) - 0.5],
            gridcolor="#1e2d3d", showline=False, zeroline=False,
            fixedrange=True,
        ),
        height=None,
        width=None,
        autosize=True,
    )
    return fig


def build_danger_figure(lik_range, size_range, danger_grid):
    z, text = [], []
    for r in range(9):
        row_z, row_t = [], []
        for c in range(9):
            d = danger_grid[r][c]
            row_z.append(DANGER_LEVELS.index(d))
            row_t.append(d)
        z.append(row_z)
        text.append(row_t)

    color_vals = [DANGER_COLORS[d] for d in DANGER_LEVELS]
    colorscale  = [[i / 5, color_vals[i]] for i in range(6)]

    x_vals = list(range(len(SIZE_LABELS)))       # [0..8]
    y_vals = list(range(len(LIKELIHOOD_LABELS))) # [0..8]

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=z, x=x_vals, y=y_vals,
        colorscale=colorscale, zmin=0, zmax=5,
        showscale=False, text=text,
        hovertemplate="Size: %{x}<br>Likelihood: %{y}<br>Danger: %{text}<extra></extra>",
    ))

    # No text labels in the danger matrix — colour alone conveys the level

    if lik_range and size_range:
        l0, l1 = lik_range
        s0, s1 = size_range
        pad_s = 0.49 if s0 != s1 else 0.42
        pad_l = 0.49 if l0 != l1 else 0.42
        fig.add_shape(
            type="path",
            path=rounded_rect_path(s0 - pad_s, l0 - pad_l, s1 + pad_s, l1 + pad_l, r=0.2),
            line=dict(color="#00e5ff", width=3),
            fillcolor="rgba(0, 229, 255, 0.18)",
        )

    fig.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        margin=dict(l=10, r=10, t=10, b=40),
        dragmode=False,
        xaxis=dict(
            tickmode="array",
            tickvals=x_vals,
            ticktext=SIZE_LABELS,
            tickfont=dict(family="Barlow Condensed", color="#bbb", size=11),
            title=dict(text="Destructive Size",
                       font=dict(family="Barlow Condensed", color="#888", size=11)),
            range=[-0.5, len(SIZE_LABELS) - 0.5],
            gridcolor="#1e2d3d", showline=False, zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 2, 4, 6, 8],
            ticktext=["Unlikely", "Possible", "Likely", "Very Likely", "Almost Certain"],
            tickfont=dict(family="Barlow Condensed", color="#bbb", size=10),
            title=dict(text="Likelihood",
                       font=dict(family="Barlow Condensed", color="#888", size=11)),
            range=[-0.5, len(LIKELIHOOD_LABELS) - 0.5],
            gridcolor="#1e2d3d", showline=False, zeroline=False,
            fixedrange=True,
        ),
        height=None,
        width=None,
        autosize=True,
    )
    return fig


# ─── Settings grid dropdowns ──────────────────────────────────────────────────

# Dropdown options with colour swatches rendered via HTML
DROPDOWN_OPTIONS = [
    {
        "label": html.Span(
            [
                html.Span(style={
                    "display": "inline-block", "width": "12px", "height": "12px",
                    "backgroundColor": DANGER_COLORS[d], "marginRight": "6px",
                    "border": "1px solid #444", "verticalAlign": "middle",
                }),
                html.Span(d, style={"fontFamily": "Barlow Condensed", "fontSize": "12px"}),
            ]
        ),
        "value": d,
    }
    for d in DANGER_LEVELS
]


def make_danger_grid_buttons(grid):
    rows = []
    for r in range(8, -1, -1):
        cells = [html.Div(
            LIKELIHOOD_LABELS[r],
            style={
                "color": "#aaa", "fontSize": "10px", "fontFamily": "Barlow Condensed",
                "width": "130px", "textAlign": "right", "paddingRight": "8px",
                "display": "flex", "alignItems": "center", "justifyContent": "flex-end",
                "flexShrink": "0",
            }
        )]
        for c in range(9):
            danger = grid[r][c]
            bg     = DANGER_COLORS[danger]
            fg     = DANGER_TEXT[danger]
            cells.append(html.Div(
                dcc.Dropdown(
                    id={"type": "grid-cell", "row": r, "col": c},
                    options=DROPDOWN_OPTIONS,
                    value=danger,
                    clearable=False,
                    searchable=False,
                    style={
                        "backgroundColor": bg,
                        "color": fg,
                        "border": "none",
                        "width": "58px",
                        "fontSize": "10px",
                        "fontFamily": "Barlow Condensed",
                        "fontWeight": "600",
                        "minHeight": "38px",
                    },
                    className="danger-cell-dropdown",
                ),
                style={"width": "58px", "flexShrink": "0"}
            ))
        rows.append(html.Div(cells, style={"display": "flex", "marginBottom": "2px"}))

    size_row = html.Div(
        [html.Div("", style={"width": "138px", "flexShrink": "0"})] +
        [html.Div(s, style={
            "width": "58px", "textAlign": "center", "color": "#aaa",
            "fontSize": "10px", "fontFamily": "Barlow Condensed", "flexShrink": "0"
        }) for s in SIZE_LABELS],
        style={"display": "flex", "marginTop": "4px"}
    )
    return html.Div([
        html.Div(rows), size_row,
        html.Div("← SIZE →", style={
            "textAlign": "center", "color": "#555", "fontSize": "10px",
            "marginTop": "4px", "fontFamily": "Barlow Condensed", "letterSpacing": "0.15em"
        })
    ])


# ─── Sliders ──────────────────────────────────────────────────────────────────

def make_range_slider(id, labels, default, half_labels=None):
    """If half_labels provided, uses those for marks with step=1 over doubled range."""
    if half_labels:
        marks = {i: {"label": l, "style": {"color": "#aaa", "fontFamily": "Barlow Condensed", "fontSize": "10px"}}
                 for i, l in enumerate(half_labels)}
        return dcc.RangeSlider(
            id=id, min=0, max=len(half_labels) - 1, step=1,
            value=[v * 2 for v in default], marks=marks, allowCross=False,
            tooltip={"always_visible": False},
        )
    marks = {i: {"label": l, "style": {"color": "#aaa", "fontFamily": "Barlow Condensed", "fontSize": "11px"}}
             for i, l in enumerate(labels)}
    return dcc.RangeSlider(
        id=id, min=0, max=len(labels) - 1, step=1,
        value=default, marks=marks, allowCross=False,
        tooltip={"always_visible": False},
    )


# ─── Layout ───────────────────────────────────────────────────────────────────

lbl = {"fontFamily": "Barlow Condensed", "fontWeight": "700", "fontSize": "13px",
       "color": "#00e5ff", "letterSpacing": "0.12em", "marginBottom": "6px"}
card = {"backgroundColor": "#0d1b2a", "border": "1px solid #1e3a4a", "marginBottom": "14px"}

def make_point_slider(id, half_labels, default_idx):
    """Single-handle slider — full labels on named steps, empty label on half-steps."""
    marks = {}
    for i, l in enumerate(half_labels):
        if i % 2 == 0:
            # Named step — show label
            marks[i] = {"label": l, "style": {"color": "#aaa", "fontFamily": "Barlow Condensed", "fontSize": "10px"}}
        else:
            # Half-step — show a small tick but no text
            marks[i] = {"label": "", "style": {"color": "transparent"}}
    return dcc.Slider(
        id=id, min=0, max=len(half_labels) - 1, step=1,
        value=default_idx, marks=marks,
        tooltip={"always_visible": False},
    )

controls = dbc.Card(dbc.CardBody([
    html.Div("DISTRIBUTION", style=lbl),
    make_point_slider("dist-slider", DISTRIBUTION_SLIDER_LABELS, 1),
    html.Div(style={"height": "26px"}),
    html.Div("SENSITIVITY", style=lbl),
    make_point_slider("sens-slider", SENSITIVITY_SLIDER_LABELS, 2),
    html.Div(style={"height": "26px"}),
    html.Div("SIZE", style=lbl),
    make_range_slider("size-slider", SIZE_LABELS, [1, 4]),
]), style=card)

forecast_tab = dbc.Row([
    # Left col: matrices (order 2 on mobile, order 1 on desktop)
    dbc.Col([
        dbc.Card(dbc.CardBody([
            html.Div("LIKELIHOOD MATRIX", style=lbl),
            dcc.Graph(id="likelihood-matrix", config={
                "displayModeBar": True,
                "modeBarButtonsToAdd": ["drawrect", "eraseshape"],
                "modeBarButtonsToRemove": ["zoom2d","pan2d","zoomIn2d","zoomOut2d",
                                           "autoScale2d","resetScale2d","toImage"],
                "editable": False,
            }, style={"width": "100%"}),
        ]), style=card),
        dbc.Card(dbc.CardBody([
            html.Div("DANGER MATRIX", style=lbl),
            dcc.Graph(id="danger-matrix", config={"displayModeBar": False},
                      style={"width": "100%", "aspectRatio": "1 / 1"}),
        ]), style=card),
    ], xs=12, md=8, className="order-2 order-md-1"),
    # Right col: sliders + summary + image (order 1 on mobile, order 2 on desktop)
    dbc.Col([
        controls,
        dbc.Card(dbc.CardBody([
            html.Div("FORECAST SUMMARY", style=lbl),
            html.Div(id="forecast-summary"),
        ]), style=card),
        html.Img(
            src="https://raw.githubusercontent.com/AndrewSchauer/CNFAC_Dashboard/main/NAPADS.png",
            style={"width": "100%", "marginTop": "4px", "borderRadius": "4px", "opacity": "0.9"},
        ),
    ], xs=12, md=4, className="order-1 order-md-2"),
], className="flex-wrap")

settings_tab = html.Div([
    html.Div("CONFIGURE DANGER GRID", style={**lbl, "fontSize": "15px"}),
    html.P("Click any cell to cycle: No Rating → Low → Moderate → Considerable → High → Extreme → …",
           style={"color": "#777", "fontFamily": "Barlow Condensed", "fontSize": "12px", "marginBottom": "14px"}),
    html.Div([
        html.Div([
            html.Div(style={"width": "16px", "height": "16px", "backgroundColor": DANGER_COLORS[d],
                            "display": "inline-block", "marginRight": "6px", "border": "1px solid #333"}),
            html.Span(d, style={"fontFamily": "Barlow Condensed", "color": "#ccc", "fontSize": "12px"})
        ], style={"display": "flex", "alignItems": "center", "marginRight": "16px"})
        for d in DANGER_LEVELS
    ], style={"display": "flex", "flexWrap": "wrap", "gap": "6px", "marginBottom": "18px"}),
    html.Div([
        html.Div("← LIKELIHOOD →", style={
            "writingMode": "vertical-rl", "transform": "rotate(180deg)",
            "color": "#555", "fontFamily": "Barlow Condensed", "letterSpacing": "0.15em",
            "fontSize": "10px", "marginRight": "6px", "display": "flex", "alignItems": "center"
        }),
        html.Div(id="danger-grid-container", style={"overflowX": "auto"}),
    ], style={"display": "flex"}),
    html.Div(style={"height": "18px"}),
    dbc.Button("Reset to Defaults", id="reset-grid-btn", color="secondary", size="sm",
               style={"fontFamily": "Barlow Condensed"}),
])

app.layout = html.Div([
    dcc.Store(id="danger-grid-store", data=DEFAULT_DANGER_GRID),

    html.Div([
        html.Div([
            html.Div([
                html.Span("AVALANCHE FORECAST DASHBOARD", style={
                    "fontFamily": "Barlow Condensed", "fontWeight": "700",
                    "fontSize": "22px", "letterSpacing": "0.25em", "color": "#ffffff"
                }),
                html.Div("RISK ASSESSMENT TOOL", style={
                    "fontFamily": "Share Tech Mono", "fontSize": "10px",
                    "color": "#00e5ff", "letterSpacing": "0.2em", "marginTop": "2px"
                }),
            ]),
            html.Img(
                src="https://raw.githubusercontent.com/AndrewSchauer/CNFAC_Dashboard/main/CNFAC_Logo.png",
                style={"height": "50px"}
            ),
        ], style={"display": "flex", "alignItems": "center", "justifyContent": "space-between", "width": "100%"}),
    ], style={"backgroundColor": "#060e1a", "borderBottom": "1px solid #1e3a4a", "padding": "14px 24px 10px"}),

    dbc.Tabs([
        dbc.Tab(
            html.Div(forecast_tab, style={"padding": "18px"}),
            label="FORECAST", tab_id="forecast",
            label_style={"fontFamily": "Barlow Condensed", "letterSpacing": "0.1em", "fontSize": "13px"},
            active_label_style={"color": "#00e5ff", "fontFamily": "Barlow Condensed", "fontSize": "13px"},
        ),
        dbc.Tab(
            html.Div(settings_tab, style={"padding": "18px"}),
            label="SETTINGS", tab_id="settings",
            label_style={"fontFamily": "Barlow Condensed", "letterSpacing": "0.1em", "fontSize": "13px"},
            active_label_style={"color": "#00e5ff", "fontFamily": "Barlow Condensed", "fontSize": "13px"},
        ),
    ], active_tab="forecast",
       style={"backgroundColor": "#060e1a", "borderBottom": "1px solid #1e3a4a"}),

], style={"backgroundColor": "#080f1a", "minHeight": "100vh", "maxWidth": "100vw", "overflowX": "hidden"})


# ─── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output("likelihood-matrix", "figure"),
    Output("danger-matrix", "figure"),
    Output("forecast-summary", "children"),
    Input("sens-slider", "value"),
    Input("dist-slider", "value"),
    Input("size-slider", "value"),
    Input("danger-grid-store", "data"),
)
def update_all(sens_val, dist_val, size_range, danger_grid):
    import math
    # sens_val / dist_val are single half-step indices
    sf = sens_val / 2.0   # float 0.0–3.0
    df = dist_val / 2.0   # float 0.0–2.0
    sz0, sz1 = size_range

    # For matrix lookup, cover both adjacent cells when on a half-step
    s_lo = max(0, min(math.floor(sf), len(SENSITIVITY_LABELS) - 1))
    s_hi = max(0, min(math.ceil(sf),  len(SENSITIVITY_LABELS) - 1))
    d_lo = max(0, min(math.floor(df), len(DISTRIBUTION_LABELS) - 1))
    d_hi = max(0, min(math.ceil(df),  len(DISTRIBUTION_LABELS) - 1))

    lik_vals = [LIKELIHOOD_MATRIX[r][c]
                for r in range(d_lo, d_hi + 1)
                for c in range(s_lo, s_hi + 1)]
    l0, l1 = min(lik_vals), max(lik_vals)

    lik_fig    = build_likelihood_figure(sf, df)
    danger_fig = build_danger_figure([l0, l1], size_range, danger_grid)

    danger_in_box = {danger_grid[r][c] for r in range(l0, l1 + 1) for c in range(sz0, sz1 + 1)}
    max_danger    = max(danger_in_box, key=lambda d: DANGER_LEVELS.index(d))

    def badge(text, d):
        return html.Span(text, style={
            "backgroundColor": DANGER_COLORS[d], "color": DANGER_TEXT[d],
            "padding": "2px 10px", "fontFamily": "Barlow Condensed",
            "fontWeight": "700", "fontSize": "14px", "borderRadius": "3px",
        })

    def row(label, value):
        return html.Div([
            html.Span(label, style={"color": "#666", "fontSize": "12px",
                                    "fontFamily": "Barlow Condensed", "width": "110px", "display": "inline-block"}),
            html.Span(value, style={"color": "#ccc", "fontSize": "12px", "fontFamily": "Barlow Condensed"}),
        ], style={"marginBottom": "6px"})

    def rng(labels, lo, hi):
        return labels[lo] if lo == hi else f"{labels[lo]} → {labels[hi]}"

    summary = html.Div([
        row("Sensitivity:",  SENSITIVITY_SLIDER_LABELS[sens_val]),
        row("Distribution:", DISTRIBUTION_SLIDER_LABELS[dist_val]),
        row("Likelihood:",   rng(LIKELIHOOD_LABELS,   l0, l1)),
        row("Size:",         rng(SIZE_LABELS,          sz0, sz1)),
        html.Hr(style={"borderColor": "#1e3a4a", "margin": "10px 0"}),
        html.Div([
            html.Span("MAX DANGER: ", style={"color": "#888", "fontSize": "12px",
                                             "fontFamily": "Barlow Condensed", "fontWeight": "700", "marginRight": "8px"}),
            badge(max_danger.upper(), max_danger),
        ]),
    ])
    return lik_fig, danger_fig, summary


@app.callback(
    Output("danger-grid-container", "children"),
    Input("danger-grid-store", "data"),
)
def refresh_grid(danger_grid):
    return make_danger_grid_buttons(danger_grid)


@app.callback(
    Output("danger-grid-store", "data"),
    Input({"type": "grid-cell", "row": ALL, "col": ALL}, "value"),
    Input("reset-grid-btn", "n_clicks"),
    State("danger-grid-store", "data"),
    prevent_initial_call=True,
)
def edit_grid(cell_values, reset_clicks, current_grid):
    ctx = callback_context
    if not ctx.triggered:
        return current_grid

    trigger_id = ctx.triggered[0]["prop_id"]

    if "reset-grid-btn" in trigger_id:
        return _build_default_grid()

    try:
        d = json.loads(trigger_id.split(".")[0])
        row, col  = d["row"], d["col"]
        new_value = ctx.triggered[0]["value"]
    except Exception:
        return current_grid

    if new_value not in DANGER_LEVELS:
        return current_grid

    grid = copy.deepcopy(current_grid)
    grid[row][col] = new_value
    return grid


# ─── Drag-to-update: likelihood matrix → sliders ─────────────────────────────

def _snap_to_half(val, max_half_idx):
    """Convert a float axis coordinate to the nearest half-step slider index."""
    import math
    # Each cell is 1 unit wide; half-step idx = round(val * 2)
    half = round(val * 2)
    return max(0, min(half, max_half_idx))

def _snap_to_int(val, max_idx):
    """Snap a float coordinate to the nearest integer grid index."""
    return max(0, min(round(val), max_idx))


@app.callback(
    Output("sens-slider", "value"),
    Output("dist-slider", "value"),
    Input("likelihood-matrix", "relayoutData"),
    State("sens-slider", "value"),
    State("dist-slider", "value"),
    prevent_initial_call=True,
)
def lik_drag_to_sliders(relayout, curr_sens, curr_dist):
    """When user draws/moves a shape on the likelihood matrix, snap sliders."""
    if not relayout:
        return curr_sens, curr_dist

    # Plotly puts drawn shape coords under shapes[N].x0 etc. or
    # relayout directly contains 'shapes[0].x0' style keys after a drag.
    x0 = x1 = y0 = y1 = None

    # Check for freshly drawn shape
    for key in ["shapes[0].x0", "shapes[0].x1", "shapes[0].y0", "shapes[0].y1"]:
        pass  # just checking structure

    if "shapes[0].x0" in relayout:
        x0 = relayout.get("shapes[0].x0")
        x1 = relayout.get("shapes[0].x1")
        y0 = relayout.get("shapes[0].y0")
        y1 = relayout.get("shapes[0].y1")
    elif "shapes" in relayout and len(relayout["shapes"]) > 0:
        s = relayout["shapes"][-1]
        x0, x1 = s.get("x0"), s.get("x1")
        y0, y1 = s.get("y0"), s.get("y1")

    if None in (x0, x1, y0, y1):
        return curr_sens, curr_dist

    # Ensure x0 < x1, y0 < y1
    if x0 > x1: x0, x1 = x1, x0
    if y0 > y1: y0, y1 = y1, y0

    # Use centre of drawn box to set the point sliders
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0

    sens_max = len(SENSITIVITY_SLIDER_LABELS) - 1   # 6
    dist_max = len(DISTRIBUTION_SLIDER_LABELS) - 1  # 4

    new_sens = _snap_to_half(cx, sens_max)
    new_dist = _snap_to_half(cy, dist_max)

    return new_sens, new_dist


@app.callback(
    Output("size-slider", "value"),
    Input("danger-matrix", "relayoutData"),
    State("size-slider", "value"),
    prevent_initial_call=True,
)
def danger_drag_to_sliders(relayout, curr_size):
    """When user draws/moves a shape on the danger matrix, update size slider."""
    if not relayout:
        return curr_size

    x0 = x1 = None

    if "shapes[0].x0" in relayout:
        x0 = relayout.get("shapes[0].x0")
        x1 = relayout.get("shapes[0].x1")
    elif "shapes" in relayout and len(relayout["shapes"]) > 0:
        s = relayout["shapes"][-1]
        x0, x1 = s.get("x0"), s.get("x1")

    if None in (x0, x1):
        return curr_size

    if x0 > x1: x0, x1 = x1, x0

    size_max = len(SIZE_LABELS) - 1  # 8
    new_s0 = _snap_to_int(x0 + 0.5, size_max)
    new_s1 = _snap_to_int(x1 - 0.5, size_max)
    if new_s0 > new_s1: new_s0, new_s1 = new_s1, new_s0

    return [new_s0, new_s1]


if __name__ == "__main__":
    import os
    # Use 0.0.0.0 on Render (or any cloud host), 127.0.0.1 locally on Windows
    host = "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1"
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host=host, port=port, jupyter_mode="external")