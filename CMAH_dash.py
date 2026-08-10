# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25

@author: andrew schauer/CNFAC
"""
import os
import urllib.error
import urllib.request

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json
import copy
import numpy as np

# The danger-matrix background is an empirical grid derived from historical
# NAC forecast data, pooled across all avalanche problem types (produced by
# build_danger_backgrounds.py, then pushed to GitHub). Falls back to a
# hand-curated grid below if the fetch fails for any reason (offline, repo
# renamed, GitHub down, etc.) so the dashboard still launches.
_GRID_URL = "https://raw.githubusercontent.com/AndrewSchauer/CNFAC_Dashboard/main/empirical_danger_grid.json"

app = dash.Dash(__name__, external_stylesheets=[
    dbc.themes.DARKLY,
    "https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow+Condensed:wght@300;400;600;700&display=swap"
], suppress_callback_exceptions=True)
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
#dist-slider .dash-slider-range,
#sens-slider .dash-slider-track,
#dist-slider .dash-slider-track {
    background-color: #1e3a4a !important;
}
/* Center graphs on desktop */
#likelihood-matrix, #danger-matrix { display: block; margin: 0 auto; }
/* Mobile: scale entire graph container down to fit screen */
@media (max-width: 767px) {
    #likelihood-matrix, #danger-matrix {
        width: 100% !important;
        overflow: hidden;
    }
    #likelihood-matrix > div, #danger-matrix > div,
    #likelihood-matrix .js-plotly-plot, #danger-matrix .js-plotly-plot,
    #likelihood-matrix .plot-container, #danger-matrix .plot-container {
        width: 100% !important;
    }
    #likelihood-matrix .main-svg, #danger-matrix .main-svg {
        width: 100% !important;
        height: auto !important;
    }
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
# The source data-entry tool never distinguished sizes above 4 -- nothing
# is ever recorded at 4.5 or 5 -- so those labels collapse into one "4+"
# bucket rather than showing two permanently-empty trailing columns.
SIZE_LABELS = ["1", "1.5", "2", "2.5", "3", "3.5", "4+"]

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

# The danger (size x likelihood) matrix uses only the 5 named likelihood
# levels -- no in-between rows -- since that's what the historical data
# actually supports. LIKELIHOOD_MATRIX above still emits 0,2,4,6,8; divide
# by 2 to get the compact row index (0-4) into a danger grid.
DANGER_ROW_LABELS = ["Unlikely", "Possible", "Likely", "Very Likely", "Almost Certain"]

def lik_val_to_row(v):
    return v // 2

# Size columns: last one ("4+") renders twice the width of the rest, since it
# represents everything the source tool ever recorded at 4 and above (nothing
# was ever recorded at 4.5 or 5) rather than one 0.5-wide size step.
SIZE_COL_WIDTHS = [1, 1, 1, 1, 1, 1, 2]
_size_col_bounds = [0]
for _w in SIZE_COL_WIDTHS:
    _size_col_bounds.append(_size_col_bounds[-1] + _w)
SIZE_COL_CENTERS = [(_size_col_bounds[i] + _size_col_bounds[i + 1]) / 2 for i in range(len(SIZE_COL_WIDTHS))]
SIZE_AXIS_MAX = _size_col_bounds[-1]

def size_col_bounds(idx):
    """(left, right) x-boundary of size column idx in the widened axis."""
    return _size_col_bounds[idx], _size_col_bounds[idx + 1]

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

# Background the danger-matrix cells blend toward, matching the card's own
# paper/plot background (see build_danger_figure's update_layout below) so a
# low-count cell fades into the surrounding UI instead of standing out.
_CELL_BG_HEX = "#0d1b2a"

def _hex_to_rgb01(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

def _rgb01_to_hex(rgb):
    return '#' + ''.join(f'{max(0, min(255, round(v * 255))):02x}' for v in rgb)

def _blend_cell_color(counts):
    """Port of build_danger_backgrounds.py's blend_cell_color: blends the
    top-2 most common danger levels by their relative share, then fades the
    result toward the card background based on total observation count (few
    observations -> mostly background; many -> mostly the blended color)."""
    bg = _hex_to_rgb01(_CELL_BG_HEX)
    if not counts:
        return bg
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:2]
    total = sum(counts.values())
    if len(items) == 1:
        rgb = _hex_to_rgb01(DANGER_COLORS[DANGER_LEVELS[int(items[0][0])]])
    else:
        (lvl1, n1), (lvl2, n2) = items
        s = n1 + n2
        w1, w2 = n1 / s, n2 / s
        rgb1 = _hex_to_rgb01(DANGER_COLORS[DANGER_LEVELS[int(lvl1)]])
        rgb2 = _hex_to_rgb01(DANGER_COLORS[DANGER_LEVELS[int(lvl2)]])
        rgb = tuple(w1 * a + w2 * b for a, b in zip(rgb1, rgb2))
    alpha = min(max(1 - 1 / (2 * total), 0.0), 1.0)
    return tuple(alpha * a + (1 - alpha) * b for a, b in zip(rgb, bg))


def _build_manual_grid():
    """5x7 danger grid: rows=named Likelihood levels (0=Unlikely..4=Almost
    Certain), cols=Size (0-6, last is the merged "4+" bucket). Hand-curated
    grid, kept as the fallback if empirical_danger_grids.json (produced by
    build_danger_backgrounds.py) isn't found alongside this file. (Compacted
    from the original 9-row/9-col grid: dropped the in-between likelihood
    rows and merged the size=4/4.5/5 columns -- see build_danger_figure.)
    """
    return [
        ["Low","Low","Low","Low","Low","Low","Low"],
        ["Low","Low","Moderate","Moderate","Moderate","Considerable","Considerable"],
        ["Low","Moderate","Moderate","Considerable","Considerable","High","High"],
        ["Low","Moderate","Considerable","High","High","Extreme","Extreme"],
        ["Low","Moderate","High","High","Extreme","Extreme","Extreme"],
    ]


def _load_grid_and_counts():
    """Fetches the pooled empirical danger grid + per-cell maxD counts from
    GitHub, falling back to the hand-curated grid (with no counts, so those
    cells render as a flat fill rather than a blend) if the request fails or
    times out. Also accepts the older grid-only (bare list) JSON format for
    backward compatibility, treating every cell as having no counts."""
    try:
        with urllib.request.urlopen(_GRID_URL, timeout=10) as resp:
            data = json.load(resp)
        if isinstance(data, dict) and "grid" in data:
            return data["grid"], data.get("counts")
        return data, None  # older grid-only format
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        print(f"[CMAH] Could not load {_GRID_URL} ({e}); using hand-curated fallback grid.")
        return _build_manual_grid(), None

DEFAULT_DANGER_GRID, DEFAULT_DANGER_COUNTS = _load_grid_and_counts()


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


def build_likelihood_figure(sf, df, fig_w=465, fig_h=350):
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
        margin=dict(l=60, r=10, t=10, b=40),
        dragmode=False,
        modebar=dict(remove=["all"]),
        xaxis=dict(
            tickmode="array",
            tickvals=x_vals,
            ticktext=SENSITIVITY_LABELS,
            tickfont=dict(family="Barlow Condensed", color="#bbb", size=12),
            title=dict(text="Sensitivity to Triggers",
                       font=dict(family="Barlow Condensed", color="#888", size=11)),
            range=[-0.5, len(SENSITIVITY_LABELS) - 0.5],
            showgrid=False, showline=False, zeroline=False,
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
            showgrid=False, showline=False, zeroline=False,
            fixedrange=True,
        ),
        width=fig_w,
        height=fig_h,
        autosize=False,
    )
    return fig


def build_danger_figure(lik_range, size_range, danger_grid, counts_grid=None, fig_w=420, fig_h=420):
    """danger_grid is 5x7 (rows=named likelihood levels, cols=size, last col
    is the merged "4+" bucket). lik_range, if given, is [row0, row1] in the
    compact 0-4 row space (convert LIKELIHOOD_MATRIX's raw 0/2/4/6/8 values
    with lik_val_to_row() before calling this). size_range is [col0, col1]
    column indices (0-6).

    counts_grid, if given, is DEFAULT_DANGER_COUNTS: a 5x7 grid of
    {danger_level_str: count} dicts holding the full historical maxD
    distribution per cell (from build_danger_backgrounds.py). Where present,
    a cell renders the same way as the matplotlib plot -- background blended
    from the top-2 most common ratings, plus a small bar per rating showing
    its share of observations. Cells with no counts (fallback-grid cells, or
    a cell the user has edited away from its historical value) fall back to
    a flat fill in the cell's current danger_grid color.

    Cells are drawn as explicit rectangles (not a Heatmap trace) so the last
    column can be exactly 2x width without Plotly's uneven-spacing logic
    also inflating the neighboring column. A fully transparent scatter layer
    carries the hover text.
    """
    n_rows = len(DANGER_ROW_LABELS)
    n_cols = len(SIZE_LABELS)

    fig = go.Figure()

    hover_x, hover_y, hover_text, hover_n = [], [], [], []
    for r in range(n_rows):
        y0, y1 = r - 0.5, r + 0.5
        for c in range(n_cols):
            x0, x1 = size_col_bounds(c)
            d = danger_grid[r][c]
            counts = counts_grid[r][c] if counts_grid else None
            # A cell the user has edited away from its historical mode no
            # longer matches that history, so it renders flat (their choice)
            # rather than a blend that would visually contradict the label.
            is_edited = counts and DANGER_LEVELS.index(d) != int(
                max(counts.items(), key=lambda kv: kv[1])[0]
            )

            if counts and not is_edited:
                blended_rgb = _blend_cell_color(counts)
                fill = _rgb01_to_hex(blended_rgb)
            else:
                blended_rgb = None
                fill = DANGER_COLORS[d]

            fig.add_shape(
                type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
                line=dict(color="#0d1b2a", width=1),
                fillcolor=fill, layer="below",
            )

            if counts and not is_edited:
                total = sum(counts.values())
                max_n = max(counts.values())
                slot_w = (x1 - x0) / 5
                bar_w = slot_w * 0.7
                max_bar_h = 0.85
                for level in range(1, 6):
                    n = counts.get(str(level), 0)
                    if n == 0:
                        continue
                    slot_cx = x0 + slot_w * (level - 0.5)
                    bar_h = max_bar_h * (n / max_n)
                    fig.add_shape(
                        type="rect",
                        x0=slot_cx - bar_w / 2, x1=slot_cx + bar_w / 2,
                        y0=y0, y1=y0 + bar_h,
                        line=dict(color="#000000", width=0.6),
                        fillcolor=DANGER_COLORS[DANGER_LEVELS[level]],
                        opacity=0.85, layer="below",
                    )
                bright = 0.299 * blended_rgb[0] + 0.587 * blended_rgb[1] + 0.114 * blended_rgb[2]
                fig.add_annotation(
                    x=x1 - 0.06 * (x1 - x0), y=y1 - 0.08, text=f"n={total}",
                    showarrow=False, xanchor="right", yanchor="top",
                    font=dict(family="Share Tech Mono", size=9,
                              color="#111111" if bright > 0.45 else "#ffffff"),
                )

            hover_x.append(SIZE_COL_CENTERS[c])
            hover_y.append(r)
            hover_text.append(d)
            hover_n.append(sum(counts.values()) if counts else 0)

    fig.add_trace(go.Scatter(
        x=hover_x, y=hover_y, mode="markers",
        marker=dict(size=1, opacity=0),
        text=hover_text, showlegend=False,
        hovertemplate="Size: %{customdata[0]}<br>Likelihood: %{y}<br>"
                      "Danger: %{text}<br>Observations: %{customdata[1]}<extra></extra>",
        customdata=list(zip(
            [SIZE_LABELS[c] for _ in range(n_rows) for c in range(n_cols)],
            hover_n,
        )),
    ))

    # No text labels beyond n= counts — color and bar heights carry the rest
    # counts and breakdown are available on hover instead of drawn in-cell

    if lik_range and size_range:
        l0, l1 = lik_range
        s0, s1 = size_range
        sx0, _ = size_col_bounds(s0)
        _, sx1 = size_col_bounds(s1)
        pad_s = 0.10
        pad_l = 0.49 if l0 != l1 else 0.42
        fig.add_shape(
            type="path",
            path=rounded_rect_path(sx0 + pad_s, l0 - pad_l, sx1 - pad_s, l1 + pad_l, r=0.2),
            line=dict(color="#00e5ff", width=3),
            fillcolor="rgba(0, 229, 255, 0.18)",
        )

    fig.update_layout(
        paper_bgcolor="#0d1b2a", plot_bgcolor="#0d1b2a",
        margin=dict(l=65, r=10, t=10, b=50),
        dragmode=False,
        modebar=dict(remove=["all"]),
        xaxis=dict(
            tickmode="array",
            tickvals=SIZE_COL_CENTERS,
            ticktext=SIZE_LABELS,
            tickfont=dict(family="Barlow Condensed", color="#bbb", size=11),
            title=dict(text="Destructive Size",
                       font=dict(family="Barlow Condensed", color="#888", size=11)),
            range=[0, SIZE_AXIS_MAX],
            showgrid=False, showline=False, zeroline=False,
            fixedrange=True,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_rows)),
            ticktext=DANGER_ROW_LABELS,
            tickfont=dict(family="Barlow Condensed", color="#bbb", size=10),
            title=dict(text="Likelihood",
                       font=dict(family="Barlow Condensed", color="#888", size=11)),
            range=[-0.5, n_rows - 0.5],
            showgrid=False, showline=False, zeroline=False,
            fixedrange=True,
        ),
        width=fig_w,
        height=fig_h,
        autosize=False,
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
    for r in range(len(DANGER_ROW_LABELS) - 1, -1, -1):
        cells = [html.Div(
            DANGER_ROW_LABELS[r],
            style={
                "color": "#aaa", "fontSize": "10px", "fontFamily": "Barlow Condensed",
                "width": "130px", "textAlign": "right", "paddingRight": "8px",
                "display": "flex", "alignItems": "center", "justifyContent": "flex-end",
                "flexShrink": "0",
            }
        )]
        for c in range(len(SIZE_LABELS)):
            danger  = grid[r][c]
            bg      = DANGER_COLORS[danger]
            fg      = DANGER_TEXT[danger]
            is_last = c == len(SIZE_LABELS) - 1
            cell_w  = 116 if is_last else 58  # "4+" column rendered 2x width, matching the matrix plot
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
                        "width": f"{cell_w}px",
                        "fontSize": "10px",
                        "fontFamily": "Barlow Condensed",
                        "fontWeight": "600",
                        "minHeight": "38px",
                    },
                    className="danger-cell-dropdown",
                ),
                style={"width": f"{cell_w}px", "flexShrink": "0"}
            ))
        rows.append(html.Div(cells, style={"display": "flex", "marginBottom": "2px"}))

    size_row = html.Div(
        [html.Div("", style={"width": "138px", "flexShrink": "0"})] +
        [html.Div(s, style={
            "width": f"{116 if i == len(SIZE_LABELS) - 1 else 58}px", "textAlign": "center", "color": "#aaa",
            "fontSize": "10px", "fontFamily": "Barlow Condensed", "flexShrink": "0"
        }) for i, s in enumerate(SIZE_LABELS)],
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
    make_range_slider("size-slider", SIZE_LABELS, [1, 2]),
]), style=card)

forecast_tab = dbc.Row([
    # Left col: sliders + matrices
    dbc.Col([
        controls,
        dbc.Card(dbc.CardBody([
            html.Div("LIKELIHOOD MATRIX", style=lbl),
            html.Div(dcc.Graph(id="likelihood-matrix", config={
                "displayModeBar": False,
            }), style={"display": "flex", "justifyContent": "center"}),
        ]), style=card),
        dbc.Card(dbc.CardBody([
            html.Div([
                html.Span("SIZE/LIKELIHOOD DANGER RATING MATRIX", style=lbl),
                html.Span(
                    "Danger rating distribution based 32,518 historical forecasts",
                    style={
                        "fontFamily": "Share Tech Mono", "fontSize": "10px",
                        "color": "#00e5ff", "letterSpacing": "0.1em",
                        "marginLeft": "10px", "opacity": "0.7",
                    },
                ),
            ]),
            html.Div(dcc.Graph(id="danger-matrix", config={
                "displayModeBar": False,
            }), style={"display": "flex", "justifyContent": "center"}),
        ]), style=card),
    ], xs=12, md=8),
    # Right col: summary + NAPADS
    dbc.Col([
        dbc.Card(dbc.CardBody([
            html.Div("SUMMARY", style=lbl),
            html.Div(id="forecast-summary"),
        ]), style=card),
        html.Img(
            src="https://raw.githubusercontent.com/AndrewSchauer/CNFAC_Dashboard/main/NAPADS.png",
            style={"width": "100%", "marginTop": "4px", "borderRadius": "4px", "opacity": "0.9"},
        ),
    ], xs=12, md=4),
])


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
                html.Span("🔮 CMAH DASHBOARD", style={
                    "fontFamily": "Barlow Condensed", "fontWeight": "700",
                    "fontSize": "22px", "letterSpacing": "0.25em", "color": "#ffffff"
                }),
                html.Div("North American Public Avalanche Danger Scale Decision Support Tool", style={
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
    # Guard against None inputs during initial load
    if sens_val is None: sens_val = 2
    if dist_val is None: dist_val = 1
    if size_range is None: size_range = [1, 4]
    if danger_grid is None: danger_grid = DEFAULT_DANGER_GRID
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
    lr0, lr1 = lik_val_to_row(l0), lik_val_to_row(l1)  # compact 0-4 rows for the 5-row danger grid

    lik_fig    = build_likelihood_figure(sf, df, fig_w=465, fig_h=350)
    danger_fig = build_danger_figure([lr0, lr1], size_range, danger_grid, DEFAULT_DANGER_COUNTS, fig_w=420, fig_h=420)

    danger_in_box = {danger_grid[r][c] for r in range(lr0, lr1 + 1) for c in range(sz0, sz1 + 1)}
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
    return make_danger_grid_buttons(danger_grid or DEFAULT_DANGER_GRID)


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
        return copy.deepcopy(DEFAULT_DANGER_GRID)

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






if __name__ == "__main__":
    import os
    # Use 0.0.0.0 on Render (or any cloud host), 127.0.0.1 locally on Windows
    host = "0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1"
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, host=host, port=port, jupyter_mode="external")
