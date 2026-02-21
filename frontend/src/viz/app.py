"""
app.py  —  Grid Placement Optimizer · Austin TX
────────────────────────────────────────────────
Plotly Dash dashboard for visualizing optimal EV-infrastructure substation
placements. Loads all data once at startup; subsequent interactions are
handled entirely via in-memory state.

Run:
    python src/viz/app.py
    → http://localhost:8050
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import dash
from dash import ALL, Input, Output, State, ctx, dcc, html
import dash_bootstrap_components as dbc

# ── Module path so map_layers is importable ───────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from map_layers import (
    FEASIBILITY_COLORS,
    build_candidates_layer,
    build_demand_layer,
    build_forbidden_layer,
    build_substations_layer,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve()
ROOT     = _HERE.parents[2]          # repo root  (src/viz/app.py → 2 levels up)
DATA     = ROOT / "data"
RESULTS  = ROOT / "results"

# ── Austin fallback center ────────────────────────────────────────────────────
_AUSTIN_CENTER = {"lat": 30.2672, "lon": -97.7431}
_DEFAULT_ZOOM  = 10

# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def _load_npy(path: Path, shape=(500, 500), fill=1.0) -> np.ndarray:
    """Load a .npy array, or return a fallback array when file is absent."""
    if path.exists():
        return np.load(path)
    arr = np.full(shape, fill, dtype=np.float32)
    return arr


def _load_json(path: Path, fallback) -> dict | list:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return fallback


def _load_all():
    bounds = _load_json(
        DATA / "city_bounds.json",
        {"south": 30.098, "north": 30.516, "west": -97.928, "east": -97.560},
    )
    mask       = _load_npy(DATA / "forbidden_mask.npy",    fill=1.0)
    heatmap    = _load_npy(DATA / "demand_heatmap.npy",    fill=0.0)
    geojson    = _load_json(DATA / "existing_substations.geojson", {"type": "FeatureCollection", "features": []})
    candidates = _load_json(RESULTS / "top_candidates.json", [])
    return bounds, mask, heatmap, geojson, candidates


BOUNDS, MASK, HEATMAP, GEOJSON, CANDIDATES = _load_all()

_HAS_CANDIDATES = bool(CANDIDATES)
_MAP_CENTER      = (
    {"lat": CANDIDATES[0]["lat"], "lon": CANDIDATES[0]["lon"]}
    if _HAS_CANDIDATES
    else _AUSTIN_CENTER
)

# ══════════════════════════════════════════════════════════════════════════════
# Map figure builder
# ══════════════════════════════════════════════════════════════════════════════

def build_map_figure(
    center_lat: float | None = None,
    center_lon: float | None = None,
    zoom: float = _DEFAULT_ZOOM,
) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(build_demand_layer(HEATMAP, BOUNDS))
    fig.add_trace(build_forbidden_layer(MASK, BOUNDS))
    fig.add_trace(build_substations_layer(GEOJSON))

    if _HAS_CANDIDATES:
        fig.add_trace(build_candidates_layer(CANDIDATES))

    center = {
        "lat": center_lat if center_lat is not None else _MAP_CENTER["lat"],
        "lon": center_lon if center_lon is not None else _MAP_CENTER["lon"],
    }

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=center,
            zoom=zoom,
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#0d0d1a",
        legend=dict(
            bgcolor="rgba(13,13,26,0.88)",
            font=dict(color="#ccd6f6", size=11),
            bordercolor="#2a2a4a",
            borderwidth=1,
            x=0.01,
            y=0.01,
            xanchor="left",
            yanchor="bottom",
        ),
        uirevision="static",   # preserve user pan/zoom between re-renders
        hoverlabel=dict(
            bgcolor="#1a1a2e",
            bordercolor="#2a2a4a",
            font=dict(color="white", size=12),
        ),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar sub-components
# ══════════════════════════════════════════════════════════════════════════════

_OBJECTIVE_SLIDERS = [
    {"label": "Demand Coverage",   "id": "slider-demand",      "value": 0.40},
    {"label": "Distance to Grid",  "id": "slider-distance",    "value": 0.30},
    {"label": "Land Cost Penalty", "id": "slider-land",        "value": 0.20},
    {"label": "Redundancy Bonus",  "id": "slider-redundancy",  "value": 0.10},
]

_SIDEBAR_BG   = "#0d0d1a"
_SURFACE      = "#12122a"
_BORDER       = "#2a2a4a"
_ACCENT       = "#00d4ff"
_TEXT_PRI     = "#ffffff"
_TEXT_SEC     = "#8892b0"
_TEXT_BODY    = "#ccd6f6"
_FONT         = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def _label_row(label: str, value: float) -> html.Div:
    return html.Div(
        [
            html.Span(label, style={"color": _TEXT_SEC, "fontSize": "11px",
                                    "fontWeight": "600", "letterSpacing": "0.4px"}),
            html.Span(f"{value:.2f}", style={"color": _ACCENT, "fontSize": "11px",
                                              "fontWeight": "700"}),
        ],
        style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"},
    )


def _make_slider(cfg: dict) -> html.Div:
    return html.Div(
        [
            _label_row(cfg["label"], cfg["value"]),
            dcc.Slider(
                id=cfg["id"],
                min=0, max=1, step=0.05,
                value=cfg["value"],
                disabled=True,
                marks=None,
                tooltip={"always_visible": False},
                className="slider-dim",
            ),
        ],
        style={"marginBottom": "16px"},
    )


def _section_header(text: str) -> html.Div:
    return html.Div(
        text,
        style={
            "fontSize": "10px", "fontWeight": "800",
            "letterSpacing": "1.8px", "color": _TEXT_SEC,
            "marginBottom": "12px",
        },
    )


def _feasibility_badge(feasibility: str) -> html.Span:
    color = FEASIBILITY_COLORS.get(feasibility, _TEXT_SEC)
    return html.Span(
        feasibility,
        style={
            "backgroundColor": f"{color}1a",
            "color": color,
            "border": f"1px solid {color}55",
            "borderRadius": "3px",
            "padding": "1px 7px",
            "fontSize": "10px",
            "fontWeight": "700",
            "letterSpacing": "0.6px",
        },
    )


def _make_candidate_row(c: dict, index: int) -> html.Div:
    feasibility = c.get("feasibility", "N/A")
    border_color = FEASIBILITY_COLORS.get(feasibility, _TEXT_SEC)
    score = c.get("composite_score", 0)

    return html.Div(
        id={"type": "candidate-btn", "index": index},
        n_clicks=0,
        children=html.Div(
            [
                html.Span(
                    f"#{c.get('rank', index + 1)}",
                    style={"color": _ACCENT, "fontWeight": "800",
                           "fontSize": "15px", "minWidth": "30px"},
                ),
                html.Div(
                    [
                        html.Div(
                            f"{c['lat']:.4f}°N,  {abs(c['lon']):.4f}°W",
                            style={"color": _TEXT_PRI, "fontSize": "12px",
                                   "fontWeight": "600", "marginBottom": "2px"},
                        ),
                        html.Div(
                            f"Score: {score:.3f}",
                            style={"color": _TEXT_SEC, "fontSize": "11px"},
                        ),
                    ],
                    style={"flex": "1"},
                ),
                _feasibility_badge(feasibility),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "10px"},
        ),
        style={
            "padding": "11px 14px",
            "borderLeft": f"3px solid {border_color}",
            "borderBottom": f"1px solid {_BORDER}",
            "cursor": "pointer",
            "backgroundColor": _SIDEBAR_BG,
            "transition": "background-color 0.15s ease",
        },
    )


def _no_candidates_panel() -> html.Div:
    return html.Div(
        [
            html.Div("⚠️", style={"fontSize": "26px", "marginBottom": "8px"}),
            html.Div("No candidates yet.",
                     style={"color": _TEXT_PRI, "fontWeight": "700",
                            "fontSize": "13px", "marginBottom": "6px"}),
            html.Div("Run the optimizer first:",
                     style={"color": _TEXT_SEC, "fontSize": "12px", "marginBottom": "6px"}),
            html.Code(
                "python src/optimizer/main.py",
                style={
                    "display": "block",
                    "backgroundColor": _SURFACE,
                    "color": _ACCENT,
                    "padding": "7px 12px",
                    "borderRadius": "5px",
                    "fontSize": "11px",
                    "border": f"1px solid {_BORDER}",
                },
            ),
            html.Div(
                "The map will update when results are ready.",
                style={"color": _TEXT_SEC, "fontSize": "11px", "marginTop": "10px"},
            ),
        ],
        style={
            "textAlign": "center",
            "padding": "22px 18px",
            "border": f"1px dashed {_BORDER}",
            "borderRadius": "8px",
            "margin": "4px 0 8px",
        },
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar assembly
# ══════════════════════════════════════════════════════════════════════════════

_SIDEBAR = html.Div(
    style={
        "backgroundColor": _SIDEBAR_BG,
        "borderLeft": f"1px solid {_BORDER}",
        "height": "100vh",
        "overflowY": "auto",
        "fontFamily": _FONT,
        "display": "flex",
        "flexDirection": "column",
    },
    children=[

        # ── Header ────────────────────────────────────────────────────────────
        html.Div(
            style={"padding": "22px 18px 0"},
            children=[
                html.Div(
                    "GRID PLACEMENT OPTIMIZER",
                    style={
                        "fontSize": "13px", "fontWeight": "900",
                        "letterSpacing": "2.5px", "color": _ACCENT,
                        "marginBottom": "3px",
                    },
                ),
                html.Div(
                    "Austin, TX  ·  EV Infrastructure AI",
                    style={"fontSize": "11px", "color": _TEXT_SEC,
                           "letterSpacing": "0.4px"},
                ),
                html.Hr(style={"borderColor": _BORDER, "margin": "14px 0 10px"}),
            ],
        ),

        # ── Objective weight sliders ───────────────────────────────────────────
        html.Div(
            style={"padding": "0 18px", "marginBottom": "6px"},
            children=[
                _section_header("OBJECTIVE WEIGHTS"),
                *[_make_slider(s) for s in _OBJECTIVE_SLIDERS],
            ],
        ),

        html.Hr(style={"borderColor": _BORDER, "margin": "2px 0 10px"}),

        # ── Candidate list ────────────────────────────────────────────────────
        html.Div(
            [
                html.Div(
                    _section_header("TOP CANDIDATE SITES"),
                    style={"padding": "0 18px"},
                ),
                html.Div(
                    id="candidate-list",
                    children=(
                        [_make_candidate_row(c, i) for i, c in enumerate(CANDIDATES[:5])]
                        if _HAS_CANDIDATES
                        else [html.Div(_no_candidates_panel(), style={"padding": "0 18px"})]
                    ),
                ),
            ]
        ),

        html.Hr(style={"borderColor": _BORDER, "margin": "10px 0"}),

        # ── Feasibility report ────────────────────────────────────────────────
        html.Div(
            style={"padding": "0 18px 24px", "flex": "1"},
            children=[
                _section_header("FEASIBILITY REPORT"),
                html.Div(
                    id="feasibility-report",
                    children=html.Span(
                        "Select a candidate site to view its full analysis.",
                        style={"color": _TEXT_SEC, "fontSize": "12px",
                               "fontStyle": "italic"},
                    ),
                ),
            ],
        ),

        # ── Hidden state store ────────────────────────────────────────────────
        dcc.Store(id="selected-candidate"),
    ],
)


# ══════════════════════════════════════════════════════════════════════════════
# App + layout
# ══════════════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap",
    ],
    title="Grid Placement Optimizer · Austin TX",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.layout = dbc.Container(
    fluid=True,
    style={"padding": 0, "margin": 0, "backgroundColor": _SIDEBAR_BG,
           "overflow": "hidden", "fontFamily": _FONT},
    children=dbc.Row(
        [
            dbc.Col(
                dcc.Graph(
                    id="map",
                    figure=build_map_figure(),
                    style={"height": "100vh"},
                    config={
                        "displayModeBar": True,
                        "modeBarButtonsToRemove": [
                            "select2d", "lasso2d", "autoScale2d",
                        ],
                        "displaylogo": False,
                    },
                ),
                width=8,
                style={"padding": 0},
            ),
            dbc.Col(
                _SIDEBAR,
                width=4,
                style={"padding": 0},
            ),
        ],
        style={"height": "100vh", "margin": 0},
    ),
)


# ══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ══════════════════════════════════════════════════════════════════════════════

@app.callback(
    Output("map", "figure"),
    Output("selected-candidate", "data"),
    Input({"type": "candidate-btn", "index": ALL}, "n_clicks"),
    State("selected-candidate", "data"),
    prevent_initial_call=True,
)
def on_candidate_select(n_clicks: list, current_idx: int | None):
    """Re-center map on clicked candidate and update selection state."""
    if not any(n for n in (n_clicks or []) if n):
        return dash.no_update, dash.no_update

    triggered = ctx.triggered_id
    if triggered is None:
        return dash.no_update, dash.no_update

    idx = triggered["index"]
    if idx >= len(CANDIDATES):
        return dash.no_update, dash.no_update

    c = CANDIDATES[idx]
    fig = build_map_figure(center_lat=c["lat"], center_lon=c["lon"], zoom=13)
    return fig, idx


@app.callback(
    Output("feasibility-report", "children"),
    Input("selected-candidate", "data"),
)
def update_feasibility_report(idx: int | None) -> html.Div:
    """Populate the feasibility report panel for the selected candidate."""
    if idx is None or not CANDIDATES:
        return html.Span(
            "Select a candidate site to view its full analysis.",
            style={"color": _TEXT_SEC, "fontSize": "12px", "fontStyle": "italic"},
        )

    c          = CANDIDATES[idx]
    feasibility = c.get("feasibility", "N/A")
    fcolor      = FEASIBILITY_COLORS.get(feasibility, _TEXT_SEC)
    score       = c.get("composite_score", 0)
    reasoning   = c.get("reasoning", "No analysis available.")
    rank        = c.get("rank", idx + 1)

    return html.Div(
        [
            # ── Score header row ───────────────────────────────────────────────
            html.Div(
                [
                    html.Span(
                        f"Rank #{rank}",
                        style={"color": _ACCENT, "fontWeight": "800",
                               "fontSize": "17px"},
                    ),
                    _feasibility_badge(feasibility),
                ],
                style={"display": "flex", "alignItems": "center",
                       "gap": "10px", "marginBottom": "10px"},
            ),

            # ── Composite score callout ────────────────────────────────────────
            html.Div(
                [
                    html.Div(
                        "COMPOSITE SCORE",
                        style={"color": _TEXT_SEC, "fontSize": "10px",
                               "fontWeight": "700", "letterSpacing": "1px"},
                    ),
                    html.Div(
                        f"{score:.3f}",
                        style={"color": _TEXT_PRI, "fontWeight": "800",
                               "fontSize": "26px", "lineHeight": "1.1"},
                    ),
                ],
                style={
                    "backgroundColor": _SURFACE,
                    "border": f"1px solid {fcolor}44",
                    "borderRadius": "6px",
                    "padding": "10px 14px",
                    "marginBottom": "10px",
                },
            ),

            # ── Coordinates ───────────────────────────────────────────────────
            html.Div(
                f"📍  {c['lat']:.5f}°N,  {abs(c['lon']):.5f}°W",
                style={"color": _TEXT_SEC, "fontSize": "11px", "marginBottom": "12px"},
            ),

            html.Hr(style={"borderColor": _BORDER, "margin": "8px 0 10px"}),

            # ── AI reasoning ──────────────────────────────────────────────────
            html.Div(
                "AI ANALYSIS",
                style={"color": _TEXT_SEC, "fontSize": "10px", "fontWeight": "700",
                       "letterSpacing": "1.2px", "marginBottom": "8px"},
            ),
            html.Div(
                reasoning,
                style={"color": _TEXT_BODY, "fontSize": "12px",
                       "lineHeight": "1.7", "whiteSpace": "pre-wrap"},
            ),
        ]
    )


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n  Grid Placement Optimizer  —  Austin TX")
    print("  ─────────────────────────────────────────")
    print(f"  Data dir   : {DATA}")
    print(f"  Results dir: {RESULTS}")
    print(f"  Candidates : {len(CANDIDATES)} loaded")
    print(f"  Map center : {_MAP_CENTER['lat']:.4f}°N, {_MAP_CENTER['lon']:.4f}°W")
    print("\n  → http://localhost:8050\n")

    app.run(debug=True, host="0.0.0.0", port=8050)
