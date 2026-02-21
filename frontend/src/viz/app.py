"""
app.py  --  Grid Placement Optimizer
--------------------------------------
Plotly Dash dashboard for visualizing optimal substation placements.
Includes a Run Optimizer button that executes the GPU scoring engine
in the background and live-updates the map when results arrive.

Run:
    python src/viz/app.py
    -> http://localhost:8050
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import plotly.graph_objects as go
import dash
from dash import ALL, Input, Output, State, ctx, dcc, html
import dash_bootstrap_components as dbc

sys.path.insert(0, str(Path(__file__).parent))

from map_layers import (
    FEASIBILITY_COLORS,
    build_candidates_layer,
    build_demand_layer,
    build_forbidden_layer,
    build_substations_layer,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve()
ROOT      = _HERE.parents[2]           # frontend/
REPO_ROOT = ROOT.parent                # tvg-hackathon-2026/
DATA      = ROOT / "data"
RESULTS   = ROOT / "results"
GRAPH_DIR = REPO_ROOT / "graph"
GPU_DIR   = REPO_ROOT / "gpu-optimization"

_AUSTIN_CENTER = {"lat": 30.2672, "lon": -97.7431}
_DEFAULT_ZOOM  = 10

# ── Background pipeline state ─────────────────────────────────────────────────
_proc_lock  = threading.Lock()
_proc_state: dict = {"status": "idle", "log": ""}   # idle | running | done | error


def _stream_proc(proc, lines: list[str]) -> None:
    """Stream subprocess stdout into lines list, updating _proc_state log."""
    for line in proc.stdout:
        line = line.rstrip()
        if line.strip():
            lines.append(line)
            with _proc_lock:
                _proc_state["log"] = "\n".join(lines[-6:])
    proc.wait()


def _run_optimizer_bg() -> None:
    """
    Background thread: run GPU optimizer then Claude feasibility analysis,
    then copy outputs to frontend/data/ and frontend/results/.
    """
    import os as _os

    with _proc_lock:
        _proc_state["status"] = "running"
        _proc_state["log"]    = "Locating input data..."

    try:
        # Prefer fresh data from graph/data/, fall back to gpu-optimization/data/
        src_data = GRAPH_DIR / "data"
        if not (src_data / "forbidden_mask.npy").exists():
            src_data = GPU_DIR / "data"

        mask_path    = src_data / "forbidden_mask.npy"
        heatmap_path = src_data / "demand_heatmap.npy"
        subs_path    = src_data / "existing_substations.geojson"
        bounds_path  = src_data / "city_bounds.json"

        RESULTS.mkdir(parents=True, exist_ok=True)
        gpu_out = RESULTS / "top_candidates_gpu.json"

        # ── Step 1: GPU optimizer ─────────────────────────────────────────────
        cmd = [
            sys.executable, "-m", "src.optimizer.run",
            "--mask",        str(mask_path),
            "--heatmap",     str(heatmap_path),
            "--substations", str(subs_path),
            "--bounds",      str(bounds_path),
            "--output",      str(gpu_out),
        ]

        with _proc_lock:
            _proc_state["log"] = "Running GPU optimizer..."

        proc = subprocess.Popen(
            cmd, cwd=str(GPU_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        lines: list[str] = []
        _stream_proc(proc, lines)

        if proc.returncode != 0:
            with _proc_lock:
                _proc_state["status"] = "error"
                _proc_state["log"]    = "GPU optimizer exited with errors. Check console."
            return

        # ── Step 2: Claude feasibility (if API key available) ─────────────────
        api_key     = _os.environ.get("ANTHROPIC_API_KEY", "")
        site_dir    = REPO_ROOT / "sitereliability"
        final_out   = RESULTS / "top_candidates.json"

        if api_key and gpu_out.exists():
            with _proc_lock:
                _proc_state["log"] = "Running Claude feasibility analysis..."

            # Read city from city_bounds or default to Austin
            city = "Austin, Texas, USA"
            city_meta = src_data / "city_meta.json"
            if city_meta.exists():
                try:
                    meta = json.loads(city_meta.read_text(encoding="utf-8"))
                    city = meta.get("city", city)
                except Exception:
                    pass

            cmd_ai = [
                sys.executable, "-m", "src.agent.feasibility",
                "--candidates", str(gpu_out),
                "--output",     str(final_out),
                "--city",       city,
            ]
            proc2 = subprocess.Popen(
                cmd_ai, cwd=str(site_dir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            _stream_proc(proc2, lines)

            if proc2.returncode != 0 or not final_out.exists():
                # Fall back to GPU output without AI enrichment
                shutil.copy2(gpu_out, final_out)
        else:
            # No API key — use GPU optimizer output directly
            shutil.copy2(gpu_out, final_out)

        # ── Step 3: Sync input data to frontend/data/ ─────────────────────────
        DATA.mkdir(parents=True, exist_ok=True)
        for fname in ("forbidden_mask.npy", "demand_heatmap.npy",
                      "existing_substations.geojson", "city_bounds.json"):
            src = src_data / fname
            if src.exists():
                shutil.copy2(src, DATA / fname)

        with _proc_lock:
            _proc_state["status"] = "done"
            _proc_state["log"]    = "Pipeline complete."

    except Exception as exc:
        with _proc_lock:
            _proc_state["status"] = "error"
            _proc_state["log"]    = str(exc)


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_npy(path: Path, fill: float = 1.0) -> np.ndarray:
    if path.exists():
        return np.load(path)
    return np.full((200, 200), fill, dtype=np.float32)


def _load_json(path: Path, fallback):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return fallback


def _normalize_candidates(raw: list) -> list:
    """
    Normalize candidates from any schema into a flat dict the frontend can consume.

    Handles two schemas:
    1. GPU optimizer only — composite_score + per-objective scores, no feasibility
    2. Sitereliability-enriched — adds a nested 'feasibility' dict with verdict + fields
    """
    # raw may be a list or a dict with a 'candidates' key (sensitivity analysis output)
    if isinstance(raw, dict):
        raw = raw.get("candidates", [])

    out = []
    for c in raw:
        c = dict(c)
        score = float(c.get("composite_score", c.get("score", 0)))
        c["composite_score"] = score

        # Unwrap nested feasibility dict (sitereliability output)
        feas_obj = c.get("feasibility")
        if isinstance(feas_obj, dict):
            verdict = feas_obj.get("feasibility", "UNKNOWN")
            c["feasibility"]           = verdict
            c["reasoning"]             = feas_obj.get("reasoning", "")
            c["land_use"]              = feas_obj.get("land_use", "")
            c["zoning_assessment"]     = feas_obj.get("zoning_assessment", "")
            c["environmental_flags"]   = feas_obj.get("environmental_flags", [])
            c["community_sensitivity"] = feas_obj.get("community_sensitivity", "")
            c["grid_proximity"]        = feas_obj.get("grid_proximity", "")
        elif not isinstance(feas_obj, str) or not feas_obj:
            # No feasibility at all — derive from score
            if score >= 0.55:
                c["feasibility"] = "HIGH"
            elif score >= 0.40:
                c["feasibility"] = "MEDIUM"
            else:
                c["feasibility"] = "LOW"

        if "reasoning" not in c or not c["reasoning"]:
            parts = [f"Composite score {score:.3f}."]
            if c.get("nearest_existing_km") is not None:
                parts.append(f"{c['nearest_existing_km']:.1f} km from nearest substation.")
            if c.get("coverage_5km_pct") is not None:
                parts.append(f"{c['coverage_5km_pct']:.1f}% of city demand within 5 km.")
            if c.get("load_relief_score") is not None:
                parts.append(f"Load relief: {c['load_relief_score']:.3f}.")
            if c.get("loss_reduction_score") is not None:
                parts.append(f"Loss reduction: {c['loss_reduction_score']:.3f}.")
            if c.get("redundancy_score") is not None:
                parts.append(f"Redundancy: {c['redundancy_score']:.3f}.")
            c["reasoning"] = "  ".join(parts)

        out.append(c)
    return out


def _load_all_fresh():
    """Reload all data from disk. Called at startup and after pipeline runs."""
    bounds     = _load_json(DATA / "city_bounds.json",
                            {"south": 30.098, "north": 30.516,
                             "west": -97.928, "east": -97.560})
    mask       = _load_npy(DATA / "forbidden_mask.npy",    fill=1.0)
    heatmap    = _load_npy(DATA / "demand_heatmap.npy",    fill=0.0)
    geojson    = _load_json(DATA / "existing_substations.geojson",
                            {"type": "FeatureCollection", "features": []})
    raw        = _load_json(RESULTS / "top_candidates.json", [])
    candidates = _normalize_candidates(raw)
    return bounds, mask, heatmap, geojson, candidates


# Load once at startup
BOUNDS, MASK, HEATMAP, GEOJSON, CANDIDATES = _load_all_fresh()


# ── Map figure builder ────────────────────────────────────────────────────────

def build_map_figure(
    bounds=None, mask=None, heatmap=None, geojson=None, candidates=None,
    center_lat=None, center_lon=None, zoom=_DEFAULT_ZOOM,
) -> go.Figure:
    b  = bounds     or BOUNDS
    m  = mask       if mask is not None else MASK
    h  = heatmap    if heatmap is not None else HEATMAP
    gj = geojson    or GEOJSON
    cs = candidates if candidates is not None else CANDIDATES

    fig = go.Figure()
    fig.add_trace(build_demand_layer(h, b))
    fig.add_trace(build_forbidden_layer(m, b))
    fig.add_trace(build_substations_layer(gj))
    if cs:
        fig.add_trace(build_candidates_layer(cs))

    map_center_lat = center_lat or (cs[0]["lat"] if cs else _AUSTIN_CENTER["lat"])
    map_center_lon = center_lon or (cs[0]["lon"] if cs else _AUSTIN_CENTER["lon"])

    fig.update_layout(
        mapbox=dict(style="open-street-map",
                    center={"lat": map_center_lat, "lon": map_center_lon},
                    zoom=zoom),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="#0d0d1a",
        plot_bgcolor="#0d0d1a",
        legend=dict(bgcolor="rgba(13,13,26,0.88)", font=dict(color="#ccd6f6", size=11),
                    bordercolor="#2a2a4a", borderwidth=1,
                    x=0.01, y=0.01, xanchor="left", yanchor="bottom"),
        uirevision="map",
        hoverlabel=dict(bgcolor="#1a1a2e", bordercolor="#2a2a4a",
                        font=dict(color="white", size=12)),
    )
    return fig


# ── Style constants ───────────────────────────────────────────────────────────
_SIDEBAR_BG = "#0d0d1a"
_SURFACE    = "#12122a"
_BORDER     = "#2a2a4a"
_ACCENT     = "#00d4ff"
_TEXT_PRI   = "#ffffff"
_TEXT_SEC   = "#8892b0"
_TEXT_BODY  = "#ccd6f6"
_FONT       = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

_OBJECTIVE_SLIDERS = [
    {"label": "Load Relief",       "id": "slider-load",        "value": 0.35},
    {"label": "Loss Reduction",    "id": "slider-loss",        "value": 0.35},
    {"label": "Sustainability",    "id": "slider-sustain",     "value": 0.15},
    {"label": "Redundancy",        "id": "slider-redundancy",  "value": 0.15},
]


# ── Sidebar component builders ────────────────────────────────────────────────

def _section_header(text):
    return html.Div(text, style={"fontSize": "10px", "fontWeight": "800",
                                  "letterSpacing": "1.8px", "color": _TEXT_SEC,
                                  "marginBottom": "12px"})


def _label_row(label, value):
    return html.Div([
        html.Span(label, style={"color": _TEXT_SEC, "fontSize": "11px",
                                 "fontWeight": "600", "letterSpacing": "0.4px"}),
        html.Span(f"{value:.2f}", style={"color": _ACCENT, "fontSize": "11px",
                                          "fontWeight": "700"}),
    ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"})


def _make_slider(cfg):
    return html.Div([
        _label_row(cfg["label"], cfg["value"]),
        dcc.Slider(id=cfg["id"], min=0, max=1, step=0.05,
                   value=cfg["value"], disabled=True, marks=None,
                   tooltip={"always_visible": False}),
    ], style={"marginBottom": "16px"})


def _feasibility_badge(feasibility):
    color = FEASIBILITY_COLORS.get(feasibility, _TEXT_SEC)
    return html.Span(feasibility, style={
        "backgroundColor": f"{color}1a", "color": color,
        "border": f"1px solid {color}55", "borderRadius": "3px",
        "padding": "1px 7px", "fontSize": "10px",
        "fontWeight": "700", "letterSpacing": "0.6px",
    })


def _make_candidate_row(c, index):
    feasibility  = c.get("feasibility", "N/A")
    border_color = FEASIBILITY_COLORS.get(feasibility, _TEXT_SEC)
    score        = c.get("composite_score", 0)
    return html.Div(
        id={"type": "candidate-btn", "index": index},
        n_clicks=0,
        children=html.Div([
            html.Span(f"#{c.get('rank', index + 1)}",
                      style={"color": _ACCENT, "fontWeight": "800",
                             "fontSize": "15px", "minWidth": "30px"}),
            html.Div([
                html.Div(f"{c['lat']:.4f}N,  {abs(c['lon']):.4f}W",
                         style={"color": _TEXT_PRI, "fontSize": "12px",
                                "fontWeight": "600", "marginBottom": "2px"}),
                html.Div(f"Score: {score:.3f}",
                         style={"color": _TEXT_SEC, "fontSize": "11px"}),
            ], style={"flex": "1"}),
            _feasibility_badge(feasibility),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px"}),
        style={"padding": "11px 14px",
               "borderLeft": f"3px solid {border_color}",
               "borderBottom": f"1px solid {_BORDER}",
               "cursor": "pointer", "backgroundColor": _SIDEBAR_BG,
               "transition": "background-color 0.15s ease"},
    )


def _build_candidate_list(candidates=None):
    cs = candidates if candidates is not None else CANDIDATES
    if not cs:
        return [html.Div([
            html.Div("No candidates yet.", style={"color": _TEXT_PRI, "fontWeight": "700",
                                                   "fontSize": "13px", "marginBottom": "6px"}),
            html.Div("Click Run Optimizer above to score candidate locations.",
                     style={"color": _TEXT_SEC, "fontSize": "12px"}),
        ], style={"textAlign": "center", "padding": "22px 18px",
                  "border": f"1px dashed {_BORDER}", "borderRadius": "8px", "margin": "4px 0 8px"})]
    return [_make_candidate_row(c, i) for i, c in enumerate(cs[:10])]


# ── App layout ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap",
    ],
    title="Grid Placement Optimizer",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

_SIDEBAR = html.Div(
    style={"backgroundColor": _SIDEBAR_BG, "borderLeft": f"1px solid {_BORDER}",
           "height": "100vh", "overflowY": "auto", "fontFamily": _FONT,
           "display": "flex", "flexDirection": "column"},
    children=[

        # Header
        html.Div(style={"padding": "18px 18px 0"}, children=[
            html.Div("GRID PLACEMENT OPTIMIZER",
                     style={"fontSize": "13px", "fontWeight": "900",
                            "letterSpacing": "2.5px", "color": _ACCENT, "marginBottom": "3px"}),
            html.Div("Austin, TX  --  EV Infrastructure AI",
                     style={"fontSize": "11px", "color": _TEXT_SEC, "letterSpacing": "0.4px"}),
            html.Hr(style={"borderColor": _BORDER, "margin": "14px 0 10px"}),
        ]),

        # Run button + status
        html.Div(style={"padding": "0 18px 10px"}, children=[
            html.Button(
                "Run Optimizer",
                id="run-btn",
                n_clicks=0,
                style={"width": "100%", "padding": "10px",
                       "backgroundColor": _ACCENT, "color": "#0d0d1a",
                       "border": "none", "borderRadius": "6px",
                       "fontWeight": "800", "fontSize": "12px",
                       "letterSpacing": "1px", "cursor": "pointer"},
            ),
            html.Div(id="run-status", style={"marginTop": "8px", "fontSize": "11px",
                                              "color": _TEXT_SEC, "minHeight": "16px"}),
        ]),

        html.Hr(style={"borderColor": _BORDER, "margin": "0 0 10px"}),

        # Objective weight sliders (display only)
        html.Div(style={"padding": "0 18px", "marginBottom": "6px"}, children=[
            _section_header("OBJECTIVE WEIGHTS"),
            *[_make_slider(s) for s in _OBJECTIVE_SLIDERS],
        ]),

        html.Hr(style={"borderColor": _BORDER, "margin": "2px 0 10px"}),

        # Candidate list
        html.Div([
            html.Div(_section_header("TOP CANDIDATE SITES"), style={"padding": "0 18px"}),
            html.Div(id="candidate-list", children=_build_candidate_list()),
        ]),

        html.Hr(style={"borderColor": _BORDER, "margin": "10px 0"}),

        # Feasibility report
        html.Div(style={"padding": "0 18px 24px", "flex": "1"}, children=[
            _section_header("FEASIBILITY REPORT"),
            html.Div(id="feasibility-report",
                     children=html.Span("Select a candidate site to view its full analysis.",
                                        style={"color": _TEXT_SEC, "fontSize": "12px",
                                               "fontStyle": "italic"})),
        ]),

        # State stores
        dcc.Store(id="selected-candidate"),
        dcc.Interval(id="poll-interval", interval=1500, n_intervals=0, disabled=True),
    ],
)

app.layout = dbc.Container(
    fluid=True,
    style={"padding": 0, "margin": 0, "backgroundColor": _SIDEBAR_BG,
           "overflow": "hidden", "fontFamily": _FONT},
    children=dbc.Row([
        dbc.Col(
            dcc.Graph(id="map", figure=build_map_figure(),
                      style={"height": "100vh"},
                      config={"displayModeBar": True,
                              "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
                              "displaylogo": False}),
            width=8, style={"padding": 0},
        ),
        dbc.Col(_SIDEBAR, width=4, style={"padding": 0}),
    ], style={"height": "100vh", "margin": 0}),
)


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output("poll-interval", "disabled"),
    Output("run-status",    "children"),
    Output("map",           "figure"),
    Output("candidate-list","children"),
    Input("run-btn",        "n_clicks"),
    Input("poll-interval",  "n_intervals"),
    prevent_initial_call=True,
)
def handle_run_and_poll(run_clicks, n_intervals):
    """Start pipeline on button click; poll status and refresh UI when done."""
    global BOUNDS, MASK, HEATMAP, GEOJSON, CANDIDATES

    triggered = ctx.triggered_id

    # ── Button clicked ────────────────────────────────────────────────────────
    if triggered == "run-btn":
        with _proc_lock:
            if _proc_state["status"] == "running":
                return False, "Already running...", dash.no_update, dash.no_update
        threading.Thread(target=_run_optimizer_bg, daemon=True).start()
        return (
            False,
            html.Span("Running optimizer...",
                      style={"color": _ACCENT, "fontStyle": "italic"}),
            dash.no_update,
            dash.no_update,
        )

    # ── Poll tick ─────────────────────────────────────────────────────────────
    with _proc_lock:
        status = _proc_state["status"]
        log    = _proc_state["log"]

    if status == "idle":
        return True, dash.no_update, dash.no_update, dash.no_update

    if status == "running":
        last_line = log.strip().split("\n")[-1] if log.strip() else "Working..."
        return (
            False,
            html.Span(last_line[:80], style={"color": _ACCENT, "fontStyle": "italic"}),
            dash.no_update,
            dash.no_update,
        )

    if status == "done":
        # Reload data fresh from disk
        BOUNDS, MASK, HEATMAP, GEOJSON, CANDIDATES = _load_all_fresh()
        with _proc_lock:
            _proc_state["status"] = "idle"
        return (
            True,
            html.Span("Done! Map updated.", style={"color": "#00c853", "fontWeight": "700"}),
            build_map_figure(),
            _build_candidate_list(),
        )

    if status == "error":
        with _proc_lock:
            _proc_state["status"] = "idle"
        return (
            True,
            html.Span(f"Error: {log[:100]}", style={"color": "#d50000"}),
            dash.no_update,
            dash.no_update,
        )

    return True, dash.no_update, dash.no_update, dash.no_update


@app.callback(
    Output("map",                "figure",   allow_duplicate=True),
    Output("selected-candidate", "data"),
    Input({"type": "candidate-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def on_candidate_select(n_clicks):
    """Re-center map on the clicked candidate."""
    if not any(n for n in (n_clicks or []) if n):
        return dash.no_update, dash.no_update

    triggered = ctx.triggered_id
    if triggered is None:
        return dash.no_update, dash.no_update

    idx = triggered["index"]
    if idx >= len(CANDIDATES):
        return dash.no_update, dash.no_update

    c   = CANDIDATES[idx]
    fig = build_map_figure(center_lat=c["lat"], center_lon=c["lon"], zoom=13)
    return fig, idx


@app.callback(
    Output("feasibility-report", "children"),
    Input("selected-candidate",  "data"),
)
def update_feasibility_report(idx):
    if idx is None or not CANDIDATES:
        return html.Span("Select a candidate site to view its full analysis.",
                         style={"color": _TEXT_SEC, "fontSize": "12px", "fontStyle": "italic"})

    c           = CANDIDATES[idx]
    feasibility = c.get("feasibility", "N/A")
    fcolor      = FEASIBILITY_COLORS.get(feasibility, _TEXT_SEC)
    score       = c.get("composite_score", 0)
    reasoning   = c.get("reasoning", "No analysis available.")
    rank        = c.get("rank", idx + 1)

    # Build extra metrics rows
    metrics = []
    for key, label in [("load_relief_score",    "Load Relief"),
                       ("loss_reduction_score",  "Loss Reduction"),
                       ("sustainability_score",  "Sustainability"),
                       ("redundancy_score",      "Redundancy")]:
        if key in c:
            metrics.append(html.Div([
                html.Span(label, style={"color": _TEXT_SEC, "fontSize": "11px", "flex": "1"}),
                html.Span(f"{c[key]:.3f}", style={"color": _TEXT_PRI, "fontWeight": "700",
                                                    "fontSize": "11px"}),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "padding": "3px 0", "borderBottom": f"1px solid {_BORDER}"}))

    coverage = []
    for r in [3, 5, 10]:
        key = f"coverage_{r}km_pct"
        if key in c:
            coverage.append(html.Span(f"{r}km: {c[key]:.1f}%",
                                      style={"color": _TEXT_SEC, "fontSize": "11px",
                                             "marginRight": "12px"}))

    # Site intelligence fields (from sitereliability)
    site_rows = []
    for key, label in [("land_use",              "Land Use"),
                       ("zoning_assessment",      "Zoning"),
                       ("community_sensitivity",  "Community Sensitivity"),
                       ("grid_proximity",         "Grid Proximity")]:
        val = c.get(key, "")
        if val:
            site_rows.append(html.Div([
                html.Span(label, style={"color": _TEXT_SEC, "fontSize": "11px",
                                        "fontWeight": "600", "minWidth": "110px"}),
                html.Span(str(val), style={"color": _TEXT_BODY, "fontSize": "11px",
                                            "flex": "1", "textAlign": "right"}),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "padding": "3px 0", "borderBottom": f"1px solid {_BORDER}"}))

    env_flags = c.get("environmental_flags", [])
    if env_flags:
        site_rows.append(html.Div([
            html.Span("Env. Flags", style={"color": _TEXT_SEC, "fontSize": "11px",
                                            "fontWeight": "600"}),
            html.Span(", ".join(env_flags), style={"color": "#ffd600", "fontSize": "11px",
                                                     "textAlign": "right", "flex": "1"}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "padding": "3px 0", "borderBottom": f"1px solid {_BORDER}"}))

    return html.Div([
        html.Div([
            html.Span(f"Rank #{rank}",
                      style={"color": _ACCENT, "fontWeight": "800", "fontSize": "17px"}),
            _feasibility_badge(feasibility),
        ], style={"display": "flex", "alignItems": "center",
                  "gap": "10px", "marginBottom": "10px"}),

        html.Div([
            html.Div("COMPOSITE SCORE",
                     style={"color": _TEXT_SEC, "fontSize": "10px",
                            "fontWeight": "700", "letterSpacing": "1px"}),
            html.Div(f"{score:.3f}",
                     style={"color": _TEXT_PRI, "fontWeight": "800",
                            "fontSize": "26px", "lineHeight": "1.1"}),
        ], style={"backgroundColor": _SURFACE, "border": f"1px solid {fcolor}44",
                  "borderRadius": "6px", "padding": "10px 14px", "marginBottom": "10px"}),

        html.Div(f"  {c['lat']:.5f}N,  {abs(c['lon']):.5f}W",
                 style={"color": _TEXT_SEC, "fontSize": "11px", "marginBottom": "10px"}),

        *([html.Div(metrics, style={"marginBottom": "10px"})] if metrics else []),
        *([html.Div(coverage, style={"marginBottom": "10px"})] if coverage else []),

        *([html.Div([
            html.Div("SITE INTELLIGENCE", style={"color": _TEXT_SEC, "fontSize": "10px",
                                                   "fontWeight": "700", "letterSpacing": "1px",
                                                   "marginBottom": "6px"}),
            *site_rows,
        ], style={"marginBottom": "10px"})] if site_rows else []),

        html.Hr(style={"borderColor": _BORDER, "margin": "8px 0 10px"}),
        html.Div("AI ANALYSIS", style={"color": _TEXT_SEC, "fontSize": "10px",
                                        "fontWeight": "700", "letterSpacing": "1.2px",
                                        "marginBottom": "8px"}),
        html.Div(reasoning, style={"color": _TEXT_BODY, "fontSize": "12px",
                                    "lineHeight": "1.7", "whiteSpace": "pre-wrap"}),
    ])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Grid Placement Optimizer  --  Austin TX")
    print("  -----------------------------------------")
    print(f"  Data dir   : {DATA}")
    print(f"  Results dir: {RESULTS}")
    print(f"  Candidates : {len(CANDIDATES)} loaded")
    if CANDIDATES:
        best = CANDIDATES[0]
        print(f"  Best site  : ({best['lat']:.4f}N, {best['lon']:.4f}W)"
              f"  score={best['composite_score']:.4f}")
    print("\n  -> http://localhost:8050\n")

    app.run(debug=False, host="0.0.0.0", port=8050)
