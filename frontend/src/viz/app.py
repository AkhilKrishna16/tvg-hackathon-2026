"""
app.py  --  Energy Grid Placement Optimizer
---------------------------------------------
Full-stack Plotly Dash dashboard for the TVG Hackathon 2026.
Users input a city, tune optimization weights, run the full pipeline
(ingest -> GPU optimize -> Claude AI feasibility), and explore results
on an interactive map.

Run:
    python src/viz/app.py
    -> http://localhost:8050
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
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
SITE_DIR  = REPO_ROOT / "sitereliability"

_DEFAULT_CENTER = {"lat": 30.2672, "lon": -97.7431}
_DEFAULT_ZOOM   = 10

# ── Background pipeline state ─────────────────────────────────────────────────
_lock  = threading.Lock()
_state: dict = {
    "status": "idle",        # idle | running | done | error
    "step":   "",
    "log":    "",
    "progress": 0,
    "metrics": {},
}


def _update(**kw):
    with _lock:
        _state.update(kw)


def _get_state():
    with _lock:
        return dict(_state)


def _stream(proc, lines):
    for line in proc.stdout:
        line = line.rstrip()
        if line.strip():
            lines.append(line)
            _update(log="\n".join(lines[-8:]))
    proc.wait()


def _run_pipeline_bg(city: str, weights: dict, grid: int = 200, top_n: int = 10):
    """Background thread: runs the full pipeline end-to-end."""
    _update(status="running", step="Initializing...", log="", progress=2, metrics={})
    t0 = time.perf_counter()
    lines: list[str] = []
    metrics: dict = {}

    try:
        # ── Step 1: Data Ingestion ────────────────────────────────────────
        _update(step="Ingesting city data from OpenStreetMap...", progress=5)
        data_dir = str(GRAPH_DIR / "data")
        os.makedirs(data_dir, exist_ok=True)

        t1 = time.perf_counter()
        cmd = [
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, r'{GRAPH_DIR}'); "
            f"from src.data.ingest import run_ingest; "
            f"run_ingest(city={city!r}, grid={grid}, output_dir=r'{data_dir}')"
        ]
        proc = subprocess.Popen(
            cmd, cwd=str(GRAPH_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        _stream(proc, lines)
        metrics["ingest_time"] = round(time.perf_counter() - t1, 1)
        if proc.returncode != 0:
            _update(status="error", step="Data ingestion failed", progress=5)
            return
        _update(progress=30)

        # ── Step 2: GPU Optimization ──────────────────────────────────────
        _update(step="Running GPU-accelerated optimization...", progress=32)
        gpu_results = str(GPU_DIR / "results")
        os.makedirs(gpu_results, exist_ok=True)
        gpu_out = os.path.join(gpu_results, "top_candidates.json")

        weights_json = json.dumps(weights)
        t2 = time.perf_counter()
        cmd = [
            sys.executable, "-m", "src.optimizer.run",
            "--mask",        os.path.join(data_dir, "forbidden_mask.npy"),
            "--heatmap",     os.path.join(data_dir, "demand_heatmap.npy"),
            "--substations", os.path.join(data_dir, "existing_substations.geojson"),
            "--bounds",      os.path.join(data_dir, "city_bounds.json"),
            "--output",      gpu_out,
            "--top-n",       str(top_n),
            "--weights",     weights_json,
        ]
        proc = subprocess.Popen(
            cmd, cwd=str(GPU_DIR),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        _stream(proc, lines)
        metrics["optimize_time"] = round(time.perf_counter() - t2, 1)

        if proc.returncode != 0:
            _update(status="error", step="GPU optimization failed", progress=35)
            return
        _update(progress=65)

        # ── Step 3: Claude AI Feasibility ─────────────────────────────────
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        final_out = gpu_out

        if api_key:
            _update(step="Running Claude AI site feasibility analysis...", progress=68)
            ai_out = os.path.join(gpu_results, "top_candidates_enriched.json")
            t3 = time.perf_counter()
            cmd = [
                sys.executable, "-m", "src.agent.feasibility",
                "--candidates", gpu_out,
                "--output",     ai_out,
                "--city",       city,
            ]
            proc = subprocess.Popen(
                cmd, cwd=str(SITE_DIR),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            _stream(proc, lines)
            metrics["ai_time"] = round(time.perf_counter() - t3, 1)
            if proc.returncode == 0 and os.path.exists(ai_out):
                final_out = ai_out
            _update(progress=88)
        else:
            _update(step="Skipping AI analysis (no ANTHROPIC_API_KEY)", progress=88)
            metrics["ai_time"] = 0

        # ── Step 4: Copy to frontend ──────────────────────────────────────
        _update(step="Preparing results...", progress=92)
        DATA.mkdir(parents=True, exist_ok=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        for fname in ("forbidden_mask.npy", "demand_heatmap.npy",
                      "existing_substations.geojson", "city_bounds.json", "city_meta.json"):
            src = Path(data_dir) / fname
            if src.exists():
                shutil.copy2(src, DATA / fname)
        if os.path.exists(final_out):
            shutil.copy2(final_out, RESULTS / "top_candidates.json")

        metrics["total_time"] = round(time.perf_counter() - t0, 1)
        try:
            mask = np.load(DATA / "forbidden_mask.npy")
            metrics["grid_size"] = f"{mask.shape[0]}x{mask.shape[1]}"
            metrics["placeable"] = int((mask > 0.5).sum())
            metrics["total_cells"] = mask.size
        except Exception:
            pass

        _update(status="done", step="Pipeline complete!", progress=100, metrics=metrics)

    except Exception as exc:
        _update(status="error", step=f"Error: {str(exc)[:120]}", log=str(exc))


# ── Data helpers ──────────────────────────────────────────────────────────────

def _load_npy(path: Path, fill: float = 1.0) -> np.ndarray:
    return np.load(path) if path.exists() else np.full((200, 200), fill, dtype=np.float32)

def _load_json(path: Path, fallback):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else fallback

def _normalize_candidates(raw) -> list:
    if isinstance(raw, dict):
        raw = raw.get("candidates", [])
    if not isinstance(raw, list):
        return []
    out = []
    for c in raw:
        c = dict(c)
        score = float(c.get("composite_score", c.get("score", 0)))
        c["composite_score"] = score
        feas_obj = c.get("feasibility")
        if isinstance(feas_obj, dict):
            c["feasibility"]           = feas_obj.get("feasibility", "UNKNOWN")
            c["reasoning"]             = feas_obj.get("reasoning", "")
            c["land_use"]              = feas_obj.get("land_use", "")
            c["zoning_assessment"]     = feas_obj.get("zoning_assessment", "")
            c["environmental_flags"]   = feas_obj.get("environmental_flags", [])
            c["community_sensitivity"] = feas_obj.get("community_sensitivity", "")
            c["grid_proximity"]        = feas_obj.get("grid_proximity", "")
        elif not isinstance(feas_obj, str) or not feas_obj:
            if score >= 0.55:   c["feasibility"] = "HIGH"
            elif score >= 0.40: c["feasibility"] = "MEDIUM"
            else:               c["feasibility"] = "LOW"
        if not c.get("reasoning"):
            parts = [f"Composite score {score:.3f}."]
            if c.get("nearest_existing_km") is not None:
                parts.append(f"{c['nearest_existing_km']:.1f} km from nearest substation.")
            if c.get("coverage_5km_pct") is not None:
                parts.append(f"{c['coverage_5km_pct']:.1f}% demand within 5 km.")
            c["reasoning"] = " ".join(parts)
        out.append(c)
    return out

def _load_all():
    bounds     = _load_json(DATA / "city_bounds.json",
                            {"south": 30.098, "north": 30.516, "west": -97.928, "east": -97.560})
    mask       = _load_npy(DATA / "forbidden_mask.npy", fill=1.0)
    heatmap    = _load_npy(DATA / "demand_heatmap.npy", fill=0.0)
    geojson    = _load_json(DATA / "existing_substations.geojson",
                            {"type": "FeatureCollection", "features": []})
    raw        = _load_json(RESULTS / "top_candidates.json", [])
    candidates = _normalize_candidates(raw)
    return bounds, mask, heatmap, geojson, candidates

BOUNDS, MASK, HEATMAP, GEOJSON, CANDIDATES = _load_all()


# ── Map builder ───────────────────────────────────────────────────────────────

def build_map(bounds=None, mask=None, heatmap=None, geojson=None,
              candidates=None, center_lat=None, center_lon=None, zoom=_DEFAULT_ZOOM):
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

    clat = center_lat or (cs[0]["lat"] if cs else _DEFAULT_CENTER["lat"])
    clon = center_lon or (cs[0]["lon"] if cs else _DEFAULT_CENTER["lon"])

    fig.update_layout(
        mapbox=dict(style="carto-darkmatter",
                    center={"lat": clat, "lon": clon}, zoom=zoom),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor="#080810",
        plot_bgcolor="#080810",
        legend=dict(bgcolor="rgba(8,8,16,0.92)", font=dict(color="#a0aec0", size=11),
                    bordercolor="#1e293b", borderwidth=1,
                    x=0.01, y=0.01, xanchor="left", yanchor="bottom"),
        uirevision="keep",
        hoverlabel=dict(bgcolor="#0f172a", bordercolor="#334155",
                        font=dict(color="white", size=12)),
    )
    return fig


# ── Style tokens ──────────────────────────────────────────────────────────────
BG       = "#080810"
SURFACE  = "#0f1628"
SURFACE2 = "#162036"
BORDER   = "#1e293b"
ACCENT   = "#38bdf8"
GREEN    = "#22c55e"
AMBER    = "#eab308"
RED      = "#ef4444"
TEXT     = "#f1f5f9"
TEXT2    = "#94a3b8"
TEXT3    = "#64748b"
FONT     = "'Inter', system-ui, -apple-system, sans-serif"
FCOLORS  = {"HIGH": GREEN, "MEDIUM": AMBER, "LOW": RED, "UNKNOWN": TEXT3}


def _section(title):
    return html.Div(title, style={
        "fontSize": "10px", "fontWeight": "700", "letterSpacing": "2px",
        "color": TEXT3, "marginBottom": "10px", "textTransform": "uppercase"})

def _badge(text, color):
    return html.Span(text, style={
        "background": f"{color}18", "color": color,
        "border": f"1px solid {color}44", "borderRadius": "4px",
        "padding": "2px 8px", "fontSize": "10px", "fontWeight": "700",
        "letterSpacing": "0.5px"})

def _metric_card(label, value):
    return html.Div([
        html.Div(label, style={"fontSize": "9px", "color": TEXT3, "fontWeight": "700",
                                "letterSpacing": "1px", "textTransform": "uppercase",
                                "marginBottom": "2px"}),
        html.Div(str(value), style={"fontSize": "20px", "fontWeight": "800",
                                     "color": TEXT, "lineHeight": "1.1"}),
    ], style={"background": SURFACE2, "borderRadius": "8px", "padding": "10px 12px",
              "border": f"1px solid {BORDER}", "flex": "1", "minWidth": "0"})


# ── App layout ────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.SLATE,
        "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap",
    ],
    title="Energy Grid Optimizer",
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

# ── Left sidebar ──────────────────────────────────────────────────────────────

_left = html.Div(style={
    "background": BG, "borderRight": f"1px solid {BORDER}",
    "height": "100vh", "overflowY": "auto", "fontFamily": FONT,
    "display": "flex", "flexDirection": "column",
}, children=[

    html.Div(style={"padding": "20px 18px 8px"}, children=[
        html.Div([
            html.Span("ENERGY GRID", style={"color": ACCENT, "fontWeight": "900",
                                              "fontSize": "14px", "letterSpacing": "2.5px"}),
            html.Span(" OPTIMIZER", style={"color": TEXT, "fontWeight": "400",
                                             "fontSize": "14px", "letterSpacing": "2.5px"}),
        ]),
        html.Div("AI-Powered Substation Placement",
                 style={"fontSize": "11px", "color": TEXT3, "marginTop": "3px"}),
    ]),

    html.Hr(style={"borderColor": BORDER, "margin": "6px 18px 14px"}),

    # City input
    html.Div(style={"padding": "0 18px 12px"}, children=[
        _section("Target City"),
        dbc.Input(id="city-input", value="Austin, Texas, USA",
                  placeholder="e.g. Houston, Texas, USA",
                  style={"background": SURFACE, "border": f"1px solid {BORDER}",
                         "color": TEXT, "fontSize": "13px", "borderRadius": "6px"}),
    ]),

    # Weight sliders
    html.Div(style={"padding": "0 18px 8px"}, children=[
        _section("Optimization Weights"),
        *[html.Div([
            html.Div([
                html.Span(label, style={"color": TEXT2, "fontSize": "11px", "fontWeight": "600"}),
                html.Span(id=f"val-{sid}", children=f"{val:.0%}",
                          style={"color": ACCENT, "fontSize": "11px", "fontWeight": "700"}),
            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "2px"}),
            dcc.Slider(id=sid, min=0, max=1, step=0.05, value=val,
                       marks=None, tooltip={"always_visible": False}),
        ], style={"marginBottom": "8px"})
        for label, sid, val in [
            ("Load Relief",    "w-load",    0.35),
            ("Loss Reduction", "w-loss",    0.35),
            ("Sustainability", "w-sustain", 0.15),
            ("Redundancy",     "w-redund",  0.15),
        ]],
    ]),

    # Grid size
    html.Div(style={"padding": "0 18px 12px"}, children=[
        _section("Grid Resolution"),
        dbc.RadioItems(id="grid-size",
            options=[{"label": " Fast (200x200)", "value": 200},
                     {"label": " Standard (500x500)", "value": 500}],
            value=200, inline=True,
            style={"fontSize": "12px"},
            input_style={"marginRight": "4px"},
            label_style={"marginRight": "14px", "fontSize": "11px", "color": TEXT2}),
    ]),

    html.Hr(style={"borderColor": BORDER, "margin": "2px 18px 12px"}),

    # Run button
    html.Div(style={"padding": "0 18px 14px"}, children=[
        html.Button("Run Full Pipeline", id="run-btn", n_clicks=0, style={
            "width": "100%", "padding": "12px", "border": "none",
            "borderRadius": "8px", "fontWeight": "800", "fontSize": "13px",
            "letterSpacing": "1px", "cursor": "pointer", "color": "#000",
            "background": f"linear-gradient(135deg, {ACCENT}, #818cf8)",
            "boxShadow": f"0 4px 14px {ACCENT}33",
        }),
    ]),

    # Progress + metrics
    html.Div(id="progress-area", style={"padding": "0 18px 10px"}),
    html.Div(id="metrics-area",  style={"padding": "0 18px 10px"}),
    html.Div(id="status-text",   style={"padding": "0 18px 10px", "fontSize": "11px",
                                         "color": TEXT3, "minHeight": "14px"}),

    dcc.Store(id="selected-idx"),
    dcc.Interval(id="poll", interval=1200, n_intervals=0, disabled=True),
])


# ── Right sidebar ─────────────────────────────────────────────────────────────

_right = html.Div(style={
    "background": BG, "borderLeft": f"1px solid {BORDER}",
    "height": "100vh", "overflowY": "auto", "fontFamily": FONT,
    "display": "flex", "flexDirection": "column",
}, children=[
    html.Div(style={"padding": "20px 16px 6px"}, children=[_section("Top Candidates")]),
    html.Div(id="candidate-list"),
    html.Hr(style={"borderColor": BORDER, "margin": "6px 16px"}),
    html.Div(style={"padding": "0 16px 20px", "flex": "1"}, children=[
        _section("Site Intelligence Report"),
        html.Div(id="report-panel", children=[
            html.Div("Click a candidate to view analysis.",
                     style={"color": TEXT3, "fontSize": "12px", "fontStyle": "italic",
                            "padding": "16px 0"}),
        ]),
    ]),
])


app.layout = html.Div(style={
    "fontFamily": FONT, "background": BG, "overflow": "hidden", "margin": 0,
}, children=[
    html.Div(style={
        "display": "grid", "gridTemplateColumns": "270px 1fr 310px", "height": "100vh",
    }, children=[
        _left,
        dcc.Graph(id="map", figure=build_map(), style={"height": "100vh"},
                  config={"displayModeBar": True,
                          "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                          "displaylogo": False}),
        _right,
    ]),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

for _sid in ("w-load", "w-loss", "w-sustain", "w-redund"):
    @app.callback(Output(f"val-{_sid}", "children"), Input(_sid, "value"))
    def _update_label(v, __sid=_sid): return f"{v:.0%}"


@app.callback(
    Output("poll", "disabled"),
    Output("progress-area", "children"),
    Output("metrics-area",  "children"),
    Output("status-text",   "children"),
    Output("map",           "figure"),
    Output("candidate-list","children"),
    Input("run-btn",        "n_clicks"),
    Input("poll",           "n_intervals"),
    State("city-input",     "value"),
    State("w-load",         "value"),
    State("w-loss",         "value"),
    State("w-sustain",      "value"),
    State("w-redund",       "value"),
    State("grid-size",      "value"),
    prevent_initial_call=True,
)
def handle_run_poll(n_clicks, n_int, city, wl, wloss, ws, wr, grid):
    global BOUNDS, MASK, HEATMAP, GEOJSON, CANDIDATES
    triggered = ctx.triggered_id
    NO = dash.no_update

    if triggered == "run-btn":
        s = _get_state()
        if s["status"] == "running":
            return NO, NO, NO, "Already running...", NO, NO
        weights = {"load_relief": wl or 0.35, "loss_reduction": wloss or 0.35,
                   "sustainability": ws or 0.15, "redundancy": wr or 0.15}
        threading.Thread(target=_run_pipeline_bg,
                         args=(city or "Austin, Texas, USA", weights, grid or 200, 10),
                         daemon=True).start()
        prog = html.Div([
            dbc.Progress(value=5, max=100, style={"height": "6px", "background": SURFACE2,
                         "borderRadius": "3px"}, color="info"),
            html.Div("Starting pipeline...", style={"fontSize": "11px", "color": ACCENT,
                                                     "marginTop": "6px", "fontWeight": "600"}),
        ])
        return False, prog, [], "", NO, NO

    # Poll
    s = _get_state()

    if s["status"] == "idle":
        return True, NO, NO, NO, NO, NO

    if s["status"] == "running":
        prog = html.Div([
            dbc.Progress(value=s["progress"], max=100, style={"height": "6px",
                         "background": SURFACE2, "borderRadius": "3px"},
                         color="info", animated=True, striped=True),
            html.Div(s["step"], style={"fontSize": "11px", "color": ACCENT,
                                        "marginTop": "6px", "fontWeight": "600"}),
        ])
        return False, prog, NO, "", NO, NO

    if s["status"] == "done":
        BOUNDS, MASK, HEATMAP, GEOJSON, CANDIDATES = _load_all()
        _update(status="idle")
        m = s.get("metrics", {})
        prog = html.Div([
            dbc.Progress(value=100, max=100, style={"height": "6px", "background": SURFACE2,
                         "borderRadius": "3px"}, color="success"),
            html.Div("Pipeline complete!", style={"fontSize": "11px", "color": GREEN,
                                                    "marginTop": "6px", "fontWeight": "700"}),
        ])
        met = html.Div([
            html.Div(style={"display": "flex", "gap": "6px", "marginBottom": "6px"}, children=[
                _metric_card("Total", f"{m.get('total_time', '?')}s"),
                _metric_card("Grid", m.get("grid_size", "?")),
            ]),
            html.Div(style={"display": "flex", "gap": "6px"}, children=[
                _metric_card("Ingest", f"{m.get('ingest_time', '?')}s"),
                _metric_card("Optimize", f"{m.get('optimize_time', '?')}s"),
                *([_metric_card("AI", f"{m.get('ai_time', '?')}s")] if m.get("ai_time") else []),
            ]),
        ])
        return True, prog, met, "", build_map(), _build_candidates()

    if s["status"] == "error":
        _update(status="idle")
        prog = html.Div([
            dbc.Progress(value=s["progress"], max=100, style={"height": "6px",
                         "background": SURFACE2, "borderRadius": "3px"}, color="danger"),
            html.Div(s["step"][:120], style={"fontSize": "11px", "color": RED,
                                               "marginTop": "6px", "fontWeight": "600"}),
        ])
        return True, prog, NO, "", NO, NO

    return True, NO, NO, NO, NO, NO


def _build_candidates(cands=None):
    cs = cands if cands is not None else CANDIDATES
    if not cs:
        return html.Div([
            html.Div("No candidates yet", style={"color": TEXT, "fontWeight": "700",
                                                   "fontSize": "13px", "marginBottom": "4px"}),
            html.Div("Configure and click Run Full Pipeline.",
                     style={"color": TEXT3, "fontSize": "12px"}),
        ], style={"textAlign": "center", "padding": "20px 16px",
                  "border": f"1px dashed {BORDER}", "borderRadius": "8px", "margin": "0 16px"})
    rows = []
    for i, c in enumerate(cs[:10]):
        f     = c.get("feasibility", "UNKNOWN")
        fc    = FCOLORS.get(f, TEXT3)
        score = c.get("composite_score", 0)
        rows.append(html.Div(
            id={"type": "cand-btn", "index": i}, n_clicks=0,
            children=html.Div([
                html.Div(f"#{c.get('rank', i+1)}", style={
                    "color": ACCENT, "fontWeight": "900", "fontSize": "14px", "minWidth": "26px"}),
                html.Div([
                    html.Div(f"{c['lat']:.4f}, {c['lon']:.4f}", style={
                        "color": TEXT, "fontSize": "11px", "fontWeight": "600", "marginBottom": "1px"}),
                    html.Div(f"Score: {score:.3f}", style={"color": TEXT3, "fontSize": "10px"}),
                ], style={"flex": "1", "minWidth": "0"}),
                _badge(f, fc),
            ], style={"display": "flex", "alignItems": "center", "gap": "8px"}),
            style={"padding": "9px 16px", "cursor": "pointer",
                   "borderLeft": f"3px solid {fc}", "borderBottom": f"1px solid {BORDER}",
                   "background": BG, "transition": "background 0.15s"},
        ))
    return html.Div(rows)


@app.callback(
    Output("map", "figure", allow_duplicate=True),
    Output("selected-idx", "data"),
    Input({"type": "cand-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def on_select(n_clicks):
    if not any(n for n in (n_clicks or []) if n):
        return dash.no_update, dash.no_update
    triggered = ctx.triggered_id
    if triggered is None:
        return dash.no_update, dash.no_update
    idx = triggered["index"]
    if idx >= len(CANDIDATES):
        return dash.no_update, dash.no_update
    c = CANDIDATES[idx]
    return build_map(center_lat=c["lat"], center_lon=c["lon"], zoom=14), idx


@app.callback(
    Output("report-panel", "children"),
    Input("selected-idx", "data"),
)
def show_report(idx):
    if idx is None or not CANDIDATES or idx >= len(CANDIDATES):
        return html.Div("Click a candidate to view analysis.",
                        style={"color": TEXT3, "fontSize": "12px", "fontStyle": "italic",
                               "padding": "16px 0"})

    c     = CANDIDATES[idx]
    f     = c.get("feasibility", "UNKNOWN")
    fc    = FCOLORS.get(f, TEXT3)
    score = c.get("composite_score", 0)
    rank  = c.get("rank", idx + 1)

    # Objective bars
    obj_bars = []
    for key, label, clr in [
        ("load_relief_score",    "Load Relief",    "#38bdf8"),
        ("loss_reduction_score", "Loss Reduction",  "#818cf8"),
        ("sustainability_score", "Sustainability",  "#22c55e"),
        ("redundancy_score",     "Redundancy",      "#f59e0b"),
    ]:
        v = c.get(key)
        if v is not None:
            pct = float(v) * 100
            obj_bars.append(html.Div([
                html.Div([
                    html.Span(label, style={"color": TEXT2, "fontSize": "10px", "fontWeight": "600"}),
                    html.Span(f"{v:.3f}", style={"color": TEXT, "fontSize": "10px", "fontWeight": "700"}),
                ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "2px"}),
                html.Div(style={"background": SURFACE2, "borderRadius": "3px",
                                "height": "4px", "overflow": "hidden"}, children=[
                    html.Div(style={"background": clr, "height": "100%",
                                    "width": f"{min(pct, 100):.0f}%", "borderRadius": "3px"}),
                ]),
            ], style={"marginBottom": "6px"}))

    # Coverage chips
    chips = []
    for r in [3, 5, 10]:
        v = c.get(f"coverage_{r}km_pct")
        if v is not None:
            chips.append(html.Span(f"{r}km: {v:.1f}%", style={
                "background": SURFACE2, "border": f"1px solid {BORDER}", "borderRadius": "4px",
                "padding": "2px 7px", "fontSize": "10px", "color": TEXT2, "marginRight": "5px"}))
    nearest = c.get("nearest_existing_km")
    if nearest is not None:
        chips.append(html.Span(f"Nearest: {nearest:.1f}km", style={
            "background": SURFACE2, "border": f"1px solid {BORDER}", "borderRadius": "4px",
            "padding": "2px 7px", "fontSize": "10px", "color": TEXT2}))

    # AI Site Intelligence fields
    intel = []
    for key, label in [("land_use", "Land Use"), ("zoning_assessment", "Zoning"),
                       ("community_sensitivity", "Community"), ("grid_proximity", "Grid Proximity")]:
        val = c.get(key, "")
        if val:
            intel.append(html.Div([
                html.Div(label, style={"fontSize": "9px", "color": TEXT3, "fontWeight": "700",
                                        "letterSpacing": "0.5px", "marginBottom": "2px",
                                        "textTransform": "uppercase"}),
                html.Div(str(val), style={"color": TEXT2, "fontSize": "11px", "lineHeight": "1.5"}),
            ], style={"marginBottom": "6px", "padding": "6px 10px",
                      "background": SURFACE2, "borderRadius": "6px",
                      "border": f"1px solid {BORDER}"}))

    env_flags = c.get("environmental_flags", [])
    if env_flags:
        intel.append(html.Div([
            html.Div("Environmental Flags", style={"fontSize": "9px", "color": TEXT3,
                                                     "fontWeight": "700", "marginBottom": "4px",
                                                     "textTransform": "uppercase"}),
            html.Div([_badge(fl, AMBER) for fl in env_flags],
                     style={"display": "flex", "flexWrap": "wrap", "gap": "4px"}),
        ], style={"marginBottom": "6px", "padding": "6px 10px",
                  "background": SURFACE2, "borderRadius": "6px",
                  "border": f"1px solid {BORDER}"}))

    reasoning = c.get("reasoning", "No detailed analysis available.")

    return html.Div([
        # Header
        html.Div([
            html.Span(f"Rank #{rank}", style={"color": ACCENT, "fontWeight": "900",
                                                "fontSize": "18px"}),
            _badge(f, fc),
        ], style={"display": "flex", "alignItems": "center", "gap": "10px",
                  "marginBottom": "10px"}),

        # Score card
        html.Div([
            html.Div("COMPOSITE SCORE", style={"fontSize": "9px", "color": TEXT3,
                                                 "fontWeight": "700", "letterSpacing": "1.5px"}),
            html.Div(f"{score:.4f}", style={"fontSize": "26px", "fontWeight": "900",
                                              "color": TEXT, "lineHeight": "1.1"}),
            html.Div(f"{c['lat']:.5f}N, {abs(c['lon']):.5f}W",
                     style={"fontSize": "10px", "color": TEXT3, "marginTop": "3px"}),
        ], style={"background": SURFACE, "border": f"1px solid {fc}44",
                  "borderRadius": "8px", "padding": "12px", "marginBottom": "10px"}),

        # Objectives
        *([html.Div(obj_bars, style={"marginBottom": "6px"})] if obj_bars else []),

        # Coverage
        *([html.Div(chips, style={"marginBottom": "10px", "display": "flex",
                                   "flexWrap": "wrap"})] if chips else []),

        # AI Intelligence
        *([html.Div([
            html.Hr(style={"borderColor": BORDER, "margin": "6px 0 8px"}),
            html.Div("AI SITE INTELLIGENCE", style={
                "fontSize": "10px", "color": ACCENT, "fontWeight": "700",
                "letterSpacing": "1.5px", "marginBottom": "8px"}),
            *intel,
        ])] if intel else []),

        # Reasoning
        html.Hr(style={"borderColor": BORDER, "margin": "6px 0 8px"}),
        html.Div("ANALYSIS", style={"fontSize": "10px", "color": TEXT3,
                                      "fontWeight": "700", "letterSpacing": "1.5px",
                                      "marginBottom": "4px"}),
        html.Div(reasoning, style={"color": TEXT2, "fontSize": "11px",
                                    "lineHeight": "1.7", "whiteSpace": "pre-wrap"}),
    ])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    n = len(CANDIDATES)
    print()
    print("  Energy Grid Optimizer")
    print("  ---------------------")
    print(f"  Candidates : {n} loaded")
    print(f"  -> http://localhost:8050")
    print()
    app.run(debug=False, host="0.0.0.0", port=8050)
