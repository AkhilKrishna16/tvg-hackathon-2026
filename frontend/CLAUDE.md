# Grid Placement Optimizer — Visualization Frontend

> **Hackathon role:** Person 4 — Visualization & Frontend
> **Branch:** `feature/viz`
> **Stack:** Python · Plotly Dash · Dash Bootstrap Components · GeoPandas · NumPy
> **Goal:** Win the demo. Make city planners lean forward in their seats.

---

## Mission

Build a Plotly Dash dashboard that visualizes optimal EV charging substation placements across Austin, TX. This is the **face of the entire project**. The ML team finds candidates; we make them undeniable. Every pixel counts.

---

## Deliverables

| File | Purpose |
|---|---|
| `src/viz/app.py` | Main Dash app — layout, callbacks, startup data loading |
| `src/viz/map_layers.py` | Pure functions returning `go.Scattermapbox` traces |

App must be launchable with:

```bash
python src/viz/app.py
# → http://localhost:8050
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Dash App (port 8050)                     │
│                                                                 │
│  ┌──────────────────────────────┐  ┌────────────────────────┐  │
│  │       Map Panel (70%)        │  │    Sidebar (30%)        │  │
│  │                              │  │                         │  │
│  │  Layer 1: Demand Heatmap     │  │  Title + Tagline        │  │
│  │  Layer 2: Forbidden Zones    │  │  Objective Sliders x4   │  │
│  │  Layer 3: Existing Substns.  │  │  Ranked Candidate List  │  │
│  │  Layer 4: Top Candidates ★  │  │  Feasibility Report     │  │
│  │                              │  │                         │  │
│  └──────────────────────────────┘  └────────────────────────┘  │
│                         dcc.Store (selected candidate state)    │
└─────────────────────────────────────────────────────────────────┘
```

Data loads **once at startup** into module-level variables. No re-reads on callback.

---

## Data Inputs

All files live in `data/` or `results/` relative to the project root (one level up from `src/`):

| File | Shape / Format | Description |
|---|---|---|
| `data/forbidden_mask.npy` | `(500, 500)` float | `0` = forbidden, `1` = feasible |
| `data/demand_heatmap.npy` | `(500, 500)` float | Normalized demand load per cell |
| `data/existing_substations.geojson` | GeoJSON FeatureCollection | Points with `name` property |
| `data/city_bounds.json` | `{south, north, west, east}` | Austin bounding box |
| `results/top_candidates.json` | JSON array | Ranked candidate placements |

### Candidate Schema (`top_candidates.json`)

```json
[
  {
    "rank": 1,
    "lat": 30.2849,
    "lon": -97.7341,
    "composite_score": 0.91,
    "feasibility": "HIGH",
    "reasoning": "High demand zone, 2.1 km from nearest substation..."
  }
]
```

---

## Coordinate Mapping

The `.npy` arrays are `500×500` grids indexed `[row, col]`. Convert to geographic coords:

```python
lat = bounds["north"] - (row / 500) * (bounds["north"] - bounds["south"])
lon = bounds["west"]  + (col / 500) * (bounds["east"]  - bounds["west"])
```

Always use `np.random.seed(42)` before sampling for reproducible point selection.

---

## `map_layers.py` — Layer Builder API

Each function returns a **single Plotly trace** ready to splice into a figure. No side effects.

### `build_demand_layer(heatmap: np.ndarray, bounds: dict) -> go.Scattermapbox`

- Sample ~2 000 points randomly from non-zero cells
- `marker.color` = demand value, `colorscale='RdYlBu_r'` (blue=low, red=high)
- `marker.opacity` = `0.4`, `marker.size` = `6`
- `name="Demand Load"`

### `build_forbidden_layer(mask: np.ndarray, bounds: dict) -> go.Scattermapbox`

- Sample ~500 points where `mask == 0`
- `marker.color = 'red'`, `marker.opacity = 0.2`, `marker.size = 4`
- `name="Forbidden Zones"`

### `build_substations_layer(geojson: dict) -> go.Scattermapbox`

- One marker per feature
- `marker.symbol = 'diamond'`, `marker.size = 12`, `marker.color = 'black'`
- `hovertext` = feature `properties.name`
- `name="Existing Substations"`

### `build_candidates_layer(candidates: list) -> go.Scattermapbox`

- Gold star markers: `marker.symbol = 'star'`
- Size scaled by rank: `size = 20 - (rank - 1) * 2` (rank 1 = biggest)
- Color by feasibility:

```python
FEASIBILITY_COLORS = {
    "HIGH":   "#00c853",   # vivid green
    "MEDIUM": "#ffd600",   # vivid amber
    "LOW":    "#d50000",   # vivid red
}
```

- Hover template:

```
Rank #1
Score: 0.91
Feasibility: HIGH
────────────────
High demand zone, 2.1 km from nearest substation...
```

- `name="Candidate Sites"`

---

## `app.py` — Application Blueprint

### Startup (module level)

```python
# Load once, cache globally
bounds     = json.load(open(DATA / "city_bounds.json"))
mask       = np.load(DATA / "forbidden_mask.npy")
heatmap    = np.load(DATA / "demand_heatmap.npy")
geojson    = json.load(open(DATA / "existing_substations.geojson"))
candidates = json.load(open(RESULTS / "top_candidates.json")) if (RESULTS / "top_candidates.json").exists() else []
```

### Layout

```
dbc.Container(fluid=True)
└── dbc.Row
    ├── dbc.Col(width=8)   ← map panel
    │   └── dcc.Graph(id="map", figure=build_map(), style={"height": "100vh"})
    └── dbc.Col(width=4)   ← sidebar
        ├── Title + tagline
        ├── Objective sliders (disabled=True, display only)
        ├── Candidate list (dbc.ListGroup)
        ├── Feasibility report (dbc.Card)
        └── dcc.Store(id="selected-candidate")
```

### Map Figure

```python
fig = go.Figure()
fig.add_trace(build_demand_layer(heatmap, bounds))
fig.add_trace(build_forbidden_layer(mask, bounds))
fig.add_trace(build_substations_layer(geojson))
if candidates:
    fig.add_trace(build_candidates_layer(candidates))
fig.update_layout(
    mapbox_style="open-street-map",
    mapbox_center={"lat": 30.2672, "lon": -97.7431},  # Austin center
    mapbox_zoom=10,
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    paper_bgcolor="#0d0d1a",
    legend=dict(bgcolor="#1a1a2e", font=dict(color="white")),
)
```

### Callbacks

**Click candidate → center map:**

```python
@app.callback(
    Output("map", "figure"),
    Output("selected-candidate", "data"),
    Input({"type": "candidate-btn", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_candidate(n_clicks):
    idx = ctx.triggered_id["index"]
    c = candidates[idx]
    # Re-center mapbox, update selected store
```

**Selected candidate → feasibility report:**

```python
@app.callback(
    Output("feasibility-report", "children"),
    Input("selected-candidate", "data"),
)
def update_report(data): ...
```

### Entry Point

```python
if __name__ == "__main__":
    app.run(debug=True, port=8050)
```

---

## Objective Sliders (Display Only)

Show the 4 optimization weights used. Labels and default values:

| Label | Default |
|---|---|
| Demand Coverage | 0.40 |
| Distance to Grid | 0.30 |
| Land Cost Penalty | 0.20 |
| Redundancy Bonus | 0.10 |

Use `dcc.Slider(min=0, max=1, step=0.05, value=<default>, disabled=True)`.

---

## Graceful Degradation

If `results/top_candidates.json` does not exist, the sidebar shows:

```
⚠️  No candidates yet.

Run the optimizer first:
  python src/optimizer/main.py

The map will auto-update when results are ready.
```

The map still renders with demand, forbidden zones, and existing substations.

---

## Styling — Dark Command Center Aesthetic

```
Background (sidebar):   #0d0d1a
Surface cards:          #1a1a2e
Accent / primary:       #00d4ff   (electric cyan)
Success / HIGH:         #00c853
Warning / MEDIUM:       #ffd600
Danger / LOW:           #d50000
Text primary:           #ffffff
Text secondary:         #8892b0
Border:                 #2a2a4a
Font:                   system-ui, -apple-system, "Segoe UI"
```

Sidebar title treatment:

```
GRID PLACEMENT OPTIMIZER
Austin, TX  ·  EV Infrastructure AI
```

Candidate list items should have a left border color matching feasibility (`border-left: 4px solid <color>`).

---

## Dependencies

```bash
pip install dash plotly dash-bootstrap-components geopandas numpy
```

Bootstrap theme: `dbc.themes.DARKLY` or `CYBORG` for a polished dark look.

---

## Quick-Start

```bash
# From repo root
pip install dash plotly dash-bootstrap-components geopandas numpy

# Run
python src/viz/app.py

# Open
open http://localhost:8050
```

---

## Execution Priorities

1. **Map renders with all 4 layers** — non-negotiable for the demo
2. **Candidate hover cards** — judges will mouse over these
3. **Sidebar candidate list with click-to-zoom** — shows interactivity
4. **Feasibility report panel** — shows the AI reasoning, impressive to non-technical judges
5. **Sliders + visual polish** — differentiates from a notebook screenshot

Build in this order. Ship something beautiful.
