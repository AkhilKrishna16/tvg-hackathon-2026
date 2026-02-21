"""
map_layers.py
─────────────
Pure builder functions — each returns a single go.Scattermapbox trace.
No I/O, no global state, fully testable in isolation.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# ── Feasibility color palette ─────────────────────────────────────────────────
FEASIBILITY_COLORS: dict[str, str] = {
    "HIGH":   "#00c853",  # vivid green
    "MEDIUM": "#ffd600",  # vivid amber
    "LOW":    "#d50000",  # vivid red
}

_FALLBACK_COLOR = "#ffd600"

# ── Internal helpers ──────────────────────────────────────────────────────────

def _grid_to_latlon(
    rows: np.ndarray,
    cols: np.ndarray,
    bounds: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Map (row, col) grid indices to (lat, lon) geographic coordinates."""
    lats = bounds["north"] - (rows / 500.0) * (bounds["north"] - bounds["south"])
    lons = bounds["west"]  + (cols / 500.0) * (bounds["east"]  - bounds["west"])
    return lats, lons


def _sample(
    rows: np.ndarray,
    cols: np.ndarray,
    n: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Randomly downsample without replacement. Reproducible via seed."""
    if len(rows) <= n:
        return rows, cols
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(rows), size=n, replace=False)
    return rows[idx], cols[idx]


# ── Public layer builders ─────────────────────────────────────────────────────

def build_demand_layer(heatmap: np.ndarray, bounds: dict) -> go.Scattermapbox:
    """
    Render demand intensity as a scatter heatmap (~2 000 sampled points).
    Blue = low demand  →  Red = high demand.
    """
    rows, cols = np.where(heatmap > 0)
    rows, cols = _sample(rows, cols, n=2_000)
    lats, lons = _grid_to_latlon(rows, cols, bounds)
    values = heatmap[rows, cols].tolist()

    return go.Scattermapbox(
        lat=lats.tolist(),
        lon=lons.tolist(),
        mode="markers",
        marker=dict(
            size=6,
            color=values,
            colorscale="RdYlBu_r",
            cmin=float(heatmap[heatmap > 0].min()) if np.any(heatmap > 0) else 0,
            cmax=float(heatmap.max()),
            opacity=0.45,
            colorbar=dict(
                title=dict(
                    text="Demand",
                    font=dict(color="#8892b0", size=10),
                    side="right",
                ),
                tickfont=dict(color="#8892b0", size=9),
                bgcolor="rgba(13,13,26,0.85)",
                bordercolor="#2a2a4a",
                borderwidth=1,
                x=0.01,
                xanchor="left",
                y=0.5,
                len=0.35,
                thickness=10,
            ),
        ),
        hoverinfo="skip",
        showlegend=True,
        name="Demand Load",
    )


def build_forbidden_layer(mask: np.ndarray, bounds: dict) -> go.Scattermapbox:
    """
    Render forbidden/infeasible zones as faint red dots (~500 sampled points).
    mask == 0  →  forbidden cell.
    """
    rows, cols = np.where(mask == 0)
    rows, cols = _sample(rows, cols, n=500, seed=43)
    lats, lons = _grid_to_latlon(rows, cols, bounds)

    return go.Scattermapbox(
        lat=lats.tolist(),
        lon=lons.tolist(),
        mode="markers",
        marker=dict(
            size=4,
            color="#ff1744",
            opacity=0.18,
        ),
        hoverinfo="skip",
        showlegend=True,
        name="Forbidden Zones",
    )


def build_substations_layer(geojson: dict) -> go.Scattermapbox:
    """
    Render existing substation locations as black diamonds with name hover.
    GeoJSON feature coordinates are [lon, lat].
    """
    features = geojson.get("features", [])
    if not features:
        return go.Scattermapbox(lat=[], lon=[], name="Existing Substations")

    lats  = [f["geometry"]["coordinates"][1] for f in features]
    lons  = [f["geometry"]["coordinates"][0] for f in features]
    names = [f.get("properties", {}).get("name", "Substation") for f in features]

    return go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=dict(
            size=14,
            color="#ffffff",
            symbol="marker",           # mapbox doesn't support 'diamond' symbol inline
            opacity=0.95,
        ),
        text=names,
        customdata=names,
        hovertemplate=(
            "<b>%{customdata}</b><br>"
            "<span style='color:#8892b0'>Existing Substation</span>"
            "<extra></extra>"
        ),
        showlegend=True,
        name="Existing Substations",
    )


def build_candidates_layer(candidates: list[dict]) -> go.Scattermapbox:
    """
    Render top candidate placements as colored stars.

    Size   → scaled by rank (rank 1 = largest)
    Color  → feasibility: HIGH=green, MEDIUM=amber, LOW=red
    Hover  → rank, score, feasibility, reasoning
    """
    if not candidates:
        return go.Scattermapbox(lat=[], lon=[], name="Candidate Sites")

    lats   = [c["lat"]  for c in candidates]
    lons   = [c["lon"]  for c in candidates]
    sizes  = [max(10, 24 - (c.get("rank", 1) - 1) * 2) for c in candidates]
    colors = [FEASIBILITY_COLORS.get(c.get("feasibility", ""), _FALLBACK_COLOR) for c in candidates]

    hover_texts = []
    for c in candidates:
        f      = c.get("feasibility", "N/A")
        fcolor = FEASIBILITY_COLORS.get(f, _FALLBACK_COLOR)
        score  = c.get("composite_score", 0)
        reason = c.get("reasoning", "No analysis available.")
        # Truncate long reasoning for hover card legibility
        if len(reason) > 180:
            reason = reason[:177] + "…"
        hover_texts.append(
            f"<b>Rank #{c.get('rank', '?')}</b>  ·  "
            f"<span style='color:{fcolor}'><b>{f}</b></span><br>"
            f"Composite Score: <b>{score:.3f}</b><br>"
            f"<br>"
            f"<span style='color:#8892b0'>{reason}</span>"
        )

    return go.Scattermapbox(
        lat=lats,
        lon=lons,
        mode="markers",
        marker=dict(
            size=sizes,
            color=colors,
            opacity=0.95,
            allowoverlap=True,
        ),
        text=hover_texts,
        customdata=[c.get("rank", i + 1) for i, c in enumerate(candidates)],
        hovertemplate="%{text}<extra></extra>",
        showlegend=True,
        name="Candidate Sites",
    )
