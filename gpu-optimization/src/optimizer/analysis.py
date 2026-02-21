"""
analysis.py — Post-processing analysis for substation placement candidates.

Provides:
  • grid_to_latlon / select_top_candidates  — candidate selection helpers
    (formerly in run.py; now centralised so they can be reused by analysis
    without creating circular imports)
  • coverage_analysis          — demand captured within radius of each candidate
  • nearest_substation_km      — distance to nearest existing substation
  • sensitivity_analysis       — stability across 8 weight-set combinations
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .objectives import _make_lat_lon_grids, _haversine_m, _extract_substations

# ---------------------------------------------------------------------------
# Weight sets used for sensitivity analysis
# ---------------------------------------------------------------------------

# (name, weights_dict) pairs — 8 combinations spanning the objective space
_WEIGHT_SETS: list[tuple[str, dict[str, float]]] = [
    ("default",       {"load_relief": 0.35, "loss_reduction": 0.35, "sustainability": 0.15, "redundancy": 0.15}),
    ("load_dominant", {"load_relief": 0.50, "loss_reduction": 0.30, "sustainability": 0.10, "redundancy": 0.10}),
    ("loss_dominant", {"load_relief": 0.30, "loss_reduction": 0.50, "sustainability": 0.10, "redundancy": 0.10}),
    ("sustain_focus", {"load_relief": 0.25, "loss_reduction": 0.25, "sustainability": 0.40, "redundancy": 0.10}),
    ("redund_focus",  {"load_relief": 0.25, "loss_reduction": 0.25, "sustainability": 0.10, "redundancy": 0.40}),
    ("equal",         {"load_relief": 0.25, "loss_reduction": 0.25, "sustainability": 0.25, "redundancy": 0.25}),
    ("economic",      {"load_relief": 0.45, "loss_reduction": 0.45, "sustainability": 0.05, "redundancy": 0.05}),
    ("resilience",    {"load_relief": 0.20, "loss_reduction": 0.20, "sustainability": 0.30, "redundancy": 0.30}),
]


# ---------------------------------------------------------------------------
# Coordinate conversion
# ---------------------------------------------------------------------------

def grid_to_latlon(
    row: int,
    col: int,
    city_bounds: dict,
    rows: int = 500,
    cols: int = 500,
) -> tuple[float, float]:
    """
    Linearly interpolate grid index (row, col) → (lat, lon).

    Convention:
        row 0   → north boundary latitude
        row N-1 → south boundary latitude
        col 0   → west boundary longitude
        col N-1 → east boundary longitude
    """
    north, south = city_bounds["north"], city_bounds["south"]
    west, east   = city_bounds["west"],  city_bounds["east"]
    lat = north - (row / (rows - 1)) * (north - south)
    lon = west  + (col / (cols - 1)) * (east  - west)
    return lat, lon


# ---------------------------------------------------------------------------
# Top-N greedy selection with anti-clustering
# ---------------------------------------------------------------------------

def select_top_candidates(
    composite: np.ndarray,
    individual: dict[str, np.ndarray],
    city_bounds: dict,
    n: int = 10,
    min_spacing_cells: int = 10,
) -> list[dict]:
    """
    Greedy best-first selection of the top ``n`` candidates.

    After each selection the composite map is zeroed within a Euclidean radius of
    ``min_spacing_cells`` grid cells to prevent spatial clustering of candidates.

    Parameters
    ----------
    composite : np.ndarray (500, 500)
        Composite score map; forbidden cells must already be zeroed.
    individual : dict[str, np.ndarray]
        Per-objective score arrays (load_relief, loss_reduction, sustainability, redundancy).
    city_bounds : dict
        Bounding box for coordinate conversion.
    n : int
        Number of candidates to return.
    min_spacing_cells : int
        Minimum grid-cell distance between any two selected candidates.

    Returns
    -------
    list of dicts, sorted by descending composite score.
    Each dict contains:
        rank, lat, lon, grid_row, grid_col, composite_score,
        load_relief_score, loss_reduction_score, sustainability_score, redundancy_score
    """
    score_map = composite.copy()
    rows, cols = score_map.shape

    # Pre-compute index grids once for suppression radius computation
    row_idx = np.arange(rows, dtype=np.int32)
    col_idx = np.arange(cols, dtype=np.int32)
    COL_GRID, ROW_GRID = np.meshgrid(col_idx, row_idx)  # both (rows, cols)

    candidates: list[dict] = []

    for rank in range(1, n + 1):
        flat_idx = int(np.argmax(score_map))
        best_i, best_j = np.unravel_index(flat_idx, score_map.shape)
        best_score = float(score_map[best_i, best_j])

        if best_score <= 0.0:
            break  # No more valid candidates

        lat, lon = grid_to_latlon(int(best_i), int(best_j), city_bounds, rows, cols)

        candidates.append(
            {
                "rank": rank,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "grid_row": int(best_i),
                "grid_col": int(best_j),
                "composite_score":      round(best_score, 6),
                "load_relief_score":    round(float(individual["load_relief"][best_i, best_j]),    6),
                "loss_reduction_score": round(float(individual["loss_reduction"][best_i, best_j]), 6),
                "sustainability_score": round(float(individual["sustainability"][best_i, best_j]), 6),
                "redundancy_score":     round(float(individual["redundancy"][best_i, best_j]),     6),
            }
        )

        # Suppress all cells within min_spacing_cells of the selected cell
        dist = np.sqrt(
            (ROW_GRID - best_i) ** 2 + (COL_GRID - best_j) ** 2
        )
        score_map[dist <= min_spacing_cells] = 0.0

    return candidates


# ---------------------------------------------------------------------------
# Coverage analysis
# ---------------------------------------------------------------------------

def coverage_analysis(
    candidates: list[dict],
    demand_heatmap: np.ndarray,
    city_bounds: dict,
    radii_km: list[float] = [3.0, 5.0, 10.0],
) -> list[dict]:
    """
    For each candidate location, compute the fraction of total city demand
    that falls within each of the given radii.

    Adds keys ``coverage_{r:.0f}km_pct`` (e.g., ``coverage_5km_pct``) to
    every candidate dict. Values are percentages rounded to 2 decimal places.

    Parameters
    ----------
    candidates : list of dicts — must have "lat" and "lon" keys.
    demand_heatmap : (500, 500) float32 array — normalised demand per cell.
    city_bounds : dict — bounding box.
    radii_km : list of radii to evaluate coverage at.

    Returns
    -------
    New list of candidate dicts with coverage keys added.
    """
    lat_grid, lon_grid = _make_lat_lon_grids(city_bounds)
    total_demand = float(demand_heatmap.sum())
    total_demand = max(total_demand, 1e-9)  # avoid division by zero

    enriched: list[dict] = []
    for c in candidates:
        entry = dict(c)
        # Distance from this candidate to every grid cell (Haversine, metres)
        dist_m = _haversine_m(
            lat_grid,
            lon_grid,
            np.float64(c["lat"]),
            np.float64(c["lon"]),
        )
        for r_km in radii_km:
            within = dist_m <= (r_km * 1_000.0)
            covered = float(demand_heatmap[within].sum())
            key = f"coverage_{r_km:.0f}km_pct"
            entry[key] = round(100.0 * covered / total_demand, 2)
        enriched.append(entry)

    return enriched


# ---------------------------------------------------------------------------
# Nearest substation distance
# ---------------------------------------------------------------------------

def nearest_substation_km(
    candidates: list[dict],
    existing_substations: dict,
) -> list[dict]:
    """
    Add ``nearest_existing_km`` to each candidate dict.

    Uses the Haversine formula; result is rounded to 3 decimal places.
    If there are no existing substations, the key is set to ``null``.

    Parameters
    ----------
    candidates : list of dicts with "lat" and "lon" keys.
    existing_substations : GeoJSON FeatureCollection.

    Returns
    -------
    New list of candidate dicts with ``nearest_existing_km`` added.
    """
    sub_lats, sub_lons = _extract_substations(existing_substations)
    enriched: list[dict] = []

    for c in candidates:
        entry = dict(c)
        if len(sub_lats) > 0:
            # Vectorized: compute distance from this candidate to all substations
            dists_m = _haversine_m(
                np.full(len(sub_lats), c["lat"]),
                np.full(len(sub_lats), c["lon"]),
                sub_lats,
                sub_lons,
            )
            entry["nearest_existing_km"] = round(float(dists_m.min()) / 1_000.0, 3)
        else:
            entry["nearest_existing_km"] = None
        enriched.append(entry)

    return enriched


# ---------------------------------------------------------------------------
# Sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    individual: dict[str, np.ndarray],
    forbidden_mask: np.ndarray,
    city_bounds: dict,
    n: int = 10,
    min_spacing_cells: int = 10,
) -> dict:
    """
    Re-run top-candidate selection across 8 weight-set combinations using
    pre-computed individual score arrays (no recomputation of objectives needed).

    This identifies "stable" candidate locations that appear consistently in the
    top-N regardless of how objectives are weighted — a proxy for robustness.

    Parameters
    ----------
    individual : dict mapping objective name → (500, 500) float32 score array.
    forbidden_mask : (500, 500) binary array; forbidden cells must be 0.
    city_bounds : dict
    n : candidates to select per weight set.
    min_spacing_cells : anti-clustering radius.

    Returns
    -------
    dict with keys:
        ``n_weight_sets``   : number of weight sets evaluated (8).
        ``weight_sets``     : list of weight dicts in the order evaluated.
        ``results_by_set``  : list of {name, weights, candidates} — one per weight set.
        ``stability_map``   : (500, 500) int32 array where each cell's value counts
                              how many weight sets selected that cell.
        ``stable_cells``    : list of (grid_row, grid_col, stability_count) tuples,
                              sorted by stability_count descending.
    """
    stability_map = np.zeros_like(forbidden_mask, dtype=np.int32)
    results_by_set: list[dict] = []

    for ws_name, ws in _WEIGHT_SETS:
        total_w = sum(ws.values())
        w = {k: v / total_w for k, v in ws.items()}

        # Weighted composite using pre-computed individual arrays
        composite = np.zeros_like(forbidden_mask, dtype=np.float32)
        for k, arr in individual.items():
            composite += w.get(k, 0.0) * arr
        composite *= forbidden_mask.astype(np.float32)

        selected = select_top_candidates(
            composite, individual, city_bounds, n, min_spacing_cells
        )
        results_by_set.append(
            {
                "name": ws_name,
                "weights": ws,
                "candidates": selected,
            }
        )

        for c in selected:
            stability_map[c["grid_row"], c["grid_col"]] += 1

    # Build stable_cells list (sorted by frequency)
    stable_cells = sorted(
        [
            (int(r), int(c), int(stability_map[r, c]))
            for r, c in zip(*np.where(stability_map > 0))
        ],
        key=lambda x: -x[2],
    )

    return {
        "n_weight_sets": len(_WEIGHT_SETS),
        "weight_sets": [ws for _, ws in _WEIGHT_SETS],
        "results_by_set": results_by_set,
        "stability_map": stability_map,
        "stable_cells": stable_cells,
    }
