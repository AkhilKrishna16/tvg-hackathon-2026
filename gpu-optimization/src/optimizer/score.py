"""
score.py — Composite scoring engine for substation placement candidates.

Combines four objective functions into a single weighted score map,
zeros out forbidden cells, and returns both the composite and individual arrays.
"""

from __future__ import annotations

import time
from typing import Optional, Union

import numpy as np

try:
    import cupy as cp  # type: ignore
    _BACKEND = "cupy"
except Exception:
    cp = None  # type: ignore
    _BACKEND = "numpy"

from .objectives import (
    load_relief_score,
    loss_reduction_score,
    redundancy_score,
    sustainability_score,
    backend_name,
)

# ---------------------------------------------------------------------------
# Default weights (must sum to 1.0)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS: dict[str, float] = {
    "load_relief":    0.35,
    "loss_reduction": 0.35,
    "sustainability": 0.15,
    "redundancy":     0.15,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

ArrayLike = Union[np.ndarray, "cp.ndarray"]  # type: ignore


def _to_numpy(arr: ArrayLike) -> np.ndarray:
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def composite_score(
    demand_heatmap: ArrayLike,
    existing_substations: dict,
    city_bounds: dict,
    forbidden_mask: Optional[ArrayLike] = None,
    weights: Optional[dict[str, float]] = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """
    Compute the weighted composite score across all four objective functions.

    Parameters
    ----------
    demand_heatmap : array-like, shape (500, 500)
        Normalised demand per grid cell.
    existing_substations : dict
        GeoJSON FeatureCollection of current substation locations.
    city_bounds : dict
        Bounding box with keys {south, north, west, east}.
    forbidden_mask : array-like, shape (500, 500), optional
        Binary mask; 1 = placeable, 0 = forbidden. If None, all cells are
        treated as placeable.
    weights : dict, optional
        Override one or more default weights. Keys must be a subset of
        {load_relief, loss_reduction, sustainability, redundancy}.
        Supplied weights are merged with defaults and then re-normalised
        so they always sum to 1.

    Returns
    -------
    composite : np.ndarray, shape (500, 500), float32
        Weighted composite score. Forbidden cells are zeroed out.
    individual : dict[str, np.ndarray]
        Individual score arrays keyed by objective name.
    timings : dict[str, float]
        Wall-clock seconds for each scoring step and a "total" entry.
    """
    # --- Resolve weights ---
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_w = sum(w.values())
    if total_w <= 0:
        raise ValueError("Weight values must be positive.")
    w = {k: v / total_w for k, v in w.items()}  # normalise to sum=1

    timings: dict[str, float] = {}
    individual: dict[str, np.ndarray] = {}

    # --- Objective 1: Load Relief ---
    t0 = time.perf_counter()
    individual["load_relief"] = _to_numpy(
        load_relief_score(demand_heatmap, existing_substations, city_bounds)
    )
    timings["load_relief"] = time.perf_counter() - t0

    # --- Objective 2: Loss Reduction ---
    t0 = time.perf_counter()
    individual["loss_reduction"] = _to_numpy(
        loss_reduction_score(demand_heatmap, existing_substations, city_bounds)
    )
    timings["loss_reduction"] = time.perf_counter() - t0

    # --- Objective 3: Sustainability ---
    t0 = time.perf_counter()
    individual["sustainability"] = _to_numpy(
        sustainability_score(city_bounds)
    )
    timings["sustainability"] = time.perf_counter() - t0

    # --- Objective 4: Redundancy ---
    t0 = time.perf_counter()
    individual["redundancy"] = _to_numpy(
        redundancy_score(existing_substations, city_bounds)
    )
    timings["redundancy"] = time.perf_counter() - t0

    # --- Weighted sum ---
    t0 = time.perf_counter()
    composite = np.zeros((500, 500), dtype=np.float32)
    for name, arr in individual.items():
        composite += w[name] * arr

    # --- Apply forbidden mask ---
    if forbidden_mask is not None:
        mask_np = _to_numpy(forbidden_mask).astype(np.float32)
        composite *= mask_np
    timings["aggregation"] = time.perf_counter() - t0

    timings["total"] = sum(timings.values())
    return composite, individual, timings


def score_summary(
    individual: dict[str, np.ndarray],
    weights: Optional[dict[str, float]] = None,
) -> str:
    """Return a formatted string summarising per-objective score statistics."""
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}

    lines = [
        f"  {'Objective':<20} {'Weight':>6}  {'Min':>6}  {'Mean':>6}  {'Max':>6}",
        "  " + "-" * 54,
    ]
    for name, arr in individual.items():
        lines.append(
            f"  {name:<20} {w.get(name, 0):.3f}   "
            f"{arr.min():.4f}   {arr.mean():.4f}   {arr.max():.4f}"
        )
    return "\n".join(lines)


def active_backend() -> str:
    return backend_name()
