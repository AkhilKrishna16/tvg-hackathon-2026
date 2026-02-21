"""
score.py — Composite scoring engine for substation placement candidates.

Combines four objective functions into a single weighted score map,
zeros out forbidden cells, and returns both the composite and individual arrays.

Improvements over v1:
  • Parallel execution (parallel=True by default): all four objectives are submitted
    to a ThreadPoolExecutor simultaneously. NumPy releases the GIL for heavy
    numerical operations, and the sustainability objective's network I/O runs
    concurrently with the CPU-bound objectives — yielding real wall-clock speedup.
  • score_summary now includes p25/p50/p75 percentiles for richer distribution insight.
  • _timed() helper cleanly measures per-objective elapsed time.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, Union

import numpy as np

try:
    import cupy as cp  # type: ignore
    _BACKEND = "cupy"
except Exception:
    cp = None  # type: ignore
    _BACKEND = "numpy"

from .objectives import (
    backend_name,
    load_relief_score,
    loss_reduction_score,
    redundancy_score,
    sustainability_score,
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


def _timed(fn, *args):
    """Call fn(*args) and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args)
    return result, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def composite_score(
    demand_heatmap: ArrayLike,
    existing_substations: dict,
    city_bounds: dict,
    forbidden_mask: Optional[ArrayLike] = None,
    weights: Optional[dict[str, float]] = None,
    parallel: bool = True,
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
        Binary mask; 1 = placeable, 0 = forbidden. If None, all cells scored.
    weights : dict, optional
        Override one or more default weights. Keys must be a subset of
        {load_relief, loss_reduction, sustainability, redundancy}.
        Supplied weights are merged with defaults and re-normalised to sum=1.
    parallel : bool, default True
        When True, all four objectives are evaluated concurrently using a
        ThreadPoolExecutor (max_workers=4).  NumPy releases the GIL for heavy
        compute, and the sustainability network request runs truly concurrently,
        reducing wall-clock time vs. sequential execution.

    Returns
    -------
    composite : np.ndarray, shape (500, 500), float32
        Weighted composite score. Forbidden cells are zeroed out.
    individual : dict[str, np.ndarray]
        Individual score arrays keyed by objective name.
    timings : dict[str, float]
        Wall-clock seconds for each scoring step and a "total" entry.
    """
    # --- Resolve and normalise weights ---
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_w = sum(w.values())
    if total_w <= 0:
        raise ValueError("Weight values must be positive (non-zero).")
    w = {k: v / total_w for k, v in w.items()}

    timings: dict[str, float] = {}
    individual: dict[str, np.ndarray] = {}

    # Derive actual grid shape from the demand array so all objectives match
    demand_arr = _to_numpy(demand_heatmap)
    grid_rows, grid_cols = demand_arr.shape

    # Map objective names → (function, positional args)
    objective_fns: dict[str, tuple] = {
        "load_relief":    (load_relief_score,    (demand_heatmap, existing_substations, city_bounds)),
        "loss_reduction": (loss_reduction_score, (demand_heatmap, existing_substations, city_bounds)),
        "sustainability": (sustainability_score, (city_bounds, grid_rows, grid_cols)),
        "redundancy":     (redundancy_score,     (existing_substations, city_bounds, grid_rows, grid_cols)),
    }

    if parallel:
        # Submit all objectives concurrently; gather results as they complete.
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures: dict[str, Future] = {
                name: executor.submit(_timed, fn, *args)
                for name, (fn, args) in objective_fns.items()
            }
            for name, future in futures.items():
                result, elapsed = future.result()
                individual[name] = _to_numpy(result)
                timings[name] = elapsed
        # Wall-clock total ≈ slowest parallel task + aggregation below
        _parallel_wall = max(timings[k] for k in objective_fns)
    else:
        for name, (fn, args) in objective_fns.items():
            result, elapsed = _timed(fn, *args)
            individual[name] = _to_numpy(result)
            timings[name] = elapsed
        _parallel_wall = None

    # --- Weighted aggregation ---
    t0 = time.perf_counter()
    composite = np.zeros((grid_rows, grid_cols), dtype=np.float32)
    for name, arr in individual.items():
        composite += w[name] * arr

    # --- Apply forbidden mask ---
    if forbidden_mask is not None:
        mask_np = _to_numpy(forbidden_mask).astype(np.float32)
        composite *= mask_np
    timings["aggregation"] = time.perf_counter() - t0

    if _parallel_wall is not None:
        # Report wall-clock total (objectives overlapped); not sum of all timings
        timings["total"] = _parallel_wall + timings["aggregation"]
    else:
        timings["total"] = sum(v for k, v in timings.items() if k != "total")

    return composite, individual, timings


def score_summary(
    individual: dict[str, np.ndarray],
    weights: Optional[dict[str, float]] = None,
) -> str:
    """
    Return a formatted table summarising per-objective score statistics.

    Columns: Objective | Weight | Min | p25 | Median | p75 | Max
    """
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    total_w = sum(w.values())
    w = {k: v / total_w for k, v in w.items()}

    header = (
        f"  {'Objective':<20} {'Weight':>6}  "
        f"{'Min':>6}  {'p25':>6}  {'p50':>6}  {'p75':>6}  {'Max':>6}"
    )
    lines = [header, "  " + "-" * 66]

    for name, arr in individual.items():
        p25, p50, p75 = np.percentile(arr, [25, 50, 75])
        lines.append(
            f"  {name:<20} {w.get(name, 0):.3f}   "
            f"{arr.min():.4f}   {p25:.4f}   {p50:.4f}   {p75:.4f}   {arr.max():.4f}"
        )
    return "\n".join(lines)


def active_backend() -> str:
    return backend_name()
