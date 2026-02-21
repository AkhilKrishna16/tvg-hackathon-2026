"""
objectives.py — Individual objective functions for substation placement scoring.

Each function accepts the full 500×500 grid and returns a 500×500 float32 array
normalized to [0, 1], where higher values indicate better placement candidates.

GPU acceleration: CuPy is used where available and falls back to NumPy silently.
FFT convolution is used for the loss_reduction objective (O(N² log N) vs O(N⁴) naive).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Union

import numpy as np

# ---------------------------------------------------------------------------
# Backend selection: CuPy → NumPy
# ---------------------------------------------------------------------------
try:
    import cupy as cp  # type: ignore
    from cupyx.scipy.signal import fftconvolve as _gpu_fftconvolve  # type: ignore

    _BACKEND = "cupy"
except Exception:
    cp = None  # type: ignore
    _BACKEND = "numpy"

try:
    from scipy.signal import fftconvolve as _cpu_fftconvolve
except ImportError:
    # Pure-NumPy fallback for fftconvolve
    def _cpu_fftconvolve(a: np.ndarray, b: np.ndarray, mode: str = "full") -> np.ndarray:  # type: ignore
        fa = np.fft.rfft2(a, s=(a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1))
        fb = np.fft.rfft2(b, s=(a.shape[0] + b.shape[0] - 1, a.shape[1] + b.shape[1] - 1))
        full = np.fft.irfft2(fa * fb)
        if mode == "same":
            start_r = (b.shape[0] - 1) // 2
            start_c = (b.shape[1] - 1) // 2
            return full[start_r : start_r + a.shape[0], start_c : start_c + a.shape[1]]
        return full


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EARTH_R_M: float = 6_371_000.0   # Earth radius, metres
_EPS: float = 1e-6                 # Small value to prevent division by zero
_GRID_SIZE: int = 500

NREL_URL_TEMPLATE = (
    "https://developer.nrel.gov/api/solar/solar_resource/v1.json"
    "?lat={lat:.3f}&lon={lon:.3f}&api_key=DEMO_KEY"
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

ArrayLike = Union[np.ndarray, "cp.ndarray"]  # type: ignore


def _to_numpy(arr: ArrayLike) -> np.ndarray:
    """Convert CuPy or NumPy array to NumPy ndarray."""
    if cp is not None and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr, dtype=np.float32)


def _to_backend(arr: np.ndarray) -> ArrayLike:
    """Move NumPy array to GPU if CuPy is available, otherwise keep on CPU."""
    if _BACKEND == "cupy":
        return cp.asarray(arr)
    return arr


def _normalize(arr: np.ndarray) -> np.ndarray:
    """
    Normalize array values to [0, 1].
    Returns a uniform 0.5 array if all values are identical (avoids NaN).
    """
    a_min, a_max = float(arr.min()), float(arr.max())
    if a_max - a_min < _EPS:
        return np.full_like(arr, 0.5, dtype=np.float32)
    return ((arr - a_min) / (a_max - a_min)).astype(np.float32)


def _make_lat_lon_grids(
    city_bounds: dict, rows: int = _GRID_SIZE, cols: int = _GRID_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (lat_grid, lon_grid) meshgrids of shape (rows, cols).

    Convention:
        row 0  → northernmost latitude
        row N-1 → southernmost latitude
        col 0  → westernmost longitude
        col N-1 → easternmost longitude
    """
    north, south = city_bounds["north"], city_bounds["south"]
    west, east = city_bounds["west"], city_bounds["east"]
    lats = np.linspace(north, south, rows, dtype=np.float64)
    lons = np.linspace(west, east, cols, dtype=np.float64)
    return np.meshgrid(lats, lons, indexing="ij")  # both (rows, cols)


def _haversine_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized Haversine distance in metres. All inputs in decimal degrees."""
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2.0 * _EARTH_R_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _extract_substations(existing_substations: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (sub_lats, sub_lons) from a GeoJSON FeatureCollection.
    Returns empty arrays if the collection has no features.
    """
    features = existing_substations.get("features", [])
    if not features:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    coords = np.array(
        [f["geometry"]["coordinates"] for f in features], dtype=np.float64
    )  # (N, 2) → [lon, lat]
    return coords[:, 1], coords[:, 0]  # sub_lats, sub_lons


def _min_dist_to_substations_m(
    lat_grid: np.ndarray,
    lon_grid: np.ndarray,
    sub_lats: np.ndarray,
    sub_lons: np.ndarray,
    fallback_m: float = 1e9,
) -> np.ndarray:
    """
    For every grid cell, compute the Haversine distance in metres to the nearest
    existing substation. Returns a (rows, cols) float32 array.

    If there are no substations, returns a uniform `fallback_m` array.
    """
    rows, cols = lat_grid.shape
    if len(sub_lats) == 0:
        return np.full((rows, cols), fallback_m, dtype=np.float32)

    # Broadcast over substations: shape (N_subs, rows, cols)
    dist = _haversine_m(
        lat_grid[np.newaxis, :, :],            # (1, rows, cols)
        lon_grid[np.newaxis, :, :],            # (1, rows, cols)
        sub_lats[:, np.newaxis, np.newaxis],   # (N, 1, 1)
        sub_lons[:, np.newaxis, np.newaxis],   # (N, 1, 1)
    )
    return np.min(dist, axis=0).astype(np.float32)  # (rows, cols)


# ---------------------------------------------------------------------------
# Objective 1 — Load Relief
# ---------------------------------------------------------------------------

def load_relief_score(
    demand_heatmap: ArrayLike,
    existing_substations: dict,
    city_bounds: dict,
) -> ArrayLike:
    """
    Rewards candidate cells with high demand that lie far from existing substations.

        score[cell] = demand[cell] / (dist_to_nearest_substation_m + ε)

    Rationale: a new substation relieves the most load when demand is high and the
    nearest existing station is already stretched across a large service area.

    Returns a (500, 500) array normalized to [0, 1], on the active backend.
    """
    demand_np = _to_numpy(demand_heatmap)
    lat_grid, lon_grid = _make_lat_lon_grids(city_bounds)
    sub_lats, sub_lons = _extract_substations(existing_substations)

    if len(sub_lats) == 0:
        # No existing substations → uniform relief score weighted by demand
        score = _normalize(demand_np)
        return _to_backend(score)

    min_dist_m = _min_dist_to_substations_m(lat_grid, lon_grid, sub_lats, sub_lons)
    score = demand_np * (1.0 / (min_dist_m + _EPS))
    return _to_backend(_normalize(score))


# ---------------------------------------------------------------------------
# Objective 2 — Loss Reduction
# ---------------------------------------------------------------------------

def loss_reduction_score(
    demand_heatmap: ArrayLike,
    existing_substations: dict,
    city_bounds: dict,
) -> ArrayLike:
    """
    Rewards placing a new substation near high-demand clusters to minimise I²R losses.

        score[candidate] = Σ_{cells} demand[cell] / (dist(cell, candidate)² + ε)

    This is a 2-D convolution with kernel K[Δi, Δj] = 1 / (Δi² + Δj² + ε).
    Computed via FFT for O(N² log N) efficiency instead of O(N⁴) brute force.

    Returns a (500, 500) array normalized to [0, 1], on the active backend.
    """
    demand_np = _to_numpy(demand_heatmap)
    rows, cols = demand_np.shape

    # Build inverse-square-distance kernel spanning the full grid extent.
    # K is symmetric so cross-correlation ≡ convolution.
    half_r, half_c = rows - 1, cols - 1
    di = np.arange(-half_r, half_r + 1, dtype=np.float32)
    dj = np.arange(-half_c, half_c + 1, dtype=np.float32)
    DI, DJ = np.meshgrid(di, dj, indexing="ij")
    K = (1.0 / (DI ** 2 + DJ ** 2 + 1.0)).astype(np.float32)  # ε=1 grid-cell²

    if _BACKEND == "cupy":
        score_gpu = _gpu_fftconvolve(
            cp.asarray(demand_np), cp.asarray(K), mode="same"
        )
        score_np = cp.asnumpy(score_gpu.real.astype(cp.float32))
    else:
        score_np = _cpu_fftconvolve(demand_np, K, mode="same").real.astype(np.float32)

    score_np = np.maximum(score_np, 0.0)
    return _to_backend(_normalize(score_np))


# ---------------------------------------------------------------------------
# Objective 3 — Sustainability
# ---------------------------------------------------------------------------

def sustainability_score(city_bounds: dict) -> ArrayLike:
    """
    Proxy for renewable integration potential, anchored to NREL solar irradiance data.

    Steps:
      1. Query the NREL Solar Resource API for Austin's annual average GHI.
      2. Build a linear spatial gradient: southern and western cells score higher
         (solar panels perform better with more southward orientation; western
         areas of Austin have marginally higher irradiance).
      3. Scale gradient by the fetched GHI so areas with higher irradiance
         potential receive a higher absolute score.

    Falls back to Austin's historical average GHI (~5.5 kWh/m²/day) if the
    network request fails.

    Returns a (500, 500) array normalized to [0, 1], on the active backend.
    """
    north, south = city_bounds["north"], city_bounds["south"]
    west, east = city_bounds["west"], city_bounds["east"]
    center_lat = (north + south) / 2.0
    center_lon = (west + east) / 2.0
    rows = cols = _GRID_SIZE

    # --- Fetch GHI from NREL ---
    ghi: float = 5.5  # Austin default kWh/m²/day
    try:
        url = NREL_URL_TEMPLATE.format(lat=center_lat, lon=center_lon)
        req = urllib.request.Request(url, headers={"User-Agent": "gpu-optimizer/1.0"})
        with urllib.request.urlopen(req, timeout=6) as resp:
            payload = json.loads(resp.read().decode())
        avg_ghi = payload.get("outputs", {}).get("avg_ghi", {}).get("annual")
        if avg_ghi is not None:
            ghi = float(avg_ghi)
    except Exception:
        pass  # Network unavailable or quota exceeded — use default

    # --- Spatial gradient ---
    # Rows: row 0 = north (lower solar), row N-1 = south (higher solar) → [0.65 … 1.0]
    lat_grad = np.linspace(0.65, 1.0, rows, dtype=np.float32)
    # Cols: col 0 = west (higher solar), col N-1 = east (lower solar) → [1.0 … 0.65]
    lon_grad = np.linspace(1.0, 0.65, cols, dtype=np.float32)

    lat_2d = lat_grad[:, np.newaxis]  # (rows, 1)
    lon_2d = lon_grad[np.newaxis, :]  # (1, cols)
    gradient = ((lat_2d + lon_2d) / 2.0).astype(np.float32)  # (rows, cols)

    # Scale by GHI factor: CONUS range ≈ 3.5–6.5 kWh/m²/day
    ghi_factor = float(np.clip((ghi - 3.5) / 3.0, 0.0, 1.0))
    score = gradient * (0.6 + 0.4 * ghi_factor)

    return _to_backend(_normalize(score))


# ---------------------------------------------------------------------------
# Objective 4 — Redundancy / Optimal Spacing
# ---------------------------------------------------------------------------

def redundancy_score(
    existing_substations: dict,
    city_bounds: dict,
) -> ArrayLike:
    """
    Rewards locations that are optimally spaced from existing substations.

    Too close → clustering, wasteful redundancy.
    Too far   → isolation, long transmission runs.
    Sweet spot → 3–5 km from nearest existing substation.

    Gaussian bell curve centred at 4 km, σ = 2 km:

        score[cell] = exp( -((dist_m − 4000)²) / (2 × 2000²) )

    Returns a (500, 500) array normalized to [0, 1], on the active backend.
    """
    lat_grid, lon_grid = _make_lat_lon_grids(city_bounds)
    sub_lats, sub_lons = _extract_substations(existing_substations)

    if len(sub_lats) == 0:
        # No existing substations → uniform score (any spacing is valid)
        uniform = np.full((_GRID_SIZE, _GRID_SIZE), 0.5, dtype=np.float32)
        return _to_backend(uniform)

    min_dist_m = _min_dist_to_substations_m(lat_grid, lon_grid, sub_lats, sub_lons)

    mu_m: float = 4_000.0   # Peak at 4 km
    sigma_m: float = 2_000.0
    score = np.exp(
        -((min_dist_m - mu_m) ** 2) / (2.0 * sigma_m ** 2)
    ).astype(np.float32)

    return _to_backend(_normalize(score))


# ---------------------------------------------------------------------------
# Utility: report active backend
# ---------------------------------------------------------------------------

def backend_name() -> str:
    """Return a human-readable string describing the active compute backend."""
    if _BACKEND == "cupy":
        device = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(device.id)
        gpu_name = props["name"].decode()
        return f"CuPy {cp.__version__} → {gpu_name}"
    return f"NumPy {np.__version__} (CPU)"
