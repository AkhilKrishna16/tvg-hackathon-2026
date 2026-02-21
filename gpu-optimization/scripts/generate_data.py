"""
generate_data.py — Generate synthetic but geographically realistic input data
for the Austin, TX substation placement optimizer.

Produces:
    data/forbidden_mask.npy   — 500×500 binary mask (1=placeable, 0=blocked)
    data/demand_heatmap.npy   — 500×500 normalized demand array

Run from the project root:
    python scripts/generate_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _ROOT / "data"

ROWS = COLS = 500
RNG = np.random.default_rng(seed=42)

# ---------------------------------------------------------------------------
# Load city bounds
# ---------------------------------------------------------------------------

def load_bounds() -> dict:
    with (_DATA_DIR / "city_bounds.json").open() as f:
        return json.load(f)


def grid_coords(bounds: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (lat_grid, lon_grid) meshgrids."""
    lats = np.linspace(bounds["north"], bounds["south"], ROWS)
    lons = np.linspace(bounds["west"],  bounds["east"],  COLS)
    return np.meshgrid(lats, lons, indexing="ij")


# ---------------------------------------------------------------------------
# Forbidden mask generation
# ---------------------------------------------------------------------------

def make_forbidden_mask(bounds: dict) -> np.ndarray:
    """
    Generate a plausible forbidden mask for Austin, TX.

    Blocked areas (value = 0) model:
      • Lady Bird Lake / Colorado River corridor (runs east-west through centre)
      • Lake Austin arm (northwest)
      • Barton Creek greenbelt (south-central)
      • Austin-Bergstrom International Airport (southeast)
      • Major highway rights-of-way (I-35, Loop 360, US-183, SH-130)
      • Random small obstacles (utility easements, parks, etc.)
    """
    lat_grid, lon_grid = grid_coords(bounds)
    mask = np.ones((ROWS, COLS), dtype=np.float32)

    # Helper: block an elliptical region
    def block_ellipse(center_lat, center_lon, r_lat_deg, r_lon_deg, value=0.0):
        dlat = (lat_grid - center_lat) / r_lat_deg
        dlon = (lon_grid - center_lon) / r_lon_deg
        mask[dlat ** 2 + dlon ** 2 <= 1.0] = value

    # Helper: block a lat-aligned corridor (highway right-of-way)
    def block_lon_corridor(lon_center, width_deg, value=0.0):
        mask[np.abs(lon_grid - lon_center) <= width_deg / 2] = value

    def block_lat_corridor(lat_center, width_deg, value=0.0):
        mask[np.abs(lat_grid - lat_center) <= width_deg / 2] = value

    def block_diagonal(lat0, lon0, lat1, lon1, width_deg=0.003, value=0.0):
        """Block a narrow strip between two (lat,lon) points."""
        dlon = lon1 - lon0
        dlat = lat1 - lat0
        length = np.sqrt(dlon ** 2 + dlat ** 2)
        # Normal vector
        nx, ny = -dlat / length, dlon / length
        # Signed perpendicular distance from the line segment
        perp = (lon_grid - lon0) * nx + (lat_grid - lat0) * ny
        # Parametric projection along the line
        proj = ((lon_grid - lon0) * dlon + (lat_grid - lat0) * dlat) / length ** 2
        in_strip = (np.abs(perp) <= width_deg / 2) & (proj >= 0) & (proj <= 1)
        mask[in_strip] = value

    # --- Lady Bird Lake / Colorado River (east-west corridor ~30.263°N) ---
    for lat_offset in np.linspace(-0.004, 0.004, 5):
        block_lat_corridor(30.263 + lat_offset, width_deg=0.003)
    # Widen at Town Lake area
    block_ellipse(30.263, -97.720, 0.008, 0.020)

    # --- Lake Austin (northwest arm) ---
    block_ellipse(30.340, -97.840, 0.012, 0.030)
    block_diagonal(30.335, -97.800, 30.348, -97.870, width_deg=0.005)

    # --- Barton Creek Greenbelt ---
    block_ellipse(30.228, -97.787, 0.012, 0.018)
    block_diagonal(30.215, -97.770, 30.240, -97.810, width_deg=0.004)

    # --- Austin-Bergstrom Airport (southeast) ---
    block_ellipse(30.197, -97.666, 0.020, 0.025)

    # --- I-35 (north-south, lon ≈ -97.720) ---
    block_lon_corridor(-97.720, width_deg=0.004)

    # --- Loop 360 / Capital of Texas Hwy (north-south, lon ≈ -97.800) ---
    block_lon_corridor(-97.800, width_deg=0.003)

    # --- US-183 (diagonal NW-SE) ---
    block_diagonal(30.435, -97.783, 30.155, -97.665, width_deg=0.003)

    # --- SH-130 Toll (eastern corridor, lon ≈ -97.630) ---
    block_lon_corridor(-97.630, width_deg=0.003)

    # --- Ben White Blvd / SH-71 (lat ≈ 30.218) ---
    block_lat_corridor(30.218, width_deg=0.002)

    # --- Random small forbidden patches (parks, easements, etc.) ---
    n_patches = 60
    patch_lats = RNG.uniform(bounds["south"] + 0.02, bounds["north"] - 0.02, n_patches)
    patch_lons = RNG.uniform(bounds["west"]  + 0.02, bounds["east"]  - 0.02, n_patches)
    patch_r    = RNG.uniform(0.002, 0.008, n_patches)
    for plat, plon, pr in zip(patch_lats, patch_lons, patch_r):
        block_ellipse(plat, plon, pr, pr * 1.3)

    # Ensure edges are also blocked (no substations on the boundary)
    mask[:5, :]  = 0
    mask[-5:, :] = 0
    mask[:, :5]  = 0
    mask[:, -5:] = 0

    return mask


# ---------------------------------------------------------------------------
# Demand heatmap generation
# ---------------------------------------------------------------------------

def make_demand_heatmap(bounds: dict, forbidden_mask: np.ndarray) -> np.ndarray:
    """
    Generate a normalized demand heatmap that reflects Austin's urban structure:
      • High demand downtown (CBD)
      • Secondary peaks at Domain (north), South Congress, Mueller, UT campus
      • Moderate demand along major arterials
      • Low demand in greenbelt / suburban fringe
      • Gaussian smooth to create realistic gradient
    """
    lat_grid, lon_grid = grid_coords(bounds)
    demand = np.zeros((ROWS, COLS), dtype=np.float64)

    def add_gaussian(center_lat, center_lon, amplitude, sigma_lat, sigma_lon):
        dlat = (lat_grid - center_lat) / sigma_lat
        dlon = (lon_grid - center_lon) / sigma_lon
        demand[:] += amplitude * np.exp(-(dlat ** 2 + dlon ** 2) / 2)

    # --- Major urban demand centres ---
    add_gaussian(30.267, -97.743, 1.00, 0.020, 0.025)   # Downtown / CBD
    add_gaussian(30.400, -97.723, 0.70, 0.018, 0.022)   # The Domain (tech hub)
    add_gaussian(30.285, -97.735, 0.55, 0.012, 0.015)   # UT Campus / West Campus
    add_gaussian(30.287, -97.696, 0.50, 0.015, 0.018)   # Mueller / Airport Blvd
    add_gaussian(30.245, -97.750, 0.45, 0.012, 0.015)   # South Congress
    add_gaussian(30.350, -97.710, 0.40, 0.014, 0.016)   # North Loop / Rundberg
    add_gaussian(30.230, -97.690, 0.35, 0.012, 0.015)   # McKinney Falls / SE Austin
    add_gaussian(30.460, -97.760, 0.55, 0.016, 0.020)   # Pflugerville border (fast growing)
    add_gaussian(30.200, -97.830, 0.30, 0.013, 0.016)   # Oak Hill / Circle C
    add_gaussian(30.310, -97.760, 0.35, 0.010, 0.012)   # Tarrytown / Westlake Hills

    # --- Arterial corridors (lower, elongated Gaussians) ---
    add_gaussian(30.330, -97.740, 0.25, 0.040, 0.008)   # Lamar Blvd (N-S)
    add_gaussian(30.267, -97.690, 0.20, 0.008, 0.035)   # E 7th / MLK (E-W)
    add_gaussian(30.370, -97.680, 0.20, 0.030, 0.008)   # E Parmer Ln

    # --- Scattered low-level residential load ---
    n_residential = 120
    res_lats = RNG.uniform(bounds["south"] + 0.04, bounds["north"] - 0.04, n_residential)
    res_lons = RNG.uniform(bounds["west"]  + 0.04, bounds["east"]  - 0.04, n_residential)
    res_amp  = RNG.uniform(0.05, 0.20, n_residential)
    res_sig  = RNG.uniform(0.005, 0.012, n_residential)
    for rlat, rlon, ramp, rsig in zip(res_lats, res_lons, res_amp, res_sig):
        add_gaussian(rlat, rlon, ramp, rsig, rsig * 1.2)

    # Smooth the raw demand with a wide Gaussian blur
    demand = gaussian_filter(demand, sigma=3.0)

    # Zero out forbidden areas (no demand from blocked cells)
    demand *= forbidden_mask.astype(np.float64)

    # Normalize to [0, 1]
    d_max = demand.max()
    if d_max > 0:
        demand /= d_max

    return demand.astype(np.float32)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    bounds = load_bounds()

    print("Generating forbidden_mask.npy …", end="  ", flush=True)
    mask = make_forbidden_mask(bounds)
    np.save(_DATA_DIR / "forbidden_mask.npy", mask)
    placeable = int(mask.sum())
    print(f"done  ({placeable:,} / {ROWS * COLS:,} cells placeable, "
          f"{100 * placeable / (ROWS * COLS):.1f}%)")

    print("Generating demand_heatmap.npy  …", end="  ", flush=True)
    heatmap = make_demand_heatmap(bounds, mask)
    np.save(_DATA_DIR / "demand_heatmap.npy", heatmap)
    print(f"done  (range=[{heatmap.min():.3f}, {heatmap.max():.3f}])")

    print("\nData written to:", _DATA_DIR)


if __name__ == "__main__":
    main()
