"""
GPU-accelerated scoring engine for the substation placement optimizer.

Scoring formula per valid cell (i, j):
    score[i,j] = forbidden_mask[i,j] * (
        w_demand    * demand_heatmap[i,j]  +
        w_isolation * isolation_score[i,j]
    )

  demand_heatmap   — from ingest; normalized [0,1]
  isolation_score  — how far cell is from nearest existing substation;
                     1 - exp(-d / ISOLATION_SCALE_M)  →  [0,1]

GPU via CuPy if available, otherwise NumPy (same code path, different xp).
"""

import argparse
import json
import os
import sys
import time

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from scipy.spatial.distance import cdist

# ── GPU / CPU backend ─────────────────────────────────────────────────────────
try:
    import cupy as cp
    xp = cp
    GPU = True
    print("[score] GPU acceleration: ENABLED (CuPy)")
except ImportError:
    xp = np
    GPU = False

# ── Scoring weights ────────────────────────────────────────────────────────────
W_DEMAND    = 0.60   # weight for demand
W_ISOLATION = 0.40   # weight for distance-from-existing-substations
ISOLATION_SCALE_M = 8_000  # metres; controls how quickly isolation decays

# Degrees → metres conversion constants
LAT_M_PER_DEG = 111_320  # metres per degree latitude (constant)


def _lon_m_per_deg(lat_mid: float) -> float:
    return LAT_M_PER_DEG * np.cos(np.radians(lat_mid))


# ─────────────────────────────────────────────────────────────────────────────
# Core computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_isolation_score(lat_grid: np.ndarray, lon_grid: np.ndarray,
                             sub_lats: np.ndarray, sub_lons: np.ndarray,
                             bounds: dict) -> np.ndarray:
    """
    For every cell compute 1 - exp(-dist_to_nearest_substation / SCALE).
    Returns float32 array of same shape as lat_grid.
    """
    grid = lat_grid.shape[0]
    lat_mid = (bounds["south"] + bounds["north"]) / 2
    lon_scale = _lon_m_per_deg(lat_mid)

    # Flatten cells → (N_cells, 2) in metres relative to SW corner
    cell_lat_m = (lat_grid.ravel() - bounds["south"]) * LAT_M_PER_DEG
    cell_lon_m = (lon_grid.ravel() - bounds["west"])  * lon_scale
    cells_m = np.column_stack([cell_lat_m, cell_lon_m])  # (N, 2)

    # Substation positions in same metre frame
    sub_lat_m = (sub_lats - bounds["south"]) * LAT_M_PER_DEG
    sub_lon_m = (sub_lons - bounds["west"])  * lon_scale
    subs_m    = np.column_stack([sub_lat_m, sub_lon_m])   # (M, 2)

    if GPU:
        # Transfer to GPU
        cells_gpu = cp.asarray(cells_m, dtype=cp.float32)
        subs_gpu  = cp.asarray(subs_m,  dtype=cp.float32)

        # Compute pairwise squared distances via broadcasting (N, M)
        diff = cells_gpu[:, None, :] - subs_gpu[None, :, :]   # (N, M, 2)
        dist2 = cp.sum(diff ** 2, axis=2)                      # (N, M)
        min_dist = cp.sqrt(cp.min(dist2, axis=1))              # (N,)
        isolation_flat = 1.0 - cp.exp(-min_dist / ISOLATION_SCALE_M)
        isolation_flat = cp.asnumpy(isolation_flat)
    else:
        # CPU via scipy cdist (highly optimised C loop)
        dist_mat  = cdist(cells_m, subs_m, metric="euclidean")  # (N, M)
        min_dist  = dist_mat.min(axis=1)                         # (N,)
        isolation_flat = 1.0 - np.exp(-min_dist / ISOLATION_SCALE_M)

    return isolation_flat.reshape(grid, grid).astype(np.float32)


def compute_scores(forbidden_mask: np.ndarray, demand_heatmap: np.ndarray,
                   isolation_score: np.ndarray) -> np.ndarray:
    """Combine all signals into a final score, zeroing forbidden cells."""
    if GPU:
        fm = cp.asarray(forbidden_mask)
        dh = cp.asarray(demand_heatmap)
        iso = cp.asarray(isolation_score)
        scores_gpu = fm * (W_DEMAND * dh + W_ISOLATION * iso)
        return cp.asnumpy(scores_gpu).astype(np.float32)
    else:
        return (forbidden_mask * (W_DEMAND * demand_heatmap +
                                   W_ISOLATION * isolation_score)).astype(np.float32)


def top_candidates(scores: np.ndarray, lats: np.ndarray, lons: np.ndarray,
                   n: int = 20) -> list[dict]:
    """Return the top-N candidate locations sorted by descending score."""
    flat_idx = np.argsort(scores.ravel())[::-1]
    results = []
    for rank, idx in enumerate(flat_idx[:n]):
        i, j = divmod(int(idx), scores.shape[1])
        results.append({
            "rank":  rank + 1,
            "score": float(scores[i, j]),
            "lat":   float(lats[i]),
            "lon":   float(lons[j]),
            "grid_i": i,
            "grid_j": j,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline step
# ─────────────────────────────────────────────────────────────────────────────

def run_scoring(data_dir: str = None, results_dir: str = None, top_n: int = 20) -> list[dict]:
    """Load data, score every cell, save scores + top candidates."""

    # ── Resolve paths ─────────────────────────────────────────────────────────
    if data_dir is None:
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    if results_dir is None:
        results_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "results"))

    print(f"\n{'='*60}")
    print(f"  Substation Placement — Scoring Engine")
    print(f"  Backend : {'GPU (CuPy)' if GPU else 'CPU (NumPy)'}")
    print(f"  Data    : {data_dir}")
    print(f"  Results : {results_dir}")
    print(f"{'='*60}\n")

    os.makedirs(results_dir, exist_ok=True)

    # ── Load inputs ───────────────────────────────────────────────────────────
    print("[1/4] Loading data...")
    forbidden_mask  = np.load(os.path.join(data_dir, "forbidden_mask.npy"))
    demand_heatmap  = np.load(os.path.join(data_dir, "demand_heatmap.npy"))
    with open(os.path.join(data_dir, "city_bounds.json")) as f:
        bounds = json.load(f)
    with open(os.path.join(data_dir, "existing_substations.geojson")) as f:
        substations = json.load(f)

    grid = forbidden_mask.shape[0]
    lats = np.linspace(bounds["south"], bounds["north"], grid)
    lons = np.linspace(bounds["west"],  bounds["east"],  grid)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    print(f"      Grid: {grid}×{grid}  valid cells: {int(forbidden_mask.sum()):,}")

    # ── Extract substation coordinates ────────────────────────────────────────
    sub_lats, sub_lons = [], []
    for feat in substations["features"]:
        try:
            lon, lat = feat["geometry"]["coordinates"][:2]
            sub_lons.append(float(lon))
            sub_lats.append(float(lat))
        except Exception:
            pass
    sub_lats = np.array(sub_lats, dtype=np.float32)
    sub_lons = np.array(sub_lons, dtype=np.float32)
    print(f"      Existing substations: {len(sub_lats)}")

    # ── Isolation score ───────────────────────────────────────────────────────
    print(f"[2/4] Computing isolation score ({grid*grid:,} cells × {len(sub_lats)} substations)...")
    t0 = time.perf_counter()

    if len(sub_lats) == 0:
        print("      WARNING: No substations found — isolation score set to 1.0 everywhere.")
        isolation_score = np.ones((grid, grid), dtype=np.float32)
    else:
        isolation_score = compute_isolation_score(lat_grid, lon_grid,
                                                   sub_lats, sub_lons, bounds)

    elapsed = time.perf_counter() - t0
    print(f"      Done in {elapsed:.2f}s  —  "
          f"isolation min={isolation_score.min():.3f} max={isolation_score.max():.3f}")

    # ── Composite score ───────────────────────────────────────────────────────
    print("[3/4] Computing composite scores...")
    scores = compute_scores(forbidden_mask, demand_heatmap, isolation_score)
    print(f"      Score range: [{scores.min():.4f}, {scores.max():.4f}]  "
          f"mean (valid cells): {scores[forbidden_mask == 1.0].mean():.4f}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("[4/4] Saving results...")
    np.save(os.path.join(results_dir, "scores.npy"),          scores)
    np.save(os.path.join(results_dir, "isolation_score.npy"), isolation_score)

    candidates = top_candidates(scores, lats, lons, n=top_n)

    # GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [c["lon"], c["lat"]]},
                "properties": {k: v for k, v in c.items() if k not in ("lat", "lon")},
            }
            for c in candidates
        ],
    }
    with open(os.path.join(results_dir, "top_candidates.geojson"), "w") as f:
        json.dump(geojson, f, indent=2)

    with open(os.path.join(results_dir, "top_candidates.json"), "w") as f:
        json.dump(candidates, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n─── Top 10 Candidate Locations ──────────────────────────────────")
    print(f"  {'Rank':>4}  {'Score':>7}  {'Lat':>10}  {'Lon':>11}")
    print(f"  {'─'*4}  {'─'*7}  {'─'*10}  {'─'*11}")
    for c in candidates[:10]:
        print(f"  {c['rank']:>4}  {c['score']:>7.4f}  {c['lat']:>10.5f}  {c['lon']:>11.5f}")
    print("─────────────────────────────────────────────────────────────────")
    best = candidates[0]
    print(f"\n  ★ BEST LOCATION: ({best['lat']:.5f}, {best['lon']:.5f})  score={best['score']:.4f}")
    print(f"    Grid index: ({best['grid_i']}, {best['grid_j']})\n")

    return candidates


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score substation placement candidates.")
    parser.add_argument("--data",    default=None, help="Data directory")
    parser.add_argument("--results", default=None, help="Results output directory")
    parser.add_argument("--top",     type=int, default=20, help="Number of top candidates")
    args = parser.parse_args()
    run_scoring(data_dir=args.data, results_dir=args.results, top_n=args.top)
