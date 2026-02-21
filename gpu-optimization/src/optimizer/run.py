"""
run.py — CLI entrypoint for the GPU Substation Placement Optimizer.

Loads all input data, runs the composite scoring engine, selects the top 10
placement candidates (with anti-clustering suppression), and writes results to
results/top_candidates.json.

Usage
-----
    # From the project root:
    python -m src.optimizer.run

    # With custom paths / weight overrides:
    python -m src.optimizer.run \\
        --mask      data/forbidden_mask.npy \\
        --heatmap   data/demand_heatmap.npy \\
        --substations data/existing_substations.geojson \\
        --bounds    data/city_bounds.json \\
        --output    results/top_candidates.json \\
        --weights   '{"load_relief": 0.4, "redundancy": 0.2}'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Allow running as a script or as a module
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent

if __name__ == "__main__":
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.optimizer.score import composite_score, score_summary, active_backend  # noqa: E402

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_DATA_DIR = _PROJECT_ROOT / "data"
_RESULTS_DIR = _PROJECT_ROOT / "results"

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
# Top-N selection with anti-clustering
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

    After each selection the scoring map is zeroed within a radius of
    ``min_spacing_cells`` grid cells to prevent spatial clustering.

    Parameters
    ----------
    composite : np.ndarray (500, 500)
        Composite score map; forbidden cells must already be zeroed.
    individual : dict[str, np.ndarray]
        Per-objective score arrays.
    city_bounds : dict
        Bounding box for coordinate conversion.
    n : int
        Number of candidates to return.
    min_spacing_cells : int
        Minimum grid-cell distance between any two selected candidates.

    Returns
    -------
    list of dicts, sorted by descending composite score.
    """
    score_map = composite.copy()
    rows, cols = score_map.shape

    # Pre-compute row/col index grids once for suppression radii
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

        # Suppress all cells within `min_spacing_cells` of the selected cell
        dist = np.sqrt(
            (ROW_GRID - best_i) ** 2 + (COL_GRID - best_j) ** 2
        )
        score_map[dist <= min_spacing_cells] = 0.0

    return candidates


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_inputs(
    mask_path: Path,
    heatmap_path: Path,
    substations_path: Path,
    bounds_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    forbidden_mask   = np.load(mask_path).astype(np.float32)
    demand_heatmap   = np.load(heatmap_path).astype(np.float32)
    with substations_path.open() as f:
        existing_substations = json.load(f)
    with bounds_path.open() as f:
        city_bounds = json.load(f)
    return forbidden_mask, demand_heatmap, existing_substations, city_bounds


def _print_banner() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      GPU Substation Placement Optimizer  v1.0            ║")
    print("║      Hackathon 2026 · Austin, TX                         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def _print_section(title: str) -> None:
    print(f"\n  ── {title} {'─' * max(0, 54 - len(title))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated substation placement optimizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mask",
        default=str(_DATA_DIR / "forbidden_mask.npy"),
        help="Path to forbidden_mask.npy",
    )
    parser.add_argument(
        "--heatmap",
        default=str(_DATA_DIR / "demand_heatmap.npy"),
        help="Path to demand_heatmap.npy",
    )
    parser.add_argument(
        "--substations",
        default=str(_DATA_DIR / "existing_substations.geojson"),
        help="Path to existing_substations.geojson",
    )
    parser.add_argument(
        "--bounds",
        default=str(_DATA_DIR / "city_bounds.json"),
        help="Path to city_bounds.json",
    )
    parser.add_argument(
        "--output",
        default=str(_RESULTS_DIR / "top_candidates.json"),
        help="Output JSON path",
    )
    parser.add_argument(
        "--weights",
        default=None,
        metavar="JSON",
        help='Weight overrides, e.g. \'{"load_relief": 0.4}\'',
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top candidates to return",
    )
    parser.add_argument(
        "--min-spacing",
        type=int,
        default=10,
        help="Minimum grid-cell spacing between candidates (anti-clustering)",
    )

    ns = parser.parse_args(argv)
    weights = json.loads(ns.weights) if ns.weights else None

    _print_banner()
    t_wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Load inputs
    # ------------------------------------------------------------------
    _print_section("1 / 4  Loading inputs")
    t0 = time.perf_counter()
    try:
        forbidden_mask, demand_heatmap, existing_substations, city_bounds = _load_inputs(
            Path(ns.mask),
            Path(ns.heatmap),
            Path(ns.substations),
            Path(ns.bounds),
        )
    except FileNotFoundError as exc:
        print(f"\n  ERROR: {exc}")
        print("  Run `python scripts/generate_data.py` first to create sample data.\n")
        sys.exit(1)

    n_subs = len(existing_substations.get("features", []))
    placeable = int(forbidden_mask.sum())
    pct_placeable = 100.0 * placeable / forbidden_mask.size

    print(f"    forbidden_mask   : {forbidden_mask.shape}  "
          f"placeable={placeable:,} ({pct_placeable:.1f}%)")
    print(f"    demand_heatmap   : {demand_heatmap.shape}  "
          f"range=[{demand_heatmap.min():.3f}, {demand_heatmap.max():.3f}]")
    print(f"    substations      : {n_subs} existing")
    print(f"    city_bounds      : {city_bounds}")
    print(f"    backend          : {active_backend()}")
    print(f"    elapsed          : {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Step 2 — Compute scores
    # ------------------------------------------------------------------
    _print_section("2 / 4  Computing objective scores")

    composite, individual, timings = composite_score(
        demand_heatmap,
        existing_substations,
        city_bounds,
        forbidden_mask=forbidden_mask,
        weights=weights,
    )

    # Print per-objective timing
    print()
    for obj in ("load_relief", "loss_reduction", "sustainability", "redundancy", "aggregation"):
        t = timings.get(obj, 0.0)
        bar_len = int(t / timings["total"] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {obj:<20}  {t:6.3f}s  [{bar}]")
    print(f"    {'TOTAL':<20}  {timings['total']:6.3f}s")

    # Per-objective statistics
    _print_section("Score statistics")
    print(score_summary(individual, weights))

    # ------------------------------------------------------------------
    # Step 3 — Select top candidates
    # ------------------------------------------------------------------
    _print_section("3 / 4  Selecting top candidates")
    t0 = time.perf_counter()
    candidates = select_top_candidates(
        composite,
        individual,
        city_bounds,
        n=ns.top_n,
        min_spacing_cells=ns.min_spacing,
    )
    print(f"    Found {len(candidates)} candidates in {time.perf_counter() - t0:.3f}s")
    print(f"    Anti-clustering radius: {ns.min_spacing} grid cells")

    # ------------------------------------------------------------------
    # Step 4 — Save results
    # ------------------------------------------------------------------
    _print_section("4 / 4  Saving results")
    out_path = Path(ns.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip grid_row/col from public output (keep them for debugging locally)
    public_candidates = [
        {k: v for k, v in c.items() if k not in ("grid_row", "grid_col")}
        for c in candidates
    ]
    with out_path.open("w") as f:
        json.dump(public_candidates, f, indent=2)
    print(f"    Saved → {out_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()
    print("  ┌──────┬────────────┬─────────────┬──────────┐")
    print("  │ Rank │    Lat     │     Lon     │  Score   │")
    print("  ├──────┼────────────┼─────────────┼──────────┤")
    for c in candidates:
        print(
            f"  │ {c['rank']:>4} │ {c['lat']:>10.5f} │ {c['lon']:>11.5f} │ {c['composite_score']:>8.4f} │"
        )
    print("  └──────┴────────────┴─────────────┴──────────┘")

    total_elapsed = time.perf_counter() - t_wall_start
    print(f"\n  Total wall time: {total_elapsed:.2f}s\n")


if __name__ == "__main__":
    main()
