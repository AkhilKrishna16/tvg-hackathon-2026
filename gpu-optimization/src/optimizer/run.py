"""
run.py — CLI entrypoint for the GPU Substation Placement Optimizer.

Loads all input data, runs the composite scoring engine, selects the top N
placement candidates (with anti-clustering suppression), enriches results with
coverage and nearest-substation metrics, and writes to results/top_candidates.json.

Usage
-----
    # Default run (parallel, top-10, default weights):
    python -m src.optimizer.run

    # Custom weights:
    python -m src.optimizer.run --weights '{"load_relief": 0.4, "redundancy": 0.2}'

    # Sensitivity analysis across 8 weight sets:
    python -m src.optimizer.run --sensitivity

    # All options:
    python -m src.optimizer.run \\
        --mask        data/forbidden_mask.npy \\
        --heatmap     data/demand_heatmap.npy \\
        --substations data/existing_substations.geojson \\
        --bounds      data/city_bounds.json \\
        --output      results/top_candidates.json \\
        --weights     '{"load_relief": 0.4}' \\
        --top-n       10 \\
        --min-spacing 10 \\
        --radii       3,5,10 \\
        --sensitivity \\
        --no-parallel
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# Allow running as a script or as a module
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent

if __name__ == "__main__":
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.optimizer.score import composite_score, score_summary, active_backend  # noqa: E402
from src.optimizer.analysis import (  # noqa: E402
    coverage_analysis,
    grid_to_latlon,
    nearest_substation_km,
    select_top_candidates,
    sensitivity_analysis,
)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
_DATA_DIR    = _PROJECT_ROOT / "data"
_RESULTS_DIR = _PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_inputs(forbidden_mask: np.ndarray, demand_heatmap: np.ndarray) -> None:
    """Validate input arrays and raise ValueError with informative messages."""
    for name, arr in [("forbidden_mask", forbidden_mask), ("demand_heatmap", demand_heatmap)]:
        if arr.ndim != 2:
            raise ValueError(f"{name}: expected 2-D array, got shape {arr.shape}")
        if arr.shape[0] < 10 or arr.shape[1] < 10:
            raise ValueError(f"{name}: grid too small {arr.shape}; minimum 10×10")
        if not np.isfinite(arr).all():
            nan_count = int(~np.isfinite(arr).sum())
            raise ValueError(f"{name}: contains {nan_count} NaN/Inf value(s)")

    if forbidden_mask.min() < -1e-6 or forbidden_mask.max() > 1.0 + 1e-6:
        raise ValueError(
            f"forbidden_mask: expected values in [0, 1], "
            f"got range [{forbidden_mask.min():.4f}, {forbidden_mask.max():.4f}]"
        )
    if demand_heatmap.min() < -1e-6:
        raise ValueError(
            f"demand_heatmap: negative values not supported (min={demand_heatmap.min():.4f})"
        )

    placeable = int((forbidden_mask > 0.5).sum())
    if placeable == 0:
        raise ValueError(
            "forbidden_mask: no placeable cells found — all cells are forbidden"
        )


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
    print("║      GPU Substation Placement Optimizer  v2.0            ║")
    print("║      Hackathon 2026 · Austin, TX                         ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()


def _print_section(title: str) -> None:
    pad = max(0, 54 - len(title))
    print(f"\n  ── {title} {'─' * pad}")


def _timing_bar(name: str, elapsed: float, reference: float) -> str:
    pct = elapsed / reference if reference > 0 else 0.0
    filled = int(pct * 30)
    bar = "█" * filled + "░" * (30 - filled)
    return f"    {name:<22}  {elapsed:6.3f}s  [{bar}]  {pct*100:4.1f}%"


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
        help="Weight overrides as JSON, e.g. '{\"load_relief\": 0.4}'",
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
        help="Anti-clustering radius in grid cells",
    )
    parser.add_argument(
        "--radii",
        default="3,5,10",
        metavar="KM",
        help="Comma-separated coverage radii in km (e.g. '3,5,10')",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run sensitivity analysis across 8 weight-set combinations",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel objective computation (sequential mode)",
    )

    ns = parser.parse_args(argv)
    weights  = json.loads(ns.weights) if ns.weights else None
    radii_km = [float(r.strip()) for r in ns.radii.split(",")]
    parallel = not ns.no_parallel

    _print_banner()
    t_wall_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Load & validate inputs
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

    try:
        _validate_inputs(forbidden_mask, demand_heatmap)
    except ValueError as exc:
        print(f"\n  ERROR (invalid input): {exc}\n")
        sys.exit(1)

    n_subs = len(existing_substations.get("features", []))
    placeable = int((forbidden_mask > 0.5).sum())
    pct_placeable = 100.0 * placeable / forbidden_mask.size

    print(f"    forbidden_mask   : {forbidden_mask.shape}  "
          f"placeable={placeable:,} ({pct_placeable:.1f}%)")
    print(f"    demand_heatmap   : {demand_heatmap.shape}  "
          f"range=[{demand_heatmap.min():.3f}, {demand_heatmap.max():.3f}]  "
          f"mean={demand_heatmap.mean():.4f}")
    print(f"    substations      : {n_subs} existing")
    print(f"    city_bounds      : lat [{city_bounds['south']:.3f}, {city_bounds['north']:.3f}]  "
          f"lon [{city_bounds['west']:.3f}, {city_bounds['east']:.3f}]")
    print(f"    backend          : {active_backend()}")
    print(f"    parallel mode    : {'enabled' if parallel else 'disabled'}")
    print(f"    elapsed          : {time.perf_counter() - t0:.3f}s")

    # ------------------------------------------------------------------
    # Step 2 — Compute objective scores
    # ------------------------------------------------------------------
    _print_section("2 / 4  Computing objective scores")

    composite, individual, timings = composite_score(
        demand_heatmap,
        existing_substations,
        city_bounds,
        forbidden_mask=forbidden_mask,
        weights=weights,
        parallel=parallel,
    )

    print()
    for obj in ("load_relief", "loss_reduction", "sustainability", "redundancy", "aggregation"):
        t = timings.get(obj, 0.0)
        print(_timing_bar(obj, t, timings["total"]))

    wall_note = ""
    if parallel:
        wall_note = (
            f"  (tasks overlapped: wall ≈ "
            f"{max(timings[k] for k in ('load_relief','loss_reduction','sustainability','redundancy')):.3f}s)"
        )
    print(f"    {'TOTAL':<22}  {timings['total']:6.3f}s{wall_note}")

    _print_section("Score statistics")
    print(score_summary(individual, weights))

    # ------------------------------------------------------------------
    # Step 3 — Select top candidates + enrich with coverage metrics
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
    print(f"    Found {len(candidates)} candidates  ({time.perf_counter() - t0:.4f}s)")
    print(f"    Anti-clustering radius: {ns.min_spacing} grid cells")

    # Enrich: coverage at multiple radii
    print(f"    Computing demand coverage at radii {radii_km} km …")
    candidates = coverage_analysis(candidates, demand_heatmap, city_bounds, radii_km)

    # Enrich: nearest existing substation
    candidates = nearest_substation_km(candidates, existing_substations)

    # Optional: sensitivity analysis (uses pre-computed individual arrays → fast)
    sensitivity_results: dict | None = None
    if ns.sensitivity:
        _print_section("3b/ 4  Sensitivity analysis")
        print(f"    Running {8} weight-set combinations …", end="  ", flush=True)
        t_sens = time.perf_counter()
        sensitivity_results = sensitivity_analysis(
            individual,
            forbidden_mask,
            city_bounds,
            n=ns.top_n,
            min_spacing_cells=ns.min_spacing,
        )
        print(f"done  ({time.perf_counter() - t_sens:.3f}s)")

        # Build stability lookup: (grid_row, grid_col) → count
        stable_lookup: dict[tuple[int, int], int] = {
            (r, c): cnt
            for r, c, cnt in sensitivity_results["stable_cells"]
        }

        # Annotate primary candidates with their stability
        for cand in candidates:
            key = (cand["grid_row"], cand["grid_col"])
            cnt = stable_lookup.get(key, 0)
            cand["stability_count"] = cnt
            cand["stability_pct"] = round(
                100.0 * cnt / sensitivity_results["n_weight_sets"], 1
            )

        # Print stability summary table
        n_sets = sensitivity_results["n_weight_sets"]
        print(f"\n    Stable candidate cells (appear in ≥ 1 of {n_sets} weight sets):")
        print(f"    {'Location':<24} {'Count':>6}  Stability bar")
        print(f"    {'─' * 55}")
        for r_idx, c_idx, cnt in sensitivity_results["stable_cells"][:15]:
            lat, lon = grid_to_latlon(r_idx, c_idx, city_bounds)
            filled = "█" * cnt + "░" * (n_sets - cnt)
            pct = 100.0 * cnt / n_sets
            print(f"    ({lat:.3f}, {lon:.3f})      {cnt:>3}/{n_sets}  [{filled}]  {pct:.0f}%")

    # ------------------------------------------------------------------
    # Step 4 — Save results
    # ------------------------------------------------------------------
    _print_section("4 / 4  Saving results")
    out_path = Path(ns.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip internal grid indices from public output
    _internal_keys = {"grid_row", "grid_col"}
    public_candidates = [
        {k: v for k, v in c.items() if k not in _internal_keys}
        for c in candidates
    ]

    if sensitivity_results is not None:
        output_payload: dict | list = {
            "candidates": public_candidates,
            "sensitivity": {
                "n_weight_sets": sensitivity_results["n_weight_sets"],
                "weight_sets": sensitivity_results["weight_sets"],
                "results_by_set": [
                    {
                        "name": r["name"],
                        "weights": r["weights"],
                        "top_candidates": [
                            {k: v for k, v in c.items() if k not in _internal_keys}
                            for c in r["candidates"]
                        ],
                    }
                    for r in sensitivity_results["results_by_set"]
                ],
            },
        }
    else:
        output_payload = public_candidates

    with out_path.open("w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"    Saved → {out_path}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print()

    # Dynamic header based on which coverage columns exist
    cov_cols   = [f"{r:.0f}km%" for r in radii_km]
    cov_header = " │ ".join(f"{h:>7}" for h in cov_cols)
    cov_top    = "─┬─".join("─" * 7 for _ in cov_cols)
    cov_mid    = "─┼─".join("─" * 7 for _ in cov_cols)
    cov_bot    = "─┴─".join("─" * 7 for _ in cov_cols)

    print(f"  ┌──────┬────────────┬─────────────┬──────────┬─────────┬─{cov_top}─┐")
    print(f"  │ Rank │    Lat     │     Lon     │  Score   │ Dist km │ {cov_header} │")
    print(f"  ├──────┼────────────┼─────────────┼──────────┼─────────┼─{cov_mid}─┤")

    for c in candidates:
        near = c.get("nearest_existing_km")
        near_str = f"{near:>7.2f}" if near is not None else "    N/A"
        cov_vals = " │ ".join(
            f"{c.get(f'coverage_{r:.0f}km_pct', 0.0):>7.2f}"
            for r in radii_km
        )
        stability_tag = ""
        if "stability_pct" in c:
            stability_tag = f"  ★{c['stability_pct']:.0f}%"

        print(
            f"  │ {c['rank']:>4} │ {c['lat']:>10.5f} │ {c['lon']:>11.5f} │ "
            f"{c['composite_score']:>8.4f} │ {near_str} │ {cov_vals} │{stability_tag}"
        )

    print(f"  └──────┴────────────┴─────────────┴──────────┴─────────┴─{cov_bot}─┘")
    if ns.sensitivity:
        print("  ★ = stability across weight sets (100% = top pick in ALL sets)")

    total_elapsed = time.perf_counter() - t_wall_start
    print(f"\n  Total wall time: {total_elapsed:.2f}s\n")


if __name__ == "__main__":
    main()
