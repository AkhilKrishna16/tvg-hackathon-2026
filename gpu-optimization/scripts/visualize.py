"""
visualize.py — Generate a publication-quality dashboard of the optimizer results.

Produces:
    results/optimizer_dashboard.png  — 2×3 panel figure (individual objectives + composite)
    results/composite_map.png        — large standalone composite score map with
                                       5 km service-area circles around top candidates

Improvements over v1:
  • _try_get_composite is called once and results passed to all consumers —
    no duplicate recomputation.
  • make_composite_map is factored so main() reuses it instead of duplicating
    rendering logic.
  • _coverage_circles: draws semi-transparent 5 km radius circles around each
    candidate to visualise the service area each placement would cover.
  • Score-distribution histograms added as small insets in each objective panel.

Run from the project root after running the optimizer:
    python scripts/visualize.py

Requirements: matplotlib, numpy (scipy optional)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR    = _ROOT / "data"
_RESULTS_DIR = _ROOT / "results"

# ---------------------------------------------------------------------------
# Custom colormaps
# ---------------------------------------------------------------------------
_SCORE_CMAP = LinearSegmentedColormap.from_list(
    "score_cmap",
    ["#0d0221", "#0a0f5e", "#1a6b9a", "#00c4a0", "#f7e200", "#e8441a"],
    N=512,
)
_DEMAND_CMAP = LinearSegmentedColormap.from_list(
    "demand_cmap",
    ["#0d0221", "#170f4f", "#2a1a7a", "#5a3fa0", "#b06fca", "#ffcaf5"],
    N=512,
)

# Service-area circle radius drawn on composite map
_COVERAGE_RADIUS_KM = 5.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_all():
    mask    = np.load(_DATA_DIR / "forbidden_mask.npy")
    heatmap = np.load(_DATA_DIR / "demand_heatmap.npy")
    with (_DATA_DIR / "existing_substations.geojson").open() as f:
        substations = json.load(f)
    with (_DATA_DIR / "city_bounds.json").open() as f:
        bounds = json.load(f)

    candidates_path = _RESULTS_DIR / "top_candidates.json"
    candidates = []
    if candidates_path.exists():
        with candidates_path.open() as f:
            raw = json.load(f)
        # Handle both plain list and {"candidates": [...]} formats
        candidates = raw if isinstance(raw, list) else raw.get("candidates", [])

    return mask, heatmap, substations, bounds, candidates


def _try_get_composite(mask, heatmap, substations, bounds):
    """
    Attempt to compute the composite score using the optimizer engine.
    Returns (composite, individual) tuple or (None, {}) on failure.
    Called once per visualization run; results are passed around.
    """
    try:
        import sys
        sys.path.insert(0, str(_ROOT))
        from src.optimizer.score import composite_score

        composite, individual, _ = composite_score(
            heatmap, substations, bounds, forbidden_mask=mask
        )
        return composite, individual
    except Exception as exc:
        print(f"  Warning: could not compute scores for visualization: {exc}")
        return None, {}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _latlon_extent(bounds):
    """Return (left, right, bottom, top) for imshow extent."""
    return [bounds["west"], bounds["east"], bounds["south"], bounds["north"]]


def _sub_coords(substations):
    features = substations.get("features", [])
    if not features:
        return np.array([]), np.array([])
    coords = np.array([f["geometry"]["coordinates"] for f in features])
    return coords[:, 0], coords[:, 1]  # lons, lats


def _apply_forbidden_overlay(ax, mask, extent, alpha=0.35):
    """Draw a semi-transparent dark overlay over forbidden (blocked) cells."""
    forbidden = np.ma.masked_where(mask > 0.5, np.ones_like(mask))
    ax.imshow(
        forbidden,
        extent=extent,
        origin="upper",
        cmap="Greys",
        vmin=0, vmax=1,
        alpha=alpha,
        interpolation="nearest",
        zorder=2,
    )


def _annotate_substations(ax, substations, marker="^", color="#00ffff", size=60, zorder=5):
    lons, lats = _sub_coords(substations)
    if len(lons) > 0:
        ax.scatter(
            lons, lats,
            marker=marker, s=size, c=color,
            edgecolors="white", linewidths=0.6,
            zorder=zorder, label="Existing substation",
        )


def _annotate_candidates(ax, candidates, zorder=6):
    if not candidates:
        return
    cmap = plt.cm.plasma
    n = len(candidates)
    for c in candidates:
        rank = c["rank"]
        colour = cmap(1.0 - rank / (n + 1))
        ax.scatter(
            c["lon"], c["lat"],
            marker="*", s=max(40, 180 - rank * 8),
            c=[colour], edgecolors="white", linewidths=0.5,
            zorder=zorder,
        )
        ax.annotate(
            str(rank),
            (c["lon"], c["lat"]),
            textcoords="offset points",
            xytext=(5, 4),
            fontsize=6.5,
            color="white",
            fontweight="bold",
            zorder=zorder + 1,
        )


def _coverage_circles(ax, candidates, bounds, radius_km=_COVERAGE_RADIUS_KM, zorder=4):
    """
    Draw semi-transparent service-area circles of `radius_km` around each candidate.

    Circle radii are approximated in degrees using the cosine-corrected lon scale
    so they appear physically correct on the lat/lon plot.
    """
    if not candidates:
        return

    import math
    center_lat_rad = math.radians((bounds["south"] + bounds["north"]) / 2.0)
    # Degrees of latitude per km (constant)
    dlat_deg = radius_km / 111.32
    # Degrees of longitude per km at center latitude
    dlon_deg = radius_km / (111.32 * math.cos(center_lat_rad))

    n = len(candidates)
    cmap = plt.cm.plasma

    for c in candidates:
        rank = c["rank"]
        colour = cmap(1.0 - rank / (n + 1))
        # Approximate circle as an Ellipse in lon/lat space
        ellipse = mpatches.Ellipse(
            xy=(c["lon"], c["lat"]),
            width=2 * dlon_deg,
            height=2 * dlat_deg,
            fill=True,
            facecolor=colour,
            edgecolor=colour,
            alpha=0.12,
            linewidth=0.8,
            linestyle="--",
            zorder=zorder,
        )
        ax.add_patch(ellipse)


def _add_score_histogram(ax, data, color="#00c4a0"):
    """
    Add a small score-distribution histogram as an inset in the top-right corner
    of a panel axes.
    """
    try:
        inset = ax.inset_axes([0.68, 0.72, 0.30, 0.26])
        flat = data.ravel()
        inset.hist(flat, bins=30, color=color, alpha=0.8, edgecolor="none")
        inset.set_xticks([])
        inset.set_yticks([])
        inset.set_facecolor("#0d0221")
        for spine in inset.spines.values():
            spine.set_edgecolor("#444")
        inset.set_title("dist", fontsize=5, color="#aaa", pad=1)
    except Exception:
        pass  # Non-critical; don't break main render on any error


def _panel(ax, data, title, extent, cmap, mask, substations, candidates=None,
           vmin=0, vmax=1, show_coverage=False, bounds=None):
    im = ax.imshow(
        data,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        interpolation="bilinear",
        zorder=1,
    )
    _apply_forbidden_overlay(ax, mask, extent)
    _annotate_substations(ax, substations)
    if show_coverage and candidates and bounds:
        _coverage_circles(ax, candidates, bounds)
    if candidates:
        _annotate_candidates(ax, candidates)

    _add_score_histogram(ax, data)

    ax.set_title(title, fontsize=9, color="white", pad=4)
    ax.set_xlabel("Longitude", fontsize=7, color="#aaa")
    ax.set_ylabel("Latitude",  fontsize=7, color="#aaa")
    ax.tick_params(colors="#aaa", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.ax.yaxis.set_tick_params(color="#aaa", labelcolor="#aaa")
    return im


# ---------------------------------------------------------------------------
# Dashboard (2 × 3 grid)
# ---------------------------------------------------------------------------

def make_dashboard(mask, heatmap, substations, bounds, candidates, individual=None):
    """
    Build and save the 2×3 panel dashboard.

    Parameters
    ----------
    individual : dict of score arrays (pre-computed by caller).
                 If None or empty, panels fall back to placeholder text.
    """
    individual = individual or {}
    extent = _latlon_extent(bounds)

    fig = plt.figure(figsize=(18, 11), facecolor="#0d0221")
    fig.suptitle(
        "GPU Substation Placement Optimizer — Austin, TX",
        fontsize=15, color="white", fontweight="bold", y=0.98,
    )
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    panels = [
        (gs[0, 0], heatmap,                        "Demand Heatmap",            _DEMAND_CMAP, False),
        (gs[0, 1], individual.get("load_relief"),   "Load Relief Score",         _SCORE_CMAP,  False),
        (gs[0, 2], individual.get("loss_reduction"),"Loss Reduction Score",      _SCORE_CMAP,  False),
        (gs[1, 0], individual.get("sustainability"),"Sustainability Score",      _SCORE_CMAP,  False),
        (gs[1, 1], individual.get("redundancy"),    "Redundancy Score",          _SCORE_CMAP,  False),
        (gs[1, 2], individual.get("_composite"),    "Composite Score + Top 10",  _SCORE_CMAP,  True),
    ]

    # Compute composite for the last panel if not pre-supplied
    if individual.get("_composite") is None:
        _comp, _ = _try_get_composite(mask, heatmap, substations, bounds)
        # Patch it in so the panel renderer picks it up
        if _comp is not None:
            individual = {**individual, "_composite": _comp}
            panels[-1] = (gs[1, 2], _comp, "Composite Score + Top 10", _SCORE_CMAP, True)

    for spec, data, title, cmap, show_cov in panels:
        ax = fig.add_subplot(spec)
        ax.set_facecolor("#0d0221")
        if data is not None:
            show_cands = candidates if "Composite" in title or "Demand" in title else None
            _panel(
                ax, data, title, extent, cmap, mask, substations,
                candidates=show_cands,
                show_coverage=show_cov,
                bounds=bounds,
            )
        else:
            ax.text(0.5, 0.5, "Run optimizer first", transform=ax.transAxes,
                    ha="center", va="center", color="#aaa", fontsize=9)
            ax.set_title(title, fontsize=9, color="white", pad=4)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#00ffff", edgecolor="white", label="Existing substation (▲)"),
        mpatches.Patch(facecolor="#f7e200", edgecolor="white", label="Top-10 candidate (★)"),
        mpatches.Patch(facecolor="#00c4a0", edgecolor="white", alpha=0.4, label=f"{_COVERAGE_RADIUS_KM:.0f} km service area"),
        mpatches.Patch(facecolor="#444",    edgecolor="white", label="Forbidden area"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        fontsize=8,
        framealpha=0.2,
        facecolor="#111",
        edgecolor="#444",
        labelcolor="white",
        bbox_to_anchor=(0.5, 0.01),
    )

    out = _RESULTS_DIR / "optimizer_dashboard.png"
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Dashboard saved → {out}")


# ---------------------------------------------------------------------------
# Standalone composite map
# ---------------------------------------------------------------------------

def make_composite_map(mask, heatmap, substations, bounds, candidates, composite):
    """
    Render a large standalone composite score map with:
      • 5 km service-area circles around top candidates
      • Candidate ranking annotations
      • Sidebar table listing top-10 with scores
    """
    if composite is None:
        print("Composite score unavailable — skipping standalone map.")
        return

    extent = _latlon_extent(bounds)
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0d0221")
    ax.set_facecolor("#0d0221")

    im = ax.imshow(
        composite,
        extent=extent,
        origin="upper",
        cmap=_SCORE_CMAP,
        vmin=0,
        interpolation="bilinear",
        zorder=1,
    )
    _apply_forbidden_overlay(ax, mask, extent, alpha=0.4)
    _coverage_circles(ax, candidates, bounds, radius_km=_COVERAGE_RADIUS_KM)
    _annotate_substations(ax, substations, size=80)
    _annotate_candidates(ax, candidates, zorder=6)

    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Composite Score")
    cb.ax.yaxis.set_tick_params(color="white", labelcolor="white")

    ax.set_title(
        "Optimal Substation Placement — Composite Score Map",
        fontsize=13, color="white", pad=8,
    )
    ax.set_xlabel("Longitude", color="#aaa")
    ax.set_ylabel("Latitude",  color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    if candidates:
        _add_candidate_sidebar(fig, candidates)

    out = _RESULTS_DIR / "composite_map.png"
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Composite map saved → {out}")


def _add_candidate_sidebar(fig, candidates):
    """Add a small inset table listing candidates with their scores."""
    ax_table = fig.add_axes([0.75, 0.10, 0.22, 0.40], facecolor="#11112a")
    ax_table.axis("off")
    ax_table.set_title("Top Candidates", fontsize=8, color="white", pad=3)

    headers = ["#", "Lat", "Lon", "Score"]
    col_widths = [0.08, 0.32, 0.35, 0.25]
    row_height = 1.0 / (len(candidates) + 1.5)

    for ci, h in enumerate(headers):
        x = sum(col_widths[:ci]) + col_widths[ci] / 2
        ax_table.text(
            x, 1.0 - row_height * 0.6, h,
            transform=ax_table.transAxes,
            ha="center", va="center",
            fontsize=6.5, color="#00ffff", fontweight="bold",
        )

    for ri, c in enumerate(candidates):
        y = 1.0 - row_height * (ri + 1.5)
        row_color = "#1e1e3a" if ri % 2 == 0 else "#131328"
        rect = mpatches.FancyBboxPatch(
            (0, y), 1.0, row_height,
            boxstyle="square,pad=0",
            transform=ax_table.transAxes,
            facecolor=row_color, edgecolor="none", zorder=0,
        )
        ax_table.add_patch(rect)
        row_vals = [
            str(c["rank"]),
            f"{c['lat']:.3f}",
            f"{c['lon']:.3f}",
            f"{c['composite_score']:.3f}",
        ]
        for ci2, val in enumerate(row_vals):
            x = sum(col_widths[:ci2]) + col_widths[ci2] / 2
            ax_table.text(
                x, y + row_height / 2, val,
                transform=ax_table.transAxes,
                ha="center", va="center",
                fontsize=6, color="white",
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Loading data …")
    mask, heatmap, substations, bounds, candidates = _load_all()

    # Compute scores once; share across both outputs
    print("Computing scores …")
    composite, individual = _try_get_composite(mask, heatmap, substations, bounds)

    if composite is None:
        individual = {
            k: np.zeros((500, 500), dtype=np.float32)
            for k in ("load_relief", "loss_reduction", "sustainability", "redundancy")
        }

    print("Rendering dashboard …")
    make_dashboard(mask, heatmap, substations, bounds, candidates, individual)

    print("Rendering composite map …")
    make_composite_map(mask, heatmap, substations, bounds, candidates, composite)

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
