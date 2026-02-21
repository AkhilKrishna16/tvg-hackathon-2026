"""
visualize.py — Generate a publication-quality dashboard of the optimizer results.

Produces:
    results/optimizer_dashboard.png  — 2×3 panel figure
    results/composite_map.png        — large standalone composite score map

Run from the project root after running the optimizer:
    python scripts/visualize.py

Requirements: matplotlib, numpy (scipy optional for contours)
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
# Custom colourmap: dark-purple → blue → teal → yellow → red
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
            candidates = json.load(f)

    return mask, heatmap, substations, bounds, candidates


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
    """Draw a semi-transparent dark overlay over forbidden cells."""
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
        ax.scatter(lons, lats, marker=marker, s=size, c=color,
                   edgecolors="white", linewidths=0.6, zorder=zorder, label="Existing substation")


def _annotate_candidates(ax, candidates, zorder=6):
    if not candidates:
        return
    cmap = plt.cm.plasma
    for c in candidates:
        rank = c["rank"]
        colour = cmap(1.0 - rank / (len(candidates) + 1))
        ax.scatter(c["lon"], c["lat"], marker="*", s=180 - rank * 8,
                   c=[colour], edgecolors="white", linewidths=0.5, zorder=zorder)
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


def _panel(ax, data, title, extent, cmap, mask, substations, candidates=None, vmin=0, vmax=1):
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
    if candidates:
        _annotate_candidates(ax, candidates)
    ax.set_title(title, fontsize=9, color="white", pad=4)
    ax.set_xlabel("Longitude", fontsize=7, color="#aaa")
    ax.set_ylabel("Latitude",  fontsize=7, color="#aaa")
    ax.tick_params(colors="#aaa", labelsize=6)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.yaxis.set_tick_params(color="#aaa", labelcolor="#aaa")
    return im


# ---------------------------------------------------------------------------
# Dashboard (2 × 3 grid)
# ---------------------------------------------------------------------------

def make_dashboard(mask, heatmap, substations, bounds, candidates, individual=None):
    """
    Build and save the 2×3 panel dashboard.

    If `individual` score arrays are supplied they are shown directly;
    otherwise we recompute them from the optimizer.
    """
    extent = _latlon_extent(bounds)

    fig = plt.figure(figsize=(18, 11), facecolor="#0d0221")
    fig.suptitle(
        "GPU Substation Placement Optimizer — Austin, TX",
        fontsize=15, color="white", fontweight="bold", y=0.98,
    )

    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    panels = [
        (gs[0, 0], heatmap,                       "Demand Heatmap",        _DEMAND_CMAP),
        (gs[0, 1], individual.get("load_relief"),  "Load Relief Score",     _SCORE_CMAP),
        (gs[0, 2], individual.get("loss_reduction"),"Loss Reduction Score",  _SCORE_CMAP),
        (gs[1, 0], individual.get("sustainability"),"Sustainability Score",  _SCORE_CMAP),
        (gs[1, 1], individual.get("redundancy"),    "Redundancy Score",      _SCORE_CMAP),
        (gs[1, 2], None,                            "Composite Score (Top 10)", _SCORE_CMAP),
    ]

    for spec, data, title, cmap in panels:
        ax = fig.add_subplot(spec)
        ax.set_facecolor("#0d0221")
        if data is None:
            # Composite — recompute on the fly if not supplied
            _comp, _ = _try_get_composite(mask, heatmap, substations, bounds)
            data = _comp
        if data is not None:
            show_candidates = candidates if "Composite" in title else None
            _panel(ax, data, title, extent, cmap, mask, substations, candidates=show_candidates)
        else:
            ax.text(0.5, 0.5, "Run optimizer first", transform=ax.transAxes,
                    ha="center", va="center", color="#aaa", fontsize=9)
            ax.set_title(title, fontsize=9, color="white", pad=4)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#00ffff", edgecolor="white", label="Existing substation (▲)"),
        mpatches.Patch(facecolor="#f7e200", edgecolor="white", label="Top-10 candidate (★)"),
        mpatches.Patch(facecolor="#444",    edgecolor="white", label="Forbidden area"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
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
# Large standalone composite map
# ---------------------------------------------------------------------------

def make_composite_map(mask, heatmap, substations, bounds, candidates):
    composite = _try_get_composite(mask, heatmap, substations, bounds)
    if composite is None:
        print("Composite score unavailable — skipping standalone map.")
        return

    extent = _latlon_extent(bounds)
    fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0d0221")
    ax.set_facecolor("#0d0221")

    im = ax.imshow(
        composite, extent=extent, origin="upper",
        cmap=_SCORE_CMAP, vmin=0, interpolation="bilinear", zorder=1,
    )
    _apply_forbidden_overlay(ax, mask, extent, alpha=0.4)
    _annotate_substations(ax, substations, size=80)
    _annotate_candidates(ax, candidates, zorder=6)

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Composite Score").ax.yaxis.set_tick_params(color="white", labelcolor="white")

    ax.set_title("Optimal Substation Placement — Composite Score Map", fontsize=13, color="white", pad=8)
    ax.set_xlabel("Longitude", color="#aaa")
    ax.set_ylabel("Latitude",  color="#aaa")
    ax.tick_params(colors="#aaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    # Score bar for top candidates
    if candidates:
        _add_candidate_sidebar(fig, candidates)

    out = _RESULTS_DIR / "composite_map.png"
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Composite map saved → {out}")


def _add_candidate_sidebar(fig, candidates):
    """Add a small inset table listing candidates."""
    ax_table = fig.add_axes([0.75, 0.12, 0.22, 0.35], facecolor="#11112a")
    ax_table.axis("off")
    ax_table.set_title("Top 10 Candidates", fontsize=8, color="white", pad=3)

    headers = ["#", "Lat", "Lon", "Score"]
    col_widths = [0.08, 0.32, 0.35, 0.25]
    row_height = 1.0 / (len(candidates) + 1.5)

    for ci, h in enumerate(headers):
        x = sum(col_widths[:ci]) + col_widths[ci] / 2
        ax_table.text(x, 1.0 - row_height * 0.6, h, transform=ax_table.transAxes,
                      ha="center", va="center", fontsize=6.5,
                      color="#00ffff", fontweight="bold")

    for ri, c in enumerate(candidates):
        y = 1.0 - row_height * (ri + 1.5)
        row_color = "#1e1e3a" if ri % 2 == 0 else "#131328"
        rect = mpatches.FancyBboxPatch(
            (0, y), 1.0, row_height,
            boxstyle="square,pad=0", transform=ax_table.transAxes,
            facecolor=row_color, edgecolor="none", zorder=0,
        )
        ax_table.add_patch(rect)
        row_vals = [str(c["rank"]), f"{c['lat']:.3f}", f"{c['lon']:.3f}",
                    f"{c['composite_score']:.3f}"]
        for ci2, val in enumerate(row_vals):
            x = sum(col_widths[:ci2]) + col_widths[ci2] / 2
            ax_table.text(x, y + row_height / 2, val, transform=ax_table.transAxes,
                          ha="center", va="center", fontsize=6, color="white")


# ---------------------------------------------------------------------------
# Helper: recompute composite score on the fly for visualization
# ---------------------------------------------------------------------------

def _try_get_composite(mask, heatmap, substations, bounds):
    try:
        import sys
        sys.path.insert(0, str(_ROOT))
        from src.optimizer.score import composite_score

        composite, individual, _ = composite_score(
            heatmap, substations, bounds, forbidden_mask=mask
        )
        return composite, individual
    except Exception as exc:
        print(f"  Warning: could not compute composite score for visualization: {exc}")
        return None, {}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    print("Loading data …")
    mask, heatmap, substations, bounds, candidates = _load_all()

    print("Computing scores for visualization …")
    composite_result = _try_get_composite(mask, heatmap, substations, bounds)
    if isinstance(composite_result, tuple):
        composite, individual = composite_result
    else:
        composite, individual = None, {}

    if composite is None:
        individual = {k: np.zeros((500, 500), dtype=np.float32)
                      for k in ("load_relief", "loss_reduction", "sustainability", "redundancy")}

    print("Rendering dashboard …")
    make_dashboard(mask, heatmap, substations, bounds, candidates, individual)

    print("Rendering composite map …")
    if composite is not None:
        extent = _latlon_extent(bounds)
        fig, ax = plt.subplots(figsize=(12, 10), facecolor="#0d0221")
        ax.set_facecolor("#0d0221")
        from matplotlib.colors import LinearSegmentedColormap
        im = ax.imshow(composite, extent=extent, origin="upper", cmap=_SCORE_CMAP, vmin=0, interpolation="bilinear", zorder=1)
        _apply_forbidden_overlay(ax, mask, extent, alpha=0.4)
        _annotate_substations(ax, substations, size=80)
        _annotate_candidates(ax, candidates, zorder=6)
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Composite Score").ax.yaxis.set_tick_params(color="white", labelcolor="white")
        ax.set_title("Optimal Substation Placement — Composite Score Map", fontsize=13, color="white", pad=8)
        ax.set_xlabel("Longitude", color="#aaa")
        ax.set_ylabel("Latitude", color="#aaa")
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

    print("\nVisualization complete.")


if __name__ == "__main__":
    main()
