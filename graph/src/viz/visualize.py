"""
Visualization for the substation placement optimizer.
Produces a single comprehensive figure with 6 panels.

Usage:
    python src/viz/visualize.py
    python src/viz/visualize.py --data data/ --results results/ --out results/map.png
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering; swap to "TkAgg" for interactive
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# ─────────────────────────────────────────────────────────────────────────────
# Colour maps
# ─────────────────────────────────────────────────────────────────────────────
_FORBIDDEN_CMAP = LinearSegmentedColormap.from_list(
    "forbidden", ["#d62728", "#2ca02c"], N=2)   # red=forbidden, green=valid

_SCORE_CMAP  = "hot"
_DEMAND_CMAP = "YlOrRd"
_ISO_CMAP    = "Blues"


def _load_data(data_dir: str, results_dir: str) -> dict:
    d = {}
    d["forbidden_mask"]  = np.load(os.path.join(data_dir,    "forbidden_mask.npy"))
    d["demand_heatmap"]  = np.load(os.path.join(data_dir,    "demand_heatmap.npy"))
    d["scores"]          = np.load(os.path.join(results_dir, "scores.npy"))
    d["isolation_score"] = np.load(os.path.join(results_dir, "isolation_score.npy"))

    with open(os.path.join(data_dir, "city_bounds.json")) as f:
        d["bounds"] = json.load(f)
    with open(os.path.join(data_dir, "existing_substations.geojson")) as f:
        d["substations"] = json.load(f)
    with open(os.path.join(results_dir, "top_candidates.json")) as f:
        d["candidates"] = json.load(f)
    with open(os.path.join(data_dir, "city_meta.json")) as f:
        d["meta"] = json.load(f)

    b = d["bounds"]
    grid = d["forbidden_mask"].shape[0]
    d["lats"] = np.linspace(b["south"], b["north"], grid)
    d["lons"] = np.linspace(b["west"],  b["east"],  grid)
    return d


def _extent(bounds: dict) -> list:
    """Matplotlib imshow extent=[left, right, bottom, top]."""
    return [bounds["west"], bounds["east"], bounds["south"], bounds["north"]]


def _draw_substations(ax, substations: dict, **kwargs):
    kw = dict(marker="^", s=60, c="cyan", edgecolors="k", linewidths=0.6,
              zorder=5, label="Existing substation")
    kw.update(kwargs)
    for feat in substations["features"]:
        try:
            lon, lat = feat["geometry"]["coordinates"][:2]
            ax.scatter(lon, lat, **kw)
            kw.pop("label", None)   # only label the first
        except Exception:
            pass


def _draw_candidates(ax, candidates: list, n: int = 10, star_best: bool = True):
    for c in candidates[:n]:
        if c["rank"] == 1 and star_best:
            ax.scatter(c["lon"], c["lat"], marker="*", s=280,
                       c="#FFD700", edgecolors="k", linewidths=0.8, zorder=10,
                       label="Best location")
        else:
            ax.scatter(c["lon"], c["lat"], marker="o", s=40,
                       c="white", edgecolors="k", linewidths=0.6, zorder=9,
                       label=("Top 10" if c["rank"] == 2 else "_nolegend_"))


def run_visualization(data_dir: str = None, results_dir: str = None,
                       out_path: str = None) -> str:
    """Generate the analysis figure and return the saved file path."""

    if data_dir is None:
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    if results_dir is None:
        results_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "results"))
    if out_path is None:
        out_path = os.path.join(results_dir, "placement_analysis.png")

    print(f"\n{'='*60}")
    print(f"  Substation Placement — Visualization")
    print(f"  Data    : {data_dir}")
    print(f"  Results : {results_dir}")
    print(f"{'='*60}\n")

    d = _load_data(data_dir, results_dir)
    bounds = d["bounds"]
    ext    = _extent(bounds)
    city   = d["meta"].get("city", "Unknown city")
    best   = d["candidates"][0]

    # ── Figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 14), dpi=120)
    fig.patch.set_facecolor("#1a1a2e")

    gs = GridSpec(2, 3, figure=fig,
                  left=0.05, right=0.97, top=0.90, bottom=0.06,
                  wspace=0.35, hspace=0.40)

    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

    title = (f"Optimal Substation Placement — {city}\n"
             f"Best candidate: ({best['lat']:.5f}°N, {best['lon']:.5f}°E)  "
             f"score={best['score']:.4f}")
    fig.suptitle(title, color="white", fontsize=14, fontweight="bold", y=0.96)

    def _style(ax, title_str):
        ax.set_facecolor("#0d0d1a")
        ax.set_title(title_str, color="white", fontsize=10, pad=4)
        ax.tick_params(colors="#888888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.set_xlabel("Longitude", color="#888888", fontsize=7)
        ax.set_ylabel("Latitude",  color="#888888", fontsize=7)

    # ── Panel 1: Forbidden mask ───────────────────────────────────────────────
    ax = axes[0]
    im = ax.imshow(d["forbidden_mask"], cmap=_FORBIDDEN_CMAP,
                   extent=ext, origin="lower", aspect="auto", vmin=0, vmax=1)
    _draw_substations(ax, d["substations"])
    _style(ax, "Forbidden Zones  (green = valid)")
    valid_pct = d["forbidden_mask"].mean() * 100
    ax.text(0.02, 0.02, f"{valid_pct:.1f}% valid", transform=ax.transAxes,
            color="white", fontsize=8, va="bottom")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Valid (1) / Forbidden (0)",
                 ).ax.yaxis.label.set_color("white")
    ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white", markerscale=0.8)

    # ── Panel 2: Demand heatmap ───────────────────────────────────────────────
    ax = axes[1]
    im = ax.imshow(d["demand_heatmap"], cmap=_DEMAND_CMAP,
                   extent=ext, origin="lower", aspect="auto", vmin=0, vmax=1)
    _draw_substations(ax, d["substations"])
    _style(ax, "Electricity Demand  (normalized)")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, label="Demand [0–1]")
    cb.ax.yaxis.label.set_color("white")
    cb.ax.tick_params(colors="white")

    # ── Panel 3: Isolation score ──────────────────────────────────────────────
    ax = axes[2]
    im = ax.imshow(d["isolation_score"], cmap=_ISO_CMAP,
                   extent=ext, origin="lower", aspect="auto", vmin=0, vmax=1)
    _draw_substations(ax, d["substations"])
    _style(ax, "Isolation from Existing Substations")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, label="Isolation [0–1]")
    cb.ax.yaxis.label.set_color("white")
    cb.ax.tick_params(colors="white")
    ax.text(0.02, 0.02, "High = underserved area", transform=ax.transAxes,
            color="white", fontsize=8, va="bottom")

    # ── Panel 4: Composite score ──────────────────────────────────────────────
    ax = axes[3]
    im = ax.imshow(d["scores"], cmap=_SCORE_CMAP,
                   extent=ext, origin="lower", aspect="auto")
    _draw_substations(ax, d["substations"])
    _draw_candidates(ax, d["candidates"], n=10)
    _style(ax, "Composite Placement Score  (higher = better)")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, label="Score")
    cb.ax.yaxis.label.set_color("white")
    cb.ax.tick_params(colors="white")
    ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white", markerscale=0.8)

    # ── Panel 5: Zoomed view of best candidate ────────────────────────────────
    ax = axes[4]
    zoom_deg = 0.08  # ~8km radius around best location
    zb = {
        "west":  best["lon"] - zoom_deg,
        "east":  best["lon"] + zoom_deg,
        "south": best["lat"] - zoom_deg,
        "north": best["lat"] + zoom_deg,
    }
    # Clip to bounds
    zb["west"]  = max(zb["west"],  bounds["west"])
    zb["east"]  = min(zb["east"],  bounds["east"])
    zb["south"] = max(zb["south"], bounds["south"])
    zb["north"] = min(zb["north"], bounds["north"])
    zext = _extent(zb)

    # Slice arrays to the zoom region
    lats, lons = d["lats"], d["lons"]
    ri = np.where((lats >= zb["south"]) & (lats <= zb["north"]))[0]
    rj = np.where((lons >= zb["west"])  & (lons <= zb["east"]))[0]
    zoom_scores = d["scores"][ri[0]:ri[-1]+1, rj[0]:rj[-1]+1]

    ax.imshow(zoom_scores, cmap=_SCORE_CMAP,
              extent=zext, origin="lower", aspect="auto")
    ax.scatter(best["lon"], best["lat"], marker="*", s=350,
               c="#FFD700", edgecolors="k", linewidths=1, zorder=10,
               label="Best location")
    _draw_substations(ax, d["substations"])
    _style(ax, f"Zoom: Best Site  ({best['lat']:.4f}°N, {best['lon']:.4f}°E)")
    ax.legend(fontsize=7, facecolor="#1a1a2e", labelcolor="white", markerscale=0.8)

    # ── Panel 6: Score histogram ──────────────────────────────────────────────
    ax = axes[5]
    ax.set_facecolor("#0d0d1a")
    valid_scores = d["scores"][d["forbidden_mask"] == 1.0]
    n_bins = 80
    counts, bin_edges, patches = ax.hist(valid_scores, bins=n_bins,
                                          color="#e07b39", edgecolor="none",
                                          alpha=0.85)
    # Colour-code by score
    for p, left in zip(patches, bin_edges):
        p.set_facecolor(plt.cm.hot(left / (valid_scores.max() or 1)))

    # Mark top-1 score
    ax.axvline(best["score"], color="#FFD700", linewidth=1.5,
               label=f"Best: {best['score']:.4f}")
    ax.axvline(valid_scores.mean(), color="white", linewidth=1, linestyle="--",
               label=f"Mean: {valid_scores.mean():.4f}")

    top_pct = (valid_scores >= best["score"] * 0.9).mean() * 100
    ax.text(0.98, 0.95,
            f"Valid cells: {len(valid_scores):,}\n"
            f"Top 10% threshold: {valid_scores.quantile(0.9) if hasattr(valid_scores, 'quantile') else np.quantile(valid_scores, 0.9):.4f}",
            transform=ax.transAxes, ha="right", va="top",
            color="white", fontsize=8)

    _style(ax, "Score Distribution  (valid cells only)")
    ax.set_xlabel("Composite Score", color="#888888", fontsize=8)
    ax.set_ylabel("Cell Count",      color="#888888", fontsize=8)
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Figure saved → {out_path}")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize placement results.")
    parser.add_argument("--data",    default=None)
    parser.add_argument("--results", default=None)
    parser.add_argument("--out",     default=None)
    args = parser.parse_args()
    run_visualization(data_dir=args.data, results_dir=args.results, out_path=args.out)
