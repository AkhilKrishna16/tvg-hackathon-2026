"""
End-to-end substation placement optimizer.

Usage:
    python run_pipeline.py                              # Austin TX, 500×500
    python run_pipeline.py "Houston, Texas, USA"
    python run_pipeline.py "Chicago, Illinois, USA" --grid 300
    python run_pipeline.py --city "Phoenix, Arizona, USA" --grid 400 --top 30
    python run_pipeline.py --skip-ingest               # re-score existing data
"""

import argparse
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Make src/ importable regardless of CWD ───────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.data.ingest       import run_ingest
from src.scoring.score     import run_scoring
from src.viz.visualize     import run_visualization


def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated substation placement optimizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py
  python run_pipeline.py "Houston, Texas, USA"
  python run_pipeline.py --city "Denver, Colorado, USA" --grid 400
  python run_pipeline.py --skip-ingest --top 50
        """,
    )
    parser.add_argument(
        "city", nargs="?", default="Austin, Texas, USA",
        help='Target city (default: "Austin, Texas, USA")',
    )
    parser.add_argument("--city", dest="city_flag", default=None,
                        help="Alternative way to pass the city name")
    parser.add_argument("--grid", type=int, default=500,
                        help="Grid resolution N → N×N cells (default: 500)")
    parser.add_argument("--top",  type=int, default=20,
                        help="Number of top candidates to output (default: 20)")
    parser.add_argument("--data-dir",    default=None,
                        help="Override data directory path")
    parser.add_argument("--results-dir", default=None,
                        help="Override results directory path")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip data ingestion (use existing data/ files)")
    parser.add_argument("--skip-viz",    action="store_true",
                        help="Skip visualization step")
    args = parser.parse_args()

    # --city flag takes priority over positional
    city = args.city_flag if args.city_flag else args.city

    # Default paths
    data_dir    = args.data_dir    or os.path.join(ROOT, "data")
    results_dir = args.results_dir or os.path.join(ROOT, "results")

    t_start = time.perf_counter()

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    if not args.skip_ingest:
        run_ingest(city=city, grid=args.grid, output_dir=data_dir)
    else:
        print(f"[pipeline] Skipping ingest — using existing data in {data_dir}")

    # ── Step 2: Score ─────────────────────────────────────────────────────────
    candidates = run_scoring(data_dir=data_dir, results_dir=results_dir, top_n=args.top)

    # ── Step 3: Visualize ─────────────────────────────────────────────────────
    if not args.skip_viz:
        fig_path = os.path.join(results_dir, "placement_analysis.png")
        run_visualization(data_dir=data_dir, results_dir=results_dir, out_path=fig_path)
    else:
        print("[pipeline] Skipping visualization.")

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    best = candidates[0]

    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  City   : {city}")
    print(f"  Best   : ({best['lat']:.5f}°N, {best['lon']:.5f}°E)  score={best['score']:.4f}")
    if not args.skip_viz:
        print(f"  Figure : {os.path.join(results_dir, 'placement_analysis.png')}")
    print(f"  Data   : {data_dir}")
    print(f"  Results: {results_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
