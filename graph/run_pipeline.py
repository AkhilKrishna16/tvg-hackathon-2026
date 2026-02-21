"""
End-to-end substation placement optimizer.

Usage:
    python run_pipeline.py                              # Austin TX, 500x500
    python run_pipeline.py "Houston, Texas, USA"
    python run_pipeline.py "Chicago, Illinois, USA" --grid 300
    python run_pipeline.py --city "Phoenix, Arizona, USA" --grid 400 --top 30
    python run_pipeline.py --skip-ingest               # re-score existing data
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Make src/ importable regardless of CWD ───────────────────────────────────
ROOT      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

from src.data.ingest       import run_ingest
from src.scoring.score     import run_scoring
from src.viz.visualize     import run_visualization


def _run_subprocess(cmd, cwd, step_name):
    """Run a subprocess, streaming output. Returns True on success."""
    print(f"\n[pipeline] {step_name}")
    proc = subprocess.Popen(
        cmd, cwd=cwd,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line.strip():
            print(f"  {line}")
    proc.wait()
    if proc.returncode != 0:
        print(f"  [WARNING] {step_name} exited with code {proc.returncode}")
        return False
    return True


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
                        help="Grid resolution N -> N×N cells (default: 500)")
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
    parser.add_argument("--skip-gpu",    action="store_true",
                        help="Skip GPU optimizer step")
    parser.add_argument("--skip-ai",     action="store_true",
                        help="Skip Claude feasibility analysis step")
    args = parser.parse_args()

    # --city flag takes priority over positional
    city = args.city_flag if args.city_flag else args.city

    # Default paths
    data_dir    = args.data_dir    or os.path.join(ROOT, "data")
    results_dir = args.results_dir or os.path.join(ROOT, "results")

    # Other module paths
    gpu_dir      = os.path.join(REPO_ROOT, "gpu-optimization")
    site_dir     = os.path.join(REPO_ROOT, "sitereliability")
    frontend_dir = os.path.join(REPO_ROOT, "frontend")

    t_start = time.perf_counter()

    # ── Step 1: Ingest ────────────────────────────────────────────────────────
    if not args.skip_ingest:
        run_ingest(city=city, grid=args.grid, output_dir=data_dir)
    else:
        print(f"[pipeline] Skipping ingest -- using existing data in {data_dir}")

    # ── Step 2: Graph scoring (simple heuristic) ──────────────────────────────
    candidates = run_scoring(data_dir=data_dir, results_dir=results_dir, top_n=args.top)

    # ── Step 3: Visualize ─────────────────────────────────────────────────────
    if not args.skip_viz:
        fig_path = os.path.join(results_dir, "placement_analysis.png")
        run_visualization(data_dir=data_dir, results_dir=results_dir, out_path=fig_path)
    else:
        print("[pipeline] Skipping visualization.")

    # ── Step 4: GPU Optimizer ─────────────────────────────────────────────────
    gpu_candidates_path = os.path.join(gpu_dir, "results", "top_candidates.json")
    if not args.skip_gpu and os.path.isdir(gpu_dir):
        mask_path   = os.path.join(data_dir, "forbidden_mask.npy")
        heatmap_path= os.path.join(data_dir, "demand_heatmap.npy")
        subs_path   = os.path.join(data_dir, "existing_substations.geojson")
        bounds_path = os.path.join(data_dir, "city_bounds.json")

        if all(os.path.exists(p) for p in [mask_path, heatmap_path, subs_path, bounds_path]):
            os.makedirs(os.path.join(gpu_dir, "results"), exist_ok=True)
            cmd = [
                sys.executable, "-m", "src.optimizer.run",
                "--mask",        mask_path,
                "--heatmap",     heatmap_path,
                "--substations", subs_path,
                "--bounds",      bounds_path,
                "--output",      gpu_candidates_path,
                "--top-n",       str(args.top),
            ]
            _run_subprocess(cmd, gpu_dir, "Step 4/6  GPU optimizer")
        else:
            print("[pipeline] Skipping GPU optimizer -- missing input data files")
    else:
        print("[pipeline] Skipping GPU optimizer step.")

    # ── Step 5: Claude Feasibility Analysis ───────────────────────────────────
    # Use GPU optimizer output if available, otherwise fall back to graph results
    import json as _json
    if os.path.exists(gpu_candidates_path):
        candidates_input = gpu_candidates_path
    else:
        candidates_input = os.path.join(results_dir, "top_candidates.json")

    ai_output = os.path.join(results_dir, "top_candidates_enriched.json")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not args.skip_ai and api_key and os.path.exists(candidates_input):
        cmd = [
            sys.executable, "-m", "src.agent.feasibility",
            "--candidates", candidates_input,
            "--output",     ai_output,
            "--city",       city,
        ]
        ok = _run_subprocess(cmd, site_dir, "Step 5/6  Claude feasibility analysis")
        if not ok or not os.path.exists(ai_output):
            ai_output = candidates_input   # fall back to unenriched
    elif not api_key:
        print("[pipeline] Skipping AI step -- ANTHROPIC_API_KEY not set")
        ai_output = candidates_input
    else:
        ai_output = candidates_input

    # ── Step 6: Copy to frontend ──────────────────────────────────────────────
    fe_data    = os.path.join(frontend_dir, "data")
    fe_results = os.path.join(frontend_dir, "results")
    os.makedirs(fe_data,    exist_ok=True)
    os.makedirs(fe_results, exist_ok=True)

    # Copy input data files
    for fname in ("forbidden_mask.npy", "demand_heatmap.npy",
                  "existing_substations.geojson", "city_bounds.json"):
        src = os.path.join(data_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(fe_data, fname))

    # Copy final candidates (enriched if available)
    final_candidates = ai_output if os.path.exists(ai_output) else candidates_input
    if os.path.exists(final_candidates):
        shutil.copy2(final_candidates, os.path.join(fe_results, "top_candidates.json"))
        print(f"[pipeline] Copied candidates -> {os.path.join(fe_results, 'top_candidates.json')}")

    # Copy feasibility report if generated
    report_src = os.path.join(results_dir, "feasibility_report.md")
    if os.path.exists(report_src):
        shutil.copy2(report_src, os.path.join(fe_results, "feasibility_report.md"))

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.perf_counter() - t_start
    best = candidates[0] if candidates else {}

    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  City   : {city}")
    if best:
        print(f"  Best   : ({best.get('lat', '?'):.5f}N, {best.get('lon', '?'):.5f}E)"
              f"  score={best.get('score', best.get('composite_score', '?'))}")
    if not args.skip_viz:
        print(f"  Figure : {os.path.join(results_dir, 'placement_analysis.png')}")
    print(f"  Data   : {data_dir}")
    print(f"  Results: {fe_results}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
