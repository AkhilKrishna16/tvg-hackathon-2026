# Energy Grid Placement Optimizer

An AI-powered tool for identifying optimal electrical substation placement sites. The system ingests real city data, runs a GPU-accelerated scoring engine across a fine-grained candidate grid, then uses Claude to assess real-world feasibility for each top site — all visualized in an interactive Plotly Dash dashboard.

Built for TVG Hackathon 2026.

---

## How It Works

The pipeline has four stages that run end-to-end:

```
OpenStreetMap / HIFLD
        │
        ▼
 [1] Data Ingestion          graph/src/data/ingest.py
     - City boundary (osmnx)
     - Forbidden zones (parks, hospitals, airports, water)
     - Demand heatmap (population proxy via Gaussian blobs)
     - Existing substations (HIFLD open data)
        │
        ▼
 [2] GPU Optimization        gpu-optimization/src/optimizer/
     - 4 objective functions scored across N×N grid
     - Load Relief · Loss Reduction · Sustainability · Redundancy
     - Anti-clustering suppression → top 10 candidates
        │
        ▼
 [3] Claude AI Feasibility   sitereliability/src/agent/feasibility.py
     - claude-sonnet-4-6 analyzes each candidate site
     - Assesses land use, zoning, environmental flags,
       community sensitivity, grid proximity
     - Verdict: HIGH / MEDIUM / LOW
        │
        ▼
 [4] Interactive Dashboard   frontend/src/viz/app.py
     - Plotly Dash on http://localhost:8050
     - Live map with demand heatmap, forbidden zones,
       existing substations, and ranked candidate markers
     - Click-to-zoom + full AI site intelligence report
```

---

## Project Structure

```
tvg-hackathon-2026/
├── graph/                        # Stage 1: Data ingestion
│   ├── src/data/ingest.py        # Main ingest script
│   ├── src/scoring/score.py      # Simple graph-based pre-scorer
│   ├── src/viz/visualize.py      # Matplotlib output chart
│   ├── run_pipeline.py           # CLI runner for all 4 stages
│   └── data/                     # Generated input data (gitignored)
│
├── gpu-optimization/             # Stage 2: Scoring engine
│   ├── src/optimizer/
│   │   ├── objectives.py         # 4 objective functions (CuPy/NumPy)
│   │   ├── score.py              # Composite scoring + parallelism
│   │   ├── analysis.py           # Candidate selection, coverage metrics
│   │   └── run.py                # CLI entrypoint
│   ├── scripts/generate_data.py  # Synthetic data generator (for testing)
│   └── results/                  # Output JSON (gitignored)
│
├── sitereliability/              # Stage 3: Claude AI agent
│   └── src/agent/
│       ├── feasibility.py        # Async Claude analysis (3 concurrent)
│       └── prompts.py            # Prompt templates
│
└── frontend/                     # Stage 4: Dash dashboard
    ├── src/viz/
    │   ├── app.py                # Main Dash app + pipeline orchestration
    │   └── map_layers.py         # Plotly map trace builders
    └── data/ / results/          # Copied here by the pipeline
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install dash plotly dash-bootstrap-components geopandas numpy scipy osmnx shapely requests matplotlib anthropic
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

The AI feasibility stage (Stage 3) is skipped gracefully if this is not set — the rest of the pipeline still runs.

### 3. Launch the dashboard

```bash
cd frontend
python src/viz/app.py
```

Open **http://localhost:8050** in your browser.

From the UI you can:
- Enter any city name (default: `Austin, Texas, USA`)
- Tune the four optimization weights with sliders
- Choose **Fast (200×200)** or **Standard (500×500)** grid resolution
- Click **Run Full Pipeline** — all four stages run in the background and the map refreshes automatically when done

---

## Running the Pipeline from the CLI

If you prefer the command line, use the orchestrator in `graph/`:

```bash
cd graph

# Default: Austin TX, 500×500 grid, top 20 candidates
python run_pipeline.py

# Custom city and grid size
python run_pipeline.py "Houston, Texas, USA" --grid 300

# Skip data ingestion (re-use existing data/ files)
python run_pipeline.py --skip-ingest

# Skip the GPU optimizer or AI steps
python run_pipeline.py --skip-gpu
python run_pipeline.py --skip-ai

# All options
python run_pipeline.py --help
```

Results are written to `graph/results/` and copied to `frontend/results/` automatically.

---

## Running Individual Stages

### Stage 1 — Data Ingestion

```bash
cd graph
python -c "from src.data.ingest import run_ingest; run_ingest('Austin, Texas, USA', grid=200, output_dir='data')"
```

Outputs to `graph/data/`:
- `forbidden_mask.npy` — 0 = forbidden, 1 = placeable
- `demand_heatmap.npy` — normalized [0, 1] demand proxy
- `existing_substations.geojson` — existing substation points
- `city_bounds.json` — bounding box `{south, north, west, east}`

### Stage 2 — GPU Optimization

```bash
cd gpu-optimization
python -m src.optimizer.run
```

Options:
```bash
python -m src.optimizer.run \
  --mask        ../graph/data/forbidden_mask.npy \
  --heatmap     ../graph/data/demand_heatmap.npy \
  --substations ../graph/data/existing_substations.geojson \
  --bounds      ../graph/data/city_bounds.json \
  --output      results/top_candidates.json \
  --top-n       10 \
  --weights     '{"load_relief": 0.4, "loss_reduction": 0.3}'
```

Sensitivity analysis across 8 weight-set combinations:
```bash
python -m src.optimizer.run --sensitivity
```

### Stage 3 — Claude AI Feasibility

```bash
cd sitereliability
python -m src.agent.feasibility \
  --candidates ../gpu-optimization/results/top_candidates.json \
  --output     ../gpu-optimization/results/top_candidates_enriched.json \
  --city       "Austin, Texas, USA"
```

Outputs:
- Enriched `top_candidates_enriched.json` with `feasibility` object per candidate
- `feasibility_report.md` — human-readable markdown report

### Stage 4 — Dashboard Only (with existing results)

```bash
cd frontend
python src/viz/app.py
```

---

## Scoring Model

The GPU optimizer scores every grid cell using a weighted sum of four objectives:

| Objective | Default Weight | Description |
|---|---|---|
| **Load Relief** | 0.35 | High demand × far from existing substations |
| **Loss Reduction** | 0.35 | Minimizes I²R transmission losses (FFT convolution) |
| **Sustainability** | 0.15 | Solar irradiance proxy — rewards south/west placement |
| **Redundancy** | 0.15 | Gaussian peak at 4 km from nearest existing substation |

All scores are normalized to [0, 1]. Forbidden cells are zeroed out. Anti-clustering suppression ensures the top candidates are spaced at least 10 grid cells apart.

**GPU acceleration:** the engine uses CuPy if an NVIDIA GPU is available and falls back to NumPy silently.

---

## Candidate Output Schema

Each entry in `top_candidates.json`:

```json
{
  "rank": 1,
  "lat": 30.28491,
  "lon": -97.73412,
  "composite_score": 0.847,
  "load_relief_score": 0.912,
  "loss_reduction_score": 0.881,
  "sustainability_score": 0.723,
  "redundancy_score": 0.651,
  "nearest_existing_km": 3.4,
  "coverage_3km_pct": 18.2,
  "coverage_5km_pct": 34.7,
  "coverage_10km_pct": 61.1,
  "feasibility": {
    "feasibility": "HIGH",
    "land_use": "Light industrial / commercial corridor",
    "zoning_assessment": "Industrial — utility use feasible",
    "environmental_flags": [],
    "community_sensitivity": "Low — minimal residential proximity",
    "grid_proximity": "~0.8 km from 138kV transmission corridor",
    "reasoning": "..."
  }
}
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `dash`, `plotly`, `dash-bootstrap-components` | Interactive frontend |
| `numpy`, `scipy` | Numerical scoring (CPU) |
| `cupy` _(optional)_ | GPU-accelerated scoring (CUDA required) |
| `osmnx`, `geopandas`, `shapely` | OpenStreetMap data ingestion |
| `requests` | HIFLD substations API + NREL solar API |
| `anthropic` | Claude AI feasibility analysis |
| `matplotlib` | Static output charts |

Install all at once:
```bash
pip install dash plotly dash-bootstrap-components geopandas numpy scipy osmnx shapely requests matplotlib anthropic
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | No (recommended) | Enables Claude AI feasibility stage |

Without `ANTHROPIC_API_KEY`, the pipeline skips Stage 3 and uses score-derived feasibility labels (`HIGH` / `MEDIUM` / `LOW` based on composite score thresholds).
