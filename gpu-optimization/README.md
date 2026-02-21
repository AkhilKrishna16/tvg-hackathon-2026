# GPU Substation Placement Optimizer

> **TVG Hackathon 2026 · Person 2 deliverable**
> Branch: `feature/optimizer`

A GPU-accelerated engine that scores every cell in a 500 × 500 city grid and
returns the **top 10 optimal locations** for a new electrical substation,
balancing four competing objectives.

---

## Architecture

```
gpu-optimization/
├── src/optimizer/
│   ├── objectives.py   ← four objective functions (CuPy / NumPy)
│   ├── score.py        ← composite_score() + timing
│   └── run.py          ← CLI entrypoint → results/top_candidates.json
├── data/
│   ├── city_bounds.json
│   ├── existing_substations.geojson
│   ├── forbidden_mask.npy      (generated)
│   └── demand_heatmap.npy      (generated)
├── scripts/
│   ├── generate_data.py        ← synthetic Austin, TX dataset
│   └── visualize.py            ← publication-quality dashboard
└── results/
    ├── top_candidates.json
    ├── optimizer_dashboard.png
    └── composite_map.png
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# (Optional) GPU acceleration — match your CUDA version
# pip install cupy-cuda12x

# 2. Generate synthetic Austin, TX input data
python scripts/generate_data.py

# 3. Run the optimizer
python -m src.optimizer.run

# 4. Visualize results
python scripts/visualize.py
```

---

## Scoring Model

```
score(cell) = w₁ · load_relief(cell)
            + w₂ · loss_reduction(cell)
            + w₃ · sustainability(cell)
            + w₄ · redundancy(cell)
```

| Objective        | Weight | Description |
|-----------------|--------|-------------|
| `load_relief`   | 0.35   | High demand × far from existing substations |
| `loss_reduction`| 0.35   | Minimise I²R transmission losses (FFT convolution) |
| `sustainability`| 0.15   | Solar integration potential via NREL GHI API |
| `redundancy`    | 0.15   | Gaussian spacing bonus, peak at 4 km from nearest substation |

### Weight overrides

```bash
python -m src.optimizer.run \
    --weights '{"load_relief": 0.5, "loss_reduction": 0.3, "sustainability": 0.1, "redundancy": 0.1}'
```

---

## Inputs

| File | Shape | Description |
|------|-------|-------------|
| `data/forbidden_mask.npy` | 500 × 500 | 1 = placeable, 0 = blocked |
| `data/demand_heatmap.npy` | 500 × 500 | Normalised demand [0, 1] |
| `data/existing_substations.geojson` | GeoJSON | Existing substation points |
| `data/city_bounds.json` | JSON | `{south, north, west, east}` |

---

## Output

`results/top_candidates.json` — array of 10 objects:

```json
[
  {
    "rank": 1,
    "lat": 30.234,
    "lon": -97.812,
    "composite_score": 0.847,
    "load_relief_score": 0.91,
    "loss_reduction_score": 0.88,
    "sustainability_score": 0.72,
    "redundancy_score": 0.65
  }
]
```

---

## Performance

| Mode | Hardware | Typical runtime |
|------|----------|-----------------|
| NumPy (CPU) | 8-core laptop | ~4–8 s |
| CuPy (GPU) | RTX 3080 | ~0.3–0.8 s |

The `loss_reduction` objective uses **FFT convolution** (O(N² log N)) instead of
naive O(N⁴) brute force — a ~250× speedup for a 500 × 500 grid.

---

## API

```python
from src.optimizer.score import composite_score

composite, individual, timings = composite_score(
    demand_heatmap,
    existing_substations,   # GeoJSON dict
    city_bounds,            # {south, north, west, east}
    forbidden_mask=mask,
    weights={"load_relief": 0.4, "redundancy": 0.2},
)
# composite    : np.ndarray (500, 500) — forbidden cells zeroed
# individual   : dict[str, np.ndarray] — per-objective maps
# timings      : dict[str, float]      — seconds per step
```

---

## Algorithm Details

### Load Relief
```
score[i,j] = demand[i,j] / (dist_to_nearest_substation_m + ε)
```
Haversine distances computed with NumPy broadcasting. Cells with high demand
that sit far from any existing substation score highest.

### Loss Reduction
```
score[candidate] = Σ_cells  demand[cell] / (dist(cell, candidate)² + ε)
```
Implemented as **2-D FFT convolution** with kernel `K[Δi, Δj] = 1 / (Δi² + Δj² + 1)`.
On GPU, uses `cupyx.scipy.signal.fftconvolve`; on CPU, uses `scipy.signal.fftconvolve`.

### Sustainability
Queries the **NREL Solar Resource API** for Austin's annual average GHI, then
applies a linear spatial gradient that rewards southern/western cells (higher
solar irradiance). Falls back to Austin's historical average (5.5 kWh/m²/day)
if the network is unavailable.

### Redundancy
```
score[i,j] = exp( -((dist_m − 4000)²) / (2 × 2000²) )
```
Gaussian bell curve centred at **4 km** from the nearest existing substation,
σ = 2 km. Prevents both clustering and isolation.

### Anti-Clustering Filter
After selecting each candidate, all cells within a **10-cell radius** are zeroed
before the next selection (greedy suppression). This ensures geographic diversity
in the final top-10 list.

---

## Development

```bash
# Run tests
pytest tests/ -v

# Benchmark (requires pytest-benchmark)
pytest tests/ --benchmark-only

# Lint
ruff check src/ scripts/
```
