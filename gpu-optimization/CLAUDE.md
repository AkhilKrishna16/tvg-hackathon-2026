# Person 2 — GPU Optimization Engine

**Branch:** `feature/optimizer`

---

## Your Job

Take the forbidden mask and demand heatmap as inputs, score every candidate location in the 500×500 grid, and return the top 10 placement candidates with scores. This is the core engine.

---

## Deliverables

- `src/optimizer/score.py` — main scoring engine
- `src/optimizer/objectives.py` — individual objective functions
- `src/optimizer/run.py` — CLI entrypoint: takes mask + heatmap, outputs `results/top_candidates.json`

---

## Scoring Model

Weighted sum across four objective functions:

```
score(cell) = w1 * load_relief(cell)
            + w2 * loss_reduction(cell)
            + w3 * sustainability(cell)
            + w4 * redundancy(cell)
```

---

## Inputs

All inputs are loaded from the `data/` directory:

| File | Shape / Type | Description |
|------|-------------|-------------|
| `data/forbidden_mask.npy` | 500×500 array | `1` = placeable, `0` = blocked |
| `data/demand_heatmap.npy` | 500×500 array | Normalized demand per cell |
| `data/existing_substations.geojson` | GeoJSON points | Existing substation locations |
| `data/city_bounds.json` | `{south, north, west, east}` | City bounding box |

---

## File Specifications

### `src/optimizer/objectives.py`

Implement four objective functions. Each takes the full 500×500 grid and returns a 500×500 score array (higher = better). **Use CuPy if available, fall back to NumPy silently if not.**

---

#### 1. `load_relief_score(demand_heatmap, existing_substations, city_bounds)`

For each candidate cell, compute how much load it would relieve from the nearest existing substation.

```
score[cell] = demand_heatmap[cell] * (1 / (distance_to_nearest_existing_substation + ε))
```

Normalize output to `[0, 1]`.

---

#### 2. `loss_reduction_score(demand_heatmap, existing_substations, city_bounds)`

Transmission loss is proportional to `distance² × load`. Rewards placing the new substation close to high-demand areas.

```
score[candidate] = Σ over all demand cells: demand[cell] / (distance(cell, candidate)² + ε)
```

Normalize output to `[0, 1]`.

---

#### 3. `sustainability_score(city_bounds)`

Proxy for renewable integration potential.

Use the NREL solar irradiance API to get GHI for Austin:

```
GET https://developer.nrel.gov/api/solar/solar_resource/v1.json
    ?lat=30.267&lon=-97.743&api_key=DEMO_KEY
```

Since irradiance is city-wide, build a gradient that rewards southern/western locations (slightly higher solar in those directions) using a simple linear spatial gradient.

Normalize output to `[0, 1]`.

---

#### 4. `redundancy_score(existing_substations, city_bounds)`

Rewards locations that are **not too close** to existing substations (avoids clustering) but also **not too far** (avoids isolation). Score peaks at ~3–5 km from the nearest existing substation.

```
score[cell] = exp( -((dist - 4000)²) / (2 × 2000²) )
```

Gaussian centered at **4 km**, with σ = 2 km. Normalize output to `[0, 1]`.

---

### `src/optimizer/score.py`

Implement:

```python
composite_score(demand_heatmap, existing_substations, city_bounds, weights=None)
```

**Default weights:**

| Objective | Weight |
|-----------|--------|
| `load_relief` | 0.35 |
| `loss_reduction` | 0.35 |
| `sustainability` | 0.15 |
| `redundancy` | 0.15 |

The `weights` parameter is a dict that overrides the defaults.

Returns a 500×500 NumPy array of composite scores with all forbidden cells already zeroed out.

---

### `src/optimizer/run.py`

CLI script that executes the full pipeline end-to-end:

1. Load all input files from `data/`
2. Run `composite_score` with default weights
3. Find top 10 candidate cells — **ignoring cells within 10 grid cells of each other** (no clustering)
4. Convert grid indices back to lat/lon coordinates using `city_bounds`
5. Save to `results/top_candidates.json`

**Output format:**

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

## Requirements

- Try to `import cupy`, fall back to `numpy` silently if unavailable
- Print timing info (how long each scoring step took, and total)
- Create the `results/` directory if it does not exist

---

## Notes

- All objective functions must handle the case where `existing_substations` has zero features gracefully (return a uniform score array rather than crashing)
- The non-clustering filter in `run.py` should use a simple greedy suppression: once a cell is selected, zero out all cells within a 10-cell radius before selecting the next candidate
- Coordinate conversion from grid index `(i, j)` to `(lat, lon)` uses linear interpolation over `city_bounds`
