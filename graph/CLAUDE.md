You are building the data ingestion layer for a GPU-accelerated power grid placement optimizer. The goal of the overall system is to find the optimal location to place a new electrical substation in Austin, Texas by scoring 250,000 candidate locations across the city. Your job is to produce all the input data that the scoring engine needs.

---

## What you are building

A single Python script: `src/data/ingest.py`

When run, it produces these files in the `data/` directory:
- `data/forbidden_mask.npy` — 500x500 numpy array. 1.0 = valid placement site, 0.0 = forbidden
- `data/demand_heatmap.npy` — 500x500 numpy array. Normalized [0,1] relative electricity demand per cell
- `data/existing_substations.geojson` — Point locations of existing substations in Austin
- `data/city_bounds.json` — Bounding box of Austin: `{"south": float, "north": float, "west": float, "east": float}`

---

## Step-by-step instructions

### Step 1 — Get Austin's bounding box
Use osmnx to get Austin's city boundary and extract its bounding box. Save as `data/city_bounds.json`.

Fallback if osmnx fails: hardcode `{"south": 30.098, "north": 30.515, "west": -97.938, "east": -97.561}`

### Step 2 — Build the 500x500 candidate grid
Discretize the bounding box into a 500x500 grid. Each cell (i, j) maps to a lat/lon centroid. Store the lat and lon arrays so downstream code can convert grid indices back to coordinates.

```python
import numpy as np
lats = np.linspace(bounds["south"], bounds["north"], 500)
lons = np.linspace(bounds["west"],  bounds["east"],  500)
# cell (i,j) is at lat=lats[i], lon=lons[j]
```

### Step 3 — Pull forbidden zones from OpenStreetMap
Use `osmnx.geometries_from_place` to pull the following geometry types for "Austin, Texas, USA":

| Tag | Value |
|-----|-------|
| amenity | hospital |
| amenity | school |
| leisure | park |
| natural | water |
| aeroway | aerodrome |

For each geometry, zero out all grid cells whose centroid falls within that geometry's polygon (use shapely `.contains()` or `.intersects()`). Build the `forbidden_mask` as a 500x500 float32 array initialized to 1.0, with forbidden cells set to 0.0.

If any individual OSM query fails, print a warning and skip that category — do not crash.

### Step 4 — Build the demand heatmap
The demand heatmap is a proxy for electricity load — where people and industry are concentrated.

**Primary method:** Use the `cenpy` library to pull Austin census tract population data, rasterize population density onto the 500x500 grid.

**Fallback (if cenpy fails or is slow):** Hardcode approximate demand by building a synthetic heatmap using known Austin landmarks as demand centers:

```python
demand_centers = [
    {"lat": 30.2672, "lon": -97.7431, "weight": 1.0},   # Downtown Austin
    {"lat": 30.3977, "lon": -97.7263, "weight": 0.7},   # North Austin / Domain
    {"lat": 30.2200, "lon": -97.7900, "weight": 0.6},   # South Congress
    {"lat": 30.2850, "lon": -97.6600, "weight": 0.5},   # East Austin
    {"lat": 30.3500, "lon": -97.7800, "weight": 0.6},   # North Loop
    {"lat": 30.2100, "lon": -97.8400, "weight": 0.4},   # Southwest Austin
]
# For each center, add a gaussian blob to the heatmap
# scipy.ndimage.gaussian_filter on the result, sigma=10
```

Normalize the final heatmap to [0, 1].

### Step 5 — Download existing substations
Fetch Austin-area substation locations from HIFLD open data:

```
GET https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/Electric_Substations/FeatureServer/0/query?where=STATE%3D%27TX%27&outFields=*&f=geojson&resultRecordCount=2000
```

Filter to only keep substations within the Austin bounding box. Save as `data/existing_substations.geojson`. If the request fails, create a minimal fallback geojson with 3 hardcoded Austin substation locations:

```python
fallback_substations = [
    {"lat": 30.2672, "lon": -97.7431, "name": "Downtown"},
    {"lat": 30.3977, "lon": -97.7263, "name": "North"},
    {"lat": 30.2100, "lon": -97.8200, "name": "Southwest"},
]
```

### Step 6 — Save all outputs
```python
np.save("data/forbidden_mask.npy", forbidden_mask.astype(np.float32))
np.save("data/demand_heatmap.npy", demand_heatmap.astype(np.float32))
# save city_bounds.json and existing_substations.geojson as described above
```

---

## Requirements

- Libraries: `osmnx`, `geopandas`, `numpy`, `scipy`, `shapely`, `requests`, `json`
- Every step must have a try/except with a fallback — do not let one failure crash the whole script
- Print progress at each step: `[1/6] Fetching Austin boundary...`, `[2/6] Building candidate grid...` etc.
- Create the `data/` directory if it doesn't exist
- Add a `if __name__ == "__main__":` block so it runs standalone
- Target runtime under 3 minutes

---

## Output validation

At the end of the script, print a summary:
```
✓ forbidden_mask.npy    shape=(500, 500)  placeable_cells=183,421 / 250,000
✓ demand_heatmap.npy    shape=(500, 500)  min=0.00  max=1.00  mean=0.34
✓ existing_substations  count=14
✓ city_bounds.json      south=30.098 north=30.515 west=-97.938 east=-97.561
```

If any file failed to generate, print `✗` and the error so the team knows what to fix before running the optimizer.