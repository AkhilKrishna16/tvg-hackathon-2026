"""
generate_mock_data.py
─────────────────────
Generates realistic synthetic data for Austin, TX so the dashboard runs
immediately without needing the real optimizer pipeline.

Run from repo root:
    python scripts/generate_mock_data.py
"""

import json
import random
from pathlib import Path

import numpy as np

ROOT    = Path(__file__).resolve().parent.parent
DATA    = ROOT / "data"
RESULTS = ROOT / "results"

DATA.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)

# ── Austin bounding box ───────────────────────────────────────────────────────
BOUNDS = {"south": 30.098, "north": 30.516, "west": -97.928, "east": -97.560}

# ── city_bounds.json ──────────────────────────────────────────────────────────
(DATA / "city_bounds.json").write_text(json.dumps(BOUNDS, indent=2))
print("✓  city_bounds.json")

# ── forbidden_mask.npy ────────────────────────────────────────────────────────
mask = np.ones((500, 500), dtype=np.float32)

# Lake Travis / Lady Bird Lake — horizontal stripe
mask[180:210, :] = 0

# Airport corridor — diagonal band
for i in range(500):
    j_start = max(0, i - 30)
    j_end   = min(500, i + 30)
    if 260 <= i <= 310:
        mask[i, j_start:j_end] = 0

# Scattered small forbidden patches (industrial zones, parks, flood plains)
for _ in range(60):
    r = RNG.integers(10, 490)
    c = RNG.integers(10, 490)
    h = RNG.integers(5, 25)
    w = RNG.integers(5, 25)
    mask[r:r+h, c:c+w] = 0

np.save(DATA / "forbidden_mask.npy", mask)
print("✓  forbidden_mask.npy")

# ── demand_heatmap.npy ────────────────────────────────────────────────────────
# Simulate demand as superposition of Gaussian blobs (urban districts)
heatmap = np.zeros((500, 500), dtype=np.float32)

demand_centers = [
    (250, 250, 80,  1.0),   # downtown core
    (200, 280, 60,  0.75),  # east Austin
    (300, 220, 55,  0.70),  # south congress
    (160, 200, 45,  0.55),  # north loop
    (350, 300, 40,  0.50),  # slaughter lane
    (120, 310, 35,  0.40),  # domain / north
    (400, 180, 30,  0.38),  # buda corridor
    (230, 340, 30,  0.35),  # manor
    (270, 170, 25,  0.30),  # bee cave
    (380, 350, 25,  0.28),  # kyle
]

grid_r, grid_c = np.mgrid[0:500, 0:500]

for (r0, c0, sigma, amp) in demand_centers:
    blob = amp * np.exp(-((grid_r - r0)**2 + (grid_c - c0)**2) / (2 * sigma**2))
    heatmap += blob

# Apply feasibility mask — no demand in forbidden zones
heatmap *= mask

# Normalize to [0, 1]
if heatmap.max() > 0:
    heatmap /= heatmap.max()

# Add subtle noise
heatmap += (RNG.random((500, 500)).astype(np.float32) * 0.03 * mask)
heatmap = np.clip(heatmap, 0, 1)

np.save(DATA / "demand_heatmap.npy", heatmap)
print("✓  demand_heatmap.npy")

# ── existing_substations.geojson ──────────────────────────────────────────────
def grid_to_latlon(row, col):
    lat = BOUNDS["north"] - (row / 500) * (BOUNDS["north"] - BOUNDS["south"])
    lon = BOUNDS["west"]  + (col / 500) * (BOUNDS["east"]  - BOUNDS["west"])
    return lat, lon

substation_cells = [
    (240, 245, "Downtown Grid Hub"),
    (170, 195, "North Austin Substation"),
    (330, 260, "South Park Substation"),
    (195, 290, "East Side Distribution"),
    (285, 215, "Barton Springs Substation"),
    (130, 320, "The Domain Substation"),
    (395, 175, "Slaughter Creek Hub"),
]

features = []
for (r, c, name) in substation_cells:
    lat, lon = grid_to_latlon(r, c)
    features.append({
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {"name": name},
    })

geojson = {"type": "FeatureCollection", "features": features}
(DATA / "existing_substations.geojson").write_text(json.dumps(geojson, indent=2))
print("✓  existing_substations.geojson")

# ── results/top_candidates.json ───────────────────────────────────────────────
feasibility_levels = ["HIGH", "HIGH", "HIGH", "MEDIUM", "MEDIUM",
                       "MEDIUM", "LOW", "LOW", "HIGH", "MEDIUM"]

candidate_locations = [
    (255, 255, 0.924, "Dense residential demand cluster 1.4 km from nearest substation. "
               "High foot traffic corridor, near planned transit hub. "
               "Land parcel owned by city — expedited permitting expected."),
    (205, 270, 0.891, "Mixed-use development zone with rapidly growing EV adoption. "
               "2.2 km service gap from existing grid. Adjacent to major arterial "
               "for easy grid tie-in."),
    (315, 235, 0.867, "High-density commercial strip with peak demand spikes. "
               "Proximity to South Congress Ave ensures high utilization. "
               "No zoning conflicts identified."),
    (155, 205, 0.812, "Growing tech corridor; anchor tenant confirmed. "
               "Moderate distance to grid (3.1 km). Minor environmental review "
               "required for creek proximity."),
    (360, 295, 0.788, "Suburban expansion zone with above-average EV adoption curve. "
               "New residential build-out will increase demand. "
               "Permitting straightforward."),
    (240, 175, 0.741, "Mid-density neighborhood with aging distribution infrastructure. "
               "Replacement would serve dual purpose. Moderate permitting risk."),
    (430, 345, 0.698, "Outer suburban ring — lower current demand but high projected "
               "growth. Land is inexpensive. Long grid tie-in required."),
    (115, 290, 0.672, "Near domain but partially in flood-adjacent zone. "
               "Elevated foundation required, adding ~15% to project cost."),
    (270, 330, 0.651, "East metro expansion — strong demand signal but "
               "right-of-way acquisition needed. 18-month permitting estimate."),
    (340, 170, 0.614, "South suburban zone; projected demand growth in 3-year window. "
               "Current utilization low. Speculative placement."),
]

candidates = []
for rank, ((r, c, score, reason), feasibility) in enumerate(
    zip(candidate_locations, feasibility_levels), start=1
):
    lat, lon = grid_to_latlon(r, c)
    # Add small jitter so points don't overlap substation markers
    lat += RNG.uniform(-0.002, 0.002)
    lon += RNG.uniform(-0.002, 0.002)
    candidates.append({
        "rank": rank,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "composite_score": round(score + RNG.uniform(-0.005, 0.005), 4),
        "feasibility": feasibility,
        "reasoning": reason,
    })

(RESULTS / "top_candidates.json").write_text(json.dumps(candidates, indent=2))
print("✓  results/top_candidates.json")

print(f"\n  All mock data written to {ROOT}")
print("  Run:  python src/viz/app.py")
