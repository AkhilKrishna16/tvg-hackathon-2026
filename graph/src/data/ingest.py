"""
Data ingestion layer for the substation placement optimizer.
Pulls real data from OSM and HIFLD; every step has a fallback so nothing crashes.

Usage:
    python src/data/ingest.py                          # defaults to Austin, TX
    python src/data/ingest.py "Houston, Texas, USA"
    python src/data/ingest.py --city "Chicago, Illinois, USA" --grid 500
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import requests
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore")

# Force UTF-8 output on Windows so Unicode chars don't crash
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Shapely 2.x / 1.x compatibility ─────────────────────────────────────────
try:
    from shapely import contains_xy as _shapely_contains_xy
    def points_in_geom(geom, lons, lats):
        return _shapely_contains_xy(geom, lons, lats)
except ImportError:
    from shapely.vectorized import contains as _shapely_contains
    def points_in_geom(geom, lons, lats):
        return _shapely_contains(geom, lons, lats)

# ── Fallback data ─────────────────────────────────────────────────────────────
CITY_FALLBACKS = {
    "Austin, Texas, USA": {
        "bounds": {"south": 30.098, "north": 30.515, "west": -97.938, "east": -97.561},
        "state": "TX",
        "demand_centers": [
            {"lat": 30.2672, "lon": -97.7431, "weight": 1.0},
            {"lat": 30.3977, "lon": -97.7263, "weight": 0.7},
            {"lat": 30.2200, "lon": -97.7900, "weight": 0.6},
            {"lat": 30.2850, "lon": -97.6600, "weight": 0.5},
            {"lat": 30.3500, "lon": -97.7800, "weight": 0.6},
            {"lat": 30.2100, "lon": -97.8400, "weight": 0.4},
        ],
        "substations": [
            {"lat": 30.2672, "lon": -97.7431, "name": "Downtown"},
            {"lat": 30.3977, "lon": -97.7263, "name": "North"},
            {"lat": 30.2100, "lon": -97.8200, "name": "Southwest"},
        ],
    }
}

FORBIDDEN_TAGS = [
    {"amenity": "hospital"},
    {"amenity": "school"},
    {"leisure": "park"},
    {"natural": "water"},
    {"aeroway": "aerodrome"},
    {"landuse": "industrial"},
    {"landuse": "military"},
    {"boundary": "protected_area"},
]

HIFLD_URL = (
    "https://services1.arcgis.com/Hp6G80Pky0om7QvQ/arcgis/rest/services/"
    "Electric_Substations/FeatureServer/0/query"
    "?where=1%3D1&outFields=NAME,CITY,STATE,LATITUDE,LONGITUDE,STATUS&"
    "f=geojson&resultRecordCount=5000"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fallback_for(city: str) -> dict:
    """Return the closest available fallback data for a city."""
    key = next((k for k in CITY_FALLBACKS if k.lower() in city.lower()), None)
    return CITY_FALLBACKS.get(key, CITY_FALLBACKS["Austin, Texas, USA"])


def _synthetic_demand(bounds: dict, lats: np.ndarray, lons: np.ndarray,
                      centers: list, grid: int) -> np.ndarray:
    demand = np.zeros((grid, grid), dtype=np.float64)
    lat_range = bounds["north"] - bounds["south"]
    lon_range = bounds["east"]  - bounds["west"]
    for c in centers:
        ci = int(np.clip((c["lat"] - bounds["south"]) / lat_range * (grid - 1), 0, grid - 1))
        cj = int(np.clip((c["lon"] - bounds["west"])  / lon_range * (grid - 1), 0, grid - 1))
        demand[ci, cj] += c["weight"]
    return gaussian_filter(demand, sigma=grid // 50)


def _extract_state_from_city(city: str) -> str | None:
    """Best-effort extraction of US state abbreviation from a city string."""
    import re
    STATE_MAP = {
        "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
        "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
        "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
        "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
        "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
        "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
        "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
        "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
        "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
        "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI",
        "south carolina": "SC", "south dakota": "SD", "tennessee": "TN",
        "texas": "TX", "utah": "UT", "vermont": "VT", "virginia": "VA",
        "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    }
    city_lower = city.lower()
    for name, abbr in STATE_MAP.items():
        if name in city_lower:
            return abbr
    # Try 2-letter abbreviation directly
    m = re.search(r'\b([A-Z]{2})\b', city)
    if m and m.group(1) in STATE_MAP.values():
        return m.group(1)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Steps
# ─────────────────────────────────────────────────────────────────────────────

def step1_get_bounds(city: str) -> dict:
    print(f"[1/6] Fetching boundary for '{city}'...")
    try:
        import osmnx as ox
        gdf = ox.geocode_to_gdf(city)
        b = gdf.geometry.iloc[0].bounds  # (minx, miny, maxx, maxy)
        bounds = {"south": b[1], "north": b[3], "west": b[0], "east": b[2]}
        print(f"      Bounds from OSMnx: S={bounds['south']:.4f} N={bounds['north']:.4f} "
              f"W={bounds['west']:.4f} E={bounds['east']:.4f}")
        return bounds
    except Exception as e:
        print(f"      WARNING: osmnx failed ({e}), using hardcoded fallback.")
        return _fallback_for(city)["bounds"]


def step2_build_grid(bounds: dict, grid: int) -> tuple[np.ndarray, np.ndarray]:
    print(f"[2/6] Building {grid}x{grid} candidate grid...")
    lats = np.linspace(bounds["south"], bounds["north"], grid)
    lons = np.linspace(bounds["west"],  bounds["east"],  grid)
    print(f"      Cell size ≈ {(bounds['north']-bounds['south'])/grid*111320:.0f}m lat × "
          f"{(bounds['east']-bounds['west'])/grid*111320*np.cos(np.radians((bounds['north']+bounds['south'])/2)):.0f}m lon")
    return lats, lons


def step3_build_forbidden_mask(city: str, bounds: dict,
                                lats: np.ndarray, lons: np.ndarray,
                                grid: int) -> np.ndarray:
    print("[3/6] Building forbidden mask from OSM...")
    forbidden_mask = np.ones((grid, grid), dtype=np.float32)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_lons = lon_grid.ravel().astype(np.float64)
    flat_lats = lat_grid.ravel().astype(np.float64)

    try:
        import osmnx as ox
        from shapely.ops import unary_union

        total_forbidden = 0
        for tag_dict in FORBIDDEN_TAGS:
            tag_label = list(tag_dict.items())[0]
            try:
                gdf = ox.features_from_place(city, tags=tag_dict)
                if gdf.empty:
                    continue
                # Project to WGS84 and union
                geom_union = unary_union(gdf.geometry.values)
                mask_flat = points_in_geom(geom_union, flat_lons, flat_lats)
                count = int(mask_flat.sum())
                if count:
                    forbidden_mask[mask_flat.reshape(grid, grid)] = 0.0
                    total_forbidden += count
                    print(f"      {tag_label}: {count:,} cells forbidden")
            except Exception as e:
                print(f"      WARNING: skipping {tag_label} — {e}")

        print(f"      Total forbidden cells: {total_forbidden:,} / {grid*grid:,}")

    except ImportError:
        print("      WARNING: osmnx not importable — mask left as all-valid.")

    return forbidden_mask


def step4_build_demand_heatmap(city: str, bounds: dict,
                                lats: np.ndarray, lons: np.ndarray,
                                grid: int) -> np.ndarray:
    print("[4/6] Building demand heatmap...")
    demand = None

    # ── Primary: OSM population/commercial land use as proxy for demand ───────
    try:
        import osmnx as ox
        from shapely.ops import unary_union

        demand = np.zeros((grid, grid), dtype=np.float64)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        flat_lons = lon_grid.ravel().astype(np.float64)
        flat_lats = lat_grid.ravel().astype(np.float64)

        # High-demand tags: residential, commercial, retail, university
        demand_tags = [
            ({"landuse": "residential"}, 0.6),
            ({"landuse": "commercial"},  1.0),
            ({"landuse": "retail"},      1.0),
            ({"amenity": "university"},  0.8),
            ({"landuse": "industrial"},  0.9),
        ]
        got_any = False
        for tag_dict, weight in demand_tags:
            try:
                gdf = ox.features_from_place(city, tags=tag_dict)
                if gdf.empty:
                    continue
                geom_union = unary_union(gdf.geometry.values)
                inside = points_in_geom(geom_union, flat_lons, flat_lats).reshape(grid, grid)
                demand[inside] += weight
                got_any = True
            except Exception:
                pass

        if got_any:
            demand = gaussian_filter(demand, sigma=grid // 40)
            print("      Demand built from OSM land-use data.")
        else:
            demand = None

    except Exception as e:
        print(f"      WARNING: OSM land-use demand failed ({e}).")
        demand = None

    # ── Fallback: synthetic gaussian blobs ────────────────────────────────────
    if demand is None or demand.max() == 0:
        centers = _fallback_for(city)["demand_centers"]
        demand = _synthetic_demand(bounds, lats, lons, centers, grid)
        print("      Demand built from synthetic landmark gaussians (fallback).")

    # Normalise to [0, 1]
    d_min, d_max = demand.min(), demand.max()
    if d_max > d_min:
        demand = (demand - d_min) / (d_max - d_min)
    else:
        demand[:] = 0.5

    demand = demand.astype(np.float32)
    print(f"      Demand heatmap: min={demand.min():.3f} max={demand.max():.3f} mean={demand.mean():.3f}")
    return demand


def step5_get_substations(city: str, bounds: dict) -> dict:
    print("[5/6] Fetching existing substations...")

    features = []

    # ── Primary: HIFLD open data ──────────────────────────────────────────────
    try:
        resp = requests.get(HIFLD_URL, timeout=60)
        resp.raise_for_status()
        raw = resp.json()
        for feat in raw.get("features", []):
            try:
                coords = feat["geometry"]["coordinates"]
                lon, lat = float(coords[0]), float(coords[1])
                if (bounds["south"] <= lat <= bounds["north"] and
                        bounds["west"]  <= lon <= bounds["east"]):
                    features.append(feat)
            except Exception:
                pass
        print(f"      HIFLD: {len(features)} substations in bounding box.")
    except Exception as e:
        print(f"      WARNING: HIFLD failed ({e}).")

    # ── Secondary: OSM power=substation ──────────────────────────────────────
    if not features:
        try:
            import osmnx as ox
            gdf = ox.features_from_place(city, tags={"power": "substation"})
            if not gdf.empty:
                for _, row in gdf.iterrows():
                    try:
                        centroid = row.geometry.centroid
                        feat = {
                            "type": "Feature",
                            "geometry": {"type": "Point",
                                         "coordinates": [centroid.x, centroid.y]},
                            "properties": {"NAME": row.get("name", "OSM substation"),
                                           "SOURCE": "OSM"},
                        }
                        features.append(feat)
                    except Exception:
                        pass
                print(f"      OSM power=substation: {len(features)} found.")
        except Exception as e:
            print(f"      WARNING: OSM substations failed ({e}).")

    # ── Fallback: hardcoded ───────────────────────────────────────────────────
    if not features:
        print("      Using hardcoded fallback substations.")
        for s in _fallback_for(city)["substations"]:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [s["lon"], s["lat"]]},
                "properties": {"NAME": s["name"], "SOURCE": "fallback"},
            })

    return {"type": "FeatureCollection", "features": features}


def step6_save(output_dir: str, bounds: dict, forbidden_mask: np.ndarray,
               demand_heatmap: np.ndarray, substations: dict, city: str) -> dict:
    print("[6/6] Saving outputs...")
    os.makedirs(output_dir, exist_ok=True)
    saved = {}
    errors = {}

    def _save(name, fn):
        try:
            fn()
            saved[name] = True
        except Exception as e:
            errors[name] = str(e)

    _save("forbidden_mask",
          lambda: np.save(os.path.join(output_dir, "forbidden_mask.npy"),
                          forbidden_mask.astype(np.float32)))
    _save("demand_heatmap",
          lambda: np.save(os.path.join(output_dir, "demand_heatmap.npy"),
                          demand_heatmap.astype(np.float32)))
    _save("existing_substations",
          lambda: open(os.path.join(output_dir, "existing_substations.geojson"), "w")
                  .write(json.dumps(substations, indent=2)))
    _save("city_bounds",
          lambda: open(os.path.join(output_dir, "city_bounds.json"), "w")
                  .write(json.dumps(bounds, indent=2)))
    # Also save city name so the scoring/viz steps know which city
    _save("city_meta",
          lambda: open(os.path.join(output_dir, "city_meta.json"), "w")
                  .write(json.dumps({"city": city}, indent=2)))

    return saved, errors


def print_summary(bounds, forbidden_mask, demand_heatmap, substations, saved, errors):
    print("\n─── Output Validation ───────────────────────────────────────────")
    grid = forbidden_mask.shape[0]

    def ok(name, msg):
        sym = "✓" if name in saved else "✗"
        print(f"  {sym} {msg}")

    placeable = int((forbidden_mask == 1.0).sum())
    ok("forbidden_mask",
       f"forbidden_mask.npy    shape={forbidden_mask.shape}  "
       f"placeable_cells={placeable:,} / {grid*grid:,}")

    ok("demand_heatmap",
       f"demand_heatmap.npy    shape={demand_heatmap.shape}  "
       f"min={demand_heatmap.min():.2f}  max={demand_heatmap.max():.2f}  "
       f"mean={demand_heatmap.mean():.2f}")

    count = len(substations["features"])
    ok("existing_substations",
       f"existing_substations  count={count}")

    b = bounds
    ok("city_bounds",
       f"city_bounds.json      S={b['south']:.4f} N={b['north']:.4f} "
       f"W={b['west']:.4f} E={b['east']:.4f}")

    for name, err in errors.items():
        print(f"  ✗ {name}: {err}")
    print("─────────────────────────────────────────────────────────────────")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_ingest(city: str = "Austin, Texas, USA", grid: int = 500,
               output_dir: str = None) -> str:
    """Run full ingestion for *city* and return the output directory path."""
    if output_dir is None:
        # data/ next to the graph/ root
        root = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        output_dir = os.path.abspath(root)

    print(f"\n{'='*60}")
    print(f"  Substation Placement — Data Ingestion")
    print(f"  City : {city}")
    print(f"  Grid : {grid}×{grid} = {grid*grid:,} candidate locations")
    print(f"  Out  : {output_dir}")
    print(f"{'='*60}\n")

    bounds         = step1_get_bounds(city)
    lats, lons     = step2_build_grid(bounds, grid)
    forbidden_mask = step3_build_forbidden_mask(city, bounds, lats, lons, grid)
    demand_heatmap = step4_build_demand_heatmap(city, bounds, lats, lons, grid)
    substations    = step5_get_substations(city, bounds)
    saved, errors  = step6_save(output_dir, bounds, forbidden_mask,
                                demand_heatmap, substations, city)
    print_summary(bounds, forbidden_mask, demand_heatmap, substations, saved, errors)
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest placement data for a city.")
    parser.add_argument("city", nargs="?", default="Austin, Texas, USA",
                        help='City string e.g. "Houston, Texas, USA"')
    parser.add_argument("--grid", type=int, default=500,
                        help="Grid resolution (default 500 → 500×500)")
    parser.add_argument("--out", default=None,
                        help="Output directory (default: ../../data/)")
    args = parser.parse_args()
    run_ingest(city=args.city, grid=args.grid, output_dir=args.out)
