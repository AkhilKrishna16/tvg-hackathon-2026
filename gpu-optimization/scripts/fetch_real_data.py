"""
fetch_real_data.py — Fetch real geographic data from public APIs for any city.

Replaces synthetic .npy files with data derived from OpenStreetMap (via Overpass
API) and optionally the US Census Bureau (population/demand).

Outputs:
    data/city_bounds.json           — bounding box (from geocoding or CLI arg)
    data/forbidden_mask.npy         — real land-use forbidden areas from OSM
    data/demand_heatmap.npy         — building/population density from OSM or Census
    data/existing_substations.geojson — real power substations from OSM

Usage:
    # By city name (geocoded via Nominatim)
    python scripts/fetch_real_data.py --city "Austin, TX"
    python scripts/fetch_real_data.py --city "Denver, CO"
    python scripts/fetch_real_data.py --city "Portland, OR" --census-key YOUR_KEY

    # By explicit bounding box (south,north,west,east)
    python scripts/fetch_real_data.py --bounds 30.098,30.516,-97.928,-97.522

    # Optional flags
    # --census-key KEY     US Census API key (improves demand data for US cities)
    # --output-dir data/   Output directory (default: data/)
    # --grid-size 500      Grid resolution (default: 500)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from scipy.ndimage import gaussian_filter
from matplotlib.path import Path as MplPath

_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Nominatim geocoding
# ---------------------------------------------------------------------------

_NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_HEADERS = {
    "User-Agent": "gpu-substation-optimizer/1.0 (hackathon project)",
    "Accept-Language": "en",
}


def geocode_city(city_name: str) -> dict:
    """
    Geocode a city name to a bounding box using OpenStreetMap Nominatim.

    Returns dict with keys: south, north, west, east, _note.
    Raises ValueError if the city cannot be found.
    """
    params = {"q": city_name, "format": "json", "limit": 1}
    resp = requests.get(
        _NOMINATIM_URL,
        params=params,
        headers=_NOMINATIM_HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json()

    if not results:
        raise ValueError(f"City not found: {city_name!r}")

    result = results[0]
    # Nominatim boundingbox: [south, north, west, east] (all as strings)
    bb = result["boundingbox"]
    south, north, west, east = (float(x) for x in bb)

    display_name = result.get("display_name", city_name)
    return {
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "_note": f"Geocoded from: {display_name}",
    }


# ---------------------------------------------------------------------------
# Overpass API
# ---------------------------------------------------------------------------

_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
_OVERPASS_HEADERS = {
    "User-Agent": "gpu-substation-optimizer/1.0 (hackathon project)",
}


def fetch_overpass(query: str) -> dict:
    """
    POST an Overpass QL query and return the parsed JSON response.

    Retries once on failure with a 5-second delay.
    """
    for attempt in range(2):
        try:
            resp = requests.post(
                _OVERPASS_URL,
                data={"data": query},
                headers=_OVERPASS_HEADERS,
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            if attempt == 0:
                time.sleep(5)
            else:
                raise RuntimeError(
                    f"Overpass API request failed after 2 attempts: {exc}"
                ) from exc
    # unreachable, but satisfies type checkers
    raise RuntimeError("Overpass query failed")


# ---------------------------------------------------------------------------
# OSM geometry helpers
# ---------------------------------------------------------------------------

def osm_elements_to_polygons(elements: list) -> list[np.ndarray]:
    """
    Convert OSM way and relation elements to arrays of (lat, lon) coordinate pairs.

    Only closed ways (first node == last node) and multipolygon relations
    are converted to polygons. Returns a list of Nx2 numpy arrays.
    """
    # Build a lookup: node_id -> (lat, lon)
    node_coords: dict[int, tuple[float, float]] = {}
    for el in elements:
        if el.get("type") == "node":
            node_coords[el["id"]] = (el["lat"], el["lon"])

    # Build way geometry from node references
    way_coords: dict[int, list[tuple[float, float]]] = {}
    for el in elements:
        if el.get("type") != "way":
            continue
        nodes = el.get("nodes", [])
        pts = [node_coords[nid] for nid in nodes if nid in node_coords]
        if len(pts) >= 3:
            way_coords[el["id"]] = pts

    polygons: list[np.ndarray] = []

    for el in elements:
        etype = el.get("type")

        if etype == "way":
            pts = way_coords.get(el["id"])
            if pts is None:
                continue
            # Must be a closed way to be a polygon
            if pts[0] == pts[-1] and len(pts) >= 4:
                polygons.append(np.array(pts, dtype=np.float64))

        elif etype == "relation":
            tags = el.get("tags", {})
            if tags.get("type") not in ("multipolygon", "boundary"):
                continue
            # Collect outer member ways
            outer_ways: list[list[tuple[float, float]]] = []
            for member in el.get("members", []):
                if member.get("type") == "way" and member.get("role") in ("outer", ""):
                    wid = member["ref"]
                    pts = way_coords.get(wid)
                    if pts:
                        outer_ways.append(pts)
            for pts in outer_ways:
                if len(pts) >= 3:
                    polygons.append(np.array(pts, dtype=np.float64))

    return polygons


def rasterize_polygons(
    polygons: list[np.ndarray],
    bounds: dict,
    shape: tuple[int, int],
) -> np.ndarray:
    """
    Rasterize a list of (lat, lon) polygons onto a grid of the given shape.

    Grid convention:
        row 0   → northernmost latitude
        col 0   → westernmost longitude

    Uses matplotlib.path.Path.contains_points for vectorized point-in-polygon.
    All grid points (~250K for a 500×500 grid) are tested in a single batch.

    Returns a boolean mask (True = inside at least one polygon).
    """
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)

    if not polygons:
        return mask

    north, south = bounds["north"], bounds["south"]
    west, east = bounds["west"], bounds["east"]

    # Build flat arrays of all grid lat/lon coordinates
    lat_lin = np.linspace(north, south, rows)
    lon_lin = np.linspace(west, east, cols)
    # Grid points in (lon, lat) order for matplotlib path
    lon_grid, lat_grid = np.meshgrid(lon_lin, lat_lin)  # both (rows, cols)
    grid_pts = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])  # (N, 2)

    for poly in polygons:
        if len(poly) < 3:
            continue
        # poly is (M, 2) as (lat, lon) — need (lon, lat) for matplotlib
        path_pts = poly[:, ::-1]  # swap to (lon, lat)
        mpl_path = MplPath(path_pts)
        inside = mpl_path.contains_points(grid_pts)
        mask |= inside.reshape(shape)

    return mask


def rasterize_linestrings(
    lines: list[list[tuple[float, float]]],
    bounds: dict,
    shape: tuple[int, int],
    buffer_cells: int = 2,
) -> np.ndarray:
    """
    Rasterize highway linestrings as buffered polygons (thin rectangles).

    Each segment (lat0,lon0)→(lat1,lon1) is converted to a 4-corner parallelogram
    of width `buffer_cells` grid cells, then rasterized via rasterize_polygons.
    """
    rows, cols = shape
    mask = np.zeros(shape, dtype=bool)

    if not lines:
        return mask

    north, south = bounds["north"], bounds["south"]
    west, east = bounds["west"], bounds["east"]

    # Degrees-per-cell for buffering
    dlat_per_cell = abs(north - south) / (rows - 1)
    dlon_per_cell = abs(east - west) / (cols - 1)
    buf_lat = buffer_cells * dlat_per_cell
    buf_lon = buffer_cells * dlon_per_cell

    buffered_polys: list[np.ndarray] = []

    for line in lines:
        if len(line) < 2:
            continue
        for k in range(len(line) - 1):
            lat0, lon0 = line[k]
            lat1, lon1 = line[k + 1]
            dlat = lat1 - lat0
            dlon = lon1 - lon0
            seg_len = np.sqrt(dlat ** 2 + dlon ** 2)
            if seg_len < 1e-12:
                continue
            # Perpendicular unit vector (normal in lat/lon space)
            nx_lat = -dlon / seg_len * (buf_lat / buf_lon if buf_lon > 0 else 1.0)
            nx_lon = dlat / seg_len
            # Normalise perpendicular direction
            n_len = np.sqrt(nx_lat ** 2 + nx_lon ** 2)
            if n_len < 1e-12:
                continue
            nx_lat /= n_len
            nx_lon /= n_len

            half_buf_lat = buf_lat / 2
            half_buf_lon = buf_lon / 2

            corners = np.array([
                [lat0 + nx_lat * half_buf_lat, lon0 + nx_lon * half_buf_lon],
                [lat1 + nx_lat * half_buf_lat, lon1 + nx_lon * half_buf_lon],
                [lat1 - nx_lat * half_buf_lat, lon1 - nx_lon * half_buf_lon],
                [lat0 - nx_lat * half_buf_lat, lon0 - nx_lon * half_buf_lon],
                [lat0 + nx_lat * half_buf_lat, lon0 + nx_lon * half_buf_lon],  # close
            ], dtype=np.float64)
            buffered_polys.append(corners)

    if buffered_polys:
        mask |= rasterize_polygons(buffered_polys, bounds, shape)

    return mask


# ---------------------------------------------------------------------------
# Forbidden mask (OSM land use)
# ---------------------------------------------------------------------------

def build_forbidden_mask(bounds: dict, grid_size: int = 500) -> np.ndarray:
    """
    Query OSM for forbidden land-use categories and rasterize them.

    Blocked (0) categories:
      • Water bodies: natural=water, waterway=riverbank, natural=wetland
      • Parks/greenbelts: leisure=park, leisure=nature_reserve, boundary=protected_area
      • Airports: aeroway=aerodrome, aeroway=runway
      • Major roads (buffered): highway=motorway, highway=trunk

    Returns a (grid_size, grid_size) float32 array: 1=placeable, 0=blocked.
    Edges are always blocked.
    """
    shape = (grid_size, grid_size)
    south, north = bounds["south"], bounds["north"]
    west, east = bounds["west"], bounds["east"]
    bbox = f"{south},{west},{north},{east}"

    forbidden = np.zeros(shape, dtype=bool)

    # --- Water ---
    water_query = f"""
[out:json][timeout:60];
(
  way["natural"="water"]({bbox});
  way["waterway"="riverbank"]({bbox});
  way["natural"="wetland"]({bbox});
  relation["natural"="water"]({bbox});
  relation["waterway"="riverbank"]({bbox});
  relation["natural"="wetland"]({bbox});
);
out body;
>;
out skel qt;
"""
    data = fetch_overpass(water_query)
    polys = osm_elements_to_polygons(data.get("elements", []))
    forbidden |= rasterize_polygons(polys, bounds, shape)

    # --- Parks / greenbelts ---
    park_query = f"""
[out:json][timeout:60];
(
  way["leisure"="park"]({bbox});
  way["leisure"="nature_reserve"]({bbox});
  way["boundary"="protected_area"]({bbox});
  relation["leisure"="park"]({bbox});
  relation["leisure"="nature_reserve"]({bbox});
  relation["boundary"="protected_area"]({bbox});
);
out body;
>;
out skel qt;
"""
    data = fetch_overpass(park_query)
    polys = osm_elements_to_polygons(data.get("elements", []))
    forbidden |= rasterize_polygons(polys, bounds, shape)

    # --- Airports ---
    airport_query = f"""
[out:json][timeout:60];
(
  way["aeroway"="aerodrome"]({bbox});
  way["aeroway"="runway"]({bbox});
  relation["aeroway"="aerodrome"]({bbox});
);
out body;
>;
out skel qt;
"""
    data = fetch_overpass(airport_query)
    polys = osm_elements_to_polygons(data.get("elements", []))
    forbidden |= rasterize_polygons(polys, bounds, shape)

    # --- Major roads (buffered linestrings) ---
    road_query = f"""
[out:json][timeout:60];
(
  way["highway"="motorway"]({bbox});
  way["highway"="trunk"]({bbox});
);
out body;
>;
out skel qt;
"""
    data = fetch_overpass(road_query)
    elements = data.get("elements", [])
    node_coords: dict[int, tuple[float, float]] = {
        el["id"]: (el["lat"], el["lon"])
        for el in elements
        if el.get("type") == "node"
    }
    road_lines: list[list[tuple[float, float]]] = []
    for el in elements:
        if el.get("type") == "way":
            nodes = el.get("nodes", [])
            pts = [node_coords[nid] for nid in nodes if nid in node_coords]
            if len(pts) >= 2:
                road_lines.append(pts)
    forbidden |= rasterize_linestrings(road_lines, bounds, shape, buffer_cells=2)

    # Build mask: 1 = placeable, 0 = blocked
    mask = (~forbidden).astype(np.float32)

    # Always block edges (5 cells)
    mask[:5, :] = 0.0
    mask[-5:, :] = 0.0
    mask[:, :5] = 0.0
    mask[:, -5:] = 0.0

    return mask


# ---------------------------------------------------------------------------
# Demand heatmap — OSM buildings
# ---------------------------------------------------------------------------

_BUILDING_WEIGHTS = {
    "commercial": 2.5,
    "retail": 2.5,
    "office": 2.5,
    "industrial": 1.5,
    "residential": 1.0,
    "apartments": 1.0,
    "house": 1.0,
}
_DEFAULT_BUILDING_WEIGHT = 0.8


def build_demand_heatmap_osm(
    bounds: dict,
    forbidden_mask: np.ndarray,
    grid_size: int = 500,
) -> np.ndarray:
    """
    Build a demand heatmap from OSM building footprint centroids.

    Uses 'out center' to get the centroid of each building way/relation without
    transferring full geometry, which is much faster for dense urban areas.

    Building type weights:
        commercial/retail/office → 2.5
        industrial               → 1.5
        residential/apartments   → 1.0
        other                    → 0.8

    Applies scipy gaussian smoothing (σ=2 cells), zeros forbidden cells,
    normalizes to [0, 1].
    """
    shape = (grid_size, grid_size)
    south, north = bounds["south"], bounds["north"]
    west, east = bounds["west"], bounds["east"]
    bbox = f"{south},{west},{north},{east}"

    query = f"""
[out:json][timeout:90];
(
  way["building"]({bbox});
  relation["building"]({bbox});
);
out center;
"""
    data = fetch_overpass(query)
    elements = data.get("elements", [])

    demand = np.zeros(shape, dtype=np.float64)
    n_buildings = 0

    lat_lin = np.linspace(north, south, grid_size)
    lon_lin = np.linspace(west, east, grid_size)

    for el in elements:
        # Get centroid
        center = el.get("center")
        if center is None:
            # For nodes (unlikely from building query but handle it)
            if el.get("type") == "node":
                clat, clon = el.get("lat"), el.get("lon")
            else:
                continue
        else:
            clat, clon = center.get("lat"), center.get("lon")

        if clat is None or clon is None:
            continue

        # Map lat/lon to grid cell
        row = int(np.searchsorted(-lat_lin, -clat))
        col = int(np.searchsorted(lon_lin, clon))
        row = max(0, min(grid_size - 1, row))
        col = max(0, min(grid_size - 1, col))

        # Building type weight
        tags = el.get("tags", {})
        btype = tags.get("building", "yes")
        weight = _BUILDING_WEIGHTS.get(btype, _DEFAULT_BUILDING_WEIGHT)

        demand[row, col] += weight
        n_buildings += 1

    # Smooth demand
    demand = gaussian_filter(demand, sigma=2.0)

    # Zero forbidden cells
    demand *= forbidden_mask.astype(np.float64)

    # Normalize
    d_max = demand.max()
    if d_max > 0:
        demand /= d_max

    return demand.astype(np.float32), n_buildings


# ---------------------------------------------------------------------------
# Demand heatmap — US Census (optional, US only)
# ---------------------------------------------------------------------------

def _get_state_fips(lat: float, lon: float) -> Optional[str]:
    """
    Determine the US state FIPS code from a lat/lon using the Census geocoder.
    Returns None if the point is outside the US or request fails.
    """
    try:
        url = (
            "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"
            f"?x={lon}&y={lat}&benchmark=Public_AR_Current&vintage=Current_Current"
            "&layers=States&format=json"
        )
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        states = (
            data.get("result", {})
            .get("geographies", {})
            .get("States", [])
        )
        if states:
            return states[0]["STATE"]
    except Exception:
        pass
    return None


def build_demand_heatmap_census(
    bounds: dict,
    api_key: str,
    forbidden_mask: np.ndarray,
    grid_size: int = 500,
) -> tuple[np.ndarray, int]:
    """
    Build demand heatmap from US Census ACS 5-year population data.

    Falls back to OSM building density on any failure.

    Steps:
      1. Determine state FIPS from bounds centroid.
      2. Fetch ACS 5-year population for all tracts in that state.
      3. Fetch TIGER/Web tract polygons intersecting the bounding box.
      4. Rasterize each tract weighted by population density (pop / km²).
      5. Apply gaussian smoothing, zero forbidden cells, normalize.
    """
    center_lat = (bounds["north"] + bounds["south"]) / 2.0
    center_lon = (bounds["west"] + bounds["east"]) / 2.0

    state_fips = _get_state_fips(center_lat, center_lon)
    if state_fips is None:
        print("  [census] Could not determine state FIPS — falling back to OSM")
        return build_demand_heatmap_osm(bounds, forbidden_mask, grid_size)

    # --- Step 1: Fetch ACS population data ---
    acs_url = (
        f"https://api.census.gov/data/2023/acs/acs5"
        f"?get=B01003_001E,NAME&for=tract:*&in=state:{state_fips}&key={api_key}"
    )
    try:
        resp = requests.get(acs_url, timeout=30)
        resp.raise_for_status()
        rows_acs = resp.json()
    except Exception as exc:
        print(f"  [census] ACS request failed ({exc}) — falling back to OSM")
        return build_demand_heatmap_osm(bounds, forbidden_mask, grid_size)

    # Build tract GEOID → population map
    # rows_acs[0] is the header: [B01003_001E, NAME, state, county, tract]
    header = rows_acs[0]
    pop_idx = header.index("B01003_001E")
    state_idx = header.index("state")
    county_idx = header.index("county")
    tract_idx = header.index("tract")

    tract_pop: dict[str, float] = {}
    for row in rows_acs[1:]:
        try:
            pop = float(row[pop_idx])
        except (ValueError, TypeError):
            pop = 0.0
        geoid = row[state_idx] + row[county_idx] + row[tract_idx]
        tract_pop[geoid] = max(0.0, pop)

    # --- Step 2: Fetch TIGER/Web tract geometries ---
    south, north = bounds["south"], bounds["north"]
    west, east = bounds["west"], bounds["east"]
    tiger_url = (
        "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb"
        "/Tracts_Blocks/MapServer/0/query"
    )
    tiger_params = {
        "geometry": f"{west},{south},{east},{north}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "GEOID,ALAND",
        "returnGeometry": "true",
        "f": "geojson",
        "inSR": "4326",
        "outSR": "4326",
    }
    try:
        resp = requests.get(tiger_url, params=tiger_params, timeout=30)
        resp.raise_for_status()
        tiger_data = resp.json()
    except Exception as exc:
        print(f"  [census] TIGER request failed ({exc}) — falling back to OSM")
        return build_demand_heatmap_osm(bounds, forbidden_mask, grid_size)

    features = tiger_data.get("features", [])
    if not features:
        print("  [census] No tract features in bbox — falling back to OSM")
        return build_demand_heatmap_osm(bounds, forbidden_mask, grid_size)

    # --- Step 3: Rasterize tracts ---
    shape = (grid_size, grid_size)
    density_grid = np.zeros(shape, dtype=np.float64)

    for feature in features:
        geoid = feature.get("properties", {}).get("GEOID", "")
        aland = float(feature.get("properties", {}).get("ALAND") or 1)  # m²
        pop = tract_pop.get(geoid, 0.0)
        area_km2 = max(aland / 1e6, 0.001)
        pop_density = pop / area_km2  # persons / km²

        geom = feature.get("geometry", {})
        gtype = geom.get("type", "")
        coords_list = geom.get("coordinates", [])

        rings: list[np.ndarray] = []
        if gtype == "Polygon":
            for ring in coords_list:
                # GeoJSON: [lon, lat]
                pts = np.array(ring, dtype=np.float64)
                # Convert to (lat, lon)
                rings.append(pts[:, [1, 0]])
        elif gtype == "MultiPolygon":
            for polygon in coords_list:
                for ring in polygon:
                    pts = np.array(ring, dtype=np.float64)
                    rings.append(pts[:, [1, 0]])

        if not rings:
            continue

        tract_mask = rasterize_polygons(rings, bounds, shape)
        density_grid += tract_mask.astype(np.float64) * pop_density

    # --- Step 4: Smooth, zero forbidden, normalize ---
    density_grid = gaussian_filter(density_grid, sigma=2.0)
    density_grid *= forbidden_mask.astype(np.float64)
    d_max = density_grid.max()
    if d_max > 0:
        density_grid /= d_max

    return density_grid.astype(np.float32), len(features)


# ---------------------------------------------------------------------------
# Substations (OSM)
# ---------------------------------------------------------------------------

def fetch_substations_osm(bounds: dict) -> dict:
    """
    Fetch power substations from OSM within the bounding box.

    Returns a GeoJSON FeatureCollection. Uses 'out center' for ways to get
    centroids without full geometry. Extracts name and voltage_kv from tags.

    Returns an empty FeatureCollection if no substations are found.
    """
    south, north = bounds["south"], bounds["north"]
    west, east = bounds["west"], bounds["east"]
    bbox = f"{south},{west},{north},{east}"

    query = f"""
[out:json][timeout:60];
(
  node["power"="substation"]({bbox});
  way["power"="substation"]({bbox});
);
out center;
"""
    data = fetch_overpass(query)
    elements = data.get("elements", [])

    features = []
    for el in elements:
        etype = el.get("type")
        if etype == "node":
            lat = el.get("lat")
            lon = el.get("lon")
        elif etype == "way":
            center = el.get("center", {})
            lat = center.get("lat")
            lon = center.get("lon")
        else:
            continue

        if lat is None or lon is None:
            continue

        tags = el.get("tags", {})
        name = tags.get("name", tags.get("operator", ""))
        voltage_str = tags.get("voltage", "")
        voltage_kv: Optional[float] = None
        if voltage_str:
            try:
                # Voltage may be "115000" (V) or "115" (kV) or "115;230"
                vparts = voltage_str.replace(",", ";").split(";")
                v = float(vparts[0])
                # If value > 1000, assume it's in volts → convert to kV
                voltage_kv = v / 1000.0 if v > 1000 else v
            except (ValueError, IndexError):
                pass

        feature = {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": name, "voltage_kv": voltage_kv},
        }
        features.append(feature)

    return {"type": "FeatureCollection", "features": features}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_bounds(bounds_str: str) -> dict:
    """Parse 'south,north,west,east' string into a bounds dict."""
    parts = [float(x.strip()) for x in bounds_str.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--bounds must be four comma-separated floats: south,north,west,east"
        )
    return {
        "south": parts[0],
        "north": parts[1],
        "west": parts[2],
        "east": parts[3],
        "_note": "Explicit bounding box from --bounds argument",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch real geographic data from public APIs for the substation optimizer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--city",
        metavar="NAME",
        help='City name to geocode, e.g. "Austin, TX" or "Berlin, Germany"',
    )
    source.add_argument(
        "--bounds",
        metavar="S,N,W,E",
        help="Explicit bounding box as south,north,west,east decimal degrees",
    )
    parser.add_argument(
        "--census-key",
        metavar="KEY",
        default=None,
        help="US Census API key (enables higher-quality demand data for US cities)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Directory to write output files (default: <project-root>/data/)",
    )
    parser.add_argument(
        "--grid-size",
        metavar="N",
        type=int,
        default=500,
        help="Grid resolution in cells (default: 500)",
    )
    args = parser.parse_args()

    # --- Resolve output directory ---
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = _ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid_size = args.grid_size

    # --- Resolve bounding box ---
    if args.city:
        print(f'Geocoding "{args.city}" ...', end="  ", flush=True)
        t0 = time.perf_counter()
        bounds = geocode_city(args.city)
        elapsed = time.perf_counter() - t0
        print(
            f"done  ({bounds['south']:.3f}–{bounds['north']:.3f}°N, "
            f"{bounds['west']:.3f}–{bounds['east']:.3f}°E)  [{elapsed:.1f}s]"
        )
    else:
        bounds = _parse_bounds(args.bounds)
        print(
            f"Bounds: {bounds['south']:.3f}–{bounds['north']:.3f}°N, "
            f"{bounds['west']:.3f}–{bounds['east']:.3f}°E"
        )

    # Save city_bounds.json
    with (out_dir / "city_bounds.json").open("w") as f:
        json.dump(bounds, f, indent=2)

    # --- Forbidden mask ---
    print(f"Fetching forbidden mask (OSM) ...", end="  ", flush=True)
    t0 = time.perf_counter()
    mask = build_forbidden_mask(bounds, grid_size=grid_size)
    elapsed = time.perf_counter() - t0
    placeable = int(mask.sum())
    total = grid_size * grid_size
    print(
        f"done  ({placeable:,} / {total:,} cells placeable, "
        f"{100 * placeable / total:.1f}%)  [{elapsed:.1f}s]"
    )
    np.save(out_dir / "forbidden_mask.npy", mask)

    # --- Demand heatmap ---
    use_census = bool(args.census_key)
    source_label = "Census" if use_census else "OSM"
    print(f"Fetching demand heatmap ({source_label}) ...", end="  ", flush=True)
    t0 = time.perf_counter()
    if use_census:
        heatmap, n_items = build_demand_heatmap_census(
            bounds, args.census_key, mask, grid_size=grid_size
        )
        item_label = "tracts"
    else:
        heatmap, n_items = build_demand_heatmap_osm(bounds, mask, grid_size=grid_size)
        item_label = "buildings"
    elapsed = time.perf_counter() - t0
    print(
        f"done  ({n_items:,} {item_label}, "
        f"range=[{heatmap.min():.3f}, {heatmap.max():.3f}])  [{elapsed:.1f}s]"
    )
    np.save(out_dir / "demand_heatmap.npy", heatmap)

    # --- Substations ---
    print(f"Fetching substations (OSM) ...", end="  ", flush=True)
    t0 = time.perf_counter()
    substations = fetch_substations_osm(bounds)
    elapsed = time.perf_counter() - t0
    n_subs = len(substations["features"])
    print(f"done  ({n_subs} substations found)  [{elapsed:.1f}s]")
    with (out_dir / "existing_substations.geojson").open("w") as f:
        json.dump(substations, f, indent=2)

    print(f"\nData written to: {out_dir}")


if __name__ == "__main__":
    main()
