"use client";

import { useState, useEffect } from "react";
import type { MapData, HeatmapPoint, Candidate, Bounds } from "@/lib/types";

const EMPTY_GEOJSON: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

const DEFAULT_BOUNDS: Bounds = {
  south: 30.0985, north: 30.5166, west: -97.9368, east: -97.5605,
};

async function fetchJSON<T>(url: string): Promise<T> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} → ${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export function useMapData(): MapData {
  const [state, setState] = useState<MapData>({
    bounds:        DEFAULT_BOUNDS,
    candidates:    [],
    heatmapPoints: [],
    maskPoints:    [],
    substation:    EMPTY_GEOJSON,
    loading:       true,
    error:         null,
  });

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        const [bounds, candidates, heatmapPoints, maskPoints, substation] = await Promise.all([
          fetchJSON<Bounds>("/api/bounds"),
          fetchJSON<Candidate[]>("/api/candidates"),
          fetchJSON<HeatmapPoint[]>("/api/heatmap"),
          fetchJSON<HeatmapPoint[]>("/api/mask"),
          fetchJSON<GeoJSON.FeatureCollection>("/api/substations"),
        ]);

        if (!cancelled) {
          setState({ bounds, candidates, heatmapPoints, maskPoints, substation, loading: false, error: null });
        }
      } catch (err) {
        if (!cancelled) {
          setState((prev) => ({ ...prev, loading: false, error: String(err) }));
        }
      }
    })();

    return () => { cancelled = true; };
  }, []);

  return state;
}
