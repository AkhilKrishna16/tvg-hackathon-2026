/**
 * Server-side data loading utilities.
 * All functions run in Node.js (API routes / Server Components only).
 */

import fs   from "fs";
import path from "path";
import type { Candidate, Bounds, HeatmapPoint } from "./types";
import { parseNpy, gridToLatLon, reservoirSample } from "./npy";

// ── Path roots ────────────────────────────────────────────────────────────────
const FRONTEND_ROOT = process.cwd();                               // frontend/
const REPO_ROOT     = path.join(FRONTEND_ROOT, "..");              // tvg-hackathon-2026/

// ── Bounds ────────────────────────────────────────────────────────────────────
const FALLBACK_BOUNDS: Bounds = {
  south: 30.0985133, north: 30.5166255,
  west:  -97.9367663, east: -97.5605288,
};

export function loadBounds(): Bounds {
  const file = path.join(FRONTEND_ROOT, "data", "city_bounds.json");
  return fs.existsSync(file)
    ? (JSON.parse(fs.readFileSync(file, "utf-8")) as Bounds)
    : FALLBACK_BOUNDS;
}

// ── Substations ───────────────────────────────────────────────────────────────
export function loadSubstationsGeoJSON(): GeoJSON.FeatureCollection {
  const file = path.join(FRONTEND_ROOT, "data", "existing_substations.geojson");
  return fs.existsSync(file)
    ? JSON.parse(fs.readFileSync(file, "utf-8"))
    : { type: "FeatureCollection", features: [] };
}

// ── Candidates ────────────────────────────────────────────────────────────────
function normalizeCandidates(raw: Record<string, unknown>[]): Candidate[] {
  return raw.map((c, i) => ({
    ...(c as unknown as Candidate),
    rank:            (c.rank as number)            ?? (c.id as number)  ?? i + 1,
    composite_score: (c.composite_score as number) ?? (c.score as number) ?? 0,
  }));
}

export function loadCandidates(): Candidate[] {
  // Priority: most-enriched source first
  const sources = [
    path.join(REPO_ROOT,     "sitereliability", "results", "top_candidates.json"),
    path.join(FRONTEND_ROOT, "results",                    "top_candidates.json"),
    path.join(REPO_ROOT,     "gpu-optimization", "results", "top_candidates.json"),
    path.join(REPO_ROOT,     "graph",             "results", "top_candidates.json"),
  ];

  for (const src of sources) {
    if (!fs.existsSync(src)) continue;
    try {
      const raw  = JSON.parse(fs.readFileSync(src, "utf-8"));
      const arr  = Array.isArray(raw) ? raw : (raw.candidates ?? []) as Record<string, unknown>[];
      if (arr.length > 0) return normalizeCandidates(arr);
    } catch {
      // try next source
    }
  }
  return [];
}

// ── Demand heatmap ────────────────────────────────────────────────────────────
export function sampleHeatmap(n = 2_000): HeatmapPoint[] {
  const file = path.join(FRONTEND_ROOT, "data", "demand_heatmap.npy");
  if (!fs.existsSync(file)) return [];

  const { data, shape } = parseNpy(fs.readFileSync(file));
  const [, cols]        = shape;
  const bounds          = loadBounds();

  const indices: number[] = [];
  for (let i = 0; i < data.length; i++) {
    if (data[i] > 0.01) indices.push(i);
  }

  return reservoirSample(indices, n, 42).map((idx) => {
    const row = Math.floor(idx / cols);
    const col = idx % cols;
    const { lat, lon } = gridToLatLon(row, col, bounds);
    return { lat, lon, value: data[idx] };
  });
}

// ── Forbidden mask ────────────────────────────────────────────────────────────
export function sampleMask(n = 500): HeatmapPoint[] {
  const file = path.join(FRONTEND_ROOT, "data", "forbidden_mask.npy");
  if (!fs.existsSync(file)) return [];

  const { data, shape } = parseNpy(fs.readFileSync(file));
  const [, cols]        = shape;
  const bounds          = loadBounds();

  const indices: number[] = [];
  for (let i = 0; i < data.length; i++) {
    if (data[i] < 0.5) indices.push(i);
  }

  return reservoirSample(indices, n, 43).map((idx) => {
    const row = Math.floor(idx / cols);
    const col = idx % cols;
    const { lat, lon } = gridToLatLon(row, col, bounds);
    return { lat, lon, value: 1 };
  });
}
