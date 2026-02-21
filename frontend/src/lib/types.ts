export interface Bounds {
  south: number;
  north: number;
  west: number;
  east: number;
}

export type FeasibilityVerdict = "HIGH" | "MEDIUM" | "LOW" | "UNKNOWN";

export interface FeasibilityAssessment {
  land_use?: string;
  zoning_assessment?: string;
  environmental_flags?: string[];
  community_sensitivity?: string;
  grid_proximity?: string;
  feasibility: FeasibilityVerdict;
  reasoning?: string;
}

/** Normalized candidate — handles gpu-optimization, graph, and sitereliability schemas */
export interface Candidate {
  rank: number;
  lat: number;
  lon: number;
  composite_score: number;
  // Individual objective scores (gpu-optimization / sitereliability)
  load_relief_score?: number;
  loss_reduction_score?: number;
  sustainability_score?: number;
  redundancy_score?: number;
  // Coverage metrics (gpu-optimization)
  coverage_3km_pct?: number;
  coverage_5km_pct?: number;
  coverage_10km_pct?: number;
  nearest_existing_km?: number;
  // Stability analysis
  stability_count?: number;
  stability_pct?: number;
  // Site reliability enrichment
  id?: number;
  name?: string;
  feasibility?: FeasibilityAssessment | FeasibilityVerdict;
}

export interface HeatmapPoint {
  lat: number;
  lon: number;
  value: number;
}

export interface MapData {
  bounds: Bounds;
  candidates: Candidate[];
  heatmapPoints: HeatmapPoint[];
  maskPoints: HeatmapPoint[];
  substation: GeoJSON.FeatureCollection;
  loading: boolean;
  error: string | null;
}

export const FEASIBILITY_COLORS: Record<FeasibilityVerdict, string> = {
  HIGH:    "#00e676",
  MEDIUM:  "#ffc107",
  LOW:     "#ff1744",
  UNKNOWN: "#00d4ff",
};

export const FEASIBILITY_COLORS_RGB: Record<FeasibilityVerdict, [number, number, number]> = {
  HIGH:    [0,   230, 118],
  MEDIUM:  [255, 193,   7],
  LOW:     [255,  23,  68],
  UNKNOWN: [0,  212, 255],
};

export function getFeasibilityVerdict(c: Candidate): FeasibilityVerdict {
  if (!c.feasibility) return "UNKNOWN";
  if (typeof c.feasibility === "string") return c.feasibility as FeasibilityVerdict;
  return (c.feasibility as FeasibilityAssessment).feasibility ?? "UNKNOWN";
}

export function getFeasibilityAssessment(c: Candidate): FeasibilityAssessment | null {
  if (!c.feasibility || typeof c.feasibility === "string") return null;
  return c.feasibility as FeasibilityAssessment;
}
