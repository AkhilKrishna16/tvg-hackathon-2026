"use client";

import { useState, useCallback, useMemo, useEffect } from "react";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer, PolygonLayer, TextLayer } from "@deck.gl/layers";
import { HeatmapLayer } from "@deck.gl/aggregation-layers";
import { FlyToInterpolator } from "@deck.gl/core";
import type { PickingInfo } from "@deck.gl/core";
import Map from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";

import type { Candidate, HeatmapPoint } from "@/lib/types";
import { FEASIBILITY_COLORS_RGB, getFeasibilityVerdict } from "@/lib/types";

const MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

const INITIAL_VIEW = {
  longitude: -97.7431,
  latitude:   30.2672,
  zoom:       10,
  pitch:       0,
  bearing:     0,
};

// ── Coordinate helpers ────────────────────────────────────────────────────────
function makeCircle(lat: number, lon: number, radiusKm: number, steps = 64): number[][] {
  const coords: number[][] = [];
  const dLat = radiusKm / 111.32;
  const dLon = radiusKm / (111.32 * Math.cos((lat * Math.PI) / 180));
  for (let i = 0; i <= steps; i++) {
    const angle = (i / steps) * Math.PI * 2;
    coords.push([lon + dLon * Math.sin(angle), lat + dLat * Math.cos(angle)]);
  }
  return coords;
}

interface CoverageRing {
  polygon:   number[][];
  fillColor: [number, number, number, number];
  lineColor: [number, number, number, number];
}

function makeCoverageRings(c: Candidate): CoverageRing[] {
  return [
    { radiusKm: 3,  fill: [0, 212, 255,  12] as [number,number,number,number], line: [0, 212, 255, 140] as [number,number,number,number] },
    { radiusKm: 5,  fill: [0, 150, 255,   8] as [number,number,number,number], line: [0, 150, 255, 110] as [number,number,number,number] },
    { radiusKm: 10, fill: [120, 0, 255,   5] as [number,number,number,number], line: [120,  0, 255,  80] as [number,number,number,number] },
  ].map(({ radiusKm, fill, line }) => ({
    polygon:   makeCircle(c.lat, c.lon, radiusKm),
    fillColor: fill,
    lineColor: line,
  }));
}

// ── Substation position from GeoJSON features ─────────────────────────────────
interface SubstationPoint {
  lon:  number;
  lat:  number;
  name: string;
}

function extractSubstations(geojson: GeoJSON.FeatureCollection): SubstationPoint[] {
  return (geojson.features ?? [])
    .filter((f) => f.geometry.type === "Point")
    .map((f) => {
      const [lon, lat] = (f.geometry as GeoJSON.Point).coordinates;
      const name = (f.properties?.NAME ?? f.properties?.name ?? "Substation") as string;
      return { lon, lat, name };
    });
}

// ── Props ─────────────────────────────────────────────────────────────────────
interface Props {
  heatmapPoints:  HeatmapPoint[];
  maskPoints:     HeatmapPoint[];
  substationGeo:  GeoJSON.FeatureCollection;
  candidates:     Candidate[];
  selectedIdx:    number | null;
  showDemand:     boolean;
  showForbidden:  boolean;
  showSubstations: boolean;
  onCandidateClick: (idx: number) => void;
}

export default function MapCanvas({
  heatmapPoints,
  maskPoints,
  substationGeo,
  candidates,
  selectedIdx,
  showDemand,
  showForbidden,
  showSubstations,
  onCandidateClick,
}: Props) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [viewState, setViewState] = useState<any>(INITIAL_VIEW);

  // Fly to selected candidate
  useEffect(() => {
    if (selectedIdx === null) return;
    const c = candidates[selectedIdx];
    if (!c) return;
    setViewState({
      longitude: c.lon,
      latitude:  c.lat,
      zoom:      13,
      pitch:     0,
      bearing:   0,
      transitionDuration:    1_000,
      transitionInterpolator: new FlyToInterpolator({ speed: 1.4 }),
    });
  }, [selectedIdx, candidates]);

  const substations = useMemo(() => extractSubstations(substationGeo), [substationGeo]);

  const selectedCandidate = selectedIdx !== null ? candidates[selectedIdx] : null;

  // Enrich candidates with color
  const enriched = useMemo(
    () =>
      candidates.map((c) => ({
        ...c,
        fillColor: [...FEASIBILITY_COLORS_RGB[getFeasibilityVerdict(c)], 230] as [number, number, number, number],
      })),
    [candidates],
  );

  const layers = useMemo(() => {
    const all = [];

    // ── Demand heatmap ──────────────────────────────────────────────────────
    if (showDemand && heatmapPoints.length > 0) {
      all.push(
        new HeatmapLayer({
          id:           "demand-heatmap",
          data:          heatmapPoints,
          getPosition:  (d: HeatmapPoint) => [d.lon, d.lat] as [number, number],
          getWeight:    (d: HeatmapPoint) => d.value,
          radiusPixels:  40,
          intensity:     1.2,
          threshold:     0.03,
          colorRange: [
            [49,  54,  149, 200],
            [116, 173, 209, 200],
            [255, 255, 191, 200],
            [253, 174,  97, 200],
            [165,   0,  38, 200],
          ] as [number, number, number, number][],
        }),
      );
    }

    // ── Forbidden zones ─────────────────────────────────────────────────────
    if (showForbidden && maskPoints.length > 0) {
      all.push(
        new ScatterplotLayer({
          id:              "forbidden-zones",
          data:             maskPoints,
          getPosition:     (d: HeatmapPoint) => [d.lon, d.lat] as [number, number],
          getRadius:        80,
          getFillColor:    [255, 23, 68, 40] as [number, number, number, number],
          radiusMinPixels:  2,
          radiusMaxPixels:  5,
        }),
      );
    }

    // ── Coverage rings ───────────────────────────────────────────────────────
    if (selectedCandidate) {
      all.push(
        new PolygonLayer<CoverageRing>({
          id:              "coverage-rings",
          data:             makeCoverageRings(selectedCandidate),
          getPolygon:      (d) => d.polygon,
          getFillColor:    (d) => d.fillColor,
          getLineColor:    (d) => d.lineColor,
          getLineWidth:    2,
          lineWidthMinPixels: 1.5,
          filled:           true,
          stroked:          true,
        }),
      );
    }

    // ── Existing substations ────────────────────────────────────────────────
    if (showSubstations && substations.length > 0) {
      all.push(
        new ScatterplotLayer<SubstationPoint>({
          id:              "substations",
          data:             substations,
          getPosition:     (d) => [d.lon, d.lat] as [number, number],
          getRadius:        220,
          getFillColor:    [240, 240, 255, 220] as [number, number, number, number],
          getLineColor:    [8, 8, 18, 255] as [number, number, number, number],
          lineWidthMinPixels: 2,
          stroked:          true,
          radiusMinPixels:  5,
          radiusMaxPixels:  14,
          pickable:         true,
        }),
      );
    }

    // ── Candidate circles ────────────────────────────────────────────────────
    if (enriched.length > 0) {
      all.push(
        new ScatterplotLayer({
          id:              "candidates",
          data:             enriched,
          getPosition:     (d: typeof enriched[0]) => [d.lon, d.lat] as [number, number],
          getRadius:       (d: typeof enriched[0]) => Math.max(180, 600 - (d.rank - 1) * 40),
          getFillColor:    (d: typeof enriched[0]) => d.fillColor,
          getLineColor:    [255, 255, 255, 180] as [number, number, number, number],
          lineWidthMinPixels: 2,
          stroked:          true,
          filled:           true,
          radiusMinPixels:  8,
          radiusMaxPixels: 22,
          pickable:         true,
          autoHighlight:    true,
          highlightColor:  [255, 255, 255, 50] as [number, number, number, number],
          onClick:         ({ index }: PickingInfo) => {
            if (index >= 0) onCandidateClick(index);
          },
        }),
      );

      // Rank number labels inside circles
      all.push(
        new TextLayer({
          id:              "candidate-labels",
          data:             enriched,
          getPosition:     (d: typeof enriched[0]) => [d.lon, d.lat] as [number, number],
          getText:         (d: typeof enriched[0]) => String(d.rank),
          getSize:          11,
          getColor:        [255, 255, 255, 240] as [number, number, number, number],
          getTextAnchor:   "middle",
          getAlignmentBaseline: "center",
          fontWeight:       "bold",
          fontFamily:       "Inter, system-ui, sans-serif",
          pickable:         false,
        }),
      );
    }

    return all;
  }, [heatmapPoints, maskPoints, substations, enriched, selectedCandidate, showDemand, showForbidden, showSubstations, onCandidateClick]);

  const getTooltip = useCallback(({ object, layer }: PickingInfo) => {
    if (!object) return null;
    if (layer?.id === "substations") {
      const sub = object as SubstationPoint;
      return { text: sub.name };
    }
    if (layer?.id === "candidates") {
      const c = object as typeof enriched[0];
      return {
        html: `
          <div style="font-family:Inter,system-ui,sans-serif;padding:8px 12px;line-height:1.5">
            <strong style="color:#00d4ff">Rank #${c.rank}</strong>
            ${c.name ? `<br/><span style="color:#eef0ff;font-size:11px">${c.name}</span>` : ""}
            <br/><span style="color:#8892b0;font-size:11px">Score: ${c.composite_score.toFixed(3)}</span>
          </div>`,
        style: {
          backgroundColor:  "#13132a",
          border:           "1px solid #1c1c38",
          borderRadius:     "8px",
          color:            "#eef0ff",
          fontSize:         "12px",
          padding:          "0",
        },
      };
    }
    return null;
  }, [enriched]);

  return (
    <DeckGL
      viewState={viewState}
      onViewStateChange={({ viewState: vs }) => setViewState(vs)}
      controller={true}
      layers={layers}
      getTooltip={getTooltip}
      style={{ position: "relative" }}
    >
      <Map mapStyle={MAP_STYLE} reuseMaps />
    </DeckGL>
  );
}
