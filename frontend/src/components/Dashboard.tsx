"use client";

import { useState } from "react";
import dynamic from "next/dynamic";

import { useMapData } from "@/hooks/useMapData";
import Sidebar from "./Sidebar";
import type { LayerState } from "./LayerControls";

// MapCanvas is WebGL — must be dynamically imported with SSR disabled
const MapCanvas = dynamic(() => import("./MapCanvas"), { ssr: false });

const DEFAULT_LAYERS: LayerState = {
  demand:      true,
  forbidden:   true,
  substations: true,
};

export default function Dashboard() {
  const data = useMapData();
  const [selectedIdx, setSelectedIdx]   = useState<number | null>(null);
  const [layers, setLayers]             = useState<LayerState>(DEFAULT_LAYERS);

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#080812]">
      {/* ── Map (left, fluid) ───────────────────────────────────────────────── */}
      <div className="flex-1 relative">
        {data.loading ? (
          <LoadingMap />
        ) : (
          <MapCanvas
            heatmapPoints   ={data.heatmapPoints}
            maskPoints      ={data.maskPoints}
            substationGeo   ={data.substation}
            candidates      ={data.candidates}
            selectedIdx     ={selectedIdx}
            showDemand      ={layers.demand}
            showForbidden   ={layers.forbidden}
            showSubstations ={layers.substations}
            onCandidateClick={setSelectedIdx}
          />
        )}

        {/* Legend overlay (bottom-left) */}
        <div className="absolute bottom-5 left-5 z-10 pointer-events-none">
          <Legend />
        </div>

        {/* Error banner */}
        {data.error && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-20 bg-[#ff1744] text-white text-xs font-semibold px-4 py-2 rounded-full shadow-lg">
            {data.error}
          </div>
        )}
      </div>

      {/* ── Sidebar (right, fixed 380px) ────────────────────────────────────── */}
      <div className="w-[380px] shrink-0">
        <Sidebar
          candidates   ={data.candidates}
          selectedIdx  ={selectedIdx}
          onSelect     ={setSelectedIdx}
          layers       ={layers}
          onLayerChange={setLayers}
          loading      ={data.loading}
        />
      </div>
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function LoadingMap() {
  return (
    <div className="w-full h-full bg-[#080812] flex flex-col items-center justify-center gap-4">
      <div className="w-8 h-8 rounded-full border-2 border-[#1c1c38] border-t-[#00d4ff] animate-spin" />
      <div className="text-[12px] text-[#6370a0] tracking-widest uppercase animate-pulse">
        Loading map data…
      </div>
    </div>
  );
}

function Legend() {
  return (
    <div
      className="bg-[#080812]/90 border border-[#1c1c38] rounded-xl px-3 py-2.5 text-[10px] space-y-1.5"
      style={{ backdropFilter: "blur(8px)" }}
    >
      <div className="text-[#6370a0] font-bold uppercase tracking-widest mb-2">Legend</div>
      <LegendRow color="#ef4444"  label="Demand intensity" />
      <LegendRow color="#ff1744"  label="Forbidden zones"   opacity="0.3" />
      <LegendRow color="#f0f0ff"  label="Existing substations" />
      <LegendRow color="#00e676"  label="Candidate — HIGH"  />
      <LegendRow color="#ffc107"  label="Candidate — MEDIUM" />
      <LegendRow color="#ff1744"  label="Candidate — LOW"   />
      <LegendRow color="#00d4ff"  label="Coverage rings (3/5/10 km)" dashed />
    </div>
  );
}

function LegendRow({
  color, label, opacity = "1", dashed = false,
}: { color: string; label: string; opacity?: string; dashed?: boolean }) {
  return (
    <div className="flex items-center gap-2 text-[#8892b0]">
      {dashed ? (
        <div className="w-4 h-[2px] rounded-full" style={{ background: `repeating-linear-gradient(90deg,${color} 0,${color} 3px,transparent 3px,transparent 6px)` }} />
      ) : (
        <div className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: color, opacity }} />
      )}
      {label}
    </div>
  );
}
