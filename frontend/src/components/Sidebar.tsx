"use client";

import { useState } from "react";
import type { Candidate } from "@/lib/types";
import ObjectiveSliders from "./ObjectiveSliders";
import LayerControls from "./LayerControls";
import CandidateCard from "./CandidateCard";
import FeasibilityReport from "./FeasibilityReport";
import type { LayerState } from "./LayerControls";
type LS = LayerState;

interface Props {
  candidates:    Candidate[];
  selectedIdx:   number | null;
  onSelect:      (idx: number | null) => void;
  layers:        LS;
  onLayerChange: (l: LS) => void;
  loading:       boolean;
}

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[10px] font-black tracking-[1.8px] text-[#6370a0] uppercase mb-3">
      {children}
    </div>
  );
}

function Divider() {
  return <div className="border-t border-[#1c1c38] my-0" />;
}

export default function Sidebar({ candidates, selectedIdx, onSelect, layers, onLayerChange, loading }: Props) {
  const selected = selectedIdx !== null ? candidates[selectedIdx] ?? null : null;
  const [reportOpen, setReportOpen] = useState(false);

  const handleSelect = (idx: number) => {
    if (idx === selectedIdx) {
      onSelect(null);
      setReportOpen(false);
    } else {
      onSelect(idx);
      setReportOpen(true);
    }
  };

  return (
    <aside
      className="flex flex-col bg-[#080812] border-l border-[#1c1c38] overflow-hidden"
      style={{ height: "100vh", width: "100%", fontFamily: "Inter, system-ui, sans-serif" }}
    >
      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div className="px-5 pt-5 pb-4 shrink-0">
        {/* Top accent line */}
        <div
          className="h-[2px] w-10 rounded-full mb-4"
          style={{ background: "linear-gradient(90deg, #00d4ff, #7c3aed)" }}
        />
        <div className="text-[13px] font-black tracking-[2.5px] text-[#00d4ff] mb-0.5">
          GRID PLACEMENT OPTIMIZER
        </div>
        <div className="text-[11px] text-[#6370a0] tracking-wide">
          Austin, TX · EV Infrastructure AI
        </div>

        {/* Stats row */}
        <div className="flex gap-4 mt-4">
          <Stat value={candidates.length} label="candidates" loading={loading} />
          <Stat value={10}                label="substations" loading={loading} />
          <Stat value={500}               label="grid cells"  loading={loading} suffix="²" />
        </div>
      </div>

      <Divider />

      {/* ── Scrollable body ──────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden scrollbar-thin">

        {/* Objective weights */}
        <div className="px-5 py-4">
          <SectionHeader>Objective Weights</SectionHeader>
          <ObjectiveSliders />
        </div>

        <Divider />

        {/* Layer controls */}
        <div className="px-5 py-4">
          <SectionHeader>Map Layers</SectionHeader>
          <LayerControls layers={layers} onChange={onLayerChange} />
        </div>

        <Divider />

        {/* Candidate list */}
        <div className="py-3">
          <div className="px-5 mb-1">
            <SectionHeader>Top Candidate Sites</SectionHeader>
          </div>

          {loading ? (
            <div className="px-5 space-y-2">
              {[1,2,3,4,5].map((i) => (
                <div key={i} className="h-14 rounded-lg bg-[#0e0e1c] animate-pulse" />
              ))}
            </div>
          ) : candidates.length === 0 ? (
            <div className="px-5">
              <NoCandidates />
            </div>
          ) : (
            candidates.slice(0, 10).map((c, i) => (
              <CandidateCard
                key={c.rank}
                candidate={c}
                selected={selectedIdx === i}
                onClick={() => handleSelect(i)}
              />
            ))
          )}
        </div>

        <Divider />

        {/* Feasibility report */}
        <div className="px-5 py-4 pb-8">
          <div
            className="flex items-center justify-between cursor-pointer mb-3"
            onClick={() => setReportOpen((o) => !o)}
          >
            <SectionHeader>Feasibility Report</SectionHeader>
            {selected && (
              <span className="text-[10px] text-[#6370a0] mb-3">
                {reportOpen ? "▲ collapse" : "▼ expand"}
              </span>
            )}
          </div>
          {(reportOpen || !selected) && (
            <FeasibilityReport candidate={selected} />
          )}
        </div>

      </div>
    </aside>
  );
}

// ── Mini components ──────────────────────────────────────────────────────────

function Stat({
  value, label, loading, suffix = "",
}: { value: number; label: string; loading: boolean; suffix?: string }) {
  return (
    <div>
      <div className="text-[16px] font-black tabular-nums text-[#eef0ff]">
        {loading ? (
          <span className="inline-block w-6 h-4 bg-[#1c1c38] rounded animate-pulse" />
        ) : (
          <>{value}{suffix}</>
        )}
      </div>
      <div className="text-[9px] text-[#6370a0] uppercase tracking-wide">{label}</div>
    </div>
  );
}

function NoCandidates() {
  return (
    <div className="text-center py-6 border border-dashed border-[#1c1c38] rounded-xl">
      <div className="text-2xl mb-2">⚠️</div>
      <div className="text-[12px] font-semibold text-[#eef0ff] mb-1">No candidates yet</div>
      <div className="text-[11px] text-[#6370a0] mb-3">Run the optimizer first:</div>
      <code className="block text-[11px] text-[#00d4ff] bg-[#0e0e1c] border border-[#1c1c38] rounded-md px-3 py-2 mx-2">
        python -m src.optimizer.run
      </code>
    </div>
  );
}
