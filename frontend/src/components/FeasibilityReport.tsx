"use client";

import type { Candidate, FeasibilityAssessment } from "@/lib/types";
import { FEASIBILITY_COLORS, getFeasibilityVerdict, getFeasibilityAssessment } from "@/lib/types";

interface Props {
  candidate: Candidate | null;
}

function ScoreBar({ value, color }: { value: number; color: string }) {
  return (
    <div className="h-[3px] rounded-full bg-[#1c1c38] overflow-hidden">
      <div
        className="h-full rounded-full transition-all duration-700"
        style={{ width: `${value * 100}%`, backgroundColor: color }}
      />
    </div>
  );
}

function ObjectiveRow({ label, value, color }: { label: string; value?: number; color: string }) {
  if (value === undefined) return null;
  return (
    <div className="mb-2">
      <div className="flex justify-between items-center mb-0.5">
        <span className="text-[10px] text-[#6370a0] uppercase tracking-wide font-semibold">{label}</span>
        <span className="text-[10px] font-bold tabular-nums" style={{ color }}>{value.toFixed(3)}</span>
      </div>
      <ScoreBar value={value} color={color} />
    </div>
  );
}

function FlagList({ flags }: { flags: string[] }) {
  return (
    <ul className="space-y-1 mt-1">
      {flags.map((f, i) => (
        <li key={i} className="flex gap-1.5 text-[11px] text-[#8892b0] leading-relaxed">
          <span className="text-[#ffc107] mt-0.5 shrink-0">▸</span>
          <span>{f}</span>
        </li>
      ))}
    </ul>
  );
}

export default function FeasibilityReport({ candidate }: Props) {
  if (!candidate) {
    return (
      <p className="text-[12px] text-[#6370a0] italic leading-relaxed">
        Select a candidate site from the list above to view its full AI-generated feasibility analysis.
      </p>
    );
  }

  const verdict    = getFeasibilityVerdict(candidate);
  const color      = FEASIBILITY_COLORS[verdict];
  const assessment = getFeasibilityAssessment(candidate);
  const score      = candidate.composite_score;

  return (
    <div className="animate-slide-up space-y-4">
      {/* ── Header ───────────────────────────────────────────────────── */}
      <div className="flex items-center gap-3">
        <span className="text-[17px] font-black" style={{ color: "#00d4ff" }}>
          Rank #{candidate.rank}
        </span>
        <span
          className="text-[10px] font-black tracking-widest px-2 py-0.5 rounded"
          style={{ backgroundColor: `${color}1a`, color, border: `1px solid ${color}44` }}
        >
          {verdict}
        </span>
      </div>

      {/* ── Score callout ─────────────────────────────────────────────── */}
      <div
        className="rounded-lg p-3 border"
        style={{ backgroundColor: "#0e0e1c", borderColor: `${color}33` }}
      >
        <div className="text-[10px] text-[#6370a0] uppercase tracking-widest font-bold mb-1">
          Composite Score
        </div>
        <div className="text-[28px] font-black tabular-nums leading-none" style={{ color: "#eef0ff" }}>
          {score.toFixed(3)}
        </div>
        <div className="mt-2">
          <ScoreBar value={score} color={color} />
        </div>
      </div>

      {/* ── Coordinates ───────────────────────────────────────────────── */}
      <div className="text-[11px] text-[#6370a0]">
        📍 {candidate.lat.toFixed(5)}°N, {Math.abs(candidate.lon).toFixed(5)}°W
        {candidate.nearest_existing_km !== undefined && (
          <span className="ml-3 text-[#4b5563]">
            · {candidate.nearest_existing_km.toFixed(2)} km to nearest substation
          </span>
        )}
      </div>

      {/* ── Objective scores ──────────────────────────────────────────── */}
      {(candidate.load_relief_score !== undefined || candidate.loss_reduction_score !== undefined) && (
        <div>
          <div className="text-[10px] font-bold text-[#6370a0] uppercase tracking-widest mb-2">
            Objective Breakdown
          </div>
          <ObjectiveRow label="Load Relief"    value={candidate.load_relief_score}    color="#00d4ff" />
          <ObjectiveRow label="Loss Reduction" value={candidate.loss_reduction_score} color="#7c3aed" />
          <ObjectiveRow label="Sustainability" value={candidate.sustainability_score}  color="#10b981" />
          <ObjectiveRow label="Redundancy"     value={candidate.redundancy_score}      color="#f59e0b" />
        </div>
      )}

      {/* ── Coverage metrics ──────────────────────────────────────────── */}
      {candidate.coverage_3km_pct !== undefined && (
        <div>
          <div className="text-[10px] font-bold text-[#6370a0] uppercase tracking-widest mb-2">
            Demand Coverage
          </div>
          <div className="grid grid-cols-3 gap-2">
            {[
              { label: "3 km",  value: candidate.coverage_3km_pct },
              { label: "5 km",  value: candidate.coverage_5km_pct },
              { label: "10 km", value: candidate.coverage_10km_pct },
            ].map(({ label, value }) => value !== undefined && (
              <div
                key={label}
                className="rounded-md p-2 text-center border border-[#1c1c38] bg-[#0e0e1c]"
              >
                <div className="text-[14px] font-black tabular-nums text-[#eef0ff]">
                  {value.toFixed(1)}%
                </div>
                <div className="text-[9px] text-[#6370a0] uppercase tracking-wide mt-0.5">{label}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* ── Stability ─────────────────────────────────────────────────── */}
      {candidate.stability_pct !== undefined && (
        <div className="text-[11px] text-[#6370a0]">
          Appears in{" "}
          <span className="text-[#00d4ff] font-bold">{candidate.stability_count}</span> of 8 weight
          sets ({candidate.stability_pct.toFixed(0)}% stability)
        </div>
      )}

      {/* ── AI feasibility assessment ─────────────────────────────────── */}
      {assessment && (
        <>
          <div className="border-t border-[#1c1c38] pt-3">
            <div className="text-[10px] font-bold text-[#6370a0] uppercase tracking-widest mb-2">
              AI Analysis
            </div>
          </div>

          {assessment.reasoning && (
            <p className="text-[12px] text-[#8892b0] leading-relaxed">{assessment.reasoning}</p>
          )}

          {assessment.land_use && (
            <div>
              <div className="text-[10px] font-bold text-[#6370a0] uppercase tracking-widest mb-1">
                Land Use
              </div>
              <p className="text-[11px] text-[#8892b0] leading-relaxed">{assessment.land_use}</p>
            </div>
          )}

          {assessment.zoning_assessment && (
            <div>
              <div className="text-[10px] font-bold text-[#6370a0] uppercase tracking-widest mb-1">
                Zoning
              </div>
              <p className="text-[11px] text-[#8892b0] leading-relaxed">{assessment.zoning_assessment}</p>
            </div>
          )}

          {assessment.environmental_flags && assessment.environmental_flags.length > 0 && (
            <div>
              <div className="text-[10px] font-bold text-[#ffc107] uppercase tracking-widest mb-1">
                Environmental Flags
              </div>
              <FlagList flags={assessment.environmental_flags} />
            </div>
          )}

          {assessment.community_sensitivity && (
            <div>
              <div className="text-[10px] font-bold text-[#6370a0] uppercase tracking-widest mb-1">
                Community Sensitivity
              </div>
              <p className="text-[11px] text-[#8892b0] leading-relaxed">{assessment.community_sensitivity}</p>
            </div>
          )}

          {assessment.grid_proximity && (
            <div>
              <div className="text-[10px] font-bold text-[#6370a0] uppercase tracking-widest mb-1">
                Grid Proximity
              </div>
              <p className="text-[11px] text-[#8892b0] leading-relaxed">{assessment.grid_proximity}</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}
