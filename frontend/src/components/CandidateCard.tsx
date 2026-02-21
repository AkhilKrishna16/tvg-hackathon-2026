"use client";

import { clsx } from "clsx";
import type { Candidate } from "@/lib/types";
import { FEASIBILITY_COLORS, getFeasibilityVerdict } from "@/lib/types";

interface Props {
  candidate: Candidate;
  selected:  boolean;
  onClick:   () => void;
}

export default function CandidateCard({ candidate, selected, onClick }: Props) {
  const verdict = getFeasibilityVerdict(candidate);
  const color   = FEASIBILITY_COLORS[verdict];
  const score   = candidate.composite_score;

  return (
    <button
      onClick={onClick}
      className={clsx(
        "w-full text-left px-4 py-3 transition-all duration-150 group relative",
        "border-b border-[#1c1c38]",
        selected
          ? "bg-[#13132a]"
          : "bg-transparent hover:bg-[#0f0f20]",
      )}
      style={{ borderLeft: `3px solid ${selected ? color : color + "55"}` }}
    >
      {/* Rank + name row */}
      <div className="flex items-start gap-3">
        {/* Rank number */}
        <span
          className="text-[15px] font-black tabular-nums min-w-[24px] mt-0.5"
          style={{ color: "#00d4ff" }}
        >
          #{candidate.rank}
        </span>

        {/* Centre column */}
        <div className="flex-1 min-w-0">
          {candidate.name ? (
            <div className="text-[12px] font-semibold text-[#eef0ff] truncate leading-snug mb-0.5">
              {candidate.name}
            </div>
          ) : (
            <div className="text-[12px] font-semibold text-[#eef0ff] truncate leading-snug mb-0.5">
              {candidate.lat.toFixed(4)}°N, {Math.abs(candidate.lon).toFixed(4)}°W
            </div>
          )}

          {/* Score bar */}
          <div className="flex items-center gap-2 mt-1">
            <div className="flex-1 h-[3px] rounded-full bg-[#1c1c38] overflow-hidden">
              <div
                className="h-full rounded-full"
                style={{ width: `${score * 100}%`, backgroundColor: color, opacity: 0.85 }}
              />
            </div>
            <span className="text-[10px] font-bold tabular-nums text-[#6370a0]">
              {score.toFixed(3)}
            </span>
          </div>

          {/* Secondary info */}
          {candidate.nearest_existing_km !== undefined && (
            <div className="text-[10px] text-[#6370a0] mt-0.5">
              {candidate.nearest_existing_km.toFixed(2)} km to grid
              {candidate.coverage_5km_pct !== undefined &&
                ` · ${candidate.coverage_5km_pct.toFixed(1)}% demand @5km`}
            </div>
          )}
        </div>

        {/* Feasibility badge */}
        <span
          className="text-[9px] font-black tracking-wider px-1.5 py-0.5 rounded shrink-0 mt-0.5"
          style={{
            backgroundColor: `${color}1a`,
            color,
            border: `1px solid ${color}44`,
          }}
        >
          {verdict === "UNKNOWN" ? "—" : verdict}
        </span>
      </div>
    </button>
  );
}
