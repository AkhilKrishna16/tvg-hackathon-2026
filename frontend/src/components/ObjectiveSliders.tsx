"use client";

interface Objective {
  label:   string;
  weight:  number;
  color:   string;
}

const OBJECTIVES: Objective[] = [
  { label: "Demand Coverage",   weight: 0.35, color: "#00d4ff" },
  { label: "Loss Reduction",    weight: 0.35, color: "#7c3aed" },
  { label: "Land Cost Penalty", weight: 0.15, color: "#f59e0b" },
  { label: "Redundancy Bonus",  weight: 0.15, color: "#10b981" },
];

export default function ObjectiveSliders() {
  return (
    <div className="space-y-3">
      {OBJECTIVES.map((obj) => (
        <div key={obj.label}>
          <div className="flex justify-between items-center mb-1">
            <span className="text-[11px] font-semibold tracking-wide text-[#6370a0] uppercase">
              {obj.label}
            </span>
            <span className="text-[11px] font-bold tabular-nums" style={{ color: obj.color }}>
              {obj.weight.toFixed(2)}
            </span>
          </div>
          {/* Track */}
          <div className="h-1 rounded-full bg-[#1c1c38] relative overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${obj.weight * 100}%`, backgroundColor: obj.color, opacity: 0.85 }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
