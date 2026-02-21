"use client";

import { clsx } from "clsx";

export interface LayerState {
  demand:     boolean;
  forbidden:  boolean;
  substations: boolean;
}

interface Props {
  layers:   LayerState;
  onChange: (layers: LayerState) => void;
}

const LAYERS: { key: keyof LayerState; label: string; color: string }[] = [
  { key: "demand",      label: "Demand",      color: "#ef4444" },
  { key: "forbidden",   label: "Forbidden",   color: "#ff1744" },
  { key: "substations", label: "Substations", color: "#ffffff" },
];

export default function LayerControls({ layers, onChange }: Props) {
  const toggle = (key: keyof LayerState) =>
    onChange({ ...layers, [key]: !layers[key] });

  return (
    <div className="flex flex-wrap gap-2">
      {LAYERS.map(({ key, label, color }) => {
        const active = layers[key];
        return (
          <button
            key={key}
            onClick={() => toggle(key)}
            className={clsx(
              "flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold",
              "border transition-all duration-150 select-none",
              active
                ? "border-[#2a2a4e] bg-[#13132a] text-[#eef0ff]"
                : "border-[#1c1c38] bg-transparent text-[#6370a0]",
            )}
          >
            <span
              className="w-1.5 h-1.5 rounded-full inline-block"
              style={{ backgroundColor: active ? color : "#374151" }}
            />
            {label}
          </button>
        );
      })}
    </div>
  );
}
