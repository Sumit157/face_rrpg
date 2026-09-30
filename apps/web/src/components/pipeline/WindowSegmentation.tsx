import { useState } from "react";
import type { VisualRegionWindows } from "@rppg/shared-types";
import { BPMChart } from "../signal/BPMChart";

interface Props {
  windows: VisualRegionWindows[];
  currentTime: number;
  onSeek: (t: number) => void;
}

export function WindowSegmentation({ windows, currentTime, onSeek }: Props) {
  const [region, setRegion] = useState(windows[0]?.region ?? "");
  const active = windows.find((w) => w.region === region) ?? windows[0];
  if (!active) return null;

  const inWindow = (startS: number, endS: number) =>
    currentTime >= startS && currentTime < endS;

  return (
    <div className="panel p-4">
      <div className="flex items-center gap-3 mb-4 flex-wrap">
        <h3 className="text-sm text-slate-300">Sliding windows</h3>
        <span className="text-[11px] text-slate-500">
          each window is analysed independently -- click one to jump there
        </span>
        <div className="ml-auto flex gap-1">
          {windows.map((w) => (
            <button
              key={w.region}
              onClick={() => setRegion(w.region)}
              className={`text-[11px] px-2 py-1 rounded border transition-colors ${
                w.region === active.region
                  ? "border-accent text-accent bg-accent/10"
                  : "border-surface-700 text-slate-400 hover:text-slate-200"
              }`}
            >
              {w.region.replace("_", " ")}
            </button>
          ))}
        </div>
      </div>

      {active.windows.length > 1 && (
        <BPMChart
          windowIndex={active.windows.map((w) => w.index)}
          bpm={active.windows.map((w) => w.bpm)}
        />
      )}

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3 mt-3">
        {active.windows.map((w) => {
          const isActive = inWindow(w.startS, w.endS);
          return (
            <button
              key={w.index}
              onClick={() => onSeek(w.startS + 0.01)}
              className={`panel p-3 text-left transition-all cursor-pointer ${
                isActive
                  ? "shadow-glow border-accent/60"
                  : "hover:border-surface-700"
              }`}
              title={`${w.startS.toFixed(1)}s - ${w.endS.toFixed(1)}s`}
            >
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] uppercase tracking-wide text-slate-500">
                  window {w.index + 1}
                </span>
                {isActive && (
                  <span className="text-[9px] text-accent uppercase">now</span>
                )}
              </div>
              <div className="text-lg font-mono text-slate-100">
                {w.bpm.toFixed(1)}
                <span className="text-xs text-slate-500 ml-1">bpm</span>
              </div>
              <div className="flex items-center justify-between mt-1 text-[10px] font-mono text-slate-500">
                <span>
                  {w.startS.toFixed(1)}-{w.endS.toFixed(1)}s
                </span>
                {w.quality != null && <span>q {w.quality.toFixed(0)}</span>}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
