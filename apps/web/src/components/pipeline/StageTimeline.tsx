import type { VisualStage } from "@rppg/shared-types";

interface Props {
  stages: VisualStage[];
}

interface StageMeta {
  label: string;
  desc: string;
  color: string;
}

const META: Record<string, StageMeta> = {
  validating: {
    label: "Validate",
    desc: "Checks the file is a readable video with usable frames",
    color: "#64748b",
  },
  extracting_faces: {
    label: "Face detection",
    desc: "Finds the face and locates the forehead and cheek regions",
    color: "#22d3ee",
  },
  extracting_signals: {
    label: "rPPG extraction",
    desc: "Measures colour changes in each region to recover the pulse",
    color: "#a78bfa",
  },
  processing_signals: {
    label: "Signal cleanup",
    desc: "Filters noise and slices signals into sliding windows",
    color: "#f472b6",
  },
  extracting_features: {
    label: "Features",
    desc: "Computes pulse timing, cross-region agreement and quality",
    color: "#fbbf24",
  },
  running_model: {
    label: "Classifier",
    desc: "Scores the feature vector against the decision model",
    color: "#34d399",
  },
  generating_report: {
    label: "Report",
    desc: "Assembles the structured result, evidence and warnings",
    color: "#60a5fa",
  },
  completed: { label: "Done", desc: "Analysis finished", color: "#34d399" },
  failed: { label: "Stopped", desc: "Analysis ended early", color: "#f87171" },
};

function fmtDuration(ms: number): string {
  return ms < 1000 ? `${ms}ms` : `${(ms / 1000).toFixed(1)}s`;
}

export function StageTimeline({ stages }: Props) {
  const totalMs = stages.reduce((s, x) => s + x.durationMs, 0);
  const measurable = stages.filter((s) => s.stage !== "completed");

  return (
    <div className="panel p-4">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-sm text-slate-300">Pipeline stages</h3>
        <span className="text-[11px] font-mono text-slate-500">
          {fmtDuration(totalMs)} total
        </span>
      </div>

      {/* Proportional time bar -- instant stages keep a minimum width so
          they stay visible and hoverable. */}
      <div className="flex h-7 gap-px rounded overflow-hidden bg-surface-900">
        {stages.map((s, i) => {
          const meta = META[s.stage] ?? {
            label: s.stage,
            desc: s.stage,
            color: "#475569",
          };
          return (
            <div
              key={`${s.stage}-${i}`}
              className="min-w-[6px] relative group"
              style={{
                flexGrow: Math.max(s.durationMs, 250),
                background: meta.color,
                opacity: 0.85,
              }}
              title={`${meta.label} -- ${fmtDuration(s.durationMs)} (${s.startS.toFixed(
                1
              )}s -> ${s.endS.toFixed(1)}s)${s.message ? `\n${s.message}` : ""}`}
            >
              <span className="absolute inset-0 flex items-center justify-center text-[9px] font-semibold text-surface-950/90 truncate px-1 select-none">
                {s.durationMs > 1200 ? meta.label : ""}
              </span>
            </div>
          );
        })}
      </div>

      <ol className="mt-4 space-y-1.5">
        {measurable.map((s, i) => {
          const meta = META[s.stage] ?? {
            label: s.stage,
            desc: s.stage,
            color: "#475569",
          };
          return (
            <li
              key={`${s.stage}-${i}`}
              className="flex items-center gap-3 text-xs"
            >
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ background: meta.color }}
              />
              <span className="text-slate-200 w-28 shrink-0 font-medium">
                {meta.label}
              </span>
              <span className="text-slate-500 flex-1 truncate" title={s.message ?? undefined}>
                {meta.desc}
              </span>
              <span className="font-mono text-slate-400 shrink-0">
                {fmtDuration(s.durationMs)}
              </span>
            </li>
          );
        })}
        {stages.some((s) => s.stage === "failed") && (
          <li className="flex items-center gap-3 text-xs pt-1 border-t border-surface-700">
            <span className="w-2 h-2 rounded-full shrink-0 bg-verdict-fake" />
            <span className="text-verdict-fake w-28 shrink-0 font-medium">
              Stopped
            </span>
            <span className="text-slate-500 flex-1">
              The pipeline ended before completing
            </span>
          </li>
        )}
      </ol>
    </div>
  );
}
