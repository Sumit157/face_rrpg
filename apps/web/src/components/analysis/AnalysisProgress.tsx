import type { AnalysisProgressEvent, AnalysisStatus } from "@rppg/shared-types";

const STAGES: { key: string; label: string }[] = [
  { key: "queued", label: "Queue" },
  { key: "validating", label: "Validate" },
  { key: "extracting_faces", label: "Faces" },
  { key: "extracting_signals", label: "Signals" },
  { key: "processing_signals", label: "Filter" },
  { key: "extracting_features", label: "Features" },
  { key: "running_model", label: "Model" },
  { key: "generating_report", label: "Report" },
  { key: "completed", label: "Done" },
];

/** Fallback progress per stage (mirrors the ML pipeline's real
 * stage percentages) for when no WebSocket event has arrived yet --
 * e.g. reloading the page while an analysis is mid-flight. */
const STAGE_PROGRESS: Record<string, number> = {
  queued: 0,
  validating: 5,
  extracting_faces: 20,
  extracting_signals: 45,
  processing_signals: 60,
  extracting_features: 75,
  running_model: 88,
  generating_report: 96,
  completed: 100,
  failed: 100,
};

interface Props {
  event: AnalysisProgressEvent | null;
  /** Current DB status -- used when no live event has arrived yet. */
  status?: AnalysisStatus;
}

export function AnalysisProgress({ event, status }: Props) {
  const stage = event?.stage ?? status ?? "queued";
  const failed = stage === "failed";
  const progress = event?.progress ?? STAGE_PROGRESS[stage] ?? 0;
  const message =
    event?.message ??
    (failed ? "Analysis failed" : `Stage: ${stage.replace("_", " ")}`);
  // Stage transitions (seconds-relative) forwarded from the ML progress
  // registry -- closed stages show how long they took, live.
  const history = event?.history ?? [];

  return (
    <div className="panel p-4">
      <div className="flex justify-between text-sm mb-2">
        <span className="text-slate-300 flex items-center gap-2">
          {!failed && progress < 100 && (
            <span className="inline-block h-2 w-2 rounded-full bg-accent animate-pulse" />
          )}
          {message}
        </span>
        <span className="text-slate-500 font-mono">{progress}%</span>
      </div>
      <div className="h-2 bg-surface-700 rounded overflow-hidden">
        <div
          className={`h-2 rounded transition-all duration-500 ${failed ? "bg-verdict-fake" : "bg-accent"}`}
          style={{ width: `${progress}%` }}
        />
      </div>
      <div className="flex justify-between text-[10px] text-slate-600 mt-2">
        {STAGES.map((s) => (
          <span
            key={s.key}
            className={
              stage === s.key || (s.key === "completed" && stage === "completed")
                ? "text-accent"
                : ""
            }
          >
            {s.label}
          </span>
        ))}
      </div>
      {history.length > 1 && (
        <div className="flex flex-wrap gap-x-3 gap-y-0.5 mt-2 pt-2 border-t border-surface-700 text-[10px] font-mono text-slate-500">
          {history.map((h, i) => {
            const next = history[i + 1];
            const label = h.stage.replace("_", " ");
            if (!next) {
              return (
                <span key={i} className="text-accent">
                  {label}&hellip;
                </span>
              );
            }
            const d = next.at - h.at;
            return (
              <span key={i}>
                {label}{" "}
                <span className="text-slate-400">
                  {d < 1 ? `${Math.round(d * 1000)}ms` : `${d.toFixed(1)}s`}
                </span>
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}
