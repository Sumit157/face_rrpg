import type { VisualExtraction } from "@rppg/shared-types";

interface Props {
  extraction: VisualExtraction;
}

function Stat({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="panel p-3" title={hint}>
      <div className="text-[10px] uppercase tracking-wide text-slate-500 mb-1">
        {label}
      </div>
      <div className="text-sm font-mono text-slate-200">{value}</div>
    </div>
  );
}

export function ExtractionStats({ extraction }: Props) {
  const rate = Math.round((extraction.detectionRate ?? 0) * 100);
  const rateColor =
    rate >= 70 ? "text-verdict-real" : rate >= 30 ? "text-verdict-uncertain" : "text-verdict-fake";

  return (
    <div className="panel p-4">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-sm text-slate-300">Frame analysis</h3>
        <span className={`text-[11px] font-mono ${rateColor}`}>
          face found in {rate}% of frames
        </span>
      </div>

      <div className="h-1.5 rounded bg-surface-900 overflow-hidden mb-4">
        <div
          className="h-full transition-all"
          style={{
            width: `${rate}%`,
            background:
              rate >= 70 ? "#34d399" : rate >= 30 ? "#fbbf24" : "#f87171",
          }}
        />
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Stat
          label="Detector"
          value={extraction.detector}
          hint={
            extraction.detector === "haar"
              ? "MediaPipe was unavailable, so OpenCV Haar cascades were used"
              : "MediaPipe face mesh"
          }
        />
        <Stat
          label="Frames read"
          value={`${extraction.framesProcessed}`}
          hint={`${extraction.fps} fps x ${extraction.durationSeconds}s`}
        />
        <Stat label="Clip analysed" value={`${extraction.durationSeconds}s`} />
        <Stat
          label="Sample stride"
          value={`1 / ${extraction.sampleStride}`}
          hint="The overlay samples every Nth frame; all frames are still analysed"
        />
        <Stat
          label="Fallback frames"
          value={`${extraction.haarFallbackFrames}`}
          hint="Frames where Haar cascades were used instead of MediaPipe"
        />
        <Stat
          label="Enhanced frames"
          value={`${extraction.enhancedFrames}`}
          hint="Frames where contrast enhancement helped detection"
        />
        <Stat label="Frame rate" value={`${extraction.fps} fps`} />
        <Stat
          label="Analysis cap"
          value={`${extraction.capSeconds}s`}
          hint="Longer videos are capped to keep analysis time bounded"
        />
      </div>
    </div>
  );
}
