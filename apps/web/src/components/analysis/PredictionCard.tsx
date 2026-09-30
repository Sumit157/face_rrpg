import type { PredictionResult, CalibratedConfidence } from "@rppg/shared-types";

interface Props {
  prediction: PredictionResult;
  confidence: CalibratedConfidence;
}

const VERDICT_STYLE: Record<string, string> = {
  REAL: "text-verdict-real border-verdict-real/60",
  FAKE: "text-verdict-fake border-verdict-fake/60",
  UNCERTAIN: "text-verdict-uncertain border-verdict-uncertain/60",
  FAILED: "text-slate-500 border-slate-600",
};

export function PredictionCard({ prediction, confidence }: Props) {
  const style = VERDICT_STYLE[prediction.label] ?? VERDICT_STYLE.UNCERTAIN;
  return (
    <div className={`panel border-2 ${style} p-6 text-center h-full flex flex-col justify-center`}>
      <div className="label mb-2">Verdict</div>
      <div className="text-4xl font-semibold tracking-tight">{prediction.label}</div>
      <div className="font-mono text-lg mt-1 text-slate-200">
        {Math.round(confidence.adjustedConfidence * 100)}% confidence
      </div>
      {prediction.explanation && (
        <p className="text-xs text-slate-300 mt-3 mx-auto max-w-md leading-relaxed border-t border-surface-700 pt-3">
          {prediction.explanation}
        </p>
      )}
      <p className="text-xs text-slate-500 mt-4 leading-relaxed">
        Model: <span className="font-mono">{prediction.modelVersion}</span>
        {prediction.isHeuristicFallback && (
          <>
            <br />
            untrained heuristic — for demonstration only
          </>
        )}
      </p>
      <p className="text-[11px] text-slate-600 mt-2 leading-snug">
        Confidence discounted by signal quality ({confidence.signalQuality.toFixed(0)}/100).
        Raw probability {Math.round(confidence.rawProbability * 100)}%.
      </p>
    </div>
  );
}
