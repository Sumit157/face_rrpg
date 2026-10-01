import { useState } from "react";
import { useParams, Link, useSearchParams } from "react-router-dom";
import { Layout } from "../components/layout/Layout";
import { AnalysisProgress } from "../components/analysis/AnalysisProgress";
import { PredictionCard } from "../components/analysis/PredictionCard";
import { SignalQuality } from "../components/analysis/SignalQuality";
import { EvidencePanel } from "../components/analysis/EvidencePanel";
import { WarningsPanel } from "../components/analysis/WarningsPanel";
import { RegionCorrelation } from "../components/signal/RegionCorrelation";
import { PipelineView } from "../components/pipeline/PipelineView";
import { useAnalysis } from "../hooks/useAnalysis";
import { useAnalysisProgress } from "../hooks/useWebSocket";

const TERMINAL = new Set(["completed", "failed"]);
type Tab = "result" | "pipeline";

export default function Analysis() {
  const { id } = useParams<{ id: string }>();
  const { data, isLoading, error } = useAnalysis(id);
  const progressEvent = useAnalysisProgress(id);
  // Deep-linkable tab: /analysis/:id?tab=pipeline
  const [searchParams, setSearchParams] = useSearchParams();
  const [tab, setTab] = useState<Tab>(
    searchParams.get("tab") === "pipeline" ? "pipeline" : "result"
  );
  const selectTab = (next: Tab) => {
    setTab(next);
    setSearchParams({ tab: next }, { replace: true });
  };

  const status = data?.status;
  const isDone = !!status && TERMINAL.has(status);
  const failed = status === "failed";
  const completed = status === "completed" && !!data?.prediction && !!data?.confidence;

  return (
    <Layout>
      <div className="flex items-center gap-3 mb-6">
        <h1 className="page-title">Analysis {id?.slice(0, 8)}</h1>
        {status && (
          <span
            className={`text-[11px] uppercase tracking-wide px-2 py-0.5 rounded border ${
              failed
                ? "text-verdict-fake border-verdict-fake"
                : completed
                  ? "text-verdict-real border-verdict-real"
                  : "text-accent border-accent"
            }`}
          >
            {status.replace("_", " ")}
          </span>
        )}
      </div>

      {isLoading && (
        <div className="space-y-4" aria-busy="true">
          <div className="skeleton h-32 w-full" />
          <div className="skeleton h-48 w-full" />
        </div>
      )}

      {error && (
        <div className="panel p-4 border border-verdict-fake">
          <p className="text-sm text-verdict-fake">Failed to load this analysis.</p>
          <Link to="/dashboard" className="text-accent text-sm">
            Back to dashboard
          </Link>
        </div>
      )}

      {!isLoading && !isDone && !error && (
        <div className="space-y-4">
          <AnalysisProgress event={progressEvent} status={status} />
          <p className="text-xs text-slate-500">
            This page updates automatically while the analysis runs.
          </p>
        </div>
      )}

      {isDone && data && (
        <>
          <div className="flex gap-1 mb-6 border-b border-surface-700" role="tablist">
            {(
              [
                ["result", "Result"],
                ["pipeline", "Pipeline"],
              ] as [Tab, string][]
            ).map(([key, label]) => (
              <button
                key={key}
                role="tab"
                aria-selected={tab === key}
                onClick={() => selectTab(key)}
                className={`px-4 py-2 text-sm border-b-2 -mb-px transition-colors ${
                  tab === key
                    ? "border-accent text-accent"
                    : "border-transparent text-slate-400 hover:text-slate-200"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {tab === "result" && (
            <>
              {failed && (
                <div className="panel p-6 border border-verdict-fake">
                  <h2 className="text-verdict-fake font-semibold mb-2">Analysis failed</h2>
                  <p className="text-sm text-slate-300">
                    {data.errorMessage || "The pipeline could not produce a result for this video."}
                  </p>
                  <p className="text-xs text-slate-500 mt-3">
                    Common causes: no detectable face in the video, too few frames, or a missing
                    face-detection model asset.
                  </p>
                  <button
                    onClick={() => selectTab("pipeline")}
                    className="text-accent text-sm mt-4 hover:underline"
                  >
                    Check the Pipeline tab to see how far it got
                  </button>
                </div>
              )}

              {completed && (
                <div className="space-y-6">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="col-span-1">
                      <PredictionCard prediction={data.prediction!} confidence={data.confidence!} />
                    </div>
                    <div className="col-span-2 grid grid-cols-2 gap-4">
                      <SignalQuality score={(data.signalQuality as any)?.overall ?? 0} />
                      <div className="panel p-4">
                        <div className="text-xs text-slate-400 mb-1">Mean BPM</div>
                        <div className="text-2xl font-semibold font-mono">
                          {data.bpm?.overall?.mean ?? "--"}
                        </div>
                      </div>
                      <EvidencePanel evidence={data.evidence ?? []} />
                      <WarningsPanel warnings={data.warnings ?? []} />
                    </div>
                  </div>

                  <div className="panel p-4">
                    <div className="flex items-center justify-between gap-3 mb-3">
                      <h3 className="text-sm text-slate-300">Decision diagnostics</h3>
                      <Link to="/signal-explorer" className="text-xs text-accent hover:underline">
                        Inspect waveforms
                      </Link>
                    </div>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                      <Diagnostic label="Quality-gated windows"
                        value={`${Math.round(((data.signalQuality as any)?.gating?.keptRatio ?? 0) * 100)}%`}
                        detail={`${(data.signalQuality as any)?.gating?.keptWindows ?? 0} / ${(data.signalQuality as any)?.gating?.totalWindows ?? 0} usable`} />
                      <Diagnostic label="Weighted usable evidence"
                        value={`${Math.round(((data.signalQuality as any)?.gating?.effectiveKeptRatio ?? 0) * 100)}%`}
                        detail="borderline compression windows down-weighted" />
                      <Diagnostic label="Measurement strength"
                        value={`${Math.round(((data.signalQuality as any)?.measurementStrength ?? 0) * 100)}%`}
                        detail="caps confidence when video is weak" />
                      <Diagnostic label="Primary rPPG method"
                        value={String((data.signalQuality as any)?.selectedMethod ?? data.method ?? "auto").toUpperCase()}
                        detail="best quality path; auto mode compares all methods" />
                    </div>
                    {Object.keys((data.signalQuality as any)?.methodScores ?? {}).length > 0 && (
                      <div className="mt-4 border-t border-surface-700 pt-3">
                        <div className="text-xs text-slate-500 mb-2">
                          Method signal quality and estimated heart rate
                          <span className="ml-2">
                            Frequency agreement: {Object.values((data.signalQuality as any)?.methodMeanBpm ?? {}).filter((bpm) => Number(bpm) > 0).length >= 2
                              ? `${Math.round(((data.signalQuality as any).methodFrequencyAgreement ?? 0) * 100)}%`
                              : "not measurable"}
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries((data.signalQuality as any).methodScores).map(([method, score]) => (
                            <span key={method} className="font-mono text-xs px-2 py-1 rounded bg-surface-800 text-slate-300">
                              {method.toUpperCase()}: {Number(score) >= 0 ? `${Number(score).toFixed(1)} quality` : "not measurable"}
                              {Number((data.signalQuality as any)?.methodMeanBpm?.[method]) > 0 &&
                                ` · ${Number((data.signalQuality as any).methodMeanBpm[method]).toFixed(0)} bpm`}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <RegionCorrelation
                    pairwiseCorrelations={(data.consistency as any)?.pairwiseCorrelations ?? {}}
                  />

                  <div className="panel p-4">
                    <h3 className="text-sm text-slate-300 mb-3">Per-region BPM</h3>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      {(data.regions ?? []).map((r) => {
                        const region = (data.bpm as any)?.[r];
                        return (
                          <div key={r} className="panel p-3">
                            <div className="text-slate-400 text-xs mb-1">{r.replace("_", " ")}</div>
                            <div className="text-lg font-mono">
                              {region?.bpmMean?.toFixed(1) ?? "--"} bpm
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              )}
            </>
          )}

          {tab === "pipeline" && (
            <PipelineView analysisId={data.id} videoId={data.videoId} />
          )}
        </>
      )}
    </Layout>
  );
}

function Diagnostic({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded bg-surface-800 p-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="mt-1 font-mono text-base text-slate-100">{value}</div>
      <div className="mt-1 text-[11px] leading-snug text-slate-500">{detail}</div>
    </div>
  );
}
