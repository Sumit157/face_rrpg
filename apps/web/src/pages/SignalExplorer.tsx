import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Layout } from "../components/layout/Layout";
import { RPPGChart } from "../components/signal/RPPGChart";
import { SpectrumChart } from "../components/signal/SpectrumChart";
import { useAnalysesList } from "../hooks/useAnalysis";
import { analysisApi } from "../services/api";

/** Inspect the raw waveform/spectrum behind a completed analysis. Arrays
 * are stored as artifact files by the ML service
 * (storage/signals/{id}.json) and served by GET /api/analyses/:id/signal --
 * deliberately kept out of the primary result payload. */
export default function SignalExplorer() {
  const { data: analyses, isLoading } = useAnalysesList();
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [region, setRegion] = useState<string | null>(null);

  const completed = useMemo(
    () => (analyses ?? []).filter((a) => a.status === "completed"),
    [analyses]
  );

  const { data: signal, isFetching } = useQuery({
    queryKey: ["signal", selectedId],
    queryFn: () => analysisApi.signal(selectedId as string),
    enabled: !!selectedId,
  });

  const artifactRegions = Object.keys(signal?.artifacts ?? {});
  const activeRegion = region ?? artifactRegions[0] ?? null;
  const artifact = activeRegion ? signal?.artifacts?.[activeRegion] : undefined;

  return (
    <Layout>
      <h1 className="page-title mb-6">Signal Explorer</h1>

      <div className="panel p-4 mb-6 flex flex-wrap items-center gap-3">
        <label className="text-sm text-slate-400">Analysis</label>
        <select
          className="bg-surface-800 text-slate-200 text-sm rounded px-3 py-1.5"
          value={selectedId ?? ""}
          onChange={(e) => {
            setSelectedId(e.target.value || null);
            setRegion(null);
          }}
        >
          <option value="">Select an analysis...</option>
          {completed.map((a) => (
            <option key={a.id} value={a.id}>
              {a.id.slice(0, 8)} - {a.prediction?.label ?? "completed"}
            </option>
          ))}
        </select>
        {isFetching && <span className="text-xs text-accent">Loading signal...</span>}
        {completed.length === 0 && !isLoading && (
          <span className="text-xs text-slate-500">No completed analyses yet.</span>
        )}
      </div>

      {!selectedId && (
        <p className="text-sm text-slate-500">
          Select a completed analysis to inspect its raw rPPG waveform and spectrum per region.
        </p>
      )}

      {selectedId && artifactRegions.length > 0 && (
        <>
          <div className="flex gap-2 mb-4">
            {artifactRegions.map((r) => (
              <button
                key={r}
                onClick={() => setRegion(r)}
                className={`px-3 py-1.5 rounded-lg text-sm transition-colors duration-200 ${
                  activeRegion === r
                    ? "bg-accent text-surface-950 font-medium shadow-glow"
                    : "bg-surface-800 text-slate-400 hover:text-slate-200 hover:bg-surface-700"
                }`}
              >
                {r.replace("_", " ")}
              </button>
            ))}
            {artifact?.bpm !== undefined && artifact?.peakHz !== undefined && (
              <span className="self-center text-xs text-slate-500 font-mono ml-2">
                peak {artifact.peakHz.toFixed(2)} Hz ({artifact.bpm.toFixed(0)} bpm)
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="panel p-4">
              <RPPGChart
                timestamps={artifact?.timestamps ?? []}
                values={artifact?.waveform ?? []}
                title={`${activeRegion?.replace("_", " ")} waveform`}
              />
            </div>
            <div className="panel p-4">
              <SpectrumChart
                freqsHz={artifact?.freqsHz ?? []}
                power={artifact?.power ?? []}
                peakHz={artifact?.peakHz}
              />
            </div>
          </div>
        </>
      )}

      {selectedId && artifactRegions.length === 0 && !isFetching && (
        <p className="text-sm text-slate-500">
          No signal artifacts stored for this analysis (it may predate artifact saving, or no
          region produced a usable signal).
        </p>
      )}
    </Layout>
  );
}
