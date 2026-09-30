import { Link } from "react-router-dom";
import { Layout } from "../components/layout/Layout";
import { useAnalysesList } from "../hooks/useAnalysis";
import type { AnalysisStatus } from "@rppg/shared-types";

const IN_FLIGHT: AnalysisStatus[] = [
  "queued",
  "validating",
  "extracting_faces",
  "extracting_signals",
  "processing_signals",
  "extracting_features",
  "running_model",
  "generating_report",
];

/** Live monitor for in-flight analyses: polls the analysis list and shows
 * current stage progress in real time (WebSocket events additionally
 * drive the per-analysis page).
 *
 * Real-time *webcam* analysis is a planned Phase 2 extension (see
 * docs/architecture.md) -- it needs a streaming ingest path (WebRTC ->
 * rolling segments -> the same pipeline); the pipeline itself already
 * processes arbitrary video sources. */
export default function LiveAnalysis() {
  const { data: analyses, isLoading } = useAnalysesList({ refetchInterval: 2000 });

  const running = (analyses ?? []).filter((a) => IN_FLIGHT.includes(a.status));
  const recent = (analyses ?? [])
    .filter((a) => a.status === "completed" || a.status === "failed")
    .slice(0, 5);

  return (
    <Layout>
      <div className="flex items-center gap-3 mb-6">
        <h1 className="page-title">Live Analysis</h1>
        <span className="flex items-center gap-1.5 text-xs text-slate-500">
          <span className="inline-block h-2 w-2 rounded-full bg-verdict-real animate-pulse" />
          auto-refresh 2s
        </span>
      </div>

      <div className="panel p-4 mb-6">
        <h2 className="text-sm text-slate-300 mb-3">In progress</h2>
        {isLoading && (
          <div className="space-y-2" aria-busy="true">
            <div className="skeleton h-6 w-2/3" />
            <div className="skeleton h-6 w-1/2" />
          </div>
        )}
        {!isLoading && running.length === 0 && (
          <p className="text-sm text-slate-500">Nothing running right now.</p>
        )}
        <ul className="space-y-3">
          {running.map((a) => (
            <li key={a.id} className="flex items-center justify-between gap-3">
              <Link
                to={`/analysis/${a.id}`}
                className="text-sm text-slate-200 font-mono hover:text-accent"
              >
                {a.id.slice(0, 8)}
              </Link>
              <span className="flex-1 text-xs text-accent uppercase tracking-wide">
                {a.status.replace("_", " ")}
                <span className="inline-block h-1.5 w-1.5 rounded-full bg-accent animate-pulse ml-2 align-middle" />
              </span>
              <span className="text-xs text-slate-500">{a.method.toUpperCase()}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="panel p-4">
        <h2 className="text-sm text-slate-300 mb-3">Recently finished</h2>
        {recent.length === 0 && <p className="text-sm text-slate-500">Nothing yet.</p>}
        <ul className="divide-y divide-surface-700">
          {recent.map((a) => (
            <li key={a.id} className="py-2 flex justify-between items-center text-sm">
              <Link
                to={`/analysis/${a.id}`}
                className="text-slate-300 font-mono hover:text-accent"
              >
                {a.id.slice(0, 8)}
              </Link>
              <span
                className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded border ${
                  a.status === "completed"
                    ? "text-verdict-real border-verdict-real"
                    : "text-verdict-fake border-verdict-fake"
                }`}
              >
                {a.prediction?.label ?? a.status}
              </span>
            </li>
          ))}
        </ul>
      </div>

      <p className="text-xs text-slate-600 mt-6">
        Real-time webcam analysis is a planned Phase 2 extension; the current pipeline works on
        uploaded video files (see docs/architecture.md).
      </p>
    </Layout>
  );
}
