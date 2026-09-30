import { useState } from "react";
import { Link } from "react-router-dom";
import { Layout } from "../components/layout/Layout";
import { useAnalysesList } from "../hooks/useAnalysis";
import { analysisApi } from "../services/api";

/** Lists completed analyses and lets the user download the structured
 * report (GET /api/analyses/:id/report) as JSON. */
export default function Reports() {
  const { data: analyses } = useAnalysesList();
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const completed = analyses?.filter((a) => a.status === "completed") ?? [];

  const download = async (id: string) => {
    setDownloadingId(id);
    setError(null);
    try {
      const report = await analysisApi.report(id);
      const blob = new Blob([JSON.stringify(report, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `rppg-report-${id.slice(0, 8)}.json`;
      link.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Download failed");
    } finally {
      setDownloadingId(null);
    }
  };

  return (
    <Layout>
      <h1 className="page-title mb-6">Reports</h1>
      {error && <p className="text-verdict-fake text-sm mb-3">{error}</p>}
      <div className="panel divide-y divide-surface-700">
        {completed.length === 0 && (
          <p className="p-4 text-sm text-slate-500">No completed analyses yet.</p>
        )}
        {completed.map((a) => (
          <div key={a.id} className="p-4 flex justify-between items-center gap-3">
            <div className="min-w-0">
              <div className="text-sm text-slate-200 font-mono">{a.id.slice(0, 8)}</div>
              <div className="text-xs text-slate-500">
                {a.prediction?.label ?? "completed"} &middot;{" "}
                {(a.completedAt ? new Date(a.completedAt) : new Date(a.createdAt)).toLocaleString()}
              </div>
            </div>
            <div className="flex items-center gap-4 shrink-0">
              <button
                onClick={() => download(a.id)}
                disabled={downloadingId === a.id}
                className="text-accent text-sm hover:underline disabled:opacity-50"
              >
                {downloadingId === a.id ? "Preparing..." : "Download JSON"}
              </button>
              <Link to={`/analysis/${a.id}`} className="text-slate-300 text-sm hover:text-accent">
                View
              </Link>
            </div>
          </div>
        ))}
      </div>
    </Layout>
  );
}
