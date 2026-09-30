import { Link } from "react-router-dom";
import { Layout } from "../components/layout/Layout";
import { StatCard } from "../components/common/StatCard";
import { useAnalysesList } from "../hooks/useAnalysis";
import type { AnalysisRecord } from "@rppg/shared-types";

const STATUS_BADGE: Record<string, string> = {
  completed: "text-verdict-real border-verdict-real",
  failed: "text-verdict-fake border-verdict-fake",
};

export default function Dashboard() {
  const { data: analyses, isLoading } = useAnalysesList();

  const total = analyses?.length ?? 0;
  const completed = analyses?.filter((a) => a.status === "completed").length ?? 0;
  const running =
    analyses?.filter((a) => a.status !== "completed" && a.status !== "failed").length ?? 0;
  const fake = analyses?.filter((a) => a.prediction?.label === "FAKE").length ?? 0;

  return (
    <Layout>
      <div className="flex justify-between items-center mb-6">
        <h1 className="page-title">Dashboard</h1>
        <Link to="/analysis/new" className="btn-primary">
          New Analysis
        </Link>
      </div>

      <div className="grid grid-cols-4 gap-4 mb-8">
        <StatCard label="Total analyses" value={total} />
        <StatCard label="Completed" value={completed} accent="text-verdict-real" />
        <StatCard label="In progress" value={running} accent="text-accent" />
        <StatCard label="Flagged FAKE" value={fake} accent="text-verdict-fake" />
      </div>

      <div className="panel p-4">
        <h2 className="text-sm text-slate-300 mb-3">Recent analyses</h2>
        {isLoading && (
          <div className="space-y-2" aria-busy="true">
            <div className="skeleton h-8 w-full" />
            <div className="skeleton h-8 w-5/6" />
            <div className="skeleton h-8 w-4/6" />
          </div>
        )}
        {!isLoading && total === 0 && (
          <p className="text-sm text-slate-500">
            No analyses yet.{" "}
            <Link to="/analysis/new" className="text-accent">
              Start one
            </Link>
            .
          </p>
        )}
        <ul className="divide-y divide-surface-700">
          {(analyses ?? []).map((a: AnalysisRecord) => (
            <li key={a.id} className="py-2 flex justify-between items-center text-sm gap-3">
              <Link to={`/analysis/${a.id}`} className="text-slate-300 hover:text-accent font-mono">
                {a.id.slice(0, 8)}
              </Link>
              <span className="text-slate-500 text-xs flex-1 text-right hidden sm:block">
                {a.method.toUpperCase()} &middot; {new Date(a.createdAt).toLocaleString()}
              </span>
              <span
                className={`text-[10px] uppercase tracking-wide px-2 py-0.5 rounded border ${
                  STATUS_BADGE[a.status] ?? "text-slate-400 border-surface-700"
                }`}
              >
                {a.prediction?.label ?? a.status.replace("_", " ")}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </Layout>
  );
}
