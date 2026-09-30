import { useQuery } from "@tanstack/react-query";
import { Layout } from "../components/layout/Layout";
import { StatCard } from "../components/common/StatCard";
import { api } from "../services/api";

interface ModelInfo {
  id: string;
  loaded: boolean;
  metrics: {
    auc?: number;
    accuracy?: number;
    f1?: number;
    classificationReport?: Record<string, unknown>;
  } | null;
}

const pct = (v: number | undefined) => (typeof v === "number" ? v.toFixed(3) : "--");

/** Shows the active model's offline evaluation metrics (written by
 * scripts/evaluate.py to experiments/results/evaluation_metrics.json and
 * served by GET /api/models), plus the live load state reported by the ML
 * service. */
export default function ModelEvaluation() {
  const { data, isLoading } = useQuery({
    queryKey: ["models"],
    queryFn: async () => {
      const { data } = await api.get<ModelInfo[]>("/models");
      return data;
    },
  });

  const model = data?.[0];
  const metrics = model?.metrics ?? null;

  return (
    <Layout>
      <h1 className="page-title mb-6">Model Evaluation</h1>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard label="AUC" value={pct(metrics?.auc)} />
        <StatCard label="Accuracy" value={pct(metrics?.accuracy)} />
        <StatCard label="F1" value={pct(metrics?.f1)} />
        <StatCard
          label="Active model"
          value={isLoading ? "..." : model?.loaded ? "baseline (trained)" : "heuristic fallback"}
        />
      </div>

      <div className="panel p-6">
        {metrics ? (
          <div className="text-sm text-slate-300 space-y-2">
            <p>
              Metrics below come from{" "}
              <code className="text-accent">experiments/results/evaluation_metrics.json</code>,
              produced by <code className="text-accent">scripts/evaluate.py</code> on a held-out
              feature table.
            </p>
            <pre className="bg-surface-900 rounded p-3 text-xs overflow-x-auto text-slate-400">
              {JSON.stringify(metrics.classificationReport ?? metrics, null, 2)}
            </pre>
          </div>
        ) : (
          <p className="text-sm text-slate-400">
            No evaluation results yet. Train a baseline with{" "}
            <code className="text-accent">scripts/train_baseline.py</code>, evaluate it with{" "}
            <code className="text-accent">scripts/evaluate.py</code>, and this page will pick up
            the metrics automatically. Until then every prediction comes from the clearly-labelled
            heuristic fallback (see <code className="text-accent">docs/limitations.md</code>).
          </p>
        )}
      </div>
    </Layout>
  );
}
