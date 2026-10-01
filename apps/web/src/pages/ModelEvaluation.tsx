import { useQuery } from "@tanstack/react-query";
import { Layout } from "../components/layout/Layout";
import { StatCard } from "../components/common/StatCard";
import { api } from "../services/api";

interface ModelInfo {
  id: string;
  loaded: boolean;
  modelVersion: string | null;
  validationStatus: string;
  metrics: {
    auc?: number;
    accuracy?: number;
    balancedAccuracy?: number;
    macroF1?: number;
    realRecall?: number;
    fakeRecall?: number;
    f1?: number;
    classificationReport?: Record<string, unknown>;
    evaluationProtocol?: string;
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
        <StatCard label="Balanced accuracy" value={pct(metrics?.balancedAccuracy)} />
        <StatCard
          label="Active model"
          value={isLoading ? "..." : model?.loaded ? "optional offline model" : "rPPG cue fusion"}
        />
      </div>

      {model?.validationStatus === "synthetic_unvalidated" && (
        <div className="panel p-4 mb-4 border border-verdict-uncertain/50">
          <h2 className="text-sm font-semibold text-verdict-uncertain mb-1">
            Optional synthetic model: excluded from live predictions
          </h2>
          <p className="text-xs leading-relaxed text-slate-400">
            The loaded {model.modelVersion ?? "Random Forest"} learned from generated rPPG traces.
            Its synthetic holdout score does not estimate accuracy on genuine or deepfake videos.
            The live analysis uses model-free rPPG cue fusion. This artifact is visible here only
            for offline evaluation; its synthetic holdout score does not estimate real-video
            accuracy.
          </p>
        </div>
      )}

      <div className="panel p-6">
        {metrics ? (
          <div className="text-sm text-slate-300 space-y-2">
            <p>
              Metrics below come from{" "}
              <code className="text-accent">experiments/results/evaluation_metrics.json</code>,
              produced by <code className="text-accent">scripts/evaluate.py</code> on a subject-
              disjoint held-out feature table ({metrics.evaluationProtocol ?? "protocol unspecified"}).
            </p>
            <pre className="bg-surface-900 rounded p-3 text-xs overflow-x-auto text-slate-400">
              {JSON.stringify(metrics.classificationReport ?? metrics, null, 2)}
            </pre>
          </div>
        ) : (
          <p className="text-sm text-slate-400">
            No labeled video evaluation results are available. Create subject-disjoint train and
            test feature tables from verified real and fake videos, train with{" "}
            <code className="text-accent">scripts/train_baseline.py</code>, and evaluate with{" "}
            <code className="text-accent">scripts/evaluate.py</code>. Synthetic-signal scores are
            not shown as real-video accuracy.
          </p>
        )}
      </div>
    </Layout>
  );
}
