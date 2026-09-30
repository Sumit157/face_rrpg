import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Layout } from "../components/layout/Layout";
import { VideoUploader } from "../components/video/VideoUploader";
import { useCreateAnalysis } from "../hooks/useAnalysis";
import { DEFAULT_ANALYSIS_PARAMS } from "../services/analysis";
import type { RPPGMethod, FaceRegion } from "@rppg/shared-types";

const METHODS: RPPGMethod[] = ["auto", "pos", "chrom", "green"];
const ALL_REGIONS: FaceRegion[] = ["forehead", "left_cheek", "right_cheek"];

export default function NewAnalysis() {
  const [videoId, setVideoId] = useState<string | null>(null);
  const [method, setMethod] = useState<RPPGMethod>(DEFAULT_ANALYSIS_PARAMS.method);
  const [regions, setRegions] = useState<FaceRegion[]>([...DEFAULT_ANALYSIS_PARAMS.regions]);
  const create = useCreateAnalysis();
  const navigate = useNavigate();

  const toggleRegion = (r: FaceRegion) =>
    setRegions((prev) => (prev.includes(r) ? prev.filter((x) => x !== r) : [...prev, r]));

  const handleStart = () => {
    if (!videoId) return;
    create.mutate(
      {
        videoId,
        method,
        regions,
        windowSeconds: DEFAULT_ANALYSIS_PARAMS.windowSeconds,
        stepSeconds: DEFAULT_ANALYSIS_PARAMS.stepSeconds,
      },
      { onSuccess: (res) => navigate(`/analysis/${res.id}`) }
    );
  };

  return (
    <Layout>
      <h1 className="page-title mb-6">New Analysis</h1>

      {!videoId ? (
        <VideoUploader onUploaded={setVideoId} />
      ) : (
        <div className="space-y-6">
          <div className="panel p-4">
            <p className="text-sm text-verdict-real">Video uploaded ({videoId.slice(0, 8)})</p>
          </div>

          <div className="panel p-4">
            <h3 className="text-sm text-slate-300 mb-3">rPPG Method</h3>
            <div className="flex gap-2">
              {METHODS.map((m) => (
                <button
                  key={m}
                  onClick={() => setMethod(m)}
                  className={`px-3 py-1.5 rounded-lg text-sm transition-colors duration-200 ${
                    method === m
                      ? "bg-accent text-surface-950 font-medium shadow-glow"
                      : "bg-surface-800 text-slate-400 hover:text-slate-200 hover:bg-surface-700"
                  }`}
                >
                  {m.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          <div className="panel p-4">
            <h3 className="text-sm text-slate-300 mb-3">Regions</h3>
            <div className="flex gap-2">
              {ALL_REGIONS.map((r) => (
                <button
                  key={r}
                  onClick={() => toggleRegion(r)}
                  className={`px-3 py-1.5 rounded-lg text-sm transition-colors duration-200 ${
                    regions.includes(r)
                      ? "bg-accent text-surface-950 font-medium shadow-glow"
                      : "bg-surface-800 text-slate-400 hover:text-slate-200 hover:bg-surface-700"
                  }`}
                >
                  {r.replace("_", " ")}
                </button>
              ))}
            </div>
          </div>

          <button
            onClick={handleStart}
            disabled={regions.length === 0 || create.isPending}
            className="btn-primary px-5 py-2.5"
          >
            {create.isPending ? "Starting..." : "Start Analysis"}
          </button>
        </div>
      )}
    </Layout>
  );
}
