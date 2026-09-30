import axios from "axios";
import { config } from "../config";
import type { RPPGMethod, FaceRegion, AnalysisProgressEvent } from "@rppg/shared-types";

export interface AnalyseVideoParams {
  analysisId: string;
  videoPath: string;
  method: RPPGMethod;
  regions: FaceRegion[];
  windowSeconds: number;
  stepSeconds: number;
}

const PROGRESS_POLL_MS = 700;

/** Thin client for the internal FastAPI ML service. This is the ONLY
 * place in the Node codebase that talks to Python -- keeps the
 * OpenCV/MediaPipe/ML logic entirely out of Node, per the spec's
 * architecture rules. */
export class PythonService {
  private readonly baseUrl: string;

  constructor(baseUrl: string = config.pythonServiceUrl) {
    this.baseUrl = baseUrl;
  }

  async health(): Promise<{ status: string; modelLoaded: boolean }> {
    const { data } = await axios.get(`${this.baseUrl}/health`, { timeout: 5000 });
    return data;
  }

  /** Run the full pipeline. While the (long, blocking) POST is in flight,
   * poll the ML service's progress endpoint and forward real stage updates
   * through `onProgress` -- this is what drives the browser's progress bar
   * over WebSocket. */
  async analyse(
    params: AnalyseVideoParams,
    onProgress?: (event: Omit<AnalysisProgressEvent, "analysisId">) => void
  ) {
    let timer: NodeJS.Timeout | null = null;
    if (onProgress) {
      timer = setInterval(() => {
        void axios
          .get(`${this.baseUrl}/v1/analyse/${params.analysisId}/progress`, { timeout: 3000 })
          .then(({ data }) => {
            if (data?.stage) {
              onProgress({
                stage: data.stage,
                progress: data.progress ?? 0,
                message: data.message ?? "",
                // Stage-transition timeline (seconds-relative), used by the
                // Pipeline tab's live stage-duration view.
                ...(Array.isArray(data.history) ? { history: data.history } : {}),
              });
            }
          })
          .catch(() => {
            // Transient poll failure is fine; the POST itself carries the
            // authoritative result.
          });
      }, PROGRESS_POLL_MS);
    }

    try {
      const { data } = await axios.post(
        `${this.baseUrl}/v1/analyse`,
        {
          analysis_id: params.analysisId,
          video_path: params.videoPath,
          method: params.method,
          regions: params.regions,
          window_seconds: params.windowSeconds,
          step_seconds: params.stepSeconds,
        },
        { timeout: 10 * 60 * 1000 } // analysis can legitimately take minutes
      );
      return data;
    } finally {
      if (timer) clearInterval(timer);
    }
  }
}

export const pythonService = new PythonService();
