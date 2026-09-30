import axios from "axios";
import type {
  CreateAnalysisRequest,
  CreateAnalysisResponse,
  AnalysisRecord,
  AnalysisResult,
  AnalysisVisuals,
  Video,
} from "@rppg/shared-types";

export const api = axios.create({ baseURL: "/api" });

export interface SignalResponse {
  bpm: AnalysisResult["bpm"] | undefined;
  consistency: AnalysisResult["consistency"] | undefined;
  artifacts: Record<string, SignalArtifact>;
}

/** Per-region waveform + spectrum arrays loaded from the artifact file the
 * ML service wrote to storage/signals/{id}.json. */
export interface SignalArtifact {
  timestamps?: number[];
  waveform?: number[];
  freqsHz?: number[];
  power?: number[];
  peakHz?: number;
  bpm?: number;
}

export const videoApi = {
  upload: async (file: File): Promise<Video> => {
    const form = new FormData();
    form.append("video", file);
    const { data } = await api.post("/videos", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    return data;
  },
  get: async (id: string): Promise<Video> => {
    const { data } = await api.get(`/videos/${id}`);
    return data;
  },
};

export const analysisApi = {
  create: async (req: CreateAnalysisRequest): Promise<CreateAnalysisResponse> => {
    const { data } = await api.post("/analyses", req);
    return data;
  },
  list: async (): Promise<AnalysisRecord[]> => {
    const { data } = await api.get("/analyses");
    return data;
  },
  get: async (id: string): Promise<AnalysisRecord> => {
    const { data } = await api.get(`/analyses/${id}`);
    return data;
  },
  remove: async (id: string): Promise<void> => {
    await api.delete(`/analyses/${id}`);
  },
  signal: async (id: string): Promise<SignalResponse> => {
    const { data } = await api.get(`/analyses/${id}/signal`);
    return data;
  },
  features: async (id: string): Promise<Record<string, number>> => {
    const { data } = await api.get(`/analyses/${id}/features`);
    return data;
  },
  report: async (id: string): Promise<Record<string, unknown>> => {
    const { data } = await api.get(`/analyses/${id}/report`);
    return data;
  },
  /** Pipeline-tab trace; 404 when the analysis predates tracing (or failed
   * before producing one) -- callers should treat that as "unavailable". */
  visuals: async (id: string): Promise<AnalysisVisuals> => {
    const { data } = await api.get(`/analyses/${id}/visuals`);
    return data;
  },
};

/** Explicit route -- storage is never served statically. */
export const streamUrl = (videoId: string) => `/api/videos/${videoId}/stream`;
