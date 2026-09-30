export type AnalysisStatus =
  | "queued"
  | "validating"
  | "extracting_faces"
  | "extracting_signals"
  | "processing_signals"
  | "extracting_features"
  | "running_model"
  | "generating_report"
  | "completed"
  | "failed";

export type RPPGMethod = "green" | "chrom" | "pos" | "auto";

export type FaceRegion = "forehead" | "left_cheek" | "right_cheek";

export type AnalysisMode = "fast" | "standard" | "research";

export interface CreateAnalysisRequest {
  videoId: string;
  method: RPPGMethod;
  regions: FaceRegion[];
  windowSeconds: number;
  stepSeconds: number;
  mode?: AnalysisMode;
}

export interface CreateAnalysisResponse {
  id: string;
  status: AnalysisStatus;
}

export interface AnalysisProgressEvent {
  analysisId: string;
  stage: AnalysisStatus;
  progress: number; // 0-100
  message: string;
  /** Stage transitions with seconds-relative timestamps (live stage
   * durations), forwarded from the ML progress registry. */
  history?: { stage: string; at: number }[];
}
