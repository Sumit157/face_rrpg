/** Pipeline visualization trace -- storage/visuals/{analysisId}.json,
 * served by GET /api/analyses/:id/visuals. Everything is normalised so
 * the frontend can draw it directly onto the video element. */

export interface VisualFrame {
  /** Frame index in the source video. */
  f: number;
  /** Timestamp in seconds. */
  t: number;
  detected: boolean;
  method: "mediapipe" | "haar" | "none";
  /** [x, y, w, h] normalised to 0-1; null when no face was found. */
  bbox: [number, number, number, number] | null;
  /** Region name -> polygon points [[x, y], ...] normalised to 0-1. */
  roi: Record<string, [number, number][]> | null;
  /** True when CLAHE enhancement was applied to aid detection (dark frame). */
  enhanced: boolean;
}

export interface VisualStage {
  stage: string;
  startS: number;
  endS: number;
  durationMs: number;
  message: string | null;
}

export interface VisualExtraction {
  fps: number;
  framesProcessed: number;
  durationSeconds: number;
  /** The trace samples every Nth frame; all frames are still analysed. */
  sampleStride: number;
  detectionRate: number;
  detector: "mediapipe" | "haar";
  haarFallbackFrames: number;
  capSeconds: number;
  enhancedFrames: number;
}

export interface VisualWindow {
  index: number;
  startS: number;
  endS: number;
  bpm: number;
  quality: number | null;
}

export interface VisualRegionWindows {
  region: string;
  windows: VisualWindow[];
}

export interface AnalysisVisuals {
  analysisId: string;
  stages: VisualStage[];
  extraction: VisualExtraction;
  frames: VisualFrame[];
  windows: VisualRegionWindows[];
  result: {
    method: string;
    regions: string[];
    windowSeconds: number;
    stepSeconds: number;
  };
}
