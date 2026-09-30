import type { AnalysisStatus } from "./analysis";

export type Prediction = "REAL" | "FAKE" | "UNCERTAIN" | "FAILED";

export interface BPMSummary {
  mean: number | null;
  std: number | null;
}

export interface ConsistencySummary {
  meanCorrelation: number | null;
  frequencyAgreement: number | null;
  pairwiseCorrelations?: Record<string, number>;
}

export interface PredictionResult {
  label: Prediction;
  probabilityReal: number;
  probabilityFake: number;
  modelVersion: string;
  isHeuristicFallback: boolean;
}

export interface CalibratedConfidence {
  rawProbability: number;
  signalQuality: number;
  adjustedConfidence: number;
}

export interface AnalysisResult {
  id: string;
  videoMetadata: Record<string, unknown>;
  faceQuality: Record<string, unknown>;
  signalQuality: Record<string, unknown>;
  method: string;
  regions: string[];
  windowSeconds: number;
  stepSeconds: number;
  bpm: Record<string, BPMSummary> & { overall: BPMSummary };
  consistency: ConsistencySummary;
  featureVector: Record<string, number>;
  prediction: PredictionResult;
  confidence: CalibratedConfidence;
  evidence: string[];
  warnings: string[];
  /** Raw rPPG waveforms/spectra per region, keyed by region (when the ML
   * service persisted them for the signal explorer). */
  artifacts?: Record<string, unknown>;
}

/** What `GET /api/analyses[/:id]` actually returns: the DB row's status
 * metadata plus every field of the ML payload flattened to the top level
 * (only present once the analysis completed). Check `status` for progress,
 * `prediction` for the verdict. */
export interface AnalysisRecord extends Partial<AnalysisResult> {
  id: string;
  videoId: string;
  status: AnalysisStatus;
  errorMessage: string | null;
  // Row columns -- always present, unlike the ML payload fields above.
  method: string;
  regions: string[];
  windowSeconds: number;
  stepSeconds: number;
  createdAt: string;
  completedAt: string | null;
}
