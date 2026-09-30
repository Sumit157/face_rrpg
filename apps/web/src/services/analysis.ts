import { analysisApi } from "./api";
export { analysisApi };

export const DEFAULT_ANALYSIS_PARAMS = {
  method: "pos" as const,
  regions: ["forehead", "left_cheek", "right_cheek"] as const,
  windowSeconds: 12,
  stepSeconds: 3,
};
