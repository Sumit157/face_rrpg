import { analysisService } from "./analysis.service";
import { ApiError } from "../middleware/error.middleware";

/** Builds the downloadable analysis report (spec item 15: "Downloadable
 * analysis report"). Returns structured JSON the frontend can render and
 * download; PDF/HTML rendering is a documented extension point. */
export class ReportService {
  async generate(analysisId: string) {
    const analysis = await analysisService.getById(analysisId);
    if (analysis.status !== "completed") {
      throw new ApiError(409, "Report is only available once the analysis has completed");
    }
    return {
      analysisId,
      generatedAt: new Date().toISOString(),
      result: analysis,
    };
  }
}

export const reportService = new ReportService();
