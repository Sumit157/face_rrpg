import { Router } from "express";
import { z } from "zod";
import { validateBody } from "../middleware/validate.middleware";
import { asyncHandler } from "../middleware/async";
import {
  createAnalysisHandler,
  listAnalysesHandler,
  getAnalysisHandler,
  deleteAnalysisHandler,
  getAnalysisSignalHandler,
  getAnalysisFeaturesHandler,
  getAnalysisReportHandler,
  getAnalysisVisualsHandler,
} from "../controllers/analysis.controller";

export const analysisRouter = Router();

const createAnalysisSchema = z.object({
  videoId: z.string().uuid(),
  method: z.enum(["green", "chrom", "pos", "auto"]).default("pos"),
  regions: z
    .array(z.enum(["forehead", "left_cheek", "right_cheek"]))
    .min(1)
    .default(["forehead", "left_cheek", "right_cheek"]),
  windowSeconds: z.number().min(4).max(30).default(12),
  stepSeconds: z.number().min(1).max(15).default(3),
});

analysisRouter.post("/", validateBody(createAnalysisSchema), asyncHandler(createAnalysisHandler));
analysisRouter.get("/", asyncHandler(listAnalysesHandler));
analysisRouter.get("/:id", asyncHandler(getAnalysisHandler));
analysisRouter.delete("/:id", asyncHandler(deleteAnalysisHandler));
analysisRouter.get("/:id/signal", asyncHandler(getAnalysisSignalHandler));
analysisRouter.get("/:id/features", asyncHandler(getAnalysisFeaturesHandler));
analysisRouter.get("/:id/report", asyncHandler(getAnalysisReportHandler));
analysisRouter.get("/:id/visuals", asyncHandler(getAnalysisVisualsHandler));
