import { Router } from "express";
import { asyncHandler } from "../middleware/async";
import { reportService } from "../services/report.service";

export const reportRouter = Router();

reportRouter.get(
  "/:analysisId",
  asyncHandler(async (req, res) => {
    res.json(await reportService.generate(req.params.analysisId));
  })
);
