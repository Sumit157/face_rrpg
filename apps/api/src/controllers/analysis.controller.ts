import fs from "node:fs";
import path from "node:path";
import type { Request, Response } from "express";
import { config } from "../config";
import { analysisService } from "../services/analysis.service";
import { reportService } from "../services/report.service";
import { ApiError } from "../middleware/error.middleware";

export async function createAnalysisHandler(req: Request, res: Response): Promise<void> {
  const result = await analysisService.create(req.body);
  res.status(202).json(result);
}

export async function listAnalysesHandler(_req: Request, res: Response): Promise<void> {
  res.json(await analysisService.list());
}

export async function getAnalysisHandler(req: Request, res: Response): Promise<void> {
  res.json(await analysisService.getById(req.params.id));
}

export async function deleteAnalysisHandler(req: Request, res: Response): Promise<void> {
  await analysisService.delete(req.params.id);
  res.status(204).send();
}

export async function getAnalysisSignalHandler(req: Request, res: Response): Promise<void> {
  const analysis = await analysisService.getById(req.params.id);
  if (!analysis.bpm) throw new ApiError(409, "Analysis has no signal data yet");
  // Large per-region arrays live in storage/signals/{id}.json (written by
  // the ML service), not in the result payload -- load them back here.
  const file = path.join(config.storageDir, "signals", `${req.params.id}.json`);
  let artifacts: Record<string, unknown> = {};
  if (fs.existsSync(file)) {
    artifacts = JSON.parse(fs.readFileSync(file, "utf8"));
  }
  res.json({
    bpm: analysis.bpm,
    consistency: analysis.consistency,
    artifacts,
  });
}

export async function getAnalysisFeaturesHandler(req: Request, res: Response): Promise<void> {
  const analysis = await analysisService.getById(req.params.id);
  if (!analysis.featureVector) throw new ApiError(409, "Analysis has no feature data yet");
  res.json(analysis.featureVector);
}

export async function getAnalysisReportHandler(req: Request, res: Response): Promise<void> {
  res.json(await reportService.generate(req.params.id));
}

export async function getAnalysisVisualsHandler(req: Request, res: Response): Promise<void> {
  await analysisService.getById(req.params.id); // 404s for unknown analyses
  // Pipeline-tab trace written by the ML service (stage timeline, frame
  // detections, window segmentation) -- convention path, like the signals.
  const file = path.join(config.storageDir, "visuals", `${req.params.id}.json`);
  if (!fs.existsSync(file)) {
    throw new ApiError(404, "No visualization trace for this analysis");
  }
  res.type("application/json").send(fs.readFileSync(file, "utf8"));
}
