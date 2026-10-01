import fs from "node:fs";
import path from "node:path";
import { Router } from "express";
import { pythonService } from "../services/python.service";
import { config } from "../config";
import { ApiError } from "../middleware/error.middleware";

export const modelRouter = Router();

/** Offline evaluation metrics written by scripts/evaluate.py
 * (experiments/results/evaluation_metrics.json), when they exist. */
function readEvaluationMetrics(): Record<string, unknown> | null {
  const file = path.join(config.repoRoot, "experiments", "results", "evaluation_metrics.json");
  try {
    return JSON.parse(fs.readFileSync(file, "utf-8"));
  } catch {
    return null;
  }
}

async function modelInfo(id: string) {
  let loaded = false;
  let modelVersion: string | null = null;
  let validationStatus = "ml_service_unavailable";
  try {
    const health = await pythonService.health();
    loaded = health.modelLoaded;
    modelVersion = health.modelVersion ?? null;
    validationStatus = health.validationStatus ?? "unverified";
  } catch {
    // ML service down -- report loaded:false rather than 500 on a read route
  }
  return {
    id,
    loaded,
    modelVersion,
    validationStatus,
    metrics: id === "baseline" ? readEvaluationMetrics() : null,
  };
}

modelRouter.get("/", async (_req, res) => {
  res.json([await modelInfo("baseline")]);
});

modelRouter.get("/:id", async (req, res) => {
  if (req.params.id !== "baseline") {
    throw new ApiError(404, `Model ${req.params.id} not found`);
  }
  res.json(await modelInfo(req.params.id));
});
