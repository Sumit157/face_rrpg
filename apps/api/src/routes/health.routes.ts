import { Router } from "express";
import { pythonService } from "../services/python.service";

export const healthRouter = Router();

healthRouter.get("/", async (_req, res) => {
  let mlStatus: unknown = "unreachable";
  try {
    mlStatus = await pythonService.health();
  } catch {
    // ML service being down shouldn't crash the health check itself
  }
  res.json({ status: "ok", service: "rppg-api", mlService: mlStatus });
});
