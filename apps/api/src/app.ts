import express, { type Express } from "express";
import cors from "cors";
import helmet from "helmet";
import pinoHttp from "pino-http";
import { config } from "./config";
import { logger } from "./config/logger";
import { healthRouter } from "./routes/health.routes";
import { videoRouter } from "./routes/video.routes";
import { analysisRouter } from "./routes/analysis.routes";
import { modelRouter } from "./routes/model.routes";
import { reportRouter } from "./routes/report.routes";
import { notFoundHandler, errorHandler } from "./middleware/error.middleware";

export function createApp(): Express {
  const app = express();

  app.use(helmet());
  app.use(cors({ origin: config.corsOrigin }));
  // pino-http BEFORE body parsing so malformed requests are logged too.
  app.use(pinoHttp({ logger }));
  app.use(express.json({ limit: "2mb" }));

  // Deliberately NOT serving the storage directory as static files --
  // uploaded videos must never be directly web-accessible (spec section 22).
  app.use("/api/health", healthRouter);
  app.use("/api/videos", videoRouter);
  app.use("/api/analyses", analysisRouter);
  app.use("/api/models", modelRouter);
  app.use("/api/reports", reportRouter);

  app.use(notFoundHandler);
  app.use(errorHandler);

  return app;
}
