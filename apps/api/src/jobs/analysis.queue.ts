import { prisma } from "../db/client";
import { pythonService } from "../services/python.service";
import { emitAnalysisEvent } from "../websocket/analysis.gateway";
import { logger } from "../config/logger";
import type { RPPGMethod, FaceRegion, AnalysisProgressEvent } from "@rppg/shared-types";

export interface AnalysisJobData {
  analysisId: string;
  videoPath: string;
  method: string;
  regions: string[];
  windowSeconds: number;
  stepSeconds: number;
}

const MAX_ATTEMPTS = 3;
const BASE_BACKOFF_MS = 5000;
/** Video analysis is CPU-heavy (OpenCV + MediaPipe in the ML service). */
const CONCURRENCY = 2;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * In-process analysis queue.
 *
 * Previously this was a BullMQ queue backed by a separate worker process.
 * That architecture could never deliver WebSocket progress events: the
 * worker had no Socket.IO instance (see docs/architecture.md), and it
 * required a Redis server that local development doesn't have. Running the
 * job inside the API process keeps retries/backoff and bounded concurrency
 * while making `emitAnalysisEvent` a direct, working call.
 *
 * Retry policy mirrors the old BullMQ config: 3 attempts, exponential
 * backoff starting at 5s.
 */
class AnalysisQueue {
  private readonly pending: AnalysisJobData[] = [];
  private running = 0;

  add(data: AnalysisJobData): void {
    this.pending.push(data);
    this.pump();
  }

  private pump(): void {
    while (this.running < CONCURRENCY && this.pending.length > 0) {
      const job = this.pending.shift() as AnalysisJobData;
      this.running += 1;
      void this.runWithRetry(job, 1)
        .catch((err) => {
          // runWithRetry already logged and persisted the final failure.
          logger.error({ err, analysisId: job.analysisId }, "Job failed after retries");
        })
        .finally(() => {
          this.running -= 1;
          this.pump();
        });
    }
  }

  private async runWithRetry(job: AnalysisJobData, attempt: number): Promise<void> {
    try {
      await processJob(job);
    } catch (err) {
      if (attempt < MAX_ATTEMPTS) {
        const delay = BASE_BACKOFF_MS * 2 ** (attempt - 1);
        logger.warn(
          { analysisId: job.analysisId, attempt, delay, err },
          "Analysis attempt failed, retrying"
        );
        await setStatus(job.analysisId, "queued", 5, `Retrying (attempt ${attempt + 1}/${MAX_ATTEMPTS})`);
        await sleep(delay);
        return this.runWithRetry(job, attempt + 1);
      }

      // Permanent failure: record it clearly (spec section 14, step 8).
      const message = err instanceof Error ? err.message : String(err);
      await updateQuietly(job.analysisId, { status: "failed", errorMessage: message });
      emitAnalysisEvent({ analysisId: job.analysisId, stage: "failed", progress: 100, message });
      throw err;
    }
  }
}

/** Flow (spec section 14): validate -> call Python -> save -> emit. */
async function processJob(job: AnalysisJobData): Promise<void> {
  const { analysisId, videoPath, method, regions, windowSeconds, stepSeconds } = job;

  await setStatus(analysisId, "validating", 5, "Validating job input");

  // The ML service reports its real stage/percentage while the single
  // blocking call runs; forward each update to the DB + WebSocket.
  const onProgress = async (
    event: Omit<AnalysisProgressEvent, "analysisId">
  ): Promise<void> => {
    emitAnalysisEvent({ analysisId, ...event });
    // Persist in-flight stages only; terminal status is written by
    // processJob itself once the result is safely stored.
    if (!["queued", "completed", "failed"].includes(event.stage)) {
      await updateQuietly(analysisId, { status: event.stage });
    }
  };

  const result = await pythonService.analyse(
    {
      analysisId,
      videoPath,
      method: method as RPPGMethod,
      regions: regions as FaceRegion[],
      windowSeconds,
      stepSeconds,
    },
    (event) => {
      // Fire-and-forget: ordering is preserved by the single-threaded
      // event loop, and a failed status write must not break the job.
      void onProgress(event).catch((err) => {
        logger.debug({ err, analysisId }, "progress update failed");
      });
    }
  );

  const failed = result?.prediction?.label === "FAILED";
  await prisma.analysis.update({
    where: { id: analysisId },
    data: {
      status: failed ? "failed" : "completed",
      result: JSON.stringify(result),
      errorMessage: failed ? (result?.prediction?.reason ?? "Analysis failed") : null,
      completedAt: new Date(),
    },
  });

  if (failed) {
    emitAnalysisEvent({
      analysisId,
      stage: "failed",
      progress: 100,
      message: result?.prediction?.reason ?? "Analysis failed",
    });
  } else {
    emitAnalysisEvent({ analysisId, stage: "completed", progress: 100, message: "Analysis complete" });
  }
}

async function setStatus(analysisId: string, stage: string, progress: number, message: string) {
  await updateQuietly(analysisId, { status: stage });
  emitAnalysisEvent({ analysisId, stage: stage as never, progress, message });
}

/** Update, tolerating the analysis having been deleted mid-flight. */
async function updateQuietly(analysisId: string, data: Record<string, unknown>): Promise<void> {
  try {
    await prisma.analysis.update({ where: { id: analysisId }, data });
  } catch (err) {
    const code = (err as { code?: string }).code;
    if (code !== "P2025") throw err; // P2025 = record already deleted
  }
}

export const analysisQueue = new AnalysisQueue();
