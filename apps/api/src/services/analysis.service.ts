import { prisma } from "../db/client";
import { analysisQueue } from "../jobs/analysis.queue";
import { videoService } from "./video.service";
import { serializeAnalysis } from "./serializers";
import { ApiError } from "../middleware/error.middleware";
import type { CreateAnalysisRequest, AnalysisRecord } from "@rppg/shared-types";

export class AnalysisService {
  async create(request: CreateAnalysisRequest) {
    const video = await videoService.getById(request.videoId);
    if (!video) {
      throw new ApiError(404, `Video ${request.videoId} not found`);
    }

    const analysis = await prisma.analysis.create({
      data: {
        videoId: request.videoId,
        method: request.method,
        // SQLite has no array columns -- regions are stored JSON-encoded.
        regions: JSON.stringify(request.regions),
        windowSeconds: request.windowSeconds,
        stepSeconds: request.stepSeconds,
        status: "queued",
      },
    });

    analysisQueue.add({
      analysisId: analysis.id,
      videoPath: video.storagePath,
      method: request.method,
      regions: request.regions,
      windowSeconds: request.windowSeconds,
      stepSeconds: request.stepSeconds,
    });

    return { id: analysis.id, status: analysis.status };
  }

  async list(): Promise<AnalysisRecord[]> {
    const rows = await prisma.analysis.findMany({
      orderBy: { createdAt: "desc" },
      take: 100,
    });
    return rows.map(serializeAnalysis);
  }

  async getById(id: string): Promise<AnalysisRecord> {
    const analysis = await prisma.analysis.findUnique({ where: { id } });
    if (!analysis) throw new ApiError(404, `Analysis ${id} not found`);
    return serializeAnalysis(analysis);
  }

  async delete(id: string) {
    try {
      await prisma.analysis.delete({ where: { id } });
    } catch (err) {
      if ((err as { code?: string }).code === "P2025") {
        throw new ApiError(404, `Analysis ${id} not found`);
      }
      throw err;
    }
  }
}

export const analysisService = new AnalysisService();
