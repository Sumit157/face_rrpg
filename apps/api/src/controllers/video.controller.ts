import fs from "node:fs";
import type { Request, Response } from "express";
import { videoService } from "../services/video.service";
import { ApiError } from "../middleware/error.middleware";

export async function uploadVideoHandler(req: Request, res: Response): Promise<void> {
  if (!req.file) {
    throw new ApiError(400, "No video file provided");
  }
  const video = await videoService.register({
    originalFilename: req.file.originalname,
    // Absolute path: the ML service runs in its own working directory and
    // would otherwise resolve the upload against the wrong base.
    storagePath: videoService.resolveAbsolutePath(req.file.path),
    sizeBytes: req.file.size,
  });
  res.status(201).json(video);
}

export async function getVideoHandler(req: Request, res: Response): Promise<void> {
  const video = await videoService.getById(req.params.id);
  if (!video) throw new ApiError(404, "Video not found");
  res.json(video);
}

/** Streams the stored file with HTTP Range support (video seeking).
 * Storage is never served statically -- this explicit route is the only
 * way to read a video back. */
export async function streamVideoHandler(req: Request, res: Response): Promise<void> {
  const video = await videoService.getById(req.params.id);
  if (!video) throw new ApiError(404, "Video not found");
  const absolute = videoService.resolveAbsolutePath(video.storagePath);
  if (!fs.existsSync(absolute)) {
    throw new ApiError(404, "Video file missing from storage");
  }
  res.sendFile(absolute, (err) => {
    if (err && !res.headersSent) {
      res.status(404).json({ error: "Could not stream video" });
    }
  });
}
