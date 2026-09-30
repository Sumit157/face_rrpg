import { Router } from "express";
import { uploadVideo } from "../middleware/upload.middleware";
import { asyncHandler } from "../middleware/async";
import {
  uploadVideoHandler,
  getVideoHandler,
  streamVideoHandler,
} from "../controllers/video.controller";

export const videoRouter = Router();

videoRouter.post("/", uploadVideo.single("video"), asyncHandler(uploadVideoHandler));
videoRouter.get("/:id", asyncHandler(getVideoHandler));
videoRouter.get("/:id/stream", asyncHandler(streamVideoHandler));
