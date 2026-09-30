import multer from "multer";
import path from "node:path";
import crypto from "node:crypto";
import { config } from "../config";

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, config.uploadsDir),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const safeName = `${crypto.randomUUID()}${ext}`;
    cb(null, safeName);
  },
});

function fileFilter(_req: unknown, file: Express.Multer.File, cb: multer.FileFilterCallback) {
  const ext = path.extname(file.originalname).toLowerCase();
  if (!config.allowedVideoExtensions.includes(ext)) {
    cb(new Error(`Unsupported file extension: ${ext}`));
    return;
  }
  cb(null, true);
}

export const uploadVideo = multer({
  storage,
  fileFilter,
  limits: { fileSize: config.maxUploadSizeMb * 1024 * 1024 },
});
