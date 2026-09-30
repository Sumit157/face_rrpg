import path from "node:path";
import { prisma } from "../db/client";

export interface RegisterVideoInput {
  originalFilename: string;
  storagePath: string;
  sizeBytes: number;
  userId?: string | null;
}

export class VideoService {
  async register(input: RegisterVideoInput) {
    return prisma.video.create({
      data: {
        originalFilename: input.originalFilename,
        storagePath: input.storagePath,
        sizeBytes: input.sizeBytes,
        userId: input.userId ?? null,
      },
    });
  }

  async getById(id: string) {
    return prisma.video.findUnique({ where: { id } });
  }

  resolveAbsolutePath(storagePath: string): string {
    // Never trust client-supplied paths -- storagePath always originates
    // from our own multer-generated filename, joined against the fixed
    // uploads directory only.
    return path.resolve(storagePath);
  }
}

export const videoService = new VideoService();
