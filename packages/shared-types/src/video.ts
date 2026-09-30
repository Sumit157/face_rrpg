export interface VideoMetadata {
  path: string;
  width: number;
  height: number;
  fps: number;
  frameCount: number;
  durationSeconds: number;
  codec: string;
}

export interface Video {
  id: string;
  originalFilename: string;
  storagePath: string;
  metadata: VideoMetadata | null;
  uploadedAt: string;
}
