import fs from "node:fs";
import path from "node:path";
import dotenv from "dotenv";

/** Locate the monorepo root by walking up from cwd until pnpm-workspace.yaml
 * is found, so storage/uploads paths resolve identically whether the API is
 * started from the repo root or from apps/api. */
function findRepoRoot(start: string): string {
  let dir = start;
  for (let i = 0; i < 8; i += 1) {
    if (fs.existsSync(path.join(dir, "pnpm-workspace.yaml"))) return dir;
    const parent = path.dirname(dir);
    if (parent === dir) break;
    dir = parent;
  }
  return start;
}

const repoRoot = findRepoRoot(process.cwd());

// Load env before reading process.env anywhere else. Prisma's CLI reads the
// root .env itself; the API process does not, so load both locations here.
dotenv.config({ path: path.join(repoRoot, ".env") });
dotenv.config({ path: path.join(process.cwd(), ".env") });

function resolveFromRoot(value: string): string {
  return path.isAbsolute(value) ? value : path.resolve(repoRoot, value);
}

export const config = {
  port: parseInt(process.env.PORT ?? "3000", 10),
  nodeEnv: process.env.NODE_ENV ?? "development",
  repoRoot,

  databaseUrl: process.env.DATABASE_URL ?? "file:./dev.db",

  pythonServiceUrl: process.env.ML_SERVICE_URL ?? "http://localhost:8000",

  storageDir: resolveFromRoot(process.env.STORAGE_DIR ?? "./storage"),
  uploadsDir: resolveFromRoot(process.env.UPLOADS_DIR ?? "./storage/uploads"),

  maxUploadSizeMb: parseInt(process.env.MAX_UPLOAD_SIZE_MB ?? "500", 10),
  allowedVideoExtensions: [".mp4", ".avi", ".mov", ".mkv", ".webm"],

  corsOrigin: process.env.CORS_ORIGIN ?? "http://localhost:5173",

  defaultAnalysis: {
    method: "auto" as const,
    regions: ["forehead", "left_cheek", "right_cheek"] as const,
    windowSeconds: 12,
    stepSeconds: 3,
  },
};
