import type { Analysis } from "@prisma/client";
import type { AnalysisRecord, AnalysisStatus } from "@rppg/shared-types";

/** Parse a JSON-encoded column (SQLite has no Json type, so `result`,
 * `regions` and `metrics` are stored as text). Tolerates an already-decoded
 * value so the code keeps working if the datasource is flipped back to
 * PostgreSQL, where Prisma returns a real Json object. */
export function parseJson<T>(raw: unknown): T | null {
  if (raw == null) return null;
  if (typeof raw === "string") {
    try {
      return JSON.parse(raw) as T;
    } catch {
      return null;
    }
  }
  return raw as T;
}

export function parseStringArray(raw: string): string[] {
  const parsed = parseJson<unknown>(raw);
  return Array.isArray(parsed) ? (parsed as string[]) : [];
}

/** DB row -> the flat shape the frontend consumes: row metadata plus every
 * ML payload field at the top level (see AnalysisRecord in shared-types). */
export function serializeAnalysis(row: Analysis): AnalysisRecord {
  const result = parseJson<Record<string, unknown>>(row.result);
  return {
    ...(result ?? {}),
    id: row.id,
    videoId: row.videoId,
    status: row.status as AnalysisStatus,
    errorMessage: row.errorMessage,
    method: row.method,
    regions: parseStringArray(row.regions),
    windowSeconds: row.windowSeconds,
    stepSeconds: row.stepSeconds,
    createdAt: row.createdAt.toISOString(),
    completedAt: row.completedAt ? row.completedAt.toISOString() : null,
  };
}
