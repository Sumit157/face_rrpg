import type { Server as HttpServer } from "node:http";
import { Server as SocketIOServer } from "socket.io";
import { config } from "../config";
import type { AnalysisProgressEvent } from "@rppg/shared-types";

let io: SocketIOServer | null = null;

export function initWebSocketGateway(httpServer: HttpServer): SocketIOServer {
  io = new SocketIOServer(httpServer, {
    cors: { origin: config.corsOrigin },
  });

  io.on("connection", (socket) => {
    socket.on("subscribe:analysis", (analysisId: string) => {
      socket.join(`analysis:${analysisId}`);
    });
  });

  return io;
}

/** Progress events originate from the in-process analysis queue (see
 * src/jobs/analysis.queue.ts), which runs in the same Node process as this
 * Socket.IO server -- so this is a direct, working call. If job execution
 * is ever moved to a separate process again, publish through Redis pub/sub
 * (e.g. the socket.io-redis adapter) instead. */
export function emitAnalysisEvent(event: AnalysisProgressEvent): void {
  if (!io) return;
  io.to(`analysis:${event.analysisId}`).emit(`analysis:${event.stage}`, event);
  io.to(`analysis:${event.analysisId}`).emit("analysis:progress", event);
}
