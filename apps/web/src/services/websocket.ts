import { io, type Socket } from "socket.io-client";

let socket: Socket | null = null;

export function getSocket(): Socket {
  if (!socket) {
    socket = io("/", { path: "/socket.io" });
  }
  return socket;
}

export function subscribeToAnalysis(analysisId: string): void {
  getSocket().emit("subscribe:analysis", analysisId);
}
