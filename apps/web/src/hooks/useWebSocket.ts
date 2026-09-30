import { useEffect, useState } from "react";
import { getSocket, subscribeToAnalysis } from "../services/websocket";
import type { AnalysisProgressEvent } from "@rppg/shared-types";

export function useAnalysisProgress(analysisId: string | undefined) {
  const [event, setEvent] = useState<AnalysisProgressEvent | null>(null);

  useEffect(() => {
    if (!analysisId) return;
    const socket = getSocket();
    subscribeToAnalysis(analysisId);

    const handler = (e: AnalysisProgressEvent) => setEvent(e);
    socket.on("analysis:progress", handler);
    return () => {
      socket.off("analysis:progress", handler);
    };
  }, [analysisId]);

  return event;
}
