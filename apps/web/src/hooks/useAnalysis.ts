import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { analysisApi } from "../services/api";
import type { CreateAnalysisRequest } from "@rppg/shared-types";

const TERMINAL_STATUSES = new Set(["completed", "failed"]);

export function useAnalysesList(options?: { refetchInterval?: number }) {
  return useQuery({
    queryKey: ["analyses"],
    queryFn: analysisApi.list,
    refetchInterval: options?.refetchInterval,
  });
}

export function useAnalysis(id: string | undefined) {
  return useQuery({
    queryKey: ["analysis", id],
    queryFn: () => analysisApi.get(id as string),
    enabled: !!id,
    // Poll while the job is still in flight; the WebSocket hook can
    // short-circuit this once real-time events arrive (see useWebSocket).
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status && TERMINAL_STATUSES.has(status) ? false : 2000;
    },
  });
}

export function useCreateAnalysis() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (req: CreateAnalysisRequest) => analysisApi.create(req),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["analyses"] }),
  });
}
