import { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { isAxiosError } from "axios";
import type { AnalysisVisuals } from "@rppg/shared-types";
import { analysisApi, streamUrl } from "../../services/api";
import { PipelinePlayer } from "./PipelinePlayer";
import { StageTimeline } from "./StageTimeline";
import { ExtractionStats } from "./ExtractionStats";
import { WindowSegmentation } from "./WindowSegmentation";

interface Props {
  analysisId: string;
  videoId: string;
}

export function PipelineView({ analysisId, videoId }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [currentTime, setCurrentTime] = useState(0);

  const { data, isLoading, error } = useQuery<AnalysisVisuals | null>({
    queryKey: ["visuals", analysisId],
    queryFn: async () => {
      try {
        return await analysisApi.visuals(analysisId);
      } catch (e) {
        if (isAxiosError(e) && e.response?.status === 404) return null;
        throw e;
      }
    },
    retry: false,
  });

  const seek = (t: number) => {
    const v = videoRef.current;
    if (v) {
      v.currentTime = t;
      setCurrentTime(t);
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4" aria-busy="true">
        <div className="skeleton h-64 w-full" />
        <div className="skeleton h-40 w-full" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="panel p-6 border border-verdict-fake">
        <p className="text-sm text-verdict-fake">
          Failed to load the pipeline trace.
        </p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="panel p-6">
        <h3 className="text-sm text-slate-300 mb-1">No pipeline trace</h3>
        <p className="text-xs text-slate-500 max-w-prose">
          This analysis ran before frame tracing was added (or stopped before
          producing one). Re-run the analysis to see the detection overlay,
          stage timeline and window segmentation.
        </p>
      </div>
    );
  }

  const hasFrames = data.frames.length > 0;
  const hasExtraction = data.extraction && data.extraction.framesProcessed > 0;

  return (
    <div className="space-y-6 animate-fadeUp">
      <PipelinePlayer
        src={streamUrl(videoId)}
        frames={data.frames}
        videoRef={videoRef}
        onTime={setCurrentTime}
      />

      <StageTimeline stages={data.stages} />

      {hasExtraction && <ExtractionStats extraction={data.extraction} />}

      {!hasFrames && (
        <div className="panel p-4 text-xs text-slate-500">
          No per-frame detections were recorded for this run (face extraction
          did not complete), but the stage timeline above still shows how far
          the pipeline got.
        </div>
      )}

      {data.windows.length > 0 && (
        <WindowSegmentation
          windows={data.windows}
          currentTime={currentTime}
          onSeek={seek}
        />
      )}
    </div>
  );
}
