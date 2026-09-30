import { useEffect, useRef, useState, type RefObject } from "react";
import type { VisualFrame } from "@rppg/shared-types";

interface Props {
  src: string;
  frames: VisualFrame[];
  videoRef: RefObject<HTMLVideoElement>;
  onTime: (t: number) => void;
}

const ROI_COLORS: Record<string, string> = {
  forehead: "#22d3ee",
  left_cheek: "#a78bfa",
  right_cheek: "#f472b6",
};

function nearestFrame(frames: VisualFrame[], t: number): VisualFrame | null {
  if (frames.length === 0) return null;
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (frames[mid].t < t) lo = mid + 1;
    else hi = mid;
  }
  const next = frames[lo];
  const prev = lo > 0 ? frames[lo - 1] : null;
  const best =
    prev && Math.abs(prev.t - t) <= Math.abs(next.t - t) ? prev : next;
  // The trace is sampled (~5 Hz); if playback has moved far beyond the
  // nearest sample there is no meaningful overlay to draw.
  const stride = frames.length > 1 ? frames[1].t - frames[0].t : 0.2;
  const tol = Math.max(0.6, stride * 1.5);
  return Math.abs(best.t - t) <= tol ? best : null;
}

function fmt(t: number): string {
  const s = Math.max(0, t);
  return `${String(Math.floor(s / 60)).padStart(2, "0")}:${(s % 60).toFixed(1).padStart(4, "0")}`;
}

/** Video + live detection overlay: draws the face box and rPPG ROI
 * polygons for whichever trace sample is nearest to the playhead. */
export function PipelinePlayer({ src, frames, videoRef, onTime }: Props) {
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    let raf = 0;
    const tick = () => {
      setCurrentTime(v.currentTime);
      onTime(v.currentTime);
      raf = requestAnimationFrame(tick);
    };
    const onPlay = () => {
      setPlaying(true);
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(tick);
    };
    const stop = () => {
      setPlaying(false);
      cancelAnimationFrame(raf);
      setCurrentTime(v.currentTime);
      onTime(v.currentTime);
    };
    v.addEventListener("play", onPlay);
    v.addEventListener("pause", stop);
    v.addEventListener("ended", stop);
    v.addEventListener("seeked", stop);
    return () => {
      cancelAnimationFrame(raf);
      v.removeEventListener("play", onPlay);
      v.removeEventListener("pause", stop);
      v.removeEventListener("ended", stop);
      v.removeEventListener("seeked", stop);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [src, frames]);

  const entry = nearestFrame(frames, currentTime);
  const stride =
    frames.length > 1 ? Math.max(0.001, frames[1].t - frames[0].t) : 0.2;
  const traceEnd = frames.length ? frames[frames.length - 1].t : 0;
  const stripEnd = duration || traceEnd || 1;

  const seek = (t: number) => {
    const v = videoRef.current;
    if (v) {
      v.currentTime = Math.max(0, Math.min(t, stripEnd));
      setCurrentTime(v.currentTime);
      onTime(v.currentTime);
    }
  };

  return (
    <div className="panel overflow-hidden">
      <div className="relative bg-black">
        <video
          ref={videoRef}
          src={src}
          controls
          preload="metadata"
          className="w-full h-auto block"
          onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        />
        <div className="absolute inset-0 pointer-events-none">
          {entry && entry.detected && entry.bbox && (
            <svg
              className="absolute inset-0 w-full h-full"
              viewBox="0 0 1 1"
              preserveAspectRatio="none"
              aria-hidden
            >
              <rect
                x={entry.bbox[0]}
                y={entry.bbox[1]}
                width={entry.bbox[2]}
                height={entry.bbox[3]}
                fill="none"
                stroke="#22d3ee"
                strokeWidth={2}
                vectorEffect="non-scaling-stroke"
              />
              {entry.roi &&
                Object.entries(entry.roi).map(([name, pts]) => (
                  <polygon
                    key={name}
                    points={pts.map(([x, y]) => `${x},${y}`).join(" ")}
                    fill={ROI_COLORS[name] ?? "#22d3ee"}
                    fillOpacity={0.16}
                    stroke={ROI_COLORS[name] ?? "#22d3ee"}
                    strokeWidth={1.5}
                    vectorEffect="non-scaling-stroke"
                  />
                ))}
            </svg>
          )}

          {/* HTML chips (SVG text would distort under preserveAspectRatio="none") */}
          <div className="absolute top-2 left-2 flex gap-1.5 text-[10px] font-mono">
            {entry ? (
              entry.detected ? (
                <span className="px-1.5 py-0.5 rounded bg-surface-900/85 border border-verdict-real/60 text-verdict-real">
                  face · {entry.method}
                </span>
              ) : (
                <span className="px-1.5 py-0.5 rounded bg-surface-900/85 border border-verdict-fake/70 text-verdict-fake animate-pulse">
                  face lost
                </span>
              )
            ) : (
              <span className="px-1.5 py-0.5 rounded bg-surface-900/85 border border-surface-700 text-slate-400">
                no trace sample
              </span>
            )}
            {entry?.enhanced && (
              <span className="px-1.5 py-0.5 rounded bg-surface-900/85 border border-verdict-uncertain/60 text-verdict-uncertain">
                enhanced
              </span>
            )}
          </div>
          <div className="absolute top-2 right-2 px-1.5 py-0.5 rounded bg-surface-900/85 border border-surface-700 text-[10px] font-mono text-slate-300">
            {fmt(currentTime)}
            {entry && (
              <span className="text-slate-500"> @ {entry.t.toFixed(2)}s</span>
            )}
          </div>
          {!playing && frames.length > 0 && (
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 text-[11px] px-2 py-1 rounded bg-surface-900/80 border border-surface-700 text-slate-300">
              Detection overlay follows playback -- press play or scrub below
            </div>
          )}
        </div>
      </div>

      {/* Scrubber + detection strip */}
      <div className="px-4 py-3 border-t border-surface-700 space-y-2">
        <div className="flex items-center gap-3">
          <span className="text-[11px] font-mono text-slate-400 w-12 shrink-0">
            {fmt(currentTime)}
          </span>
          <input
            type="range"
            min={0}
            max={stripEnd}
            step={0.05}
            value={Math.min(currentTime, stripEnd)}
            onChange={(e) => seek(parseFloat(e.target.value))}
            aria-label="Scrub video"
            className="flex-1 accent-cyan-400 h-1"
          />
          <span className="text-[11px] font-mono text-slate-500 w-12 shrink-0 text-right">
            {fmt(stripEnd)}
          </span>
        </div>
        <div
          className="relative h-1.5 flex gap-px cursor-pointer"
          onClick={(e) => {
            const r = e.currentTarget.getBoundingClientRect();
            seek(((e.clientX - r.left) / r.width) * stripEnd);
          }}
          title="Per-sample face detection (green = detected)"
          role="presentation"
        >
          {frames.map((f) => (
            <div
              key={f.f}
              className="flex-1"
              style={{
                background: f.detected ? "#34d399" : "#f87171",
                opacity: 0.75,
              }}
            />
          ))}
          <div
            className="absolute top-0 bottom-0 w-px bg-white/90 pointer-events-none"
            style={{ left: `${(currentTime / stripEnd) * 100}%` }}
          />
        </div>
        <div className="flex justify-between text-[10px] text-slate-500">
          <span>frame samples (~{stride.toFixed(2)}s apart)</span>
          <span>
            <span className="text-verdict-real">&#9632;</span> detected{" "}
            <span className="text-verdict-fake">&#9632;</span> missed
          </span>
        </div>
      </div>
    </div>
  );
}
