interface Props {
  detectionRate: number; // 0-1
}

/** Simple status chip shown over the video while a full landmark overlay
 * (drawn from per-frame ROI polygons) is wired up to a real detection
 * stream. See ROIOverlay for the region-box variant. */
export function FaceOverlay({ detectionRate }: Props) {
  const ok = detectionRate >= 0.7;
  return (
    <div
      className={`absolute top-2 left-2 text-xs px-2 py-1 rounded ${
        ok ? "bg-verdict-real/20 text-verdict-real" : "bg-verdict-uncertain/20 text-verdict-uncertain"
      }`}
    >
      Face detected {Math.round(detectionRate * 100)}% of frames
    </div>
  );
}
