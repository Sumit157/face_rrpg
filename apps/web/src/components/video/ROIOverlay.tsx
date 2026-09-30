interface Region {
  name: string;
  x: number; // 0-1 normalised
  y: number;
  width: number;
  height: number;
}

interface Props {
  regions: Region[];
}

const COLORS: Record<string, string> = {
  forehead: "#22d3ee",
  left_cheek: "#a78bfa",
  right_cheek: "#f472b6",
};

export function ROIOverlay({ regions }: Props) {
  return (
    <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1 1" preserveAspectRatio="none">
      {regions.map((r) => (
        <rect
          key={r.name}
          x={r.x}
          y={r.y}
          width={r.width}
          height={r.height}
          fill="none"
          stroke={COLORS[r.name] ?? "#22d3ee"}
          strokeWidth={0.004}
        />
      ))}
    </svg>
  );
}
