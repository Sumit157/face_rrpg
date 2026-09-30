interface Props {
  score: number; // 0-100
}

export function SignalQuality({ score }: Props) {
  const color = score >= 65 ? "text-verdict-real" : score >= 40 ? "text-verdict-uncertain" : "text-verdict-fake";
  return (
    <div className="panel p-4 text-center">
      <div className="text-xs text-slate-400 mb-1">Signal Quality</div>
      <div className={`text-2xl font-semibold ${color}`}>{Math.round(score)}/100</div>
    </div>
  );
}
