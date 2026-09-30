interface Props {
  pairwiseCorrelations: Record<string, number>;
}

export function RegionCorrelation({ pairwiseCorrelations }: Props) {
  const entries = Object.entries(pairwiseCorrelations);
  return (
    <div className="panel p-4">
      <h3 className="text-sm text-slate-300 mb-3">Region correlation</h3>
      <div className="space-y-2">
        {entries.map(([pair, value]) => {
          const label = pair.replace("__", " \u2194 ").replace(/_/g, " ");
          const pct = Math.round(((value + 1) / 2) * 100); // -1..1 -> 0..100
          return (
            <div key={pair}>
              <div className="flex justify-between text-xs text-slate-400 mb-1">
                <span>{label}</span>
                <span>{value.toFixed(2)}</span>
              </div>
              <div className="h-1.5 bg-surface-700 rounded">
                <div
                  className="h-1.5 rounded bg-accent"
                  style={{ width: `${pct}%` }}
                />
              </div>
            </div>
          );
        })}
        {entries.length === 0 && <p className="text-xs text-slate-500">No data yet.</p>}
      </div>
    </div>
  );
}
