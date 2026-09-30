interface Props {
  evidence: string[];
}

export function EvidencePanel({ evidence }: Props) {
  return (
    <div className="panel p-4">
      <h3 className="label mb-2.5">Physiological evidence</h3>
      {evidence.length === 0 ? (
        <p className="text-xs text-slate-500">No strong evidence identified.</p>
      ) : (
        <ul className="space-y-1.5">
          {evidence.map((e, i) => (
            <li key={i} className="text-sm flex gap-2">
              <span className="text-verdict-real" aria-hidden="true">
                ✓
              </span>
              <span className="text-slate-300">{e}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
