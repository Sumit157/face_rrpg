interface Props {
  warnings: string[];
}

export function WarningsPanel({ warnings }: Props) {
  return (
    <div className="panel p-4">
      <h3 className="label mb-2.5">Warnings</h3>
      {warnings.length === 0 ? (
        <p className="text-xs text-slate-500">No warnings.</p>
      ) : (
        <ul className="space-y-1.5">
          {warnings.map((w, i) => (
            <li key={i} className="text-sm flex gap-2">
              <span className="text-verdict-uncertain" aria-hidden="true">
                ⚠
              </span>
              <span className="text-slate-300">{w}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
