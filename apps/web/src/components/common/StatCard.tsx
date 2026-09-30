interface Props {
  label: string;
  value: string | number;
  accent?: string;
}

export function StatCard({ label, value, accent = "text-slate-100" }: Props) {
  return (
    <div className="panel p-4 transition-shadow duration-200 hover:shadow-lift">
      <div className="label mb-1.5">{label}</div>
      <div className={`font-mono text-2xl font-semibold tracking-tight ${accent}`}>{value}</div>
    </div>
  );
}
