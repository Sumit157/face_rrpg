import { NavLink } from "react-router-dom";

const GROUPS: { heading: string; links: { to: string; label: string }[] }[] = [
  {
    heading: "Analysis",
    links: [
      { to: "/dashboard", label: "Dashboard" },
      { to: "/analysis/new", label: "New analysis" },
      { to: "/live", label: "Live" },
    ],
  },
  {
    heading: "Research",
    links: [
      { to: "/signal-explorer", label: "Signal explorer" },
      { to: "/models", label: "Model evaluation" },
      { to: "/reports", label: "Reports" },
    ],
  },
  {
    heading: "Project",
    links: [{ to: "/about", label: "About" }],
  },
];

function BrandMark() {
  return (
    <svg viewBox="0 0 32 32" className="h-8 w-8" aria-hidden="true">
      <rect width="32" height="32" rx="8" className="fill-surface-800" />
      <path
        d="M4 17h6l3-8 4 14 3-6h8"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        className="text-accent"
      />
    </svg>
  );
}

export function Sidebar() {
  return (
    <aside className="w-60 shrink-0 border-r border-surface-700/60 bg-surface-900/60 backdrop-blur-sm flex flex-col">
      <div className="flex items-center gap-3 px-5 py-6 border-b border-surface-700/60">
        <BrandMark />
        <div className="leading-tight">
          <div className="text-[15px] font-semibold tracking-tight text-slate-100">
            rPPG Forensic
          </div>
          <div className="font-mono text-[10px] tracking-wider text-slate-500 uppercase">
            Analyser
          </div>
        </div>
      </div>

      <nav className="flex-1 px-3 py-5 space-y-6 overflow-y-auto">
        {GROUPS.map((group) => (
          <div key={group.heading}>
            <div className="label px-3 mb-2">{group.heading}</div>
            <ul className="space-y-0.5">
              {group.links.map((link) => (
                <li key={link.to}>
                  <NavLink
                    to={link.to}
                    className={({ isActive }) =>
                      `group relative flex items-center rounded-lg px-3 py-2 text-sm transition-colors duration-200 ${
                        isActive
                          ? "bg-accent/10 text-accent-soft font-medium"
                          : "text-slate-400 hover:text-slate-100 hover:bg-surface-800/70"
                      }`
                    }
                  >
                    {({ isActive }) => (
                      <>
                        <span
                          className={`absolute left-0 top-1/2 -translate-y-1/2 h-4 w-0.5 rounded-full transition-all duration-200 ${
                            isActive ? "bg-accent" : "bg-transparent group-hover:bg-surface-700"
                          }`}
                        />
                        {link.label}
                      </>
                    )}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </nav>

      <div className="px-5 py-4 border-t border-surface-700/60 font-mono text-[10px] text-slate-600 tracking-wide">
        v0.1.0 · local mode
      </div>
    </aside>
  );
}
