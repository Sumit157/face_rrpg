import type { PropsWithChildren } from "react";
import { Sidebar } from "./Sidebar";

export function Layout({ children }: PropsWithChildren) {
  return (
    <div className="flex min-h-dvh">
      <a href="#content" className="skip-link">
        Skip to content
      </a>
      <Sidebar />
      <main id="content" className="flex-1 min-w-0 p-6 md:p-8 max-w-6xl mx-auto w-full">
        <div className="animate-fadeUp">{children}</div>
      </main>
    </div>
  );
}
