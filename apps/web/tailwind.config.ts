import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Forensic/technical palette: dark surface, cyan accent for
        // signal traces, distinct semantic colours for verdicts.
        surface: {
          950: "#0a0e14",
          900: "#0f1520",
          800: "#161d2b",
          700: "#212a3b",
        },
        accent: {
          DEFAULT: "#22d3ee",
          dim: "#0e7490",
        },
        verdict: {
          real: "#22c55e",
          fake: "#ef4444",
          uncertain: "#eab308",
        },
      },
      fontFamily: {
        mono: ["JetBrains Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
