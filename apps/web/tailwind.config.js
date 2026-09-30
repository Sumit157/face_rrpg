import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Forensic/technical palette: blue-tinted dark surfaces, one cyan
        // accent for signal traces, semantic colours only for verdicts.
        surface: {
          950: "#0a0e14",
          900: "#0f1520",
          800: "#161d2b",
          700: "#212a3b",
        },
        accent: {
          DEFAULT: "#22d3ee",
          dim: "#0e7490",
          soft: "#67e8f9",
        },
        verdict: {
          real: "#34d399",
          fake: "#f87171",
          uncertain: "#fbbf24",
        },
      },
      fontFamily: {
        sans: ["Outfit", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        // Tinted shadows: light comes from a cool top-left source.
        panel:
          "inset 0 1px 0 0 rgb(255 255 255 / 0.04), 0 12px 32px -16px rgb(2 10 22 / 0.9)",
        lift: "inset 0 1px 0 0 rgb(255 255 255 / 0.06), 0 20px 44px -18px rgb(2 10 22 / 0.95)",
        glow: "0 0 0 1px rgb(34 211 238 / 0.25), 0 12px 32px -16px rgb(34 211 238 / 0.15)",
      },
      keyframes: {
        shimmer: {
          "0%": { backgroundPosition: "-400px 0" },
          "100%": { backgroundPosition: "400px 0" },
        },
        fadeUp: {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
      },
      animation: {
        shimmer: "shimmer 1.4s linear infinite",
        fadeUp: "fadeUp 0.4s cubic-bezier(0.22, 1, 0.36, 1) both",
      },
    },
  },
  plugins: [],
} satisfies Config;
