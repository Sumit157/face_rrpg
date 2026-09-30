import ReactECharts from "echarts-for-react";

interface Props {
  timestamps: number[];
  values: number[];
  title?: string;
}

export function RPPGChart({ timestamps, values, title = "rPPG waveform" }: Props) {
  const option = {
    backgroundColor: "transparent",
    title: { text: title, textStyle: { color: "#cbd5e1", fontSize: 13 } },
    grid: { left: 40, right: 20, top: 40, bottom: 30 },
    xAxis: {
      type: "value",
      name: "seconds",
      axisLine: { lineStyle: { color: "#334155" } },
      axisLabel: { color: "#94a3b8" },
    },
    yAxis: {
      type: "value",
      axisLine: { lineStyle: { color: "#334155" } },
      axisLabel: { color: "#94a3b8" },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: [
      {
        type: "line",
        data: timestamps.map((t, i) => [t, values[i]]),
        showSymbol: false,
        lineStyle: { color: "#22d3ee", width: 1.5 },
      },
    ],
  };
  return <ReactECharts option={option} style={{ height: 220 }} notMerge />;
}
