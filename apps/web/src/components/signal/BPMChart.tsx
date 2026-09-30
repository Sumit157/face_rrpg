import ReactECharts from "echarts-for-react";

interface Props {
  windowIndex: number[];
  bpm: number[];
}

export function BPMChart({ windowIndex, bpm }: Props) {
  const option = {
    backgroundColor: "transparent",
    title: { text: "BPM over time", textStyle: { color: "#cbd5e1", fontSize: 13 } },
    grid: { left: 40, right: 20, top: 40, bottom: 30 },
    xAxis: {
      type: "category",
      data: windowIndex.map(String),
      name: "window",
      axisLine: { lineStyle: { color: "#334155" } },
      axisLabel: { color: "#94a3b8" },
    },
    yAxis: {
      type: "value",
      name: "BPM",
      axisLine: { lineStyle: { color: "#334155" } },
      axisLabel: { color: "#94a3b8" },
      splitLine: { lineStyle: { color: "#1e293b" } },
    },
    series: [
      {
        type: "line",
        data: bpm,
        symbolSize: 6,
        lineStyle: { color: "#a78bfa", width: 2 },
        itemStyle: { color: "#a78bfa" },
      },
    ],
  };
  return <ReactECharts option={option} style={{ height: 200 }} notMerge />;
}
