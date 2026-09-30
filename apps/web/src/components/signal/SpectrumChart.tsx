import ReactECharts from "echarts-for-react";

interface Props {
  freqsHz: number[];
  power: number[];
  peakHz?: number;
}

export function SpectrumChart({ freqsHz, power, peakHz }: Props) {
  const option = {
    backgroundColor: "transparent",
    title: { text: "Frequency spectrum", textStyle: { color: "#cbd5e1", fontSize: 13 } },
    grid: { left: 40, right: 20, top: 40, bottom: 30 },
    xAxis: {
      type: "value",
      name: "Hz",
      min: 0,
      max: 4.5,
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
        data: freqsHz.map((f, i) => [f, power[i]]),
        showSymbol: false,
        areaStyle: { color: "rgba(34, 211, 238, 0.15)" },
        lineStyle: { color: "#22d3ee", width: 1.5 },
        markLine: peakHz
          ? {
              data: [{ xAxis: peakHz }],
              lineStyle: { color: "#eab308" },
              label: { formatter: `peak: ${peakHz.toFixed(2)}Hz`, color: "#eab308" },
            }
          : undefined,
      },
    ],
  };
  return <ReactECharts option={option} style={{ height: 220 }} notMerge />;
}
