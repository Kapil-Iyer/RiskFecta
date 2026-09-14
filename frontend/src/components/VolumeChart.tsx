import { useMemo } from "react";
import type { Data, Layout } from "plotly.js-cartesian-dist-min";
import type { PriceObservation } from "../api/types";
import PlotlyChart from "./PlotlyChart";

interface VolumeChartProps {
  ticker: string;
  prices: PriceObservation[];
}

/** Chronological daily volume bar chart, straight from `prices_raw.volume`. */
export default function VolumeChart({ ticker, prices }: VolumeChartProps) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        name: `${ticker} volume`,
        x: prices.map((p) => p.date),
        y: prices.map((p) => p.volume),
        marker: { color: "#94a3b8" },
        hovertemplate: "%{x}<br>Volume: %{y:,}<extra></extra>",
      },
    ],
    [ticker, prices],
  );

  const layout = useMemo<Partial<Layout>>(
    () => ({
      title: { text: `${ticker} — daily volume` },
      xaxis: { type: "date" },
      yaxis: { title: { text: "Shares traded" } },
      margin: { t: 48, r: 24, b: 40, l: 64 },
      autosize: true,
      font: { family: "Inter, system-ui, sans-serif", size: 12 },
    }),
    [ticker],
  );

  return <PlotlyChart data={data} layout={layout} className="chart chart--volume" ariaLabel={`${ticker} daily trading volume chart`} />;
}
