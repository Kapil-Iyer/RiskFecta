import { useMemo } from "react";
import type { Data, Layout } from "plotly.js-cartesian-dist-min";
import type { PriceObservation } from "../api/types";
import PlotlyChart from "./PlotlyChart";
import { DARK_CHART_LAYOUT } from "./chartTheme";

interface VolumeChartProps {
  ticker: string;
  prices: PriceObservation[];
  accentColor: string;
}

/** Chronological daily volume bar chart, straight from `prices_raw.volume`. */
export default function VolumeChart({ ticker, prices, accentColor }: VolumeChartProps) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        name: `${ticker} volume`,
        x: prices.map((p) => p.date),
        y: prices.map((p) => p.volume),
        marker: { color: accentColor, opacity: 0.55 },
        hovertemplate: "%{x}<br>Volume: %{y:,}<extra></extra>",
      },
    ],
    [ticker, prices, accentColor],
  );

  const layout = useMemo<Partial<Layout>>(
    () => ({
      ...DARK_CHART_LAYOUT,
      title: { text: `${ticker} — daily volume`, font: { size: 13 } },
      xaxis: { ...DARK_CHART_LAYOUT.xaxis, type: "date" },
      yaxis: { ...DARK_CHART_LAYOUT.yaxis, title: { text: "Shares traded" } },
    }),
    [ticker],
  );

  return <PlotlyChart data={data} layout={layout} className="chart chart--volume" ariaLabel={`${ticker} daily trading volume chart`} />;
}
