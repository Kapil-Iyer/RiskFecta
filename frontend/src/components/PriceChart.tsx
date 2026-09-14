import { useMemo } from "react";
import type { Data, Layout } from "plotly.js-cartesian-dist-min";
import type { PriceObservation } from "../api/types";
import PlotlyChart from "./PlotlyChart";

interface PriceChartProps {
  ticker: string;
  prices: PriceObservation[];
}

/** Chronological close-price line chart. No indicators, no forecasts —
 * exactly what's stored in `prices_raw`. */
export default function PriceChart({ ticker, prices }: PriceChartProps) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "scatter",
        mode: "lines",
        name: `${ticker} close`,
        x: prices.map((p) => p.date),
        y: prices.map((p) => p.close),
        line: { color: "#2563eb", width: 1.75 },
        hovertemplate: "%{x}<br>Close: $%{y:.2f}<extra></extra>",
      },
    ],
    [ticker, prices],
  );

  const layout = useMemo<Partial<Layout>>(
    () => ({
      title: { text: `${ticker} — historical close price` },
      xaxis: { type: "date", title: { text: "" } },
      yaxis: { title: { text: "Close (USD)" }, tickprefix: "$" },
      margin: { t: 48, r: 24, b: 40, l: 56 },
      autosize: true,
      font: { family: "Inter, system-ui, sans-serif", size: 12 },
    }),
    [ticker],
  );

  return <PlotlyChart data={data} layout={layout} className="chart chart--price" ariaLabel={`${ticker} historical close price chart`} />;
}
