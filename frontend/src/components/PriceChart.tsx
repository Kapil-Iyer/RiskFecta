import { useMemo } from "react";
import type { Data, Layout } from "plotly.js-cartesian-dist-min";
import type { PriceObservation } from "../api/types";
import PlotlyChart from "./PlotlyChart";
import { DARK_CHART_LAYOUT } from "./chartTheme";

interface PriceChartProps {
  ticker: string;
  prices: PriceObservation[];
  /** Sector accent hex (theme.ts) — Plotly needs a real color string, not a
   * CSS variable, since it draws to canvas/SVG outside the page's own CSS
   * cascade. */
  accentColor: string;
}

/** Chronological close-price line chart. No indicators, no forecasts —
 * exactly what's stored in `prices_raw`. */
export default function PriceChart({ ticker, prices, accentColor }: PriceChartProps) {
  const data = useMemo<Data[]>(
    () => [
      {
        type: "scatter",
        mode: "lines",
        name: `${ticker} close`,
        x: prices.map((p) => p.date),
        y: prices.map((p) => p.close),
        line: { color: accentColor, width: 1.75 },
        hovertemplate: "%{x}<br>Close: $%{y:.2f}<extra></extra>",
      },
    ],
    [ticker, prices, accentColor],
  );

  const layout = useMemo<Partial<Layout>>(
    () => ({
      ...DARK_CHART_LAYOUT,
      title: { text: `${ticker} — historical close price`, font: { size: 13 } },
      xaxis: { ...DARK_CHART_LAYOUT.xaxis, type: "date" },
      yaxis: { ...DARK_CHART_LAYOUT.yaxis, title: { text: "Close (USD)" }, tickprefix: "$" },
    }),
    [ticker],
  );

  return <PlotlyChart data={data} layout={layout} className="chart chart--price" ariaLabel={`${ticker} historical close price chart`} />;
}
