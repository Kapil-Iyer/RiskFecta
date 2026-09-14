import type { Layout } from "plotly.js-cartesian-dist-min";

/**
 * Shared dark-theme Plotly layout, reused by PriceChart and VolumeChart.
 * Transparent paper/plot backgrounds let the surrounding dark card show
 * through; gridlines and axis lines are deliberately low-contrast so the
 * data line/bars stay the visual focus.
 */
const TEXT_MUTED = "#93a1b8";
const GRID = "rgba(148, 163, 184, 0.12)";
const AXIS_LINE = "rgba(148, 163, 184, 0.28)";

export const DARK_CHART_LAYOUT: Partial<Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor: "transparent",
  font: { family: "Inter, system-ui, sans-serif", size: 12, color: TEXT_MUTED },
  margin: { t: 44, r: 20, b: 40, l: 56 },
  autosize: true,
  xaxis: {
    title: { text: "" },
    gridcolor: GRID,
    linecolor: AXIS_LINE,
    zeroline: false,
    tickfont: { color: TEXT_MUTED },
  },
  yaxis: {
    gridcolor: GRID,
    linecolor: AXIS_LINE,
    zeroline: false,
    tickfont: { color: TEXT_MUTED },
  },
  hoverlabel: {
    bgcolor: "#121b2e",
    bordercolor: "#2a3652",
    font: { color: "#e7ebf3", size: 12 },
  },
};
