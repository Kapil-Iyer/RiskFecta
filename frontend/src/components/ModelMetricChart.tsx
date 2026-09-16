import { useMemo } from "react";
import type { Data, Layout } from "plotly.js-cartesian-dist-min";
import type { MetricDefinition, ModelMetricRow } from "../api/types";
import PlotlyChart from "./PlotlyChart";
import { DARK_CHART_LAYOUT } from "./chartTheme";

interface ModelMetricChartProps {
  /** Pre-sorted (best-to-worst, or a stable order for context-dependent
   * metrics) — this component renders in the order given, it doesn't sort. */
  models: ModelMetricRow[];
  metric: MetricDefinition;
}

// Neutral categorical colors — baseline vs. ML model, never "winner" coding.
const BASELINE_COLOR = "#5b6478";
const MODEL_COLOR = "#5b9dff";

const CORRELATION_METRICS = new Set(["pearson_corr", "spearman_corr"]);

/** Restrained horizontal bar comparison for one metric across all six
 * approaches — used by the Model Comparison metric explorer. Horizontal
 * bars read cleanly with six variable-length model labels and handle the
 * (small, sometimes negative) correlation values without a diverging-axis
 * layout change. */
export default function ModelMetricChart({ models, metric }: ModelMetricChartProps) {
  const isCorrelation = CORRELATION_METRICS.has(metric.key);

  const data = useMemo<Data[]>(
    () => [
      {
        type: "bar",
        orientation: "h",
        x: models.map((m) => m[metric.key]),
        y: models.map((m) => m.label),
        marker: { color: models.map((m) => (m.source === "frozen_baseline_constant" ? BASELINE_COLOR : MODEL_COLOR)) },
        hovertemplate: isCorrelation ? "%{y}: %{x:.4f}<extra></extra>" : "%{y}: %{x:.2%}<extra></extra>",
      },
    ],
    [models, metric, isCorrelation],
  );

  const layout = useMemo<Partial<Layout>>(
    () => ({
      ...DARK_CHART_LAYOUT,
      margin: { t: 20, r: 24, b: 40, l: 132 },
      xaxis: {
        ...DARK_CHART_LAYOUT.xaxis,
        tickformat: isCorrelation ? ".3f" : ".1%",
        zeroline: true,
        zerolinecolor: "rgba(148, 163, 184, 0.28)",
      },
      yaxis: DARK_CHART_LAYOUT.yaxis,
    }),
    [isCorrelation],
  );

  return (
    <PlotlyChart
      data={data}
      layout={layout}
      className="chart"
      ariaLabel={`${metric.label} compared across Historical Mean, Momentum 3M, Ridge, XGBoost, LSTM, and Ensemble`}
    />
  );
}
