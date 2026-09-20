import { useCallback, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import type { Data, Layout } from "plotly.js-cartesian-dist-min";
import { getBacktest } from "../api/client";
import type { BacktestSeries } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import PlotlyChart from "../components/PlotlyChart";
import { DARK_CHART_LAYOUT } from "../components/chartTheme";
import { sectionMotion, sectionTransition } from "../pageMotion";

function formatPct(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

function formatSignedPct(value: number, decimals = 1): string {
  const pct = value * 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(decimals)}%`;
}

const DEFAULT_SELECTED = "LW_MAXSHARPE"; // one of six series — not "the best one"

const SERIES_COLOR: Record<string, string> = {
  SAMPLE_MINVOL: "#5b9dff",
  SAMPLE_MAXSHARPE: "#a78bfa",
  LW_MINVOL: "#34d399",
  LW_MAXSHARPE: "#f59e0b",
  EQUAL_WEIGHT: "#93a1b8",
  SPXT: "#f472b6",
};

/** The chart's real 47 x-axis points: the anchor (first formation date,
 * growth=1.0, before any period has been realized) plus one point per
 * period at its target_date — never a fabricated intra-period value. */
function toGrowthTrace(series: BacktestSeries): Data {
  const first = series.periods[0];
  const x = [first.formation_date, ...series.periods.map((p) => p.target_date)];
  const y = [1.0, ...series.periods.map((p) => p.growth_of_one)];
  return {
    type: "scatter",
    mode: "lines",
    name: series.label,
    x,
    y,
    line: {
      color: SERIES_COLOR[series.key] ?? "#5b9dff",
      width: series.kind === "benchmark" ? 2 : 2.5,
      dash: series.kind === "benchmark" ? "dot" : "solid",
    },
    hovertemplate: `<b>${series.label}</b><br>%{x}: growth of $1 = %{y:.3f}<extra></extra>`,
  };
}

export default function HistoricalEvidencePage() {
  const [attempt, setAttempt] = useState(0);
  const backtestState = useAsync(getBacktest, [attempt]);
  const retry = useCallback(() => setAttempt((n) => n + 1), []);
  const [selectedKey, setSelectedKey] = useState<string | undefined>(undefined);

  const series = backtestState.status === "success" ? backtestState.data.series : [];
  const effectiveSelectedKey = selectedKey ?? DEFAULT_SELECTED;
  const selected = series.find((s) => s.key === effectiveSelectedKey) ?? series[0];

  const growthChartData = useMemo<Data[]>(() => series.map(toGrowthTrace), [series]);
  const growthChartLayout = useMemo<Partial<Layout>>(
    () => ({
      ...DARK_CHART_LAYOUT,
      margin: { t: 20, r: 20, b: 40, l: 56 },
      xaxis: { ...DARK_CHART_LAYOUT.xaxis, title: { text: "Formation / evaluation date" } },
      yaxis: { ...DARK_CHART_LAYOUT.yaxis, title: { text: "Growth of $1" } },
      legend: { orientation: "h", y: -0.2, font: { color: "#93a1b8", size: 11 } },
      showlegend: true,
    }),
    [],
  );

  const periodReturnData = useMemo<Data[]>(() => {
    if (!selected) return [];
    return [
      {
        type: "bar",
        name: "21-session return",
        x: selected.periods.map((p) => p.formation_date),
        y: selected.periods.map((p) => p.realized_return_21),
        marker: {
          color: selected.periods.map((p) => (p.realized_return_21 >= 0 ? "#34d399" : "#f87171")),
        },
        hovertemplate: "%{x}: %{y:.2%}<extra></extra>",
      },
    ];
  }, [selected]);

  const periodReturnLayout = useMemo<Partial<Layout>>(
    () => ({
      ...DARK_CHART_LAYOUT,
      margin: { t: 10, r: 20, b: 40, l: 56 },
      xaxis: { ...DARK_CHART_LAYOUT.xaxis, title: { text: "Formation date" } },
      yaxis: { ...DARK_CHART_LAYOUT.yaxis, title: { text: "Realized 21-session return" }, tickformat: ".0%", zeroline: true, zerolinecolor: "rgba(148,163,184,0.4)" },
    }),
    [],
  );

  const turnoverData = useMemo<Data[]>(() => {
    if (!selected || selected.kind !== "portfolio") return [];
    return [
      {
        type: "bar",
        name: "Turnover",
        x: selected.periods.map((p) => p.formation_date),
        y: selected.periods.map((p) => p.turnover ?? null),
        marker: { color: "#5b9dff" },
        hovertemplate: "%{x}: %{y:.1%}<extra></extra>",
      },
    ];
  }, [selected]);

  const turnoverLayout = useMemo<Partial<Layout>>(
    () => ({
      ...DARK_CHART_LAYOUT,
      margin: { t: 10, r: 20, b: 40, l: 56 },
      xaxis: { ...DARK_CHART_LAYOUT.xaxis, title: { text: "Formation date" } },
      yaxis: { ...DARK_CHART_LAYOUT.yaxis, title: { text: "Turnover" }, tickformat: ".0%" },
    }),
    [],
  );

  return (
    <motion.div {...sectionMotion} transition={sectionTransition} className="app-page">
      <div className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold text-foreground">Historical Evidence</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          Frozen walk-forward portfolio evidence across 46 non-overlapping 21-session periods — the already-computed
          Phase 7 experiment, presented as-is. This is not a new backtest and not a strategy ranking.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-md border border-border bg-secondary/40 px-4 py-2 text-xs text-muted-foreground">
        <span>Historical walk-forward</span>
        <span aria-hidden="true">·</span>
        <span>46 periods</span>
        <span aria-hidden="true">·</span>
        <span>21-session horizon</span>
        <span aria-hidden="true">·</span>
        <span>
          Data through{" "}
          {backtestState.status === "success" ? backtestState.data.experiment.data_through : "…"}
        </span>
        <span aria-hidden="true">·</span>
        <span className="font-medium text-foreground">Sealed March holdout pending</span>
      </div>

      <Card>
        <CardContent className="flex flex-col gap-2 pt-5 text-sm text-muted-foreground">
          <p>
            These are historical walk-forward results — realized outcomes from a frozen, already-run experiment, not
            a live or forward-looking guarantee. The sealed March 2026 holdout remains untouched and is not
            reflected anywhere on this page. No transaction costs or slippage are modeled; this is especially
            relevant for the Max-Sharpe portfolios, which exhibited substantially higher turnover than the Min-Vol
            portfolios or Equal Weight.
          </p>
          <p>
            Standalone return forecasts showed weak/mixed predictive performance (see{" "}
            <Link to="/models" className="underline underline-offset-2 hover:text-foreground">
              Model Comparison
            </Link>
            ). The results below test whether those frozen signals nevertheless contained useful cross-sectional
            information once passed through constrained portfolio optimization — historical evidence does not
            establish future performance.
          </p>
          <Link to="/methodology" className="w-fit underline underline-offset-2 hover:text-foreground">
            Methodology &amp; limitations
          </Link>
        </CardContent>
      </Card>

      {backtestState.status === "loading" && (
        <Card>
          <CardContent className="pt-5">
            <LoadingState label="Loading historical evidence…" />
          </CardContent>
        </Card>
      )}
      {backtestState.status === "error" && (
        <Card>
          <CardContent className="pt-5">
            <ErrorState message={backtestState.message} onRetry={retry} />
          </CardContent>
        </Card>
      )}

      {backtestState.status === "success" && selected && (
        <>
          <Card className="p-0">
            <CardHeader className="border-b border-border py-4">
              <CardTitle>Cumulative growth of $1</CardTitle>
              <CardDescription>
                All six series start at exactly $1.00 on {backtestState.data.experiment.first_formation_date} and
                compound sequentially across the 46 periods. Click a legend entry to toggle a series.
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-4">
              <PlotlyChart
                data={growthChartData}
                layout={growthChartLayout}
                className="chart h-[420px] w-full"
                ariaLabel="Cumulative growth of $1 for all six strategies and SPXT across 46 historical walk-forward periods"
              />
            </CardContent>
          </Card>

          <Card>
            <CardContent className="flex flex-wrap items-end gap-5 pt-5">
              <div className="flex flex-col gap-1">
                <span className="text-xs font-medium text-muted-foreground">Inspect a series</span>
                <div className="flex flex-wrap gap-1" role="group" aria-label="Series to inspect">
                  {series.map((s) => (
                    <Button
                      key={s.key}
                      type="button"
                      size="sm"
                      variant={effectiveSelectedKey === s.key ? "default" : "outline"}
                      aria-pressed={effectiveSelectedKey === s.key}
                      onClick={() => setSelectedKey(s.key)}
                    >
                      {s.label}
                    </Button>
                  ))}
                </div>
              </div>
            </CardContent>
          </Card>

          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
            {[
              { label: "Mean 21-session return", value: formatSignedPct(selected.summary.mean_return_21) },
              { label: "Std deviation", value: formatPct(selected.summary.std_return_21) },
              { label: "Median", value: formatSignedPct(selected.summary.median_return_21) },
              { label: "Min", value: formatSignedPct(selected.summary.min_return_21) },
              { label: "Max", value: formatSignedPct(selected.summary.max_return_21) },
              { label: "Positive-period rate", value: formatPct(selected.summary.positive_period_rate) },
              { label: "Cumulative (compounded)", value: formatSignedPct(selected.summary.cumulative_return) },
            ].map((s) => (
              <div key={s.label} className="rounded-md border border-border bg-secondary/30 px-3 py-2.5">
                <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{s.label}</dt>
                <dd className="mt-0.5 font-mono text-sm font-semibold text-foreground">{s.value}</dd>
              </div>
            ))}
          </dl>

          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              { label: "Mean turnover", value: selected.summary.mean_turnover !== null ? formatPct(selected.summary.mean_turnover) : "N/A" },
              { label: "Median turnover", value: selected.summary.median_turnover !== null ? formatPct(selected.summary.median_turnover) : "N/A" },
              { label: "Max turnover", value: selected.summary.max_turnover !== null ? formatPct(selected.summary.max_turnover) : "N/A" },
              {
                label: "Avg / max observed weight",
                value:
                  selected.summary.avg_max_weight !== null
                    ? `${formatPct(selected.summary.avg_max_weight)} / ${formatPct(selected.summary.max_observed_weight ?? 0)}`
                    : "N/A",
              },
            ].map((s) => (
              <div key={s.label} className="rounded-md border border-border bg-secondary/20 px-3 py-2.5">
                <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{s.label}</dt>
                <dd className="mt-0.5 font-mono text-sm font-semibold text-foreground">{s.value}</dd>
              </div>
            ))}
          </dl>
          {selected.max_weight_constraint !== null && (
            <p className="text-xs text-muted-foreground">
              Hard {formatPct(selected.max_weight_constraint, 0)} single-name cap — the constraint frequently bound
              for optimized strategies (average and max observed weight above).
            </p>
          )}
          {selected.kind === "benchmark" && (
            <p className="text-xs text-muted-foreground">Turnover and weight concentration are not applicable to an index benchmark.</p>
          )}

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            <Card className="p-0">
              <CardHeader className="border-b border-border py-4">
                <CardTitle>Period returns — {selected.label}</CardTitle>
                <CardDescription>Each of the 46 realized, non-overlapping 21-session returns.</CardDescription>
              </CardHeader>
              <CardContent className="pt-4">
                <PlotlyChart
                  data={periodReturnData}
                  layout={periodReturnLayout}
                  className="chart h-[320px] w-full"
                  ariaLabel={`Realized 21-session period returns for ${selected.label}`}
                />
              </CardContent>
            </Card>

            <Card className="p-0">
              <CardHeader className="border-b border-border py-4">
                <CardTitle>Turnover — {selected.label}</CardTitle>
                <CardDescription>
                  {selected.kind === "portfolio"
                    ? "Portfolio turnover at each rebalance. The first formation has no prior portfolio to compare against."
                    : "Not applicable — SPXT is an index benchmark, not a rebalanced portfolio."}
                </CardDescription>
              </CardHeader>
              <CardContent className="pt-4">
                {selected.kind === "portfolio" ? (
                  <PlotlyChart
                    data={turnoverData}
                    layout={turnoverLayout}
                    className="chart h-[320px] w-full"
                    ariaLabel={`Turnover at each formation date for ${selected.label}`}
                  />
                ) : (
                  <p className="py-10 text-center text-sm text-muted-foreground">N/A</p>
                )}
              </CardContent>
            </Card>
          </div>

          <p className="text-xs text-muted-foreground">
            First-formation turnover ({selected.periods[0]?.formation_date}) is{" "}
            <span className="font-mono font-semibold text-foreground">
              {selected.kind !== "portfolio"
                ? "not applicable"
                : selected.periods[0].turnover === null
                  ? "not defined for the first formation"
                  : formatPct(selected.periods[0].turnover)}
            </span>{" "}
            — there is no prior portfolio to compare against, and this is never shown as zero.
          </p>

          <Card>
            <CardHeader>
              <CardTitle>Methodology</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-2 text-sm text-muted-foreground">
              <p>
                Every figure above is either read directly from the official, persisted Phase 7 experiment
                (realized return, turnover, max weight observed) or computed via the exact frozen aggregation and
                sequential-compounding formulas used to produce that experiment's original report — never a new
                backtest, never re-tuned using these outcomes, never annualized. Sample and Ledoit-Wolf covariance
                produced nearly identical realized portfolio behavior in this historical sample, despite Ledoit-Wolf
                substantially improving covariance conditioning — shrinkage is not shown here as having improved
                returns.
              </p>
              <Link to="/methodology" className="w-fit underline underline-offset-2 hover:text-foreground">
                Read the full methodology
              </Link>
            </CardContent>
          </Card>
        </>
      )}
    </motion.div>
  );
}
