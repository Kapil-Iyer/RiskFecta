import { useCallback, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import type { Data, Layout } from "plotly.js-cartesian-dist-min";
import { getFrontier, getPortfolioDates } from "../api/client";
import type { FrontierMarker } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import PlotlyChart from "../components/PlotlyChart";
import { DARK_CHART_LAYOUT } from "../components/chartTheme";
import { sectionMotion, sectionTransition } from "../pageMotion";

function formatPct(value: number | null, decimals = 2): string {
  if (value === null) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

function formatSignedPct(value: number, decimals = 2): string {
  const pct = value * 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(decimals)}%`;
}

const COVARIANCE_OPTIONS: { key: string; label: string }[] = [
  { key: "LW", label: "Ledoit-Wolf" },
  { key: "SAMPLE", label: "Sample" },
];

const CURVE_COLOR = "#5b9dff";
const MIN_VOL_COLOR = "#34d399";
const MAX_SHARPE_COLOR = "#f59e0b";
const EQUAL_WEIGHT_COLOR = "#93a1b8";

function markerTrace(marker: FrontierMarker, color: string, symbol: string): Data {
  return {
    type: "scatter",
    mode: "markers",
    name: marker.label,
    x: [marker.volatility_21],
    y: [marker.expected_return_21],
    marker: { color, size: 13, symbol, line: { color: "#05070d", width: 1.5 } },
    hovertemplate:
      `<b>${marker.label}</b><br>` +
      "Expected return: %{y:.2%}<br>" +
      "Predicted volatility: %{x:.2%}<br>" +
      (marker.sharpe_21 !== null ? `Sharpe: ${marker.sharpe_21.toFixed(3)}<br>` : "") +
      (marker.provenance === "official_phase7_persisted" ? "Official Phase 7 result" : "Reconstructed benchmark") +
      "<extra></extra>",
  };
}

export default function FrontierPage() {
  const [datesAttempt, setDatesAttempt] = useState(0);
  const datesState = useAsync(getPortfolioDates, [datesAttempt]);
  const retryDates = useCallback(() => setDatesAttempt((n) => n + 1), []);

  const [selectedDate, setSelectedDate] = useState<string | undefined>(undefined);
  const [selectedCovariance, setSelectedCovariance] = useState<string>("LW");
  const [frontierAttempt, setFrontierAttempt] = useState(0);
  const frontierState = useAsync(
    () => getFrontier(selectedDate, selectedCovariance),
    [selectedDate, selectedCovariance, frontierAttempt],
  );
  const retryFrontier = useCallback(() => setFrontierAttempt((n) => n + 1), []);

  const dateOptions = datesState.status === "success" ? [...datesState.data].reverse() : [];
  const effectiveSelectedDate =
    selectedDate ?? (frontierState.status === "success" ? frontierState.data.formation_date : "");

  const chartData = useMemo<Data[]>(() => {
    if (frontierState.status !== "success") return [];
    const { points, markers } = frontierState.data;
    const curve: Data = {
      type: "scatter",
      mode: "lines+markers",
      name: "Reconstructed frontier",
      x: points.map((p) => p.volatility_21),
      y: points.map((p) => p.expected_return_21),
      line: { color: CURVE_COLOR, width: 2 },
      marker: { color: CURVE_COLOR, size: 5 },
      hovertemplate:
        "Expected return: %{y:.2%}<br>Predicted volatility: %{x:.2%}<extra>Frontier point</extra>",
    };
    return [
      curve,
      markerTrace(markers.equal_weight, EQUAL_WEIGHT_COLOR, "diamond"),
      markerTrace(markers.min_vol, MIN_VOL_COLOR, "star"),
      markerTrace(markers.max_sharpe, MAX_SHARPE_COLOR, "star"),
    ];
  }, [frontierState]);

  const layout = useMemo<Partial<Layout>>(
    () => ({
      ...DARK_CHART_LAYOUT,
      margin: { t: 20, r: 20, b: 56, l: 64 },
      xaxis: {
        ...DARK_CHART_LAYOUT.xaxis,
        title: { text: "Predicted 21-session volatility" },
        tickformat: ".1%",
      },
      yaxis: {
        ...DARK_CHART_LAYOUT.yaxis,
        title: { text: "Expected 21-session return" },
        tickformat: ".1%",
        zeroline: true,
        zerolinecolor: "rgba(148, 163, 184, 0.28)",
      },
      legend: { orientation: "h", y: -0.22, font: { color: "#93a1b8", size: 11 } },
      showlegend: true,
    }),
    [],
  );

  return (
    <motion.div {...sectionMotion} transition={sectionTransition} className="app-page">
      <div className="flex flex-col gap-1">
        <h1 className="text-xl font-semibold text-foreground">Efficient Frontier</h1>
        <p className="max-w-3xl text-sm text-muted-foreground">
          At this historical formation date, what risk/expected-return trade-offs were feasible under RiskFecta's
          frozen portfolio constraints? Reconstructed from the frozen expected-return signal and a historical
          covariance estimate — never persisted, never a live/forward-looking promise.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-md border border-border bg-secondary/40 px-4 py-2 text-xs text-muted-foreground">
        <span>21-session horizon</span>
        <span aria-hidden="true">·</span>
        <span>252-session covariance window</span>
        <span aria-hidden="true">·</span>
        <span>10% max position weight</span>
        <span aria-hidden="true">·</span>
        <span>Historical research — not investment advice</span>
        <span aria-hidden="true">·</span>
        <Link to="/portfolio" className="underline underline-offset-2 hover:text-foreground">
          Portfolio Construction
        </Link>
        <span aria-hidden="true">·</span>
        <Link to="/methodology" className="underline underline-offset-2 hover:text-foreground">
          Methodology &amp; limitations
        </Link>
      </div>

      <Card>
        <CardContent className="flex flex-wrap items-end gap-5 pt-5">
          <div className="flex flex-col gap-1">
            <label htmlFor="frontier-date-select" className="text-xs font-medium text-muted-foreground">
              Formation date (historical)
            </label>
            {datesState.status === "error" ? (
              <ErrorState message={datesState.message} onRetry={retryDates} />
            ) : (
              <select
                id="frontier-date-select"
                className="h-9 min-w-[10rem] rounded-md border border-input bg-secondary px-3 text-sm text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                value={effectiveSelectedDate}
                disabled={datesState.status === "loading"}
                onChange={(e) => setSelectedDate(e.target.value)}
              >
                {datesState.status === "loading" && <option>Loading dates…</option>}
                {dateOptions.map((d) => (
                  <option key={d} value={d}>
                    {d}
                  </option>
                ))}
              </select>
            )}
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Covariance estimator</span>
            <div className="flex flex-wrap gap-1" role="group" aria-label="Covariance estimator">
              {COVARIANCE_OPTIONS.map((c) => (
                <Button
                  key={c.key}
                  type="button"
                  size="sm"
                  variant={selectedCovariance === c.key ? "default" : "outline"}
                  aria-pressed={selectedCovariance === c.key}
                  onClick={() => setSelectedCovariance(c.key)}
                >
                  {c.label}
                </Button>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>

      {frontierState.status === "loading" && (
        <Card>
          <CardContent className="pt-5">
            <LoadingState label="Reconstructing frontier — several SLSQP solves, may take a few seconds…" />
          </CardContent>
        </Card>
      )}
      {frontierState.status === "error" && (
        <Card>
          <CardContent className="pt-5">
            <ErrorState message={frontierState.message} onRetry={retryFrontier} />
          </CardContent>
        </Card>
      )}

      {frontierState.status === "success" && (
        <>
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              { label: "Covariance estimator", value: frontierState.data.covariance_estimator },
              { label: "Frontier points", value: `${frontierState.data.points.length}` },
              { label: "Max weight (hard cap)", value: formatPct(frontierState.data.max_weight_constraint) },
              { label: "Risk-free rate (21-session)", value: formatPct(frontierState.data.risk_free_rate_21, 3) },
            ].map((s) => (
              <div key={s.label} className="rounded-md border border-border bg-secondary/30 px-3 py-2.5">
                <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{s.label}</dt>
                <dd className="mt-0.5 font-mono text-sm font-semibold text-foreground">{s.value}</dd>
              </div>
            ))}
          </dl>

          <Card className="p-0">
            <CardHeader className="border-b border-border py-4">
              <CardTitle>Risk/return frontier</CardTitle>
              <CardDescription>
                Predicted 21-session volatility (x) vs. expected 21-session return (y), both construction-time
                estimates — never realized outcomes. Moving right means more predicted risk; moving up means higher
                expected return.
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-4">
              <PlotlyChart
                data={chartData}
                layout={layout}
                className="chart h-[420px] w-full"
                ariaLabel="Reconstructed efficient frontier with official Min-Vol, Max-Sharpe, and Equal Weight markers"
              />
            </CardContent>
          </Card>

          <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
            {[
              { marker: frontierState.data.markers.min_vol, color: MIN_VOL_COLOR },
              { marker: frontierState.data.markers.max_sharpe, color: MAX_SHARPE_COLOR },
              { marker: frontierState.data.markers.equal_weight, color: EQUAL_WEIGHT_COLOR },
            ].map(({ marker, color }) => (
              <Card key={marker.label}>
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-sm">
                    <span aria-hidden="true" className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: color }} />
                    {marker.label}
                  </CardTitle>
                  <CardDescription>
                    {marker.provenance === "official_phase7_persisted"
                      ? "Official Phase 7 persisted result"
                      : "Reconstructed benchmark — not implied to lie on the frontier"}
                  </CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col gap-1.5 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Expected return</span>
                    <span className="font-mono font-semibold text-foreground">
                      {formatSignedPct(marker.expected_return_21)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Predicted volatility</span>
                    <span className="font-mono font-semibold text-foreground">{formatPct(marker.volatility_21)}</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Sharpe</span>
                    <span className="font-mono font-semibold text-foreground">
                      {marker.sharpe_21 !== null ? marker.sharpe_21.toFixed(3) : "—"}
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Methodology</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-2 text-sm text-muted-foreground">
              <p>
                The dense curve was never persisted in Phase 7 — only five discrete strategies were. It is
                reconstructed here from the frozen 50/50 XGBoost + LSTM ensemble expected returns and a 252-session
                historical covariance estimate, scaled to the 21-session horizon, under the same long-only, fully
                invested, 10% max-weight constraints. Min-Vol and Max-Sharpe markers are the OFFICIAL persisted
                Phase 7 results, never a freshly re-optimized copy. Equal Weight is a fixed 1/50 benchmark and is not
                implied to lie on the frontier. Expected return and predicted volatility are construction-time
                estimates only — see{" "}
                <Link to="/portfolio" className="underline underline-offset-2 hover:text-foreground">
                  Portfolio Construction
                </Link>{" "}
                for realized (ex-post) outcomes.
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
