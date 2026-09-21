import { useCallback, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import type { Data, Layout } from "plotly.js-cartesian-dist-min";
import { getPortfolioDates, getPortfolioStrategies, getRisk } from "../api/client";
import { useAsync } from "../hooks/useAsync";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import PageHeader from "../components/PageHeader";
import PlotlyChart from "../components/PlotlyChart";
import { DARK_CHART_LAYOUT } from "../components/chartTheme";
import { sectionMotion, sectionTransition } from "../pageMotion";

function formatPct(value: number, decimals = 2): string {
  return `${(value * 100).toFixed(decimals)}%`;
}

const WEIGHT_COLOR = "#5b9dff";
const RISK_SHARE_COLOR = "#f59e0b";

export default function RiskAnalyticsPage() {
  const [datesAttempt, setDatesAttempt] = useState(0);
  const datesState = useAsync(getPortfolioDates, [datesAttempt]);
  const retryDates = useCallback(() => setDatesAttempt((n) => n + 1), []);

  const strategiesState = useAsync(getPortfolioStrategies, []);

  const [selectedDate, setSelectedDate] = useState<string | undefined>(undefined);
  const [selectedStrategy, setSelectedStrategy] = useState<string | undefined>(undefined);
  const [equalWeightCovariance, setEqualWeightCovariance] = useState<string>("LW");
  const [riskAttempt, setRiskAttempt] = useState(0);

  const isEqualWeight = selectedStrategy === "EQUAL_WEIGHT";
  const riskState = useAsync(
    () => getRisk(selectedDate, selectedStrategy, isEqualWeight ? equalWeightCovariance : undefined),
    [selectedDate, selectedStrategy, isEqualWeight, equalWeightCovariance, riskAttempt],
  );
  const retryRisk = useCallback(() => setRiskAttempt((n) => n + 1), []);

  const dateOptions = datesState.status === "success" ? [...datesState.data].reverse() : [];
  const effectiveSelectedDate = selectedDate ?? (riskState.status === "success" ? riskState.data.formation_date : "");
  const effectiveSelectedStrategy =
    selectedStrategy ?? (riskState.status === "success" ? riskState.data.strategy.key : "");

  const rankedAssets = useMemo(() => {
    if (riskState.status !== "success") return [];
    return [...riskState.data.assets].sort((a, b) => b.component_risk_contribution - a.component_risk_contribution);
  }, [riskState]);

  const chartData = useMemo<Data[]>(() => {
    if (rankedAssets.length === 0) return [];
    return [
      {
        type: "bar",
        orientation: "h",
        name: "Portfolio weight",
        x: rankedAssets.map((a) => a.weight),
        y: rankedAssets.map((a) => a.ticker),
        marker: { color: WEIGHT_COLOR },
        hovertemplate: "%{y} weight: %{x:.2%}<extra></extra>",
      },
      {
        type: "bar",
        orientation: "h",
        name: "Risk share",
        x: rankedAssets.map((a) => a.risk_share),
        y: rankedAssets.map((a) => a.ticker),
        marker: { color: RISK_SHARE_COLOR },
        hovertemplate: "%{y} risk share: %{x:.2%}<extra></extra>",
      },
    ];
  }, [rankedAssets]);

  const chartLayout = useMemo<Partial<Layout>>(
    () => ({
      ...DARK_CHART_LAYOUT,
      barmode: "group",
      margin: { t: 20, r: 24, b: 40, l: 56 },
      xaxis: {
        ...DARK_CHART_LAYOUT.xaxis,
        tickformat: ".1%",
        zeroline: true,
        zerolinecolor: "rgba(148, 163, 184, 0.4)",
      },
      yaxis: { ...DARK_CHART_LAYOUT.yaxis, autorange: "reversed", tickfont: { size: 10, color: "#93a1b8" } },
      legend: { orientation: "h", y: -0.03, font: { color: "#93a1b8", size: 11 } },
      showlegend: true,
    }),
    [],
  );

  return (
    <motion.div {...sectionMotion} transition={sectionTransition} className="app-page">
      <PageHeader
        title="Risk Analytics"
        description="Formation-time predicted portfolio risk under the frozen Phase 7 covariance methodology — which positions actually drive the portfolio's predicted volatility, not just which positions are largest by capital."
      />

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-md border border-border bg-secondary/40 px-4 py-2 text-xs text-muted-foreground">
        <span>21-session horizon</span>
        <span aria-hidden="true">·</span>
        <span>252-session covariance window</span>
        <span aria-hidden="true">·</span>
        <span>Bloomberg data through 2026-02-27</span>
        <span aria-hidden="true">·</span>
        <span>Historical research — not investment advice</span>
        <span aria-hidden="true">·</span>
        <Link to="/portfolio" className="underline underline-offset-2 hover:text-foreground">
          Portfolio Construction
        </Link>
        <span aria-hidden="true">·</span>
        <Link to="/frontier" className="underline underline-offset-2 hover:text-foreground">
          Efficient Frontier
        </Link>
        <span aria-hidden="true">·</span>
        <Link to="/methodology" className="underline underline-offset-2 hover:text-foreground">
          Methodology &amp; limitations
        </Link>
      </div>

      <Card>
        <CardContent className="flex flex-wrap items-end gap-5 pt-5">
          <div className="flex flex-col gap-1">
            <label htmlFor="risk-date-select" className="text-xs font-medium text-muted-foreground">
              Formation date (historical)
            </label>
            {datesState.status === "error" ? (
              <ErrorState message={datesState.message} onRetry={retryDates} />
            ) : (
              <select
                id="risk-date-select"
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
            <span className="text-xs font-medium text-muted-foreground">Strategy</span>
            <div className="flex flex-wrap gap-1" role="group" aria-label="Portfolio strategy">
              {strategiesState.status === "success" &&
                strategiesState.data.map((s) => (
                  <Button
                    key={s.key}
                    type="button"
                    size="sm"
                    variant={effectiveSelectedStrategy === s.key ? "default" : "outline"}
                    aria-pressed={effectiveSelectedStrategy === s.key}
                    onClick={() => setSelectedStrategy(s.key)}
                  >
                    {s.label}
                  </Button>
                ))}
            </div>
          </div>

          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-muted-foreground">Covariance</span>
            {isEqualWeight ? (
              <div className="flex flex-col gap-1">
                <div className="flex flex-wrap gap-1" role="group" aria-label="Covariance estimator for Equal Weight risk analysis">
                  {[
                    { key: "LW", label: "Ledoit-Wolf" },
                    { key: "SAMPLE", label: "Sample" },
                  ].map((c) => (
                    <Button
                      key={c.key}
                      type="button"
                      size="sm"
                      variant={equalWeightCovariance === c.key ? "default" : "outline"}
                      aria-pressed={equalWeightCovariance === c.key}
                      onClick={() => setEqualWeightCovariance(c.key)}
                    >
                      {c.label}
                    </Button>
                  ))}
                </div>
                <span className="text-[11px] text-muted-foreground">
                  Risk analysis using {equalWeightCovariance === "LW" ? "Ledoit-Wolf" : "Sample"} covariance — Equal
                  Weight itself is never constructed from a covariance estimate.
                </span>
              </div>
            ) : (
              <div className="flex h-9 items-center rounded-md border border-input bg-secondary/40 px-3 text-sm text-foreground">
                {riskState.status === "success" ? riskState.data.covariance_estimator : "…"}
                <span className="ml-2 text-[11px] text-muted-foreground">(determined by strategy)</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {riskState.status === "loading" && (
        <Card>
          <CardContent className="pt-5">
            <LoadingState label="Reconstructing formation-time risk…" />
          </CardContent>
        </Card>
      )}
      {riskState.status === "error" && (
        <Card>
          <CardContent className="pt-5">
            <ErrorState message={riskState.message} onRetry={retryRisk} />
          </CardContent>
        </Card>
      )}

      {riskState.status === "success" && (
        <>
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {[
              { label: "Predicted 21-session volatility", value: formatPct(riskState.data.portfolio.predicted_volatility_21) },
              { label: "Largest position", value: formatPct(riskState.data.portfolio.largest_weight) },
              { label: "Active holdings", value: `${riskState.data.portfolio.active_holdings_count}` },
              { label: "Concentration (HHI)", value: riskState.data.portfolio.concentration_hhi.toFixed(4) },
              { label: "Effective holdings", value: riskState.data.portfolio.effective_holdings.toFixed(1) },
              {
                label: "Max weight (hard cap)",
                value:
                  riskState.data.portfolio.max_weight_constraint !== null
                    ? formatPct(riskState.data.portfolio.max_weight_constraint)
                    : "N/A",
              },
            ].map((s) => (
              <div key={s.label} className="rounded-md border border-border bg-secondary/30 px-3 py-2.5">
                <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{s.label}</dt>
                <dd className="mt-0.5 font-mono text-sm font-semibold text-foreground">{s.value}</dd>
              </div>
            ))}
          </dl>

          <Card className="p-0">
            <CardHeader className="border-b border-border py-4">
              <CardTitle>Component risk contribution</CardTitle>
              <CardDescription>
                Ranked by contribution to predicted portfolio volatility, highest first. Portfolio weight (capital)
                vs. risk share (predicted-risk) side by side — a larger risk share is not inherently better or worse
                than a larger weight, just a different lens on the same portfolio.
              </CardDescription>
            </CardHeader>
            <CardContent className="pt-4">
              <PlotlyChart
                data={chartData}
                layout={chartLayout}
                className="chart h-[900px] w-full"
                ariaLabel="Component risk contribution: portfolio weight vs. risk share, ranked by risk contribution"
              />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Sector risk aggregation</CardTitle>
              <CardDescription>
                Component contributions are additive, so sector totals are exact sums — not a separate sector
                covariance model.
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3 text-sm">
              {riskState.data.sectors.map((s) => (
                <div key={s.sector} className="flex flex-col gap-1">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">{s.sector}</span>
                    <span className="font-mono text-xs text-muted-foreground">
                      weight {formatPct(s.weight)} · risk share {formatPct(s.risk_share)}
                    </span>
                  </div>
                  <div className="flex gap-1">
                    <div className="h-2 flex-1 rounded bg-secondary/40">
                      <div className="h-2 rounded" style={{ width: `${Math.max(0, s.weight * 100)}%`, backgroundColor: WEIGHT_COLOR }} />
                    </div>
                    <div className="h-2 flex-1 rounded bg-secondary/40">
                      <div
                        className="h-2 rounded"
                        style={{ width: `${Math.max(0, Math.min(100, s.risk_share * 100))}%`, backgroundColor: RISK_SHARE_COLOR }}
                      />
                    </div>
                  </div>
                </div>
              ))}
              <div className="flex gap-4 text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1">
                  <span aria-hidden="true" className="h-2 w-2 rounded-full" style={{ backgroundColor: WEIGHT_COLOR }} />
                  Portfolio weight
                </span>
                <span className="flex items-center gap-1">
                  <span aria-hidden="true" className="h-2 w-2 rounded-full" style={{ backgroundColor: RISK_SHARE_COLOR }} />
                  Risk share
                </span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Methodology</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-2 text-sm text-muted-foreground">
              <p>
                Covariance is estimated from 252 sessions of historical realized returns as of the formation date,
                scaled to the 21-session horizon — the same reconstruction Efficient Frontier uses. Each asset's
                component contribution to portfolio volatility is computed as RC_i = w_i · (Σw)_i / σ_p, and these
                contributions sum exactly to the portfolio's predicted volatility. Diversification and covariance
                between positions mean risk contribution can differ materially from capital weight — and, because
                covariance can contain negative cross-asset terms, an individual contribution can occasionally be
                negative (a genuine diversifying/hedging effect), which is shown as-is, never hidden or floored at
                zero. These are formation-time, predicted estimates — not realized future risk.
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
