import { useCallback, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Link, useOutletContext } from "react-router-dom";
import type { AppOutletContext } from "../AppShell";
import { getPortfolio, getPortfolioDates, getPortfolioStrategies, getUniverse } from "../api/client";
import type { HoldingRow } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { sectorForTicker } from "../data/tickerUniverse";
import { sectorAttr, themeForSector } from "../theme";
import { cn } from "../lib/utils";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import PageHeader from "../components/PageHeader";
import { sectionMotion, sectionTransition } from "../pageMotion";

function formatPct(value: number | null, decimals = 2): string {
  if (value === null) return "—";
  return `${(value * 100).toFixed(decimals)}%`;
}

function formatSignedPct(value: number | null, decimals = 2): string {
  if (value === null) return "—";
  const pct = value * 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(decimals)}%`;
}

function returnColorClass(value: number | null): string {
  if (value === null) return "text-muted-foreground";
  return value >= 0 ? "text-[var(--color-live)]" : "text-[var(--color-error)]";
}

interface RankedHolding extends HoldingRow {
  rank: number;
}

function rankHoldings(holdings: HoldingRow[]): RankedHolding[] {
  return [...holdings]
    .sort((a, b) => (a.weight === b.weight ? a.ticker.localeCompare(b.ticker) : b.weight - a.weight))
    .map((h, i) => ({ ...h, rank: i + 1 }));
}

export default function PortfolioConstructionPage() {
  const { summaryState } = useOutletContext<AppOutletContext>();

  const [datesAttempt, setDatesAttempt] = useState(0);
  const datesState = useAsync(getPortfolioDates, [datesAttempt]);
  const retryDates = useCallback(() => setDatesAttempt((n) => n + 1), []);

  const strategiesState = useAsync(getPortfolioStrategies, []);
  const universeState = useAsync(getUniverse, []);

  const [selectedDate, setSelectedDate] = useState<string | undefined>(undefined);
  const [selectedStrategy, setSelectedStrategy] = useState<string | undefined>(undefined);
  const [portfolioAttempt, setPortfolioAttempt] = useState(0);
  const portfolioState = useAsync(
    () => getPortfolio(selectedDate, selectedStrategy),
    [selectedDate, selectedStrategy, portfolioAttempt],
  );
  const retryPortfolio = useCallback(() => setPortfolioAttempt((n) => n + 1), []);

  const sectorByTicker = useMemo(() => {
    const map = new Map<string, string | null>();
    if (universeState.status === "success") {
      for (const row of universeState.data) map.set(row.ticker, row.sector);
    }
    return map;
  }, [universeState]);

  const ranked = useMemo(() => {
    if (portfolioState.status !== "success") return [];
    return rankHoldings(portfolioState.data.holdings);
  }, [portfolioState]);

  const sectorExposure = useMemo(() => {
    if (portfolioState.status !== "success") return null;
    let it = 0;
    let fin = 0;
    for (const h of portfolioState.data.holdings) {
      const sector = sectorByTicker.get(h.ticker) ?? sectorForTicker(h.ticker);
      if (sector === "Information Technology") it += h.weight;
      else if (sector === "Financials") fin += h.weight;
    }
    return { it, fin };
  }, [portfolioState, sectorByTicker]);

  const dateOptions = datesState.status === "success" ? [...datesState.data].reverse() : [];
  const effectiveSelectedDate =
    selectedDate ?? (portfolioState.status === "success" ? portfolioState.data.formation_date : "");
  const effectiveSelectedStrategy =
    selectedStrategy ?? (portfolioState.status === "success" ? portfolioState.data.strategy.key : "");

  const scaleMax: number =
    portfolioState.status === "success"
      ? (portfolioState.data.max_weight_constraint ?? Math.max(portfolioState.data.largest_weight, 0.02))
      : 0.1;

  return (
    <motion.div {...sectionMotion} transition={sectionTransition} className="app-page">
      <PageHeader
        title="Historical Portfolio Construction"
        description="Given the frozen ML expected-return signal and a historical covariance estimate, what portfolio did RiskFecta construct at this historical formation date? This is a historical research construction — not a live recommendation, current allocation, or forward-looking promise."
      />

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-md border border-border bg-secondary/40 px-4 py-2 text-xs text-muted-foreground">
        <span>
          Bloomberg data through{" "}
          <span className="font-mono text-foreground">
            {summaryState.status === "success" ? (summaryState.data.last_date ?? "—") : "…"}
          </span>
        </span>
        <span aria-hidden="true">·</span>
        <span>
          Portfolio calendar begins 2022-03-28 — the 252-session covariance estimator needs 253 prior TRI levels, one
          more than the first forecasting formation (2022-02-25) had available.
        </span>
        <span aria-hidden="true">·</span>
        <span>Historical research — not investment advice</span>
        <span aria-hidden="true">·</span>
        <Link to="/methodology" className="underline underline-offset-2 hover:text-foreground">
          Methodology &amp; limitations
        </Link>
      </div>

      <Card>
        <CardContent className="flex flex-wrap items-end gap-5 pt-5">
          <div className="flex flex-col gap-1">
            <label htmlFor="portfolio-date-select" className="text-xs font-medium text-muted-foreground">
              Formation date (historical)
            </label>
            {datesState.status === "error" ? (
              <ErrorState message={datesState.message} onRetry={retryDates} />
            ) : (
              <select
                id="portfolio-date-select"
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
        </CardContent>
      </Card>

      {portfolioState.status === "loading" && (
        <Card>
          <CardContent className="pt-5">
            <LoadingState label="Loading portfolio…" />
          </CardContent>
        </Card>
      )}
      {portfolioState.status === "error" && (
        <Card>
          <CardContent className="pt-5">
            <ErrorState message={portfolioState.message} onRetry={retryPortfolio} />
          </CardContent>
        </Card>
      )}

      {portfolioState.status === "success" && (
        <>
          <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
            {[
              { label: "Universe", value: "50 equities" },
              { label: "Active holdings", value: `${portfolioState.data.active_holdings_count}` },
              { label: "Largest weight", value: formatPct(portfolioState.data.largest_weight) },
              {
                label: "Max weight (hard cap)",
                value: portfolioState.data.max_weight_constraint !== null ? formatPct(portfolioState.data.max_weight_constraint) : "N/A",
              },
              { label: "Covariance estimator", value: portfolioState.data.strategy.covariance_estimator },
              { label: "Concentration (HHI)", value: portfolioState.data.concentration_hhi.toFixed(4) },
            ].map((s) => (
              <div key={s.label} className="rounded-md border border-border bg-secondary/30 px-3 py-2.5">
                <dt className="text-[11px] uppercase tracking-wide text-muted-foreground">{s.label}</dt>
                <dd className="mt-0.5 font-mono text-sm font-semibold text-foreground">{s.value}</dd>
              </div>
            ))}
          </dl>

          <p className="text-xs text-muted-foreground">
            {portfolioState.data.strategy.objective}
            {portfolioState.data.max_weight_constraint !== null && " The 10% cap frequently binds — a 10% weight may reflect the optimizer constraint, not model conviction alone."}
          </p>

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
            <Card className="p-0">
              <CardHeader className="border-b border-border py-4">
                <CardTitle>Portfolio weights</CardTitle>
                <CardDescription>
                  Ranked by weight, highest first. The dashed line marks the {portfolioState.data.max_weight_constraint !== null ? "10% hard cap" : "equal-weight reference"}.
                </CardDescription>
              </CardHeader>
              <CardContent className="px-0 pb-0">
                <div className="overflow-x-auto">
                  <div role="table" aria-label="Portfolio weights" className="min-w-[28rem] text-sm">
                    <div
                      role="row"
                      className="grid grid-cols-[3rem_5.5rem_1fr_5rem] items-center border-b border-border bg-secondary/30 px-4 py-2 text-xs font-medium uppercase tracking-wide text-muted-foreground"
                    >
                      <span role="columnheader">#</span>
                      <span role="columnheader">Ticker</span>
                      <span role="columnheader">Weight</span>
                      <span role="columnheader" className="text-right">
                        Value
                      </span>
                    </div>
                    <div role="rowgroup">
                      {ranked.map((h) => {
                        const sector = sectorByTicker.get(h.ticker) ?? sectorForTicker(h.ticker);
                        const theme = themeForSector(sector);
                        const barPct = Math.min(100, (h.weight / scaleMax) * 100);
                        const capPct = portfolioState.data.max_weight_constraint !== null ? Math.min(100, (portfolioState.data.max_weight_constraint / scaleMax) * 100) : null;
                        return (
                          <motion.div
                            key={h.ticker}
                            layout
                            transition={{ duration: 0.22 }}
                            role="row"
                            data-sector={sectorAttr(sector)}
                            style={{ ["--local-accent" as string]: theme.cssVar }}
                            className="grid grid-cols-[3rem_5.5rem_1fr_5rem] items-center border-b border-border px-4 py-2 hover:bg-secondary/40"
                          >
                            <span role="cell" className="font-mono text-xs text-muted-foreground">
                              {h.rank}
                            </span>
                            <span role="cell" className="font-mono font-semibold text-foreground">
                              {h.ticker}
                            </span>
                            <span role="cell" className="relative h-4 rounded bg-secondary/40">
                              <span
                                className="absolute inset-y-0 left-0 rounded"
                                style={{ width: `${barPct}%`, backgroundColor: "var(--local-accent)" }}
                              />
                              {capPct !== null && (
                                <span
                                  className="absolute inset-y-0 border-l border-dashed border-[var(--color-error)]"
                                  style={{ left: `${capPct}%` }}
                                  aria-hidden="true"
                                />
                              )}
                            </span>
                            <span role="cell" className="text-right font-mono tabular-nums text-foreground">
                              {formatPct(h.weight)}
                            </span>
                          </motion.div>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="flex flex-col gap-4">
              <Card>
                <CardHeader>
                  <CardTitle>Sector exposure</CardTitle>
                </CardHeader>
                <CardContent className="flex flex-col gap-2 text-sm">
                  {sectorExposure && (
                    <>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Information Technology</span>
                        <span className="font-mono font-semibold text-foreground">{formatPct(sectorExposure.it)}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Financials</span>
                        <span className="font-mono font-semibold text-foreground">{formatPct(sectorExposure.fin)}</span>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Construction — ex-ante</CardTitle>
                  <CardDescription>Predicted at formation time, from the frozen expected-return/covariance inputs.</CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col gap-2 text-sm">
                  {portfolioState.data.strategy.is_optimized ? (
                    <>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Expected 21-session return</span>
                        <span className={cn("font-mono font-semibold", returnColorClass(portfolioState.data.construction.expected_return_21))}>
                          {formatSignedPct(portfolioState.data.construction.expected_return_21)}
                        </span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Predicted volatility</span>
                        <span className="font-mono font-semibold text-foreground">{formatPct(portfolioState.data.construction.predicted_volatility_21)}</span>
                      </div>
                      <div className="flex items-center justify-between">
                        <span className="text-muted-foreground">Expected Sharpe</span>
                        <span className="font-mono font-semibold text-foreground">
                          {portfolioState.data.construction.expected_sharpe_21?.toFixed(3) ?? "—"}
                        </span>
                      </div>
                    </>
                  ) : (
                    <p className="text-xs text-muted-foreground">
                      Not applicable — Equal Weight is a benchmark, not an optimized construction.
                    </p>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Realized outcome — ex-post</CardTitle>
                  <CardDescription>Known only after the 21-session period passed — never construction-time information.</CardDescription>
                </CardHeader>
                <CardContent className="flex flex-col gap-2 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Realized 21-session return</span>
                    <span className={cn("font-mono font-semibold", returnColorClass(portfolioState.data.evaluation.realized_return_21))}>
                      {formatSignedPct(portfolioState.data.evaluation.realized_return_21)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Turnover vs. prior formation</span>
                    <span className="font-mono font-semibold text-foreground">
                      {portfolioState.data.evaluation.turnover === null ? "Not defined for first formation" : formatPct(portfolioState.data.evaluation.turnover)}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-muted-foreground">Max weight observed</span>
                    <span className="font-mono font-semibold text-foreground">{formatPct(portfolioState.data.evaluation.max_weight_observed)}</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>Methodology</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-2 text-sm text-muted-foreground">
              <p>
                Long-only, fully invested, hard 10% maximum single-position weight. Expected returns are the frozen
                50/50 XGBoost + LSTM ensemble forecast (Max-Sharpe strategies only — Min-Vol does not use expected
                returns at all). Covariance is estimated from 252 sessions of historical realized returns, scaled to
                the 21-session horizon. Equal Weight is a fixed 1/50 benchmark, never optimized.
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
