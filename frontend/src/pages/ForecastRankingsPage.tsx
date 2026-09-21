import { useCallback, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Link, useOutletContext } from "react-router-dom";
import type { AppOutletContext } from "../AppShell";
import { getPredictionDates, getPredictions, getUniverse } from "../api/client";
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
import { MODEL_OPTIONS, rankPredictions, type ModelKey } from "./forecastRanking";

const GRID_COLS_BASE = "grid-cols-[3rem_5.5rem_minmax(7rem,1fr)_7rem_7rem_7rem]";
const GRID_COLS_EVAL = "grid-cols-[3rem_5.5rem_minmax(7rem,1fr)_7rem_7rem_7rem_9rem]";

function formatReturn(value: number | null): string {
  if (value === null) return "—";
  const pct = value * 100;
  const sign = pct > 0 ? "+" : "";
  return `${sign}${pct.toFixed(2)}%`;
}

function returnColorClass(value: number | null): string {
  if (value === null) return "text-muted-foreground";
  return value >= 0 ? "text-[var(--color-live)]" : "text-[var(--color-error)]";
}

export default function ForecastRankingsPage() {
  const { summaryState } = useOutletContext<AppOutletContext>();

  const [datesAttempt, setDatesAttempt] = useState(0);
  const datesState = useAsync(getPredictionDates, [datesAttempt]);
  const retryDates = useCallback(() => setDatesAttempt((n) => n + 1), []);

  const [selectedDate, setSelectedDate] = useState<string | undefined>(undefined);
  const [predictionsAttempt, setPredictionsAttempt] = useState(0);
  const predictionsState = useAsync(() => getPredictions(selectedDate), [selectedDate, predictionsAttempt]);
  const retryPredictions = useCallback(() => setPredictionsAttempt((n) => n + 1), []);

  const universeState = useAsync(getUniverse, []);

  const [selectedModel, setSelectedModel] = useState<ModelKey>("ensemble");
  const [showEvaluation, setShowEvaluation] = useState(false);

  const sectorByTicker = useMemo(() => {
    const map = new Map<string, string | null>();
    if (universeState.status === "success") {
      for (const row of universeState.data) map.set(row.ticker, row.sector);
    }
    return map;
  }, [universeState]);

  const ranked = useMemo(() => {
    if (predictionsState.status !== "success") return [];
    return rankPredictions(predictionsState.data.predictions, selectedModel);
  }, [predictionsState, selectedModel]);

  // Latest-first for the dropdown; the API's own base order (ticker
  // ascending within a formation date) is unrelated and unaffected.
  const dateOptions = datesState.status === "success" ? [...datesState.data].reverse() : [];
  const effectiveSelectedDate =
    selectedDate ?? (predictionsState.status === "success" ? predictionsState.data.formation_date : "");

  return (
    <motion.div {...sectionMotion} transition={sectionTransition} className="app-page">
      <PageHeader
        title="Forecast Rankings"
        description="21-trading-session forward-return forecasts from the frozen pooled XGBoost model, pooled LSTM model, and their 50/50 ensemble — historical walk-forward out-of-sample evidence, not a live or current-day market forecast."
      />

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-md border border-border bg-secondary/40 px-4 py-2 text-xs text-muted-foreground">
        <span>
          Bloomberg data through{" "}
          <span className="font-mono text-foreground">
            {summaryState.status === "success" ? (summaryState.data.last_date ?? "—") : "…"}
          </span>
        </span>
        <span aria-hidden="true">·</span>
        <span>Historical walk-forward out-of-sample evidence (Phase 4–6)</span>
        <span aria-hidden="true">·</span>
        <span>Research &amp; education only — not investment advice</span>
        <span aria-hidden="true">·</span>
        <Link to="/methodology" className="underline underline-offset-2 hover:text-foreground">
          Methodology &amp; limitations
        </Link>
      </div>

      <Card>
        <CardContent className="flex flex-wrap items-end gap-5 pt-5">
          <div className="flex flex-col gap-1">
            <label htmlFor="formation-date-select" className="text-xs font-medium text-muted-foreground">
              Formation date (historical walk-forward)
            </label>
            {datesState.status === "error" ? (
              <ErrorState message={datesState.message} onRetry={retryDates} />
            ) : (
              <select
                id="formation-date-select"
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
            <span className="text-xs font-medium text-muted-foreground">Rank by</span>
            <div className="flex gap-1" role="group" aria-label="Ranking model">
              {MODEL_OPTIONS.map((m) => (
                <Button
                  key={m.key}
                  type="button"
                  size="sm"
                  variant={selectedModel === m.key ? "default" : "outline"}
                  aria-pressed={selectedModel === m.key}
                  onClick={() => setSelectedModel(m.key)}
                >
                  {m.label}
                </Button>
              ))}
            </div>
          </div>

          <div className="ml-auto flex flex-col items-end gap-1">
            <span className="text-xs font-medium text-muted-foreground">Realized outcome</span>
            <Button
              type="button"
              size="sm"
              variant={showEvaluation ? "default" : "outline"}
              aria-pressed={showEvaluation}
              onClick={() => setShowEvaluation((v) => !v)}
            >
              {showEvaluation ? "Hide evaluation" : "Show evaluation"}
            </Button>
          </div>
        </CardContent>
      </Card>

      <Card className="overflow-hidden p-0">
        <CardHeader className="border-b border-border py-4">
          <CardTitle>50-stock forecast cross-section</CardTitle>
          <CardDescription>
            {predictionsState.status === "success"
              ? `Formation date ${predictionsState.data.formation_date} — realized outcome known as of ${predictionsState.data.target_date} (21 trading sessions later).`
              : "Ranked highest predicted 21-session return to lowest."}
          </CardDescription>
        </CardHeader>
        <CardContent className="px-0 pb-0">
          {predictionsState.status === "loading" && (
            <div className="px-5 py-6">
              <LoadingState label="Loading forecast cross-section…" />
            </div>
          )}
          {predictionsState.status === "error" && (
            <div className="px-5 py-6">
              <ErrorState message={predictionsState.message} onRetry={retryPredictions} />
            </div>
          )}
          {predictionsState.status === "success" && (
            <div className="overflow-x-auto">
              <div role="table" aria-label="Forecast rankings" className="min-w-[42rem] text-sm">
                <div
                  role="row"
                  className={cn(
                    "grid items-center border-b border-border bg-secondary/30 px-4 py-2 text-xs font-medium uppercase tracking-wide text-muted-foreground",
                    showEvaluation ? GRID_COLS_EVAL : GRID_COLS_BASE,
                  )}
                >
                  <span role="columnheader">#</span>
                  <span role="columnheader">Ticker</span>
                  <span role="columnheader">Sector</span>
                  {MODEL_OPTIONS.map((m) => (
                    <span
                      key={m.key}
                      role="columnheader"
                      className={cn("text-right", selectedModel === m.key && "text-foreground")}
                    >
                      {m.label}
                    </span>
                  ))}
                  {showEvaluation && (
                    <span role="columnheader" className="text-right">
                      Realized (21d)
                    </span>
                  )}
                </div>

                <div role="rowgroup">
                  {ranked.map((p) => {
                    const sector = sectorByTicker.get(p.ticker) ?? sectorForTicker(p.ticker);
                    const theme = themeForSector(sector);
                    return (
                      <motion.div
                        key={p.ticker}
                        layout
                        transition={{ duration: 0.22 }}
                        role="row"
                        data-sector={sectorAttr(sector)}
                        style={{ ["--local-accent" as string]: theme.cssVar }}
                        className={cn(
                          "grid items-center border-b border-border px-4 py-2 hover:bg-secondary/40",
                          showEvaluation ? GRID_COLS_EVAL : GRID_COLS_BASE,
                        )}
                      >
                        <span role="cell" className="font-mono text-xs text-muted-foreground">
                          {p.rank}
                        </span>
                        <span role="cell" className="font-mono font-semibold text-foreground">
                          {p.ticker}
                        </span>
                        <span role="cell" className="truncate text-xs" style={{ color: "var(--local-accent)" }}>
                          {sector ?? "—"}
                        </span>
                        {MODEL_OPTIONS.map((m) => (
                          <span
                            key={m.key}
                            role="cell"
                            className={cn(
                              "text-right font-mono tabular-nums",
                              returnColorClass(p[m.field]),
                              selectedModel === m.key ? "font-semibold" : "opacity-70",
                            )}
                          >
                            {formatReturn(p[m.field])}
                          </span>
                        ))}
                        {showEvaluation && (
                          <span role="cell" className="flex flex-col items-end">
                            <span className={cn("font-mono tabular-nums", returnColorClass(p.actual_return))}>
                              {formatReturn(p.actual_return)}
                            </span>
                            <span className="text-[10px] text-muted-foreground">
                              {p.directional_correct === null
                                ? ""
                                : p.directional_correct
                                  ? "✓ direction correct"
                                  : "✗ direction wrong"}
                            </span>
                          </span>
                        )}
                      </motion.div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
