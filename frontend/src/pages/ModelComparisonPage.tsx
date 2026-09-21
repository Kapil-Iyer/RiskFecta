import { useCallback, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Link, useOutletContext } from "react-router-dom";
import type { AppOutletContext } from "../AppShell";
import { getModelComparison } from "../api/client";
import type { MetricDefinition, ModelMetricRow } from "../api/types";
import { useAsync } from "../hooks/useAsync";
import { cn } from "../lib/utils";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../components/ui/card";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "../components/ui/table";
import LoadingState from "../components/LoadingState";
import ErrorState from "../components/ErrorState";
import ModelMetricChart from "../components/ModelMetricChart";
import PageHeader from "../components/PageHeader";
import { sectionMotion, sectionTransition } from "../pageMotion";

function formatMetricValue(row: ModelMetricRow, metric: MetricDefinition): string {
  const value = row[metric.key];
  if (metric.key === "pearson_corr" || metric.key === "spearman_corr") {
    return value.toFixed(4);
  }
  return `${(value * 100).toFixed(2)}%`;
}

function isExtremum(models: ModelMetricRow[], row: ModelMetricRow, metric: MetricDefinition): boolean {
  if (metric.direction === "context_dependent") return false; // never mark a "best" tiny correlation
  const values = models.map((m) => m[metric.key]);
  const extreme = metric.direction === "lower_is_better" ? Math.min(...values) : Math.max(...values);
  return row[metric.key] === extreme;
}

export default function ModelComparisonPage() {
  const { summaryState } = useOutletContext<AppOutletContext>();
  const [attempt, setAttempt] = useState(0);
  const state = useAsync(getModelComparison, [attempt]);
  const retry = useCallback(() => setAttempt((n) => n + 1), []);

  const [selectedMetricKey, setSelectedMetricKey] = useState<MetricDefinition["key"]>("mae");

  const metricDefinitions = state.status === "success" ? state.data.metric_definitions : [];
  const selectedMetric = metricDefinitions.find((m) => m.key === selectedMetricKey) ?? metricDefinitions[0];

  const chartModels = useMemo(() => {
    if (state.status !== "success" || !selectedMetric) return [];
    return [...state.data.models].sort((a, b) => {
      const av = a[selectedMetric.key];
      const bv = b[selectedMetric.key];
      return selectedMetric.direction === "lower_is_better" ? av - bv : bv - av;
    });
  }, [state, selectedMetric]);

  return (
    <motion.div {...sectionMotion} transition={sectionTransition} className="app-page">
      <PageHeader
        title="Model Comparison"
        description="What each forecasting approach actually achieved during the frozen historical walk-forward experiment — baselines and machine-learning models, scored identically, side by side."
      />

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 rounded-md border border-border bg-secondary/40 px-4 py-2 text-xs text-muted-foreground">
        <span>
          Bloomberg data through{" "}
          <span className="font-mono text-foreground">
            {summaryState.status === "success" ? (summaryState.data.last_date ?? "—") : "…"}
          </span>
        </span>
        <span aria-hidden="true">·</span>
        {state.status === "success" && (
          <span>
            {state.data.fold_count} formation dates · {state.data.prediction_count.toLocaleString()} OOS forecasts
            · {state.data.formation_date_start} → {state.data.formation_date_end} ·{" "}
            {state.data.target_horizon_sessions}-session target
          </span>
        )}
        <span aria-hidden="true">·</span>
        <span>Research &amp; education only — not investment advice</span>
        <span aria-hidden="true">·</span>
        <Link to="/methodology" className="underline underline-offset-2 hover:text-foreground">
          Methodology &amp; limitations
        </Link>
      </div>

      {state.status === "loading" && (
        <Card>
          <CardContent className="pt-5">
            <LoadingState label="Loading model comparison…" />
          </CardContent>
        </Card>
      )}
      {state.status === "error" && (
        <Card>
          <CardContent className="pt-5">
            <ErrorState message={state.message} onRetry={retry} />
          </CardContent>
        </Card>
      )}

      {state.status === "success" && selectedMetric && (
        <>
          <Card className="p-0">
            <CardHeader className="border-b border-border py-4">
              <CardTitle>Comparison matrix</CardTitle>
              <CardDescription>
                All six approaches, scored identically against the same realized 21-session returns.
                Bolded values mark the best result in each column — not an overall winner.
              </CardDescription>
            </CardHeader>
            <CardContent className="px-0 pb-0">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Model / baseline</TableHead>
                    {metricDefinitions.map((m) => (
                      <TableHead key={m.key} className="text-right">
                        {m.label}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {state.data.models.map((row) => (
                    <TableRow key={row.model}>
                      <TableCell className="font-medium text-foreground">
                        {row.label}
                        {row.source === "frozen_baseline_constant" && (
                          <span className="ml-2 text-[10px] uppercase tracking-wide text-muted-foreground">
                            baseline
                          </span>
                        )}
                      </TableCell>
                      {metricDefinitions.map((m) => (
                        <TableCell
                          key={m.key}
                          className={cn(
                            "text-right font-mono tabular-nums",
                            isExtremum(state.data.models, row, m) ? "font-semibold text-foreground" : "text-muted-foreground",
                          )}
                        >
                          {formatMetricValue(row, m)}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Metric explorer</CardTitle>
              <CardDescription>{selectedMetric.description}</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div className="flex flex-wrap gap-1" role="group" aria-label="Select metric">
                {metricDefinitions.map((m) => (
                  <Button
                    key={m.key}
                    type="button"
                    size="sm"
                    variant={selectedMetricKey === m.key ? "default" : "outline"}
                    aria-pressed={selectedMetricKey === m.key}
                    onClick={() => setSelectedMetricKey(m.key)}
                  >
                    {m.label}
                  </Button>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                {selectedMetric.direction === "lower_is_better" && "Lower is better."}
                {selectedMetric.direction === "higher_is_better" && "Higher is better."}
                {selectedMetric.direction === "context_dependent" &&
                  "Interpret in context — a small value here reflects weak signal, not strong predictive power."}
              </p>
              <ModelMetricChart models={chartModels} metric={selectedMetric} />
            </CardContent>
          </Card>

          {state.data.disagreement && (
            <Card>
              <CardHeader>
                <CardTitle>Model disagreement</CardTitle>
                <CardDescription>
                  How often the XGBoost and LSTM branches disagreed on direction, before the ensemble
                  averages them together.
                </CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-4 text-sm sm:grid-cols-4">
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">Prediction correlation</span>
                  <span className="font-mono text-foreground">
                    {state.data.disagreement.xgb_lstm_pred_pearson.toFixed(3)}
                  </span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">Residual correlation</span>
                  <span className="font-mono text-foreground">{state.data.disagreement.residual_pearson.toFixed(3)}</span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">Directional disagreement</span>
                  <span className="font-mono text-foreground">
                    {state.data.disagreement.n_disagree} / {state.data.disagreement.n_total}
                  </span>
                </div>
                <div className="flex flex-col gap-1">
                  <span className="text-xs text-muted-foreground">Disagreement rate</span>
                  <span className="font-mono text-foreground">
                    {((state.data.disagreement.n_disagree / state.data.disagreement.n_total) * 100).toFixed(1)}%
                  </span>
                </div>
              </CardContent>
              <CardContent className="pt-0 text-xs text-muted-foreground">
                The two branches' predictions correlate only moderately (~0.34) but their residuals against the
                realized return correlate strongly (~0.83) — most of what separates them is shared error, not
                complementary signal. This is a modest, not a strong, diversification benefit.
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Research conclusion</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-2 text-sm text-muted-foreground">
              <p>
                Standalone predictive signal was weak across every approach — no model or baseline showed a
                strong, dependable relationship between its forecast and the realized 21-session return.
              </p>
              <p>
                Historical Mean remained difficult to beat on conventional error and directional metrics —
                lowest MAE, lowest RMSE, and the highest directional accuracy of the six approaches. XGBoost
                showed the strongest linear (Pearson) relationship with realized returns. The 50/50 Ensemble
                slightly improved rank correlation (Spearman) and reduced some model-specific variance relative
                to either branch alone.
              </p>
              <p>
                The Ensemble was frozen for downstream portfolio construction ahead of the sealed March 2026
                holdout evaluation — a methodology decision made before observing further results, not a claim
                that it "won" this comparison. No single approach dominated across all five metrics.
              </p>
            </CardContent>
          </Card>
        </>
      )}
    </motion.div>
  );
}
