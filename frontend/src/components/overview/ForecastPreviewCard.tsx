import { Link } from "react-router-dom";
import type { PredictionCrossSectionResponse } from "../../api/types";
import { rankPredictions } from "../../pages/forecastRanking";
import { Button } from "../ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import LoadingState from "../LoadingState";
import ErrorState from "../ErrorState";
import type { AsyncState } from "../../hooks/useAsync";

interface ForecastPreviewCardProps {
  state: AsyncState<PredictionCrossSectionResponse>;
  onRetry: () => void;
}

function formatReturn(value: number | null): string {
  if (value === null) return "—";
  const pct = value * 100;
  return `${pct > 0 ? "+" : ""}${pct.toFixed(2)}%`;
}

/** A small, real-data preview of the Forecast Rankings page — never the
 * full 50-row table, and never framed as a live/current recommendation. */
export default function ForecastPreviewCard({ state, onRetry }: ForecastPreviewCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Forecast preview</CardTitle>
        <CardDescription>
          {state.status === "success"
            ? `Historical walk-forward forecast — formation date ${state.data.formation_date}, ranked by the 50/50 ensemble.`
            : "Historical walk-forward forecast ranking, ensemble-ranked."}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {state.status === "loading" && <LoadingState label="Loading forecast preview…" />}
        {state.status === "error" && <ErrorState message={state.message} onRetry={onRetry} />}
        {state.status === "success" && (
          <ul className="flex flex-col gap-1.5" aria-label="Top 3 ensemble-ranked stocks, historical forecast">
            {rankPredictions(state.data.predictions, "ensemble")
              .slice(0, 3)
              .map((p) => (
                <li key={p.ticker} className="flex items-center justify-between rounded-md bg-secondary/30 px-3 py-1.5 text-sm">
                  <span className="font-mono font-semibold text-foreground">
                    #{p.rank} {p.ticker}
                  </span>
                  <span className={p.ensemble_pred !== null && p.ensemble_pred >= 0 ? "text-[var(--color-live)]" : "text-[var(--color-error)]"}>
                    {formatReturn(p.ensemble_pred)}
                  </span>
                </li>
              ))}
          </ul>
        )}
        <Button asChild size="sm" variant="outline" className="self-start">
          <Link to="/forecasts">See all 50 rankings</Link>
        </Button>
      </CardContent>
    </Card>
  );
}
