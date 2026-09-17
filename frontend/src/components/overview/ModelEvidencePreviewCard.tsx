import { Link } from "react-router-dom";
import type { ModelComparisonResponse } from "../../api/types";
import type { AsyncState } from "../../hooks/useAsync";
import { Button } from "../ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "../ui/card";
import LoadingState from "../LoadingState";
import ErrorState from "../ErrorState";

interface ModelEvidencePreviewCardProps {
  state: AsyncState<ModelComparisonResponse>;
  onRetry: () => void;
}

/** Honestly previews the real Model Comparison result — standalone forecast
 * evidence was mixed/weak; this is never hidden to look more impressive. */
export default function ModelEvidencePreviewCard({ state, onRetry }: ModelEvidencePreviewCardProps) {
  const historicalMean = state.status === "success" ? state.data.models.find((m) => m.model === "historical_mean") : undefined;
  const ensemble = state.status === "success" ? state.data.models.find((m) => m.model === "ensemble_pred") : undefined;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Model evidence preview</CardTitle>
        <CardDescription>How each forecasting approach performed in the frozen historical walk-forward evaluation.</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-3">
        {state.status === "loading" && <LoadingState label="Loading model evidence…" />}
        {state.status === "error" && <ErrorState message={state.message} onRetry={onRetry} />}
        {state.status === "success" && historicalMean && ensemble && (
          <p className="text-sm text-muted-foreground">
            Standalone predictive signal was weak overall. <strong className="text-foreground">Historical Mean</strong>{" "}
            remained the hardest to beat on conventional error/direction metrics (MAE {(historicalMean.mae * 100).toFixed(2)}%,
            directional accuracy {(historicalMean.directional_accuracy * 100).toFixed(2)}%), while the{" "}
            <strong className="text-foreground">50/50 Ensemble</strong> showed only limited correlation with realized
            returns (Pearson {ensemble.pearson_corr.toFixed(3)}). No approach dominated across every metric.
          </p>
        )}
        <Button asChild size="sm" variant="outline" className="self-start">
          <Link to="/models">See the full comparison</Link>
        </Button>
      </CardContent>
    </Card>
  );
}
