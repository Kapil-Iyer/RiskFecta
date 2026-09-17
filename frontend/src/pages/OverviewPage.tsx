import { useCallback, useState } from "react";
import { motion } from "framer-motion";
import { Link, useOutletContext } from "react-router-dom";
import type { AppOutletContext } from "../AppShell";
import { getModelComparison, getPortfolioDates, getPredictions, getUniverse } from "../api/client";
import { useAsync } from "../hooks/useAsync";
import OverviewHero from "../components/overview/OverviewHero";
import ResearchSnapshot from "../components/overview/ResearchSnapshot";
import ResearchPipelineStory from "../components/overview/ResearchPipelineStory";
import ForecastPreviewCard from "../components/overview/ForecastPreviewCard";
import ModelEvidencePreviewCard from "../components/overview/ModelEvidencePreviewCard";
import ResearchStatusSection from "../components/overview/ResearchStatusSection";
import { sectionMotion, sectionTransition } from "../pageMotion";

function SectionHeading({ children }: { children: string }) {
  return <h2 className="text-lg font-semibold text-foreground">{children}</h2>;
}

export default function OverviewPage() {
  const { summaryState } = useOutletContext<AppOutletContext>();

  const [universeAttempt, setUniverseAttempt] = useState(0);
  const universeState = useAsync(getUniverse, [universeAttempt]);
  const retryUniverse = useCallback(() => setUniverseAttempt((n) => n + 1), []);

  const [predictionsAttempt, setPredictionsAttempt] = useState(0);
  const predictionsState = useAsync(() => getPredictions(), [predictionsAttempt]);
  const retryPredictions = useCallback(() => setPredictionsAttempt((n) => n + 1), []);

  const [modelsAttempt, setModelsAttempt] = useState(0);
  const modelComparisonState = useAsync(getModelComparison, [modelsAttempt]);
  const retryModels = useCallback(() => setModelsAttempt((n) => n + 1), []);

  // Lightweight reachability check only (dates list, not a full portfolio
  // fetch) — just enough real evidence for the status section below.
  const portfolioDatesState = useAsync(getPortfolioDates, []);

  const universe = universeState.status === "success" ? universeState.data : [];

  return (
    <motion.div {...sectionMotion} transition={sectionTransition} className="app-page">
      <OverviewHero universe={universe} />

      {universeState.status === "error" && (
        <p className="text-xs text-muted-foreground">
          Universe visualization unavailable: {universeState.message}{" "}
          <button type="button" className="underline" onClick={retryUniverse}>
            Retry
          </button>
        </p>
      )}

      <ResearchSnapshot
        universe={universe}
        foldCount={modelComparisonState.status === "success" ? modelComparisonState.data.fold_count : null}
        predictionCount={modelComparisonState.status === "success" ? modelComparisonState.data.prediction_count : null}
        targetHorizonSessions={
          modelComparisonState.status === "success" ? modelComparisonState.data.target_horizon_sessions : null
        }
        dataThrough={summaryState.status === "success" ? summaryState.data.last_date : undefined}
      />

      <div className="flex flex-col gap-3">
        <SectionHeading>Research pipeline</SectionHeading>
        <ResearchPipelineStory />
      </div>

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ForecastPreviewCard state={predictionsState} onRetry={retryPredictions} />
        <ModelEvidencePreviewCard state={modelComparisonState} onRetry={retryModels} />
      </div>

      <div className="flex flex-col gap-3">
        <SectionHeading>Research engine vs. dashboard status</SectionHeading>
        <ResearchStatusSection
          universeReachable={universeState.status === "success"}
          forecastsReachable={predictionsState.status === "success"}
          modelComparisonReachable={modelComparisonState.status === "success"}
          portfolioReachable={portfolioDatesState.status === "success"}
        />
      </div>

      <div className="rounded-md border border-border bg-secondary/30 px-4 py-3">
        <p className="text-sm text-muted-foreground">
          Historical Bloomberg data only, through {summaryState.status === "success" ? (summaryState.data.last_date ?? "—") : "…"}.
          March 2026 is a sealed holdout reserved for a one-time Phase 9 evaluation — not yet touched. See{" "}
          <Link to="/methodology" className="underline underline-offset-2 hover:text-foreground">
            Methodology &amp; limitations
          </Link>{" "}
          for the full picture.
        </p>
      </div>
    </motion.div>
  );
}
