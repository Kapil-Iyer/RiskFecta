import { Link } from "react-router-dom";
import type { UniverseTicker } from "../../api/types";
import { Button } from "../ui/button";
import UniverseVisualization from "./UniverseVisualization";

interface OverviewHeroProps {
  universe: UniverseTicker[];
}

export default function OverviewHero({ universe }: OverviewHeroProps) {
  return (
    <div className="grid grid-cols-1 items-center gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
      <div className="flex flex-col gap-4">
        <span className="w-fit rounded-full border border-border bg-secondary/40 px-3 py-1 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          Quantitative Portfolio Intelligence
        </span>
        <h1 className="text-3xl font-semibold leading-tight text-foreground sm:text-4xl">
          From Bloomberg market data to constrained portfolio research.
        </h1>
        <p className="max-w-lg text-sm text-muted-foreground sm:text-base">
          RiskFecta turns five years of historical Bloomberg data for a fixed 50-equity universe into
          21-session forward-return forecasts (pooled XGBoost + LSTM, frozen 50/50 ensemble), historical
          covariance estimates, and long-only constrained mean-variance portfolios — every step walk-forward
          validated, never randomly split.
        </p>
        <div className="flex flex-wrap gap-2 pt-1">
          <Button asChild size="sm">
            <Link to="/forecasts">Explore Forecasts</Link>
          </Button>
          <Button asChild size="sm" variant="outline">
            <Link to="/models">Compare Models</Link>
          </Button>
          <Button asChild size="sm" variant="outline">
            <Link to="/universe">Explore Universe</Link>
          </Button>
          <Button asChild size="sm" variant="ghost">
            <Link to="/methodology">Read the methodology</Link>
          </Button>
        </div>
      </div>
      <UniverseVisualization universe={universe} />
    </div>
  );
}
