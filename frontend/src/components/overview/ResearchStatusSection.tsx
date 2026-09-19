import { Link } from "react-router-dom";
import { NAV_ITEMS } from "../../navigation";

interface ResearchStatusSectionProps {
  /** Real evidence that the underlying research is actually reachable —
   * never a hardcoded assumption. */
  universeReachable: boolean;
  forecastsReachable: boolean;
  modelComparisonReachable: boolean;
  portfolioReachable: boolean;
}

interface StatusColumn {
  heading: string;
  items: string[];
}

const DASHBOARD_LIVE_PATHS = new Set([
  "/", "/universe", "/forecasts", "/models", "/portfolio", "/frontier", "/risk", "/methodology",
]);

/**
 * Replaces the old `BuildStatusSection` with a clearer three-way split:
 * research-engine completion (Phases 3-7) vs. which dashboard surfaces are
 * actually live vs. which are still being built — never conflating "the
 * research is done" with "the page exists," in either direction.
 */
export default function ResearchStatusSection({
  universeReachable,
  forecastsReachable,
  modelComparisonReachable,
  portfolioReachable,
}: ResearchStatusSectionProps) {
  const liveNav = NAV_ITEMS.filter((n) => DASHBOARD_LIVE_PATHS.has(n.path));
  const comingNav = NAV_ITEMS.filter((n) => !DASHBOARD_LIVE_PATHS.has(n.path));

  const columns: StatusColumn[] = [
    {
      heading: "Research engine — complete",
      items: [
        "Bloomberg historical data pipeline" + (universeReachable ? " (verified live)" : ""),
        "Feature engineering",
        "XGBoost + LSTM walk-forward forecasting" + (forecastsReachable ? " (verified live)" : ""),
        "Frozen 50/50 ensemble" + (modelComparisonReachable ? " (verified live)" : ""),
        "Sample + Ledoit-Wolf covariance",
        "Constrained Min-Vol / Max-Sharpe optimization",
        "Phase 7 historical portfolio research" + (portfolioReachable ? " (verified live)" : ""),
      ],
    },
    {
      heading: "Dashboard — live now",
      items: liveNav.map((n) => n.label),
    },
    {
      heading: "Dashboard — coming next",
      items: comingNav.map((n) => n.label),
    },
  ];

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      {columns.map((col) => (
        <div key={col.heading} className="rounded-md border border-border bg-secondary/20 p-4">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">{col.heading}</h3>
          <ul className="flex flex-col gap-1.5 text-sm text-foreground">
            {col.items.map((item) => (
              <li key={item} className="flex items-start gap-2">
                <span aria-hidden="true" className="mt-1.5 h-1 w-1 shrink-0 rounded-full bg-current" />
                {item}
              </li>
            ))}
          </ul>
        </div>
      ))}
      <p className="md:col-span-3 text-xs text-muted-foreground">
        Research completion and dashboard-surface completion are tracked separately: a "complete" research item may not
        yet have its own interactive page. See{" "}
        <Link to="/methodology" className="underline underline-offset-2 hover:text-foreground">
          Methodology &amp; limitations
        </Link>{" "}
        for the full picture.
      </p>
    </div>
  );
}
