import { useCallback, useState } from "react";
import { MotionConfig } from "framer-motion";
import { NavLink, Outlet } from "react-router-dom";
import { getMarketSummary } from "./api/client";
import type { MarketSummaryResponse } from "./api/types";
import { useAsync, type AsyncState } from "./hooks/useAsync";
import { cn } from "./lib/utils";
import { NAV_ITEMS } from "./navigation";
import Header from "./components/Header";
import DataFreshnessBanner from "./components/DataFreshnessBanner";
import BackendStatusBanner from "./components/BackendStatusBanner";

/** Shared with routed pages (currently just UniversePage) via
 * `useOutletContext` so the market-summary fetch stays here — one fetch on
 * app load, reused by the freshness banner and by any page that needs it,
 * instead of every page re-fetching it independently. */
export interface AppOutletContext {
  summaryState: AsyncState<MarketSummaryResponse>;
  retrySummary: () => void;
}

export default function AppShell() {
  const [summaryAttempt, setSummaryAttempt] = useState(0);
  const summaryState = useAsync(getMarketSummary, [summaryAttempt]);
  const retrySummary = useCallback(() => setSummaryAttempt((n) => n + 1), []);

  const lastDate = summaryState.status === "success" ? summaryState.data.last_date : undefined;

  return (
    <MotionConfig reducedMotion="user">
      <div className="app-shell">
        <Header />
        <DataFreshnessBanner lastDate={lastDate} />
        <BackendStatusBanner />

        <nav
          aria-label="Research surfaces"
          className="flex gap-1 overflow-x-auto border-b border-border bg-background px-[clamp(1rem,4vw,3rem)] py-2"
        >
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              end={item.path === "/"}
              className={({ isActive }) =>
                cn(
                  "shrink-0 rounded-md px-3 py-1.5 text-sm font-medium text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground",
                  isActive && "bg-secondary text-foreground",
                )
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <main className="app-main">
          <Outlet context={{ summaryState, retrySummary } satisfies AppOutletContext} />
        </main>

        <footer className="app-footer">
          <p>RiskFecta — academic / research project. Not investment advice.</p>
        </footer>
      </div>
    </MotionConfig>
  );
}
