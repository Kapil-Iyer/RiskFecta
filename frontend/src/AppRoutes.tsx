import { Route, Routes } from "react-router-dom";
import AppShell from "./AppShell";
import OverviewPage from "./pages/OverviewPage";
import UniversePage from "./pages/UniversePage";
import ForecastRankingsPage from "./pages/ForecastRankingsPage";
import ModelComparisonPage from "./pages/ModelComparisonPage";
import PortfolioConstructionPage from "./pages/PortfolioConstructionPage";
import MethodologyPage from "./pages/MethodologyPage";
import ComingSoonPage from "./pages/ComingSoonPage";

/** Route tree, separated from `App.tsx` so tests can render it inside a
 * `MemoryRouter` (controlling the initial route) instead of the real
 * `BrowserRouter`. */
export default function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/universe" element={<UniversePage />} />
        <Route path="/forecasts" element={<ForecastRankingsPage />} />
        <Route path="/models" element={<ModelComparisonPage />} />
        <Route path="/portfolio" element={<PortfolioConstructionPage />} />
        <Route
          path="/frontier"
          element={
            <ComingSoonPage
              title="Efficient Frontier"
              description="Interactive risk/return frontier with minimum-volatility, maximum-Sharpe, and benchmark markers."
            />
          }
        />
        <Route
          path="/risk"
          element={
            <ComingSoonPage
              title="Risk Analytics"
              description="Portfolio volatility, concentration, and per-asset risk contribution."
            />
          }
        />
        <Route
          path="/backtest"
          element={
            <ComingSoonPage
              title="Historical Evidence"
              description="The walk-forward record model and methodology selection was based on, vs. SPXT and equal-weight."
            />
          }
        />
        <Route path="/methodology" element={<MethodologyPage />} />
      </Route>
    </Routes>
  );
}
