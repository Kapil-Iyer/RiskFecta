import { Route, Routes } from "react-router-dom";
import AppShell from "./AppShell";
import OverviewPage from "./pages/OverviewPage";
import UniversePage from "./pages/UniversePage";
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
        <Route
          path="/forecasts"
          element={
            <ComingSoonPage
              title="Forecast Rankings"
              description="XGBoost, LSTM, and ensemble forward-return forecasts for the current formation date, ranked."
            />
          }
        />
        <Route
          path="/models"
          element={
            <ComingSoonPage
              title="Model Comparison"
              description="Walk-forward out-of-sample evidence across baselines, XGBoost, LSTM, and the ensemble."
            />
          }
        />
        <Route
          path="/portfolio"
          element={
            <ComingSoonPage
              title="Portfolio Construction"
              description="Optimized portfolio weights and statistics from the constrained mean-variance optimizer."
            />
          }
        />
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
