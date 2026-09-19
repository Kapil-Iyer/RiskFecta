import { render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter, Outlet, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import OverviewPage from "./OverviewPage";
import * as client from "../api/client";
import type { AppOutletContext } from "../AppShell";
import { TICKER_UNIVERSE, sectorForTicker } from "../data/tickerUniverse";
import type { ModelComparisonResponse, PredictionCrossSectionResponse, UniverseTicker } from "../api/types";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    getUniverse: vi.fn(),
    getPredictions: vi.fn(),
    getModelComparison: vi.fn(),
    getPortfolioDates: vi.fn(),
  };
});

// The real 50-ticker universe (mirrors config.py), with real sectors — not
// a synthetic/example fixture.
const REAL_UNIVERSE: UniverseTicker[] = TICKER_UNIVERSE.map((ticker) => ({
  ticker,
  sector: sectorForTicker(ticker),
}));

const PREDICTIONS: PredictionCrossSectionResponse = {
  formation_date: "2026-01-02",
  target_date: "2026-02-03",
  count: 3,
  predictions: [
    { ticker: "ORCL", xgb_pred: 0.15, lstm_pred: 0.12, ensemble_pred: 0.14, actual_return: -0.2, directional_correct: false },
    { ticker: "NOW", xgb_pred: 0.1, lstm_pred: 0.09, ensemble_pred: 0.1, actual_return: -0.25, directional_correct: false },
    { ticker: "AMD", xgb_pred: 0.05, lstm_pred: 0.02, ensemble_pred: 0.04, actual_return: 0.08, directional_correct: true },
  ],
};

const MODEL_COMPARISON: ModelComparisonResponse = {
  experiment_type: "historical_walk_forward_oos",
  target_horizon_sessions: 21,
  fold_count: 47,
  prediction_count: 2350,
  formation_date_start: "2022-02-25",
  formation_date_end: "2026-01-02",
  models: [
    { model: "historical_mean", label: "Historical Mean", source: "frozen_baseline_constant", mae: 0.07312, rmse: 0.09722, directional_accuracy: 0.5366, pearson_corr: 0.02854, spearman_corr: 0.01673, n_obs: 2350 },
    { model: "momentum_3m", label: "Momentum 3M", source: "frozen_baseline_constant", mae: 0.14254, rmse: 0.18803, directional_accuracy: 0.5077, pearson_corr: -0.00132, spearman_corr: -0.03109, n_obs: 2350 },
    { model: "ridge", label: "Ridge", source: "frozen_baseline_constant", mae: 0.07831, rmse: 0.10421, directional_accuracy: 0.4761, pearson_corr: 0.00158, spearman_corr: -0.03153, n_obs: 2350 },
    { model: "xgb_pred", label: "XGBoost", source: "computed_from_persisted_predictions", mae: 0.07915, rmse: 0.10453, directional_accuracy: 0.4996, pearson_corr: 0.07105, spearman_corr: 0.00007, n_obs: 2350 },
    { model: "lstm_pred", label: "LSTM", source: "computed_from_persisted_predictions", mae: 0.08156, rmse: 0.1103, directional_accuracy: 0.5085, pearson_corr: 0.02197, spearman_corr: 0.00805, n_obs: 2350 },
    { model: "ensemble_pred", label: "Ensemble", source: "computed_from_persisted_predictions", mae: 0.07717, rmse: 0.10292, directional_accuracy: 0.50724, pearson_corr: 0.05479, spearman_corr: 0.01721, n_obs: 2350 },
  ],
  metric_definitions: [
    { key: "mae", label: "MAE", direction: "lower_is_better", description: "Mean absolute error." },
    { key: "rmse", label: "RMSE", direction: "lower_is_better", description: "Root mean squared error." },
    { key: "directional_accuracy", label: "Directional Accuracy", direction: "higher_is_better", description: "Sign match rate." },
    { key: "pearson_corr", label: "Pearson Correlation", direction: "context_dependent", description: "Linear correlation." },
    { key: "spearman_corr", label: "Spearman Correlation", direction: "context_dependent", description: "Rank correlation." },
  ],
  disagreement: null,
};

const SUCCESS_SUMMARY: AppOutletContext = {
  summaryState: {
    status: "success",
    data: { ticker_count: 50, price_row_count: 62800, first_date: "2021-03-01", last_date: "2026-02-27" },
  },
  retrySummary: vi.fn(),
};

function renderPage(context: AppOutletContext = SUCCESS_SUMMARY) {
  function Wrapper() {
    return <Outlet context={context} />;
  }
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route element={<Wrapper />}>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/forecasts" element={<h1>Forecast Rankings</h1>} />
          <Route path="/models" element={<h1>Model Comparison</h1>} />
          <Route path="/universe" element={<h1>Universe page</h1>} />
          <Route path="/methodology" element={<h1>Methodology &amp; roadmap</h1>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

describe("OverviewPage", () => {
  it("renders core product identity and real universe/research metadata", async () => {
    vi.mocked(client.getUniverse).mockResolvedValue(REAL_UNIVERSE);
    vi.mocked(client.getPredictions).mockResolvedValue(PREDICTIONS);
    vi.mocked(client.getModelComparison).mockResolvedValue(MODEL_COMPARISON);
    vi.mocked(client.getPortfolioDates).mockResolvedValue(["2022-03-28", "2026-01-02"]);

    renderPage();

    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(/Bloomberg market data/);

    await waitFor(() => expect(screen.getByText("50")).toBeInTheDocument()); // Equities stat
    expect(screen.getByText("25 IT / 25 Financials")).toBeInTheDocument();
    expect(screen.getByText("47")).toBeInTheDocument(); // walk-forward formations
    expect(screen.getByText("2,350")).toBeInTheDocument(); // OOS forecasts
    expect(screen.getByText("21 sessions")).toBeInTheDocument();
  });

  it("renders exactly 50 universe nodes with a real 25/25 sector split (fallback grid, no WebGL in jsdom)", async () => {
    vi.mocked(client.getUniverse).mockResolvedValue(REAL_UNIVERSE);
    vi.mocked(client.getPredictions).mockResolvedValue(PREDICTIONS);
    vi.mocked(client.getModelComparison).mockResolvedValue(MODEL_COMPARISON);
    vi.mocked(client.getPortfolioDates).mockResolvedValue(["2022-03-28", "2026-01-02"]);

    renderPage();

    const list = await screen.findByRole("list", { name: "50-stock RiskFecta universe" });
    const items = within(list).getAllByRole("listitem");
    expect(items).toHaveLength(50);

    const itCount = items.filter((el) => el.getAttribute("title") === "Information Technology").length;
    const finCount = items.filter((el) => el.getAttribute("title") === "Financials").length;
    expect(itCount).toBe(25);
    expect(finCount).toBe(25);
  });

  it("provides CTAs into Forecasts, Model Comparison, Universe, and Methodology", async () => {
    vi.mocked(client.getUniverse).mockResolvedValue(REAL_UNIVERSE);
    vi.mocked(client.getPredictions).mockResolvedValue(PREDICTIONS);
    vi.mocked(client.getModelComparison).mockResolvedValue(MODEL_COMPARISON);
    vi.mocked(client.getPortfolioDates).mockResolvedValue(["2022-03-28", "2026-01-02"]);

    renderPage();
    await waitFor(() => expect(screen.getByText("50")).toBeInTheDocument());

    expect(screen.getByRole("link", { name: "Explore Forecasts" })).toHaveAttribute("href", "/forecasts");
    expect(screen.getByRole("link", { name: "Compare Models" })).toHaveAttribute("href", "/models");
    expect(screen.getByRole("link", { name: "Explore Universe" })).toHaveAttribute("href", "/universe");
    expect(screen.getByRole("link", { name: "Read the methodology" })).toHaveAttribute("href", "/methodology");
  });

  it("shows a real forecast preview labeled as historical walk-forward, never a live pick", async () => {
    vi.mocked(client.getUniverse).mockResolvedValue(REAL_UNIVERSE);
    vi.mocked(client.getPredictions).mockResolvedValue(PREDICTIONS);
    vi.mocked(client.getModelComparison).mockResolvedValue(MODEL_COMPARISON);
    vi.mocked(client.getPortfolioDates).mockResolvedValue(["2022-03-28", "2026-01-02"]);

    renderPage();

    expect(await screen.findByText(/Historical walk-forward forecast/)).toBeInTheDocument();
    expect(screen.getByText(/#1 ORCL/)).toBeInTheDocument();
    expect(screen.queryByText(/today's picks/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/current opportunities/i)).not.toBeInTheDocument();
  });

  it("honestly previews mixed model evidence, never claiming the ensemble won", async () => {
    vi.mocked(client.getUniverse).mockResolvedValue(REAL_UNIVERSE);
    vi.mocked(client.getPredictions).mockResolvedValue(PREDICTIONS);
    vi.mocked(client.getModelComparison).mockResolvedValue(MODEL_COMPARISON);
    vi.mocked(client.getPortfolioDates).mockResolvedValue(["2022-03-28", "2026-01-02"]);

    renderPage();

    expect(await screen.findByText(/Standalone predictive signal was weak overall/)).toBeInTheDocument();
    expect(screen.getByText(/Historical Mean/)).toBeInTheDocument();
    expect(screen.queryByText(/best model/i)).not.toBeInTheDocument();
  });

  it("distinguishes research-complete from dashboard-not-yet-built", async () => {
    vi.mocked(client.getUniverse).mockResolvedValue(REAL_UNIVERSE);
    vi.mocked(client.getPredictions).mockResolvedValue(PREDICTIONS);
    vi.mocked(client.getModelComparison).mockResolvedValue(MODEL_COMPARISON);
    vi.mocked(client.getPortfolioDates).mockResolvedValue(["2022-03-28", "2026-01-02"]);

    renderPage();

    expect(await screen.findByText("Research engine — complete")).toBeInTheDocument();
    const liveColumn = screen.getByText("Dashboard — live now").closest("div");
    const comingColumn = screen.getByText("Dashboard — coming next").closest("div");
    expect(liveColumn).not.toBeNull();
    expect(comingColumn).not.toBeNull();
    // Portfolio Construction, Efficient Frontier (Phase 8C), and now Risk
    // Analytics (Phase 8D-1) are all real — they must have moved out of
    // "coming next" and into "live now". Historical Evidence remains
    // pending.
    expect(within(liveColumn as HTMLElement).getByText("Portfolio Construction")).toBeInTheDocument();
    expect(within(liveColumn as HTMLElement).getByText("Efficient Frontier")).toBeInTheDocument();
    expect(within(liveColumn as HTMLElement).getByText("Risk Analytics")).toBeInTheDocument();
    expect(within(comingColumn as HTMLElement).getByText("Historical Evidence")).toBeInTheDocument();
    expect(within(comingColumn as HTMLElement).queryByText("Portfolio Construction")).not.toBeInTheDocument();
    expect(within(comingColumn as HTMLElement).queryByText("Efficient Frontier")).not.toBeInTheDocument();
    expect(within(comingColumn as HTMLElement).queryByText("Risk Analytics")).not.toBeInTheDocument();
  });

  it("discloses historical-only data and the sealed March holdout, never implying live market data", async () => {
    vi.mocked(client.getUniverse).mockResolvedValue(REAL_UNIVERSE);
    vi.mocked(client.getPredictions).mockResolvedValue(PREDICTIONS);
    vi.mocked(client.getModelComparison).mockResolvedValue(MODEL_COMPARISON);
    vi.mocked(client.getPortfolioDates).mockResolvedValue(["2022-03-28", "2026-01-02"]);

    renderPage();

    expect(await screen.findByText(/Historical Bloomberg data only, through 2026-02-27/)).toBeInTheDocument();
    expect(screen.getByText(/sealed holdout reserved for a one-time Phase 9 evaluation/)).toBeInTheDocument();
    expect(screen.queryByText(/live market data/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/real-time/i)).not.toBeInTheDocument();
  });

  it("degrades gracefully when the model comparison and predictions APIs fail", async () => {
    vi.mocked(client.getUniverse).mockResolvedValue(REAL_UNIVERSE);
    vi.mocked(client.getPredictions).mockRejectedValue(new client.ApiError("Unable to reach the RiskFecta API."));
    vi.mocked(client.getModelComparison).mockRejectedValue(new client.ApiError("Unable to reach the RiskFecta API."));
    vi.mocked(client.getPortfolioDates).mockResolvedValue(["2022-03-28", "2026-01-02"]);

    renderPage();

    const errors = await screen.findAllByText("Unable to reach the RiskFecta API.");
    expect(errors.length).toBeGreaterThanOrEqual(2);
    // The universe visualization/hero still renders even if research previews fail.
    expect(await screen.findByRole("list", { name: "50-stock RiskFecta universe" })).toBeInTheDocument();
  });

  it("shows loading states before data resolves", () => {
    vi.mocked(client.getUniverse).mockReturnValue(new Promise(() => {}));
    vi.mocked(client.getPredictions).mockReturnValue(new Promise(() => {}));
    vi.mocked(client.getModelComparison).mockReturnValue(new Promise(() => {}));
    vi.mocked(client.getPortfolioDates).mockReturnValue(new Promise(() => {}));

    renderPage();

    expect(screen.getByText("Loading forecast preview…")).toBeInTheDocument();
    expect(screen.getByText("Loading model evidence…")).toBeInTheDocument();
  });
});
