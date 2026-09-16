import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Outlet, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import ModelComparisonPage from "./ModelComparisonPage";
import * as client from "../api/client";
import type { AppOutletContext } from "../AppShell";
import type { ModelComparisonResponse } from "../api/types";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    getModelComparison: vi.fn(),
  };
});

const COMPARISON: ModelComparisonResponse = {
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
    { model: "lstm_pred", label: "LSTM", source: "computed_from_persisted_predictions", mae: 0.08156, rmse: 0.11030, directional_accuracy: 0.5085, pearson_corr: 0.02197, spearman_corr: 0.00805, n_obs: 2350 },
    { model: "ensemble_pred", label: "Ensemble", source: "computed_from_persisted_predictions", mae: 0.07717, rmse: 0.10292, directional_accuracy: 0.50724, pearson_corr: 0.05479, spearman_corr: 0.01721, n_obs: 2350 },
  ],
  metric_definitions: [
    { key: "mae", label: "MAE", direction: "lower_is_better", description: "Mean absolute error." },
    { key: "rmse", label: "RMSE", direction: "lower_is_better", description: "Root mean squared error." },
    { key: "directional_accuracy", label: "Directional Accuracy", direction: "higher_is_better", description: "Sign match rate." },
    { key: "pearson_corr", label: "Pearson Correlation", direction: "context_dependent", description: "Linear correlation; small values are weak signal, not strong performance." },
    { key: "spearman_corr", label: "Spearman Correlation", direction: "context_dependent", description: "Rank correlation." },
  ],
  disagreement: {
    source: "computed_from_persisted_predictions",
    xgb_lstm_pred_pearson: 0.3407,
    xgb_lstm_pred_spearman: 0.3795,
    residual_pearson: 0.8335,
    residual_spearman: 0.8263,
    n_disagree: 773,
    n_total: 2350,
  },
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
    <MemoryRouter initialEntries={["/models"]}>
      <Routes>
        <Route element={<Wrapper />}>
          <Route path="/models" element={<ModelComparisonPage />} />
          <Route path="/methodology" element={<h1>Methodology &amp; roadmap</h1>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

describe("ModelComparisonPage", () => {
  it("renders real API-backed comparison data for all six approaches", async () => {
    vi.mocked(client.getModelComparison).mockResolvedValue(COMPARISON);

    renderPage();

    expect(screen.getByRole("heading", { name: "Model Comparison" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Comparison matrix")).toBeInTheDocument());

    for (const label of ["Historical Mean", "Momentum 3M", "Ridge", "XGBoost", "LSTM", "Ensemble"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
    // Real values from the (mocked) backend, not hardcoded in the component.
    expect(screen.getByText("7.31%")).toBeInTheDocument(); // Historical Mean MAE
  });

  it("represents all five metrics with direction, and defaults the explorer to MAE", async () => {
    vi.mocked(client.getModelComparison).mockResolvedValue(COMPARISON);
    renderPage();
    await waitFor(() => expect(screen.getByText("Comparison matrix")).toBeInTheDocument());

    for (const label of ["MAE", "RMSE", "Directional Accuracy", "Pearson Correlation", "Spearman Correlation"]) {
      expect(screen.getAllByText(label).length).toBeGreaterThan(0);
    }
    expect(screen.getByRole("button", { name: "MAE" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByText("Mean absolute error.")).toBeInTheDocument();
    expect(screen.getByText("Lower is better.")).toBeInTheDocument();
  });

  it("switches the metric explorer and shows the correct direction/definition", async () => {
    vi.mocked(client.getModelComparison).mockResolvedValue(COMPARISON);
    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getByText("Comparison matrix")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "Pearson Correlation" }));
    expect(screen.getByRole("button", { name: "Pearson Correlation" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "MAE" })).toHaveAttribute("aria-pressed", "false");
    expect(
      screen.getByText("Interpret in context — a small value here reflects weak signal, not strong predictive power."),
    ).toBeInTheDocument();
  });

  it("presents a mixed-result interpretation and never claims a single universal winner", async () => {
    vi.mocked(client.getModelComparison).mockResolvedValue(COMPARISON);
    renderPage();
    await waitFor(() => expect(screen.getByText("Research conclusion")).toBeInTheDocument());

    expect(screen.getByText(/Historical Mean remained difficult to beat/)).toBeInTheDocument();
    expect(screen.getByText(/XGBoost showed the strongest linear/)).toBeInTheDocument();
    expect(screen.getByText(/No single approach dominated across all five metrics/)).toBeInTheDocument();
    // No giant "best model" declaration — e.g. a heading naming a single winner.
    expect(screen.queryByRole("heading", { name: /best model/i })).not.toBeInTheDocument();
    expect(screen.queryByText(/ensemble (is|was) the (best|overall) (model|winner)/i)).not.toBeInTheDocument();
  });

  it("shows a loading state before the comparison resolves", () => {
    vi.mocked(client.getModelComparison).mockReturnValue(new Promise(() => {}));
    renderPage();
    expect(screen.getByText("Loading model comparison…")).toBeInTheDocument();
  });

  it("shows an error state with retry when the API call fails", async () => {
    vi.mocked(client.getModelComparison).mockRejectedValue(new client.ApiError("Unable to reach the RiskFecta API."));
    renderPage();
    expect(await screen.findByText("Unable to reach the RiskFecta API.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("links to the methodology page", async () => {
    vi.mocked(client.getModelComparison).mockResolvedValue(COMPARISON);
    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getByText("Comparison matrix")).toBeInTheDocument());

    await user.click(screen.getByRole("link", { name: /Methodology & limitations/ }));
    expect(await screen.findByRole("heading", { name: /Methodology & roadmap/ })).toBeInTheDocument();
  });
});
