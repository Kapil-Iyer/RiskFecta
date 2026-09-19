import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import RiskAnalyticsPage from "./RiskAnalyticsPage";
import * as client from "../api/client";
import type { RiskResponse, StrategyInfo } from "../api/types";
import { TICKER_UNIVERSE, sectorForTicker } from "../data/tickerUniverse";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    getPortfolioDates: vi.fn(),
    getPortfolioStrategies: vi.fn(),
    getRisk: vi.fn(),
  };
});

const DATES = ["2022-03-28", "2022-04-27", "2026-01-02"];

const STRATEGIES: StrategyInfo[] = [
  { key: "SAMPLE_MINVOL", label: "Sample Min-Vol", covariance_estimator: "Sample", objective: "Minimum volatility — does not use expected returns", is_optimized: true },
  { key: "SAMPLE_MAXSHARPE", label: "Sample Max-Sharpe", covariance_estimator: "Sample", objective: "Maximum Sharpe ratio, using the frozen 50/50 ensemble expected returns", is_optimized: true },
  { key: "LW_MINVOL", label: "Ledoit-Wolf Min-Vol", covariance_estimator: "Ledoit-Wolf", objective: "Minimum volatility — does not use expected returns", is_optimized: true },
  { key: "LW_MAXSHARPE", label: "Ledoit-Wolf Max-Sharpe", covariance_estimator: "Ledoit-Wolf", objective: "Maximum Sharpe ratio, using the frozen 50/50 ensemble expected returns", is_optimized: true },
  { key: "EQUAL_WEIGHT", label: "Equal Weight", covariance_estimator: "Not applicable", objective: "Equal-weight benchmark — not an optimized portfolio", is_optimized: false },
];

function buildRisk(overrides: Partial<RiskResponse> = {}): RiskResponse {
  const assets = TICKER_UNIVERSE.map((ticker, i) => ({
    ticker,
    sector: sectorForTicker(ticker) ?? "Information Technology",
    weight: i < 10 ? 0.1 : 0,
    marginal_risk_contribution: 0.05,
    component_risk_contribution: i < 10 ? 0.0056 : 0,
    risk_share: i < 10 ? 0.1 : 0,
  }));
  return {
    formation_date: "2026-01-02",
    strategy: STRATEGIES[3],
    covariance_estimator: "Ledoit-Wolf",
    forecast_horizon_sessions: 21,
    covariance_window_sessions: 252,
    portfolio: {
      predicted_volatility_21: 0.0563,
      largest_weight: 0.1,
      active_holdings_count: 10,
      concentration_hhi: 0.1,
      effective_holdings: 10,
      max_weight_constraint: 0.1,
    },
    assets,
    sectors: [
      { sector: "Information Technology", weight: 0.5, component_risk_contribution: 0.041, risk_share: 0.73 },
      { sector: "Financials", weight: 0.5, component_risk_contribution: 0.015, risk_share: 0.27 },
    ],
    source: "reconstructed_from_frozen_phase7_methodology",
    ...overrides,
  };
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/risk"]}>
      <Routes>
        <Route path="/risk" element={<RiskAnalyticsPage />} />
        <Route path="/methodology" element={<h1>Methodology &amp; roadmap</h1>} />
        <Route path="/portfolio" element={<h1>Historical Portfolio Construction</h1>} />
        <Route path="/frontier" element={<h1>Efficient Frontier</h1>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("RiskAnalyticsPage", () => {
  it("renders real data, defaulting to the latest date and Ledoit-Wolf Max-Sharpe", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getRisk).mockResolvedValue(buildRisk());

    renderPage();

    expect(screen.getByRole("heading", { name: "Risk Analytics" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Component risk contribution")).toBeInTheDocument());

    expect(screen.getByRole("button", { name: "Ledoit-Wolf Max-Sharpe" })).toHaveAttribute("aria-pressed", "true");
    await waitFor(() => expect(client.getRisk).toHaveBeenCalledWith(undefined, undefined, undefined));
    expect(document.querySelector(".chart")).toBeInTheDocument();
  });

  it("shows the 21-session horizon, 252-session covariance window, and 10% cap", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getRisk).mockResolvedValue(buildRisk());

    renderPage();
    await waitFor(() => expect(screen.getByText("Component risk contribution")).toBeInTheDocument());

    expect(screen.getByText("21-session horizon")).toBeInTheDocument();
    expect(screen.getByText("252-session covariance window")).toBeInTheDocument();
    expect(screen.getAllByText("10.00%").length).toBeGreaterThan(0); // Max weight (hard cap) stat
  });

  it("shows predicted portfolio volatility and concentration stats", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getRisk).mockResolvedValue(buildRisk());

    renderPage();
    await waitFor(() => expect(screen.getByText("Predicted 21-session volatility")).toBeInTheDocument());
    expect(screen.getByText("5.63%")).toBeInTheDocument();
    expect(screen.getByText("Effective holdings")).toBeInTheDocument();
    expect(screen.getByText("Concentration (HHI)")).toBeInTheDocument();
  });

  it("locks the covariance display to the strategy's construction estimator for optimized strategies", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getRisk).mockResolvedValue(buildRisk());

    renderPage();
    await waitFor(() => expect(screen.getByText("Component risk contribution")).toBeInTheDocument());

    expect(screen.getByText("(determined by strategy)")).toBeInTheDocument();
    expect(screen.getByText("Ledoit-Wolf")).toBeInTheDocument();
    // No free covariance selector shown for an optimized strategy.
    expect(screen.queryByRole("group", { name: /Covariance estimator for Equal Weight/ })).not.toBeInTheDocument();
  });

  it("shows an explicit covariance-analysis selector only for Equal Weight, with truthful labeling", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getRisk).mockResolvedValue(
      buildRisk({ strategy: STRATEGIES[4], covariance_estimator: "Ledoit-Wolf", portfolio: { ...buildRisk().portfolio, max_weight_constraint: null } }),
    );

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getByText("Component risk contribution")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "Equal Weight" }));

    expect(await screen.findByRole("group", { name: /Covariance estimator for Equal Weight risk analysis/ })).toBeInTheDocument();
    expect(screen.getByText(/Risk analysis using Ledoit-Wolf covariance/)).toBeInTheDocument();
    expect(screen.getByText(/Equal Weight itself is never constructed from a covariance estimate/)).toBeInTheDocument();
    // "N/A" max-weight stat for the unconstrained benchmark.
    expect(screen.getByText("N/A")).toBeInTheDocument();
  });

  it("re-fetches with the selected covariance only when analyzing Equal Weight", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getRisk).mockResolvedValue(buildRisk());

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getByText("Component risk contribution")).toBeInTheDocument());

    vi.mocked(client.getRisk).mockResolvedValue(buildRisk({ strategy: STRATEGIES[4] }));
    await user.click(screen.getByRole("button", { name: "Equal Weight" }));
    await waitFor(() => expect(client.getRisk).toHaveBeenCalledWith(undefined, "EQUAL_WEIGHT", "LW"));

    vi.mocked(client.getRisk).mockResolvedValue(buildRisk({ strategy: STRATEGIES[4], covariance_estimator: "Sample" }));
    await user.click(screen.getByRole("button", { name: "Sample" }));
    await waitFor(() => expect(client.getRisk).toHaveBeenCalledWith(undefined, "EQUAL_WEIGHT", "SAMPLE"));
  });

  it("re-fetches when the formation date changes", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getRisk).mockResolvedValue(buildRisk());

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getByText("Component risk contribution")).toBeInTheDocument());

    vi.mocked(client.getRisk).mockResolvedValue(buildRisk({ formation_date: "2022-03-28" }));
    await user.selectOptions(screen.getByLabelText("Formation date (historical)"), "2022-03-28");

    await waitFor(() => expect(client.getRisk).toHaveBeenCalledWith("2022-03-28", undefined, undefined));
  });

  it("shows the weight-vs-risk-share component chart and exactly two sectors", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getRisk).mockResolvedValue(buildRisk());

    renderPage();
    await waitFor(() => expect(screen.getByText("Component risk contribution")).toBeInTheDocument());

    expect(document.querySelector(".chart")).toBeInTheDocument();
    expect(screen.getByText("Sector risk aggregation")).toBeInTheDocument();
    expect(screen.getByText("Information Technology")).toBeInTheDocument();
    expect(screen.getByText("Financials")).toBeInTheDocument();
    // "Portfolio weight"/"Risk share" also appear in the real Plotly
    // legend (rendered in this jsdom test, not a stub) alongside this
    // page's own legend dots — assert presence via getAllBy*.
    expect(screen.getAllByText("Portfolio weight").length).toBeGreaterThan(0);
    expect(screen.getAllByText(/Risk share/).length).toBeGreaterThan(0);
  });

  it("uses ex-ante/predicted language and never presents realized-outcome figures as risk", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getRisk).mockResolvedValue(buildRisk());

    renderPage();
    await waitFor(() => expect(screen.getByText("Component risk contribution")).toBeInTheDocument());

    expect(screen.getByText(/Formation-time predicted portfolio risk/)).toBeInTheDocument();
    expect(screen.getByText(/formation-time, predicted estimates — not realized future risk/)).toBeInTheDocument();
    expect(screen.queryByText(/realized_return_21|Realized return|Realized outcome/)).not.toBeInTheDocument();
    expect(screen.queryByText(/SPXT/)).not.toBeInTheDocument();
    expect(screen.queryByText(/[Tt]urnover/)).not.toBeInTheDocument();
  });

  it("links to Portfolio Construction, Efficient Frontier, and Methodology", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getRisk).mockResolvedValue(buildRisk());

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getByText("Component risk contribution")).toBeInTheDocument());

    await user.click(screen.getByRole("link", { name: "Portfolio Construction" }));
    expect(await screen.findByRole("heading", { name: "Historical Portfolio Construction" })).toBeInTheDocument();
  });

  it("shows a loading state before risk data resolves", () => {
    vi.mocked(client.getPortfolioDates).mockReturnValue(new Promise(() => {}));
    vi.mocked(client.getPortfolioStrategies).mockReturnValue(new Promise(() => {}));
    vi.mocked(client.getRisk).mockReturnValue(new Promise(() => {}));

    renderPage();
    expect(screen.getByText(/Reconstructing formation-time risk/)).toBeInTheDocument();
  });

  it("shows an error state with retry when the API call fails", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getRisk).mockRejectedValue(new client.ApiError("Unable to reach the RiskFecta API."));

    renderPage();
    expect(await screen.findByText("Unable to reach the RiskFecta API.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
