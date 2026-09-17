import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Outlet, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import PortfolioConstructionPage from "./PortfolioConstructionPage";
import * as client from "../api/client";
import type { AppOutletContext } from "../AppShell";
import type { PortfolioResponse, StrategyInfo, UniverseTicker } from "../api/types";
import { TICKER_UNIVERSE, sectorForTicker } from "../data/tickerUniverse";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    getPortfolioDates: vi.fn(),
    getPortfolioStrategies: vi.fn(),
    getPortfolio: vi.fn(),
    getUniverse: vi.fn(),
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

const UNIVERSE: UniverseTicker[] = TICKER_UNIVERSE.map((t) => ({ ticker: t, sector: sectorForTicker(t) }));

function buildPortfolio(overrides: Partial<PortfolioResponse> = {}): PortfolioResponse {
  const holdings = TICKER_UNIVERSE.map((t, i) => ({ ticker: t, weight: i < 5 ? 0.1 : (1 - 0.5) / 45 }));
  return {
    formation_date: "2026-01-02",
    strategy: STRATEGIES[3],
    max_weight_constraint: 0.1,
    holdings,
    active_holdings_count: 50,
    largest_weight: 0.1,
    concentration_hhi: 0.055,
    construction: { expected_return_21: 0.041, predicted_volatility_21: 0.056, expected_sharpe_21: 0.667 },
    evaluation: { realized_return_21: -0.0123, turnover: 0.42, max_weight_observed: 0.1 },
    source: "official_phase7_persisted_experiment",
    ...overrides,
  };
}

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
    <MemoryRouter initialEntries={["/portfolio"]}>
      <Routes>
        <Route element={<Wrapper />}>
          <Route path="/portfolio" element={<PortfolioConstructionPage />} />
          <Route path="/methodology" element={<h1>Methodology &amp; roadmap</h1>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

describe("PortfolioConstructionPage", () => {
  it("renders real data with all 46 dates, all 5 strategies, and defaults to Ledoit-Wolf Max-Sharpe", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);
    vi.mocked(client.getPortfolio).mockResolvedValue(buildPortfolio());

    renderPage();

    expect(screen.getByRole("heading", { name: "Historical Portfolio Construction" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Portfolio weights")).toBeInTheDocument());

    expect(screen.getByRole("button", { name: "Ledoit-Wolf Max-Sharpe" })).toHaveAttribute("aria-pressed", "true");
    for (const s of STRATEGIES) {
      expect(screen.getByRole("button", { name: s.label })).toBeInTheDocument();
    }
    expect(screen.getAllByRole("row").length).toBeGreaterThan(50); // header + 50 holdings
  });

  it("shows the 50 holdings and the 10% hard cap", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);
    vi.mocked(client.getPortfolio).mockResolvedValue(buildPortfolio());

    renderPage();
    await waitFor(() => expect(screen.getByText("Portfolio weights")).toBeInTheDocument());

    expect(screen.getByText("Max weight (hard cap)")).toBeInTheDocument();
    expect(screen.getAllByText("10.00%").length).toBeGreaterThan(0);
  });

  it("shows Equal Weight as a benchmark, not an optimized construction", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);
    vi.mocked(client.getPortfolio).mockResolvedValue(
      buildPortfolio({
        strategy: STRATEGIES[4],
        holdings: TICKER_UNIVERSE.map((t) => ({ ticker: t, weight: 0.02 })),
        max_weight_constraint: null,
        largest_weight: 0.02,
        construction: { expected_return_21: null, predicted_volatility_21: null, expected_sharpe_21: null },
      }),
    );

    renderPage();
    await waitFor(() => expect(screen.getByText("Portfolio weights")).toBeInTheDocument());

    expect(screen.getByText(/Not applicable — Equal Weight is a benchmark/)).toBeInTheDocument();
    expect(screen.getByText("N/A")).toBeInTheDocument(); // max weight hard cap stat
  });

  it("distinguishes Min-Vol (no expected-return objective) from Max-Sharpe (ensemble-driven)", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);
    vi.mocked(client.getPortfolio).mockResolvedValue(buildPortfolio({ strategy: STRATEGIES[2] })); // LW_MINVOL

    renderPage();
    await waitFor(() => expect(screen.getAllByText(/does not use expected returns/).length).toBeGreaterThan(0));
  });

  it("shows sector exposure summing to ~100%", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);
    vi.mocked(client.getPortfolio).mockResolvedValue(buildPortfolio());

    renderPage();
    await waitFor(() => expect(screen.getByText("Sector exposure")).toBeInTheDocument());
    expect(screen.getByText("Information Technology")).toBeInTheDocument();
    expect(screen.getByText("Financials")).toBeInTheDocument();
  });

  it("labels realized return as ex-post, separate from construction", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);
    vi.mocked(client.getPortfolio).mockResolvedValue(buildPortfolio());

    renderPage();
    await waitFor(() => expect(screen.getByText("Realized outcome — ex-post")).toBeInTheDocument());
    expect(screen.getByText("Construction — ex-ante")).toBeInTheDocument();
  });

  it("shows first-formation turnover as undefined, never zero", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);
    vi.mocked(client.getPortfolio).mockResolvedValue(buildPortfolio({ evaluation: { realized_return_21: -0.05, turnover: null, max_weight_observed: 0.1 } }));

    renderPage();
    await waitFor(() => expect(screen.getByText("Not defined for first formation")).toBeInTheDocument());
  });

  it("discloses historical research context and freshness, and links to methodology", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);
    vi.mocked(client.getPortfolio).mockResolvedValue(buildPortfolio());

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getByText("Portfolio weights")).toBeInTheDocument());

    expect(screen.getByText(/historical research construction — not a live recommendation/)).toBeInTheDocument();
    expect(screen.getByText("2026-02-27")).toBeInTheDocument();

    await user.click(screen.getByRole("link", { name: /Methodology & limitations/ }));
    expect(await screen.findByRole("heading", { name: /Methodology & roadmap/ })).toBeInTheDocument();
  });

  it("re-fetches when the strategy selector changes", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);
    vi.mocked(client.getPortfolio).mockResolvedValue(buildPortfolio());

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getByText("Portfolio weights")).toBeInTheDocument());

    vi.mocked(client.getPortfolio).mockResolvedValue(buildPortfolio({ strategy: STRATEGIES[0] }));
    await user.click(screen.getByRole("button", { name: "Sample Min-Vol" }));

    await waitFor(() => expect(client.getPortfolio).toHaveBeenCalledWith(undefined, "SAMPLE_MINVOL"));
  });

  it("shows a loading state before the portfolio resolves", () => {
    vi.mocked(client.getPortfolioDates).mockReturnValue(new Promise(() => {}));
    vi.mocked(client.getPortfolioStrategies).mockReturnValue(new Promise(() => {}));
    vi.mocked(client.getUniverse).mockReturnValue(new Promise(() => {}));
    vi.mocked(client.getPortfolio).mockReturnValue(new Promise(() => {}));

    renderPage();
    expect(screen.getByText("Loading portfolio…")).toBeInTheDocument();
  });

  it("shows an error state with retry when the API call fails", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getPortfolioStrategies).mockResolvedValue(STRATEGIES);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);
    vi.mocked(client.getPortfolio).mockRejectedValue(new client.ApiError("Unable to reach the RiskFecta API."));

    renderPage();
    expect(await screen.findByText("Unable to reach the RiskFecta API.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
