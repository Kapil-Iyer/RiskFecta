import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import HistoricalEvidencePage from "./HistoricalEvidencePage";
import * as client from "../api/client";
import type { BacktestResponse, BacktestSeries } from "../api/types";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    getBacktest: vi.fn(),
  };
});

function buildPeriods(returns: number[], turnovers: (number | null)[]) {
  const dates = ["2022-03-28", "2022-04-27", "2022-05-26"];
  const targets = ["2022-04-27", "2022-05-26", "2022-06-28"];
  let growth = 1.0;
  return returns.map((r, i) => {
    growth *= 1 + r;
    return {
      formation_date: dates[i],
      target_date: targets[i],
      realized_return_21: r,
      growth_of_one: growth,
      turnover: turnovers[i],
      max_weight_observed: 0.1,
    };
  });
}

function buildSeries(overrides: Partial<BacktestSeries> = {}): BacktestSeries {
  const returns = [0.02, -0.01, 0.03];
  const turnovers = [null, 0.4, 0.35];
  return {
    key: "LW_MAXSHARPE",
    label: "Ledoit-Wolf Max-Sharpe",
    kind: "portfolio",
    is_optimized: true,
    covariance_estimator: "Ledoit-Wolf",
    max_weight_constraint: 0.1,
    periods: buildPeriods(returns, turnovers),
    summary: {
      mean_return_21: 0.0133,
      std_return_21: 0.0208,
      median_return_21: 0.02,
      min_return_21: -0.01,
      max_return_21: 0.03,
      positive_period_rate: 0.667,
      cumulative_return: 0.0403,
      mean_turnover: 0.375,
      median_turnover: 0.375,
      max_turnover: 0.4,
      avg_max_weight: 0.1,
      max_observed_weight: 0.1,
    },
    ...overrides,
  };
}

function buildBacktest(overrides: Partial<BacktestResponse> = {}): BacktestResponse {
  const strategies: BacktestSeries[] = [
    buildSeries({ key: "SAMPLE_MINVOL", label: "Sample Min-Vol", covariance_estimator: "Sample" }),
    buildSeries({ key: "SAMPLE_MAXSHARPE", label: "Sample Max-Sharpe", covariance_estimator: "Sample" }),
    buildSeries({ key: "LW_MINVOL", label: "Ledoit-Wolf Min-Vol" }),
    buildSeries({ key: "LW_MAXSHARPE", label: "Ledoit-Wolf Max-Sharpe" }),
    buildSeries({
      key: "EQUAL_WEIGHT",
      label: "Equal Weight",
      is_optimized: false,
      covariance_estimator: null,
      max_weight_constraint: null,
      periods: buildPeriods([0.015, 0.01, 0.02], [null, 0, 0]),
      summary: {
        mean_return_21: 0.015, std_return_21: 0.005, median_return_21: 0.015, min_return_21: 0.01, max_return_21: 0.02,
        positive_period_rate: 1.0, cumulative_return: 0.046,
        mean_turnover: 0, median_turnover: 0, max_turnover: 0, avg_max_weight: 0.02, max_observed_weight: 0.02,
      },
    }),
    buildSeries({
      key: "SPXT",
      label: "SPXT (S&P 500 Total Return)",
      kind: "benchmark",
      is_optimized: null,
      covariance_estimator: null,
      max_weight_constraint: null,
      periods: buildPeriods([0.011, 0.015, 0.018], [null, null, null]).map((p) => ({
        ...p,
        turnover: null,
        max_weight_observed: null,
      })),
      summary: {
        mean_return_21: 0.0147, std_return_21: 0.0035, median_return_21: 0.015, min_return_21: 0.011, max_return_21: 0.018,
        positive_period_rate: 1.0, cumulative_return: 0.045,
        mean_turnover: null, median_turnover: null, max_turnover: null, avg_max_weight: null, max_observed_weight: null,
      },
    }),
  ];

  return {
    experiment: {
      period_count: 3,
      first_formation_date: "2022-03-28",
      last_formation_date: "2022-05-26",
      horizon_sessions: 21,
      covariance_window_sessions: 252,
      benchmark: "SPXT",
      data_through: "2026-02-27",
    },
    series: strategies,
    source: "official_phase7_persisted_experiment",
    ...overrides,
  };
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/backtest"]}>
      <Routes>
        <Route path="/backtest" element={<HistoricalEvidencePage />} />
        <Route path="/methodology" element={<h1>Methodology &amp; roadmap</h1>} />
        <Route path="/models" element={<h1>Model Comparison</h1>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("HistoricalEvidencePage", () => {
  it("renders the header, 46-period disclosure context, and historical-vs-sealed distinction", async () => {
    vi.mocked(client.getBacktest).mockResolvedValue(buildBacktest());

    renderPage();
    expect(screen.getByRole("heading", { name: "Historical Evidence" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Cumulative growth of $1")).toBeInTheDocument());

    expect(screen.getByText("46 periods")).toBeInTheDocument();
    expect(screen.getByText("21-session horizon")).toBeInTheDocument();
    expect(screen.getByText("Sealed March holdout pending")).toBeInTheDocument();
    expect(screen.getByText(/sealed March 2026 holdout remains untouched/)).toBeInTheDocument();
    const dataThrough = screen.getByText(/Data through/);
    expect(dataThrough.textContent).toContain("2026-02-27");
  });

  it("shows all six series as inspectable, defaulting to Ledoit-Wolf Max-Sharpe", async () => {
    vi.mocked(client.getBacktest).mockResolvedValue(buildBacktest());

    renderPage();
    await waitFor(() => expect(screen.getByText("Cumulative growth of $1")).toBeInTheDocument());

    for (const label of ["Sample Min-Vol", "Sample Max-Sharpe", "Ledoit-Wolf Min-Vol", "Ledoit-Wolf Max-Sharpe", "Equal Weight", "SPXT (S&P 500 Total Return)"]) {
      expect(screen.getByRole("button", { name: label })).toBeInTheDocument();
    }
    expect(screen.getByRole("button", { name: "Ledoit-Wolf Max-Sharpe" })).toHaveAttribute("aria-pressed", "true");
    expect(document.querySelector(".chart")).toBeInTheDocument();
    expect(screen.getByText(/All six series start at exactly \$1\.00 on 2022-03-28/)).toBeInTheDocument();
  });

  it("updates the summary stats when a different series is selected", async () => {
    vi.mocked(client.getBacktest).mockResolvedValue(buildBacktest());
    const user = userEvent.setup();

    renderPage();
    await waitFor(() => expect(screen.getByText("Cumulative growth of $1")).toBeInTheDocument());

    const meanStat = () => screen.getByText("Mean 21-session return").closest("div");
    expect(meanStat()!.textContent).toContain("+1.3%"); // LW_MAXSHARPE mean, default selection

    await user.click(screen.getByRole("button", { name: "Equal Weight" }));
    await waitFor(() => expect(meanStat()!.textContent).toContain("+1.5%")); // Equal Weight mean
    expect(screen.getByRole("button", { name: "Equal Weight" })).toHaveAttribute("aria-pressed", "true");
  });

  it("shows first-formation turnover as not defined, never zero, for portfolio strategies", async () => {
    vi.mocked(client.getBacktest).mockResolvedValue(buildBacktest());

    renderPage();
    await waitFor(() => expect(screen.getByText("Cumulative growth of $1")).toBeInTheDocument());

    expect(screen.getByText(/not defined for the first formation/)).toBeInTheDocument();
  });

  it("shows Equal Weight's later turnover as truthfully zero", async () => {
    vi.mocked(client.getBacktest).mockResolvedValue(buildBacktest());
    const user = userEvent.setup();

    renderPage();
    await waitFor(() => expect(screen.getByText("Cumulative growth of $1")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "Equal Weight" }));
    await waitFor(() => expect(screen.getByText("Mean turnover")).toBeInTheDocument());
    const meanTurnoverStat = screen.getByText("Mean turnover").closest("div");
    expect(meanTurnoverStat).not.toBeNull();
    expect(meanTurnoverStat!.textContent).toContain("0.0%");
  });

  it("shows SPXT turnover and concentration as not applicable", async () => {
    vi.mocked(client.getBacktest).mockResolvedValue(buildBacktest());
    const user = userEvent.setup();

    renderPage();
    await waitFor(() => expect(screen.getByText("Cumulative growth of $1")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "SPXT (S&P 500 Total Return)" }));
    await waitFor(() => expect(screen.getAllByText("N/A").length).toBeGreaterThan(0));
    expect(screen.getByText(/Turnover and weight concentration are not applicable to an index benchmark/)).toBeInTheDocument();
    expect(screen.getByText(/Not applicable — SPXT is an index benchmark/)).toBeInTheDocument();
  });

  it("discloses no transaction costs / high turnover and the weak-mixed forecast evidence, linking to Model Comparison", async () => {
    vi.mocked(client.getBacktest).mockResolvedValue(buildBacktest());
    const user = userEvent.setup();

    renderPage();
    await waitFor(() => expect(screen.getByText("Cumulative growth of $1")).toBeInTheDocument());

    expect(screen.getByText(/No transaction costs or slippage are modeled/)).toBeInTheDocument();
    expect(screen.getByText(/weak\/mixed predictive performance/)).toBeInTheDocument();

    await user.click(screen.getByRole("link", { name: "Model Comparison" }));
    expect(await screen.findByRole("heading", { name: "Model Comparison" })).toBeInTheDocument();
  });

  it("links to Methodology", async () => {
    vi.mocked(client.getBacktest).mockResolvedValue(buildBacktest());
    const user = userEvent.setup();

    renderPage();
    await waitFor(() => expect(screen.getByText("Cumulative growth of $1")).toBeInTheDocument());

    await user.click(screen.getAllByRole("link", { name: /Methodology & limitations/ })[0]);
    expect(await screen.findByRole("heading", { name: /Methodology & roadmap/ })).toBeInTheDocument();
  });

  it("never uses best/winner/recommendation wording anywhere on the page", async () => {
    vi.mocked(client.getBacktest).mockResolvedValue(buildBacktest());

    renderPage();
    await waitFor(() => expect(screen.getByText("Cumulative growth of $1")).toBeInTheDocument());

    const bodyText = document.body.textContent ?? "";
    for (const forbidden of [/\bbest strategy\b/i, /\bwinner\b/i, /\bsuperior\b/i, /\boptimal performer\b/i, /\brecommended\b/i, /\bmost successful\b/i]) {
      expect(bodyText).not.toMatch(forbidden);
    }
  });

  it("shows a loading state before the experiment resolves", () => {
    vi.mocked(client.getBacktest).mockReturnValue(new Promise(() => {}));

    renderPage();
    expect(screen.getByText(/Loading historical evidence/)).toBeInTheDocument();
  });

  it("shows an error state with retry when the API call fails", async () => {
    vi.mocked(client.getBacktest).mockRejectedValue(new client.ApiError("Unable to reach the RiskFecta API."));

    renderPage();
    expect(await screen.findByText("Unable to reach the RiskFecta API.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
