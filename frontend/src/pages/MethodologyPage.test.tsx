import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it } from "vitest";
import MethodologyPage from "./MethodologyPage";

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/methodology"]}>
      <Routes>
        <Route path="/methodology" element={<MethodologyPage />} />
        <Route path="/universe" element={<h1>Universe</h1>} />
        <Route path="/forecasts" element={<h1>Forecast Rankings</h1>} />
        <Route path="/models" element={<h1>Model Comparison</h1>} />
        <Route path="/portfolio" element={<h1>Historical Portfolio Construction</h1>} />
        <Route path="/frontier" element={<h1>Efficient Frontier</h1>} />
        <Route path="/risk" element={<h1>Risk Analytics</h1>} />
        <Route path="/backtest" element={<h1>Historical Evidence</h1>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("MethodologyPage", () => {
  it("renders the header and status chips", () => {
    renderPage();
    expect(screen.getByRole("heading", { name: "Methodology", level: 1 })).toBeInTheDocument();
    expect(screen.getByText("Historical Bloomberg data")).toBeInTheDocument();
    expect(screen.getByText("Walk-forward OOS")).toBeInTheDocument();
    expect(screen.getByText("21-session horizon")).toBeInTheDocument();
    expect(screen.getByText("50-stock universe")).toBeInTheDocument();
    expect(screen.getByText("Sealed holdout pending")).toBeInTheDocument();
  });

  it("describes the 50-stock, 25/25-sector universe and Bloomberg data freshness", () => {
    renderPage();
    expect(screen.getByText(/fixed 50-stock universe \(25 Information Technology, 25 Financials\)/)).toBeInTheDocument();
    expect(screen.getByText(/2021-03-01 through 2026-02-27/)).toBeInTheDocument();
    expect(screen.getByText(/Session validity is gated on PX_LAST \(close\)/)).toBeInTheDocument();
  });

  it("states the 21-session TRI target definition, never a price return", () => {
    renderPage();
    expect(screen.getByText(/21-trading-session forward total return/)).toBeInTheDocument();
    expect(screen.getByText("target_t = TRI(t+21) / TRI(t) − 1 — 21 valid trading sessions, never 21 calendar days.")).toBeInTheDocument();
  });

  it("explains walk-forward causality and the 47-vs-46 formation date distinction", () => {
    renderPage();
    expect(screen.getByText(/No random train\/test split anywhere in this pipeline/)).toBeInTheDocument();
    expect(screen.getByText("47 formation dates")).toBeInTheDocument();
    expect(screen.getByText("2022-02-25 → 2026-01-02")).toBeInTheDocument();
    expect(screen.getByText("46 formation dates")).toBeInTheDocument();
    expect(screen.getByText("2022-03-28 → 2026-01-02")).toBeInTheDocument();
    expect(screen.getByText(/one short of the 253 levels needed/)).toBeInTheDocument();
  });

  it("describes XGBoost, LSTM, and the fixed 50/50 ensemble", () => {
    renderPage();
    expect(screen.getByRole("heading", { name: "XGBoost" })).toBeInTheDocument();
    expect(screen.getByText(/VIX, the US 10-year yield, 3-month momentum, 6-month momentum, and 20-day realized volatility/)).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "LSTM" })).toBeInTheDocument();
    expect(screen.getByText(/60-session sequence of 13 OHLCV\/technical features/)).toBeInTheDocument();
    // "Ensemble" appears both as a pipeline step and its own Modeling
    // sub-heading — assert at least one of each real element.
    expect(screen.getAllByRole("heading", { name: "Ensemble" }).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("ensemble_pred = 0.5 · XGB_pred + 0.5 · LSTM_pred")).toBeInTheDocument();
    expect(screen.getByText(/frozen in advance — never tuned or reweighted/)).toBeInTheDocument();
  });

  it("expands LSTM technical detail via progressive disclosure", async () => {
    // jsdom implements <details>'s `open` attribute toggling but not the
    // UA stylesheet that hides closed content, so this checks the
    // semantic `open` state directly rather than computed visibility.
    renderPage();
    const user = userEvent.setup();
    const details = screen.getByText("Technical detail").closest("details");
    expect(details).not.toBeNull();
    expect(details).not.toHaveAttribute("open");

    await user.click(screen.getByText("Technical detail"));

    expect(details).toHaveAttribute("open");
    expect(screen.getByText(/hidden_size=32, num_layers=1, dropout=0\.0/)).toBeInTheDocument();
  });

  it("discloses weak/mixed standalone forecast evidence and links to Model Comparison", async () => {
    renderPage();
    expect(screen.getByText(/Historical Mean baseline outperformed both XGBoost and the ensemble on MAE, RMSE, and directional accuracy/)).toBeInTheDocument();
    const user = userEvent.setup();
    await user.click(screen.getByRole("link", { name: "See the full Model Comparison" }));
    expect(await screen.findByRole("heading", { name: "Model Comparison" })).toBeInTheDocument();
  });

  it("states the covariance methodology without ranking Sample vs. Ledoit-Wolf", () => {
    renderPage();
    expect(screen.getByText(/252 sessions of historical realized TRI returns/)).toBeInTheDocument();
    // The Σ_21 scaling formula is intentionally repeated (pipeline step +
    // Portfolio research detail) — assert it appears, not that it's unique.
    expect(screen.getAllByText(/Σ_21 = 21 · Σ_session/).length).toBeGreaterThanOrEqual(2);
    expect(screen.getByText(/shrinkage is not shown anywhere as having produced better returns/)).toBeInTheDocument();
  });

  it("states the frozen constraints and that Min-Vol never uses expected returns", () => {
    renderPage();
    const expectedReturnsHeading = screen.getByRole("heading", { name: "Expected returns" });
    const paragraph = expectedReturnsHeading.parentElement?.querySelector("p");
    expect(paragraph?.textContent).toContain("mu_21");
    expect(paragraph?.textContent).toContain("the expected 21-session return vector");
    expect(screen.getByText(/Min-Vol does not use expected returns in its objective at all/)).toBeInTheDocument();
    expect(screen.getByText(/hard 10% maximum single-name weight — constrained portfolios, never unconstrained mean-variance optimization/)).toBeInTheDocument();
  });

  it("states the risk-contribution identity and that negative contributions are preserved", () => {
    renderPage();
    expect(screen.getByText(/RC_i = w_i · \(Σw\)_i \/ σ_p/)).toBeInTheDocument();
    expect(screen.getByText(/an individual contribution can be negative/)).toBeInTheDocument();
    expect(screen.getByText(/never floored at zero/)).toBeInTheDocument();
  });

  it("states historical-evidence methodology: compounding, turnover, and SPXT identity", () => {
    renderPage();
    expect(screen.getByText(/product_t\(1 \+ r_t\) − 1/)).toBeInTheDocument();
    expect(screen.getByText(/no daily interpolation and no/)).toBeInTheDocument();
    expect(screen.getByText(/genuinely undefined turnover .* never shown as zero/)).toBeInTheDocument();
    expect(screen.getByText(/official Bloomberg S&P 500 Total Return Index \(SPXT Index, PX_LAST\)/)).toBeInTheDocument();
    expect(screen.getByText(/never used in portfolio construction, never a predictive feature, and never a risk coordinate/)).toBeInTheDocument();
  });

  it("labels historical portfolio results as historical walk-forward evidence with required caveats", () => {
    renderPage();
    expect(screen.getByText("Historical walk-forward evidence")).toBeInTheDocument();
    expect(screen.getByText(/No transaction costs or slippage/)).toBeInTheDocument();
    expect(screen.getByText(/not a guarantee of future performance/)).toBeInTheDocument();
  });

  it("never presents the sealed March holdout as evaluated", () => {
    renderPage();
    expect(screen.getByText("Sealed — not yet evaluated")).toBeInTheDocument();
    expect(screen.getByText(/deliberately withheld/)).toBeInTheDocument();
    expect(screen.getByText(/not a fresh out-of-sample test\. March/)).toBeInTheDocument();
    expect(screen.getByText(/no retuning regardless of outcome/)).toBeInTheDocument();

    const bodyText = document.body.textContent ?? "";
    // No March 2026 performance figure of any kind may appear anywhere.
    expect(bodyText).not.toMatch(/march.{0,40}(result|return|performance|outcome)s?\s*(is|was|were|:)/i);
  });

  it("shows a scannable limitations section covering the required items", () => {
    renderPage();
    expect(screen.getByRole("heading", { name: "Limitations" })).toBeInTheDocument();
    expect(screen.getByText(/not proof of future performance/)).toBeInTheDocument();
    expect(screen.getByText(/only 50 stocks/)).toBeInTheDocument();
    expect(screen.getByText(/Information Technology and Financials/)).toBeInTheDocument();
    expect(screen.getByText(/Transaction costs and slippage are not modeled/)).toBeInTheDocument();
    expect(screen.getByText(/substantially higher turnover than Min-Vol or Equal Weight/)).toBeInTheDocument();
    expect(screen.getByText(/hard 10% single-name cap frequently bound/)).toBeInTheDocument();
    expect(screen.getByText(/Static snapshot fields.*excluded from predictive features/)).toBeInTheDocument();
    expect(screen.getByText(/historical realized covariance, not a predicted/)).toBeInTheDocument();
    expect(screen.getByText(/long-only and fully invested — no leverage, no shorting/)).toBeInTheDocument();
    expect(screen.getByText(/different evidence categories/)).toBeInTheDocument();
    expect(screen.getByText(/not by itself sufficient to establish broad generalization/)).toBeInTheDocument();
  });

  it("shows what is frozen vs. pending, and what RiskFecta does not claim", () => {
    renderPage();
    expect(screen.getByRole("heading", { name: "What is frozen" })).toBeInTheDocument();
    expect(screen.getByText(/21-session forward TRI target definition/)).toBeInTheDocument();
    expect(screen.getByText("the sealed March 2026 outcome.")).toBeInTheDocument();

    expect(screen.getByRole("heading", { name: "RiskFecta does not claim" })).toBeInTheDocument();
    expect(screen.getByText("Guaranteed returns")).toBeInTheDocument();
    expect(screen.getByText("Investment advice")).toBeInTheDocument();
    expect(screen.getByText("Production trading readiness")).toBeInTheDocument();
  });

  it("links to every relevant live research surface", async () => {
    renderPage();
    const user = userEvent.setup();

    for (const [label, heading] of [
      ["Universe", "Universe"],
      ["Forecast Rankings", "Forecast Rankings"],
      ["Portfolio Construction", "Historical Portfolio Construction"],
      ["Efficient Frontier", "Efficient Frontier"],
      ["Risk Analytics", "Risk Analytics"],
      ["Historical Evidence", "Historical Evidence"],
    ] as const) {
      const links = screen.getAllByRole("link", { name: label });
      await user.click(links[0]);
      expect(await screen.findByRole("heading", { name: heading })).toBeInTheDocument();
      renderPage(); // reset for the next link
    }
  });

  it("never uses live/current-day market-data language or unsupported performance claims", () => {
    renderPage();
    const bodyText = document.body.textContent ?? "";
    // "Guaranteed returns" legitimately appears once, under "does not
    // claim" — the forbidden check is that nothing ever CLAIMS it.
    for (const forbidden of [/\blive market\b/i, /\bcurrent-day data\b/i, /\breal-time\b/i, /\btoday's market\b/i, /\bbest strategy\b/i, /\bwinner\b/i]) {
      expect(bodyText).not.toMatch(forbidden);
    }
    expect(bodyText).not.toMatch(/riskfecta (guarantees|promises|ensures)/i);
  });

  it("frames the project as research/educational, not investment advice", () => {
    renderPage();
    expect(screen.getByText("Investment advice")).toBeInTheDocument(); // under "does not claim"
  });
});
