import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import FrontierPage from "./FrontierPage";
import * as client from "../api/client";
import type { FrontierResponse } from "../api/types";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    getPortfolioDates: vi.fn(),
    getFrontier: vi.fn(),
  };
});

const DATES = ["2022-03-28", "2022-04-27", "2026-01-02"];

function buildFrontier(overrides: Partial<FrontierResponse> = {}): FrontierResponse {
  const points = Array.from({ length: 41 }, (_, i) => ({
    expected_return_21: -0.01 + (i / 40) * 0.06,
    volatility_21: 0.03 + (i / 40) * 0.03,
    sharpe_21: 0.5,
  }));
  return {
    formation_date: "2026-01-02",
    covariance_estimator: "Ledoit-Wolf",
    forecast_horizon_sessions: 21,
    covariance_window_sessions: 252,
    max_weight_constraint: 0.1,
    risk_free_rate_21: 0.0035,
    points,
    markers: {
      min_vol: { label: "Min-Vol (official)", expected_return_21: 0.002, volatility_21: 0.039, sharpe_21: -0.036, provenance: "official_phase7_persisted" },
      max_sharpe: { label: "Max-Sharpe (official)", expected_return_21: 0.041, volatility_21: 0.056, sharpe_21: 0.668, provenance: "official_phase7_persisted" },
      equal_weight: { label: "Equal Weight", expected_return_21: 0.018, volatility_21: 0.045, sharpe_21: 0.31, provenance: "reconstructed_benchmark" },
    },
    source: "reconstructed_from_frozen_phase7_methodology",
    ...overrides,
  };
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/frontier"]}>
      <Routes>
        <Route path="/frontier" element={<FrontierPage />} />
        <Route path="/methodology" element={<h1>Methodology &amp; roadmap</h1>} />
        <Route path="/portfolio" element={<h1>Historical Portfolio Construction</h1>} />
      </Routes>
    </MemoryRouter>,
  );
}

describe("FrontierPage", () => {
  it("renders the reconstructed frontier, defaulting to the latest date and Ledoit-Wolf", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getFrontier).mockResolvedValue(buildFrontier());

    renderPage();

    expect(screen.getByRole("heading", { name: "Efficient Frontier" })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText("Risk/return frontier")).toBeInTheDocument());

    expect(screen.getByRole("button", { name: "Ledoit-Wolf" })).toHaveAttribute("aria-pressed", "true");
    expect(document.querySelector(".chart")).toBeInTheDocument();
    await waitFor(() => expect(client.getFrontier).toHaveBeenCalledWith(undefined, "LW"));
  });

  it("shows the 21-session horizon, 252-session covariance window, and 10% cap", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getFrontier).mockResolvedValue(buildFrontier());

    renderPage();
    await waitFor(() => expect(screen.getByText("Risk/return frontier")).toBeInTheDocument());

    expect(screen.getByText("21-session horizon")).toBeInTheDocument();
    expect(screen.getByText("252-session covariance window")).toBeInTheDocument();
    expect(screen.getByText("10% max position weight")).toBeInTheDocument();
    expect(screen.getAllByText("10.00%").length).toBeGreaterThan(0); // Max weight (hard cap) stat
  });

  it("shows Min-Vol and Max-Sharpe as official Phase 7 results, and Equal Weight as a reconstructed benchmark", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getFrontier).mockResolvedValue(buildFrontier());

    renderPage();
    // The reconstructed marker names also appear in the Plotly legend
    // (real Plotly renders in this jsdom test, not a stub) — assert
    // presence via getAllBy*, not a single unambiguous match.
    await waitFor(() => expect(screen.getAllByText("Min-Vol (official)").length).toBeGreaterThan(0));

    expect(screen.getAllByText("Max-Sharpe (official)").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Equal Weight").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Official Phase 7 persisted result").length).toBe(2);
    expect(screen.getByText(/Reconstructed benchmark — not implied to lie on the frontier/)).toBeInTheDocument();
  });

  it("never shows an SPXT marker or reference", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getFrontier).mockResolvedValue(buildFrontier());

    renderPage();
    await waitFor(() => expect(screen.getByText("Risk/return frontier")).toBeInTheDocument());

    expect(screen.queryByText(/SPXT/)).not.toBeInTheDocument();
  });

  it("discloses expected (construction-time) vs. realized (ex-post) and links to Portfolio Construction / Methodology", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getFrontier).mockResolvedValue(buildFrontier());

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getByText("Risk/return frontier")).toBeInTheDocument());

    expect(screen.getByText(/construction-time estimates only/)).toBeInTheDocument();
    expect(screen.getByText(/realized \(ex-post\) outcomes/)).toBeInTheDocument();

    await user.click(screen.getAllByRole("link", { name: "Portfolio Construction" })[0]);
    expect(await screen.findByRole("heading", { name: "Historical Portfolio Construction" })).toBeInTheDocument();
  });

  it("links to Methodology from the disclosure bar", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getFrontier).mockResolvedValue(buildFrontier());

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getByText("Risk/return frontier")).toBeInTheDocument());

    await user.click(screen.getAllByRole("link", { name: /Methodology & limitations/ })[0]);
    expect(await screen.findByRole("heading", { name: /Methodology & roadmap/ })).toBeInTheDocument();
  });

  it("re-fetches when the covariance selector changes", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getFrontier).mockResolvedValue(buildFrontier());

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getByText("Risk/return frontier")).toBeInTheDocument());

    vi.mocked(client.getFrontier).mockResolvedValue(buildFrontier({ covariance_estimator: "Sample" }));
    await user.click(screen.getByRole("button", { name: "Sample" }));

    await waitFor(() => expect(client.getFrontier).toHaveBeenCalledWith(undefined, "SAMPLE"));
  });

  it("re-fetches when the formation date changes", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getFrontier).mockResolvedValue(buildFrontier());

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getByText("Risk/return frontier")).toBeInTheDocument());

    vi.mocked(client.getFrontier).mockResolvedValue(buildFrontier({ formation_date: "2022-03-28" }));
    await user.selectOptions(screen.getByLabelText("Formation date (historical)"), "2022-03-28");

    await waitFor(() => expect(client.getFrontier).toHaveBeenCalledWith("2022-03-28", "LW"));
  });

  it("shows a loading state before the frontier resolves", () => {
    vi.mocked(client.getPortfolioDates).mockReturnValue(new Promise(() => {}));
    vi.mocked(client.getFrontier).mockReturnValue(new Promise(() => {}));

    renderPage();
    expect(screen.getByText(/Reconstructing frontier/)).toBeInTheDocument();
  });

  it("shows an error state with retry when the API call fails", async () => {
    vi.mocked(client.getPortfolioDates).mockResolvedValue(DATES);
    vi.mocked(client.getFrontier).mockRejectedValue(new client.ApiError("Unable to reach the RiskFecta API."));

    renderPage();
    expect(await screen.findByText("Unable to reach the RiskFecta API.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });
});
