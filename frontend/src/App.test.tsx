import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import AppRoutes from "./AppRoutes";
import * as client from "./api/client";

vi.mock("./api/client", async () => {
  const actual = await vi.importActual<typeof import("./api/client")>("./api/client");
  return {
    ...actual,
    getMarketSummary: vi.fn(),
    getUniverse: vi.fn(),
    getReadiness: vi.fn(),
    getPrices: vi.fn(),
    getPredictions: vi.fn(),
    getPredictionDates: vi.fn(),
    getModelComparison: vi.fn(),
    getPortfolioDates: vi.fn(),
    getPortfolioStrategies: vi.fn(),
    getPortfolio: vi.fn(),
  };
});

/** Overview fetches universe/predictions/model-comparison/portfolio-dates
 * on mount; every test that renders "/" needs these mocked to avoid an
 * uncontrolled real fetch, even when the test itself only cares about
 * shell-level behavior. */
function mockOverviewDataAsPending() {
  vi.mocked(client.getPredictions).mockReturnValue(new Promise(() => {}));
  vi.mocked(client.getModelComparison).mockReturnValue(new Promise(() => {}));
  vi.mocked(client.getPortfolioDates).mockReturnValue(new Promise(() => {}));
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <AppRoutes />
    </MemoryRouter>,
  );
}

describe("App routing shell", () => {
  it("renders the RiskFecta shell, nav, and universe data on /universe", async () => {
    vi.mocked(client.getMarketSummary).mockResolvedValue({
      ticker_count: 50,
      price_row_count: 62800,
      first_date: "2021-03-01",
      last_date: "2026-02-27",
    });
    vi.mocked(client.getUniverse).mockResolvedValue([{ ticker: "AAPL", sector: "Information Technology" }]);
    vi.mocked(client.getReadiness).mockResolvedValue({ status: "ok", database: "connected" });

    renderAt("/universe");

    // Shell chrome (persistent across routes).
    expect(screen.getByRole("heading", { name: "RiskFecta" })).toBeInTheDocument();
    expect(screen.getByText("Quantitative Portfolio Intelligence Platform")).toBeInTheDocument();
    expect(screen.getByRole("navigation", { name: "Research surfaces" })).toBeInTheDocument();

    // Universe-page content.
    await waitFor(() => expect(screen.getByText("62,800")).toBeInTheDocument());
    expect(screen.getByText("2026-02-27")).toBeInTheDocument();
    expect(screen.getByText("AAPL")).toBeInTheDocument();

    // Data-freshness banner is present and honest about the cutoff.
    expect(screen.getByText(/Historical data through/)).toBeInTheDocument();

    // Backend is healthy — no unavailability banner.
    expect(screen.queryByText(/Backend unavailable/)).not.toBeInTheDocument();
  });

  it("shows a backend-unavailable banner on any route when readiness fails", async () => {
    vi.mocked(client.getMarketSummary).mockResolvedValue({
      ticker_count: 50,
      price_row_count: 62800,
      first_date: "2021-03-01",
      last_date: "2026-02-27",
    });
    vi.mocked(client.getUniverse).mockResolvedValue([]);
    vi.mocked(client.getReadiness).mockRejectedValue(new client.ApiError("Unable to reach the RiskFecta API."));
    mockOverviewDataAsPending();

    renderAt("/");

    await waitFor(() => expect(screen.getByText(/Backend unavailable/)).toBeInTheDocument());
  });

  it("navigates between routes via the nav bar without page reload", async () => {
    vi.mocked(client.getMarketSummary).mockResolvedValue({
      ticker_count: 50,
      price_row_count: 62800,
      first_date: "2021-03-01",
      last_date: "2026-02-27",
    });
    vi.mocked(client.getUniverse).mockResolvedValue([]);
    vi.mocked(client.getReadiness).mockResolvedValue({ status: "ok", database: "connected" });
    mockOverviewDataAsPending();

    const user = userEvent.setup();
    renderAt("/");

    // Overview page content.
    expect(screen.getByRole("heading", { name: "Research pipeline" })).toBeInTheDocument();

    await user.click(screen.getByRole("link", { name: "Methodology" }));
    expect(await screen.findByRole("heading", { name: /Methodology & roadmap/ })).toBeInTheDocument();

    // Efficient Frontier is still a not-yet-built Coming Soon surface
    // (Forecast Rankings, Model Comparison, and Portfolio Construction each
    // have their own dedicated test suites in pages/*.test.tsx).
    await user.click(screen.getByRole("link", { name: "Efficient Frontier" }));
    expect(await screen.findByRole("heading", { name: "Efficient Frontier" })).toBeInTheDocument();
    // A not-yet-built surface must never show fabricated numbers/charts.
    expect(document.querySelector(".chart")).not.toBeInTheDocument();
    expect(screen.getByText(/not yet implemented/)).toBeInTheDocument();
  });
});
