import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import App from "./App";
import * as client from "./api/client";

vi.mock("./api/client", async () => {
  const actual = await vi.importActual<typeof import("./api/client")>("./api/client");
  return {
    ...actual,
    getMarketSummary: vi.fn(),
    getUniverse: vi.fn(),
    getReadiness: vi.fn(),
    getPrices: vi.fn(),
  };
});

describe("App", () => {
  it("renders the RiskFecta shell and market summary from mocked API data", async () => {
    vi.mocked(client.getMarketSummary).mockResolvedValue({
      ticker_count: 50,
      price_row_count: 62800,
      first_date: "2021-03-01",
      last_date: "2026-02-27",
    });
    vi.mocked(client.getUniverse).mockResolvedValue([{ ticker: "AAPL", sector: "Information Technology" }]);
    vi.mocked(client.getReadiness).mockResolvedValue({ status: "ok", database: "connected" });

    render(<App />);

    expect(screen.getByRole("heading", { name: "RiskFecta" })).toBeInTheDocument();
    expect(screen.getByText("Quantitative Portfolio Intelligence Platform")).toBeInTheDocument();

    await waitFor(() => expect(screen.getByText("62,800")).toBeInTheDocument());
    expect(screen.getByText("2026-02-27")).toBeInTheDocument();
    expect(screen.getByText("AAPL")).toBeInTheDocument();

    // Data-freshness banner is present and honest about the cutoff.
    expect(screen.getByText(/Historical data through/)).toBeInTheDocument();
    expect(screen.getByText(/Model forecasts in development/)).toBeInTheDocument();

    // Backend is healthy — no unavailability banner.
    expect(screen.queryByText(/Backend unavailable/)).not.toBeInTheDocument();
  });

  it("shows a backend-unavailable banner when readiness fails", async () => {
    vi.mocked(client.getMarketSummary).mockResolvedValue({
      ticker_count: 50,
      price_row_count: 62800,
      first_date: "2021-03-01",
      last_date: "2026-02-27",
    });
    vi.mocked(client.getUniverse).mockResolvedValue([]);
    vi.mocked(client.getReadiness).mockRejectedValue(new client.ApiError("Unable to reach the RiskFecta API."));

    render(<App />);

    await waitFor(() => expect(screen.getByText(/Backend unavailable/)).toBeInTheDocument());
  });
});
