import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import TickerDetail from "./TickerDetail";
import { getPrices } from "../api/client";
import type { PriceHistoryResponse } from "../api/types";

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return { ...actual, getPrices: vi.fn() };
});

// PriceChart/VolumeChart wrap plotly.js, which isn't meaningfully renderable
// under jsdom (no canvas/layout). Stubbing them here keeps this test focused
// on TickerDetail's own data flow — verifying the chart components receive
// correctly-shaped price data — not Plotly's internal rendering.
vi.mock("./PriceChart", () => ({
  default: ({ ticker, prices }: { ticker: string; prices: unknown[] }) => (
    <div data-testid="price-chart">{`${ticker}:${prices.length}`}</div>
  ),
}));
vi.mock("./VolumeChart", () => ({
  default: ({ ticker, prices }: { ticker: string; prices: unknown[] }) => (
    <div data-testid="volume-chart">{`${ticker}:${prices.length}`}</div>
  ),
}));

const FIXTURE: PriceHistoryResponse = {
  ticker: "AAPL",
  start: null,
  end: null,
  count: 2,
  prices: [
    { date: "2021-03-01", open: 123.75, high: 127.93, low: 122.79, close: 127.79, volume: 116307892, total_return_idx: 1.0 },
    { date: "2021-03-02", open: 128.41, high: 128.72, low: 125.01, close: 125.12, volume: 102260945, total_return_idx: 0.9791 },
  ],
};

const mockGetPrices = vi.mocked(getPrices);

beforeEach(() => {
  mockGetPrices.mockReset();
});

describe("TickerDetail", () => {
  it("shows a loading state, then renders charts with real-shaped price data", async () => {
    mockGetPrices.mockResolvedValue(FIXTURE);
    render(<TickerDetail ticker="AAPL" />);

    expect(screen.getByText(/Loading AAPL price history/)).toBeInTheDocument();

    await waitFor(() => expect(screen.getByTestId("price-chart")).toHaveTextContent("AAPL:2"));
    expect(screen.getByTestId("volume-chart")).toHaveTextContent("AAPL:2");
    expect(screen.getByText(/2021-03-01 → 2021-03-02/)).toBeInTheDocument();
  });

  it("shows an empty-state message when the API returns zero observations", async () => {
    mockGetPrices.mockResolvedValue({ ticker: "AAPL", start: "2099-01-01", end: "2099-01-31", count: 0, prices: [] });
    render(<TickerDetail ticker="AAPL" />);

    await waitFor(() => expect(screen.getByText(/No price observations found/)).toBeInTheDocument());
    expect(screen.queryByTestId("price-chart")).not.toBeInTheDocument();
  });

  it("shows an API error without ever rendering a raw exception", async () => {
    mockGetPrices.mockRejectedValue(Object.assign(new Error("Database unavailable"), { name: "ApiError" }));
    render(<TickerDetail ticker="AAPL" />);

    await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
  });

  it("blocks an invalid date range client-side without calling the API", async () => {
    mockGetPrices.mockResolvedValue(FIXTURE);
    render(<TickerDetail ticker="AAPL" />);
    await waitFor(() => expect(mockGetPrices).toHaveBeenCalledTimes(1));

    // A start date alone (end still empty) is a valid, real request.
    fireEvent.change(screen.getByLabelText("Start"), { target: { value: "2025-02-01" } });
    await waitFor(() => expect(mockGetPrices).toHaveBeenCalledTimes(2));

    // Now end < start — must be rejected client-side, no further request.
    fireEvent.change(screen.getByLabelText("End"), { target: { value: "2025-01-01" } });

    expect(await screen.findByText(/Start date must not be after end date/)).toBeInTheDocument();
    expect(mockGetPrices).toHaveBeenCalledTimes(2);
  });
});
