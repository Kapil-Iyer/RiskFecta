import { render, screen, waitFor } from "@testing-library/react";
import { fireEvent } from "@testing-library/react";
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
    { date: "2026-02-27", open: 272.81, high: 272.81, low: 262.89, close: 264.18, volume: 72333238, total_return_idx: 2.1223 },
  ],
};

const mockGetPrices = vi.mocked(getPrices);

beforeEach(() => {
  mockGetPrices.mockReset();
});

describe("TickerDetail", () => {
  it("shows the company name, sector, and 'historical data' framing — never live/current-price wording", async () => {
    mockGetPrices.mockResolvedValue(FIXTURE);
    render(<TickerDetail ticker="AAPL" />);

    expect(await screen.findByRole("heading", { name: "Apple Inc." })).toBeInTheDocument();
    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("Information Technology")).toBeInTheDocument();
    expect(screen.getByText("Historical data")).toBeInTheDocument();

    const bodyText = document.body.textContent ?? "";
    expect(bodyText).not.toMatch(/current price/i);
    expect(bodyText).not.toMatch(/live price/i);
    expect(bodyText).not.toMatch(/today'?s price/i);
  });

  it("shows the latest dataset observation and session stats, then renders charts with real-shaped data", async () => {
    mockGetPrices.mockResolvedValue(FIXTURE);
    render(<TickerDetail ticker="AAPL" />);

    expect(await screen.findByText("Latest dataset observation")).toBeInTheDocument();
    expect(screen.getByText("$264.18")).toBeInTheDocument();
    expect(screen.getByText("Feb 27, 2026")).toBeInTheDocument();
    expect(screen.getByText("Mar 2021 — Feb 2026")).toBeInTheDocument();

    await waitFor(() => expect(screen.getByTestId("price-chart")).toHaveTextContent("AAPL:2"));
    expect(screen.getByTestId("volume-chart")).toHaveTextContent("AAPL:2");
  });

  it("shows an empty-state message when the API returns zero observations", async () => {
    mockGetPrices.mockResolvedValue({ ticker: "AAPL", start: null, end: null, count: 0, prices: [] });
    render(<TickerDetail ticker="AAPL" />);

    await waitFor(() => expect(screen.getByText(/No price observations found/)).toBeInTheDocument());
    expect(screen.queryByTestId("price-chart")).not.toBeInTheDocument();
  });

  it("shows an API error without ever rendering a raw exception", async () => {
    mockGetPrices.mockRejectedValue(Object.assign(new Error("Database unavailable"), { name: "ApiError" }));
    render(<TickerDetail ticker="AAPL" />);

    await waitFor(() => expect(screen.getAllByRole("alert").length).toBeGreaterThan(0));
  });

  it("blocks an invalid date range client-side without an extra API call", async () => {
    mockGetPrices.mockResolvedValue(FIXTURE);
    render(<TickerDetail ticker="AAPL" />);
    await waitFor(() => expect(mockGetPrices).toHaveBeenCalledTimes(1)); // initial unfiltered fetch

    // A start date alone (end still empty) is a valid, real filtered request.
    fireEvent.change(screen.getByLabelText("Start"), { target: { value: "2025-02-01" } });
    await waitFor(() => expect(mockGetPrices).toHaveBeenCalledTimes(2));

    // Now end < start — must be rejected client-side, no further request.
    fireEvent.change(screen.getByLabelText("End"), { target: { value: "2025-01-01" } });

    expect(await screen.findByText(/Start date must not be after end date/)).toBeInTheDocument();
    expect(mockGetPrices).toHaveBeenCalledTimes(2);
  });

  it("quick-range buttons are disabled until the ticker's own date range is known, then apply a real filter", async () => {
    mockGetPrices.mockResolvedValue(FIXTURE);
    render(<TickerDetail ticker="AAPL" />);

    const oneMonthButton = screen.getByRole("button", { name: "1M" });
    expect(oneMonthButton).toBeDisabled();

    await waitFor(() => expect(oneMonthButton).toBeEnabled());
    mockGetPrices.mockClear();

    fireEvent.click(oneMonthButton);

    await waitFor(() => expect(mockGetPrices).toHaveBeenCalledWith("AAPL", { start: "2026-01-27", end: "2026-02-27" }));
    expect((screen.getByLabelText("Start") as HTMLInputElement).value).toBe("2026-01-27");
  });

  it("resets any date filter when the selected ticker changes", async () => {
    mockGetPrices.mockResolvedValue(FIXTURE);
    const { rerender } = render(<TickerDetail ticker="AAPL" />);
    await waitFor(() => expect(mockGetPrices).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByLabelText("Start"), { target: { value: "2025-02-01" } });
    await waitFor(() => expect((screen.getByLabelText("Start") as HTMLInputElement).value).toBe("2025-02-01"));

    rerender(<TickerDetail ticker="JPM" />);
    expect((screen.getByLabelText("Start") as HTMLInputElement).value).toBe("");
  });
});
