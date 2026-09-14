import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import MarketSummarySection from "./MarketSummarySection";

describe("MarketSummarySection", () => {
  it("renders the real market summary values from mocked API data", () => {
    render(
      <MarketSummarySection
        state={{
          status: "success",
          data: { ticker_count: 50, price_row_count: 62800, first_date: "2021-03-01", last_date: "2026-02-27" },
        }}
        onRetry={() => {}}
      />,
    );

    expect(screen.getByText("50")).toBeInTheDocument();
    expect(screen.getByText("62,800")).toBeInTheDocument();
    expect(screen.getByText("2021-03-01")).toBeInTheDocument();
    expect(screen.getByText("2026-02-27")).toBeInTheDocument();
  });

  it("shows a loading state", () => {
    render(<MarketSummarySection state={{ status: "loading" }} onRetry={() => {}} />);
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("shows an error state with a working retry button", () => {
    const onRetry = vi.fn();
    render(
      <MarketSummarySection state={{ status: "error", message: "Database unavailable" }} onRetry={onRetry} />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent("Database unavailable");
    screen.getByRole("button", { name: "Retry" }).click();
    expect(onRetry).toHaveBeenCalledOnce();
  });
});
