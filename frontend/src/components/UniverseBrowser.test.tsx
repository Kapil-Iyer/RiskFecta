import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import UniverseBrowser from "./UniverseBrowser";

const UNIVERSE = [
  { ticker: "AAPL", sector: "Information Technology" },
  { ticker: "JPM", sector: "Financials" },
];

describe("UniverseBrowser", () => {
  it("lists real API tickers with their company name and lets the user select one", async () => {
    const onSelect = vi.fn();
    render(
      <UniverseBrowser state={{ status: "success", data: UNIVERSE }} selected={null} onSelect={onSelect} onRetry={() => {}} />,
    );

    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("Apple Inc.")).toBeInTheDocument();
    expect(screen.getByText("JPM")).toBeInTheDocument();
    expect(screen.getByText("JPMorgan Chase & Co.")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("option", { name: /AAPL/ }));
    expect(onSelect).toHaveBeenCalledWith("AAPL");
  });

  it("filters the universe by ticker", async () => {
    render(
      <UniverseBrowser state={{ status: "success", data: UNIVERSE }} selected={null} onSelect={() => {}} onRetry={() => {}} />,
    );

    await userEvent.type(screen.getByRole("searchbox"), "JPM");

    expect(screen.getByText("JPM")).toBeInTheDocument();
    expect(screen.queryByText("AAPL")).not.toBeInTheDocument();
  });

  it("filters the universe by company name", async () => {
    render(
      <UniverseBrowser state={{ status: "success", data: UNIVERSE }} selected={null} onSelect={() => {}} onRetry={() => {}} />,
    );

    await userEvent.type(screen.getByRole("searchbox"), "apple");

    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.queryByText("JPM")).not.toBeInTheDocument();
  });

  it("filters the universe by sector", async () => {
    render(
      <UniverseBrowser state={{ status: "success", data: UNIVERSE }} selected={null} onSelect={() => {}} onRetry={() => {}} />,
    );

    await userEvent.type(screen.getByRole("searchbox"), "financial");

    expect(screen.getByText("JPM")).toBeInTheDocument();
    expect(screen.queryByText("AAPL")).not.toBeInTheDocument();
  });

  it("shows a loading state (still announced, not hidden) and an error state", () => {
    const { rerender } = render(
      <UniverseBrowser state={{ status: "loading" }} selected={null} onSelect={() => {}} onRetry={() => {}} />,
    );
    expect(screen.getByRole("status")).toBeInTheDocument();

    rerender(
      <UniverseBrowser
        state={{ status: "error", message: "Backend unavailable" }}
        selected={null}
        onSelect={() => {}}
        onRetry={() => {}}
      />,
    );
    expect(screen.getByRole("alert")).toHaveTextContent("Backend unavailable");
  });
});
