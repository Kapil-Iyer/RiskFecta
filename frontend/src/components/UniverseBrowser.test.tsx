import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import UniverseBrowser from "./UniverseBrowser";

const UNIVERSE = [
  { ticker: "AAPL", sector: "Information Technology" },
  { ticker: "JPM", sector: "Financials" },
];

describe("UniverseBrowser", () => {
  it("lists real API tickers and lets the user select one", async () => {
    const onSelect = vi.fn();
    render(
      <UniverseBrowser state={{ status: "success", data: UNIVERSE }} selected={null} onSelect={onSelect} onRetry={() => {}} />,
    );

    expect(screen.getByText("AAPL")).toBeInTheDocument();
    expect(screen.getByText("JPM")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("option", { name: /AAPL/ }));
    expect(onSelect).toHaveBeenCalledWith("AAPL");
  });

  it("filters the universe by search query", async () => {
    render(
      <UniverseBrowser state={{ status: "success", data: UNIVERSE }} selected={null} onSelect={() => {}} onRetry={() => {}} />,
    );

    await userEvent.type(screen.getByRole("searchbox"), "JPM");

    expect(screen.getByText("JPM")).toBeInTheDocument();
    expect(screen.queryByText("AAPL")).not.toBeInTheDocument();
  });

  it("shows a loading state and an error state", () => {
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
