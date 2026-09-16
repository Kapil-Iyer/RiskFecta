import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter, Outlet, Route, Routes } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import ForecastRankingsPage from "./ForecastRankingsPage";
import * as client from "../api/client";
import type { AppOutletContext } from "../AppShell";

vi.mock("../api/client", async () => {
  const actual = await vi.importActual<typeof import("../api/client")>("../api/client");
  return {
    ...actual,
    getPredictionDates: vi.fn(),
    getPredictions: vi.fn(),
    getUniverse: vi.fn(),
  };
});

const DATES = ["2022-02-25", "2022-03-28", "2026-01-02"];

const CROSS_SECTION_LATEST = {
  formation_date: "2026-01-02",
  target_date: "2026-02-02",
  count: 3,
  predictions: [
    { ticker: "AAPL", xgb_pred: 0.09, lstm_pred: -0.02, ensemble_pred: 0.035, actual_return: 0.04, directional_correct: true },
    { ticker: "MSFT", xgb_pred: -0.01, lstm_pred: 0.08, ensemble_pred: 0.035, actual_return: -0.01, directional_correct: false },
    { ticker: "NVDA", xgb_pred: 0.02, lstm_pred: 0.06, ensemble_pred: 0.038, actual_return: 0.05, directional_correct: true },
  ],
};

const UNIVERSE = [
  { ticker: "AAPL", sector: "Information Technology" },
  { ticker: "MSFT", sector: "Information Technology" },
  { ticker: "NVDA", sector: "Information Technology" },
];

const SUCCESS_SUMMARY: AppOutletContext = {
  summaryState: {
    status: "success",
    data: { ticker_count: 50, price_row_count: 62800, first_date: "2021-03-01", last_date: "2026-02-27" },
  },
  retrySummary: vi.fn(),
};

function renderPage(context: AppOutletContext = SUCCESS_SUMMARY) {
  function Wrapper() {
    return <Outlet context={context} />;
  }
  return render(
    <MemoryRouter initialEntries={["/forecasts"]}>
      <Routes>
        <Route element={<Wrapper />}>
          <Route path="/forecasts" element={<ForecastRankingsPage />} />
          <Route path="/methodology" element={<h1>Methodology &amp; roadmap</h1>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  );
}

function tickerOrder() {
  return screen.getAllByRole("row").slice(1).map((row) => within(row).getAllByRole("cell")[1].textContent);
}

describe("ForecastRankingsPage", () => {
  it("renders real, API-backed rankings defaulting to Ensemble", async () => {
    vi.mocked(client.getPredictionDates).mockResolvedValue(DATES);
    vi.mocked(client.getPredictions).mockResolvedValue(CROSS_SECTION_LATEST);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);

    renderPage();

    expect(screen.getByRole("heading", { name: "Forecast Rankings" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ensemble" })).toHaveAttribute("aria-pressed", "true");

    await waitFor(() => expect(screen.getAllByRole("row")).toHaveLength(4)); // header + 3 tickers
    // Ensemble: NVDA (0.038) > AAPL (0.035) = MSFT (0.035) → tie breaks alphabetically.
    expect(tickerOrder()).toEqual(["NVDA", "AAPL", "MSFT"]);

    // Real freshness value from the (mocked) backend, not a hardcoded string.
    expect(screen.getByText("2026-02-27")).toBeInTheDocument();
    // Evaluation data is not shown by default.
    expect(screen.queryByText("Realized (21d)")).not.toBeInTheDocument();
  });

  it("re-ranks when the selected model changes, without a new predictions request", async () => {
    vi.mocked(client.getPredictionDates).mockResolvedValue(DATES);
    vi.mocked(client.getPredictions).mockResolvedValue(CROSS_SECTION_LATEST);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getAllByRole("row")).toHaveLength(4));

    await user.click(screen.getByRole("button", { name: "XGBoost" }));
    expect(screen.getByRole("button", { name: "XGBoost" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "Ensemble" })).toHaveAttribute("aria-pressed", "false");
    // XGBoost: AAPL (0.09) > NVDA (0.02) > MSFT (-0.01).
    expect(tickerOrder()).toEqual(["AAPL", "NVDA", "MSFT"]);

    expect(client.getPredictions).toHaveBeenCalledTimes(1); // model switch is client-side only
  });

  it("re-fetches predictions when a different formation date is selected", async () => {
    vi.mocked(client.getPredictionDates).mockResolvedValue(DATES);
    vi.mocked(client.getPredictions).mockResolvedValue(CROSS_SECTION_LATEST);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getAllByRole("row")).toHaveLength(4));

    const earlierCrossSection = { ...CROSS_SECTION_LATEST, formation_date: "2022-02-25", target_date: "2022-03-25" };
    vi.mocked(client.getPredictions).mockResolvedValue(earlierCrossSection);

    await user.selectOptions(screen.getByLabelText(/Formation date/), "2022-02-25");

    await waitFor(() => expect(client.getPredictions).toHaveBeenCalledWith("2022-02-25"));
    await waitFor(() => expect(screen.getByText(/Formation date 2022-02-25/)).toBeInTheDocument());
  });

  it("shows realized outcome only after the explicit evaluation toggle", async () => {
    vi.mocked(client.getPredictionDates).mockResolvedValue(DATES);
    vi.mocked(client.getPredictions).mockResolvedValue(CROSS_SECTION_LATEST);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getAllByRole("row")).toHaveLength(4));

    expect(screen.queryByText("Realized (21d)")).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Show evaluation" }));
    expect(screen.getByText("Realized (21d)")).toBeInTheDocument();
    expect(screen.getByText("+4.00%")).toBeInTheDocument(); // AAPL's actual_return
  });

  it("shows a loading state before the cross-section resolves", () => {
    vi.mocked(client.getPredictionDates).mockReturnValue(new Promise(() => {}));
    vi.mocked(client.getPredictions).mockReturnValue(new Promise(() => {}));
    vi.mocked(client.getUniverse).mockReturnValue(new Promise(() => {}));

    renderPage();

    expect(screen.getByText("Loading forecast cross-section…")).toBeInTheDocument();
  });

  it("shows an error state with retry when the API call fails", async () => {
    vi.mocked(client.getPredictionDates).mockResolvedValue(DATES);
    vi.mocked(client.getPredictions).mockRejectedValue(new client.ApiError("Unable to reach the RiskFecta API."));
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);

    renderPage();

    expect(await screen.findByText("Unable to reach the RiskFecta API.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("links to the methodology page", async () => {
    vi.mocked(client.getPredictionDates).mockResolvedValue(DATES);
    vi.mocked(client.getPredictions).mockResolvedValue(CROSS_SECTION_LATEST);
    vi.mocked(client.getUniverse).mockResolvedValue(UNIVERSE);

    const user = userEvent.setup();
    renderPage();
    await waitFor(() => expect(screen.getAllByRole("row")).toHaveLength(4));

    await user.click(screen.getByRole("link", { name: /Methodology & limitations/ }));
    expect(await screen.findByRole("heading", { name: /Methodology & roadmap/ })).toBeInTheDocument();
  });
});
