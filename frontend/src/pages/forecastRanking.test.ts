import { describe, expect, it } from "vitest";
import { rankPredictions } from "./forecastRanking";
import type { PredictionRow } from "../api/types";

function row(overrides: Partial<PredictionRow> & { ticker: string }): PredictionRow {
  return {
    xgb_pred: null,
    lstm_pred: null,
    ensemble_pred: null,
    actual_return: null,
    directional_correct: null,
    ...overrides,
  };
}

describe("rankPredictions", () => {
  it("ranks by ensemble_pred descending and assigns 1-based rank", () => {
    const rows = [
      row({ ticker: "AAPL", ensemble_pred: 0.01 }),
      row({ ticker: "MSFT", ensemble_pred: 0.05 }),
      row({ ticker: "NVDA", ensemble_pred: 0.03 }),
    ];

    const ranked = rankPredictions(rows, "ensemble");

    expect(ranked.map((r) => r.ticker)).toEqual(["MSFT", "NVDA", "AAPL"]);
    expect(ranked.map((r) => r.rank)).toEqual([1, 2, 3]);
  });

  it("re-ranks by xgb_pred or lstm_pred when a different model is selected", () => {
    const rows = [
      row({ ticker: "AAPL", ensemble_pred: 0.01, xgb_pred: 0.09, lstm_pred: -0.02 }),
      row({ ticker: "MSFT", ensemble_pred: 0.05, xgb_pred: -0.01, lstm_pred: 0.08 }),
    ];

    expect(rankPredictions(rows, "xgb").map((r) => r.ticker)).toEqual(["AAPL", "MSFT"]);
    expect(rankPredictions(rows, "lstm").map((r) => r.ticker)).toEqual(["MSFT", "AAPL"]);
  });

  it("sinks null predictions to the bottom regardless of input order", () => {
    const rows = [
      row({ ticker: "ZZZ", ensemble_pred: null }),
      row({ ticker: "AAA", ensemble_pred: 0.02 }),
    ];

    expect(rankPredictions(rows, "ensemble").map((r) => r.ticker)).toEqual(["AAA", "ZZZ"]);
  });

  it("breaks exact ties alphabetically by ticker, deterministically", () => {
    const rows = [
      row({ ticker: "ZETA", ensemble_pred: 0.04 }),
      row({ ticker: "ALPHA", ensemble_pred: 0.04 }),
    ];

    // Same input in reverse order must produce the same ranked output.
    const forward = rankPredictions(rows, "ensemble").map((r) => r.ticker);
    const reversed = rankPredictions([...rows].reverse(), "ensemble").map((r) => r.ticker);

    expect(forward).toEqual(["ALPHA", "ZETA"]);
    expect(reversed).toEqual(forward);
  });

  it("does not mutate the input array", () => {
    const rows = [row({ ticker: "B", ensemble_pred: 0.01 }), row({ ticker: "A", ensemble_pred: 0.02 })];
    const original = [...rows];

    rankPredictions(rows, "ensemble");

    expect(rows).toEqual(original);
  });
});
