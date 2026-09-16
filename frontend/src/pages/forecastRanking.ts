import type { PredictionRow } from "../api/types";

export type ModelKey = "ensemble" | "xgb" | "lstm";

export interface ModelOption {
  key: ModelKey;
  label: string;
  field: "ensemble_pred" | "xgb_pred" | "lstm_pred";
}

export const MODEL_OPTIONS: ModelOption[] = [
  { key: "ensemble", label: "Ensemble", field: "ensemble_pred" },
  { key: "xgb", label: "XGBoost", field: "xgb_pred" },
  { key: "lstm", label: "LSTM", field: "lstm_pred" },
];

export interface RankedPrediction extends PredictionRow {
  rank: number;
}

/**
 * Ranks a formation date's 50-ticker cross-section by the chosen model's
 * predicted return, highest first. Deterministic regardless of input order:
 * nulls always sink to the bottom, and exact ties break by ticker
 * (alphabetical) rather than by array position — so re-ranking is a pure
 * function of (rows, model), never a source of surprising reordering.
 */
export function rankPredictions(rows: PredictionRow[], model: ModelKey): RankedPrediction[] {
  const field = MODEL_OPTIONS.find((m) => m.key === model)!.field;

  const sorted = [...rows].sort((a, b) => {
    const av = a[field];
    const bv = b[field];
    if (av === null && bv === null) return a.ticker.localeCompare(b.ticker);
    if (av === null) return 1;
    if (bv === null) return -1;
    if (av === bv) return a.ticker.localeCompare(b.ticker);
    return bv - av;
  });

  return sorted.map((row, index) => ({ ...row, rank: index + 1 }));
}
