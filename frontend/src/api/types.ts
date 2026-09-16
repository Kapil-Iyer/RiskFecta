/**
 * TypeScript interfaces mirroring the ACTUAL Phase 2A FastAPI response shapes
 * (see app/schemas.py and the live responses captured during Phase 2B
 * development — not invented ahead of the real API).
 */

export interface HealthResponse {
  status: string;
}

export interface ReadinessResponse {
  status: string;
  database: string;
}

/** One row of GET /api/universe. `sector` is null when the local read-only
 * static snapshot isn't available (e.g. the backend has no data/raw/ CSVs). */
export interface UniverseTicker {
  ticker: string;
  sector: string | null;
}

/** One row of GET /api/prices/{ticker}.prices — a single trading session. */
export interface PriceObservation {
  date: string; // ISO date, e.g. "2021-03-01"
  open: number | null;
  high: number | null;
  low: number | null;
  close: number;
  volume: number | null;
  total_return_idx: number | null;
}

/** Full body of GET /api/prices/{ticker}. */
export interface PriceHistoryResponse {
  ticker: string;
  start: string | null;
  end: string | null;
  count: number;
  prices: PriceObservation[];
}

/** Body of GET /api/market/summary. */
export interface MarketSummaryResponse {
  ticker_count: number;
  price_row_count: number;
  first_date: string | null;
  last_date: string | null;
}

/**
 * One ticker's forecast cross-section at a formation date — frozen Phase
 * 4-6 historical walk-forward evidence, never a live/current forecast.
 * `actual_return`/`directional_correct` are ex-post EVALUATION fields,
 * filled in only after the 21-session target_date passed — they were not
 * available at formation time. See ForecastRankingsPage's explicit
 * "Show realized outcome" control before rendering either field.
 */
export interface PredictionRow {
  ticker: string;
  xgb_pred: number | null;
  lstm_pred: number | null;
  ensemble_pred: number | null;
  actual_return: number | null;
  directional_correct: boolean | null;
}

/** Full body of GET /api/predictions. */
export interface PredictionCrossSectionResponse {
  formation_date: string; // ISO date, e.g. "2022-02-25"
  target_date: string; // formation_date + 21 valid trading sessions
  count: number;
  /** Base order is deterministic (ticker ascending) — ranking by a chosen
   * model is done client-side, see pages/forecastRanking.ts. */
  predictions: PredictionRow[];
}
