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
