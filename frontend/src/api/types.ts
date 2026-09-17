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

/**
 * One model/baseline's historical walk-forward OOS metrics (frozen Phase
 * 4-6 evaluation). `source` says exactly how the backend obtained this row
 * — "frozen_baseline_constant" (Historical Mean/Momentum 3M/Ridge, never
 * persisted, exposed from a frozen research-result artifact) or
 * "computed_from_persisted_predictions" (XGBoost/LSTM/Ensemble, recomputed
 * from the real `predictions` table on every request).
 */
export interface ModelMetricRow {
  model: string;
  label: string;
  source: "frozen_baseline_constant" | "computed_from_persisted_predictions";
  mae: number;
  rmse: number;
  directional_accuracy: number;
  pearson_corr: number;
  spearman_corr: number;
  n_obs: number;
}

export interface MetricDefinition {
  key: "mae" | "rmse" | "directional_accuracy" | "pearson_corr" | "spearman_corr";
  label: string;
  direction: "lower_is_better" | "higher_is_better" | "context_dependent";
  description: string;
}

export interface ModelDisagreementSummary {
  source: string;
  xgb_lstm_pred_pearson: number;
  xgb_lstm_pred_spearman: number;
  residual_pearson: number;
  residual_spearman: number;
  n_disagree: number;
  n_total: number;
}

/** Full body of GET /api/models/comparison. */
export interface ModelComparisonResponse {
  experiment_type: string;
  target_horizon_sessions: number;
  fold_count: number;
  prediction_count: number;
  formation_date_start: string;
  formation_date_end: string;
  models: ModelMetricRow[];
  metric_definitions: MetricDefinition[];
  disagreement: ModelDisagreementSummary | null;
}

/** One ticker's persisted weight within an official Phase 7 portfolio.
 * Base order is deterministic (ticker ascending) — ranking by weight is a
 * client-side concern, matching the Forecast Rankings convention. */
export interface HoldingRow {
  ticker: string;
  weight: number;
}

/** Frozen Phase 7 strategy identity/metadata — never a "best strategy"
 * judgment. `covariance_estimator` is "Not applicable" for EQUAL_WEIGHT (a
 * benchmark, never routed through the optimizer or MAX_WEIGHT). */
export interface StrategyInfo {
  key: string;
  label: string;
  covariance_estimator: string;
  objective: string;
  is_optimized: boolean;
}

/** Ex-ante (construction-time) figures, already persisted per-row on
 * `portfolios` — read directly by the backend, never reconstructed. All
 * `null` for EQUAL_WEIGHT (a benchmark, not an optimized construction). */
export interface PortfolioConstructionMetrics {
  expected_return_21: number | null;
  predicted_volatility_21: number | null;
  expected_sharpe_21: number | null;
}

/** Ex-post (realized-after-the-fact) figures. `turnover` is `null` — never
 * zero — for each strategy's first official formation date, where turnover
 * is genuinely undefined (no prior portfolio to compare against). */
export interface PortfolioEvaluationMetrics {
  realized_return_21: number;
  turnover: number | null;
  max_weight_observed: number;
}

/** Full body of GET /api/portfolios. */
export interface PortfolioResponse {
  formation_date: string;
  strategy: StrategyInfo;
  max_weight_constraint: number | null;
  holdings: HoldingRow[];
  active_holdings_count: number;
  largest_weight: number;
  concentration_hhi: number;
  construction: PortfolioConstructionMetrics;
  evaluation: PortfolioEvaluationMetrics;
  source: string;
}
