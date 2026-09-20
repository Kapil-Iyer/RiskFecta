"""
Pydantic request/response schemas for the RiskFecta API.

Every field here reflects data that already exists in `prices_raw`, the
Phase 1 static snapshot, project config, the frozen Phase 4-6 `predictions`
table, or (as of Phase 8C) the frozen, official Phase 7 portfolio experiment
persisted in `portfolios`/`risk_metrics`.
"""
from __future__ import annotations

from datetime import date
from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str


class ReadinessResponse(BaseModel):
    status: str
    database: str


class UniverseTicker(BaseModel):
    ticker: str
    # Single-snapshot metadata only (pipeline.normalize.normalize_static_fields),
    # never a historical/predictive field. None if the local static snapshot
    # CSV isn't available (e.g. in CI, where data/raw/ is gitignored).
    sector: Optional[str] = None


class PriceObservation(BaseModel):
    date: date
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: float
    volume: Optional[int] = None
    total_return_idx: Optional[float] = None


class PriceHistoryResponse(BaseModel):
    ticker: str
    start: Optional[date] = None
    end: Optional[date] = None
    count: int
    prices: List[PriceObservation]


class MarketSummaryResponse(BaseModel):
    ticker_count: int
    price_row_count: int
    first_date: Optional[date] = None
    last_date: Optional[date] = None


class PredictionRow(BaseModel):
    """One ticker's forecast cross-section at a formation date (Phase 4-6,
    frozen historical walk-forward — never a live/current forecast).

    `actual_return`/`directional_correct` are ex-post evaluation fields
    filled in only after the 21-session target_date has passed — they were
    NOT available at formation time and must never be presented as if they
    were an input to the forecast itself.
    """

    ticker: str
    xgb_pred: Optional[float] = None
    lstm_pred: Optional[float] = None
    ensemble_pred: Optional[float] = None
    actual_return: Optional[float] = None
    directional_correct: Optional[bool] = None


class PredictionCrossSectionResponse(BaseModel):
    formation_date: date
    target_date: date
    count: int
    # Base order is deterministic (ticker ascending) — see
    # app/routes/predictions.py. Ranking by a chosen model is a client-side
    # concern, not an API parameter.
    predictions: List[PredictionRow]


class ModelMetricRow(BaseModel):
    """One model/baseline's historical walk-forward OOS metrics.

    `source` distinguishes how this row was obtained — never left implicit:
    - "frozen_baseline_constant": Historical Mean / Momentum 3M / Ridge —
      baseline forecasts are not persisted, so these are a frozen research
      artifact (models/frozen_phase4_baselines.py), not a live computation.
    - "computed_from_persisted_predictions": XGBoost / LSTM / Ensemble —
      recomputed on each request from the real `predictions` table using
      the frozen evaluation code (models/metrics.py, models/ensemble.py).
    """

    model: str
    label: str
    source: str
    mae: float
    rmse: float
    directional_accuracy: float
    pearson_corr: float
    spearman_corr: float
    n_obs: int


class MetricDefinition(BaseModel):
    key: str
    label: str
    direction: str  # "lower_is_better" | "higher_is_better" | "context_dependent"
    description: str


class ModelDisagreementSummary(BaseModel):
    """Optional XGBoost/LSTM disagreement diagnostics (ML_SPEC.md/Phase 6A
    task brief) — recomputed from persisted predictions, same provenance
    rule as the ML model rows above."""

    source: str
    xgb_lstm_pred_pearson: float
    xgb_lstm_pred_spearman: float
    residual_pearson: float
    residual_spearman: float
    n_disagree: int
    n_total: int


class ModelComparisonResponse(BaseModel):
    experiment_type: str
    target_horizon_sessions: int
    fold_count: int
    prediction_count: int
    formation_date_start: date
    formation_date_end: date
    models: List[ModelMetricRow]
    metric_definitions: List[MetricDefinition]
    disagreement: Optional[ModelDisagreementSummary] = None


class HoldingRow(BaseModel):
    """One ticker's persisted weight within an official Phase 7 portfolio.
    Base order is deterministic (ticker ascending) — ranking by weight is a
    client-side concern, matching the Forecast Rankings convention."""

    ticker: str
    weight: float


class StrategyInfo(BaseModel):
    """Frozen Phase 7 strategy identity/metadata — never a "best strategy"
    judgment. `covariance_estimator` is "Not applicable" for EQUAL_WEIGHT
    (a benchmark, never routed through the optimizer or MAX_WEIGHT)."""

    key: str
    label: str
    covariance_estimator: str
    objective: str
    is_optimized: bool


class PortfolioConstructionMetrics(BaseModel):
    """Ex-ante (construction-time) figures, already persisted per-row on
    `portfolios` (`target_return`/`portfolio_vol`/`sharpe_ratio`) — read
    directly, never reconstructed from mu/Sigma/rf here. All `None` for
    EQUAL_WEIGHT (a benchmark, not an optimized construction)."""

    expected_return_21: Optional[float] = None
    predicted_volatility_21: Optional[float] = None
    expected_sharpe_21: Optional[float] = None


class PortfolioEvaluationMetrics(BaseModel):
    """Ex-post (realized-after-the-fact) figures from `risk_metrics`.
    `turnover` is `None` — not zero — for each strategy's first official
    formation date, where turnover is genuinely undefined (no prior
    portfolio to compare against)."""

    realized_return_21: float
    turnover: Optional[float] = None
    max_weight_observed: float


class PortfolioResponse(BaseModel):
    formation_date: date
    strategy: StrategyInfo
    max_weight_constraint: Optional[float] = None
    holdings: List[HoldingRow]
    active_holdings_count: int
    largest_weight: float
    concentration_hhi: float
    construction: PortfolioConstructionMetrics
    evaluation: PortfolioEvaluationMetrics
    source: str


class FrontierPoint(BaseModel):
    """One reconstructed constrained-efficient-frontier point at formation
    time (`app/routes/frontier.py`) — always ex-ante. `sharpe_21` is
    `None` only when volatility is exactly zero (never a divide-by-zero)."""

    expected_return_21: float
    volatility_21: float
    sharpe_21: Optional[float] = None


class FrontierMarker(BaseModel):
    """A named reference point on the frontier chart.

    Min-Vol/Max-Sharpe markers carry `provenance="official_phase7_persisted"`
    — their coordinates are the OFFICIAL persisted Phase 7
    `target_return`/`portfolio_vol`/`sharpe_ratio`, read directly, never a
    freshly re-optimized copy. Equal Weight carries
    `provenance="reconstructed_benchmark"` — its weights are the exact
    conceptual 1/50 benchmark, but its coordinates are recomputed from the
    SAME request's reconstructed mu_21/Sigma_21 so they sit in the curve's
    coordinate system. Equal Weight is never implied to lie on the
    frontier."""

    label: str
    expected_return_21: float
    volatility_21: float
    sharpe_21: Optional[float] = None
    provenance: str


class FrontierMarkers(BaseModel):
    min_vol: FrontierMarker
    max_sharpe: FrontierMarker
    equal_weight: FrontierMarker


class FrontierResponse(BaseModel):
    """Reconstructed (never persisted) Phase 7 efficient frontier at one
    historical formation date. Entirely construction-time / ex-ante — see
    app/routes/frontier.py for the exact reconstruction path and the
    causality guarantee."""

    formation_date: date
    covariance_estimator: str  # "Sample" | "Ledoit-Wolf"
    forecast_horizon_sessions: int
    covariance_window_sessions: int
    max_weight_constraint: float
    risk_free_rate_21: float
    points: List[FrontierPoint]
    markers: FrontierMarkers
    source: str


class AssetRiskRow(BaseModel):
    """One ticker's formation-time risk decomposition within an official
    Phase 7 portfolio (`app/routes/risk.py`). `component_risk_contribution`
    (RC_i = w_i * (Sigma_21 w)_i / sigma_p) is the AUTHORITATIVE
    decomposition, in the same volatility units as the portfolio's
    predicted volatility — summing it across all 50 assets reproduces
    `portfolio.predicted_volatility_21` exactly (within numerical
    tolerance). It is NEVER clamped/abs'd/renormalized: a genuine
    diversifying position can carry a negative contribution, and that
    sign is preserved. `risk_share` (`RC_i / sigma_p`) is a convenience
    normalization for UI display only — never the authoritative figure."""

    ticker: str
    sector: str
    weight: float
    marginal_risk_contribution: float
    component_risk_contribution: float
    risk_share: float


class SectorRiskRow(BaseModel):
    """Sector aggregate: sum of `AssetRiskRow.component_risk_contribution`
    across every asset in that sector — valid because component
    contributions are additive by construction (no separate sector
    covariance model is estimated)."""

    sector: str
    weight: float
    component_risk_contribution: float
    risk_share: float


class PortfolioRiskSummary(BaseModel):
    """`max_weight_constraint` is `None` for EQUAL_WEIGHT (a benchmark,
    never routed through MAX_WEIGHT) — same convention as
    `PortfolioResponse.max_weight_constraint`. `effective_holdings` =
    `1 / concentration_hhi` (the number of equally-weighted names that
    would produce the same HHI) — a direct, transparent function of HHI,
    not a new methodology."""

    predicted_volatility_21: float
    largest_weight: float
    active_holdings_count: int
    concentration_hhi: float
    effective_holdings: float
    max_weight_constraint: Optional[float] = None


class RiskResponse(BaseModel):
    """Formation-time (ex-ante) risk decomposition of an official Phase 7
    portfolio — never realized/ex-post figures (no `realized_return_21`,
    no SPXT, no turnover). `covariance_estimator` is DETERMINED by
    `strategy` for the four optimized strategies (Sample_MinVol/MaxSharpe
    -> "Sample", LW_MinVol/MaxSharpe -> "Ledoit-Wolf") and is a genuine
    user choice only for EQUAL_WEIGHT, which has no covariance identity
    of its own (see app/routes/risk.py's Equal Weight handling)."""

    formation_date: date
    strategy: StrategyInfo
    covariance_estimator: str  # "Sample" | "Ledoit-Wolf"
    forecast_horizon_sessions: int
    covariance_window_sessions: int
    portfolio: PortfolioRiskSummary
    assets: List[AssetRiskRow]
    sectors: List[SectorRiskRow]
    source: str


class BacktestPeriod(BaseModel):
    """One of the 46 non-overlapping, sequential 21-session evaluation
    periods (`app/routes/backtest.py`). `growth_of_one` is the compounded
    wealth level of $1 AT `target_date`, i.e. immediately after this
    period's return has been realized — `product(1+r_1..r_i)`, never a
    fabricated intra-period/daily value. `turnover`/`max_weight_observed`
    are `None` for the SPXT benchmark series (no portfolio weights exist
    for an index); `turnover` is additionally `None` for every strategy's
    own first formation period (genuinely undefined, never zero)."""

    formation_date: date
    target_date: date
    realized_return_21: float
    growth_of_one: float
    turnover: Optional[float] = None
    max_weight_observed: Optional[float] = None


class BacktestSummary(BaseModel):
    """Direct, unmodified output of the frozen Phase 7 aggregation
    (`optimizer.walkforward.aggregate_statistics` /
    `cumulative_compounded_return`) for portfolio strategies, and the
    identical arithmetic for SPXT (mean/std/median/min/max/hit-rate —
    matching `scripts/run_phase7b_official.py`'s `_aggregate_spxt_returns`,
    which deliberately excludes turnover/max-weight because SPXT is not a
    weighted portfolio). Never annualized, never a Sharpe-style
    risk-adjusted aggregate (BUILD_PLAN.md's Phase 7 report explicitly
    scopes this out)."""

    mean_return_21: float
    std_return_21: float
    median_return_21: float
    min_return_21: float
    max_return_21: float
    positive_period_rate: float
    cumulative_return: float
    mean_turnover: Optional[float] = None
    median_turnover: Optional[float] = None
    max_turnover: Optional[float] = None
    avg_max_weight: Optional[float] = None
    max_observed_weight: Optional[float] = None


class BacktestSeries(BaseModel):
    """One of the six displayed series: the five official Phase 7
    strategies (`kind="portfolio"`) plus the official SPXT benchmark
    (`kind="benchmark"`). Never labeled "best"/"winner"/"recommended" —
    `label`/`is_optimized`/`covariance_estimator` for portfolio series are
    read from the same `STRATEGY_REGISTRY` Portfolio Construction and Risk
    Analytics use, never a second identity scheme."""

    key: str
    label: str
    kind: str  # "portfolio" | "benchmark"
    is_optimized: Optional[bool] = None
    covariance_estimator: Optional[str] = None
    max_weight_constraint: Optional[float] = None
    periods: List[BacktestPeriod]
    summary: BacktestSummary


class BacktestExperiment(BaseModel):
    period_count: int
    first_formation_date: date
    last_formation_date: date
    horizon_sessions: int
    covariance_window_sessions: int
    benchmark: str
    data_through: Optional[date] = None


class BacktestResponse(BaseModel):
    """Frozen, official Phase 7 historical walk-forward evidence — never a
    new backtest, never a ranking, never a sealed-holdout (March 2026)
    result. `experiment` carries the shared calendar/horizon context;
    `series` carries all six displayed series."""

    experiment: BacktestExperiment
    series: List[BacktestSeries]
    source: str
