"""
RiskFecta — frozen Phase 4A historical baseline results (Historical Mean,
Momentum 3M, Ridge).

These values are NOT computed live and are NOT derived from the database.
Baseline forecasts are deliberately never persisted: `pipeline/predictions.py`'s
module docstring states plainly that "baseline forecasts are evaluation-only...
never passed to `build_predictions_rows` or persisted" — only
`xgb_pred`/`lstm_pred`/`ensemble_pred` have columns in the frozen `predictions`
schema (`schema.sql`). There is therefore no read-only query that can
reproduce these numbers; recovering them would mean re-running the Phase 4A
historical walk-forward baseline experiment, which this artifact deliberately
avoids doing on every API request (or at all, without separate explicit
approval).

The values below are the authoritative, already-completed Phase 4A
historical walk-forward out-of-sample evaluation results — the same 47
formation dates / 2,350-observation experiment window as the persisted
XGBoost/LSTM/Ensemble metrics (`models/ensemble.py::compare_models`),
computed with `models.baselines.historical_mean_baseline` /
`momentum_baseline` (using `OFFICIAL_MOMENTUM_BASELINE_COL = "momentum_3m"`) /
`ridge_baseline`, scored with `models.metrics.compute_all_metrics` — the
identical evaluation code used for every other model in this comparison,
never a separately invented metric definition.

This is the historical-evaluation equivalent of `BUILD_PLAN.md`'s frozen
Phase 7 results table: a one-time, already-computed research result written
down once rather than re-derived by every reader. Every number here is
independently regression-tested (`tests/test_frozen_phase4_baselines.py`)
against these exact literals — changing any value requires a deliberate,
reviewed code edit, never a live recomputation and never a silent drift.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrozenBaselineMetrics:
    model: str
    mae: float
    rmse: float
    directional_accuracy: float
    pearson_corr: float
    spearman_corr: float
    n_obs: int


# Source: Phase 4A historical walk-forward OOS evaluation — 47 formation
# dates (2022-02-25 -> 2026-01-02), 50 tickers/date, 2,350 observations —
# the identical experiment window independently confirmed for
# XGBoost/LSTM/Ensemble by recomputing from the persisted `predictions`
# table (see app/routes/models.py).
FROZEN_BASELINE_RESULTS: tuple[FrozenBaselineMetrics, ...] = (
    FrozenBaselineMetrics(
        model="historical_mean",
        mae=0.07312,
        rmse=0.09722,
        directional_accuracy=0.5366,
        pearson_corr=0.02854,
        spearman_corr=0.01673,
        n_obs=2350,
    ),
    FrozenBaselineMetrics(
        model="momentum_3m",
        mae=0.14254,
        rmse=0.18803,
        directional_accuracy=0.5077,
        pearson_corr=-0.00132,
        spearman_corr=-0.03109,
        n_obs=2350,
    ),
    FrozenBaselineMetrics(
        model="ridge",
        mae=0.07831,
        rmse=0.10421,
        directional_accuracy=0.4761,
        pearson_corr=0.00158,
        spearman_corr=-0.03153,
        n_obs=2350,
    ),
)
