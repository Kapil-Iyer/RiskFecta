"""GET /api/models/comparison — frozen Phase 4-6 historical walk-forward
model comparison (ML_SPEC.md §15, §19-20; BUILD_PLAN.md Phase 4-6).

Two provenance categories, both read-only, both labeled explicitly in the
response (`ModelMetricRow.source`):

- XGBoost / LSTM / Ensemble: recomputed on every request from the real
  `predictions` table using the exact frozen evaluation code
  (models.metrics.compute_all_metrics via models.ensemble.compare_models)
  — never retrained, never re-forecast, just re-scored against the
  already-frozen (prediction, realized) pairs. Independently verified to
  reproduce the authoritative Phase 6 results exactly before this route
  was written (see BUILD_PLAN.md / the Phase 8B implementation report).

- Historical Mean / Momentum 3M / Ridge: baseline forecasts are never
  persisted (pipeline/predictions.py) — reproducing them here would mean
  re-running the Phase 4A baseline experiment, which this route does not
  do. They are exposed from `models.frozen_phase4_baselines`, a frozen
  research-result artifact, not a live computation.

This router never trains, tunes, retries a forecast, or writes anything.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends

from app.db import get_db
from app.schemas import (
    MetricDefinition,
    ModelComparisonResponse,
    ModelDisagreementSummary,
    ModelMetricRow,
)
from models.ensemble import (
    build_ensemble_frame,
    compare_models,
    directional_agreement,
    load_aligned_predictions,
    prediction_correlation,
    residual_correlation,
)
from models.frozen_phase4_baselines import FROZEN_BASELINE_RESULTS

router = APIRouter(prefix="/api", tags=["models"])

_LABELS = {
    "historical_mean": "Historical Mean",
    "momentum_3m": "Momentum 3M",
    "ridge": "Ridge",
    "xgb_pred": "XGBoost",
    "lstm_pred": "LSTM",
    "ensemble_pred": "Ensemble",
}

_METRIC_DEFINITIONS = [
    MetricDefinition(
        key="mae",
        label="MAE",
        direction="lower_is_better",
        description="Mean absolute error between the predicted and realized 21-session forward return.",
    ),
    MetricDefinition(
        key="rmse",
        label="RMSE",
        direction="lower_is_better",
        description="Root mean squared error — penalizes large misses more heavily than MAE.",
    ),
    MetricDefinition(
        key="directional_accuracy",
        label="Directional Accuracy",
        direction="higher_is_better",
        description=(
            "Fraction of predictions with the same sign as the realized return "
            "(observations with a zero realized return are excluded)."
        ),
    ),
    MetricDefinition(
        key="pearson_corr",
        label="Pearson Correlation",
        direction="context_dependent",
        description=(
            "Linear correlation between predicted and realized returns. Values here are small — "
            "a small positive number reflects weak, not strong, linear predictive signal."
        ),
    ),
    MetricDefinition(
        key="spearman_corr",
        label="Spearman Correlation",
        direction="context_dependent",
        description=(
            "Rank correlation between predicted and realized returns — how well the model orders "
            "stocks by expected return, independent of scale. Values here are small."
        ),
    ),
]


@router.get("/models/comparison", response_model=ModelComparisonResponse)
def get_model_comparison(conn=Depends(get_db)) -> ModelComparisonResponse:
    df = load_aligned_predictions(conn)
    ensemble_df = build_ensemble_frame(df)
    live_metrics = compare_models(ensemble_df)  # indexed by xgb_pred / lstm_pred / ensemble_pred

    rows: list[ModelMetricRow] = [
        ModelMetricRow(
            model=b.model,
            label=_LABELS[b.model],
            source="frozen_baseline_constant",
            mae=b.mae,
            rmse=b.rmse,
            directional_accuracy=b.directional_accuracy,
            pearson_corr=b.pearson_corr,
            spearman_corr=b.spearman_corr,
            n_obs=b.n_obs,
        )
        for b in FROZEN_BASELINE_RESULTS
    ]
    for col in ("xgb_pred", "lstm_pred", "ensemble_pred"):
        m = live_metrics.loc[col]
        rows.append(
            ModelMetricRow(
                model=col,
                label=_LABELS[col],
                source="computed_from_persisted_predictions",
                mae=float(m["mae"]),
                rmse=float(m["rmse"]),
                directional_accuracy=float(m["directional_accuracy"]),
                pearson_corr=float(m["pearson_corr"]),
                spearman_corr=float(m["spearman_corr"]),
                n_obs=int(m["n_obs"]),
            )
        )

    corr = prediction_correlation(ensemble_df["xgb_pred"], ensemble_df["lstm_pred"])
    resid = residual_correlation(ensemble_df["xgb_pred"], ensemble_df["lstm_pred"], ensemble_df["actual_return"])
    agree = directional_agreement(ensemble_df["xgb_pred"], ensemble_df["lstm_pred"])

    disagreement = ModelDisagreementSummary(
        source="computed_from_persisted_predictions",
        xgb_lstm_pred_pearson=corr.pearson,
        xgb_lstm_pred_spearman=corr.spearman,
        residual_pearson=resid.pearson,
        residual_spearman=resid.spearman,
        n_disagree=agree.n_disagree,
        n_total=agree.n_total,
    )

    return ModelComparisonResponse(
        experiment_type="historical_walk_forward_oos",
        target_horizon_sessions=21,
        fold_count=int(df["forecast_date"].nunique()),
        prediction_count=int(len(df)),
        formation_date_start=df["forecast_date"].min().date(),
        formation_date_end=df["forecast_date"].max().date(),
        models=rows,
        metric_definitions=_METRIC_DEFINITIONS,
        disagreement=disagreement,
    )
