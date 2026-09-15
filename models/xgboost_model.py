"""
RiskFecta Phase 4A — pooled XGBoost regression (ML_SPEC.md §9, §16).

Training/prediction plumbing only. No fold construction lives here (that's
`pipeline.folds`, independently testable without touching this module) and
no metrics computation lives here (`models.metrics`) — kept separate per
task brief §11 ("keep training code separate from fold construction and
metrics").

HARD GATE (task brief §18, §25): nothing in this module is called against
the real 62,800-row Bloomberg `features`/`prices_raw` data during Phase 4A.
Tests exercise it only against small synthetic fixtures to prove the
wrapper's plumbing (feature surface, pooling, reproducibility) is correct.
"""
from __future__ import annotations

from typing import Optional, Sequence

import pandas as pd
import xgboost as xgb

import config
from pipeline.sessions import TICKER_COL
from pipeline.targets import TARGET_COL

# ---------------------------------------------------------------------------
# Locked feature surface (task brief §7) — re-exported from config.py, the
# single source of truth, never independently redefined here.
# ---------------------------------------------------------------------------
XGBOOST_FEATURE_COLS = list(config.XGBOOST_FEATURE_COLS)

# ---------------------------------------------------------------------------
# Fixed Phase 4A hyperparameters (task brief §12) — a small, documented,
# non-tuned configuration. Never selected/adjusted using real OOS results;
# any future tuning must go through
# pipeline.folds.internal_validation_split + rows_in_session_range only.
# ---------------------------------------------------------------------------
DEFAULT_XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="reg:squarederror",
    n_jobs=-1,
)
DEFAULT_SEED = 42


def train_xgboost(
    training_rows: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
    target_col: str = TARGET_COL,
    params: Optional[dict] = None,
    seed: int = DEFAULT_SEED,
) -> xgb.XGBRegressor:
    """Fit one pooled XGBRegressor on `training_rows` (already purge-safe —
    see pipeline.folds.training_rows_for_fold). Pooled by construction:
    `training_rows` is expected to already contain observations from every
    ticker present in that fold's window, and this function adds no
    per-ticker branching, no ticker-identity feature, and no per-ticker
    model — it fits exactly one model over exactly `feature_cols` (task
    brief §6, §7).
    """
    feature_cols = list(feature_cols) if feature_cols is not None else XGBOOST_FEATURE_COLS
    _assert_locked_feature_surface(feature_cols)

    merged_params = {**DEFAULT_XGB_PARAMS, **(params or {})}
    merged_params["random_state"] = seed

    X = training_rows[feature_cols].to_numpy(dtype=float)
    y = training_rows[target_col].to_numpy(dtype=float)

    model = xgb.XGBRegressor(**merged_params)
    model.fit(X, y)
    return model


def predict_xgboost(
    model: xgb.XGBRegressor,
    eval_rows: pd.DataFrame,
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Predict for `eval_rows` (already purge-safe formation-date rows —
    see pipeline.folds.eval_rows_for_fold). Returns a Series indexed by
    ticker; caller is responsible for attaching forecast_date/target_date
    identity (models/xgboost_model.py never drops ticker/date identity —
    task brief §6 "predictions retain ticker/date identity" — the caller
    supplies eval_rows already keyed by ticker for exactly this reason).
    """
    feature_cols = list(feature_cols) if feature_cols is not None else XGBOOST_FEATURE_COLS
    _assert_locked_feature_surface(feature_cols)

    X = eval_rows[feature_cols].to_numpy(dtype=float)
    preds = model.predict(X)
    return pd.Series(preds, index=eval_rows[TICKER_COL].to_numpy())


def _assert_locked_feature_surface(feature_cols: Sequence[str]) -> None:
    """Defense in depth: every call site must use exactly the locked five
    columns (task brief §7) unless a caller deliberately overrides
    `feature_cols` for a test — never a ticker-ID, static field, or
    RSI/MACD/Bollinger column silently smuggled in via a default.
    """
    forbidden = {"ticker_id", "ticker", "beta", "mkt_cap_log", "sector", "div_yield",
                 "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower"}
    leaked = forbidden.intersection(feature_cols)
    if leaked:
        raise ValueError(f"train/predict_xgboost: forbidden feature(s) present: {sorted(leaked)}")
