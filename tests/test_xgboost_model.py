"""
RiskFecta Phase 4A — pooled XGBoost wrapper tests (models/xgboost_model.py).

SYNTHETIC DATA ONLY (task brief §18, §25 hard gate) — never fits against
real Bloomberg-sourced features/prices_raw. Covers task brief §17-I, §7.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from models.xgboost_model import (
    DEFAULT_SEED,
    XGBOOST_FEATURE_COLS,
    predict_xgboost,
    train_xgboost,
)

LOCKED_FEATURES = ["vix", "yield_10y", "momentum_3m", "momentum_6m", "volatility_20d"]


def _synthetic_training_rows(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in LOCKED_FEATURES})
    df["ticker"] = [f"T{i % 10}" for i in range(n)]  # pooled across 10 synthetic tickers
    df["target_21d"] = df[LOCKED_FEATURES].sum(axis=1) * 0.01 + rng.normal(scale=0.001, size=n)
    return df


def _synthetic_eval_rows(n=10, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(size=n) for c in LOCKED_FEATURES})
    df["ticker"] = [f"T{i}" for i in range(n)]
    return df


# ---------------------------------------------------------------------------
# I. XGBOOST FEATURE SURFACE — exactly the locked five columns
# ---------------------------------------------------------------------------
def test_feature_cols_exactly_match_locked_config_list():
    assert XGBOOST_FEATURE_COLS == config.XGBOOST_FEATURE_COLS
    assert XGBOOST_FEATURE_COLS == ["vix", "yield_10y", "momentum_3m", "momentum_6m", "volatility_20d"]
    assert len(XGBOOST_FEATURE_COLS) == 5


def test_no_static_or_ticker_id_or_indicator_columns_in_locked_list():
    forbidden = {"rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower",
                 "beta", "mkt_cap_log", "sector", "div_yield", "ticker", "ticker_id"}
    assert forbidden.isdisjoint(XGBOOST_FEATURE_COLS)


def test_train_xgboost_rejects_forbidden_feature_override():
    train = _synthetic_training_rows()
    with pytest.raises(ValueError):
        train_xgboost(train, feature_cols=LOCKED_FEATURES + ["sector"])


def test_predict_xgboost_rejects_forbidden_feature_override():
    train = _synthetic_training_rows()
    model = train_xgboost(train)
    eval_rows = _synthetic_eval_rows()
    with pytest.raises(ValueError):
        predict_xgboost(model, eval_rows, feature_cols=LOCKED_FEATURES + ["beta"])


# ---------------------------------------------------------------------------
# Plumbing: pooled fit/predict roundtrip, reproducibility
# ---------------------------------------------------------------------------
def test_train_and_predict_roundtrip_shapes():
    train = _synthetic_training_rows()
    eval_rows = _synthetic_eval_rows()
    model = train_xgboost(train)
    preds = predict_xgboost(model, eval_rows)
    assert len(preds) == len(eval_rows)
    assert set(preds.index) == set(eval_rows["ticker"])
    assert preds.notna().all()


def test_training_is_reproducible_given_fixed_seed():
    train = _synthetic_training_rows()
    eval_rows = _synthetic_eval_rows()
    model_a = train_xgboost(train, seed=DEFAULT_SEED)
    model_b = train_xgboost(train, seed=DEFAULT_SEED)
    preds_a = predict_xgboost(model_a, eval_rows)
    preds_b = predict_xgboost(model_b, eval_rows)
    np.testing.assert_allclose(preds_a.values, preds_b.values)


def test_one_pooled_model_not_per_ticker_models():
    """A single fitted model applied identically to any ticker's feature
    vector — two DIFFERENT tickers with the IDENTICAL feature vector must
    get the IDENTICAL prediction (proves no per-ticker branching / no
    ticker-identity leakage into the fitted model)."""
    train = _synthetic_training_rows()
    model = train_xgboost(train)

    identical_features = {c: 0.3 for c in LOCKED_FEATURES}
    eval_rows = pd.DataFrame(
        [{"ticker": "AAPL", **identical_features}, {"ticker": "ZZZZ", **identical_features}]
    )
    preds = predict_xgboost(model, eval_rows)
    assert preds["AAPL"] == pytest.approx(preds["ZZZZ"])


def test_training_uses_pooled_rows_from_multiple_tickers():
    train = _synthetic_training_rows()
    assert train["ticker"].nunique() > 1
    model = train_xgboost(train)  # must not raise / must not require single-ticker input
    assert model is not None
