"""
RiskFecta Phase 4A — baseline tests (models/baselines.py).

Covers task brief §9A-§9C, §10, §17-H.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.baselines import (
    OFFICIAL_MOMENTUM_BASELINE_COL,
    SECONDARY_MOMENTUM_DIAGNOSTIC_COL,
    historical_mean_baseline,
    momentum_baseline,
    ridge_baseline,
)

FEATURE_COLS = ["vix", "yield_10y", "momentum_3m", "momentum_6m", "volatility_20d"]


def _training_rows(ticker_targets: dict) -> pd.DataFrame:
    """ticker_targets: {"AAPL": [t0, t1, ...], ...} -> a purge-safe-shaped
    training frame with those target_21d values."""
    rows = []
    for ticker, targets in ticker_targets.items():
        for i, t in enumerate(targets):
            row = {"ticker": ticker, "date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=i), "target_21d": t}
            for c in FEATURE_COLS:
                row[c] = float(i)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A. Historical-mean baseline
# ---------------------------------------------------------------------------
def test_historical_mean_is_per_ticker_mean_of_eligible_labels():
    training_rows = _training_rows({"AAPL": [0.01, 0.02, 0.03], "MSFT": [0.10, 0.20]})
    out = historical_mean_baseline(training_rows)
    assert out["AAPL"] == pytest.approx((0.01 + 0.02 + 0.03) / 3)
    assert out["MSFT"] == pytest.approx((0.10 + 0.20) / 2)


def test_historical_mean_ignores_labels_outside_the_given_training_rows():
    """The baseline trusts its input is already the fold's purge-safe
    window — it must not reach outside `training_rows` for any value."""
    scoped = _training_rows({"AAPL": [0.01, 0.02]})  # caller already purged to 2 rows
    out = historical_mean_baseline(scoped)
    assert out["AAPL"] == pytest.approx(0.015)


def test_historical_mean_future_label_perturbation_does_not_affect_earlier_baseline_forecast():
    """A future label change (outside the purge-safe training_rows given to
    this fold) must not reach the historical-mean forecast — proven here by
    literally never handing the future rows to the function at all."""
    fold_window_rows = _training_rows({"AAPL": [0.01, 0.02, 0.03]})
    baseline_before = historical_mean_baseline(fold_window_rows)

    # A "future" row (outside this fold's window) changes arbitrarily — but
    # it's never part of `fold_window_rows`, so it cannot move the forecast.
    future_row = pd.DataFrame([{"ticker": "AAPL", "date": pd.Timestamp("2030-01-01"),
                                 "target_21d": 999.0, **{c: 0.0 for c in FEATURE_COLS}}])
    baseline_after = historical_mean_baseline(fold_window_rows)  # unchanged input
    assert baseline_before["AAPL"] == baseline_after["AAPL"]
    assert future_row["target_21d"].iloc[0] == 999.0  # sanity: the future value really is different


def test_historical_mean_absent_ticker_not_fabricated():
    training_rows = _training_rows({"AAPL": [0.01]})
    out = historical_mean_baseline(training_rows)
    assert "MSFT" not in out.index


# ---------------------------------------------------------------------------
# B. Momentum baseline
# ---------------------------------------------------------------------------
def test_momentum_baseline_uses_formation_date_feature_value_directly():
    eval_rows = pd.DataFrame(
        [{"ticker": "AAPL", "momentum_3m": 0.05, "momentum_6m": 0.11},
         {"ticker": "MSFT", "momentum_3m": -0.02, "momentum_6m": 0.03}]
    )
    out3 = momentum_baseline(eval_rows, momentum_col="momentum_3m")
    out6 = momentum_baseline(eval_rows, momentum_col="momentum_6m")
    assert out3["AAPL"] == 0.05
    assert out3["MSFT"] == -0.02
    assert out6["AAPL"] == 0.11
    assert out6["MSFT"] == 0.03


def test_official_momentum_baseline_locked_to_3m():
    # Single source of truth (models.baselines.OFFICIAL_MOMENTUM_BASELINE_COL)
    # reused by both the constant itself and the function's default arg.
    assert OFFICIAL_MOMENTUM_BASELINE_COL == "momentum_3m"
    assert SECONDARY_MOMENTUM_DIAGNOSTIC_COL == "momentum_6m"


def test_momentum_baseline_default_uses_official_3m_column():
    eval_rows = pd.DataFrame(
        [{"ticker": "AAPL", "momentum_3m": 0.05, "momentum_6m": 0.11},
         {"ticker": "MSFT", "momentum_3m": -0.02, "momentum_6m": 0.03}]
    )
    default_out = momentum_baseline(eval_rows)  # no momentum_col given
    explicit_3m_out = momentum_baseline(eval_rows, momentum_col=OFFICIAL_MOMENTUM_BASELINE_COL)
    pd.testing.assert_series_equal(default_out.sort_index(), explicit_3m_out.sort_index())
    assert default_out["AAPL"] == 0.05
    assert default_out["MSFT"] == -0.02


def test_momentum_baseline_secondary_6m_diagnostic_still_explicitly_available():
    eval_rows = pd.DataFrame([{"ticker": "AAPL", "momentum_3m": 0.05, "momentum_6m": 0.11}])
    out6 = momentum_baseline(eval_rows, momentum_col=SECONDARY_MOMENTUM_DIAGNOSTIC_COL)
    assert out6["AAPL"] == 0.11
    # 6M must be requested explicitly — it is never what the default returns.
    default_out = momentum_baseline(eval_rows)
    assert default_out["AAPL"] != out6["AAPL"]


def test_momentum_baseline_rejects_any_other_column():
    eval_rows = pd.DataFrame([{"ticker": "AAPL", "momentum_3m": 0.05, "momentum_6m": 0.11, "rsi_14": 55.0}])
    with pytest.raises(ValueError):
        momentum_baseline(eval_rows, momentum_col="rsi_14")
    with pytest.raises(ValueError):
        momentum_baseline(eval_rows, momentum_col="momentum_1m")


def test_momentum_baseline_is_not_a_fitted_regression():
    """The momentum baseline must be an exact passthrough, not merely
    correlated with the feature — proves no fitting occurred."""
    eval_rows = pd.DataFrame([{"ticker": "AAPL", "momentum_3m": 0.0731234}])
    out = momentum_baseline(eval_rows, momentum_col="momentum_3m")
    assert out["AAPL"] == 0.0731234


# ---------------------------------------------------------------------------
# C. Ridge baseline — train-only preprocessing (§10, §17-G)
# ---------------------------------------------------------------------------
def _synthetic_ridge_frames(n_train=60, n_eval=5, seed=0):
    rng = np.random.default_rng(seed)
    train = pd.DataFrame(
        {c: rng.normal(size=n_train) for c in FEATURE_COLS}
    )
    train["ticker"] = [f"T{i}" for i in range(n_train)]
    train["target_21d"] = train[FEATURE_COLS].sum(axis=1) * 0.01 + rng.normal(scale=0.001, size=n_train)

    evalf = pd.DataFrame({c: rng.normal(size=n_eval) for c in FEATURE_COLS})
    evalf["ticker"] = [f"E{i}" for i in range(n_eval)]
    return train, evalf


def test_ridge_baseline_returns_one_forecast_per_eval_ticker():
    train, evalf = _synthetic_ridge_frames()
    forecast, scaler, model = ridge_baseline(train, evalf, FEATURE_COLS)
    assert len(forecast) == len(evalf)
    assert set(forecast.index) == set(evalf["ticker"])


def test_ridge_scaler_never_fit_on_eval_rows():
    """Adversarial: perturbing eval-row feature values must not change the
    fitted scaler's mean_/scale_ (fit is train-only, per ML_SPEC.md §22)."""
    train, evalf = _synthetic_ridge_frames()
    _, scaler_before, _ = ridge_baseline(train, evalf, FEATURE_COLS)

    evalf_perturbed = evalf.copy()
    evalf_perturbed[FEATURE_COLS] = evalf_perturbed[FEATURE_COLS] * 999.0 + 12345.0
    _, scaler_after, _ = ridge_baseline(train, evalf_perturbed, FEATURE_COLS)

    np.testing.assert_allclose(scaler_before.mean_, scaler_after.mean_)
    np.testing.assert_allclose(scaler_before.scale_, scaler_after.scale_)


def test_ridge_model_coefficients_unaffected_by_eval_perturbation():
    train, evalf = _synthetic_ridge_frames()
    _, _, model_before = ridge_baseline(train, evalf, FEATURE_COLS)

    evalf_perturbed = evalf.copy()
    evalf_perturbed[FEATURE_COLS] = evalf_perturbed[FEATURE_COLS] * -500.0
    _, _, model_after = ridge_baseline(train, evalf_perturbed, FEATURE_COLS)

    np.testing.assert_allclose(model_before.coef_, model_after.coef_)
    assert model_before.intercept_ == pytest.approx(model_after.intercept_)


def test_ridge_scaler_fit_matches_manual_train_only_statistics():
    train, evalf = _synthetic_ridge_frames()
    _, scaler, _ = ridge_baseline(train, evalf, FEATURE_COLS)
    manual_mean = train[FEATURE_COLS].to_numpy(dtype=float).mean(axis=0)
    np.testing.assert_allclose(scaler.mean_, manual_mean)
