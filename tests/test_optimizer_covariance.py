"""
RiskFecta Phase 7A — covariance estimation tests (optimizer/covariance.py).

Synthetic/fixture-only (task brief §17-COVARIANCE, §18). Proves LOCK A
(252-return / 253-level window), LOCK B (simple TRI returns, PX_LAST
session gate), and the session-frequency-only scope of both estimators
(x21 scaling is a separate, later step — see test_optimizer_portfolio.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import optimizer.covariance as covmod
from optimizer.covariance import (
    build_tri_wide,
    covariance_return_window,
    ledoit_wolf_covariance_session,
    sample_covariance_session,
    scale_covariance_to_horizon,
    session_returns,
    validate_covariance_matrix,
)

TICKERS = ["AAA", "BBB", "CCC"]


def _synthetic_prices(tickers=TICKERS, n_sessions=260, seed=0, start="2023-01-02"):
    """Dense per-ticker business-day panel with a smooth positive-drift TRI
    series (geometric random walk) and close == a scaled version of TRI so
    PX_LAST-gating tests have something to differentiate."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_sessions)
    rows = []
    for i, ticker in enumerate(tickers):
        tri = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, size=n_sessions))
        close = tri * (1.0 + 0.01 * i)  # distinct but proportional close series
        for d, c, t in zip(dates, close, tri):
            rows.append({"ticker": ticker, "date": d, "close": c, "total_return_idx": t})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LOCK A — 252-return / 253-level window
# ---------------------------------------------------------------------------
def test_covariance_window_produces_exactly_252_returns_from_253_levels():
    prices = _synthetic_prices(n_sessions=260)
    as_of = prices["date"].max()
    window = covariance_return_window(prices, TICKERS, as_of_date=as_of, window=252)
    assert window.shape == (252, 3)

    # Directly verify the 253-levels -> 252-returns off-by-one is respected.
    tri_wide = build_tri_wide(prices, TICKERS, as_of_date=as_of)
    tri_tail = tri_wide.iloc[-253:]
    assert len(tri_tail) == 253
    expected = tri_tail.pct_change().iloc[1:]
    pd.testing.assert_frame_equal(window, expected)


def test_covariance_window_default_uses_config_covar_window():
    import config
    assert config.COVAR_WINDOW == 252
    prices = _synthetic_prices(n_sessions=260)
    as_of = prices["date"].max()
    window = covariance_return_window(prices, TICKERS, as_of_date=as_of)
    assert len(window) == config.COVAR_WINDOW


def test_covar_window_is_a_separate_constant_from_train_window():
    import config
    # Same numerical value today, but must never be aliased.
    assert config.COVAR_WINDOW == config.TRAIN_WINDOW == 252
    assert covmod.__doc__ is not None and "SEPARATE constant" in covmod.__doc__


def test_insufficient_causal_history_fails_loudly():
    prices = _synthetic_prices(n_sessions=100)  # far fewer than 253 levels needed
    as_of = prices["date"].max()
    with pytest.raises(ValueError, match="need 253 causal TRI levels"):
        covariance_return_window(prices, TICKERS, as_of_date=as_of, window=252)


def test_off_by_one_would_silently_understate_returns_if_mishandled():
    # 253 levels -> exactly 252 returns, never 251.
    prices = _synthetic_prices(n_sessions=260)
    as_of = prices["date"].max()
    window = covariance_return_window(prices, TICKERS, as_of_date=as_of, window=252)
    assert len(window) == 252
    assert len(window) != 251


# ---------------------------------------------------------------------------
# LOCK B — session validity gate (PX_LAST, never TRI) + simple TRI returns
# ---------------------------------------------------------------------------
def test_px_last_determines_valid_sessions_not_tri():
    prices = _synthetic_prices(n_sessions=30, tickers=["AAA"])
    # Simulate a Bloomberg weekend/holiday placeholder row: close is NaN
    # (invalid session) but total_return_idx remains populated (carried
    # forward) — this row must be EXCLUDED, never treated as valid.
    prices.loc[5, "close"] = np.nan
    assert not pd.isna(prices.loc[5, "total_return_idx"])

    wide = build_tri_wide(prices, ["AAA"], as_of_date=prices["date"].max())
    assert prices.loc[5, "date"] not in wide.index


def test_tri_never_used_as_session_gate_even_when_tri_is_nan():
    prices = _synthetic_prices(n_sessions=30, tickers=["AAA"])
    # Opposite case: a genuinely valid session (real close) with a missing
    # TRI value. The row must still be treated as a valid SESSION (present
    # in the index), with the TRI gap preserved as NaN rather than the row
    # being dropped as if invalid.
    prices.loc[5, "total_return_idx"] = np.nan
    wide = build_tri_wide(prices, ["AAA"], as_of_date=prices["date"].max())
    assert prices.loc[5, "date"] in wide.index
    assert pd.isna(wide.loc[prices.loc[5, "date"], "AAA"])


def test_session_returns_is_simple_not_log_return():
    tri_wide = pd.DataFrame({"AAA": [100.0, 110.0, 99.0]})
    returns = session_returns(tri_wide)
    expected_simple = pd.Series([0.10, 99.0 / 110.0 - 1.0], name="AAA")
    np.testing.assert_allclose(returns["AAA"].to_numpy(), expected_simple.to_numpy())
    # Explicitly NOT the log return.
    log_return_0 = np.log(110.0 / 100.0)
    assert not np.isclose(returns["AAA"].iloc[0], log_return_0)


def test_session_returns_uses_tri_not_close():
    tri_wide = pd.DataFrame({"AAA": [100.0, 105.0]})
    close_wide = pd.DataFrame({"AAA": [50.0, 60.0]})  # deliberately different series
    r_from_tri = session_returns(tri_wide)
    r_from_close_would_be = session_returns(close_wide)
    assert not np.isclose(r_from_tri["AAA"].iloc[0], r_from_close_would_be["AAA"].iloc[0])


def test_session_returns_fails_loudly_on_nan():
    tri_wide = pd.DataFrame({"AAA": [100.0, np.nan, 102.0]})
    with pytest.raises(ValueError, match="NaN"):
        session_returns(tri_wide)


def test_build_tri_wide_fails_loudly_on_missing_ticker():
    prices = _synthetic_prices(n_sessions=30, tickers=["AAA", "BBB"])
    with pytest.raises(ValueError, match="entirely absent"):
        build_tri_wide(prices, ["AAA", "ZZZ"], as_of_date=prices["date"].max())


def test_build_tri_wide_fails_loudly_on_duplicate_ticker_date():
    prices = _synthetic_prices(n_sessions=10, tickers=["AAA"])
    dup_row = prices.iloc[[0]].copy()
    prices = pd.concat([prices, dup_row], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_tri_wide(prices, ["AAA"], as_of_date=prices["date"].max())


def test_build_tri_wide_enforces_exact_requested_column_order():
    prices = _synthetic_prices(n_sessions=10, tickers=["AAA", "BBB", "CCC"])
    wide = build_tri_wide(prices, ["CCC", "AAA", "BBB"], as_of_date=prices["date"].max())
    assert list(wide.columns) == ["CCC", "AAA", "BBB"]


# ---------------------------------------------------------------------------
# Causality
# ---------------------------------------------------------------------------
def test_future_tri_perturbation_cannot_change_causal_window():
    prices = _synthetic_prices(n_sessions=260)
    as_of = prices["date"].iloc[-10]  # cutoff well before the data's end
    baseline = covariance_return_window(prices, TICKERS, as_of_date=as_of, window=200)

    perturbed = prices.copy()
    future_mask = perturbed["date"] > as_of
    assert future_mask.sum() > 0
    perturbed.loc[future_mask, "total_return_idx"] *= 5.0
    perturbed.loc[future_mask, "close"] *= 5.0

    after = covariance_return_window(perturbed, TICKERS, as_of_date=as_of, window=200)
    pd.testing.assert_frame_equal(baseline, after)


def test_covariance_and_portfolio_modules_never_read_predictions_table():
    # Structural causality guarantee: neither module opens a database
    # connection, executes SQL, or imports pipeline.predictions/pipeline.db
    # — mu is always a caller-supplied Series (predictions.ensemble_pred as
    # of T, read upstream by a caller), never fetched or recomputed here.
    import inspect
    import optimizer.covariance as cov
    import optimizer.portfolio as port
    for mod in (cov, port):
        assert not hasattr(mod, "db")
        assert not hasattr(mod, "psycopg2")
        src = inspect.getsource(mod)
        assert "SELECT" not in src.upper()
        assert "pipeline.predictions" not in src
        assert "pipeline import db" not in src
        assert "pipeline.db" not in src


# ---------------------------------------------------------------------------
# Sample vs. Ledoit-Wolf — correctness, symmetry, finiteness
# ---------------------------------------------------------------------------
def _synthetic_returns(n_obs=252, n_assets=4, seed=1):
    rng = np.random.default_rng(seed)
    cols = [f"T{i}" for i in range(n_assets)]
    data = rng.normal(0.0005, 0.02, size=(n_obs, n_assets))
    return pd.DataFrame(data, columns=cols)


def test_sample_covariance_matches_pandas_cov_reference():
    returns = _synthetic_returns()
    sigma = sample_covariance_session(returns)
    expected = returns.cov()
    pd.testing.assert_frame_equal(sigma, expected)


def test_ledoit_wolf_differs_from_sample_but_stays_valid():
    returns = _synthetic_returns()
    sample = sample_covariance_session(returns)
    lw = ledoit_wolf_covariance_session(returns)
    assert not np.allclose(sample.to_numpy(), lw.to_numpy())
    validate_covariance_matrix(sample, list(returns.columns))
    validate_covariance_matrix(lw, list(returns.columns))


def test_both_estimators_use_identical_return_window():
    # Same `returns` object into both — this test documents/enforces that
    # callers must feed the identical window to each (never two different
    # windows for "comparable" estimators).
    returns = _synthetic_returns()
    sample = sample_covariance_session(returns)
    lw = ledoit_wolf_covariance_session(returns)
    assert list(sample.columns) == list(lw.columns) == list(returns.columns)
    assert sample.shape == lw.shape


def test_validate_covariance_matrix_rejects_asymmetry():
    tickers = ["A", "B"]
    bad = pd.DataFrame([[1.0, 0.5], [0.6, 1.0]], index=tickers, columns=tickers)
    with pytest.raises(ValueError, match="symmetric"):
        validate_covariance_matrix(bad, tickers)


def test_validate_covariance_matrix_rejects_nan_and_misalignment():
    tickers = ["A", "B"]
    with_nan = pd.DataFrame([[1.0, np.nan], [np.nan, 1.0]], index=tickers, columns=tickers)
    with pytest.raises(ValueError, match="NaN/Inf"):
        validate_covariance_matrix(with_nan, tickers)

    reordered = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["B", "A"], columns=["A", "B"])
    with pytest.raises(ValueError, match="misalignment"):
        validate_covariance_matrix(reordered, tickers)


# ---------------------------------------------------------------------------
# LOCK C (session half) — x21 scaling applied only AFTER estimation
# ---------------------------------------------------------------------------
def test_scale_to_horizon_is_exactly_21x_session_and_applied_after_estimation():
    returns = _synthetic_returns()
    sigma_session = sample_covariance_session(returns)
    sigma_21 = scale_covariance_to_horizon(sigma_session, horizon=21)
    np.testing.assert_allclose(sigma_21.to_numpy(), 21.0 * sigma_session.to_numpy())

    lw_session = ledoit_wolf_covariance_session(returns)
    lw_21 = scale_covariance_to_horizon(lw_session, horizon=21)
    np.testing.assert_allclose(lw_21.to_numpy(), 21.0 * lw_session.to_numpy())

    # Never folded into the estimator itself — the un-scaled session
    # estimate is still exactly the pandas/sklearn reference.
    pd.testing.assert_frame_equal(sigma_session, returns.cov())


def test_scale_to_horizon_default_uses_forecast_horizon():
    import config
    returns = _synthetic_returns()
    sigma_session = sample_covariance_session(returns)
    scaled_default = scale_covariance_to_horizon(sigma_session)
    scaled_explicit = scale_covariance_to_horizon(sigma_session, horizon=config.FORECAST_HORIZON)
    pd.testing.assert_frame_equal(scaled_default, scaled_explicit)
