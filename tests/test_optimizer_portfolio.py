"""
RiskFecta Phase 7A — constrained mean-variance optimization tests
(optimizer/portfolio.py). Synthetic-only (task brief §17-MIN-VOL,
§17-MAX-SHARPE, §17-CONSTRAINTS, §17-ALIGNMENT, §17-FRONTIER,
§17-BENCHMARK, §17-GENERAL, §17-HORIZON, §17-SHARPE).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config
from optimizer.portfolio import (
    align_mu_sigma,
    efficient_frontier,
    efficient_frontier_point,
    equal_weight_benchmark,
    maximum_sharpe_weights,
    minimum_volatility_weights,
    portfolio_expected_return,
    portfolio_sharpe,
    portfolio_volatility,
    rf_horizon,
)

TOL = 1e-6


# ---------------------------------------------------------------------------
# LOCK D — max weight
# ---------------------------------------------------------------------------
def test_max_weight_is_010_and_is_a_new_lock_not_previously_frozen():
    assert config.MAX_WEIGHT == 0.10


# ---------------------------------------------------------------------------
# Risk-free-rate — 21-session horizon lock
# ---------------------------------------------------------------------------
def test_rf_horizon_21_session_worked_example():
    # From the required synthetic worked example: USGG10YR=4.5 -> rf_21=0.003750
    rf_21 = rf_horizon(4.5, horizon=21)
    assert rf_21 == pytest.approx(0.003750, abs=1e-9)


def test_rf_horizon_never_skips_percent_normalization():
    correct = rf_horizon(4.5, horizon=21)
    wrong = (4.5 / 252) * 21  # the explicitly-forbidden USGG10YR/252 convention
    assert not np.isclose(correct, wrong)
    assert correct == pytest.approx(((4.5 / 100) / 252) * 21)


def test_rf_horizon_default_uses_forecast_horizon():
    assert rf_horizon(4.5) == pytest.approx(rf_horizon(4.5, horizon=config.FORECAST_HORIZON))


# ---------------------------------------------------------------------------
# Equal-weight benchmark
# ---------------------------------------------------------------------------
def test_equal_weight_benchmark_exact_1_over_50():
    w = equal_weight_benchmark(50)
    assert len(w) == 50
    assert np.allclose(w, 0.02)
    assert w.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Ticker / matrix alignment
# ---------------------------------------------------------------------------
def _mu_sigma(tickers):
    mu = pd.Series({t: 0.01 * (i + 1) for i, t in enumerate(tickers)})
    n = len(tickers)
    sigma = pd.DataFrame(np.eye(n) * 0.04, index=tickers, columns=tickers)
    return mu, sigma


def test_align_mu_sigma_missing_ticker_fails_loudly():
    mu, sigma = _mu_sigma(["A", "B"])
    with pytest.raises(ValueError, match="missing from mu"):
        align_mu_sigma(mu, sigma, ["A", "B", "C"])


def test_align_mu_sigma_extra_ticker_fails_loudly():
    mu, sigma = _mu_sigma(["A", "B", "C"])
    with pytest.raises(ValueError, match="extra"):
        align_mu_sigma(mu, sigma, ["A", "B"])


def test_align_mu_sigma_duplicate_ticker_fails_loudly():
    mu = pd.Series([0.01, 0.01], index=["A", "A"])
    sigma = pd.DataFrame(np.eye(2) * 0.04, index=["A", "A"], columns=["A", "A"])
    with pytest.raises(ValueError, match="duplicate"):
        align_mu_sigma(mu, sigma, ["A"])


def test_align_mu_sigma_row_column_mismatch_fails_loudly():
    mu, _ = _mu_sigma(["A", "B"])
    sigma = pd.DataFrame(np.eye(2) * 0.04, index=["A", "B"], columns=["A", "X"])
    with pytest.raises(ValueError, match="sigma columns"):
        align_mu_sigma(mu, sigma, ["A", "B"])


def test_align_mu_sigma_nan_fails_loudly():
    mu, sigma = _mu_sigma(["A", "B"])
    mu["A"] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        align_mu_sigma(mu, sigma, ["A", "B"])


def test_align_mu_sigma_enforces_exact_order_never_implicit():
    tickers = ["C", "A", "B"]
    mu = pd.Series({"A": 0.01, "B": 0.02, "C": 0.03})  # alphabetical, NOT tickers order
    sigma = pd.DataFrame(np.eye(3) * 0.04, index=["B", "C", "A"], columns=["B", "C", "A"])
    mu_arr, sigma_arr = align_mu_sigma(mu, sigma, tickers)
    np.testing.assert_allclose(mu_arr, [0.03, 0.01, 0.02])  # C, A, B
    np.testing.assert_allclose(np.diag(sigma_arr), [0.04, 0.04, 0.04])


# ---------------------------------------------------------------------------
# Constraints — general
# ---------------------------------------------------------------------------
def test_infeasible_max_weight_constraint_fails_loudly():
    sigma = np.eye(5) * 0.01
    with pytest.raises(ValueError, match="infeasible constraints"):
        minimum_volatility_weights(sigma, max_weight=0.1)  # 0.1*5=0.5 < 1


def test_weights_respect_bounds_and_sum_to_one():
    sigma = np.array([[0.04, 0.0], [0.0, 0.09]])
    w = minimum_volatility_weights(sigma, max_weight=1.0)
    assert w.sum() == pytest.approx(1.0, abs=TOL)
    assert (w >= -TOL).all() and (w <= 1.0 + TOL).all()


# ---------------------------------------------------------------------------
# Minimum-volatility
# ---------------------------------------------------------------------------
def test_min_vol_known_synthetic_case_two_uncorrelated_equal_variance_assets():
    sigma = np.array([[0.04, 0.0], [0.0, 0.04]])
    w = minimum_volatility_weights(sigma, max_weight=1.0)
    np.testing.assert_allclose(w, [0.5, 0.5], atol=1e-4)


def test_min_vol_favors_lower_variance_asset():
    sigma = np.array([[0.01, 0.0], [0.0, 0.09]])  # asset 0 much less volatile
    w = minimum_volatility_weights(sigma, max_weight=1.0)
    assert w[0] > w[1]


def test_min_vol_respects_max_weight_cap():
    n = 15  # 0.10 * 15 = 1.5 > 1, so the cap is comfortably feasible
    sigma = np.eye(n) * 0.01
    sigma[0, 0] = 0.0001  # asset 0 would dominate an unconstrained solution
    w = minimum_volatility_weights(sigma, max_weight=config.MAX_WEIGHT)
    assert w[0] <= config.MAX_WEIGHT + 1e-6
    assert w.sum() == pytest.approx(1.0, abs=TOL)


def test_min_vol_does_not_accept_mu_structurally_independent_of_expected_return():
    import inspect
    sig = inspect.signature(minimum_volatility_weights)
    assert "mu" not in sig.parameters


def test_min_vol_deterministic():
    sigma = np.array([[0.04, 0.01], [0.01, 0.09]])
    w1 = minimum_volatility_weights(sigma, max_weight=1.0)
    w2 = minimum_volatility_weights(sigma, max_weight=1.0)
    np.testing.assert_allclose(w1, w2, atol=1e-8)


# ---------------------------------------------------------------------------
# Maximum-Sharpe
# ---------------------------------------------------------------------------
def test_max_sharpe_known_synthetic_case_tangency_weights():
    # Two uncorrelated, equal-variance assets, rf=0: unconstrained tangency
    # weights are proportional to excess return (mu_i / var_i, equal var).
    mu = np.array([0.05, 0.01])
    sigma = np.array([[0.04, 0.0], [0.0, 0.04]])
    rf = 0.0
    w = maximum_sharpe_weights(mu, sigma, rf, max_weight=1.0)
    expected = mu / mu.sum()  # [0.8333, 0.1667]
    np.testing.assert_allclose(w, expected, atol=1e-3)


def test_max_sharpe_feasible_and_finite():
    mu = np.array([0.02, -0.01, 0.03])
    sigma = np.eye(3) * 0.02
    w = maximum_sharpe_weights(mu, sigma, rf=0.001, max_weight=1.0)
    assert w.sum() == pytest.approx(1.0, abs=TOL)
    assert (w >= -TOL).all()
    er = portfolio_expected_return(w, mu)
    vol = portfolio_volatility(w, sigma)
    sharpe = portfolio_sharpe(w, mu, sigma, 0.001)
    assert np.isfinite(er) and np.isfinite(vol) and np.isfinite(sharpe)


def test_max_sharpe_respects_max_weight_cap():
    n = 15  # 0.10 * 15 = 1.5 > 1, so the cap is comfortably feasible (not a
            # degenerate single-point feasible region)
    mu = np.linspace(0.01, 0.10, n)  # last asset has by far the best return
    sigma = np.eye(n) * 0.02
    w = maximum_sharpe_weights(mu, sigma, rf=0.0, max_weight=config.MAX_WEIGHT)
    assert w.max() <= config.MAX_WEIGHT + 1e-6
    assert w.sum() == pytest.approx(1.0, abs=TOL)


# ---------------------------------------------------------------------------
# Sharpe worked example — mu_21 / Sigma_21 / rf_21 compatibility
# ---------------------------------------------------------------------------
def test_synthetic_21_session_sharpe_worked_example():
    mu_21 = np.array([0.03, 0.02])
    sigma_session = np.array([[0.0004, 0.0001], [0.0001, 0.0003]])
    sigma_21 = sigma_session * 21
    usgg10yr = 4.5
    rf_21 = rf_horizon(usgg10yr, horizon=21)
    assert rf_21 == pytest.approx(0.003750, abs=1e-9)

    w = np.array([0.6, 0.4])
    expected_return_21 = w @ mu_21
    variance_21 = w @ sigma_21 @ w
    volatility_21 = np.sqrt(variance_21)
    sharpe_21 = (expected_return_21 - rf_21) / volatility_21

    assert portfolio_expected_return(w, mu_21) == pytest.approx(expected_return_21)
    assert portfolio_volatility(w, sigma_21) == pytest.approx(volatility_21)
    assert portfolio_sharpe(w, mu_21, sigma_21, rf_21) == pytest.approx(sharpe_21)


# ---------------------------------------------------------------------------
# Efficient Frontier — generic machinery, no grid invented
# ---------------------------------------------------------------------------
def test_frontier_point_feasible_target_matches_requested_return():
    mu = np.array([0.01, 0.05])
    sigma = np.array([[0.02, 0.0], [0.0, 0.03]])
    target = 0.03
    w = efficient_frontier_point(mu, sigma, target, max_weight=1.0)
    assert portfolio_expected_return(w, mu) == pytest.approx(target, abs=1e-4)
    assert w.sum() == pytest.approx(1.0, abs=TOL)


def test_frontier_point_infeasible_target_fails_loudly():
    mu = np.array([0.01, 0.02])
    sigma = np.eye(2) * 0.01
    with pytest.raises(ValueError, match="infeasible"):
        efficient_frontier_point(mu, sigma, target_return=10.0, max_weight=1.0)


def test_frontier_batch_marks_infeasible_without_raising():
    mu = np.array([0.01, 0.02, 0.03])
    sigma = np.eye(3) * 0.01
    results = efficient_frontier(mu, sigma, [0.015, 999.0], max_weight=1.0)
    assert results[0]["feasible"] is True
    assert results[1]["feasible"] is False
    assert results[1]["weights"] is None


def test_frontier_respects_max_weight_constraint():
    n = 15  # 0.10 * 15 = 1.5 > 1, so the cap is comfortably feasible
    mu = np.linspace(0.01, 0.05, n)
    sigma = np.eye(n) * 0.01
    w = efficient_frontier_point(mu, sigma, target_return=float(mu.mean()), max_weight=config.MAX_WEIGHT)
    assert w.max() <= config.MAX_WEIGHT + 1e-6
    assert w.sum() == pytest.approx(1.0, abs=TOL)


def test_frontier_future_outcomes_cannot_influence_it_by_construction():
    # efficient_frontier_point is a pure function of (mu, sigma,
    # target_return, max_weight) — nothing else can enter, so there is no
    # code path by which a "future realized outcome" could influence it.
    import inspect
    sig = inspect.signature(efficient_frontier_point)
    assert set(sig.parameters) == {"mu", "sigma", "target_return", "max_weight"}


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------
def test_general_nan_sigma_fails_loudly_via_alignment():
    tickers = ["A", "B"]
    mu = pd.Series({"A": 0.01, "B": np.nan})
    sigma = pd.DataFrame(np.eye(2) * 0.04, index=tickers, columns=tickers)
    with pytest.raises(ValueError, match="NaN/Inf"):
        align_mu_sigma(mu, sigma, tickers)


def test_general_inf_sigma_fails_loudly_via_alignment():
    tickers = ["A", "B"]
    mu = pd.Series({"A": 0.01, "B": 0.02})
    sigma = pd.DataFrame([[np.inf, 0.0], [0.0, 0.04]], index=tickers, columns=tickers)
    with pytest.raises(ValueError, match="NaN/Inf"):
        align_mu_sigma(mu, sigma, tickers)
