"""
RiskFecta Phase 7 — SLSQP memory-layout robustness fix tests
(optimizer/portfolio.py's `_ensure_c_contiguous`, applied in
`minimum_volatility_weights`, `maximum_sharpe_weights`,
`efficient_frontier_point`). Task brief §4.

Root cause (independently reproduced before the fix was implemented):
`align_mu_sigma`'s `DataFrame.reindex(...).to_numpy()` can return an
F-contiguous (Fortran-order) 2-D array. SciPy SLSQP's internal
finite-difference line search failed ("Positive directional derivative
for linesearch") on a real 50-asset Sigma_21 at formation date
2023-02-28 when F-contiguous, while a C-contiguous copy of the IDENTICAL
values (max abs difference 0.0) converged normally. The fix forces a
C-contiguous copy at the SciPy boundary only — no numerical value is
ever changed.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import config
from optimizer.portfolio import (
    _ensure_c_contiguous,
    efficient_frontier_point,
    maximum_sharpe_weights,
    minimum_volatility_weights,
    portfolio_expected_return,
    portfolio_volatility,
)

_skip_no_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set in the environment (expected in CI / clean checkouts)",
)


def _synthetic_mu_sigma(n=12, seed=3):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.02, 0.02, size=n)
    a = rng.normal(0.0, 0.02, size=(n, n))
    sigma = a @ a.T + np.eye(n) * 0.01  # PD by construction
    return mu, sigma


# ---------------------------------------------------------------------------
# 1-5. Both C- and F-contiguous inputs converge to equivalent results
# ---------------------------------------------------------------------------
def test_ensure_c_contiguous_never_changes_values():
    _, sigma = _synthetic_mu_sigma()
    sigma_f = np.asfortranarray(sigma)
    assert sigma_f.flags["F_CONTIGUOUS"] and not sigma_f.flags["C_CONTIGUOUS"]
    fixed = _ensure_c_contiguous(sigma_f)
    assert fixed.flags["C_CONTIGUOUS"]
    np.testing.assert_array_equal(fixed, sigma_f)  # bit-identical, not just close
    assert np.max(np.abs(fixed - sigma_f)) == 0.0


def test_max_sharpe_accepts_c_contiguous_inputs():
    mu, sigma = _synthetic_mu_sigma()
    assert sigma.flags["C_CONTIGUOUS"]
    w = maximum_sharpe_weights(mu, sigma, rf=0.002, max_weight=1.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_max_sharpe_accepts_f_contiguous_inputs():
    mu, sigma = _synthetic_mu_sigma()
    sigma_f = np.asfortranarray(sigma)
    assert sigma_f.flags["F_CONTIGUOUS"] and not sigma_f.flags["C_CONTIGUOUS"]
    w = maximum_sharpe_weights(mu, sigma_f, rf=0.002, max_weight=1.0)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_max_sharpe_c_and_f_contiguous_inputs_are_bit_identical_values():
    mu, sigma = _synthetic_mu_sigma()
    sigma_f = np.asfortranarray(sigma)
    assert np.max(np.abs(sigma - sigma_f)) == 0.0


def test_max_sharpe_c_and_f_contiguous_converge_to_equivalent_weights():
    mu, sigma = _synthetic_mu_sigma()
    sigma_f = np.asfortranarray(sigma)
    w_c = maximum_sharpe_weights(mu, sigma, rf=0.002, max_weight=1.0)
    w_f = maximum_sharpe_weights(mu, sigma_f, rf=0.002, max_weight=1.0)
    np.testing.assert_allclose(w_c, w_f, atol=1e-6)
    obj_c = portfolio_expected_return(w_c, mu) - portfolio_volatility(w_c, sigma)
    obj_f = portfolio_expected_return(w_f, mu) - portfolio_volatility(w_f, sigma)
    assert obj_c == pytest.approx(obj_f, abs=1e-8)


# ---------------------------------------------------------------------------
# 6. minimum_volatility_weights robust to both layouts
# ---------------------------------------------------------------------------
def test_min_vol_robust_to_both_layouts():
    _, sigma = _synthetic_mu_sigma()
    sigma_f = np.asfortranarray(sigma)
    w_c = minimum_volatility_weights(sigma, max_weight=1.0)
    w_f = minimum_volatility_weights(sigma_f, max_weight=1.0)
    np.testing.assert_allclose(w_c, w_f, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Efficient Frontier path robust to both layouts
# ---------------------------------------------------------------------------
def test_efficient_frontier_point_robust_to_both_layouts():
    mu, sigma = _synthetic_mu_sigma()
    sigma_f = np.asfortranarray(sigma)
    target = float(mu.mean())
    w_c = efficient_frontier_point(mu, sigma, target, max_weight=1.0)
    w_f = efficient_frontier_point(mu, sigma_f, target, max_weight=1.0)
    np.testing.assert_allclose(w_c, w_f, atol=1e-6)


# ---------------------------------------------------------------------------
# 8. constraints still satisfied under both layouts
# ---------------------------------------------------------------------------
def test_constraints_hold_under_f_contiguous_input():
    mu, sigma = _synthetic_mu_sigma(n=15)
    sigma_f = np.asfortranarray(sigma)
    w = maximum_sharpe_weights(mu, sigma_f, rf=0.001, max_weight=config.MAX_WEIGHT)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-6).all()
    assert (w <= config.MAX_WEIGHT + 1e-6).all()


# ---------------------------------------------------------------------------
# 10. normalization changes layout only, never financial/methodological inputs
# ---------------------------------------------------------------------------
def test_ensure_c_contiguous_is_a_pure_layout_operation():
    mu, sigma = _synthetic_mu_sigma()
    sigma_f = np.asfortranarray(sigma)
    out = _ensure_c_contiguous(sigma_f)
    assert out.shape == sigma_f.shape
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, sigma)  # same values as the original C-order matrix


# ---------------------------------------------------------------------------
# 9. Real historical formation 2023-02-28 now constructs successfully
# (Stage A construction only — no evaluation, no persistence, no
# performance reported; this is a narrow regression check, not the
# official experiment).
# ---------------------------------------------------------------------------
@pytest.mark.db
@_skip_no_db
def test_real_formation_2023_02_28_constructs_all_five_strategies():
    from pipeline import db as db_mod
    from pipeline import normalize
    import optimizer.walkforward as wf

    conn = db_mod.get_connection()
    try:
        conn.autocommit = True
        calendar = wf.load_formation_calendar(conn)
        prices = pd.read_sql(
            "SELECT ticker, date, close, total_return_idx FROM prices_raw ORDER BY ticker, date", conn
        )
    finally:
        conn.close()
    prices["date"] = pd.to_datetime(prices["date"])
    for col in ("close", "total_return_idx"):
        prices[col] = prices[col].astype(float)

    eligible, _ = wf.covariance_eligible_calendar(calendar, prices)
    fd = pd.Timestamp("2023-02-28")
    assert fd in set(eligible[wf.FORECAST_DATE_COL])

    mu = eligible[eligible[wf.FORECAST_DATE_COL] == fd].set_index("ticker")[wf.ENSEMBLE_PRED_COL]
    macro = normalize.normalize_macro()
    usgg10yr = float(macro.set_index("date")["yield_10y"].loc[fd])

    tickers = list(config.TICKER_UNIVERSE)
    construction = wf.construct_portfolios_at(fd, mu, prices, usgg10yr, tickers=tickers)

    assert set(construction.weights.keys()) == set(wf.STRATEGIES)
    for name, w in construction.weights.items():
        assert np.isfinite(w).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-6).all()
        if name != "EQUAL_WEIGHT":
            assert (w <= config.MAX_WEIGHT + 1e-6).all()
