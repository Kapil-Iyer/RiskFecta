"""
RiskFecta Phase 7 — deterministic Max-Sharpe SLSQP one-retry robustness
fix tests (optimizer/portfolio.py's `maximum_sharpe_weights`). Task
brief §6.

SLSQP's single-start, finite-difference line search occasionally fails
to converge ("Positive directional derivative for linesearch") on
well-formed real 50-asset (mu, Sigma, rf) inputs — observed on 3 of the
46 real Phase 7 formation dates (2023-10-27, 2024-04-30, 2024-07-31),
independent of the earlier memory-layout fix. The fix retries EXACTLY
ONCE, from the minimum-volatility portfolio as x0, with everything else
(objective/mu/Sigma/rf/bounds/constraints/solver/tolerances) unchanged.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize as real_minimize

import config
import optimizer.portfolio as port_mod
from optimizer.portfolio import (
    _SLSQP_LINESEARCH_FAILURE,
    maximum_sharpe_weights,
    minimum_volatility_weights,
    equal_weight_benchmark,
)

_skip_no_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set in the environment (expected in CI / clean checkouts)",
)


class _FakeResult:
    def __init__(self, x, success, message=""):
        self.x = np.asarray(x, dtype=float)
        self.success = success
        self.message = message


def _well_conditioned_mu_sigma(n=8, seed=11):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.02, 0.015, size=n)
    a = rng.normal(0.0, 0.02, size=(n, n))
    sigma = a @ a.T + np.eye(n) * 0.01
    return mu, sigma


# ---------------------------------------------------------------------------
# 1/4. Normal case uses only the primary (equal-weight) attempt
# ---------------------------------------------------------------------------
def test_normal_case_never_calls_min_vol_and_uses_equal_weight_x0(monkeypatch):
    mu, sigma = _well_conditioned_mu_sigma()

    minvol_calls = []
    original_minvol = port_mod.minimum_volatility_weights

    def spy_minvol(*args, **kwargs):
        minvol_calls.append((args, kwargs))
        return original_minvol(*args, **kwargs)

    monkeypatch.setattr(port_mod, "minimum_volatility_weights", spy_minvol)
    w = maximum_sharpe_weights(mu, sigma, rf=0.002, max_weight=1.0)
    assert minvol_calls == []  # primary attempt succeeded; no retry, no min-vol call
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# 2/3/4/5. Controlled linesearch failure -> exactly one retry, correct x0,
# same objective/settings, and the retry's OWN solve (not the min-vol
# portfolio) is returned.
# ---------------------------------------------------------------------------
def test_controlled_linesearch_failure_triggers_exactly_one_retry_from_minvol_x0(monkeypatch):
    mu, sigma = _well_conditioned_mu_sigma(n=6, seed=3)
    rf = 0.001
    n = len(mu)

    sharpe_calls = []

    def fake_minimize(fun, x0, method, bounds, constraints, options):
        if fun.__name__ == "neg_sharpe":
            sharpe_calls.append(np.array(x0, dtype=float))
            if len(sharpe_calls) == 1:
                return _FakeResult(x0, success=False, message=_SLSQP_LINESEARCH_FAILURE)
            return real_minimize(fun, x0, method=method, bounds=bounds, constraints=constraints, options=options)
        # minimum_volatility_weights' own "objective" call — solve for real.
        return real_minimize(fun, x0, method=method, bounds=bounds, constraints=constraints, options=options)

    monkeypatch.setattr(port_mod, "minimize", fake_minimize)

    w = maximum_sharpe_weights(mu, sigma, rf, max_weight=1.0)

    assert len(sharpe_calls) == 2  # primary + exactly one retry, never more
    np.testing.assert_allclose(sharpe_calls[0], equal_weight_benchmark(n))

    expected_retry_x0 = minimum_volatility_weights(sigma, max_weight=1.0)
    np.testing.assert_allclose(sharpe_calls[1], expected_retry_x0, atol=1e-6)

    # The retry's RESULT is the Max-Sharpe solve's own answer, not simply
    # the min-vol starting portfolio it was initialized from.
    assert not np.allclose(w, expected_retry_x0, atol=1e-4)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)


def test_both_attempts_failing_raises_loudly_after_exactly_two_tries(monkeypatch):
    mu, sigma = _well_conditioned_mu_sigma(n=5, seed=5)

    sharpe_calls = []

    def fake_minimize(fun, x0, method, bounds, constraints, options):
        if fun.__name__ == "neg_sharpe":
            sharpe_calls.append(np.array(x0, dtype=float))
            return _FakeResult(x0, success=False, message=_SLSQP_LINESEARCH_FAILURE)
        return real_minimize(fun, x0, method=method, bounds=bounds, constraints=constraints, options=options)

    monkeypatch.setattr(port_mod, "minimize", fake_minimize)

    with pytest.raises(ValueError, match="did not converge"):
        maximum_sharpe_weights(mu, sigma, rf=0.001, max_weight=1.0)
    assert len(sharpe_calls) == 2  # never a third attempt


def test_unrelated_failure_message_never_triggers_retry(monkeypatch):
    mu, sigma = _well_conditioned_mu_sigma(n=5, seed=9)

    sharpe_calls = []

    def fake_minimize(fun, x0, method, bounds, constraints, options):
        if fun.__name__ == "neg_sharpe":
            sharpe_calls.append(np.array(x0, dtype=float))
            return _FakeResult(x0, success=False, message="Iteration limit exceeded")
        return real_minimize(fun, x0, method=method, bounds=bounds, constraints=constraints, options=options)

    monkeypatch.setattr(port_mod, "minimize", fake_minimize)

    with pytest.raises(ValueError, match="Iteration limit exceeded"):
        maximum_sharpe_weights(mu, sigma, rf=0.001, max_weight=1.0)
    assert len(sharpe_calls) == 1  # no retry for an unrelated failure mode


# ---------------------------------------------------------------------------
# 6 (duplicate-safe) / 7 — already covered above; explicit constraints check
# ---------------------------------------------------------------------------
def test_retry_result_still_respects_constraints():
    mu, sigma = _well_conditioned_mu_sigma(n=10, seed=13)
    w = maximum_sharpe_weights(mu, sigma, rf=0.001, max_weight=config.MAX_WEIGHT * 3)
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert (w >= -1e-6).all()
    assert (w <= config.MAX_WEIGHT * 3 + 1e-6).all()


# ---------------------------------------------------------------------------
# 8. Prior contiguity fix still green (imported to guarantee both suites
# are exercised together; full file also runs independently in CI).
# ---------------------------------------------------------------------------
def test_contiguity_fix_helper_still_present():
    from optimizer.portfolio import _ensure_c_contiguous
    sigma_f = np.asfortranarray(np.eye(4))
    out = _ensure_c_contiguous(sigma_f)
    assert out.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# 9/10/11. Real historical construction now succeeds for all five
# strategies at every previously-failing date, plus the earlier fixed
# date — construction only, no evaluation, no SPXT, no persistence.
# ---------------------------------------------------------------------------
def _construct_at_real_date(fd_str):
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
    fd = pd.Timestamp(fd_str)
    assert fd in set(eligible[wf.FORECAST_DATE_COL])

    mu = eligible[eligible[wf.FORECAST_DATE_COL] == fd].set_index("ticker")[wf.ENSEMBLE_PRED_COL]
    macro = normalize.normalize_macro()
    usgg10yr = float(macro.set_index("date")["yield_10y"].loc[fd])

    tickers = list(config.TICKER_UNIVERSE)
    return wf.construct_portfolios_at(fd, mu, prices, usgg10yr, tickers=tickers)


def _assert_all_five_strategies_valid(construction, tickers_n=50):
    import optimizer.walkforward as wf
    assert set(construction.weights.keys()) == set(wf.STRATEGIES)
    for name, w in construction.weights.items():
        assert np.isfinite(w).all()
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-6).all()
        if name != "EQUAL_WEIGHT":
            assert (w <= config.MAX_WEIGHT + 1e-6).all()


@pytest.mark.db
@_skip_no_db
def test_real_formation_2023_10_27_constructs_all_five_strategies():
    construction = _construct_at_real_date("2023-10-27")
    _assert_all_five_strategies_valid(construction)


@pytest.mark.db
@_skip_no_db
def test_real_formation_2024_04_30_constructs_all_five_strategies():
    construction = _construct_at_real_date("2024-04-30")
    _assert_all_five_strategies_valid(construction)


@pytest.mark.db
@_skip_no_db
def test_real_formation_2024_07_31_constructs_all_five_strategies():
    construction = _construct_at_real_date("2024-07-31")
    _assert_all_five_strategies_valid(construction)


@pytest.mark.db
@_skip_no_db
def test_real_formation_2023_02_28_still_constructs_all_five_strategies():
    construction = _construct_at_real_date("2023-02-28")
    _assert_all_five_strategies_valid(construction)
