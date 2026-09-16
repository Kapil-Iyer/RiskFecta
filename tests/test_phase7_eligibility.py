"""
RiskFecta Phase 7 covariance-eligibility amendment tests
(optimizer.walkforward.covariance_eligible_calendar). Task brief §6.

Synthetic by default; a few tests are marked `db` and READ-ONLY against
the real database (never write) — skipped automatically when
DATABASE_URL is unset, same convention as the rest of the suite.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import config
import optimizer.walkforward as wf
from optimizer.walkforward import (
    CovarianceEligibilityReport,
    covariance_eligible_calendar,
)
from optimizer.covariance import covariance_return_window

_skip_no_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set in the environment (expected in CI / clean checkouts)",
)

TICKERS = ["AAA", "BBB", "CCC"]


def _synthetic_prices(tickers=TICKERS, n_sessions=300, seed=0, start="2022-01-03"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_sessions)
    rows = []
    for i, ticker in enumerate(tickers):
        tri = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, size=n_sessions))
        close = tri * (1.0 + 0.01 * i)
        for d, c, t in zip(dates, close, tri):
            rows.append({"ticker": ticker, "date": d, "close": c, "total_return_idx": t})
    return pd.DataFrame(rows)


def _synthetic_calendar(dates, tickers=TICKERS, horizon=21):
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "ticker": t, "forecast_date": d,
                "target_date": d + pd.tseries.offsets.BDay(horizon),
                "ensemble_pred": 0.01,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Synthetic: boundary behavior
# ---------------------------------------------------------------------------
def test_formation_with_exactly_252_levels_is_excluded():
    prices = _synthetic_prices(n_sessions=300)
    # session index 251 (0-based) -> 252 TRI levels available -> excluded
    boundary_date = prices["date"].unique()[251]
    calendar = _synthetic_calendar([pd.Timestamp(boundary_date)])
    eligible, report = covariance_eligible_calendar(calendar, prices, tickers=TICKERS, window=252)
    assert len(eligible) == 0
    assert report.n_total_formations == 1
    assert report.n_eligible_formations == 0
    assert len(report.excluded) == 1
    exc = report.excluded[0]
    assert exc["tri_levels_available"] == 252
    assert exc["returns_available"] == 251
    assert exc["returns_required"] == 252


def test_formation_with_exactly_253_levels_is_eligible():
    prices = _synthetic_prices(n_sessions=300)
    # session index 252 (0-based) -> 253 TRI levels available -> eligible
    boundary_date = prices["date"].unique()[252]
    calendar = _synthetic_calendar([pd.Timestamp(boundary_date)])
    eligible, report = covariance_eligible_calendar(calendar, prices, tickers=TICKERS, window=252)
    assert len(eligible) == len(TICKERS)  # one row per ticker for the one eligible date
    assert report.n_eligible_formations == 1
    assert report.excluded == []


def test_eligible_formation_produces_exactly_252_returns_from_253_levels():
    prices = _synthetic_prices(n_sessions=300)
    boundary_date = pd.Timestamp(prices["date"].unique()[252])
    window = covariance_return_window(prices, TICKERS, as_of_date=boundary_date, window=252)
    assert window.shape == (252, len(TICKERS))


def test_covariance_eligible_calendar_never_mutates_input():
    prices = _synthetic_prices(n_sessions=300)
    dates = pd.to_datetime(sorted(prices["date"].unique()))[[251, 252, 260]]
    calendar = _synthetic_calendar(dates)
    calendar_copy = calendar.copy(deep=True)
    covariance_eligible_calendar(calendar, prices, tickers=TICKERS, window=252)
    pd.testing.assert_frame_equal(calendar, calendar_copy)


def test_covariance_eligible_calendar_returns_report_dataclass():
    prices = _synthetic_prices(n_sessions=300)
    dates = pd.to_datetime(sorted(prices["date"].unique()))[[252, 260]]
    calendar = _synthetic_calendar(dates)
    _, report = covariance_eligible_calendar(calendar, prices, tickers=TICKERS, window=252)
    assert isinstance(report, CovarianceEligibilityReport)


def test_t_plus_horizon_alignment_holds_for_eligible_dates():
    prices = _synthetic_prices(n_sessions=300)
    dates = pd.to_datetime(sorted(prices["date"].unique()))[[252, 260, 270]]
    calendar = _synthetic_calendar(dates)
    eligible, _ = covariance_eligible_calendar(calendar, prices, tickers=TICKERS, window=252)
    wf.verify_t_plus_horizon_alignment(eligible, prices, horizon=21)  # must not raise


def test_config_covar_window_unchanged():
    assert config.COVAR_WINDOW == 252


def test_eligibility_function_never_references_spxt_or_sealed_files():
    import inspect
    src = inspect.getsource(covariance_eligible_calendar)
    assert "spxt" not in src.lower()
    for forbidden in ("prices_sealed.csv", "macro_sealed.csv", "prices_extension.csv", "macro_extension.csv"):
        assert forbidden not in src


# ---------------------------------------------------------------------------
# Real data (read-only)
# ---------------------------------------------------------------------------
@pytest.mark.db
@_skip_no_db
def test_real_forecasting_calendar_still_47_dates():
    from pipeline import db as db_mod
    conn = db_mod.get_connection()
    try:
        conn.autocommit = True
        full_calendar = wf.load_formation_calendar(conn)
    finally:
        conn.close()
    assert full_calendar[wf.FORECAST_DATE_COL].nunique() == 47
    assert full_calendar[wf.FORECAST_DATE_COL].min() == pd.Timestamp("2022-02-25")
    assert full_calendar[wf.FORECAST_DATE_COL].max() == pd.Timestamp("2026-01-02")


@pytest.mark.db
@_skip_no_db
def test_real_phase7_eligible_calendar_excludes_exactly_20220225():
    from pipeline import db as db_mod
    conn = db_mod.get_connection()
    try:
        conn.autocommit = True
        full_calendar = wf.load_formation_calendar(conn)
        prices = pd.read_sql(
            "SELECT ticker, date, close, total_return_idx FROM prices_raw ORDER BY ticker, date", conn
        )
    finally:
        conn.close()
    prices["date"] = pd.to_datetime(prices["date"])
    for col in ("close", "total_return_idx"):
        prices[col] = prices[col].astype(float)

    eligible, report = covariance_eligible_calendar(full_calendar, prices)

    assert report.n_total_formations == 47
    assert report.n_eligible_formations == 46
    assert len(report.excluded) == 1
    exc = report.excluded[0]
    assert pd.Timestamp(exc["formation_date"]) == pd.Timestamp("2022-02-25")
    assert exc["tri_levels_available"] == 252
    assert exc["returns_available"] == 251
    assert exc["returns_required"] == 252

    eligible_dates = sorted(eligible[wf.FORECAST_DATE_COL].unique())
    assert pd.Timestamp(eligible_dates[0]) == pd.Timestamp("2022-03-28")
    assert pd.Timestamp(eligible_dates[-1]) == pd.Timestamp("2026-01-02")
    assert len(eligible_dates) == 46

    # Phase 4-6 predictions calendar itself remains untouched.
    assert full_calendar[wf.FORECAST_DATE_COL].nunique() == 47


@pytest.mark.db
@_skip_no_db
def test_real_boundary_date_2022_03_28_produces_exactly_252_returns():
    from pipeline import db as db_mod
    conn = db_mod.get_connection()
    try:
        conn.autocommit = True
        prices = pd.read_sql(
            "SELECT ticker, date, close, total_return_idx FROM prices_raw ORDER BY ticker, date", conn
        )
    finally:
        conn.close()
    prices["date"] = pd.to_datetime(prices["date"])
    for col in ("close", "total_return_idx"):
        prices[col] = prices[col].astype(float)

    window = covariance_return_window(
        prices, list(config.TICKER_UNIVERSE), as_of_date=pd.Timestamp("2022-03-28")
    )
    assert window.shape == (252, 50)


@pytest.mark.db
@_skip_no_db
def test_real_spxt_covers_the_46_date_eligible_phase7_calendar_specifically():
    # The 47-date forecasting calendar is a superset of what Phase 7
    # portfolio construction actually uses; SPXT coverage against the
    # full 47 dates (tests/test_optimizer_spxt.py) doesn't by itself
    # prove coverage of the 46-date ELIGIBLE calendar the official driver
    # actually evaluates. Check that directly.
    from pipeline import db as db_mod
    import optimizer.benchmark_spxt as spxt_mod

    conn = db_mod.get_connection()
    try:
        conn.autocommit = True
        full_calendar = wf.load_formation_calendar(conn)
        prices = pd.read_sql(
            "SELECT ticker, date, close, total_return_idx FROM prices_raw ORDER BY ticker, date", conn
        )
    finally:
        conn.close()
    prices["date"] = pd.to_datetime(prices["date"])
    for col in ("close", "total_return_idx"):
        prices[col] = prices[col].astype(float)

    eligible, report = covariance_eligible_calendar(full_calendar, prices)
    assert report.n_eligible_formations == 46

    spxt_raw = spxt_mod.load_spxt_raw()
    spxt_mod.verify_spxt_covers_formation_calendar(spxt_raw, eligible)  # must not raise


@pytest.mark.db
@_skip_no_db
def test_real_excluded_date_2022_02_25_still_fails_loudly_at_252_window():
    from pipeline import db as db_mod
    conn = db_mod.get_connection()
    try:
        conn.autocommit = True
        prices = pd.read_sql(
            "SELECT ticker, date, close, total_return_idx FROM prices_raw ORDER BY ticker, date", conn
        )
    finally:
        conn.close()
    prices["date"] = pd.to_datetime(prices["date"])
    for col in ("close", "total_return_idx"):
        prices[col] = prices[col].astype(float)

    with pytest.raises(ValueError, match="need 253 causal TRI levels"):
        covariance_return_window(
            prices, list(config.TICKER_UNIVERSE), as_of_date=pd.Timestamp("2022-02-25")
        )
