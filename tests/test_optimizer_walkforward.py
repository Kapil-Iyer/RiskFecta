"""
RiskFecta Phase 7B — historical walk-forward driver tests
(optimizer/walkforward.py). Task brief §20. Synthetic/mock by default;
a handful of tests are marked `db`/`integration` and READ-ONLY against
the real database / real Bloomberg export (never write) — skipped
automatically when unavailable, same convention as
tests/test_db_integration.py / tests/test_integration_real_data.py.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import config
from optimizer.walkforward import (
    STRATEGIES,
    OPTIMIZED_STRATEGIES,
    ConstructionResult,
    StrategyPeriodResult,
    aggregate_statistics,
    construct_portfolios_at,
    covariance_diagnostics,
    cumulative_compounded_return,
    load_formation_calendar,
    realized_portfolio_return,
    realized_stock_returns,
    spx_price_return_diagnostic,
    turnover,
    validate_formation_calendar,
    verify_sequential_non_overlapping_path,
    verify_t_plus_horizon_alignment,
)
import optimizer.walkforward as wf_mod

TICKERS = ["AAA", "BBB", "CCC"]


def _synthetic_prices(tickers=TICKERS, n_sessions=280, seed=0, start="2022-01-03"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_sessions)
    rows = []
    for i, ticker in enumerate(tickers):
        tri = 100.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, size=n_sessions))
        close = tri * (1.0 + 0.01 * i)
        for d, c, t in zip(dates, close, tri):
            rows.append({"ticker": ticker, "date": d, "close": c, "total_return_idx": t})
    return pd.DataFrame(rows)


def _synthetic_mu(tickers=TICKERS, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series({t: float(v) for t, v in zip(tickers, rng.normal(0.02, 0.01, size=len(tickers)))})


# ---------------------------------------------------------------------------
# FORMATION CALENDAR
# ---------------------------------------------------------------------------
def _calendar_df(tickers=TICKERS, n_dates=3, preds=None):
    dates = pd.bdate_range("2022-01-03", periods=n_dates * 30, freq="B")[::30][:n_dates]
    rows = []
    for i, d in enumerate(dates):
        for j, t in enumerate(tickers):
            rows.append({
                "ticker": t, "forecast_date": d,
                "target_date": d + pd.tseries.offsets.BDay(21),
                "ensemble_pred": 0.01 * (i + j) if preds is None else preds,
            })
    return pd.DataFrame(rows)


def test_validate_formation_calendar_accepts_clean_fixture():
    df = _calendar_df()
    validate_formation_calendar(df, TICKERS)  # must not raise


def test_validate_formation_calendar_missing_ticker_fails_loudly():
    df = _calendar_df()
    df = df[~((df["ticker"] == "AAA") & (df["forecast_date"] == df["forecast_date"].iloc[0]))]
    with pytest.raises(ValueError, match="missing ticker"):
        validate_formation_calendar(df, TICKERS)


def test_validate_formation_calendar_extra_ticker_fails_loudly():
    df = _calendar_df()
    extra = df.iloc[[0]].copy()
    extra["ticker"] = "ZZZ"
    df = pd.concat([df, extra], ignore_index=True)
    with pytest.raises(ValueError, match="extra ticker"):
        validate_formation_calendar(df, TICKERS)


def test_validate_formation_calendar_duplicate_identity_fails_loudly():
    df = _calendar_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_formation_calendar(df, TICKERS)


def test_validate_formation_calendar_nan_prediction_fails_loudly():
    df = _calendar_df()
    df.loc[0, "ensemble_pred"] = np.nan
    with pytest.raises(ValueError, match="NULL/NaN/Inf"):
        validate_formation_calendar(df, TICKERS)


def test_validate_formation_calendar_inf_prediction_fails_loudly():
    df = _calendar_df()
    df.loc[0, "ensemble_pred"] = np.inf
    with pytest.raises(ValueError, match="NULL/NaN/Inf"):
        validate_formation_calendar(df, TICKERS)


def test_verify_t_plus_horizon_alignment_passes_for_correct_gap():
    prices = _synthetic_prices(n_sessions=280)
    calendar = build_calendar_from_prices(prices, gap=21)
    verify_t_plus_horizon_alignment(calendar, prices, horizon=21)  # must not raise


def test_verify_t_plus_horizon_alignment_fails_on_wrong_gap():
    prices = _synthetic_prices(n_sessions=280)
    calendar = build_calendar_from_prices(prices, gap=20)  # wrong gap
    with pytest.raises(ValueError, match="valid session"):
        verify_t_plus_horizon_alignment(calendar, prices, horizon=21)


def build_calendar_from_prices(prices, gap):
    from pipeline.folds import build_global_calendar
    cal = build_global_calendar(prices)
    fd, td = cal[100], cal[100 + gap]
    return pd.DataFrame({"forecast_date": [fd], "target_date": [td]})


def test_verify_sequential_path_true_for_chained_dates():
    df = pd.DataFrame({
        "forecast_date": pd.to_datetime(["2022-01-03", "2022-02-01", "2022-03-01"]),
        "target_date": pd.to_datetime(["2022-02-01", "2022-03-01", "2022-04-01"]),
    })
    assert verify_sequential_non_overlapping_path(df) is True


def test_verify_sequential_path_false_for_gap():
    df = pd.DataFrame({
        "forecast_date": pd.to_datetime(["2022-01-03", "2022-02-15", "2022-03-01"]),
        "target_date": pd.to_datetime(["2022-02-01", "2022-03-01", "2022-04-01"]),
    })
    assert verify_sequential_non_overlapping_path(df) is False


# ---------------------------------------------------------------------------
# CONSTRUCTION
# ---------------------------------------------------------------------------
def test_construct_portfolios_produces_exactly_five_strategies():
    prices = _synthetic_prices(n_sessions=280)
    mu = _synthetic_mu()
    formation_date = prices["date"].iloc[-10]
    result = construct_portfolios_at(formation_date, mu, prices, usgg10yr=4.5, tickers=TICKERS, max_weight=1.0)
    assert isinstance(result, ConstructionResult)
    assert set(result.weights.keys()) == set(STRATEGIES)
    assert len(STRATEGIES) == 5
    assert set(OPTIMIZED_STRATEGIES) == {"SAMPLE_MINVOL", "SAMPLE_MAXSHARPE", "LW_MINVOL", "LW_MAXSHARPE"}


def test_construct_portfolios_all_sum_to_one_and_respect_bounds():
    prices = _synthetic_prices(n_sessions=280)
    mu = _synthetic_mu()
    formation_date = prices["date"].iloc[-10]
    max_weight = 0.5  # n=3 tickers here; config.MAX_WEIGHT=0.10 would be infeasible for n<10
    result = construct_portfolios_at(formation_date, mu, prices, usgg10yr=4.5, tickers=TICKERS, max_weight=max_weight)
    for name, w in result.weights.items():
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-6).all()
        if name != "EQUAL_WEIGHT":
            assert (w <= max_weight + 1e-6).all()


def test_equal_weight_is_exactly_1_over_n_independent_of_optimizer():
    from optimizer.portfolio import equal_weight_benchmark
    import inspect
    w = equal_weight_benchmark(50)
    assert np.allclose(w, 0.02)
    # Structural proof it cannot be affected by MAX_WEIGHT: no such parameter exists.
    assert "max_weight" not in inspect.signature(equal_weight_benchmark).parameters


def test_sample_and_lw_use_identical_causal_window(monkeypatch):
    calls = []
    original = wf_mod.covariance_return_window

    def spy(*args, **kwargs):
        window = original(*args, **kwargs)
        calls.append(window)
        return window

    monkeypatch.setattr(wf_mod, "covariance_return_window", spy)
    prices = _synthetic_prices(n_sessions=280)
    mu = _synthetic_mu()
    formation_date = prices["date"].iloc[-10]
    construct_portfolios_at(formation_date, mu, prices, usgg10yr=4.5, tickers=TICKERS, max_weight=1.0)
    # Exactly ONE causal window fetched and fed to BOTH estimators — never
    # two separately-fetched windows for "comparable" estimators.
    assert len(calls) == 1


def test_construction_deterministic():
    prices = _synthetic_prices(n_sessions=280)
    mu = _synthetic_mu()
    formation_date = prices["date"].iloc[-10]
    r1 = construct_portfolios_at(formation_date, mu, prices, usgg10yr=4.5, tickers=TICKERS, max_weight=1.0)
    r2 = construct_portfolios_at(formation_date, mu, prices, usgg10yr=4.5, tickers=TICKERS, max_weight=1.0)
    for name in STRATEGIES:
        np.testing.assert_allclose(r1.weights[name], r2.weights[name], atol=1e-8)


def test_construction_fails_loudly_on_nan_mu():
    prices = _synthetic_prices(n_sessions=280)
    mu = _synthetic_mu()
    mu["AAA"] = np.nan
    formation_date = prices["date"].iloc[-10]
    with pytest.raises(ValueError, match="NaN/Inf"):
        construct_portfolios_at(formation_date, mu, prices, usgg10yr=4.5, tickers=TICKERS, max_weight=1.0)


# ---------------------------------------------------------------------------
# CAUSALITY
# ---------------------------------------------------------------------------
def test_future_tri_and_close_perturbation_cannot_alter_weights_T():
    prices = _synthetic_prices(n_sessions=280)
    mu = _synthetic_mu()
    formation_date = prices["date"].iloc[-15]

    baseline = construct_portfolios_at(formation_date, mu, prices, usgg10yr=4.5, tickers=TICKERS, max_weight=1.0)

    perturbed = prices.copy()
    future_mask = perturbed["date"] > formation_date
    assert future_mask.sum() > 0
    perturbed.loc[future_mask, "total_return_idx"] *= 3.0
    perturbed.loc[future_mask, "close"] *= 3.0

    after = construct_portfolios_at(formation_date, mu, perturbed, usgg10yr=4.5, tickers=TICKERS, max_weight=1.0)
    for name in STRATEGIES:
        np.testing.assert_allclose(baseline.weights[name], after.weights[name], atol=1e-8)


def test_construct_portfolios_cannot_see_actual_return_or_spx_or_macro_structurally():
    # Structural causality guarantee: construct_portfolios_at's signature
    # accepts no realized-return, SPX, or macro-history argument at all —
    # there is no code path for "future actual_return"/"future SPX"/
    # "future macro" to reach construction.
    import inspect
    sig = inspect.signature(construct_portfolios_at)
    assert set(sig.parameters) == {"formation_date", "mu", "prices", "usgg10yr", "tickers", "max_weight"}
    # usgg10yr is a single as-of-T scalar the caller supplies — not a
    # macro time series this function could reach into the future with.
    # (`from __future__ import annotations` in walkforward.py makes this a
    # string annotation rather than the `float` type object.)
    assert sig.parameters["usgg10yr"].annotation in (float, "float", inspect.Parameter.empty)


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
def test_realized_stock_returns_exact_tri_formula():
    prices = pd.DataFrame([
        {"ticker": "AAA", "date": pd.Timestamp("2022-01-03"), "close": 100.0, "total_return_idx": 100.0},
        {"ticker": "AAA", "date": pd.Timestamp("2022-02-01"), "close": 105.0, "total_return_idx": 110.0},
    ])
    out = realized_stock_returns(prices, ["AAA"], pd.Timestamp("2022-01-03"), pd.Timestamp("2022-02-01"))
    assert out["AAA"] == pytest.approx(110.0 / 100.0 - 1.0)


def test_realized_stock_returns_never_uses_close():
    prices_a = pd.DataFrame([
        {"ticker": "AAA", "date": pd.Timestamp("2022-01-03"), "close": 100.0, "total_return_idx": 100.0},
        {"ticker": "AAA", "date": pd.Timestamp("2022-02-01"), "close": 999.0, "total_return_idx": 110.0},
    ])
    out = realized_stock_returns(prices_a, ["AAA"], pd.Timestamp("2022-01-03"), pd.Timestamp("2022-02-01"))
    assert out["AAA"] == pytest.approx(0.10)  # driven by TRI, not the wildly different close


def test_realized_portfolio_return_exact_weighted_sum_with_explicit_alignment():
    realized = pd.Series({"AAA": 0.10, "BBB": -0.05, "CCC": 0.02})
    weights = np.array([0.5, 0.3, 0.2])
    tickers = ["AAA", "BBB", "CCC"]
    out = realized_portfolio_return(weights, tickers, realized)
    expected = 0.5 * 0.10 + 0.3 * -0.05 + 0.2 * 0.02
    assert out == pytest.approx(expected)


def test_realized_portfolio_return_alignment_is_explicit_not_positional():
    # `realized` is deliberately in a DIFFERENT order than `tickers`/`weights`
    # — a positional (non-aligned) implementation would give the wrong answer.
    realized = pd.Series({"CCC": 0.02, "AAA": 0.10, "BBB": -0.05})
    weights = np.array([0.5, 0.3, 0.2])  # AAA, BBB, CCC
    tickers = ["AAA", "BBB", "CCC"]
    out = realized_portfolio_return(weights, tickers, realized)
    expected = 0.5 * 0.10 + 0.3 * -0.05 + 0.2 * 0.02
    assert out == pytest.approx(expected)


def test_realized_portfolio_return_missing_ticker_fails_loudly():
    realized = pd.Series({"AAA": 0.10, "BBB": -0.05})
    with pytest.raises(ValueError, match="missing realized return"):
        realized_portfolio_return(np.array([0.5, 0.3, 0.2]), ["AAA", "BBB", "CCC"], realized)


# ---------------------------------------------------------------------------
# TURNOVER
# ---------------------------------------------------------------------------
def test_turnover_exact_formula():
    w_t = np.array([0.5, 0.3, 0.2])
    w_prev = np.array([0.4, 0.4, 0.2])
    out = turnover(w_t, w_prev, TICKERS)
    expected = 0.5 * (abs(0.5 - 0.4) + abs(0.3 - 0.4) + abs(0.2 - 0.2))
    assert out == pytest.approx(expected)


def test_turnover_first_period_is_nan_not_zero():
    out = turnover(np.array([0.5, 0.3, 0.2]), None, TICKERS)
    assert np.isnan(out)


def test_turnover_ticker_alignment_enforced():
    w_t = np.array([0.5, 0.5])
    tickers_t = ["AAA", "BBB"]
    w_prev = np.array([0.5, 0.5])
    tickers_prev = ["BBB", "AAA"]  # reversed order
    out = turnover(w_t, w_prev, tickers_t, tickers_prev)
    assert out == pytest.approx(0.5 * (abs(0.5 - 0.5) + abs(0.5 - 0.5)))  # 0, since values match once aligned

    # Now make them genuinely different once aligned.
    w_prev2 = np.array([0.5, 0.5])  # BBB=0.5, AAA=0.5 aligns to [AAA=0.5, BBB=0.5] -> identical
    w_t2 = np.array([0.6, 0.4])     # AAA=0.6, BBB=0.4
    out2 = turnover(w_t2, w_prev2, tickers_t, tickers_prev)
    assert out2 == pytest.approx(0.5 * (abs(0.6 - 0.5) + abs(0.4 - 0.5)))


def test_turnover_equal_weight_naturally_zero_when_universe_unchanged():
    from optimizer.portfolio import equal_weight_benchmark
    w1 = equal_weight_benchmark(50)
    w2 = equal_weight_benchmark(50)
    out = turnover(w2, w1, list(config.TICKER_UNIVERSE))
    assert out == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# AGGREGATE STATISTICS
# ---------------------------------------------------------------------------
def _periods(returns, turnovers, max_weights):
    dates = pd.bdate_range("2022-01-03", periods=len(returns), freq="21B")
    return [
        StrategyPeriodResult(formation_date=d, realized_return=r, turnover=t, max_weight_observed=m)
        for d, r, t, m in zip(dates, returns, turnovers, max_weights)
    ]


def test_aggregate_statistics_hand_computed():
    periods = _periods([0.02, -0.01, 0.03], [np.nan, 0.1, 0.2], [0.10, 0.09, 0.08])
    stats = aggregate_statistics(periods)
    assert stats["n_periods"] == 3
    assert stats["mean_return_21"] == pytest.approx(np.mean([0.02, -0.01, 0.03]))
    assert stats["median_return_21"] == pytest.approx(np.median([0.02, -0.01, 0.03]))
    assert stats["min_return_21"] == pytest.approx(-0.01)
    assert stats["max_return_21"] == pytest.approx(0.03)
    assert stats["hit_rate"] == pytest.approx(2 / 3)
    assert stats["mean_turnover"] == pytest.approx(np.mean([0.1, 0.2]))  # first NaN excluded
    assert stats["max_turnover"] == pytest.approx(0.2)
    assert stats["avg_max_weight"] == pytest.approx(np.mean([0.10, 0.09, 0.08]))
    assert stats["max_observed_weight"] == pytest.approx(0.10)


def test_aggregate_statistics_no_annualization_keys():
    periods = _periods([0.01, 0.02], [np.nan, 0.1], [0.1, 0.1])
    stats = aggregate_statistics(periods)
    forbidden_substrings = ["annual", "sqrt(252", "sharpe_ratio", "max_drawdown"]
    for key in stats:
        for bad in forbidden_substrings:
            assert bad not in key.lower()


def test_cumulative_return_requires_sequential_path():
    periods = _periods([0.01, 0.02, -0.01], [np.nan, 0.1, 0.1], [0.1, 0.1, 0.1])
    good_calendar = pd.DataFrame({
        "forecast_date": pd.to_datetime(["2022-01-03", "2022-02-01", "2022-03-01"]),
        "target_date": pd.to_datetime(["2022-02-01", "2022-03-01", "2022-04-01"]),
    })
    out = cumulative_compounded_return(periods, good_calendar)
    expected = (1.01 * 1.02 * 0.99) - 1.0
    assert out == pytest.approx(expected)

    bad_calendar = pd.DataFrame({
        "forecast_date": pd.to_datetime(["2022-01-03", "2022-02-15", "2022-03-01"]),
        "target_date": pd.to_datetime(["2022-02-01", "2022-03-01", "2022-04-01"]),
    })
    with pytest.raises(ValueError, match="sequential"):
        cumulative_compounded_return(periods, bad_calendar)


# ---------------------------------------------------------------------------
# COVARIANCE DIAGNOSTICS
# ---------------------------------------------------------------------------
def test_covariance_diagnostics_known_eigenvalues():
    sigma = np.diag([0.01, 0.04, 0.09])
    w = np.array([1 / 3, 1 / 3, 1 / 3])
    diag = covariance_diagnostics(sigma, w)
    assert diag["min_eigenvalue"] == pytest.approx(0.01)
    assert diag["max_eigenvalue"] == pytest.approx(0.09)
    assert diag["condition_number"] == pytest.approx(9.0)
    assert diag["is_psd"] is True
    expected_vol = np.sqrt(w @ sigma @ w)
    assert diag["predicted_volatility_21"] == pytest.approx(expected_vol)
    assert diag["max_weight_observed"] == pytest.approx(1 / 3)


def test_covariance_diagnostics_flags_non_psd():
    sigma = np.array([[1.0, 2.0], [2.0, 1.0]])  # indefinite: eigenvalues -1, 3
    w = np.array([0.5, 0.5])
    diag = covariance_diagnostics(sigma, w)
    assert diag["is_psd"] is False
    assert diag["min_eigenvalue"] < 0


# ---------------------------------------------------------------------------
# SPX — HARD STOP evidence + non-official diagnostic
# ---------------------------------------------------------------------------
def test_strategies_never_includes_an_spx_entry():
    for s in STRATEGIES:
        assert "SPX" not in s


def test_spx_price_return_diagnostic_matches_manual_ratio():
    macro = pd.DataFrame({
        "date": pd.to_datetime(["2022-01-03", "2022-02-01"]),
        "spx": [4000.0, 4200.0],
        "vix": [20.0, 21.0],
        "yield_10y": [1.5, 1.6],
    })
    out = spx_price_return_diagnostic(macro, pd.Timestamp("2022-01-03"), pd.Timestamp("2022-02-01"))
    assert out == pytest.approx(4200.0 / 4000.0 - 1.0)


def test_spx_diagnostic_is_never_called_total_return_in_docstring():
    doc = spx_price_return_diagnostic.__doc__ or ""
    assert "NON-OFFICIAL" in doc
    assert "total return" in doc.lower() or "total-return" in doc.lower()


@pytest.mark.integration
@pytest.mark.skipif(
    not (config.DATA_RAW / "macro.csv").exists(),
    reason="real Bloomberg macro export not present under data/raw/ (expected in CI / clean checkouts)",
)
def test_real_macro_csv_only_has_spx_px_last_not_total_return():
    raw = pd.read_csv(config.DATA_RAW / "macro.csv", nrows=1)
    cols = list(raw.columns)
    spx_cols = [c for c in cols if "SPX" in c.upper()]
    assert any("PX_LAST" in c.upper() for c in spx_cols)
    assert not any("TOTAL_RETURN" in c.upper() or "TOT_RETURN" in c.upper() for c in spx_cols)


# ---------------------------------------------------------------------------
# MARCH / real formation calendar (read-only DB checks)
# ---------------------------------------------------------------------------
_skip_no_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set in the environment (expected in CI / clean checkouts)",
)


@pytest.mark.db
@_skip_no_db
def test_real_formation_calendar_exactly_47_dates_and_bounds():
    from pipeline import db as db_mod
    conn = db_mod.get_connection()
    try:
        conn.autocommit = True
        df = load_formation_calendar(conn)
    finally:
        conn.close()
    assert df[wf_mod.FORECAST_DATE_COL].nunique() == 47
    assert df[wf_mod.FORECAST_DATE_COL].min() == pd.Timestamp("2022-02-25")
    assert df[wf_mod.FORECAST_DATE_COL].max() == pd.Timestamp("2026-01-02")
    assert set(df["ticker"].unique()) == set(config.TICKER_UNIVERSE)
    # No March 2026+ data anywhere in the frozen calendar.
    assert df[wf_mod.TARGET_DATE_COL].max() < pd.Timestamp("2026-03-01")


@pytest.mark.db
@_skip_no_db
def test_real_formation_calendar_is_sequential_non_overlapping():
    from pipeline import db as db_mod
    conn = db_mod.get_connection()
    try:
        conn.autocommit = True
        df = load_formation_calendar(conn)
    finally:
        conn.close()
    assert verify_sequential_non_overlapping_path(df) is True
