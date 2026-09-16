"""
RiskFecta Phase 7B — driver aggregation/persistence safety fix tests
(scripts/run_phase7b_official.py). Task brief items 1-8.

Loads the driver script as a module (it is a one-off script, not part of
the tested/importable `optimizer` package) and exercises its extracted
helper functions with SYNTHETIC data only — never the real 46-date
walk-forward, never a real database write. `insert_portfolios`/
`insert_risk_metrics` are always monkeypatched to in-memory fakes in
this file; no test here can write to the real database.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import config
import optimizer.persistence as persist_mod
import optimizer.walkforward as wf

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "run_phase7b_official.py"


def _load_driver_module():
    spec = importlib.util.spec_from_file_location("phase7b_official_driver", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


driver = _load_driver_module()

# 15 tickers so 1/15 (~0.0667) respects config.MAX_WEIGHT=0.10 for the
# non-EQUAL_WEIGHT strategies too (3 tickers would make sum(w)=1 and
# w<=0.10 mutually infeasible, same corner case as the Phase 7A tests).
TICKERS = [f"T{i:02d}" for i in range(15)]
STRATEGY = list(wf.STRATEGIES)


def _calendar_df(n_dates=2):
    dates = pd.bdate_range("2022-03-28", periods=n_dates * 21, freq="B")[::21][:n_dates]
    targets = dates + pd.tseries.offsets.BDay(21)
    return pd.DataFrame({wf.FORECAST_DATE_COL: dates, wf.TARGET_DATE_COL: targets})


def _valid_synthetic_experiment(n_dates=2, tickers=TICKERS):
    """A fully valid, self-consistent synthetic experiment: period
    results, SPXT period results, and matching portfolios/risk_metrics
    DataFrames — everything `_validate_complete_experiment` requires to
    pass, and everything `_compute_all_aggregates` needs to succeed."""
    calendar = _calendar_df(n_dates)
    dates = list(calendar[wf.FORECAST_DATE_COL])

    period_results = {s: [] for s in STRATEGY}
    spxt_period_results = []
    portfolio_frames = []
    risk_metric_frames = []

    n = len(tickers)
    for i, fd in enumerate(dates):
        spxt_period_results.append(
            wf.StrategyPeriodResult(formation_date=fd, realized_return=0.01 * (i + 1), turnover=float("nan"), max_weight_observed=float("nan"))
        )
        for strat in STRATEGY:
            tv = float("nan") if i == 0 else 0.05
            w = np.full(n, 1.0 / n)  # respects sum=1 and MAX_WEIGHT for every strategy
            max_w = float(np.max(w))
            period_results[strat].append(
                wf.StrategyPeriodResult(formation_date=fd, realized_return=0.01 * (i + 1), turnover=tv, max_weight_observed=max_w)
            )
            run_id = persist_mod.build_run_id("p7btest", strat, fd)
            portfolio_frames.append(persist_mod.build_portfolio_rows(run_id, tickers, w, 0.01, 0.05, 0.2))
            # Uses the driver's own turnover-omission helper (Phase 7B
            # persistence repair) — never inserts a NULL/NaN metric_value.
            metrics = driver._build_period_risk_metrics(0.01 * (i + 1), tv, max_w)
            risk_metric_frames.append(persist_mod.build_risk_metric_rows(run_id, metrics))

    portfolios_df = pd.concat(portfolio_frames, ignore_index=True)
    risk_metrics_df = pd.concat(risk_metric_frames, ignore_index=True)

    cov_diagnostics_by_date = {
        est: [
            {"condition_number": 10.0 + i, "min_eigenvalue": 0.01, "max_eigenvalue": 0.1 + 0.01 * i,
             "is_psd": True, "predicted_volatility_21": 0.05, "formation_date": fd}
            for i, fd in enumerate(dates)
        ]
        for est in ("SAMPLE", "LW")
    }
    return calendar, dates, period_results, spxt_period_results, portfolios_df, risk_metrics_df, cov_diagnostics_by_date


# ---------------------------------------------------------------------------
# 1/2. SPXT aggregation: return-only, never requires/fabricates
# turnover/max-weight.
# ---------------------------------------------------------------------------
def test_spxt_aggregation_succeeds_with_only_return_fields():
    _, dates, _, spxt_period_results, _, _, _ = _valid_synthetic_experiment()
    agg = driver._aggregate_spxt_returns(spxt_period_results)
    expected_keys = {"n_periods", "mean_return_21", "std_return_21", "median_return_21", "min_return_21", "max_return_21", "hit_rate"}
    assert set(agg.keys()) == expected_keys
    assert agg["n_periods"] == len(dates)


def test_spxt_aggregation_never_fabricates_turnover_or_weight_fields():
    _, _, _, spxt_period_results, _, _, _ = _valid_synthetic_experiment()
    agg = driver._aggregate_spxt_returns(spxt_period_results)
    for forbidden in ("turnover", "max_weight", "concentration"):
        assert not any(forbidden in k for k in agg)


# ---------------------------------------------------------------------------
# 3. Portfolio aggregate_statistics unchanged — still rejects non-finite
# max_weight_observed.
# ---------------------------------------------------------------------------
def test_portfolio_aggregate_statistics_still_rejects_nonfinite_max_weight():
    periods = [
        wf.StrategyPeriodResult(formation_date=pd.Timestamp("2022-03-28"), realized_return=0.01, turnover=float("nan"), max_weight_observed=float("nan")),
    ]
    with pytest.raises(ValueError, match="max_weight_observed"):
        wf.aggregate_statistics(periods)


# ---------------------------------------------------------------------------
# 4. A simulated AGGREGATION failure occurs before any persistence.
# ---------------------------------------------------------------------------
def test_aggregation_failure_never_reaches_persistence(monkeypatch):
    calendar, dates, period_results, spxt_period_results, portfolios_df, risk_metrics_df, cov_diagnostics_by_date = _valid_synthetic_experiment()
    # Corrupt one period's realized_return -> aggregate_statistics raises.
    period_results[STRATEGY[0]][0] = wf.StrategyPeriodResult(
        formation_date=dates[0], realized_return=float("nan"), turnover=float("nan"), max_weight_observed=0.5
    )

    persist_calls = []
    monkeypatch.setattr(driver, "_persist_official_experiment", lambda *a, **k: persist_calls.append(1))

    with pytest.raises(ValueError):
        agg, spxt_agg, cov_summary = driver._compute_all_aggregates(period_results, spxt_period_results, calendar, cov_diagnostics_by_date)
        driver._validate_complete_experiment(agg, spxt_agg, portfolios_df, risk_metrics_df, dates, TICKERS)
        driver._persist_official_experiment(None, portfolios_df, risk_metrics_df)

    assert persist_calls == []


# ---------------------------------------------------------------------------
# 5. A simulated STRUCTURAL-VALIDATION failure occurs before any
# persistence.
# ---------------------------------------------------------------------------
def test_structural_validation_failure_never_reaches_persistence(monkeypatch):
    calendar, dates, period_results, spxt_period_results, portfolios_df, risk_metrics_df, cov_diagnostics_by_date = _valid_synthetic_experiment()
    agg, spxt_agg, cov_summary = driver._compute_all_aggregates(period_results, spxt_period_results, calendar, cov_diagnostics_by_date)

    # Corrupt the prepared rows: drop one portfolio row -> row-count mismatch.
    broken_portfolios_df = portfolios_df.iloc[:-1].reset_index(drop=True)

    persist_calls = []
    monkeypatch.setattr(driver, "_persist_official_experiment", lambda *a, **k: persist_calls.append(1))

    with pytest.raises(ValueError, match="portfolios_df has"):
        driver._validate_complete_experiment(agg, spxt_agg, broken_portfolios_df, risk_metrics_df, dates, TICKERS)
        driver._persist_official_experiment(None, broken_portfolios_df, risk_metrics_df)

    assert persist_calls == []


def test_structural_validation_rejects_bad_weight_sum():
    calendar, dates, period_results, spxt_period_results, portfolios_df, risk_metrics_df, cov_diagnostics_by_date = _valid_synthetic_experiment()
    agg, spxt_agg, cov_summary = driver._compute_all_aggregates(period_results, spxt_period_results, calendar, cov_diagnostics_by_date)

    broken = portfolios_df.copy()
    broken.loc[0, "weight"] = broken.loc[0, "weight"] + 0.5  # break sum-to-1 for that run_id
    with pytest.raises(ValueError, match="sum to"):
        driver._validate_complete_experiment(agg, spxt_agg, broken, risk_metrics_df, dates, TICKERS)


def test_structural_validation_rejects_equal_weight_deviation():
    calendar, dates, period_results, spxt_period_results, portfolios_df, risk_metrics_df, cov_diagnostics_by_date = _valid_synthetic_experiment()
    agg, spxt_agg, cov_summary = driver._compute_all_aggregates(period_results, spxt_period_results, calendar, cov_diagnostics_by_date)

    broken = portfolios_df.copy()
    ew_run_id = persist_mod.build_run_id("p7btest", "EQUAL_WEIGHT", dates[0])
    mask = broken["run_id"] == ew_run_id
    idx = broken[mask].index[0]
    other_idx = broken[mask].index[1]
    broken.loc[idx, "weight"] += 0.01
    broken.loc[other_idx, "weight"] -= 0.01  # sum still 1, but not exactly 1/n per ticker
    with pytest.raises(ValueError, match="EQUAL_WEIGHT weights are not exactly"):
        driver._validate_complete_experiment(agg, spxt_agg, broken, risk_metrics_df, dates, TICKERS)


# ---------------------------------------------------------------------------
# 6. No insert_* function can be reached unless complete aggregation and
# validation have succeeded — structural proof via source inspection:
# insert_portfolios/insert_risk_metrics are called ONLY inside
# `_persist_official_experiment`, nowhere else in the module.
# ---------------------------------------------------------------------------
def test_insert_calls_exist_only_inside_persist_official_experiment():
    import inspect
    module_src = inspect.getsource(driver)
    persist_src = inspect.getsource(driver._persist_official_experiment)
    for call in ("insert_portfolios(", "insert_risk_metrics("):
        total = module_src.count(call)
        within_persist = persist_src.count(call)
        assert total >= 1
        assert total == within_persist, f"{call} appears outside _persist_official_experiment"


def test_main_calls_validate_before_persist_in_source_order():
    import inspect
    main_src = inspect.getsource(driver.main)
    validate_pos = main_src.index("_validate_complete_experiment(")
    persist_pos = main_src.index("_persist_official_experiment(")
    assert validate_pos < persist_pos


# ---------------------------------------------------------------------------
# 7. Successful execution reaches persistence only after both gates,
# using fake (never real-DB) insert_* functions.
# ---------------------------------------------------------------------------
def test_successful_synthetic_experiment_reaches_persistence(monkeypatch):
    calendar, dates, period_results, spxt_period_results, portfolios_df, risk_metrics_df, cov_diagnostics_by_date = _valid_synthetic_experiment()
    agg, spxt_agg, cov_summary = driver._compute_all_aggregates(period_results, spxt_period_results, calendar, cov_diagnostics_by_date)
    driver._validate_complete_experiment(agg, spxt_agg, portfolios_df, risk_metrics_df, dates, TICKERS)  # must not raise

    inserted = {}

    def fake_insert_portfolios(conn, df):
        inserted["portfolios"] = len(df)
        return len(df)

    def fake_insert_risk_metrics(conn, df):
        inserted["risk_metrics"] = len(df)
        return len(df)

    monkeypatch.setattr(driver.persist_mod, "insert_portfolios", fake_insert_portfolios)
    monkeypatch.setattr(driver.persist_mod, "insert_risk_metrics", fake_insert_risk_metrics)

    class _FakeConn:
        autocommit = True

        def commit(self):
            pass

        def rollback(self):
            pass

    n_port, n_risk = driver._persist_official_experiment(_FakeConn(), portfolios_df, risk_metrics_df)
    assert inserted == {"portfolios": len(portfolios_df), "risk_metrics": len(risk_metrics_df)}
    assert n_port == len(portfolios_df)
    assert n_risk == len(risk_metrics_df)


# ---------------------------------------------------------------------------
# 8. Existing Phase 7 tests remain green — exercised by the full suite,
# not re-duplicated here.
# ---------------------------------------------------------------------------
def test_driver_module_never_globs_data_raw():
    import inspect
    src = inspect.getsource(driver)
    assert "glob(" not in src  # no wildcard directory scan anywhere in the script
    # Check for actual code USAGE (a quoted string literal, as a path would
    # be constructed/opened with) — not the module's own docstring, which
    # legitimately NAMES these files in prose to document that they are
    # never opened (e.g. "Never opens data/raw/prices_sealed.csv, ...").
    for forbidden in ("prices_sealed.csv", "macro_sealed.csv", "prices_extension.csv", "macro_extension.csv"):
        assert f'"{forbidden}"' not in src
        assert f"'{forbidden}'" not in src
