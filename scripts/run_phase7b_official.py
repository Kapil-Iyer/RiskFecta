"""
RiskFecta Phase 7B — SINGLE OFFICIAL historical walk-forward execution.

One-off driver script (not part of the tested/importable `optimizer`
package). Authorized for exactly one run under the frozen Phase 7
methodology (COVAR_WINDOW=252, MAX_WEIGHT=0.10, TRI simple returns,
Sigma_21=21*Sigma_session, rf_21=((USGG10YR/100)/252)*21, mu_21=persisted
ensemble_pred, official SPXT benchmark via optimizer.benchmark_spxt).

Inputs, all read-only:
  - `predictions` (persisted ensemble_pred, formation calendar)
  - `prices_raw` (TRI/close, for covariance + realized stock returns)
  - data/raw/macro.csv (USGG10YR, via pipeline.normalize.normalize_macro)
  - data/raw/spxt_benchmark.csv (official SPXT benchmark)

Never opens data/raw/prices_sealed.csv, macro_sealed.csv,
prices_extension.csv, or macro_extension.csv — no glob of data/raw/*.csv
anywhere in this script; every input path is named explicitly.

Writes: `portfolios` and `risk_metrics` rows only, via the collision-safe
optimizer.persistence run-identity machinery. Does not touch any other
table. Run collision is checked for EVERY (strategy, date) run_id before
ANY row is inserted.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import config
from pipeline import db as db_mod
from pipeline import normalize
import optimizer.benchmark_spxt as spxt_mod
import optimizer.covariance as cov_mod
import optimizer.persistence as persist_mod
import optimizer.portfolio as port_mod
import optimizer.walkforward as wf

EXPERIMENT_ID = "p7bv1"
TICKERS = list(config.TICKER_UNIVERSE)

# Expected row shape for the complete official experiment — used by
# `_validate_complete_experiment` as a hard structural check before
# persistence may even be attempted.
#
# NEW (Phase 7B persistence repair): `risk_metrics.metric_value` is
# `NOT NULL` in the frozen schema, but the locked turnover methodology
# requires each strategy's FIRST Phase 7 formation to have an undefined/
# NaN turnover (there is no prior formation to compare against). Rather
# than inventing a sentinel value or making the schema nullable, the
# turnover `risk_metrics` row is OMITTED ENTIRELY for that one
# (strategy, first-date) combination — absence represents "not
# applicable," never a fabricated 0/-1/NaN-as-string. `realized_return_21`
# and `max_weight_observed` are always finite and always persisted, every
# date. So each strategy persists exactly `n_dates` realized_return_21
# rows, `n_dates` max_weight_observed rows, and `n_dates - 1` turnover
# rows (missing only its own first formation).
ALWAYS_PRESENT_METRICS = ("realized_return_21", "max_weight_observed")
CONDITIONAL_METRICS = ("turnover",)  # omitted at each strategy's first formation only


def _build_period_risk_metrics(realized_return_21: float, turnover: float, max_weight_observed: float) -> dict:
    """Build the {metric_name: metric_value} dict for one (strategy, date)
    risk_metrics row group. `turnover` is OMITTED (key absent) when it is
    NaN — never inserted as a NULL/sentinel value. `realized_return_21`/
    `max_weight_observed` are always required to be finite (they always
    are, by construction, for a real formation)."""
    if not np.isfinite(realized_return_21):
        raise ValueError("_build_period_risk_metrics: realized_return_21 must be finite")
    if not np.isfinite(max_weight_observed):
        raise ValueError("_build_period_risk_metrics: max_weight_observed must be finite")
    metrics = {"realized_return_21": realized_return_21, "max_weight_observed": max_weight_observed}
    if not np.isnan(turnover):
        metrics["turnover"] = turnover
    return metrics


# ---------------------------------------------------------------------------
# SPXT is a BENCHMARK, not a constructed portfolio — it has no turnover or
# position-weight concept. Deliberately does NOT reuse the portfolio-
# oriented `wf.aggregate_statistics` (which requires finite turnover/
# max-weight fields that don't apply here); never fabricates them either.
# ---------------------------------------------------------------------------
def _aggregate_spxt_returns(spxt_period_results) -> dict:
    returns = np.array([r.realized_return for r in spxt_period_results], dtype=float)
    if len(returns) == 0:
        raise ValueError("_aggregate_spxt_returns: no periods supplied")
    if not np.isfinite(returns).all():
        raise ValueError("_aggregate_spxt_returns: non-finite realized_return present")
    return {
        "n_periods": int(len(returns)),
        "mean_return_21": float(np.mean(returns)),
        "std_return_21": float(np.std(returns, ddof=1)) if len(returns) > 1 else float("nan"),
        "median_return_21": float(np.median(returns)),
        "min_return_21": float(np.min(returns)),
        "max_return_21": float(np.max(returns)),
        "hit_rate": float(np.mean(returns > 0)),
    }


def _cov_summary(records):
    cond = [r["condition_number"] for r in records]
    min_eig = [r["min_eigenvalue"] for r in records]
    max_eig = [r["max_eigenvalue"] for r in records]
    psd = [r["is_psd"] for r in records]
    pred_vol = [r["predicted_volatility_21"] for r in records]
    return {
        "n_dates": len(records),
        "condition_number_mean": float(np.mean(cond)),
        "condition_number_median": float(np.median(cond)),
        "condition_number_min": float(np.min(cond)),
        "condition_number_max": float(np.max(cond)),
        "min_eigenvalue_mean": float(np.mean(min_eig)),
        "min_eigenvalue_min": float(np.min(min_eig)),
        "max_eigenvalue_mean": float(np.mean(max_eig)),
        "psd_always": bool(all(psd)),
        "psd_violations": int(sum(1 for p in psd if not p)),
        "predicted_vol_21_mean": float(np.mean(pred_vol)),
    }


def _compute_all_aggregates(period_results, spxt_period_results, calendar, cov_diagnostics_by_date):
    """Pure computation — never touches the database. Returns
    (agg, spxt_agg, cov_summary)."""
    agg = {strat: wf.aggregate_statistics(period_results[strat]) for strat in wf.STRATEGIES}
    for strat in wf.STRATEGIES:
        agg[strat]["cumulative_return"] = wf.cumulative_compounded_return(period_results[strat], calendar)

    spxt_agg = _aggregate_spxt_returns(spxt_period_results)
    spxt_agg["cumulative_return"] = wf.cumulative_compounded_return(spxt_period_results, calendar)

    cov_summary = {est: _cov_summary(recs) for est, recs in cov_diagnostics_by_date.items()}
    return agg, spxt_agg, cov_summary


def _validate_complete_experiment(agg, spxt_agg, portfolios_df, risk_metrics_df, formation_dates, tickers) -> None:
    """ALL structural/integrity/constraint checks on the COMPLETE,
    already-computed experiment (NEW — see the Phase 7B driver
    aggregation/persistence safety fix report). Raises loudly on any
    violation. `main()` calls this, and it must succeed, strictly BEFORE
    `_persist_official_experiment` is ever called — there is no other
    call site for `insert_portfolios`/`insert_risk_metrics` in this
    script, so persistence is structurally unreachable until this
    function has returned without raising."""
    n_dates = len(formation_dates)
    n_tickers = len(tickers)

    for strat in wf.STRATEGIES:
        if agg[strat]["n_periods"] != n_dates:
            raise ValueError(
                f"_validate_complete_experiment: {strat} has {agg[strat]['n_periods']} periods, expected {n_dates}"
            )
    if spxt_agg["n_periods"] != n_dates:
        raise ValueError(
            f"_validate_complete_experiment: SPXT has {spxt_agg['n_periods']} periods, expected {n_dates}"
        )

    expected_portfolio_rows = n_dates * len(wf.STRATEGIES) * n_tickers
    if len(portfolios_df) != expected_portfolio_rows:
        raise ValueError(
            f"_validate_complete_experiment: portfolios_df has {len(portfolios_df)} rows, "
            f"expected {expected_portfolio_rows}"
        )
    # Per-metric-type row counts (NEW — Phase 7B persistence repair):
    # ALWAYS_PRESENT_METRICS must appear exactly once per (strategy, date);
    # CONDITIONAL_METRICS (turnover) must appear exactly once per
    # (strategy, date) EXCEPT each strategy's own first formation —
    # n_dates - 1 rows per strategy, never n_dates (that would mean a
    # sentinel/NULL was inserted for the undefined first period).
    expected_n_strategies = len(wf.STRATEGIES)
    for metric_name in ALWAYS_PRESENT_METRICS:
        actual = int((risk_metrics_df["metric_name"] == metric_name).sum())
        expected = n_dates * expected_n_strategies
        if actual != expected:
            raise ValueError(
                f"_validate_complete_experiment: {actual} '{metric_name}' risk_metrics rows, expected {expected}"
            )
    for metric_name in CONDITIONAL_METRICS:
        actual = int((risk_metrics_df["metric_name"] == metric_name).sum())
        expected = (n_dates - 1) * expected_n_strategies
        if actual != expected:
            raise ValueError(
                f"_validate_complete_experiment: {actual} '{metric_name}' risk_metrics rows, expected {expected} "
                f"(exactly one omitted per strategy, at its own first formation)"
            )
    expected_risk_metric_rows = n_dates * expected_n_strategies * len(ALWAYS_PRESENT_METRICS) + \
        (n_dates - 1) * expected_n_strategies * len(CONDITIONAL_METRICS)
    if len(risk_metrics_df) != expected_risk_metric_rows:
        raise ValueError(
            f"_validate_complete_experiment: risk_metrics_df has {len(risk_metrics_df)} rows, "
            f"expected {expected_risk_metric_rows}"
        )

    # Never let a NULL slip through to the schema's NOT NULL constraint —
    # the exact failure that motivated this fix. Catches ANY metric ever
    # going non-finite, not just turnover.
    if risk_metrics_df["metric_value"].isna().any():
        n_bad = int(risk_metrics_df["metric_value"].isna().sum())
        raise ValueError(
            f"_validate_complete_experiment: {n_bad} risk_metrics row(s) have a NULL/NaN metric_value "
            "— schema.sql's risk_metrics.metric_value is NOT NULL; omit the row instead of inserting a NULL"
        )

    for run_id, g in portfolios_df.groupby("run_id"):
        total = float(g["weight"].sum())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"_validate_complete_experiment: run_id {run_id!r} weights sum to {total}, expected 1")
        if (g["weight"] < -1e-6).any():
            raise ValueError(f"_validate_complete_experiment: run_id {run_id!r} has negative weight(s)")
        parsed = persist_mod.parse_run_id(run_id)
        if parsed["strategy"] == "EQUAL_WEIGHT":
            if not np.allclose(g["weight"].to_numpy(dtype=float), 1.0 / n_tickers, atol=1e-9):
                raise ValueError(
                    f"_validate_complete_experiment: run_id {run_id!r} EQUAL_WEIGHT weights are not exactly "
                    f"1/{n_tickers}"
                )
        elif (g["weight"] > config.MAX_WEIGHT + 1e-6).any():
            raise ValueError(f"_validate_complete_experiment: run_id {run_id!r} has a weight exceeding MAX_WEIGHT")


def _persist_official_experiment(conn, portfolios_df, risk_metrics_df):
    """The ONLY call site for `insert_portfolios`/`insert_risk_metrics` in
    this script. Callers MUST have already run
    `_validate_complete_experiment` successfully — this function does not
    re-validate, it only persists.

    ATOMIC (NEW — Phase 7B persistence repair): `portfolios` and
    `risk_metrics` are written inside ONE transaction — BEGIN, insert
    both, COMMIT only if both succeed; ROLLBACK both on any failure.
    Never relies on the connection's ambient autocommit setting (turned
    off for exactly the duration of this call, always restored
    afterward, on success or failure alike) — this is what prevents a
    repeat of the earlier incident where `portfolios` committed
    (autocommit) before `risk_metrics` failed, leaving a half-persisted
    official experiment."""
    prev_autocommit = conn.autocommit
    conn.autocommit = False
    try:
        n_portfolios_inserted = persist_mod.insert_portfolios(conn, portfolios_df)
        n_risk_metrics_inserted = persist_mod.insert_risk_metrics(conn, risk_metrics_df)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.autocommit = prev_autocommit
    return n_portfolios_inserted, n_risk_metrics_inserted


def main(dry_run: bool = False) -> dict:
    """`dry_run=True` runs the identical construction/evaluation/
    aggregation logic against the real data but skips both `insert_*`
    calls — a pre-flight correctness check (shape/finiteness/constraint
    validation only, never a basis for choosing methodology) before the
    single persisting official run. `dry_run=False` is the official,
    persisting execution."""
    conn = db_mod.get_connection()
    conn.autocommit = True

    # -----------------------------------------------------------------
    # PRE-RUN GATE (re-verified inside the script itself, not just by
    # the shell commands run before invoking it)
    # -----------------------------------------------------------------
    # Phase 4-6 forecasting calendar — unchanged, still 47 formations.
    full_calendar = wf.load_formation_calendar(conn, tickers=TICKERS)
    full_formation_dates = sorted(full_calendar[wf.FORECAST_DATE_COL].unique())
    assert len(full_formation_dates) == 47, f"expected 47 forecasting formations, got {len(full_formation_dates)}"
    assert pd.Timestamp(full_formation_dates[0]) == pd.Timestamp("2022-02-25")
    assert pd.Timestamp(full_formation_dates[-1]) == pd.Timestamp("2026-01-02")
    assert wf.verify_sequential_non_overlapping_path(full_calendar) is True

    prices = pd.read_sql(
        "SELECT ticker, date, close, total_return_idx FROM prices_raw ORDER BY ticker, date", conn
    )
    prices["date"] = pd.to_datetime(prices["date"])
    for col in ("close", "total_return_idx"):
        prices[col] = prices[col].astype(float)

    # Phase 7 PORTFOLIO-eligible calendar — excludes 2022-02-25 (insufficient
    # covariance history; see the Phase 7 covariance-eligibility amendment
    # report). Does not modify `full_calendar`/`predictions` in any way.
    calendar, eligibility_report = wf.covariance_eligible_calendar(full_calendar, prices, tickers=TICKERS)
    formation_dates = sorted(calendar[wf.FORECAST_DATE_COL].unique())
    assert eligibility_report.n_total_formations == 47
    assert eligibility_report.n_eligible_formations == 46, (
        f"expected 46 Phase 7 covariance-eligible formations, got {eligibility_report.n_eligible_formations}"
    )
    assert len(formation_dates) == 46
    assert pd.Timestamp(formation_dates[0]) == pd.Timestamp("2022-03-28")
    assert pd.Timestamp(formation_dates[-1]) == pd.Timestamp("2026-01-02")
    assert wf.verify_sequential_non_overlapping_path(calendar) is True
    target_dates = sorted(calendar[wf.TARGET_DATE_COL].unique())
    assert pd.Timestamp(target_dates[-1]) < pd.Timestamp("2026-03-01"), "March 2026+ target date detected"

    wf.verify_t_plus_horizon_alignment(calendar, prices, horizon=config.FORECAST_HORIZON)

    macro = normalize.normalize_macro()  # data/raw/macro.csv only — frozen, authorized
    yield_10y_by_date = macro.set_index("date")["yield_10y"]

    spxt_raw = spxt_mod.load_spxt_raw()  # data/raw/spxt_benchmark.csv only
    spxt_mod.verify_spxt_covers_formation_calendar(spxt_raw, calendar)

    # Collision check for EVERY (strategy, date) run_id, in BOTH tables,
    # before any insert whatsoever.
    all_run_ids = [
        persist_mod.build_run_id(EXPERIMENT_ID, strat, fd)
        for strat in wf.STRATEGIES
        for fd in formation_dates
    ]
    assert len(all_run_ids) == len(set(all_run_ids)), "duplicate run_ids generated — bug"
    for rid in all_run_ids:
        persist_mod.assert_run_id_available(conn, "portfolios", rid)
        persist_mod.assert_run_id_available(conn, "risk_metrics", rid)

    # -----------------------------------------------------------------
    # WALK-FORWARD LOOP
    # -----------------------------------------------------------------
    ensemble_by_date = {
        fd: g.set_index("ticker")[wf.ENSEMBLE_PRED_COL]
        for fd, g in calendar.groupby(wf.FORECAST_DATE_COL)
    }

    period_results = {strat: [] for strat in wf.STRATEGIES}
    spxt_period_results = []
    cov_diagnostics_by_date = {"SAMPLE": [], "LW": []}
    prev_weights = {strat: None for strat in wf.STRATEGIES}

    portfolio_row_frames = []
    risk_metric_row_frames = []

    for fd in formation_dates:
        row0 = calendar[calendar[wf.FORECAST_DATE_COL] == fd].iloc[0]
        td = row0[wf.TARGET_DATE_COL]
        mu = ensemble_by_date[fd]

        if fd not in yield_10y_by_date.index or pd.isna(yield_10y_by_date.loc[fd]):
            raise RuntimeError(f"USGG10YR not available exactly at formation date {fd.date()} — HARD STOP")
        usgg10yr = float(yield_10y_by_date.loc[fd])

        # STAGE A — construction (causal only; construct_portfolios_at
        # itself cuts `prices` at `fd` before anything is estimated).
        try:
            construction = wf.construct_portfolios_at(fd, mu, prices, usgg10yr, tickers=TICKERS)
        except Exception as exc:
            raise RuntimeError(
                f"construct_portfolios_at FAILED at formation_date={fd.date()} "
                f"(dry_run={dry_run}, {len(period_results[wf.STRATEGIES[0]])} prior formation(s) "
                f"already processed this run, zero DB writes issued by this script so far): {exc}"
            ) from exc

        # STAGE B — evaluation. Only now do we touch target_date (T+21).
        realized = wf.realized_stock_returns(prices, TICKERS, fd, td)
        spxt_ret = spxt_mod.spxt_total_return(spxt_raw, fd, td)
        spxt_period_results.append(
            wf.StrategyPeriodResult(formation_date=fd, realized_return=spxt_ret, turnover=float("nan"), max_weight_observed=float("nan"))
        )

        for est_name, sigma21 in construction.sigma_21.items():
            ref_w = construction.weights[f"{est_name}_MINVOL"]
            diag = wf.covariance_diagnostics(sigma21, ref_w)
            diag["formation_date"] = fd
            cov_diagnostics_by_date[est_name].append(diag)

        for strat in wf.STRATEGIES:
            w = construction.weights[strat]
            port_ret = wf.realized_portfolio_return(w, TICKERS, realized)
            tv = wf.turnover(w, prev_weights[strat], TICKERS)
            max_w = float(np.max(w))
            period_results[strat].append(
                wf.StrategyPeriodResult(formation_date=fd, realized_return=port_ret, turnover=tv, max_weight_observed=max_w)
            )
            prev_weights[strat] = w

            # Ex-ante (construction-time) predicted figures for the
            # `portfolios` row; realized/turnover/diagnostics go to
            # `risk_metrics` instead (see module docstring / report for
            # the documented schema mapping).
            if strat == "EQUAL_WEIGHT":
                pred_return, pred_vol, pred_sharpe = float("nan"), float("nan"), float("nan")
            else:
                estimator = "SAMPLE" if strat.startswith("SAMPLE") else "LW"
                sigma21 = construction.sigma_21[estimator]
                pred_return = port_mod.portfolio_expected_return(w, construction.mu_21)
                pred_vol = port_mod.portfolio_volatility(w, sigma21)
                pred_sharpe = port_mod.portfolio_sharpe(w, construction.mu_21, sigma21, construction.rf_21)

            run_id = persist_mod.build_run_id(EXPERIMENT_ID, strat, fd)
            portfolio_row_frames.append(
                persist_mod.build_portfolio_rows(run_id, TICKERS, w, pred_return, pred_vol, pred_sharpe)
            )
            metrics = _build_period_risk_metrics(port_ret, tv, max_w)
            risk_metric_row_frames.append(persist_mod.build_risk_metric_rows(run_id, metrics))

    # -----------------------------------------------------------------
    # REQUIRED ORDER (NEW — see the Phase 7B driver aggregation/
    # persistence safety fix report): aggregate everything, validate the
    # COMPLETE experiment, and ONLY THEN may persistence be attempted.
    # `insert_portfolios`/`insert_risk_metrics` have exactly one call
    # site in this entire script (`_persist_official_experiment`), which
    # is reached only after `_validate_complete_experiment` returns
    # without raising. A failure at any earlier step — construction,
    # evaluation, aggregation, or validation — leaves `portfolios`/
    # `risk_metrics` untouched, because no code path reaches the persist
    # call before that point.
    # -----------------------------------------------------------------
    portfolios_df = pd.concat(portfolio_row_frames, ignore_index=True)
    risk_metrics_df = pd.concat(risk_metric_row_frames, ignore_index=True)

    agg, spxt_agg, cov_summary = _compute_all_aggregates(
        period_results, spxt_period_results, calendar, cov_diagnostics_by_date
    )

    _validate_complete_experiment(agg, spxt_agg, portfolios_df, risk_metrics_df, formation_dates, TICKERS)

    if dry_run:
        n_portfolios_inserted = 0
        n_risk_metrics_inserted = 0
    else:
        n_portfolios_inserted, n_risk_metrics_inserted = _persist_official_experiment(
            conn, portfolios_df, risk_metrics_df
        )

    return {
        "dry_run": dry_run,
        "n_forecasting_formations": eligibility_report.n_total_formations,
        "n_covariance_eligible_formations": eligibility_report.n_eligible_formations,
        "excluded_formations": eligibility_report.excluded,
        "formation_dates": formation_dates,
        "target_dates": target_dates,
        "portfolio_strategy_aggregate": agg,
        "spxt_aggregate": spxt_agg,
        "covariance_summary": cov_summary,
        "n_portfolios_inserted": n_portfolios_inserted,
        "n_risk_metrics_inserted": n_risk_metrics_inserted,
        "n_portfolio_rows_prepared": len(portfolios_df),
        "n_risk_metric_rows_prepared": len(risk_metrics_df),
        "run_ids": all_run_ids,
    }


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    result = main(dry_run=dry_run)
    import json

    if dry_run:
        # STRUCTURAL confirmation only — no realized/SPXT/aggregate
        # financial figures are printed or inspected at this stage (task
        # brief §7: "do NOT print, summarize, rank, interpret, or inspect
        # aggregate realized strategy performance or SPXT performance
        # during the dry-run stage").
        for strat in result["portfolio_strategy_aggregate"]:
            n = result["portfolio_strategy_aggregate"][strat]["n_periods"]
            assert n == 46, f"{strat}: expected 46 periods, got {n}"
        assert result["spxt_aggregate"]["n_periods"] == 46
        assert result["n_portfolios_inserted"] == 0
        assert result["n_risk_metrics_inserted"] == 0
        print(json.dumps(
            {
                "dry_run": True,
                "structural_check": "PASSED",
                "n_forecasting_formations": result["n_forecasting_formations"],
                "n_covariance_eligible_formations": result["n_covariance_eligible_formations"],
                "excluded_formations": result["excluded_formations"],
                "n_formation_dates": len(result["formation_dates"]),
                "first_formation": str(result["formation_dates"][0]),
                "last_formation": str(result["formation_dates"][-1]),
                "last_target": str(result["target_dates"][-1]),
                "n_portfolio_rows_prepared": result["n_portfolio_rows_prepared"],
                "n_risk_metric_rows_prepared": result["n_risk_metric_rows_prepared"],
                "n_portfolios_inserted": result["n_portfolios_inserted"],
                "n_risk_metrics_inserted": result["n_risk_metrics_inserted"],
                "strategies_present": sorted(result["portfolio_strategy_aggregate"].keys()),
                "n_periods_per_strategy": {
                    s: result["portfolio_strategy_aggregate"][s]["n_periods"]
                    for s in result["portfolio_strategy_aggregate"]
                },
            },
            indent=2,
            default=str,
        ))
    else:
        print(json.dumps(
            {
                "dry_run": False,
                "n_forecasting_formations": result["n_forecasting_formations"],
                "n_covariance_eligible_formations": result["n_covariance_eligible_formations"],
                "excluded_formations": result["excluded_formations"],
                "n_formation_dates": len(result["formation_dates"]),
                "first_formation": str(result["formation_dates"][0]),
                "last_formation": str(result["formation_dates"][-1]),
                "last_target": str(result["target_dates"][-1]),
                "n_portfolio_rows_prepared": result["n_portfolio_rows_prepared"],
                "n_risk_metric_rows_prepared": result["n_risk_metric_rows_prepared"],
                "n_portfolios_inserted": result["n_portfolios_inserted"],
                "n_risk_metrics_inserted": result["n_risk_metrics_inserted"],
                "portfolio_strategy_aggregate": result["portfolio_strategy_aggregate"],
                "spxt_aggregate": result["spxt_aggregate"],
                "covariance_summary": result["covariance_summary"],
            },
            indent=2,
            default=str,
        ))
