"""GET /api/backtest — Historical Evidence: the frozen, official Phase 7
walk-forward portfolio experiment, presented across all 46 non-overlapping
21-session evaluation periods (Phase 8D-2; PRD.md §9's Historical Evidence
surface; BUILD_PLAN.md Phase 7).

This is NOT a new backtest, NOT a strategy-ranking exercise, and NOT a
transaction-cost-adjusted simulation — it is a read-only presentation of
numbers that were already computed and persisted (or, for SPXT, already
frozen as a raw benchmark artifact) BEFORE this slice existed.

Provenance, by field:
  - `realized_return_21`, `turnover`, `max_weight_observed` (portfolio
    strategies): read directly from `risk_metrics` via the official
    `p7bv1_{strategy}_{date}` run_id (`optimizer.persistence.build_run_id`)
    — never recomputed.
  - `growth_of_one`, aggregate summary statistics, and
    `cumulative_return` (portfolio strategies): computed via
    `optimizer.walkforward.aggregate_statistics` /
    `cumulative_compounded_return` — the SAME frozen helpers
    `scripts/run_phase7b_official.py` used to produce the original Phase 7
    report. No second compounding/statistics implementation.
  - SPXT: read via `optimizer.benchmark_spxt.load_spxt_raw` /
    `spxt_total_return` (exact-date lookup only, no nearest-date fallback)
    against `data/raw/spxt_benchmark.csv` — the same official benchmark
    artifact/helper Phase 7B used. SPXT's descriptive statistics use the
    identical arithmetic as `_aggregate_spxt_returns` in
    `scripts/run_phase7b_official.py` (mean/std/median/min/max/hit-rate —
    deliberately no turnover/max-weight, since SPXT is not a weighted
    portfolio), and its cumulative return reuses
    `cumulative_compounded_return` directly (turnover/max_weight fields on
    the dummy `StrategyPeriodResult` are unused by that function).

Sealed-data guard: this module never opens `prices_sealed.csv`,
`macro_sealed.csv`, `prices_extension.csv`, or `macro_extension.csv` — only
`prices_raw` (DB), `predictions` (DB, for the formation/target calendar),
`risk_metrics`/`portfolios` (DB), and `data/raw/spxt_benchmark.csv`. No
March 2026 result can appear here — the persisted Phase 7 experiment and
the SPXT artifact both predate and are structurally independent of the
sealed holdout.

Read-only: no INSERT/UPDATE/DELETE anywhere in this module; nothing here
is persisted.
"""
from __future__ import annotations

from datetime import date as date_type
from typing import List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

import config
import optimizer.benchmark_spxt as spxt_mod
from app.db import get_db
from app.routes.portfolios import STRATEGY_REGISTRY, _official_formation_dates
from app.schemas import BacktestExperiment, BacktestPeriod, BacktestResponse, BacktestSeries, BacktestSummary
from optimizer.persistence import VALID_STRATEGIES, build_run_id
from optimizer.walkforward import (
    FORECAST_DATE_COL,
    TARGET_DATE_COL,
    StrategyPeriodResult,
    aggregate_statistics,
    cumulative_compounded_return,
    verify_sequential_non_overlapping_path,
)

router = APIRouter(prefix="/api", tags=["backtest"])

EXPERIMENT_ID = "p7bv1"
SPXT_KEY = "SPXT"


def _load_formation_target_calendar(conn, dates: List[date_type]) -> pd.DataFrame:
    """The exact (forecast_date, target_date) pairing already frozen in
    `predictions` for the 46 official Phase 7 portfolio dates — never an
    independently regenerated calendar."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT DISTINCT forecast_date, target_date FROM predictions WHERE forecast_date = ANY(%s)",
            (dates,),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=[FORECAST_DATE_COL, TARGET_DATE_COL])
    df[FORECAST_DATE_COL] = pd.to_datetime(df[FORECAST_DATE_COL])
    df[TARGET_DATE_COL] = pd.to_datetime(df[TARGET_DATE_COL])
    return df.sort_values(FORECAST_DATE_COL).reset_index(drop=True)


def _load_strategy_period_results(conn, strategy: str, calendar_df: pd.DataFrame) -> List[StrategyPeriodResult]:
    """One batched query for all 46 formations of this strategy (never 46
    round-trips) — purely a query-efficiency choice, identical values and
    identical missing-row-means-undefined semantics as querying one
    run_id at a time."""
    run_id_by_date = {row[FORECAST_DATE_COL]: build_run_id(EXPERIMENT_ID, strategy, row[FORECAST_DATE_COL]) for _, row in calendar_df.iterrows()}
    run_ids = list(run_id_by_date.values())
    with conn.cursor() as cur:
        cur.execute(
            "SELECT run_id, metric_name, metric_value FROM risk_metrics WHERE run_id = ANY(%s)",
            (run_ids,),
        )
        rows = cur.fetchall()
    metrics_by_run_id: dict = {}
    for run_id, metric_name, metric_value in rows:
        metrics_by_run_id.setdefault(run_id, {})[metric_name] = float(metric_value)

    results = []
    for _, row in calendar_df.iterrows():
        fd = row[FORECAST_DATE_COL]
        run_id = run_id_by_date[fd]
        metrics = metrics_by_run_id.get(run_id, {})
        if "realized_return_21" not in metrics or "max_weight_observed" not in metrics:
            raise HTTPException(
                status_code=500,
                detail=f"Official portfolio {run_id!r} is missing required risk metrics",
            )
        results.append(
            StrategyPeriodResult(
                formation_date=fd,
                realized_return=metrics["realized_return_21"],
                turnover=metrics.get("turnover", float("nan")),  # absent row == genuinely undefined, never 0
                max_weight_observed=metrics["max_weight_observed"],
            )
        )
    return results


def _growth_series(returns: List[float]) -> List[float]:
    """growth_t = growth_(t-1) * (1 + r_t), with growth_0 implicitly 1.0
    (never returned as a period row — see the module/page docs on the
    47-point convention: the frontend prepends (first_formation_date, 1.0)
    itself). Sequential compounding only — never a sum, never
    interpolated, never annualized."""
    growth = []
    level = 1.0
    for r in returns:
        level *= 1.0 + r
        growth.append(level)
    return growth


def _none_if_nan(value: float):
    return None if (value is None or np.isnan(value)) else value


def _portfolio_series(conn, strategy: str, calendar_df: pd.DataFrame) -> BacktestSeries:
    period_results = _load_strategy_period_results(conn, strategy, calendar_df)
    agg = aggregate_statistics(period_results)
    cumulative = cumulative_compounded_return(period_results, calendar_df)
    growth = _growth_series([r.realized_return for r in period_results])

    periods = [
        BacktestPeriod(
            formation_date=r.formation_date.date(),
            target_date=calendar_df.iloc[i][TARGET_DATE_COL].date(),
            realized_return_21=r.realized_return,
            growth_of_one=growth[i],
            turnover=_none_if_nan(r.turnover),
            max_weight_observed=r.max_weight_observed,
        )
        for i, r in enumerate(period_results)
    ]

    strategy_info = STRATEGY_REGISTRY[strategy]
    return BacktestSeries(
        key=strategy,
        label=strategy_info.label,
        kind="portfolio",
        is_optimized=strategy_info.is_optimized,
        covariance_estimator=strategy_info.covariance_estimator,
        max_weight_constraint=config.MAX_WEIGHT if strategy_info.is_optimized else None,
        periods=periods,
        summary=BacktestSummary(
            mean_return_21=agg["mean_return_21"],
            std_return_21=agg["std_return_21"],
            median_return_21=agg["median_return_21"],
            min_return_21=agg["min_return_21"],
            max_return_21=agg["max_return_21"],
            positive_period_rate=agg["hit_rate"],
            cumulative_return=cumulative,
            mean_turnover=_none_if_nan(agg["mean_turnover"]),
            median_turnover=_none_if_nan(agg["median_turnover"]),
            max_turnover=_none_if_nan(agg["max_turnover"]),
            avg_max_weight=agg["avg_max_weight"],
            max_observed_weight=agg["max_observed_weight"],
        ),
    )


def _spxt_series(calendar_df: pd.DataFrame) -> BacktestSeries:
    spxt_raw = spxt_mod.load_spxt_raw()
    spxt_mod.verify_spxt_covers_formation_calendar(spxt_raw, calendar_df)

    rows = list(calendar_df.iterrows())
    returns = [
        spxt_mod.spxt_total_return(spxt_raw, row[FORECAST_DATE_COL], row[TARGET_DATE_COL]) for _, row in rows
    ]
    growth = _growth_series(returns)

    # Reuses the SAME canonical compounding helper as the portfolio
    # strategies — turnover/max_weight_observed are unused by this
    # function (it only reads `.realized_return`) and are set to NaN here
    # purely as required-but-ignored placeholders, never persisted or
    # displayed.
    dummy_period_results = [
        StrategyPeriodResult(
            formation_date=row[FORECAST_DATE_COL], realized_return=r,
            turnover=float("nan"), max_weight_observed=float("nan"),
        )
        for r, (_, row) in zip(returns, rows)
    ]
    cumulative = cumulative_compounded_return(dummy_period_results, calendar_df)

    returns_arr = np.array(returns, dtype=float)
    if not np.isfinite(returns_arr).all():
        raise HTTPException(status_code=500, detail="Non-finite SPXT period return computed")

    periods = [
        BacktestPeriod(
            formation_date=row[FORECAST_DATE_COL].date(),
            target_date=row[TARGET_DATE_COL].date(),
            realized_return_21=float(returns_arr[i]),
            growth_of_one=growth[i],
            turnover=None,
            max_weight_observed=None,
        )
        for i, (_, row) in enumerate(rows)
    ]

    return BacktestSeries(
        key=SPXT_KEY,
        label="SPXT (S&P 500 Total Return)",
        kind="benchmark",
        is_optimized=None,
        covariance_estimator=None,
        max_weight_constraint=None,
        periods=periods,
        summary=BacktestSummary(
            mean_return_21=float(returns_arr.mean()),
            std_return_21=float(returns_arr.std(ddof=1)),
            median_return_21=float(np.median(returns_arr)),
            min_return_21=float(returns_arr.min()),
            max_return_21=float(returns_arr.max()),
            positive_period_rate=float((returns_arr > 0).mean()),
            cumulative_return=cumulative,
            mean_turnover=None,
            median_turnover=None,
            max_turnover=None,
            avg_max_weight=None,
            max_observed_weight=None,
        ),
    )


@router.get("/backtest", response_model=BacktestResponse)
def get_backtest(conn=Depends(get_db)) -> BacktestResponse:
    dates = _official_formation_dates(conn)
    if not dates:
        raise HTTPException(status_code=404, detail="No official Phase 7 portfolio data available")
    if len(dates) != 46:
        raise HTTPException(
            status_code=500,
            detail=f"Expected 46 Phase 7 portfolio-eligible formations, found {len(dates)}",
        )

    calendar_df = _load_formation_target_calendar(conn, dates)
    if len(calendar_df) != len(dates):
        raise HTTPException(status_code=500, detail="Formation/target date calendar is incomplete")
    if not verify_sequential_non_overlapping_path(calendar_df):
        raise HTTPException(
            status_code=500,
            detail="Phase 7 formation/target dates do not form a sequential non-overlapping path",
        )

    series = [_portfolio_series(conn, strategy, calendar_df) for strategy in VALID_STRATEGIES]
    series.append(_spxt_series(calendar_df))

    with conn.cursor() as cur:
        cur.execute("SELECT MAX(date) FROM prices_raw")
        data_through = cur.fetchone()[0]

    return BacktestResponse(
        experiment=BacktestExperiment(
            period_count=len(dates),
            first_formation_date=dates[0],
            last_formation_date=dates[-1],
            horizon_sessions=config.FORECAST_HORIZON,
            covariance_window_sessions=config.COVAR_WINDOW,
            benchmark="SPXT",
            data_through=data_through,
        ),
        series=series,
        source="official_phase7_persisted_experiment",
    )
