"""GET /api/frontier — Efficient Frontier: constrained mean-variance
reconstruction at a historical Phase 7 formation date (Phase 8C, final
sub-slice; ML_SPEC.md §23-27; BUILD_PLAN.md Phase 7).

UNLIKE Portfolio Construction (app/routes/portfolios.py), the dense
frontier itself was never persisted in Phase 7 — only five discrete
strategies were. This route therefore RECONSTRUCTS the exact
formation-time (mu_21, Sigma_21, rf_21) context using the SAME frozen
helper call sequence `scripts/run_phase7b_official.py` and
`optimizer.walkforward.construct_portfolios_at` use
(`optimizer.covariance.covariance_return_window` /
`sample_covariance_session` / `ledoit_wolf_covariance_session` /
`scale_covariance_to_horizon` / `validate_covariance_matrix`,
`optimizer.portfolio.align_mu_sigma` / `rf_horizon`) — never a second,
independently-invented covariance/rf implementation. It then sweeps
`optimizer.portfolio.efficient_frontier` (GENERIC machinery — that module
never chooses a target-return grid itself) over a caller-chosen grid
whose DENSITY is a UI/presentation parameter, never a financial
hyperparameter.

Data-access boundary: prices come from `prices_raw` (the same table
`prices_raw.csv` populated — historical Bloomberg data through
2026-02-27), cut at `date <= formation_date` in SQL before a single row
leaves the database. The risk-free input reads the already-persisted
`features.yield_10y` column (Phase 3 loaded it from `data/raw/macro.csv`'s
USGG10YR field into Postgres for every (ticker, date) row — this route
reads that same persisted value back rather than re-opening the
gitignored, deployment-unavailable CSV at request time; see
`app/routes/frontier.py::_load_rf_21` and the Phase 8F production-
remediation report for the exact reconciliation proving this is
numerically identical to the CSV for every official formation date, never
a second/independent risk-free source). No sealed/extension file is
opened anywhere in this module.

Causality: nothing here calls `optimizer.walkforward.realized_stock_returns`
/`realized_portfolio_return` (Stage B) or `optimizer.benchmark_spxt`
(the official market benchmark, evaluation-only) — this module has no
import of either. SPXT is deliberately never placed on this ex-ante
chart (it has no formation-time expected-return/volatility coordinate
under this framework; see the frontend page's disclosure).

Official markers (Min-Vol, Max-Sharpe) reuse the OFFICIAL persisted
Phase 7 portfolio weights/metrics from `portfolios` — the identical
read-directly philosophy as app/routes/portfolios.py (never reconstruct
an already-persisted construction-time metric). Equal Weight is the
exact conceptual 1/50 benchmark; its expected-return/volatility are
recomputed from this request's reconstructed mu_21/Sigma_21 so its
coordinates sit in the same system as the drawn curve — it is never
implied to lie on the frontier.

Read-only: no INSERT/UPDATE/DELETE anywhere in this module; no frontier
row is ever persisted (schema/table additions were explicitly out of
scope for this slice).
"""
from __future__ import annotations

from datetime import date as date_type
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

import config
from app.db import get_db
from app.routes.portfolios import _official_formation_dates
from app.schemas import FrontierMarker, FrontierMarkers, FrontierPoint, FrontierResponse
from optimizer.covariance import (
    covariance_return_window,
    ledoit_wolf_covariance_session,
    sample_covariance_session,
    scale_covariance_to_horizon,
    validate_covariance_matrix,
)
from optimizer.persistence import build_run_id
from optimizer.portfolio import (
    _achievable_return_range,
    align_mu_sigma,
    efficient_frontier,
    equal_weight_benchmark,
    minimum_volatility_weights,
    portfolio_expected_return,
    portfolio_volatility,
    rf_horizon,
)

router = APIRouter(prefix="/api", tags=["frontier"])

EXPERIMENT_ID = "p7bv1"
TICKERS = list(config.TICKER_UNIVERSE)

# UI-density parameter only (§4/§5 of the task brief) — never a financial
# hyperparameter. `optimizer.portfolio.efficient_frontier` itself never
# chooses or freezes a grid; this route owns that choice entirely.
N_FRONTIER_POINTS = 41

_ESTIMATOR_LABELS = {"SAMPLE": "Sample", "LW": "Ledoit-Wolf"}
VALID_COVARIANCE_ESTIMATORS = ("SAMPLE", "LW")


def _load_mu(conn, formation_date: date_type) -> pd.Series:
    """Persisted `ensemble_pred` as of `formation_date` only — the same
    frozen expected-return signal Portfolio Construction reads, never a
    second forecast source."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ticker, ensemble_pred FROM predictions WHERE forecast_date = %s ORDER BY ticker",
            (formation_date,),
        )
        rows = cur.fetchall()
    if not rows:
        raise HTTPException(status_code=404, detail=f"No persisted predictions for formation_date={formation_date}")
    return pd.Series({ticker: float(pred) for ticker, pred in rows})


def _load_causal_prices(conn, formation_date: date_type) -> pd.DataFrame:
    """Causal cutoff enforced HERE, in SQL, before a single row leaves the
    database — defense in depth alongside `covariance_return_window`'s own
    `as_of_date` filter, never relying on the python-side filter alone."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ticker, date, close, total_return_idx FROM prices_raw WHERE date <= %s ORDER BY ticker, date",
            (formation_date,),
        )
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=["ticker", "date", "close", "total_return_idx"])
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    df["total_return_idx"] = df["total_return_idx"].astype(float)
    return df


def _load_rf_21(conn, formation_date: date_type) -> float:
    """USGG10YR at `formation_date`, read from the already-persisted
    `features.yield_10y` column rather than re-opening `data/raw/macro.csv`
    at request time.

    `features.yield_10y` was itself loaded from that exact CSV column at
    Phase 3 ingestion time (`pipeline.normalize.normalize_macro`) — this is
    the same frozen macro source, not a second/independent one, just read
    from Postgres (always available in every deployment) instead of a
    gitignored local file (absent on Render — the CSV is intentionally
    never committed; see BUILD_PLAN.md's raw-data licensing note). Every
    ticker shares one macro-level value per date, so any of the 50 rows for
    `formation_date` gives the same answer; `DISTINCT` both fetches it and
    defensively verifies that invariant instead of assuming it."""
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT yield_10y FROM features WHERE date = %s", (formation_date,))
        rows = cur.fetchall()
    if len(rows) != 1 or rows[0][0] is None:
        raise HTTPException(
            status_code=500,
            detail=f"USGG10YR not available exactly at formation_date={formation_date}",
        )
    return rf_horizon(float(rows[0][0]))


def reconstruct_mu_sigma_rf(conn, formation_date: date_type) -> Tuple[np.ndarray, Dict[str, np.ndarray], float]:
    """Formation-time (mu_21, {"SAMPLE"/"LW": Sigma_21}, rf_21) —
    reconstructed via the IDENTICAL helper call sequence as
    `optimizer.walkforward.construct_portfolios_at`. Exposed at module
    level (not nested) so tests can exercise it directly against a live
    causal-boundary/reconciliation check without going through HTTP."""
    mu = _load_mu(conn, formation_date)
    prices = _load_causal_prices(conn, formation_date)

    returns_window = covariance_return_window(prices, TICKERS, as_of_date=formation_date)
    sigma_sample_21 = scale_covariance_to_horizon(sample_covariance_session(returns_window))
    sigma_lw_21 = scale_covariance_to_horizon(ledoit_wolf_covariance_session(returns_window))
    validate_covariance_matrix(sigma_sample_21, TICKERS)
    validate_covariance_matrix(sigma_lw_21, TICKERS)

    mu_arr_sample, sigma_sample_arr = align_mu_sigma(mu, sigma_sample_21, TICKERS)
    mu_arr_lw, sigma_lw_arr = align_mu_sigma(mu, sigma_lw_21, TICKERS)
    if not np.allclose(mu_arr_sample, mu_arr_lw):
        raise AssertionError("reconstruct_mu_sigma_rf: mu differs between estimator branches — alignment bug")
    mu_arr = mu_arr_sample

    rf_21 = _load_rf_21(conn, formation_date)
    return mu_arr, {"SAMPLE": sigma_sample_arr, "LW": sigma_lw_arr}, rf_21


def _sharpe_or_none(expected_return: float, volatility: float, rf_21: float) -> Optional[float]:
    if volatility <= 0:
        return None
    return float((expected_return - rf_21) / volatility)


def _official_marker(conn, formation_date: date_type, strategy: str, label: str) -> FrontierMarker:
    """Min-Vol/Max-Sharpe markers: read the OFFICIAL persisted Phase 7
    target_return/portfolio_vol/sharpe_ratio DIRECTLY — never reconstructed
    (identical philosophy to app/routes/portfolios.py)."""
    run_id = build_run_id(EXPERIMENT_ID, strategy, formation_date)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT target_return, portfolio_vol, sharpe_ratio FROM portfolios WHERE run_id = %s LIMIT 1",
            (run_id,),
        )
        row = cur.fetchone()
    if row is None or row[0] is None:
        raise HTTPException(
            status_code=500,
            detail=f"Official portfolio {run_id!r} is missing required construction metrics",
        )
    target_return, portfolio_vol, sharpe_ratio = row
    return FrontierMarker(
        label=label,
        expected_return_21=float(target_return),
        volatility_21=float(portfolio_vol),
        sharpe_21=float(sharpe_ratio) if sharpe_ratio is not None else None,
        provenance="official_phase7_persisted",
    )


def _equal_weight_marker(mu_arr: np.ndarray, sigma_arr: np.ndarray, rf_21: float) -> FrontierMarker:
    """Equal Weight is a fixed 1/50 BENCHMARK, never optimized and never
    implied to lie on the efficient frontier — its coordinates are
    recomputed from THIS request's reconstructed mu_21/Sigma_21 so they sit
    in the same coordinate system as the drawn curve."""
    w = equal_weight_benchmark(len(TICKERS))
    expected_return = portfolio_expected_return(w, mu_arr)
    volatility = portfolio_volatility(w, sigma_arr)
    return FrontierMarker(
        label="Equal Weight",
        expected_return_21=expected_return,
        volatility_21=volatility,
        sharpe_21=_sharpe_or_none(expected_return, volatility, rf_21),
        provenance="reconstructed_benchmark",
    )


def _frontier_points(mu_arr: np.ndarray, sigma_arr: np.ndarray, rf_21: float) -> List[FrontierPoint]:
    """A single infeasible/failed target is honestly OMITTED (never
    fabricated/interpolated) — see `optimizer.portfolio.efficient_frontier`'s
    own per-point `feasible` contract.

    Sweeping the FULL achievable target-return range `[lo, hi]` produces
    the whole minimum-variance parabola, which includes a dominated
    (inefficient) lower branch — for any point below the global
    minimum-variance return, some OTHER feasible portfolio has the same
    or lower volatility with a strictly higher return. A page titled
    "Efficient Frontier" must never present those dominated points as
    part of the frontier, so the sweep starts at the global minimum-
    variance return (computed via the same frozen
    `minimum_volatility_weights`, never a second implementation) rather
    than at `lo`. This does not change how any point is computed, only
    which end of the achievable range the caller-chosen grid starts
    from — still entirely a UI/presentation decision, never a
    methodology change."""
    lo, hi = _achievable_return_range(mu_arr, config.MAX_WEIGHT)
    min_vol_w = minimum_volatility_weights(sigma_arr, max_weight=config.MAX_WEIGHT)
    min_vol_return = portfolio_expected_return(min_vol_w, mu_arr)
    sweep_lo = min(max(min_vol_return, lo), hi)  # clamp defensively; must stay within the achievable range
    target_grid = np.linspace(sweep_lo, hi, N_FRONTIER_POINTS)
    results = efficient_frontier(mu_arr, sigma_arr, target_grid, max_weight=config.MAX_WEIGHT)

    points: List[FrontierPoint] = []
    for r in results:
        if not r["feasible"]:
            continue
        vol = float(r["volatility"])
        ret = float(r["target_return"])
        if not (np.isfinite(vol) and np.isfinite(ret)) or vol < 0:
            continue
        points.append(FrontierPoint(expected_return_21=ret, volatility_21=vol, sharpe_21=_sharpe_or_none(ret, vol, rf_21)))
    return points


@router.get("/frontier", response_model=FrontierResponse)
def get_frontier(
    formation_date: Optional[date_type] = Query(
        None,
        description="Phase 7 portfolio-eligible formation date (46 available, 2022-03-28 to 2026-01-02). Defaults to the latest.",
    ),
    covariance: str = Query("LW", description="One of: " + ", ".join(VALID_COVARIANCE_ESTIMATORS)),
    conn=Depends(get_db),
) -> FrontierResponse:
    if covariance not in VALID_COVARIANCE_ESTIMATORS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown covariance estimator {covariance!r}; expected one of {VALID_COVARIANCE_ESTIMATORS}",
        )

    eligible_dates = _official_formation_dates(conn)
    if formation_date is None:
        if not eligible_dates:
            raise HTTPException(status_code=404, detail="No official Phase 7 portfolio data available")
        formation_date = eligible_dates[-1]
    elif formation_date not in eligible_dates:
        raise HTTPException(
            status_code=404,
            detail=f"formation_date={formation_date} is not a Phase 7 portfolio-eligible formation",
        )

    mu_arr, sigma_by_estimator, rf_21 = reconstruct_mu_sigma_rf(conn, formation_date)
    sigma_arr = sigma_by_estimator[covariance]

    points = _frontier_points(mu_arr, sigma_arr, rf_21)

    min_vol_strategy = f"{covariance}_MINVOL"
    max_sharpe_strategy = f"{covariance}_MAXSHARPE"
    markers = FrontierMarkers(
        min_vol=_official_marker(conn, formation_date, min_vol_strategy, "Min-Vol (official)"),
        max_sharpe=_official_marker(conn, formation_date, max_sharpe_strategy, "Max-Sharpe (official)"),
        equal_weight=_equal_weight_marker(mu_arr, sigma_arr, rf_21),
    )

    return FrontierResponse(
        formation_date=formation_date,
        covariance_estimator=_ESTIMATOR_LABELS[covariance],
        forecast_horizon_sessions=config.FORECAST_HORIZON,
        covariance_window_sessions=config.COVAR_WINDOW,
        max_weight_constraint=config.MAX_WEIGHT,
        risk_free_rate_21=rf_21,
        points=points,
        markers=markers,
        source="reconstructed_from_frozen_phase7_methodology",
    )
