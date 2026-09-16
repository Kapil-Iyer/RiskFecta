"""
RiskFecta Phase 7B — historical walk-forward experiment driver
(optimizer/walkforward.py). IMPLEMENTATION ONLY.

Every function here is exercised against synthetic fixtures
(tests/test_optimizer_walkforward.py) and, for the formation-calendar
loader/validators, read-only SELECTs against the live database (never a
write) — until the real historical experiment is explicitly authorized
(see the Phase 7B / Phase 7 covariance-eligibility reports). The 47-date
figure below and elsewhere in this module refers to the Phase 4-6
FORECASTING calendar (`load_formation_calendar`) — distinct from the
Phase 7 PORTFOLIO experiment's 46-date covariance-eligible calendar
(`covariance_eligible_calendar`); see the dedicated docstring section
below for the exact boundary and which date is excluded.

================================================================================
STAGE A / STAGE B SEPARATION (task brief §10)
================================================================================
`construct_portfolios_at` (STAGE A) accepts only: a formation date, a
caller-supplied `mu` (persisted `ensemble_pred` as of that date), causal
`prices` (gated to <= formation date by `optimizer.covariance`), and a
caller-supplied `usgg10yr` (as of that date). It returns frozen weights.

`realized_stock_returns` / `realized_portfolio_return` (STAGE B) accept
already-frozen weights as plain numbers and separately-supplied realized
TRI data — they never recompute or adjust weights, and Stage A never
calls into Stage B. There is no code path from Stage B back into Stage A.

================================================================================
PHASE 7 COVARIANCE-ELIGIBILITY BOUNDARY (see the Phase 7 covariance-
eligibility amendment report)
================================================================================
A frozen Phase 4-6 forecasting formation is not automatically Phase 7
PORTFOLIO-eligible: LOCK A below needs 253 causal TRI levels (252
one-session returns), which is ONE MORE session than TRAIN_WINDOW=252
needed to make that date selectable as a forecasting formation in the
first place (a return is a first difference of levels). This was
discovered on the real 47-date calendar during Phase 7B's pre-flight dry
run — the FIRST forecasting formation, 2022-02-25, sits at global session
index 251 (252 TRI levels from the 2021-03-01 start of price history),
one session short of LOCK A's 253-level requirement. No portfolio weight
was ever constructed at that point, and this was resolved by explicit
planner decision BEFORE any portfolio/realized/SPXT result was observed:
`covariance_eligible_calendar` below excludes 2022-02-25 from the Phase 7
PORTFOLIO experiment (46 eligible formations) WITHOUT touching the
underlying Phase 4-6 `predictions` calendar (still 47 forecasting
formations) at all — it filters a copy, never mutates or deletes
anything upstream.

================================================================================
LOCKS CARRIED FORWARD FROM PHASE 7A (never altered here)
================================================================================
LOCK A: config.COVAR_WINDOW = 252 one-session returns from 253 causal TRI
        levels (optimizer.covariance.covariance_return_window).
LOCK B: simple one-session TRI returns; PX_LAST is the session gate, never
        TRI (optimizer.covariance.build_tri_wide/session_returns).
LOCK C: Sigma_21 = 21 * Sigma_session; rf_21 = ((USGG10YR/100)/252)*21;
        mu_21 = persisted ensemble_pred, unchanged.
LOCK D: MAX_WEIGHT = config.MAX_WEIGHT = 0.10 for the four OPTIMIZED
        strategies only — EQUAL_WEIGHT is never routed through it (task
        brief §7: "Do not run it through MAX_WEIGHT logic").

================================================================================
SPX / SPXT — SEE THE PHASE 7B SPXT AMENDMENT REPORT
================================================================================
The official Phase 7 market benchmark is now the S&P 500 Total Return
Index (SPXT) — `optimizer.benchmark_spxt.spxt_total_return`, sourced from
`data/raw/spxt_benchmark.csv` (Bloomberg `SPXT Index`, field `PX_LAST`),
obtained specifically for Phase 7 before any Phase 7 execution. See
ML_SPEC.md §27 (amended) and the Phase 7B SPXT amendment report.

`spx_price_return_diagnostic` below is now DEPRECATED: it predates the
SPXT artifact and was never the official benchmark (it computes a PRICE
return, dividends excluded, from the pre-existing `SPX_PX_LAST` macro
series). It is retained only for backward compatibility with earlier
Phase 7B tests/callers, remains unmistakably labeled NON-OFFICIAL in its
own docstring, and must never be used as, or confused with, the official
SPXT benchmark. STRATEGIES intentionally excludes any SPX/SPXT entry —
neither is one of the five official portfolio strategies; both are
evaluation-only market benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import config
from optimizer.covariance import (
    build_tri_wide,
    covariance_return_window,
    ledoit_wolf_covariance_session,
    sample_covariance_session,
    scale_covariance_to_horizon,
    validate_covariance_matrix,
)
from optimizer.portfolio import (
    align_mu_sigma,
    equal_weight_benchmark,
    maximum_sharpe_weights,
    minimum_volatility_weights,
    portfolio_volatility,
    rf_horizon,
)
from pipeline.folds import build_global_calendar
from pipeline.sessions import DATE_COL, TICKER_COL

FORECAST_DATE_COL = "forecast_date"
TARGET_DATE_COL = "target_date"
ENSEMBLE_PRED_COL = "ensemble_pred"

# The official Phase 7B strategy set (task brief §7) — nothing else.
OPTIMIZED_STRATEGIES = ("SAMPLE_MINVOL", "SAMPLE_MAXSHARPE", "LW_MINVOL", "LW_MAXSHARPE")
STRATEGIES = OPTIMIZED_STRATEGIES + ("EQUAL_WEIGHT",)


# ---------------------------------------------------------------------------
# §3 — formation calendar: EXACTLY the already-persisted 47 dates, never an
# independently generated calendar.
# ---------------------------------------------------------------------------
CANONICAL_ENSEMBLE_SQL = """
SELECT ticker, forecast_date, target_date, ensemble_pred
FROM predictions
ORDER BY forecast_date, ticker
"""


def load_formation_calendar(conn, tickers: Sequence[str] = None) -> pd.DataFrame:
    """Read-only load of the exact persisted (ticker, forecast_date,
    target_date, ensemble_pred) rows from `predictions`. Validates the
    frozen structural invariants (§3) before returning — never silently
    drops a bad row."""
    tickers = list(config.TICKER_UNIVERSE) if tickers is None else list(tickers)
    df = pd.read_sql(CANONICAL_ENSEMBLE_SQL, conn)
    df[FORECAST_DATE_COL] = pd.to_datetime(df[FORECAST_DATE_COL])
    df[TARGET_DATE_COL] = pd.to_datetime(df[TARGET_DATE_COL])
    validate_formation_calendar(df, tickers)
    return df


def validate_formation_calendar(df: pd.DataFrame, tickers: Sequence[str]) -> None:
    """Fails loudly (§3) on: duplicate (ticker, forecast_date) identity;
    NULL/NaN/Inf ensemble_pred; any formation date missing a requested
    ticker, carrying an extra one, or not having exactly len(tickers)
    rows. Never drops a bad date/row silently."""
    tickers = list(tickers)
    if len(set(tickers)) != len(tickers):
        raise ValueError("validate_formation_calendar: duplicate tickers in requested universe")
    expected_set = set(tickers)

    dup_mask = df.duplicated(subset=[TICKER_COL, FORECAST_DATE_COL], keep=False)
    if dup_mask.any():
        n = int(df.duplicated(subset=[TICKER_COL, FORECAST_DATE_COL]).sum())
        raise ValueError(f"validate_formation_calendar: {n} duplicate (ticker, forecast_date) identities")

    pred = df[ENSEMBLE_PRED_COL].astype(float)
    bad_pred = pred.isna() | ~np.isfinite(pred)
    if bad_pred.any():
        raise ValueError(f"validate_formation_calendar: {int(bad_pred.sum())} row(s) with NULL/NaN/Inf ensemble_pred")

    for date, g in df.groupby(FORECAST_DATE_COL):
        present = set(g[TICKER_COL])
        missing = expected_set - present
        extra = present - expected_set
        if missing:
            raise ValueError(f"validate_formation_calendar: {date} missing ticker(s): {sorted(missing)}")
        if extra:
            raise ValueError(f"validate_formation_calendar: {date} unexpected extra ticker(s): {sorted(extra)}")
        if len(g) != len(tickers):
            raise ValueError(f"validate_formation_calendar: {date} has {len(g)} rows, expected {len(tickers)}")


def verify_t_plus_horizon_alignment(calendar_df: pd.DataFrame, prices: pd.DataFrame, horizon: int = None) -> None:
    """Independently verifies each persisted (forecast_date, target_date)
    pair really is exactly `horizon` (default config.FORECAST_HORIZON)
    VALID trading sessions apart, using the real session calendar built
    from `prices` — never assumed merely because config.STEP ==
    config.FORECAST_HORIZON (§4, §12)."""
    horizon = config.FORECAST_HORIZON if horizon is None else horizon
    global_calendar = build_global_calendar(prices)
    pos = {d: i for i, d in enumerate(global_calendar)}
    pairs = calendar_df[[FORECAST_DATE_COL, TARGET_DATE_COL]].drop_duplicates()
    for _, row in pairs.iterrows():
        fd, td = row[FORECAST_DATE_COL], row[TARGET_DATE_COL]
        if fd not in pos:
            raise ValueError(f"verify_t_plus_horizon_alignment: forecast_date {fd} is not a valid session")
        if td not in pos:
            raise ValueError(f"verify_t_plus_horizon_alignment: target_date {td} is not a valid session")
        gap = pos[td] - pos[fd]
        if gap != horizon:
            raise ValueError(
                f"verify_t_plus_horizon_alignment: {fd} -> {td} is {gap} valid session(s) apart, expected exactly {horizon}"
            )


def verify_sequential_non_overlapping_path(calendar_df: pd.DataFrame) -> bool:
    """True only if, for every consecutive pair of formation dates
    (sorted), target_date(T_i) == forecast_date(T_{i+1}) — i.e. the
    periods chain into one continuous, non-overlapping investment path.
    Never assumed merely because STEP == FORECAST_HORIZON (§12) — checked
    against the actual persisted dates."""
    dates = (
        calendar_df[[FORECAST_DATE_COL, TARGET_DATE_COL]]
        .drop_duplicates()
        .sort_values(FORECAST_DATE_COL)
        .reset_index(drop=True)
    )
    for i in range(len(dates) - 1):
        if dates.loc[i, TARGET_DATE_COL] != dates.loc[i + 1, FORECAST_DATE_COL]:
            return False
    return True


# ---------------------------------------------------------------------------
# Phase 7 covariance-eligibility filter (NEW lock, planner-resolved — see
# module docstring above and the Phase 7 covariance-eligibility amendment
# report). Applied to a COPY of the Phase 4-6 calendar; never mutates or
# deletes anything in `predictions`.
# ---------------------------------------------------------------------------
@dataclass
class CovarianceEligibilityReport:
    n_total_formations: int
    n_eligible_formations: int
    excluded: List[dict]  # each: formation_date, tri_levels_available, returns_available, returns_required


def covariance_eligible_calendar(
    calendar_df: pd.DataFrame,
    prices: pd.DataFrame,
    tickers: Sequence[str] = None,
    window: int = None,
) -> Tuple[pd.DataFrame, CovarianceEligibilityReport]:
    """Phase 7 PORTFOLIO-construction eligibility filter. A frozen Phase
    4-6 forecasting formation T is Phase 7 portfolio-eligible only if the
    full `window` (default `config.COVAR_WINDOW`) causal one-session TRI
    returns are available as of T — i.e. `window + 1` valid TRI levels
    exist from the start of price history through T inclusive (§2 of the
    amendment: "No shorter covariance window is permitted. No padding...
    No interpolation... No use of future data...").

    Returns `(eligible_calendar_df, report)`. `eligible_calendar_df` is a
    FILTERED COPY of `calendar_df` — this function never mutates its
    input and never touches `predictions`; an excluded formation date
    remains a fully valid Phase 4-6 ML forecasting record, simply not
    Phase 7 PORTFOLIO-eligible. `report` names every excluded date with
    its exact TRI-level/return counts, for the driver to report
    explicitly (never a silent drop)."""
    tickers = list(config.TICKER_UNIVERSE) if tickers is None else list(tickers)
    window = config.COVAR_WINDOW if window is None else window

    formation_dates = sorted(pd.Timestamp(d) for d in calendar_df[FORECAST_DATE_COL].unique())
    global_calendar = build_global_calendar(prices)
    pos = {d: i for i, d in enumerate(global_calendar)}

    eligible_dates: List[pd.Timestamp] = []
    excluded: List[dict] = []
    for fd in formation_dates:
        if fd not in pos:
            raise ValueError(f"covariance_eligible_calendar: formation_date {fd.date()} is not a valid session in prices")
        tri_levels_available = pos[fd] + 1  # sessions from history start through fd, inclusive
        returns_available = tri_levels_available - 1
        if returns_available >= window:
            eligible_dates.append(fd)
        else:
            excluded.append({
                "formation_date": fd,
                "tri_levels_available": tri_levels_available,
                "returns_available": returns_available,
                "returns_required": window,
            })

    eligible_calendar = calendar_df[
        pd.to_datetime(calendar_df[FORECAST_DATE_COL]).isin(eligible_dates)
    ].reset_index(drop=True)

    report = CovarianceEligibilityReport(
        n_total_formations=len(formation_dates),
        n_eligible_formations=len(eligible_dates),
        excluded=excluded,
    )
    return eligible_calendar, report


# ---------------------------------------------------------------------------
# STAGE A — construction. Pure function of (formation_date, mu, causal
# prices, usgg10yr) — never sees anything dated after formation_date.
# ---------------------------------------------------------------------------
@dataclass
class ConstructionResult:
    formation_date: pd.Timestamp
    tickers: List[str]
    weights: Dict[str, np.ndarray]        # strategy -> weight array, in `tickers` order
    sigma_21: Dict[str, np.ndarray]        # "SAMPLE"/"LW" -> Sigma_21 actually used
    mu_21: np.ndarray
    rf_21: float


def construct_portfolios_at(
    formation_date,
    mu: pd.Series,
    prices: pd.DataFrame,
    usgg10yr: float,
    tickers: Sequence[str] = None,
    max_weight: float = None,
) -> ConstructionResult:
    """STAGE A only (§10 steps 1-13). `prices` may contain rows dated
    after `formation_date` — those are always excluded via
    `optimizer.covariance`'s `as_of_date` cutoff before anything is
    estimated, so passing a full price history in is safe and does not
    leak information forward."""
    tickers = list(config.TICKER_UNIVERSE) if tickers is None else list(tickers)
    max_weight = config.MAX_WEIGHT if max_weight is None else max_weight

    returns_window = covariance_return_window(prices, tickers, as_of_date=formation_date)
    sigma_sample_session = sample_covariance_session(returns_window)
    sigma_lw_session = ledoit_wolf_covariance_session(returns_window)
    sigma_sample_21 = scale_covariance_to_horizon(sigma_sample_session)
    sigma_lw_21 = scale_covariance_to_horizon(sigma_lw_session)
    validate_covariance_matrix(sigma_sample_21, tickers)
    validate_covariance_matrix(sigma_lw_21, tickers)

    mu_arr_sample, sigma_sample_arr = align_mu_sigma(mu, sigma_sample_21, tickers)
    mu_arr_lw, sigma_lw_arr = align_mu_sigma(mu, sigma_lw_21, tickers)
    if not np.allclose(mu_arr_sample, mu_arr_lw):
        raise AssertionError("construct_portfolios_at: mu differs between estimator branches — alignment bug")
    mu_arr = mu_arr_sample

    rf_21 = rf_horizon(usgg10yr)

    weights: Dict[str, np.ndarray] = {
        "SAMPLE_MINVOL": minimum_volatility_weights(sigma_sample_arr, max_weight=max_weight),
        "SAMPLE_MAXSHARPE": maximum_sharpe_weights(mu_arr, sigma_sample_arr, rf_21, max_weight=max_weight),
        "LW_MINVOL": minimum_volatility_weights(sigma_lw_arr, max_weight=max_weight),
        "LW_MAXSHARPE": maximum_sharpe_weights(mu_arr, sigma_lw_arr, rf_21, max_weight=max_weight),
        # EQUAL_WEIGHT is a BENCHMARK, never routed through MAX_WEIGHT (§7).
        "EQUAL_WEIGHT": equal_weight_benchmark(len(tickers)),
    }
    for name, w in weights.items():
        _validate_frozen_weights(w, name, max_weight if name in OPTIMIZED_STRATEGIES else None)

    return ConstructionResult(
        formation_date=pd.Timestamp(formation_date),
        tickers=tickers,
        weights=weights,
        sigma_21={"SAMPLE": sigma_sample_arr, "LW": sigma_lw_arr},
        mu_21=mu_arr,
        rf_21=rf_21,
    )


def _validate_frozen_weights(w: np.ndarray, name: str, max_weight: Optional[float], tol: float = 1e-6) -> None:
    if not np.isfinite(w).all():
        raise ValueError(f"_validate_frozen_weights[{name}]: non-finite weight(s)")
    if abs(float(np.sum(w)) - 1.0) > tol:
        raise ValueError(f"_validate_frozen_weights[{name}]: weights sum to {np.sum(w):.6g}, expected 1")
    if w.min() < -tol:
        raise ValueError(f"_validate_frozen_weights[{name}]: negative weight present")
    if max_weight is not None and w.max() > max_weight + tol:
        raise ValueError(f"_validate_frozen_weights[{name}]: weight {w.max():.6g} exceeds max_weight={max_weight}")


# ---------------------------------------------------------------------------
# STAGE B — evaluation. Accepts already-frozen weights as plain numbers;
# never recomputes or adjusts them. This is the only part of the driver
# permitted to see data dated after formation_date, and only because it
# runs strictly after Stage A has already returned.
# ---------------------------------------------------------------------------
def realized_stock_returns(prices: pd.DataFrame, tickers: Sequence[str], formation_date, target_date) -> pd.Series:
    """realized_stock_return_i = TRI_i(target_date)/TRI_i(formation_date) - 1
    (LOCK, §5) — raw TRI total return; never PX_LAST, never log return,
    never `target_21d`/`actual_return`."""
    tickers = list(tickers)
    wide = build_tri_wide(prices, tickers, as_of_date=target_date)
    if formation_date not in wide.index:
        raise ValueError(f"realized_stock_returns: formation_date {formation_date} is not a valid session")
    if target_date not in wide.index:
        raise ValueError(f"realized_stock_returns: target_date {target_date} is not a valid session")
    tri_t = wide.loc[formation_date]
    tri_t21 = wide.loc[target_date]
    if tri_t.isna().any() or tri_t21.isna().any():
        raise ValueError("realized_stock_returns: missing TRI at formation_date or target_date")
    return tri_t21 / tri_t - 1.0


def realized_portfolio_return(weights: np.ndarray, tickers: Sequence[str], realized_returns: pd.Series) -> float:
    """realized_portfolio_return = sum_i(w_i * realized_stock_return_i)
    (§6), with EXPLICIT ticker alignment before the weighted sum (never
    relies on positional/array-order matching)."""
    tickers = list(tickers)
    aligned = realized_returns.reindex(tickers)
    if aligned.isna().any():
        missing = [t for t in tickers if pd.isna(aligned[t])]
        raise ValueError(f"realized_portfolio_return: missing realized return for ticker(s): {missing}")
    return float(np.asarray(weights, dtype=float) @ aligned.to_numpy(dtype=float))


# ---------------------------------------------------------------------------
# §9 — turnover (NEW Phase 7B lock)
# ---------------------------------------------------------------------------
def turnover(
    w_t: np.ndarray,
    w_prev: Optional[np.ndarray],
    tickers_t: Sequence[str],
    tickers_prev: Optional[Sequence[str]] = None,
) -> float:
    """0.5 * sum(|w_i,T - w_i,T-1|). NaN for the first formation date
    (`w_prev is None`) — never 0 (§9). Ticker alignment is enforced
    explicitly rather than assumed positional."""
    if w_prev is None:
        return float("nan")
    tickers_t = list(tickers_t)
    tickers_prev = tickers_t if tickers_prev is None else list(tickers_prev)
    if tickers_t == tickers_prev:
        wt = np.asarray(w_t, dtype=float)
        wp = np.asarray(w_prev, dtype=float)
    else:
        union = sorted(set(tickers_t) | set(tickers_prev))
        st = pd.Series(np.asarray(w_t, dtype=float), index=tickers_t).reindex(union, fill_value=0.0)
        sp = pd.Series(np.asarray(w_prev, dtype=float), index=tickers_prev).reindex(union, fill_value=0.0)
        wt, wp = st.to_numpy(), sp.to_numpy()
    return float(0.5 * np.sum(np.abs(wt - wp)))


# ---------------------------------------------------------------------------
# §12 — direct 21-session aggregate statistics only. §13: no annualization.
# ---------------------------------------------------------------------------
@dataclass
class StrategyPeriodResult:
    formation_date: pd.Timestamp
    realized_return: float
    turnover: float
    max_weight_observed: float


def aggregate_statistics(period_results: Sequence[StrategyPeriodResult]) -> dict:
    """Direct 21-session statistics only (§12) — arithmetic mean, std,
    median, min, max, positive-period hit rate, turnover summary,
    concentration. Deliberately excludes any annualized/interpolated
    metric (§13) — see the Phase 7B report for what was deferred and why."""
    if len(period_results) == 0:
        raise ValueError("aggregate_statistics: no periods supplied")
    returns = np.array([r.realized_return for r in period_results], dtype=float)
    turnovers = np.array([r.turnover for r in period_results], dtype=float)
    max_weights = np.array([r.max_weight_observed for r in period_results], dtype=float)
    if not np.isfinite(returns).all():
        raise ValueError("aggregate_statistics: non-finite realized_return present")
    if not np.isfinite(max_weights).all():
        raise ValueError("aggregate_statistics: non-finite max_weight_observed present")

    turnovers_valid = turnovers[~np.isnan(turnovers)]

    return {
        "n_periods": int(len(returns)),
        "mean_return_21": float(np.mean(returns)),
        "std_return_21": float(np.std(returns, ddof=1)) if len(returns) > 1 else float("nan"),
        "median_return_21": float(np.median(returns)),
        "min_return_21": float(np.min(returns)),
        "max_return_21": float(np.max(returns)),
        "hit_rate": float(np.mean(returns > 0)),
        "mean_turnover": float(np.mean(turnovers_valid)) if len(turnovers_valid) > 0 else float("nan"),
        "median_turnover": float(np.median(turnovers_valid)) if len(turnovers_valid) > 0 else float("nan"),
        "max_turnover": float(np.max(turnovers_valid)) if len(turnovers_valid) > 0 else float("nan"),
        "avg_max_weight": float(np.mean(max_weights)),
        "max_observed_weight": float(np.max(max_weights)),
    }


def cumulative_compounded_return(period_results: Sequence[StrategyPeriodResult], calendar_df: pd.DataFrame) -> float:
    """product_T(1+R) - 1 — ONLY computed after
    `verify_sequential_non_overlapping_path` confirms the periods form one
    continuous, non-overlapping investment path (§12). Raises rather than
    silently fabricating a misleading compounded figure otherwise."""
    if not verify_sequential_non_overlapping_path(calendar_df):
        raise ValueError(
            "cumulative_compounded_return: formation/target dates do not form a clean "
            "sequential non-overlapping path — refusing to fabricate a continuous "
            "compounded investment path"
        )
    compounded = 1.0
    for r in period_results:
        compounded *= 1.0 + r.realized_return
    return compounded - 1.0


# ---------------------------------------------------------------------------
# §14 — covariance diagnostics (mathematically defined only; never selects
# Sample vs. Ledoit-Wolf).
# ---------------------------------------------------------------------------
def covariance_diagnostics(sigma_21: np.ndarray, weights: np.ndarray, psd_tol: float = -1e-8) -> dict:
    sigma_21 = np.asarray(sigma_21, dtype=float)
    eigvals = np.linalg.eigvalsh(sigma_21)
    min_eig, max_eig = float(eigvals.min()), float(eigvals.max())
    is_psd = bool(min_eig >= psd_tol)
    cond = float(max_eig / min_eig) if min_eig > 0 else float("inf")
    return {
        "condition_number": cond,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": max_eig,
        "is_psd": is_psd,
        "psd_tolerance": psd_tol,
        "predicted_volatility_21": portfolio_volatility(weights, sigma_21),
        "max_weight_observed": float(np.max(weights)),
    }


# ---------------------------------------------------------------------------
# DEPRECATED — see optimizer.benchmark_spxt for the official benchmark.
# NOT the official ML_SPEC.md §27 benchmark (SPXT total return). Retained
# only for backward compatibility with earlier Phase 7B callers/tests.
# ---------------------------------------------------------------------------
def spx_price_return_diagnostic(macro: pd.DataFrame, formation_date, target_date) -> float:
    """DEPRECATED — NON-OFFICIAL. A labeled diagnostic: SPX PRICE return
    (dividends EXCLUDED) over [formation_date, target_date], computed
    directly from `macro['spx']` (== the raw `SPX_PX_LAST` Bloomberg
    field, per `pipeline.normalize.MACRO_COLUMN_RENAME` — never a
    total-return series, because none was ever pulled from this source).
    NEVER referred to as "the SPX benchmark" or "the official benchmark"
    anywhere in this module or in `STRATEGIES`. The official Phase 7
    market benchmark is `optimizer.benchmark_spxt.spxt_total_return`
    (S&P 500 TOTAL RETURN, `SPXT Index`/`PX_LAST`) — use that instead for
    any official Phase 7 evaluation/reporting."""
    if "spx" not in macro.columns:
        raise ValueError("spx_price_return_diagnostic: macro frame missing 'spx' column")
    levels = macro.dropna(subset=["spx"]).sort_values(DATE_COL).set_index(DATE_COL)["spx"]
    if formation_date not in levels.index:
        raise ValueError(f"spx_price_return_diagnostic: formation_date {formation_date} has no spx quote")
    if target_date not in levels.index:
        raise ValueError(f"spx_price_return_diagnostic: target_date {target_date} has no spx quote")
    return float(levels.loc[target_date] / levels.loc[formation_date] - 1.0)
