"""
RiskFecta Phase 7A — covariance estimation (optimizer/covariance.py).

NEW PHASE 7 PLANNER LOCKS. ML_SPEC.md §23 only fixes causality ("historical
realized returns available as of each formation date — never future
realized returns, never predicted returns") — it never fixed a lookback
window, a return definition, or a horizon-scaling convention. Those three
methodology gaps were the Phase 7A HARD STOP; the planner has since
resolved them explicitly, and this module implements exactly that
resolution (never invented independently here):

  LOCK A — Covariance lookback: `config.COVAR_WINDOW = 252` most recent
    valid-session ONE-SESSION returns ending at/before the formation date.
    Deliberately a SEPARATE constant from `config.TRAIN_WINDOW` (model
    training, ML_SPEC.md §12) even though both currently equal 252 — never
    aliased or documented as "the same window."

  LOCK B — Covariance return definition: simple one-session Bloomberg
    TOTAL_RETURN_INDEX returns,

        r(i, s) = TRI(i, s) / TRI(i, s-1) - 1

    Never log returns, never PX_LAST/close-based returns, never
    target_21d/actual_return/predicted returns. PX_LAST/close remains the
    SESSION-VALIDITY gate (ML_SPEC.md §3, schema.sql header comment):
    valid trading sessions are established from `close` first, and only
    THEN are one-session TRI returns computed within those valid sessions.
    TRI itself never decides session validity.

  LOCK C (session-frequency half) — covariance is estimated here strictly
    at SESSION frequency. Horizon scaling (x21) is a separate, explicit
    step (`scale_covariance_to_horizon`) applied by the caller only AFTER
    session-frequency estimation — never folded into the estimator, never
    applied before estimation.

Causality: every function here only ever sees rows the caller has
restricted to date <= the formation date (`as_of_date` in
`covariance_return_window`/`build_tri_wide`). Nothing here reaches for
"the latest data" on its own — same discipline as `pipeline.folds`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

import config
from pipeline.sessions import DATE_COL, TICKER_COL, valid_sessions

TRI_COL = "total_return_idx"
CLOSE_COL = "close"


def build_tri_wide(
    prices: pd.DataFrame,
    tickers: Sequence[str],
    as_of_date=None,
    close_col: str = CLOSE_COL,
    tri_col: str = TRI_COL,
) -> pd.DataFrame:
    """Wide TRI-level frame: index = valid-session date (ascending),
    columns = exactly `tickers` in the given order.

    Session validity is gated on `close_col` (PX_LAST) via
    `pipeline.sessions.valid_sessions` — never on `tri_col` (ML_SPEC.md
    §3). `as_of_date`, if given, is a causal cutoff applied BEFORE the
    pivot: rows dated after it are dropped, so nothing downstream can ever
    see a post-cutoff TRI level.

    Fails loudly (never silently drops) on a requested ticker entirely
    absent from `prices` as of the cutoff, or a duplicate (ticker, date)
    identity. A genuine mid-window missing TRI value is preserved as NaN
    here (ML_SPEC.md §5 missing-data policy: never forward-filled/
    imputed) — callers that need a complete window enforce that
    themselves (see `session_returns`/`covariance_return_window`).
    """
    tickers = list(tickers)
    if len(set(tickers)) != len(tickers):
        raise ValueError(f"build_tri_wide: duplicate tickers in requested universe: {tickers}")

    sess = valid_sessions(prices, close_col=close_col)
    if as_of_date is not None:
        sess = sess[pd.to_datetime(sess[DATE_COL]) <= pd.to_datetime(as_of_date)]

    present = set(sess[TICKER_COL].unique())
    missing = [t for t in tickers if t not in present]
    if missing:
        raise ValueError(f"build_tri_wide: ticker(s) entirely absent from prices as of cutoff: {missing}")

    sess = sess[sess[TICKER_COL].isin(tickers)]
    dup_mask = sess.duplicated(subset=[TICKER_COL, DATE_COL], keep=False)
    if dup_mask.any():
        raise ValueError("build_tri_wide: duplicate (ticker, date) identity in prices")

    wide = sess.pivot(index=DATE_COL, columns=TICKER_COL, values=tri_col).sort_index()
    wide = wide.reindex(columns=tickers)  # enforce exact requested order, never implicit alphabetical
    return wide


def session_returns(tri_wide: pd.DataFrame) -> pd.DataFrame:
    """Simple one-session returns from a TRI-level wide frame (LOCK B):
    r(i, s) = TRI(i, s)/TRI(i, s-1) - 1. N TRI levels produce exactly
    N-1 returns (the first row has no prior level).

    Fails loudly on any NaN in `tri_wide` (a genuine missing TRI value
    inside the window actually being used for estimation is a real
    ambiguity the caller must resolve, not something this function
    forward-fills or drops around) or any non-finite resulting return
    (e.g. a zero/negative TRI level, which should never occur for a
    genuine Bloomberg TOTAL_RETURN_INDEX).
    """
    if tri_wide.isna().any().any():
        raise ValueError("session_returns: tri_wide contains NaN — resolve before computing returns")
    returns = tri_wide.pct_change().iloc[1:]
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("session_returns: non-finite return produced (check TRI levels for zero/negative values)")
    return returns


def covariance_return_window(
    prices: pd.DataFrame,
    tickers: Sequence[str],
    as_of_date,
    window: int = None,
) -> pd.DataFrame:
    """The exact causal return window covariance estimation uses (LOCK A +
    LOCK B): the most recent `window` (default `config.COVAR_WINDOW`)
    one-session TRI returns ending at/before `as_of_date`.

    `window` returns require `window + 1` TRI levels — this function
    fetches causal TRI levels, trims to exactly the trailing `window + 1`
    of them, and only then computes returns (never the other order, which
    would silently produce `window - 1` returns from `window` levels).

    Raises if fewer than `window + 1` causal TRI levels exist, rather than
    silently returning a shorter window.
    """
    window = config.COVAR_WINDOW if window is None else window
    tri_wide = build_tri_wide(prices, tickers, as_of_date=as_of_date)
    if len(tri_wide) < window + 1:
        raise ValueError(
            f"covariance_return_window: need {window + 1} causal TRI levels to produce "
            f"{window} one-session returns, only {len(tri_wide)} available as of {as_of_date}"
        )
    tri_window = tri_wide.iloc[-(window + 1):]
    returns = session_returns(tri_window)
    assert len(returns) == window, f"covariance_return_window: expected {window} returns, got {len(returns)}"
    return returns


def sample_covariance_session(returns_window: pd.DataFrame) -> pd.DataFrame:
    """Sample covariance (pandas default, ddof=1) of one-session returns,
    at SESSION frequency — no horizon scaling here (LOCK C)."""
    return returns_window.cov()


def ledoit_wolf_covariance_session(returns_window: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance (sklearn), at SESSION frequency —
    no horizon scaling here (LOCK C). Uses exactly the same
    `returns_window` (identical universe/window/return-definition/cutoff)
    as `sample_covariance_session` so the two estimators are directly
    comparable, per the Phase 7 covariance experiment (ML_SPEC.md §23)."""
    lw = LedoitWolf().fit(returns_window.to_numpy())
    return pd.DataFrame(lw.covariance_, index=returns_window.columns, columns=returns_window.columns)


def scale_covariance_to_horizon(sigma_session: pd.DataFrame, horizon: int = None) -> pd.DataFrame:
    """LOCK C: Sigma_21 = horizon * Sigma_session (default
    `config.FORECAST_HORIZON` = 21). A NEW Phase 7 MVP approximation
    assuming approximately stationary, i.i.d. session returns — the
    frozen docs never required or endorsed this scaling; it is a
    planner-locked convention, not claimed to be exact. Applied strictly
    AFTER session-frequency estimation, identically for both Sample and
    Ledoit-Wolf."""
    horizon = config.FORECAST_HORIZON if horizon is None else horizon
    return sigma_session * horizon


def validate_covariance_matrix(sigma: pd.DataFrame, tickers: Sequence[str], atol: float = 1e-8) -> None:
    """Fail loudly on wrong shape, ticker misalignment (missing/extra/
    duplicate/out-of-order axis labels), NaN, Inf, or asymmetry beyond
    `atol`. Never silently reorders or coerces."""
    tickers = list(tickers)
    if list(sigma.index) != tickers or list(sigma.columns) != tickers:
        raise ValueError(
            f"validate_covariance_matrix: ticker misalignment — expected exactly {tickers} "
            f"(in order) on both axes, got index={list(sigma.index)}, columns={list(sigma.columns)}"
        )
    values = sigma.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("validate_covariance_matrix: covariance matrix contains NaN/Inf")
    if not np.allclose(values, values.T, atol=atol):
        raise ValueError("validate_covariance_matrix: covariance matrix is not symmetric within tolerance")
