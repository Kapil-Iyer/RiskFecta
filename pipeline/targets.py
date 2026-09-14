"""
RiskFecta Phase 3 — 21-valid-trading-session forward TRI return target.

Locked formula (ML_SPEC.md §8):

    y(i, t) = TRI(i, t+21) / TRI(i, t) - 1

where `i` is a ticker and `t` / `t+21` are VALID trading sessions
(ML_SPEC.md §3, §29) — never raw calendar-row offsets. Concretely: for
ticker i reduced to its own ordered sequence of valid sessions, `t+21` means
"21 positions further down that same ticker's own sorted sequence", not
"21 calendar days later" and never a raw-row shift on unfiltered data.

This module is Bloomberg-shape-agnostic: it needs only `ticker`, `date`,
`close` (the session-validity gate — see pipeline.sessions) and
`total_return_idx` (TRI). `close` is used *only* for the session gate; the
target formula itself is computed on TRI, never on close.

Target/feature isolation (ML_SPEC.md §29, task brief §15):
    Changing a future TRI value may legitimately change a TARGET at an
    earlier formation date whose 21-session horizon reaches that future
    date. It must NEVER change a FEATURE value at that earlier date. This
    module never reads or writes pipeline.features output, and vice versa
    — the two are computed independently from prices_raw-shaped input.

Missing-target taxonomy (task brief §15, §18-A):
    - The final FORECAST_HORIZON valid sessions of each ticker have no
      t+21 row yet -> target is NaN (never filled/zeroed/dropped silently).
    - A row whose own TRI, or whose t+21 TRI, is itself genuinely missing
      -> target is NaN (propagates naturally through division).
"""
from __future__ import annotations

from typing import Optional

import pandas as pd

import config
from pipeline.sessions import DATE_COL, TICKER_COL, valid_sessions

TRI_COL = "total_return_idx"
TARGET_COL = "target_21d"


def compute_targets(
    prices: pd.DataFrame,
    horizon: Optional[int] = None,
    close_col: str = "close",
    tri_col: str = TRI_COL,
) -> pd.DataFrame:
    """Compute y(i, t) = TRI(i, t+horizon) / TRI(i, t) - 1 for every valid
    session of every ticker in `prices`.

    `prices` may be calendar-inclusive (e.g. a fixture with weekend
    placeholder rows where TOTAL_RETURN_INDEX is still populated but
    `close_col` is NaN) — this function applies the session gate itself via
    pipeline.sessions.valid_sessions before computing anything, so a
    weekend/holiday placeholder row can never occupy a "t" or "t+horizon"
    slot regardless of its TRI value.

    Returns columns: ticker, date, target_21d — one row per valid session,
    sorted by (ticker, date). `target_21d` is NaN where unavailable (see
    module docstring).
    """
    horizon = config.FORECAST_HORIZON if horizon is None else horizon
    if horizon <= 0:
        raise ValueError("compute_targets: horizon must be a positive number of valid trading sessions")

    sess = valid_sessions(prices, close_col=close_col)
    df = sess[[TICKER_COL, DATE_COL, tri_col]].copy()

    # groupby(...).shift(-horizon) shifts strictly WITHIN each ticker's own
    # group (aligned on the original index), so:
    #   - it never reaches across a ticker boundary (no cross-ticker leakage)
    #   - the trailing `horizon` rows of each group naturally become NaN
    #     (there is no future row to shift in) — exactly the "final 21
    #     sessions have no target" requirement, with no manual fill.
    future_tri = df.groupby(TICKER_COL)[tri_col].shift(-horizon)
    df[TARGET_COL] = future_tri / df[tri_col] - 1

    return df[[TICKER_COL, DATE_COL, TARGET_COL]].reset_index(drop=True)
