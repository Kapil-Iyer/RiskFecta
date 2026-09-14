"""
RiskFecta Phase 3 — shared trading-session ordering.

Single authoritative definition of "valid trading session," reused by
pipeline.features and pipeline.targets so the session gate is never
re-implemented ad hoc in two places (ML_SPEC.md §3, §29):

  - A valid session for ticker i is a row where PX_LAST (close) is a genuine
    observed value.
  - TOTAL_RETURN_INDEX is NEVER used to decide session validity — it may
    remain populated (held flat / carried forward by Bloomberg) on
    weekend/holiday calendar rows, so gating on it would silently admit
    invalid rows.

`prices_raw` (schema.sql) already enforces `close NOT NULL` at load time
(pipeline.normalize.normalize_prices gates on PX_LAST before any row is
inserted), so a `SELECT ... FROM prices_raw` result is already
valid-session-only. The filter below is applied defensively so any caller
that hands in calendar-inclusive rows (e.g. a raw/synthetic fixture that
still has weekend placeholder rows with TOTAL_RETURN_INDEX populated) gets
exactly the same gate — never gating on `total_return_idx`.
"""
from __future__ import annotations

import pandas as pd

TICKER_COL = "ticker"
DATE_COL = "date"


def valid_sessions(df: pd.DataFrame, close_col: str = "close") -> pd.DataFrame:
    """Filter `df` to genuine-close rows only, sorted ascending by
    (ticker, date). Never gates on total_return_idx/TOTAL_RETURN_INDEX —
    only `close_col` (PX_LAST) decides validity (ML_SPEC.md §3).
    """
    if close_col not in df.columns:
        raise ValueError(f"valid_sessions: expected a '{close_col}' column to gate on, not found")
    out = df[df[close_col].notna()].copy()
    out = out.sort_values([TICKER_COL, DATE_COL], kind="mergesort").reset_index(drop=True)
    return out


def session_number(df: pd.DataFrame) -> pd.Series:
    """0-based valid-session index within each ticker's own sorted sequence:
    row 0 is that ticker's first valid session, row 1 its second, etc. Use
    this (not calendar-row position) to express any lookback/horizon in
    valid-session counts. `df` must already be `valid_sessions(...)`-filtered
    and sorted; this function does not re-filter or re-sort.
    """
    return df.groupby(TICKER_COL).cumcount()
