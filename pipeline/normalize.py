"""
RiskFecta Phase 1A — wide -> long normalization + trading-session filtering.

Consumes the structurally-validated raw frames from pipeline.validate and
produces schema.sql-shaped, database-ready DataFrames:
  - normalize_prices()        -> prices_raw-shaped rows (session-filtered)
  - normalize_macro()         -> one row per date across SPX/VIX/USGG10YR
  - normalize_static_fields() -> one row per ticker, snapshot fields

Locked rules (ML_SPEC.md §3-§5, §29; schema.sql):
  - A valid trading session requires a genuine PX_LAST (close). NEVER
    TOTAL_RETURN_INDEX — TOTAL_RETURN_INDEX may remain populated on
    weekend/holiday calendar rows and must never be used to decide whether a
    row is a real trading session.
  - Genuine missing values are preserved as NaN/NULL — never zero-filled,
    never forward/back-filled, never interpolated. Forward-fill is a Phase 3
    (feature engineering) concern only and is not implemented here.
  - Output is returned as in-memory DataFrames / can be written under
    data/processed/ (gitignored). data/raw/ is never written to.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

from pipeline.validate import (
    ParsedMacroRaw,
    ParsedPricesRaw,
    ParsedStaticFieldsRaw,
    parse_macro_raw,
    parse_prices_raw,
    parse_static_fields_raw,
)

PathLike = Union[str, Path]

PRICE_COLUMN_RENAME = {
    "PX_OPEN": "open",
    "PX_HIGH": "high",
    "PX_LOW": "low",
    "PX_LAST": "close",
    "PX_VOLUME": "volume",
    "TOTAL_RETURN_INDEX": "total_return_idx",
}

MACRO_COLUMN_RENAME = {"SPX": "spx", "VIX": "vix", "USGG10YR": "yield_10y"}

PRICES_OUTPUT_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume", "total_return_idx"]
MACRO_OUTPUT_COLUMNS = ["date", "spx", "vix", "yield_10y"]
STATIC_OUTPUT_COLUMNS = ["ticker", "market_cap", "beta", "div_yield", "sector"]


def normalize_prices(
    parsed: Optional[ParsedPricesRaw] = None,
    path: Optional[PathLike] = None,
    expected_universe: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Wide Bloomberg price panel -> long, trading-session-filtered rows.

    Session-validity gate: PX_LAST (close) must be a genuine observed value
    (ML_SPEC.md §3). TOTAL_RETURN_INDEX is NEVER used for this gate — it can
    remain populated on non-trading calendar rows and is preserved as-is on
    every row that survives the PX_LAST gate. Rows without a genuine close
    are excluded entirely (not NaN-filled), which also removes every
    weekend/holiday calendar row (ML_SPEC.md §4 item 2).

    No forward-fill, back-fill, interpolation, or zero-fill of any field
    happens here — a genuinely missing OHLCV/TRI value on an otherwise valid
    trading session is preserved as NaN.
    """
    if parsed is None:
        parsed = parse_prices_raw(path=path, expected_universe=expected_universe)

    df = parsed.frame
    df = df[df["PX_LAST"].notna()].copy()  # the ONLY session-validity gate — never TOTAL_RETURN_INDEX
    df = df.rename(columns=PRICE_COLUMN_RENAME)
    df = df[PRICES_OUTPUT_COLUMNS]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if df.duplicated(subset=["ticker", "date"]).any():
        # Unreachable given parse_prices_raw's own uniqueness check, but kept
        # as an explicit, cheap guarantee on the function's own output.
        raise ValueError("normalize_prices: duplicate (ticker, date) rows after session filtering")

    return df


def normalize_macro(parsed: Optional[ParsedMacroRaw] = None, path: Optional[PathLike] = None) -> pd.DataFrame:
    """Three independent Bloomberg macro series -> one long table keyed by date.

    Each series keeps only the calendar dates it was genuinely quoted on. A
    date on which one series has no genuine observation (e.g. USGG10YR
    quoted on a day SPX was not) leaves that series' column NaN on that
    date's row — never fabricated, never dropped from the other series.
    """
    if parsed is None:
        parsed = parse_macro_raw(path=path)

    out: Optional[pd.DataFrame] = None
    for name, sub in parsed.series.items():
        renamed = sub.rename(columns={"px_last": MACRO_COLUMN_RENAME[name]})
        out = renamed if out is None else out.merge(renamed, on="date", how="outer")

    out = out.sort_values("date").reset_index(drop=True)
    return out[MACRO_OUTPUT_COLUMNS]


def normalize_static_fields(
    parsed: Optional[ParsedStaticFieldsRaw] = None,
    path: Optional[PathLike] = None,
    expected_universe: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Bloomberg static snapshot fields -> one row per ticker.

    These are single current-day values, not historical time series
    (ML_SPEC.md §2, §10) — this function does not attach them to historical
    dates and does not compute derived/engineered features (e.g. a log
    market-cap transform); that belongs to Phase 3's feature pipeline, out of
    this subphase's scope. Genuine missing fields are preserved as NaN.
    """
    if parsed is None:
        parsed = parse_static_fields_raw(path=path, expected_universe=expected_universe)

    df = parsed.frame.rename(
        columns={
            "CUR_MKT_CAP": "market_cap",
            "BETA_RAW_OVERRIDABLE": "beta",
            "DIVIDEND_INDICATED_YIELD": "div_yield",
            "GICS_SECTOR_NAME": "sector",
        }
    )
    return df[STATIC_OUTPUT_COLUMNS].sort_values("ticker").reset_index(drop=True)
