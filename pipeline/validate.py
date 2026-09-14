"""
RiskFecta Phase 1A — Bloomberg raw-export structural validation.

Parses the ACTUAL Bloomberg CSV export layouts (multi-header wide price
panel, independent-length macro series, single-snapshot static fields) and
validates their structure against config.TICKER_UNIVERSE before any
normalization runs (pipeline/normalize.py).

Scope (locked, see BUILD_PLAN.md Phase 1A):
- Read-only against data/raw/*.csv. These files are NEVER written to.
- No forward/back-fill, no zero-fill, no interpolation anywhere in this
  module. Genuine Bloomberg "#N/A"-style missing values are preserved as
  NaN.
- TOTAL_RETURN_INDEX is parsed and returned like any other field, but this
  module never uses it to decide trading-session validity (ML_SPEC.md §3,
  §29). Session-validity filtering itself happens in pipeline/normalize.py,
  not here — this module only parses and structurally validates.

Raw Bloomberg export quirks discovered during the Phase 1A audit of the real
files (documented, not silently "fixed"):
- prices_raw.csv column 0 is NOT a data column tied to that row. It is a
  stray vertical ticker list (an Excel/BQL export artifact — the real
  TICKER_UNIVERSE list spilling down column A) whose values are unrelated to
  the date or values on that row. It is read and discarded, never treated as
  "the ticker for this row".
- macro.csv has three fully independent (DATE, PX_LAST) column pairs (SPX,
  VIX, USGG10YR) of *different natural lengths* (1256 / 1285 / 1303 valid
  rows respectively in the real export), not one shared date column. Row
  position does NOT imply the same date across the three series — each pair
  is parsed and validated independently.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import pandas as pd

import config

# ---------------------------------------------------------------------------
# Exceptions — explicit, typed failure modes (never silent repair)
# ---------------------------------------------------------------------------
class BloombergValidationError(Exception):
    """Base class for all Phase 1A raw-export validation failures."""


class MalformedExportError(BloombergValidationError):
    """Raw file structure does not match the expected Bloomberg layout."""


class UniverseMismatchError(BloombergValidationError):
    """Ticker universe found in a raw export does not match the expected universe."""

    def __init__(self, missing: set, extra: set):
        self.missing = missing
        self.extra = extra
        super().__init__(
            f"Ticker universe mismatch — missing from export: {sorted(missing)}; "
            f"unexpected tickers in export: {sorted(extra)}"
        )


class DuplicateObservationError(BloombergValidationError):
    """Duplicate date/ticker rows found where uniqueness is required."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_PRICE_FIELDS: List[str] = [
    "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME", "TOTAL_RETURN_INDEX",
]
N_FIELDS_PER_TICKER = len(EXPECTED_PRICE_FIELDS)

# Bloomberg NA placeholders beyond pandas' own defaults (defensive superset).
# Anything NOT in this list that still isn't numeric fails loudly at coercion
# (pd.to_numeric(..., errors="raise")) rather than being silently dropped.
BLOOMBERG_NA_VALUES = ["#N/A", "#N/A N/A", "#N/A Invalid Security", "#VALUE!", "N/A"]

STATIC_FIELDS: List[str] = [
    "CUR_MKT_CAP", "BETA_RAW_OVERRIDABLE", "DIVIDEND_INDICATED_YIELD", "GICS_SECTOR_NAME",
]

MACRO_SERIES: List[str] = ["SPX", "VIX", "USGG10YR"]

PathLike = Union[str, Path]


def _strip_bbg_suffix(security: str) -> str:
    """'AAPL US Equity' -> 'AAPL'."""
    return str(security).split(" ")[0]


# ---------------------------------------------------------------------------
# Prices — wide Bloomberg multi-header panel
# ---------------------------------------------------------------------------
@dataclass
class ParsedPricesRaw:
    """Result of structurally parsing prices_raw.csv: one row per
    (ticker, calendar date) carrying the raw Bloomberg field names, BEFORE
    any trading-session filtering. Includes weekend/holiday placeholder rows
    (PX_* fields NaN, TOTAL_RETURN_INDEX may still be populated).
    """

    frame: pd.DataFrame  # columns: ticker, date, PX_OPEN, PX_HIGH, PX_LOW, PX_LAST, PX_VOLUME, TOTAL_RETURN_INDEX
    tickers: List[str]


def parse_prices_raw(
    path: Optional[PathLike] = None,
    expected_universe: Optional[Sequence[str]] = None,
) -> ParsedPricesRaw:
    """Parse the actual Bloomberg wide price export.

    Structure (confirmed against the real export, not assumed):
      - Row 0: ticker label repeated across each ticker's 6-column block
        (e.g. "AAPL US Equity" x6), after 2 leading columns.
      - Row 1: field names per column ("DATES" in column 1, then
        PX_OPEN/PX_HIGH/PX_LOW/PX_LAST/PX_VOLUME/TOTAL_RETURN_INDEX repeating
        per ticker block).
      - Row 2+: data. Column 0 is a stray, row-position-only ticker label
        UNRELATED to that row's date/values and is discarded (see module
        docstring). Column 1 is the shared calendar-date column used by every
        ticker block. Columns 2.. are the 6-column-per-ticker data blocks.

    Raises MalformedExportError / UniverseMismatchError / DuplicateObservationError
    on any structural problem. Never fabricates or silently repairs data.
    """
    path = Path(path) if path is not None else config.DATA_RAW / "prices_raw.csv"
    expected = set(expected_universe) if expected_universe is not None else set(config.TICKER_UNIVERSE)

    raw = pd.read_csv(path, header=None, low_memory=False, na_values=BLOOMBERG_NA_VALUES)

    n_cols = raw.shape[1]
    n_data_cols = n_cols - 2
    if n_data_cols <= 0 or n_data_cols % N_FIELDS_PER_TICKER != 0:
        raise MalformedExportError(
            f"{path.name}: unexpected column count {n_cols}; expected 2 leading columns "
            f"+ a multiple of {N_FIELDS_PER_TICKER} ticker-block columns"
        )
    n_tickers = n_data_cols // N_FIELDS_PER_TICKER

    if raw.shape[0] < 2:
        raise MalformedExportError(f"{path.name}: expected at least 2 header rows, found {raw.shape[0]} total rows")

    ticker_row = raw.iloc[0]
    field_row = raw.iloc[1]

    tickers: List[str] = []
    for i in range(n_tickers):
        cols = list(range(2 + i * N_FIELDS_PER_TICKER, 2 + (i + 1) * N_FIELDS_PER_TICKER))
        block_labels = ticker_row[cols].tolist()
        if len(set(block_labels)) != 1:
            raise MalformedExportError(
                f"{path.name}: ticker header block at columns {cols} is not a single "
                f"repeated ticker label: {block_labels}"
            )
        block_fields = field_row[cols].tolist()
        if block_fields != EXPECTED_PRICE_FIELDS:
            raise MalformedExportError(
                f"{path.name}: field-name block at columns {cols} does not match expected "
                f"{EXPECTED_PRICE_FIELDS}, found {block_fields}"
            )
        tickers.append(_strip_bbg_suffix(block_labels[0]))

    if len(set(tickers)) != len(tickers):
        dupes = sorted({t for t in tickers if tickers.count(t) > 1})
        raise MalformedExportError(f"{path.name}: duplicate ticker header blocks found: {dupes}")

    found = set(tickers)
    if found != expected:
        raise UniverseMismatchError(missing=expected - found, extra=found - expected)

    data = raw.iloc[2:].reset_index(drop=True)
    dates = pd.to_datetime(data.iloc[:, 1], errors="raise")
    if dates.isna().any():
        raise MalformedExportError(f"{path.name}: unparseable date value(s) in the DATES column")
    if dates.duplicated().any():
        dup_dates = dates[dates.duplicated()].unique()
        raise DuplicateObservationError(f"{path.name}: duplicate calendar date row(s): {list(dup_dates)[:5]}")

    frames = []
    for i, ticker in enumerate(tickers):
        cols = list(range(2 + i * N_FIELDS_PER_TICKER, 2 + (i + 1) * N_FIELDS_PER_TICKER))
        block = data.iloc[:, cols].copy()
        block.columns = EXPECTED_PRICE_FIELDS
        for col in EXPECTED_PRICE_FIELDS:
            block[col] = pd.to_numeric(block[col], errors="raise")
        block.insert(0, "date", dates.values)
        block.insert(0, "ticker", ticker)
        frames.append(block)

    long_df = pd.concat(frames, ignore_index=True)

    dup_mask = long_df.duplicated(subset=["ticker", "date"])
    if dup_mask.any():
        raise DuplicateObservationError(
            f"{path.name}: duplicate (ticker, date) rows after reshape: "
            f"{long_df.loc[dup_mask, ['ticker', 'date']].head().to_dict('records')}"
        )

    return ParsedPricesRaw(frame=long_df, tickers=tickers)


# ---------------------------------------------------------------------------
# Macro — three independent (DATE, PX_LAST) series
# ---------------------------------------------------------------------------
@dataclass
class ParsedMacroRaw:
    """Each Bloomberg macro series (SPX, VIX, USGG10YR) is exported as an
    independent (DATE, PX_LAST) column pair. The three pairs are NOT
    row-aligned by date (confirmed against the real export — see module
    docstring) and have different natural lengths. Each series is parsed,
    date-deduplicated, and validated independently; ``series[name]`` has
    columns ``date`` and ``px_last``.
    """

    series: Dict[str, pd.DataFrame]


def parse_macro_raw(path: Optional[PathLike] = None) -> ParsedMacroRaw:
    """Parse the actual Bloomberg macro export.

    The file is NOT a single date-indexed table: SPX_DATE/VIX_DATE/
    USGG10YR_DATE are independent columns with independent lengths (the
    shorter series' trailing rows are simply blank in the CSV — a padding
    artifact, not a shared calendar). Rows with no date for a given series
    are the padding and are dropped for that series only; a genuine
    (date-present, value-missing) observation is preserved as NaN, never
    dropped or fabricated.
    """
    path = Path(path) if path is not None else config.DATA_RAW / "macro.csv"
    raw = pd.read_csv(path, na_values=BLOOMBERG_NA_VALUES)

    expected_cols: List[str] = []
    for name in MACRO_SERIES:
        expected_cols += [f"{name}_DATE", f"{name}_PX_LAST"]
    if list(raw.columns) != expected_cols:
        raise MalformedExportError(f"{path.name}: expected columns {expected_cols}, found {list(raw.columns)}")

    series: Dict[str, pd.DataFrame] = {}
    for name in MACRO_SERIES:
        date_col, val_col = f"{name}_DATE", f"{name}_PX_LAST"
        sub = raw[[date_col, val_col]].dropna(subset=[date_col]).copy()
        sub[date_col] = pd.to_datetime(sub[date_col], errors="raise")
        sub[val_col] = pd.to_numeric(sub[val_col], errors="raise")
        if sub[date_col].duplicated().any():
            dup = sub.loc[sub[date_col].duplicated(), date_col].unique()
            raise DuplicateObservationError(f"{path.name}: duplicate {date_col} value(s): {list(dup)[:5]}")
        sub = sub.rename(columns={date_col: "date", val_col: "px_last"})
        sub = sub.sort_values("date").reset_index(drop=True)
        series[name] = sub

    return ParsedMacroRaw(series=series)


# ---------------------------------------------------------------------------
# Static fields — single current-day snapshot per ticker
# ---------------------------------------------------------------------------
@dataclass
class ParsedStaticFieldsRaw:
    """One row per ticker: ticker + the 4 raw Bloomberg static fields."""

    frame: pd.DataFrame  # ticker, CUR_MKT_CAP, BETA_RAW_OVERRIDABLE, DIVIDEND_INDICATED_YIELD, GICS_SECTOR_NAME


def parse_static_fields_raw(
    path: Optional[PathLike] = None,
    expected_universe: Optional[Sequence[str]] = None,
) -> ParsedStaticFieldsRaw:
    """Parse the actual Bloomberg static-fields snapshot export.

    These are single current-day values (ML_SPEC.md §2, §10), not a
    historical series — this function does not attach them to any date.
    Genuine missing fields (e.g. no indicated dividend for a non-dividend
    payer) are preserved as NaN, never zero-filled.
    """
    path = Path(path) if path is not None else config.DATA_RAW / "static_fields.csv"
    expected = set(expected_universe) if expected_universe is not None else set(config.TICKER_UNIVERSE)

    raw = pd.read_csv(path, na_values=BLOOMBERG_NA_VALUES)
    if raw.shape[1] != 1 + len(STATIC_FIELDS):
        raise MalformedExportError(
            f"{path.name}: expected {1 + len(STATIC_FIELDS)} columns (ticker + {len(STATIC_FIELDS)} "
            f"static fields), found {raw.shape[1]}"
        )

    ticker_col = raw.columns[0]
    found_fields = list(raw.columns[1:])
    if found_fields != STATIC_FIELDS:
        raise MalformedExportError(f"{path.name}: expected fields {STATIC_FIELDS}, found {found_fields}")

    frame = raw.copy()
    frame[ticker_col] = frame[ticker_col].map(_strip_bbg_suffix)
    frame = frame.rename(columns={ticker_col: "ticker"})

    if frame["ticker"].duplicated().any():
        dupes = frame.loc[frame["ticker"].duplicated(), "ticker"].tolist()
        raise DuplicateObservationError(f"{path.name}: duplicate ticker row(s): {dupes}")

    for col in STATIC_FIELDS:
        if col == "GICS_SECTOR_NAME":
            continue
        frame[col] = pd.to_numeric(frame[col], errors="raise")

    found = set(frame["ticker"])
    if found != expected:
        raise UniverseMismatchError(missing=expected - found, extra=found - expected)

    return ParsedStaticFieldsRaw(frame=frame)
