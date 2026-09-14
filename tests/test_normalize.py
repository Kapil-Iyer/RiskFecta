"""
Tests for pipeline.normalize — trading-session filtering and wide->long
normalization. Fixtures only (tests/bloomberg_fixtures.py); never the real
data/raw/ export (TRD.md §17).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline import normalize
from tests.bloomberg_fixtures import write_macro_fixture, write_prices_fixture, write_static_fields_fixture

TICKERS = ["AAA", "BBB", "CCC"]


def _row(date, aaa=None, bbb=None, ccc=None):
    row = {"date": date}
    if aaa is not None:
        row["AAA"] = aaa
    if bbb is not None:
        row["BBB"] = bbb
    if ccc is not None:
        row["CCC"] = ccc
    return row


def _full(px_last, tri=1.0, open_=None, high=None, low=None, volume=1000):
    return {
        "PX_OPEN": open_ if open_ is not None else px_last - 0.5,
        "PX_HIGH": high if high is not None else px_last + 0.5,
        "PX_LOW": low if low is not None else px_last - 1,
        "PX_LAST": px_last,
        "PX_VOLUME": volume,
        "TOTAL_RETURN_INDEX": tri,
    }


# ---------------------------------------------------------------------------
# normalize_prices — session gating
# ---------------------------------------------------------------------------
def test_normalize_prices_excludes_weekend_and_holiday_rows(tmp_path):
    p = tmp_path / "prices_raw.csv"
    rows = [
        _row("2021-03-06"),  # Saturday, no data given -> all-#N/A placeholder
        _row("2021-03-08", aaa=_full(10.5), bbb=_full(20.5), ccc=_full(30.5)),  # Monday, valid
    ]
    write_prices_fixture(p, TICKERS, rows)

    out = normalize.normalize_prices(path=p, expected_universe=TICKERS)

    assert set(out["date"]) == {pd.Timestamp("2021-03-08")}
    assert len(out) == len(TICKERS)


def test_normalize_prices_never_uses_total_return_index_as_session_gate(tmp_path):
    """The exact footgun flagged for Phase 1A: a populated TOTAL_RETURN_INDEX
    on a non-trading row must NOT cause that row to be treated as a valid
    session, and a genuinely missing TOTAL_RETURN_INDEX on an otherwise valid
    PX_LAST row must NOT cause that row to be excluded."""
    p = tmp_path / "prices_raw.csv"
    rows = [
        # Non-trading placeholder: PX_LAST is #N/A but TRI is a real-looking,
        # non-trivial value (not just the default "1") for AAA.
        {
            "date": "2021-03-06",
            "AAA": {"PX_OPEN": None, "PX_HIGH": None, "PX_LOW": None, "PX_LAST": None, "PX_VOLUME": None, "TOTAL_RETURN_INDEX": 1.2345},
            "BBB": None,
            "CCC": None,
        },
        # Valid trading session for BBB with PX_LAST populated but TRI itself missing.
        {
            "date": "2021-03-08",
            "AAA": _full(10.5),
            "BBB": {"PX_OPEN": 20, "PX_HIGH": 21, "PX_LOW": 19, "PX_LAST": 20.5, "PX_VOLUME": 2000, "TOTAL_RETURN_INDEX": None},
            "CCC": _full(30.5),
        },
    ]
    write_prices_fixture(p, TICKERS, rows)

    out = normalize.normalize_prices(path=p, expected_universe=TICKERS)

    # AAA's 2021-03-06 row must be excluded despite a populated TRI value.
    aaa_dates = set(out.loc[out["ticker"] == "AAA", "date"])
    assert pd.Timestamp("2021-03-06") not in aaa_dates

    # BBB's 2021-03-08 row must be INCLUDED despite a missing TRI value,
    # and total_return_idx must be preserved as NaN, not fabricated.
    bbb_row = out[(out["ticker"] == "BBB") & (out["date"] == pd.Timestamp("2021-03-08"))]
    assert len(bbb_row) == 1
    assert pd.isna(bbb_row.iloc[0]["total_return_idx"])
    assert bbb_row.iloc[0]["close"] == 20.5


def test_normalize_prices_preserves_genuine_missing_values_no_fill(tmp_path):
    p = tmp_path / "prices_raw.csv"
    rows = [
        _row(
            "2021-03-08",
            aaa={"PX_OPEN": 10, "PX_HIGH": 11, "PX_LOW": None, "PX_LAST": 10.5, "PX_VOLUME": None, "TOTAL_RETURN_INDEX": 1.0},
            bbb=_full(20.5),
            ccc=_full(30.5),
        ),
        _row("2021-03-09", aaa=_full(11), bbb=_full(21), ccc=_full(31)),
    ]
    write_prices_fixture(p, TICKERS, rows)

    out = normalize.normalize_prices(path=p, expected_universe=TICKERS)

    first = out[(out["ticker"] == "AAA") & (out["date"] == pd.Timestamp("2021-03-08"))].iloc[0]
    assert pd.isna(first["low"])
    assert pd.isna(first["volume"])
    # No forward/back-fill: the next valid day's own volume must be exactly
    # what was supplied, not bled from the prior NaN row or vice versa.
    second = out[(out["ticker"] == "AAA") & (out["date"] == pd.Timestamp("2021-03-09"))].iloc[0]
    assert second["volume"] == 1000


def test_normalize_prices_output_schema_and_uniqueness(tmp_path):
    p = tmp_path / "prices_raw.csv"
    rows = [_row("2021-03-08", aaa=_full(10.5), bbb=_full(20.5), ccc=_full(30.5))]
    write_prices_fixture(p, TICKERS, rows)

    out = normalize.normalize_prices(path=p, expected_universe=TICKERS)

    assert list(out.columns) == ["ticker", "date", "open", "high", "low", "close", "volume", "total_return_idx"]
    assert not out.duplicated(subset=["ticker", "date"]).any()
    assert out["close"].notna().all()  # schema.sql: close is NOT NULL for loaded rows


# ---------------------------------------------------------------------------
# normalize_macro
# ---------------------------------------------------------------------------
def test_normalize_macro_preserves_nan_for_series_missing_a_date(tmp_path):
    p = tmp_path / "macro.csv"
    spx = [{"date": "2021-03-08", "value": 100}]
    vix = [{"date": "2021-03-08", "value": 20}, {"date": "2021-03-09", "value": 21}]
    usgg10yr = [{"date": "2021-03-08", "value": 1.5}, {"date": "2021-03-09", "value": 1.6}]
    write_macro_fixture(p, spx, vix, usgg10yr)

    out = normalize.normalize_macro(path=p)

    assert list(out.columns) == ["date", "spx", "vix", "yield_10y"]
    row_09 = out[out["date"] == pd.Timestamp("2021-03-09")].iloc[0]
    assert np.isnan(row_09["spx"])  # SPX had no quote that day — preserved as NaN, not dropped/fabricated
    assert row_09["vix"] == 21
    assert row_09["yield_10y"] == 1.6


# ---------------------------------------------------------------------------
# normalize_static_fields
# ---------------------------------------------------------------------------
def test_normalize_static_fields_renames_and_preserves_missing(tmp_path):
    p = tmp_path / "static_fields.csv"
    rows = [
        {"ticker": "AAA", "CUR_MKT_CAP": 1e12, "BETA_RAW_OVERRIDABLE": 1.1, "DIVIDEND_INDICATED_YIELD": 0.5, "GICS_SECTOR_NAME": "Information Technology"},
        {"ticker": "BBB", "CUR_MKT_CAP": 2e11, "BETA_RAW_OVERRIDABLE": 0.9, "DIVIDEND_INDICATED_YIELD": None, "GICS_SECTOR_NAME": "Financials"},
        {"ticker": "CCC", "CUR_MKT_CAP": 3e11, "BETA_RAW_OVERRIDABLE": 1.3, "DIVIDEND_INDICATED_YIELD": 1.2, "GICS_SECTOR_NAME": "Financials"},
    ]
    write_static_fields_fixture(p, rows)

    out = normalize.normalize_static_fields(path=p, expected_universe=TICKERS)

    assert list(out.columns) == ["ticker", "market_cap", "beta", "div_yield", "sector"]
    bbb = out.set_index("ticker").loc["BBB"]
    assert pd.isna(bbb["div_yield"])
    assert bbb["market_cap"] == 2e11
