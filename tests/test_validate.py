"""
Tests for pipeline.validate — Bloomberg raw-export structural parsing.

All tests run against small hand-built fixtures (tests/bloomberg_fixtures.py),
never against the real, gitignored data/raw/ export (TRD.md §17).
"""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline import validate
from tests.bloomberg_fixtures import (
    write_macro_fixture,
    write_prices_fixture,
    write_static_fields_fixture,
)

TICKERS = ["AAA", "BBB", "CCC"]


def _happy_rows():
    return [
        {"date": "2021-03-06"},  # Saturday — no field dict given -> all-#N/A / TRI=1 placeholder row
        {
            "date": "2021-03-08",  # Monday
            "AAA": {"PX_OPEN": 10, "PX_HIGH": 11, "PX_LOW": 9, "PX_LAST": 10.5, "PX_VOLUME": 1000, "TOTAL_RETURN_INDEX": 1.0},
            "BBB": {"PX_OPEN": 20, "PX_HIGH": 21, "PX_LOW": 19, "PX_LAST": 20.5, "PX_VOLUME": 2000, "TOTAL_RETURN_INDEX": 1.0},
            "CCC": {"PX_OPEN": 30, "PX_HIGH": 31, "PX_LOW": 29, "PX_LAST": 30.5, "PX_VOLUME": 3000, "TOTAL_RETURN_INDEX": 1.0},
        },
        {
            "date": "2021-03-09",  # Tuesday
            "AAA": {"PX_OPEN": 10.5, "PX_HIGH": 11.5, "PX_LOW": 10, "PX_LAST": 11, "PX_VOLUME": 1100, "TOTAL_RETURN_INDEX": 1.05},
            "BBB": {"PX_OPEN": 20.5, "PX_HIGH": 21.5, "PX_LOW": 20, "PX_LAST": 21, "PX_VOLUME": 2100, "TOTAL_RETURN_INDEX": 1.02},
            "CCC": {"PX_OPEN": 30.5, "PX_HIGH": 31.5, "PX_LOW": 30, "PX_LAST": 31, "PX_VOLUME": 3100, "TOTAL_RETURN_INDEX": 0.99},
        },
    ]


# ---------------------------------------------------------------------------
# parse_prices_raw
# ---------------------------------------------------------------------------
def test_parse_prices_raw_happy_path(tmp_path):
    p = tmp_path / "prices_raw.csv"
    write_prices_fixture(p, TICKERS, _happy_rows())

    result = validate.parse_prices_raw(path=p, expected_universe=TICKERS)

    assert sorted(result.tickers) == sorted(TICKERS)
    assert len(result.frame) == 3 * len(TICKERS)  # 3 calendar rows x 3 tickers
    assert set(result.frame.columns) == {
        "ticker", "date", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_LAST", "PX_VOLUME", "TOTAL_RETURN_INDEX",
    }
    # Weekend row: PX_LAST NaN for every ticker, but TOTAL_RETURN_INDEX populated.
    weekend = result.frame[result.frame["date"] == pd.Timestamp("2021-03-06")]
    assert weekend["PX_LAST"].isna().all()
    assert (weekend["TOTAL_RETURN_INDEX"] == 1.0).all()


def test_parse_prices_raw_stray_column_zero_is_ignored(tmp_path):
    """Column 0 is a leftover ticker-list artifact unrelated to the row's
    actual date/ticker content (see module docstring) — parsing must not use
    it as "the ticker for this row"."""
    p = tmp_path / "prices_raw.csv"
    write_prices_fixture(p, TICKERS, _happy_rows())

    result = validate.parse_prices_raw(path=p, expected_universe=TICKERS)

    # Every real ticker's rows come from its own header block, not column 0.
    for t in TICKERS:
        sub = result.frame[result.frame["ticker"] == t]
        assert len(sub) == 3


def test_parse_prices_raw_universe_mismatch(tmp_path):
    p = tmp_path / "prices_raw.csv"
    write_prices_fixture(p, TICKERS, _happy_rows())

    with pytest.raises(validate.UniverseMismatchError) as exc_info:
        validate.parse_prices_raw(path=p, expected_universe=TICKERS + ["DDD"])

    assert "DDD" in str(exc_info.value)


def test_parse_prices_raw_malformed_field_order(tmp_path):
    p = tmp_path / "prices_raw.csv"
    write_prices_fixture(p, TICKERS, _happy_rows(), corrupt_field_order_for="BBB")

    with pytest.raises(validate.MalformedExportError, match="field-name block"):
        validate.parse_prices_raw(path=p, expected_universe=TICKERS)


def test_parse_prices_raw_wrong_column_count(tmp_path):
    p = tmp_path / "prices_raw.csv"
    write_prices_fixture(p, TICKERS, _happy_rows(), drop_last_column=True)

    with pytest.raises(validate.MalformedExportError, match="column count"):
        validate.parse_prices_raw(path=p, expected_universe=TICKERS)


def test_parse_prices_raw_duplicate_ticker_header_block(tmp_path):
    p = tmp_path / "prices_raw.csv"
    dup_tickers = ["AAA", "BBB", "AAA"]
    write_prices_fixture(p, dup_tickers, _happy_rows())

    with pytest.raises(validate.MalformedExportError, match="duplicate ticker header"):
        validate.parse_prices_raw(path=p, expected_universe=dup_tickers)


def test_parse_prices_raw_duplicate_date_row(tmp_path):
    p = tmp_path / "prices_raw.csv"
    rows = _happy_rows()
    rows.append(rows[1])  # duplicate 2021-03-08
    write_prices_fixture(p, TICKERS, rows)

    with pytest.raises(validate.DuplicateObservationError, match="duplicate calendar date"):
        validate.parse_prices_raw(path=p, expected_universe=TICKERS)


# ---------------------------------------------------------------------------
# parse_macro_raw
# ---------------------------------------------------------------------------
def test_parse_macro_raw_independent_series_lengths(tmp_path):
    p = tmp_path / "macro.csv"
    spx = [{"date": "2021-03-08", "value": 100}, {"date": "2021-03-09", "value": 101}]
    vix = [{"date": "2021-03-08", "value": 20}, {"date": "2021-03-09", "value": 21}, {"date": "2021-03-10", "value": 22}]
    usgg10yr = [
        {"date": "2021-03-08", "value": 1.5},
        {"date": "2021-03-09", "value": 1.6},
        {"date": "2021-03-10", "value": 1.7},
        {"date": "2021-03-11", "value": 1.8},
    ]
    write_macro_fixture(p, spx, vix, usgg10yr)

    result = validate.parse_macro_raw(path=p)

    assert len(result.series["SPX"]) == 2
    assert len(result.series["VIX"]) == 3
    assert len(result.series["USGG10YR"]) == 4
    # The longer series' extra trailing dates must not leak into the shorter ones.
    assert pd.Timestamp("2021-03-11") not in set(result.series["SPX"]["date"])
    assert pd.Timestamp("2021-03-11") in set(result.series["USGG10YR"]["date"])


def test_parse_macro_raw_duplicate_date_in_series(tmp_path):
    p = tmp_path / "macro.csv"
    spx = [{"date": "2021-03-08", "value": 100}, {"date": "2021-03-08", "value": 101}]
    vix = [{"date": "2021-03-08", "value": 20}]
    usgg10yr = [{"date": "2021-03-08", "value": 1.5}]
    write_macro_fixture(p, spx, vix, usgg10yr)

    with pytest.raises(validate.DuplicateObservationError, match="SPX_DATE"):
        validate.parse_macro_raw(path=p)


def test_parse_macro_raw_wrong_columns(tmp_path):
    p = tmp_path / "macro.csv"
    p.write_text("DATE,SPX\n2021-03-08,100\n")

    with pytest.raises(validate.MalformedExportError, match="expected columns"):
        validate.parse_macro_raw(path=p)


# ---------------------------------------------------------------------------
# parse_static_fields_raw
# ---------------------------------------------------------------------------
def test_parse_static_fields_raw_happy_path_preserves_missing(tmp_path):
    p = tmp_path / "static_fields.csv"
    rows = [
        {"ticker": "AAA", "CUR_MKT_CAP": 1e12, "BETA_RAW_OVERRIDABLE": 1.1, "DIVIDEND_INDICATED_YIELD": 0.5, "GICS_SECTOR_NAME": "Information Technology"},
        {"ticker": "BBB", "CUR_MKT_CAP": 2e11, "BETA_RAW_OVERRIDABLE": 0.9, "DIVIDEND_INDICATED_YIELD": None, "GICS_SECTOR_NAME": "Financials"},
        {"ticker": "CCC", "CUR_MKT_CAP": 3e11, "BETA_RAW_OVERRIDABLE": 1.3, "DIVIDEND_INDICATED_YIELD": 1.2, "GICS_SECTOR_NAME": "Financials"},
    ]
    write_static_fields_fixture(p, rows)

    result = validate.parse_static_fields_raw(path=p, expected_universe=TICKERS)

    assert sorted(result.frame["ticker"]) == sorted(TICKERS)
    bbb = result.frame.set_index("ticker").loc["BBB"]
    assert pd.isna(bbb["DIVIDEND_INDICATED_YIELD"])  # genuine missing preserved, not zero-filled


def test_parse_static_fields_raw_universe_mismatch(tmp_path):
    p = tmp_path / "static_fields.csv"
    rows = [
        {"ticker": "AAA", "CUR_MKT_CAP": 1e12, "BETA_RAW_OVERRIDABLE": 1.1, "DIVIDEND_INDICATED_YIELD": 0.5, "GICS_SECTOR_NAME": "Information Technology"},
        {"ticker": "BBB", "CUR_MKT_CAP": 2e11, "BETA_RAW_OVERRIDABLE": 0.9, "DIVIDEND_INDICATED_YIELD": 0.1, "GICS_SECTOR_NAME": "Financials"},
    ]
    write_static_fields_fixture(p, rows)  # missing CCC

    with pytest.raises(validate.UniverseMismatchError):
        validate.parse_static_fields_raw(path=p, expected_universe=TICKERS)


def test_parse_static_fields_raw_duplicate_ticker(tmp_path):
    p = tmp_path / "static_fields.csv"
    rows = [
        {"ticker": "AAA", "CUR_MKT_CAP": 1e12, "BETA_RAW_OVERRIDABLE": 1.1, "DIVIDEND_INDICATED_YIELD": 0.5, "GICS_SECTOR_NAME": "Information Technology"},
        {"ticker": "AAA", "CUR_MKT_CAP": 1e12, "BETA_RAW_OVERRIDABLE": 1.1, "DIVIDEND_INDICATED_YIELD": 0.5, "GICS_SECTOR_NAME": "Information Technology"},
        {"ticker": "BBB", "CUR_MKT_CAP": 2e11, "BETA_RAW_OVERRIDABLE": 0.9, "DIVIDEND_INDICATED_YIELD": 0.1, "GICS_SECTOR_NAME": "Financials"},
        {"ticker": "CCC", "CUR_MKT_CAP": 3e11, "BETA_RAW_OVERRIDABLE": 1.3, "DIVIDEND_INDICATED_YIELD": 1.2, "GICS_SECTOR_NAME": "Financials"},
    ]
    write_static_fields_fixture(p, rows)

    with pytest.raises(validate.DuplicateObservationError):
        validate.parse_static_fields_raw(path=p, expected_universe=TICKERS)
