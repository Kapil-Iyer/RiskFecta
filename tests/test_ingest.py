"""
CI-safe unit tests for pipeline.ingest — volume BIGINT casting and DB-record
preparation. No database connection required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.ingest import PRICE_COLUMNS, _prepare_prices_for_insert, cast_volume_to_bigint


# ---------------------------------------------------------------------------
# cast_volume_to_bigint
# ---------------------------------------------------------------------------
def test_cast_volume_to_bigint_preserves_integral_values():
    s = pd.Series([1000.0, 2000.0, 0.0])
    out = cast_volume_to_bigint(s)
    assert out.dtype == "Int64"
    assert list(out) == [1000, 2000, 0]


def test_cast_volume_to_bigint_preserves_null_as_na():
    s = pd.Series([1000.0, np.nan, 3000.0])
    out = cast_volume_to_bigint(s)
    assert out.iloc[1] is pd.NA
    assert out.iloc[0] == 1000
    assert out.iloc[2] == 3000


def test_cast_volume_to_bigint_rejects_genuinely_fractional_values():
    s = pd.Series([1000.0, 2000.5, 3000.0])
    with pytest.raises(ValueError, match="fractional"):
        cast_volume_to_bigint(s)


def test_cast_volume_to_bigint_does_not_silently_round():
    """A fractional value must raise, not be quietly rounded into a plausible-looking int."""
    s = pd.Series([1234.9])
    with pytest.raises(ValueError):
        cast_volume_to_bigint(s)


def test_cast_volume_to_bigint_handles_large_realistic_volumes():
    # Real Bloomberg PX_VOLUME magnitudes (hundreds of millions of shares).
    s = pd.Series([116307892.0, 4025093.0])
    out = cast_volume_to_bigint(s)
    assert list(out) == [116307892, 4025093]


def test_cast_volume_to_bigint_rejects_non_numeric_garbage():
    s = pd.Series(["not-a-number"])
    with pytest.raises((ValueError, TypeError)):
        cast_volume_to_bigint(s)


# ---------------------------------------------------------------------------
# _prepare_prices_for_insert
# ---------------------------------------------------------------------------
def _sample_df():
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": pd.to_datetime(["2021-03-08", "2021-03-08"]),
            "open": [10.0, np.nan],
            "high": [11.0, 21.0],
            "low": [9.0, 19.0],
            "close": [10.5, 20.5],
            "volume": [1000.0, np.nan],
            "total_return_idx": [1.0, np.nan],
        }
    )[PRICE_COLUMNS]


def test_prepare_prices_for_insert_converts_nan_to_none():
    records = _prepare_prices_for_insert(_sample_df())
    assert len(records) == 2
    # BBB row: open, volume, total_return_idx were NaN -> must become None, not NaN/0.
    bbb = records[1]
    ticker, date, open_, high, low, close, volume, tri = bbb
    assert ticker == "BBB"
    assert open_ is None
    assert volume is None
    assert tri is None
    assert close == 20.5


def test_prepare_prices_for_insert_converts_date_to_python_date():
    import datetime

    records = _prepare_prices_for_insert(_sample_df())
    assert isinstance(records[0][1], datetime.date)


def test_prepare_prices_for_insert_volume_is_plain_int_not_float():
    records = _prepare_prices_for_insert(_sample_df())
    volume = records[0][6]
    assert isinstance(volume, int)
    assert volume == 1000


def test_prepare_prices_for_insert_propagates_fractional_volume_error():
    df = _sample_df()
    df.loc[0, "volume"] = 1000.5
    with pytest.raises(ValueError, match="fractional"):
        _prepare_prices_for_insert(df)
