"""
Optional local-only integration tests against the REAL Bloomberg export
(data/raw/*.csv). Per TRD.md §17, CI must not require this gitignored,
machine-local data — every test here is skipped automatically if the file is
absent (e.g. in CI). Run locally with: pytest -m integration
"""
from __future__ import annotations

import pandas as pd
import pytest

import config
from pipeline import normalize, validate

pytestmark = pytest.mark.integration

PRICES_PATH = config.DATA_RAW / "prices_raw.csv"
MACRO_PATH = config.DATA_RAW / "macro.csv"
STATIC_PATH = config.DATA_RAW / "static_fields.csv"

_skip_no_data = pytest.mark.skipif(
    not (PRICES_PATH.exists() and MACRO_PATH.exists() and STATIC_PATH.exists()),
    reason="real Bloomberg exports not present under data/raw/ (expected in CI / clean checkouts)",
)


@_skip_no_data
def test_real_universe_is_exactly_config_universe():
    parsed = validate.parse_prices_raw()
    assert len(parsed.tickers) == 50
    assert set(parsed.tickers) == set(config.TICKER_UNIVERSE)
    assert "MRSH" in parsed.tickers
    assert "MMC" not in parsed.tickers


@_skip_no_data
def test_real_prices_normalize_session_filtering_and_uniqueness():
    out = normalize.normalize_prices()

    assert set(out["ticker"]) == set(config.TICKER_UNIVERSE)
    assert not out.duplicated(subset=["ticker", "date"]).any()
    assert out["close"].notna().all()

    # Every ticker in this export has an identical, fully clean trading
    # calendar (confirmed during the Phase 1A audit): exactly 1256 valid
    # sessions each, starting 2021-03-01. This test intentionally fixes that
    # observed fact so a future re-export that changes it is caught, not
    # silently accepted.
    counts = out.groupby("ticker").size()
    assert (counts == 1256).all()
    assert out["date"].min() == pd.Timestamp("2021-03-01")


@_skip_no_data
def test_real_prices_normalize_excludes_all_weekend_rows():
    out = normalize.normalize_prices()
    assert (out["date"].dt.dayofweek < 5).all()


@_skip_no_data
def test_real_macro_normalize_has_no_fabricated_values():
    out = normalize.normalize_macro()
    # Each series' known real length (from the Phase 1A audit) — a stray
    # fabricated/extra row would move these counts.
    assert out["spx"].notna().sum() == 1256
    assert out["vix"].notna().sum() == 1285
    assert out["yield_10y"].notna().sum() == 1303


@_skip_no_data
def test_real_static_fields_preserve_known_missing_dividend_yield():
    out = normalize.normalize_static_fields()
    assert len(out) == 50
    # ADBE is a known non-dividend-payer in the real export (Phase 1A audit).
    adbe = out.set_index("ticker").loc["ADBE"]
    assert pd.isna(adbe["div_yield"])
