"""
RiskFecta Phase 7B SPXT amendment — official S&P 500 Total Return (SPXT)
benchmark tests (optimizer/benchmark_spxt.py). Task brief §12.

Synthetic by default. A few tests are marked `integration`/`db` and are
READ-ONLY against the real `data/raw/spxt_benchmark.csv` artifact / the
real database (never write) — skipped automatically when unavailable,
same convention as the rest of the suite.
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pandas as pd
import pytest

import config
import optimizer.benchmark_spxt as spxt_mod
import optimizer.covariance as cov_mod
import optimizer.portfolio as port_mod
import optimizer.walkforward as wf_mod
from optimizer.benchmark_spxt import (
    SPXT_RAW_PATH,
    load_spxt_raw,
    spxt_total_return,
    verify_spxt_covers_formation_calendar,
)

_skip_no_spxt_file = pytest.mark.skipif(
    not SPXT_RAW_PATH.exists(),
    reason="real data/raw/spxt_benchmark.csv not present (expected in CI / clean checkouts)",
)
_skip_no_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set in the environment (expected in CI / clean checkouts)",
)


# ---------------------------------------------------------------------------
# 1. Loader reads the real artifact correctly
# ---------------------------------------------------------------------------
@pytest.mark.integration
@_skip_no_spxt_file
def test_load_spxt_raw_reads_real_artifact():
    df = load_spxt_raw()
    assert list(df.columns) == ["date", "spxt_px_last"]
    assert len(df) > 1000
    assert df["date"].min() <= pd.Timestamp("2021-03-01")
    assert df["date"].max() >= pd.Timestamp("2026-02-27")
    assert df["spxt_px_last"].gt(0).all()
    assert df["spxt_px_last"].apply(np.isfinite).all()


# ---------------------------------------------------------------------------
# 2. duplicate dates fail
# ---------------------------------------------------------------------------
def test_load_spxt_raw_duplicate_date_fails_loudly(tmp_path):
    p = tmp_path / "spxt.csv"
    p.write_text("SPXT_DATE,SPXT_PX_LAST\n2022-01-03,4000\n2022-01-03,4001\n2022-01-04,4010\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_spxt_raw(p)


# ---------------------------------------------------------------------------
# 3. non-numeric/invalid required values fail
# ---------------------------------------------------------------------------
def test_load_spxt_raw_non_numeric_fails_loudly(tmp_path):
    p = tmp_path / "spxt.csv"
    p.write_text("SPXT_DATE,SPXT_PX_LAST\n2022-01-03,not_a_number\n2022-01-04,4010\n")
    with pytest.raises(ValueError, match="non-finite|non-positive|non-numeric"):
        load_spxt_raw(p)


def test_load_spxt_raw_non_positive_fails_loudly(tmp_path):
    p = tmp_path / "spxt.csv"
    p.write_text("SPXT_DATE,SPXT_PX_LAST\n2022-01-03,0\n2022-01-04,4010\n")
    with pytest.raises(ValueError, match="non-finite|non-positive|non-numeric"):
        load_spxt_raw(p)


def test_load_spxt_raw_missing_column_fails_loudly(tmp_path):
    p = tmp_path / "spxt.csv"
    p.write_text("SPXT_DATE,WRONG_COL\n2022-01-03,4000\n")
    with pytest.raises(ValueError, match="missing expected column"):
        load_spxt_raw(p)


# ---------------------------------------------------------------------------
# 4/5. missing exact formation/T+21 endpoint fails
# ---------------------------------------------------------------------------
def _spxt_df():
    dates = pd.bdate_range("2022-01-03", periods=30)
    return pd.DataFrame({"date": dates, "spxt_px_last": np.linspace(4000, 4100, len(dates))})


def test_spxt_total_return_missing_formation_endpoint_fails_loudly():
    df = _spxt_df()
    missing_date = pd.Timestamp("2019-01-01")
    with pytest.raises(ValueError, match="formation_date"):
        spxt_total_return(df, missing_date, df["date"].iloc[5])


def test_spxt_total_return_missing_target_endpoint_fails_loudly():
    df = _spxt_df()
    missing_date = pd.Timestamp("2019-01-01")
    with pytest.raises(ValueError, match="target_date"):
        spxt_total_return(df, df["date"].iloc[0], missing_date)


# ---------------------------------------------------------------------------
# 6. exact formula
# ---------------------------------------------------------------------------
def test_spxt_total_return_exact_formula():
    df = _spxt_df()
    fd, td = df["date"].iloc[0], df["date"].iloc[10]
    out = spxt_total_return(df, fd, td)
    levels = df.set_index("date")["spxt_px_last"]
    expected = levels[td] / levels[fd] - 1.0
    assert out == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 7/8/9. no nearest-date / interpolation / fill substitution
# ---------------------------------------------------------------------------
def test_spxt_total_return_never_substitutes_nearest_date():
    df = _spxt_df()
    saturday = df["date"].iloc[0] + pd.Timedelta(days=5)  # 2022-01-03 (Mon) + 5d = 2022-01-08 (Sat)
    assert saturday.day_name() == "Saturday"
    assert saturday not in set(df["date"])
    with pytest.raises(ValueError):
        spxt_total_return(df, saturday, df["date"].iloc[5])


def test_spxt_module_never_calls_interpolation_or_fill_helpers():
    # Check actual call sites / method-argument literals, not docstring
    # prose (this module's docstrings deliberately explain, in English,
    # what it does NOT do — e.g. "never a nearest-date lookup" — which
    # would otherwise false-positive a naive substring search).
    src = inspect.getsource(spxt_mod)
    forbidden_calls = (".interpolate(", ".fillna(", ".ffill(", ".bfill(", ".reindex(", ".asof(")
    for forbidden in forbidden_calls:
        assert forbidden not in src, f"found forbidden fill/substitution call: {forbidden!r}"
    for literal in ('"nearest"', "'nearest'"):
        assert literal not in src, f"found forbidden method-argument literal: {literal!r}"


# ---------------------------------------------------------------------------
# verify_spxt_covers_formation_calendar
# ---------------------------------------------------------------------------
def test_verify_spxt_covers_formation_calendar_passes_when_complete():
    dates = pd.bdate_range("2022-01-03", periods=60)
    spxt = pd.DataFrame({"date": dates, "spxt_px_last": np.linspace(4000, 4200, len(dates))})
    calendar = pd.DataFrame({
        wf_mod.FORECAST_DATE_COL: [dates[0], dates[20]],
        wf_mod.TARGET_DATE_COL: [dates[21], dates[41]],
    })
    verify_spxt_covers_formation_calendar(spxt, calendar)  # must not raise


def test_verify_spxt_covers_formation_calendar_fails_loudly_on_gap():
    dates = pd.bdate_range("2022-01-03", periods=60)
    spxt = pd.DataFrame({"date": dates[:30], "spxt_px_last": np.linspace(4000, 4100, 30)})  # too short
    calendar = pd.DataFrame({
        wf_mod.FORECAST_DATE_COL: [dates[0], dates[40]],
        wf_mod.TARGET_DATE_COL: [dates[21], dates[59]],
    })
    with pytest.raises(ValueError, match="missing an exact SPXT quote"):
        verify_spxt_covers_formation_calendar(spxt, calendar)


@pytest.mark.db
@_skip_no_db
@pytest.mark.integration
@_skip_no_spxt_file
def test_real_spxt_covers_every_endpoint_of_the_real_47_date_calendar():
    from pipeline import db as db_mod
    conn = db_mod.get_connection()
    try:
        conn.autocommit = True
        calendar = wf_mod.load_formation_calendar(conn)
    finally:
        conn.close()
    spxt = load_spxt_raw()
    verify_spxt_covers_formation_calendar(spxt, calendar)  # must not raise: full coverage


# ---------------------------------------------------------------------------
# 10. confirmation SPX_PX_LAST is no longer the official benchmark /
#     old diagnostic is clearly deprecated
# ---------------------------------------------------------------------------
def test_old_spx_price_diagnostic_is_clearly_deprecated():
    doc = wf_mod.spx_price_return_diagnostic.__doc__ or ""
    assert "DEPRECATED" in doc
    assert "NON-OFFICIAL" in doc
    assert "optimizer.benchmark_spxt" in doc or "spxt_total_return" in doc


def test_strategies_never_includes_an_spx_or_spxt_entry():
    for s in wf_mod.STRATEGIES:
        assert "SPX" not in s  # neither SPX nor SPXT is one of the 5 official strategies


# ---------------------------------------------------------------------------
# 12/19. SPXT is evaluation-only / construction code is unchanged
# ---------------------------------------------------------------------------
def test_construction_modules_never_import_benchmark_spxt():
    for mod in (cov_mod, port_mod):
        assert not hasattr(mod, "benchmark_spxt")
        src = inspect.getsource(mod)
        assert "benchmark_spxt" not in src
        assert "spxt" not in src.lower()


def test_construct_portfolios_at_source_never_references_spxt():
    src = inspect.getsource(wf_mod.construct_portfolios_at)
    assert "spxt" not in src.lower()
    assert "benchmark_spxt" not in src


def test_construct_portfolios_at_signature_unchanged_by_spxt_amendment():
    sig = inspect.signature(wf_mod.construct_portfolios_at)
    assert set(sig.parameters) == {"formation_date", "mu", "prices", "usgg10yr", "tickers", "max_weight"}


# ---------------------------------------------------------------------------
# 13-18. covariance / expected-return / optimizer / strategy definitions /
# equal-weight all unchanged by this amendment (regression-style checks).
# ---------------------------------------------------------------------------
def test_locks_unchanged_by_spxt_amendment():
    assert config.COVAR_WINDOW == 252
    assert config.MAX_WEIGHT == 0.10
    assert wf_mod.OPTIMIZED_STRATEGIES == ("SAMPLE_MINVOL", "SAMPLE_MAXSHARPE", "LW_MINVOL", "LW_MAXSHARPE")
    assert wf_mod.STRATEGIES == wf_mod.OPTIMIZED_STRATEGIES + ("EQUAL_WEIGHT",)


def test_equal_weight_still_exactly_1_over_50():
    from optimizer.portfolio import equal_weight_benchmark
    w = equal_weight_benchmark(50)
    assert np.allclose(w, 0.02)


# ---------------------------------------------------------------------------
# 19/20. sealed / extension files are not referenced by any Phase 7 code path
# ---------------------------------------------------------------------------
FORBIDDEN_FILENAMES = (
    "prices_sealed.csv", "macro_sealed.csv", "prices_extension.csv", "macro_extension.csv",
)


def test_no_phase7_module_references_sealed_or_extension_filenames():
    modules = [cov_mod, port_mod, wf_mod, spxt_mod]
    import optimizer.persistence as persist_mod
    modules.append(persist_mod)
    for mod in modules:
        src = inspect.getsource(mod)
        for forbidden in FORBIDDEN_FILENAMES:
            assert forbidden not in src, f"{mod.__name__} references forbidden filename {forbidden!r}"


# ---------------------------------------------------------------------------
# 17. formula sanity: exact 47 formation dates / T+21 mappings unchanged
#     (re-confirms the Phase 7B finding; read-only)
# ---------------------------------------------------------------------------
@pytest.mark.db
@_skip_no_db
def test_real_formation_calendar_unchanged_by_spxt_amendment():
    from pipeline import db as db_mod
    conn = db_mod.get_connection()
    try:
        conn.autocommit = True
        df = wf_mod.load_formation_calendar(conn)
    finally:
        conn.close()
    assert df[wf_mod.FORECAST_DATE_COL].nunique() == 47
    assert df[wf_mod.FORECAST_DATE_COL].min() == pd.Timestamp("2022-02-25")
    assert df[wf_mod.FORECAST_DATE_COL].max() == pd.Timestamp("2026-01-02")
    assert wf_mod.verify_sequential_non_overlapping_path(df) is True
    assert df[wf_mod.TARGET_DATE_COL].max() < pd.Timestamp("2026-03-01")
