"""
Phase 1B local-only integration tests against the REAL Supabase-hosted
PostgreSQL database. Per TRD.md §17/§18, ordinary unit tests must never
depend on credentials — every test here is skipped automatically when
DATABASE_URL is not set (e.g. in CI). Run locally with: pytest -m db

These tests never print, log, or assert on the DATABASE_URL value itself.
"""
from __future__ import annotations

import os

import pandas as pd
import pytest

import config
from pipeline import db, ingest, normalize

pytestmark = pytest.mark.db

_skip_no_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set in the environment (expected in CI / clean checkouts)",
)

EXPECTED_TABLES = ["prices_raw", "features", "predictions", "portfolios", "risk_metrics"]
EXPECTED_ROW_COUNT = 62_800  # 50 tickers x 1256 valid sessions each (Phase 1A audit)


def _fetch_prices_raw_as_frame(conn) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT ticker, date, open, high, low, close, volume, total_return_idx "
            "FROM prices_raw ORDER BY ticker, date"
        )
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    out = pd.DataFrame(rows, columns=cols)
    out["date"] = pd.to_datetime(out["date"])
    for c in ["open", "high", "low", "close", "total_return_idx"]:
        out[c] = out[c].astype(float)
    out["volume"] = out["volume"].astype("Float64")
    return out


@pytest.fixture(scope="module")
def live_conn():
    """One reusable read connection for the module, in autocommit mode so
    read-only assertions never hold an idle transaction open. Schema is
    (re-)applied and prices are (re-)ingested first so these tests observe a
    known, current state regardless of what ran before."""
    conn = db.get_connection()
    conn.autocommit = True
    db.apply_schema(conn)
    ingest.ingest_prices(conn=conn)
    yield conn
    conn.close()


# ---------------------------------------------------------------------------
# Connection + schema
# ---------------------------------------------------------------------------
@_skip_no_db
def test_connection_succeeds():
    conn = db.get_connection()
    try:
        assert conn.closed == 0
    finally:
        conn.close()


@_skip_no_db
def test_schema_apply_creates_all_expected_tables(live_conn):
    for table in EXPECTED_TABLES:
        assert db.table_exists(live_conn, table), f"missing table: {table}"


@_skip_no_db
def test_prices_raw_column_types_and_nullability(live_conn):
    cols = {c.name: c for c in db.list_columns(live_conn, "prices_raw")}
    assert cols["close"].is_nullable is False  # session-gated NOT NULL, per schema.sql
    assert cols["total_return_idx"].is_nullable is True
    assert cols["volume"].data_type == "bigint"


@_skip_no_db
def test_prices_raw_has_unique_ticker_date_constraint(live_conn):
    constraints = db.unique_constraint_columns(live_conn, "prices_raw")
    assert ["ticker", "date"] in constraints


# ---------------------------------------------------------------------------
# Ingested data integrity
# ---------------------------------------------------------------------------
@_skip_no_db
def test_price_row_count_matches_expected(live_conn):
    n = db.fetch_scalar(live_conn, "SELECT COUNT(*) FROM prices_raw")
    assert n == EXPECTED_ROW_COUNT


@_skip_no_db
def test_universe_is_exactly_config_universe_mrsh_present_mmc_absent(live_conn):
    with live_conn.cursor() as cur:
        cur.execute("SELECT DISTINCT ticker FROM prices_raw")
        tickers = {r[0] for r in cur.fetchall()}
    assert len(tickers) == 50
    assert tickers == set(config.TICKER_UNIVERSE)
    assert "MRSH" in tickers
    assert "MMC" not in tickers


@_skip_no_db
def test_ticker_date_logical_uniqueness(live_conn):
    dup = db.fetch_scalar(
        live_conn,
        "SELECT COUNT(*) FROM (SELECT ticker, date FROM prices_raw GROUP BY ticker, date HAVING COUNT(*) > 1) t",
    )
    assert dup == 0


@_skip_no_db
def test_no_weekend_rows(live_conn):
    n = db.fetch_scalar(live_conn, "SELECT COUNT(*) FROM prices_raw WHERE EXTRACT(DOW FROM date) IN (0, 6)")
    assert n == 0


@_skip_no_db
def test_min_max_date_range(live_conn):
    lo, hi = db.fetch_scalar(live_conn, "SELECT MIN(date) FROM prices_raw"), db.fetch_scalar(
        live_conn, "SELECT MAX(date) FROM prices_raw"
    )
    assert str(lo) == "2021-03-01"
    assert str(hi) == "2026-02-27"


@_skip_no_db
def test_close_is_never_null(live_conn):
    n = db.fetch_scalar(live_conn, "SELECT COUNT(*) FROM prices_raw WHERE close IS NULL")
    assert n == 0


@_skip_no_db
def test_full_normalized_vs_db_value_fidelity(live_conn):
    """Phase 1C closeout check: compare EVERY normalized row (all 62,800),
    not a sample, against the live DB. NUMERIC(12,4) columns are compared
    with a 5e-4 tolerance to account for the schema's intentional 4-decimal
    storage precision (the raw export carries up to 8-9 decimal places on
    some fields) — anything beyond that tolerance is a genuine mismatch.
    """
    normalized = normalize.normalize_prices().sort_values(["ticker", "date"]).reset_index(drop=True)
    db_df = _fetch_prices_raw_as_frame(live_conn)

    assert len(normalized) == len(db_df) == EXPECTED_ROW_COUNT

    merged = normalized.merge(db_df, on=["ticker", "date"], suffixes=("_norm", "_db"), how="outer", indicator=True)
    assert (merged["_merge"] == "both").all(), "row present in only one of normalize output / DB"

    for col in ["open", "high", "low", "close", "total_return_idx"]:
        a, b = merged[f"{col}_norm"], merged[f"{col}_db"]
        both_nan = a.isna() & b.isna()
        mismatched_nullness = a.isna() != b.isna()
        value_diff_too_large = ~both_nan & ~mismatched_nullness & ((a - b).abs() > 5e-4)
        assert not mismatched_nullness.any(), f"{col}: NULL-ness disagreement between normalize output and DB"
        assert not value_diff_too_large.any(), f"{col}: value(s) differ by more than schema NUMERIC(12,4) precision"

    vol_norm, vol_db = merged["volume_norm"], merged["volume_db"]
    both_nan_v = vol_norm.isna() & vol_db.isna()
    mismatched_v = ~both_nan_v & ((vol_norm.isna() != vol_db.isna()) | ((vol_norm - vol_db).abs() > 0.5))
    assert not mismatched_v.any(), "volume: mismatch between normalize output and DB"


@_skip_no_db
def test_every_ticker_has_exactly_1256_sessions(live_conn):
    with live_conn.cursor() as cur:
        cur.execute("SELECT ticker, COUNT(*) FROM prices_raw GROUP BY ticker")
        counts = dict(cur.fetchall())
    assert set(counts.values()) == {1256}


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
@_skip_no_db
def test_reingestion_does_not_duplicate_rows():
    conn = db.get_connection()
    try:
        r1 = ingest.ingest_prices(conn=conn)
        r2 = ingest.ingest_prices(conn=conn)
        assert r1.rows_in_table_after == EXPECTED_ROW_COUNT
        assert r2.rows_in_table_after == EXPECTED_ROW_COUNT
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# NULL preservation + failure/rollback behavior
# (each uses its own dedicated, explicitly-rolled-back connection/transaction
#  so the shared read fixture's view of the real data is never disturbed)
# ---------------------------------------------------------------------------
@_skip_no_db
def test_null_values_round_trip_as_sql_null():
    conn = db.get_connection()
    try:
        synthetic = pd.DataFrame(
            {
                "ticker": ["ZZZTEST_NULLCHECK"],
                "date": pd.to_datetime(["2099-01-01"]),
                "open": [None],
                "high": [None],
                "low": [None],
                "close": [123.45],
                "volume": [None],
                "total_return_idx": [None],
            }
        )
        ingest.upsert_prices(conn, synthetic)  # not yet committed
        with conn.cursor() as cur:
            cur.execute(
                "SELECT open, volume, total_return_idx, close FROM prices_raw WHERE ticker = 'ZZZTEST_NULLCHECK'"
            )
            open_, volume, tri, close = cur.fetchone()
        assert open_ is None
        assert volume is None
        assert tri is None
        assert float(close) == 123.45
    finally:
        conn.rollback()  # never persist the synthetic row
        conn.close()


@_skip_no_db
def test_not_null_violation_rolls_back_entire_batch_no_partial_write():
    conn = db.get_connection()
    try:
        before = db.fetch_scalar(conn, "SELECT COUNT(*) FROM prices_raw")

        bad_batch = pd.DataFrame(
            {
                "ticker": ["ZZZTEST_ROLLBACK_A", "ZZZTEST_ROLLBACK_B"],
                "date": pd.to_datetime(["2099-01-02", "2099-01-03"]),
                "open": [1.0, 1.0],
                "high": [1.0, 1.0],
                "low": [1.0, 1.0],
                "close": [100.0, None],  # violates prices_raw.close NOT NULL
                "volume": [1.0, 1.0],
                "total_return_idx": [1.0, 1.0],
            }
        )
        with pytest.raises(Exception):
            ingest.upsert_prices(conn, bad_batch)
        conn.rollback()

        after = db.fetch_scalar(conn, "SELECT COUNT(*) FROM prices_raw")
        assert after == before  # no partial write from the failed batch

        leaked = db.fetch_scalar(
            conn, "SELECT COUNT(*) FROM prices_raw WHERE ticker LIKE 'ZZZTEST_ROLLBACK%'"
        )
        assert leaked == 0
    finally:
        conn.rollback()
        conn.close()


@_skip_no_db
def test_ingest_prices_transaction_rolls_back_on_bad_input():
    """ingest_prices itself (not just the low-level upsert helper) must not
    leave a committed partial write when the batch is rejected by the DB."""
    conn = db.get_connection()
    conn.autocommit = True
    try:
        before = db.fetch_scalar(conn, "SELECT COUNT(*) FROM prices_raw")
        bad = pd.DataFrame(
            {
                "ticker": ["ZZZTEST_INGEST_FAIL"],
                "date": pd.to_datetime(["2099-01-04"]),
                "open": [1.0],
                "high": [1.0],
                "low": [1.0],
                "close": [None],  # violates NOT NULL
                "volume": [1.0],
                "total_return_idx": [1.0],
            }
        )
        with pytest.raises(Exception):
            ingest.ingest_prices(conn=conn, normalized=bad)
        after = db.fetch_scalar(conn, "SELECT COUNT(*) FROM prices_raw")
        assert after == before
    finally:
        conn.close()
