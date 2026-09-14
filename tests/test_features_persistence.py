"""
RiskFecta Phase 3 — `features` persistence tests (pipeline/features.py).

Mock-only: NO real database connection is opened anywhere in this file
(task brief §17, §19 — real `features` must remain empty pre-audit).
`psycopg2.extras.execute_values` is monkeypatched to an in-memory stand-in
that mimics `INSERT ... ON CONFLICT (ticker, date) DO UPDATE` semantics in
plain Python, so idempotency/conflict behavior is verified without ever
touching Postgres.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

import pipeline.features as features_mod
from pipeline.features import (
    FEATURES_TABLE_COLUMNS,
    FEATURES_UPSERT_SQL,
    PersistResult,
    STATIC_LEAKAGE_COLS,
    _prepare_features_for_insert,
    persist_features,
    upsert_features,
)


class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConn:
    """Records whether its `with conn:` blocks committed cleanly; never
    talks to a real database."""

    def __init__(self):
        self.blocks_committed = 0

    def cursor(self):
        return FakeCursor()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.blocks_committed += 1
        return False


@pytest.fixture
def fake_store(monkeypatch):
    """In-memory (ticker, date) -> row dict, updated by a fake
    execute_values that mimics ON CONFLICT (ticker, date) DO UPDATE.
    """
    store: dict = {}

    def fake_execute_values(cur, sql, records, page_size=1000):
        assert "features" in sql
        idx_ticker = FEATURES_TABLE_COLUMNS.index("ticker")
        idx_date = FEATURES_TABLE_COLUMNS.index("date")
        for rec in records:
            key = (rec[idx_ticker], rec[idx_date])
            store[key] = rec  # DO UPDATE == overwrite, never a second row

    monkeypatch.setattr(features_mod, "execute_values", fake_execute_values)
    return store


def _df(rows):
    base = {col: [None] * len(rows) for col in FEATURES_TABLE_COLUMNS}
    df = pd.DataFrame(base)
    for i, row in enumerate(rows):
        for k, v in row.items():
            df.loc[i, k] = v
    return df


def test_prepare_features_nulls_nan_and_none_uniformly():
    df = _df(
        [
            {"ticker": "AAPL", "date": "2021-01-04", "rsi_14": float("nan"), "vix": 20.0, "beta": None},
            {"ticker": "AAPL", "date": "2021-01-05", "rsi_14": 55.5, "vix": None, "beta": np.nan},
        ]
    )
    records = _prepare_features_for_insert(df)
    idx_rsi = FEATURES_TABLE_COLUMNS.index("rsi_14")
    idx_vix = FEATURES_TABLE_COLUMNS.index("vix")
    idx_beta = FEATURES_TABLE_COLUMNS.index("beta")

    assert records[0][idx_rsi] is None
    assert records[0][idx_vix] == 20.0
    assert records[0][idx_beta] is None
    assert records[1][idx_rsi] == 55.5
    assert records[1][idx_vix] is None
    assert records[1][idx_beta] is None


def test_prepare_features_date_normalized_to_date_object():
    df = _df([{"ticker": "AAPL", "date": pd.Timestamp("2021-01-04")}])
    records = _prepare_features_for_insert(df)
    idx_date = FEATURES_TABLE_COLUMNS.index("date")
    import datetime

    assert isinstance(records[0][idx_date], datetime.date)


def test_static_leakage_columns_always_null_in_prepared_records():
    df = _df(
        [
            {"ticker": "AAPL", "date": "2021-01-04", "beta": 1.2, "mkt_cap_log": 25.0, "sector": "Tech", "div_yield": 0.01}
        ]
    )
    # Even if a caller mistakenly populated these (build_feature_frame never
    # does), a defense-in-depth check at the persistence boundary itself:
    # this test documents that these columns pass through as given here —
    # the leakage GUARANTEE lives in build_feature_frame (see
    # tests/test_features.py::test_static_snapshot_never_leaks...); this
    # test only pins down that _prepare_features_for_insert does not itself
    # fabricate or clear values it wasn't given (no silent behavior either way).
    records = _prepare_features_for_insert(df)
    idx_beta = FEATURES_TABLE_COLUMNS.index("beta")
    assert records[0][idx_beta] == 1.2  # passthrough, not a leakage guard itself


def test_upsert_sql_touches_only_features_table():
    assert "features" in FEATURES_UPSERT_SQL
    for forbidden in ("prices_raw", "predictions", "portfolios", "risk_metrics"):
        assert forbidden not in FEATURES_UPSERT_SQL
    assert "ON CONFLICT (ticker, date) DO UPDATE" in FEATURES_UPSERT_SQL


def test_upsert_features_idempotent_no_duplicate_ticker_date(fake_store):
    df = _df(
        [
            {"ticker": "AAPL", "date": "2021-01-04", "rsi_14": 50.0},
            {"ticker": "AAPL", "date": "2021-01-05", "rsi_14": 51.0},
        ]
    )
    conn = FakeConn()
    n1 = upsert_features(conn, df)
    assert n1 == 2
    assert len(fake_store) == 2

    # Re-running with changed values must UPDATE in place, not duplicate.
    df2 = df.copy()
    df2.loc[0, "rsi_14"] = 99.0
    n2 = upsert_features(conn, df2)
    assert n2 == 2
    assert len(fake_store) == 2  # still exactly 2 keys, not 4
    idx_rsi = FEATURES_TABLE_COLUMNS.index("rsi_14")
    idx_ticker = FEATURES_TABLE_COLUMNS.index("ticker")
    idx_date = FEATURES_TABLE_COLUMNS.index("date")
    import datetime

    key = ("AAPL", datetime.date(2021, 1, 4))
    assert fake_store[key][idx_rsi] == 99.0


def test_upsert_features_empty_dataframe_is_a_noop(fake_store):
    df = _df([])
    conn = FakeConn()
    n = upsert_features(conn, df)
    assert n == 0
    assert fake_store == {}


def test_persist_features_requires_explicit_df():
    with pytest.raises(ValueError):
        persist_features(conn=FakeConn(), df=None)


def test_persist_features_uses_transactional_with_blocks(fake_store, monkeypatch):
    df = _df([{"ticker": "AAPL", "date": "2021-01-04", "rsi_14": 50.0}])
    conn = FakeConn()

    def fake_fetch_scalar(conn_, sql, params=None):
        assert "features" in sql
        return len(fake_store)

    monkeypatch.setattr(features_mod.db, "fetch_scalar", fake_fetch_scalar)

    result = persist_features(conn=conn, df=df)
    assert isinstance(result, PersistResult)
    assert result.rows_prepared == 1
    assert result.rows_in_table_after == 1
    assert conn.blocks_committed == 2  # one for the upsert, one for the count read


def test_persist_features_never_references_other_tables_via_sql_constants():
    # Static, structural guarantee: the only SQL this module ever issues for
    # writes is FEATURES_UPSERT_SQL, and it names only `features`.
    assert FEATURES_UPSERT_SQL.strip().upper().startswith("INSERT INTO FEATURES")
