"""
RiskFecta Phase 6A — `predictions` ensemble narrow-upsert tests
(pipeline/predictions.py's build_ensemble_predictions_rows/
upsert_ensemble_predictions/persist_ensemble_predictions).

Mock-only: NO real database connection is opened anywhere in this file
(task brief §12, §21 — real `predictions.ensemble_pred` must remain
entirely NULL pre-audit). Mirrors tests/test_lstm_predictions_persistence
.py's FakeConn/FakeCursor + realistic_fake_store pattern (genuinely
emulates Postgres's column-level ON CONFLICT ... DO UPDATE SET semantics)
so narrow-upsert column preservation is actually proven, not merely
asserted on the SQL string.
"""
from __future__ import annotations

import datetime
import re

import pandas as pd
import pytest

import pipeline.predictions as predictions_mod
from pipeline.predictions import (
    ENSEMBLE_PRED_COL,
    ENSEMBLE_PREDICTIONS_UPSERT_SQL,
    LSTM_PREDICTIONS_UPSERT_SQL,
    PREDICTIONS_TABLE_COLUMNS,
    PREDICTIONS_UPSERT_SQL,
    PersistResult,
    build_ensemble_predictions_rows,
    persist_ensemble_predictions,
    upsert_ensemble_predictions,
)


class FakeCursor:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConn:
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


def _parse_conflict_set_columns(sql: str) -> set:
    m = re.search(r"DO UPDATE SET(.*)$", sql, re.S)
    assert m is not None
    return set(re.findall(r"(\w+)\s*=\s*EXCLUDED", m.group(1)))


@pytest.fixture
def fake_store(monkeypatch):
    store: dict = {}

    def fake_execute_values(cur, sql, records, page_size=1000):
        assert "predictions" in sql
        idx_ticker = PREDICTIONS_TABLE_COLUMNS.index("ticker")
        idx_fdate = PREDICTIONS_TABLE_COLUMNS.index("forecast_date")
        for rec in records:
            key = (rec[idx_ticker], rec[idx_fdate])
            store[key] = rec

    monkeypatch.setattr(predictions_mod, "execute_values", fake_execute_values)
    return store


@pytest.fixture
def realistic_fake_store(monkeypatch):
    """Genuinely emulates Postgres INSERT ... ON CONFLICT DO UPDATE SET:
    an existing key has ONLY the columns named in the SQL's own SET
    clause overwritten — every sibling column is left untouched."""
    store: dict = {}

    def fake_execute_values(cur, sql, records, page_size=1000):
        assert "predictions" in sql
        set_cols = _parse_conflict_set_columns(sql)
        set_idxs = [PREDICTIONS_TABLE_COLUMNS.index(c) for c in set_cols]
        idx_ticker = PREDICTIONS_TABLE_COLUMNS.index("ticker")
        idx_fdate = PREDICTIONS_TABLE_COLUMNS.index("forecast_date")
        for rec in records:
            key = (rec[idx_ticker], rec[idx_fdate])
            if key not in store:
                store[key] = list(rec)
            else:
                existing = store[key]
                for i in set_idxs:
                    existing[i] = rec[i]

    monkeypatch.setattr(predictions_mod, "execute_values", fake_execute_values)
    return store


def _ensemble_df(rows):
    """rows: list of {ticker, forecast_date, target_date, ensemble_pred}."""
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# build_ensemble_predictions_rows
# ---------------------------------------------------------------------------
def test_build_ensemble_predictions_rows_uses_only_existing_columns():
    df = _ensemble_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"),
         "target_date": pd.Timestamp("2022-03-28"), "ensemble_pred": 0.015},
    ])
    out = build_ensemble_predictions_rows(df)
    assert list(out.columns) == PREDICTIONS_TABLE_COLUMNS
    assert out.loc[0, ENSEMBLE_PRED_COL] == pytest.approx(0.015)
    assert out.loc[0, "xgb_pred"] is None
    assert out.loc[0, "lstm_pred"] is None
    assert out.loc[0, "actual_return"] is None


def test_build_ensemble_predictions_rows_multi_row_retains_identity():
    df = _ensemble_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"),
         "target_date": pd.Timestamp("2022-03-28"), "ensemble_pred": 0.01},
        {"ticker": "MSFT", "forecast_date": pd.Timestamp("2022-02-25"),
         "target_date": pd.Timestamp("2022-03-28"), "ensemble_pred": -0.02},
    ])
    out = build_ensemble_predictions_rows(df)
    assert out.set_index("ticker")[ENSEMBLE_PRED_COL].to_dict() == {"AAPL": 0.01, "MSFT": -0.02}


# ---------------------------------------------------------------------------
# Upsert SQL — narrow ON CONFLICT, table scope
# ---------------------------------------------------------------------------
def test_ensemble_upsert_sql_touches_only_predictions_table():
    assert "predictions" in ENSEMBLE_PREDICTIONS_UPSERT_SQL
    for forbidden in ("prices_raw", "features", "portfolios", "risk_metrics"):
        assert forbidden not in ENSEMBLE_PREDICTIONS_UPSERT_SQL
    assert "ON CONFLICT (ticker, forecast_date) DO UPDATE" in ENSEMBLE_PREDICTIONS_UPSERT_SQL


def test_ensemble_upsert_sql_only_updates_target_date_and_ensemble_pred_on_conflict():
    assert _parse_conflict_set_columns(ENSEMBLE_PREDICTIONS_UPSERT_SQL) == {"target_date", "ensemble_pred"}
    assert "xgb_pred = EXCLUDED" not in ENSEMBLE_PREDICTIONS_UPSERT_SQL
    assert "lstm_pred = EXCLUDED" not in ENSEMBLE_PREDICTIONS_UPSERT_SQL
    assert "actual_return = EXCLUDED" not in ENSEMBLE_PREDICTIONS_UPSERT_SQL
    assert "directional_correct = EXCLUDED" not in ENSEMBLE_PREDICTIONS_UPSERT_SQL


def test_xgb_lstm_ensemble_upsert_sql_disagree_only_on_their_own_column():
    xgb_cols = _parse_conflict_set_columns(PREDICTIONS_UPSERT_SQL)
    lstm_cols = _parse_conflict_set_columns(LSTM_PREDICTIONS_UPSERT_SQL)
    ens_cols = _parse_conflict_set_columns(ENSEMBLE_PREDICTIONS_UPSERT_SQL)
    assert xgb_cols == {"target_date", "xgb_pred"}
    assert lstm_cols == {"target_date", "lstm_pred"}
    assert ens_cols == {"target_date", "ensemble_pred"}


# ---------------------------------------------------------------------------
# L. narrow ensemble upsert preserves xgb_pred, lstm_pred, actual_return,
#    directional_correct
# ---------------------------------------------------------------------------
def test_ensemble_upsert_preserves_existing_xgb_lstm_actual_directional(realistic_fake_store):
    key_date = datetime.date(2022, 2, 25)
    idx = {c: i for i, c in enumerate(PREDICTIONS_TABLE_COLUMNS)}
    existing = [None] * len(PREDICTIONS_TABLE_COLUMNS)
    existing[idx["ticker"]] = "AAPL"
    existing[idx["forecast_date"]] = key_date
    existing[idx["target_date"]] = datetime.date(2022, 3, 28)
    existing[idx["xgb_pred"]] = 0.045
    existing[idx["lstm_pred"]] = 0.025
    existing[idx["actual_return"]] = 0.05
    existing[idx["directional_correct"]] = True
    realistic_fake_store[("AAPL", key_date)] = existing

    df = _ensemble_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"),
         "target_date": pd.Timestamp("2022-03-28"), "ensemble_pred": 0.035},
    ])
    ens_rows = build_ensemble_predictions_rows(df)
    conn = FakeConn()
    n = upsert_ensemble_predictions(conn, ens_rows)
    assert n == 1

    row = realistic_fake_store[("AAPL", key_date)]
    assert row[idx["xgb_pred"]] == pytest.approx(0.045)             # untouched
    assert row[idx["lstm_pred"]] == pytest.approx(0.025)            # untouched
    assert row[idx["actual_return"]] == pytest.approx(0.05)          # untouched
    assert row[idx["directional_correct"]] is True                  # untouched
    assert row[idx["ensemble_pred"]] == pytest.approx(0.035)        # updated
    assert row[idx["target_date"]] == datetime.date(2022, 3, 28)    # re-affirmed, not nulled


# ---------------------------------------------------------------------------
# M. ensemble upsert idempotence
# ---------------------------------------------------------------------------
def test_ensemble_upsert_idempotent_no_duplicate_identity(fake_store):
    df = _ensemble_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"),
         "target_date": pd.Timestamp("2022-03-28"), "ensemble_pred": 0.01},
    ])
    ens_rows = build_ensemble_predictions_rows(df)
    conn = FakeConn()
    n1 = upsert_ensemble_predictions(conn, ens_rows)
    assert n1 == 1
    assert len(fake_store) == 1

    df2 = _ensemble_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"),
         "target_date": pd.Timestamp("2022-03-28"), "ensemble_pred": 0.02},
    ])
    n2 = upsert_ensemble_predictions(conn, build_ensemble_predictions_rows(df2))
    assert n2 == 1
    assert len(fake_store) == 1  # updated in place, not duplicated


# ---------------------------------------------------------------------------
# N. ensemble persistence does not alter row count
# ---------------------------------------------------------------------------
def test_persist_ensemble_predictions_does_not_change_row_count(fake_store, monkeypatch):
    df = _ensemble_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"),
         "target_date": pd.Timestamp("2022-03-28"), "ensemble_pred": 0.01},
    ])
    ens_rows = build_ensemble_predictions_rows(df)
    conn = FakeConn()

    def fake_fetch_scalar(conn_, sql, params=None):
        assert "predictions" in sql
        return len(fake_store)

    monkeypatch.setattr(predictions_mod.db, "fetch_scalar", fake_fetch_scalar)

    def fail_get_connection():
        raise AssertionError("persist_ensemble_predictions must not open a real DB connection when conn is given")
    monkeypatch.setattr(predictions_mod.db, "get_connection", fail_get_connection)

    result = persist_ensemble_predictions(conn=conn, df=ens_rows)
    assert isinstance(result, PersistResult)
    assert result.rows_prepared == 1
    assert result.rows_in_table_after == 1  # existing row updated, not a new row added
    assert conn.blocks_committed == 2

    # Re-running with the SAME identity again must not change the row count.
    result2 = persist_ensemble_predictions(conn=conn, df=ens_rows)
    assert result2.rows_in_table_after == 1


def test_upsert_ensemble_predictions_empty_dataframe_is_noop(fake_store):
    empty = pd.DataFrame(columns=PREDICTIONS_TABLE_COLUMNS)
    conn = FakeConn()
    n = upsert_ensemble_predictions(conn, empty)
    assert n == 0
    assert fake_store == {}


def test_persist_ensemble_predictions_requires_explicit_df():
    with pytest.raises(ValueError):
        persist_ensemble_predictions(conn=FakeConn(), df=None)


def test_persist_ensemble_predictions_never_opens_a_real_connection_when_conn_given(fake_store, monkeypatch):
    df = _ensemble_df([
        {"ticker": "AAPL", "forecast_date": pd.Timestamp("2022-02-25"),
         "target_date": pd.Timestamp("2022-03-28"), "ensemble_pred": 0.01},
    ])
    ens_rows = build_ensemble_predictions_rows(df)
    conn = FakeConn()

    def fake_fetch_scalar(conn_, sql, params=None):
        return len(fake_store)
    monkeypatch.setattr(predictions_mod.db, "fetch_scalar", fake_fetch_scalar)

    def fail_get_connection():
        raise AssertionError("must not open a real DB connection when conn is given")
    monkeypatch.setattr(predictions_mod.db, "get_connection", fail_get_connection)

    result = persist_ensemble_predictions(conn=conn, df=ens_rows)
    assert result.rows_prepared == 1
