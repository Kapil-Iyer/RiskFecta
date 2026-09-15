"""
RiskFecta Phase 5A — `predictions` LSTM narrow-upsert tests
(pipeline/predictions.py's build_lstm_predictions_rows/upsert_lstm_predictions/
persist_lstm_predictions).

Mock-only: NO real database connection is opened anywhere in this file
(task brief §17, §21 — real `predictions.lstm_pred` must remain entirely
NULL pre-audit). Mirrors tests/test_predictions_persistence.py's
FakeConn/FakeCursor + monkeypatched execute_values pattern, plus a second,
stricter fixture that actually emulates Postgres's column-level
`ON CONFLICT ... DO UPDATE SET` semantics (see `realistic_fake_store`
below) — needed to genuinely prove the persistence-safety requirement
(task brief §17, §20-R), not just assert on the SQL string.
"""
from __future__ import annotations

import datetime
import re

import numpy as np
import pandas as pd
import pytest

import pipeline.predictions as predictions_mod
from pipeline.predictions import (
    LSTM_PRED_COL,
    LSTM_PREDICTIONS_UPSERT_SQL,
    PREDICTIONS_TABLE_COLUMNS,
    PREDICTIONS_UPSERT_SQL,
    PersistResult,
    XGB_PRED_COL,
    build_lstm_predictions_rows,
    persist_lstm_predictions,
    upsert_lstm_predictions,
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
    """Extract the column names named in an `ON CONFLICT ... DO UPDATE SET
    ...` clause, e.g. {'target_date', 'lstm_pred'}."""
    m = re.search(r"DO UPDATE SET(.*)$", sql, re.S)
    assert m is not None
    return set(re.findall(r"(\w+)\s*=\s*EXCLUDED", m.group(1)))


@pytest.fixture
def fake_store(monkeypatch):
    """Blind full-tuple replace — same convention as
    tests/test_predictions_persistence.py. Adequate for idempotency/shape
    checks; NOT sufficient to prove column-preservation (see
    `realistic_fake_store`)."""
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
    """Genuinely emulates Postgres INSERT ... ON CONFLICT DO UPDATE SET
    semantics: a NEW key inserts the full record; an EXISTING key has ONLY
    the columns named in the SQL's own SET clause overwritten — every
    sibling column of the existing row is left untouched. This is what
    actually proves a narrow upsert cannot null out sibling columns (task
    brief §17, §20-R); a blind full-tuple replace would not catch a
    widened ON CONFLICT clause bug."""
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


# ---------------------------------------------------------------------------
# build_lstm_predictions_rows
# ---------------------------------------------------------------------------
def test_build_lstm_predictions_rows_uses_only_existing_columns():
    eval_rows = pd.DataFrame([{"ticker": "AAPL", "date": pd.Timestamp("2021-06-01")}])
    lstm_preds = pd.Series({"AAPL": 0.0088})
    out = build_lstm_predictions_rows(eval_rows, lstm_preds, horizon=21)
    assert list(out.columns) == PREDICTIONS_TABLE_COLUMNS
    assert out.loc[0, LSTM_PRED_COL] == pytest.approx(0.0088)
    assert out.loc[0, XGB_PRED_COL] is None
    assert out.loc[0, "ensemble_pred"] is None
    assert out.loc[0, "actual_return"] is None


def test_build_lstm_predictions_rows_multi_ticker_retains_identity():
    eval_rows = pd.DataFrame(
        [{"ticker": "AAPL", "date": pd.Timestamp("2021-06-01")},
         {"ticker": "MSFT", "date": pd.Timestamp("2021-06-01")}]
    )
    lstm_preds = pd.Series({"AAPL": 0.01, "MSFT": -0.02})
    out = build_lstm_predictions_rows(eval_rows, lstm_preds, horizon=21)
    assert out.set_index("ticker")[LSTM_PRED_COL].to_dict() == {"AAPL": 0.01, "MSFT": -0.02}


# ---------------------------------------------------------------------------
# Upsert SQL — narrow ON CONFLICT, table scope
# ---------------------------------------------------------------------------
def test_lstm_upsert_sql_touches_only_predictions_table():
    assert "predictions" in LSTM_PREDICTIONS_UPSERT_SQL
    for forbidden in ("prices_raw", "features", "portfolios", "risk_metrics"):
        assert forbidden not in LSTM_PREDICTIONS_UPSERT_SQL
    assert "ON CONFLICT (ticker, forecast_date) DO UPDATE" in LSTM_PREDICTIONS_UPSERT_SQL


def test_lstm_upsert_sql_only_updates_target_date_and_lstm_pred_on_conflict():
    assert _parse_conflict_set_columns(LSTM_PREDICTIONS_UPSERT_SQL) == {"target_date", "lstm_pred"}
    assert "xgb_pred = EXCLUDED" not in LSTM_PREDICTIONS_UPSERT_SQL
    assert "ensemble_pred = EXCLUDED" not in LSTM_PREDICTIONS_UPSERT_SQL
    assert "actual_return = EXCLUDED" not in LSTM_PREDICTIONS_UPSERT_SQL
    assert "directional_correct = EXCLUDED" not in LSTM_PREDICTIONS_UPSERT_SQL


def test_xgb_and_lstm_upsert_sql_disagree_only_on_their_own_column():
    xgb_cols = _parse_conflict_set_columns(PREDICTIONS_UPSERT_SQL)
    lstm_cols = _parse_conflict_set_columns(LSTM_PREDICTIONS_UPSERT_SQL)
    assert xgb_cols == {"target_date", "xgb_pred"}
    assert lstm_cols == {"target_date", "lstm_pred"}


# ---------------------------------------------------------------------------
# R. Persistence safety — the critical requirement (task brief §17, §20-R):
# an LSTM upsert must never null out xgb_pred/actual_return/
# directional_correct/ensemble_pred that a prior phase already wrote.
# ---------------------------------------------------------------------------
def test_lstm_upsert_preserves_existing_xgb_pred_and_actual_return(realistic_fake_store):
    key_date = datetime.date(2021, 6, 1)
    idx = {c: i for i, c in enumerate(PREDICTIONS_TABLE_COLUMNS)}
    existing = [None] * len(PREDICTIONS_TABLE_COLUMNS)
    existing[idx["ticker"]] = "AAPL"
    existing[idx["forecast_date"]] = key_date
    existing[idx["target_date"]] = datetime.date(2021, 7, 1)
    existing[idx["xgb_pred"]] = 0.045
    existing[idx["actual_return"]] = 0.05
    existing[idx["directional_correct"]] = True
    realistic_fake_store[("AAPL", key_date)] = existing

    # Realistic caller: target_date is a deterministic function of
    # (ticker, forecast_date, horizon) shared by every model branch, so a
    # genuine caller resolves and passes the SAME target_date the Phase 4A
    # xgb_pred row already carries — not left NaT (see
    # pipeline.predictions.build_predictions_rows's own target_date note).
    eval_rows = pd.DataFrame(
        [{"ticker": "AAPL", "date": pd.Timestamp("2021-06-01"), "target_date": pd.Timestamp("2021-07-01")}]
    )
    lstm_df = build_lstm_predictions_rows(eval_rows, pd.Series({"AAPL": 0.011}), horizon=21)
    conn = FakeConn()
    n = upsert_lstm_predictions(conn, lstm_df)
    assert n == 1

    row = realistic_fake_store[("AAPL", key_date)]
    assert row[idx["xgb_pred"]] == pytest.approx(0.045)          # untouched
    assert row[idx["actual_return"]] == pytest.approx(0.05)       # untouched
    assert row[idx["directional_correct"]] is True                # untouched
    assert row[idx["lstm_pred"]] == pytest.approx(0.011)          # updated
    assert row[idx["target_date"]] == datetime.date(2021, 7, 1)   # re-affirmed, not nulled


def test_lstm_upsert_on_new_row_leaves_xgb_and_ensemble_null(realistic_fake_store):
    eval_rows = pd.DataFrame([{"ticker": "NEWCO", "date": pd.Timestamp("2021-06-01")}])
    lstm_df = build_lstm_predictions_rows(eval_rows, pd.Series({"NEWCO": 0.02}), horizon=21)
    conn = FakeConn()
    upsert_lstm_predictions(conn, lstm_df)

    idx = {c: i for i, c in enumerate(PREDICTIONS_TABLE_COLUMNS)}
    row = realistic_fake_store[("NEWCO", datetime.date(2021, 6, 1))]
    assert row[idx["lstm_pred"]] == pytest.approx(0.02)
    assert row[idx["xgb_pred"]] is None
    assert row[idx["ensemble_pred"]] is None
    assert row[idx["actual_return"]] is None


def test_lstm_upsert_idempotent_no_duplicate_ticker_forecast_date(fake_store):
    eval_rows = pd.DataFrame([{"ticker": "AAPL", "date": pd.Timestamp("2021-06-01")}])
    df = build_lstm_predictions_rows(eval_rows, pd.Series({"AAPL": 0.01}), horizon=21)
    conn = FakeConn()
    n1 = upsert_lstm_predictions(conn, df)
    assert n1 == 1
    assert len(fake_store) == 1

    df2 = build_lstm_predictions_rows(eval_rows, pd.Series({"AAPL": 0.02}), horizon=21)
    n2 = upsert_lstm_predictions(conn, df2)
    assert n2 == 1
    assert len(fake_store) == 1  # updated in place, not duplicated


def test_upsert_lstm_predictions_empty_dataframe_is_noop(fake_store):
    empty = pd.DataFrame(columns=PREDICTIONS_TABLE_COLUMNS)
    conn = FakeConn()
    n = upsert_lstm_predictions(conn, empty)
    assert n == 0
    assert fake_store == {}


def test_persist_lstm_predictions_requires_explicit_df():
    with pytest.raises(ValueError):
        persist_lstm_predictions(conn=FakeConn(), df=None)


def test_persist_lstm_predictions_never_opens_a_real_connection_when_conn_given(fake_store, monkeypatch):
    eval_rows = pd.DataFrame([{"ticker": "AAPL", "date": pd.Timestamp("2021-06-01")}])
    df = build_lstm_predictions_rows(eval_rows, pd.Series({"AAPL": 0.01}), horizon=21)
    conn = FakeConn()

    def fake_fetch_scalar(conn_, sql, params=None):
        assert "predictions" in sql
        return len(fake_store)

    monkeypatch.setattr(predictions_mod.db, "fetch_scalar", fake_fetch_scalar)

    def fail_get_connection():
        raise AssertionError("persist_lstm_predictions must not open a real DB connection when conn is given")

    monkeypatch.setattr(predictions_mod.db, "get_connection", fail_get_connection)

    result = persist_lstm_predictions(conn=conn, df=df)
    assert isinstance(result, PersistResult)
    assert result.rows_prepared == 1
    assert result.rows_in_table_after == 1
    assert conn.blocks_committed == 2
