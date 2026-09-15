"""
RiskFecta Phase 4A — `predictions` persistence tests (pipeline/predictions.py).

Mock-only: NO real database connection is opened anywhere in this file
(task brief §15, §18-L, §19 — real `predictions` must remain empty
pre-audit). Mirrors tests/test_features_persistence.py's FakeConn/FakeCursor
+ monkeypatched execute_values pattern exactly.
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import pytest

import pipeline.predictions as predictions_mod
from pipeline.predictions import (
    PREDICTIONS_TABLE_COLUMNS,
    PREDICTIONS_UPSERT_SQL,
    PersistResult,
    build_predictions_rows,
    persist_predictions,
    upsert_predictions,
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


# ---------------------------------------------------------------------------
# build_predictions_rows
# ---------------------------------------------------------------------------
def test_build_predictions_rows_uses_only_existing_columns():
    eval_rows = pd.DataFrame([{"ticker": "AAPL", "date": pd.Timestamp("2021-06-01")}])
    xgb_preds = pd.Series({"AAPL": 0.0123})
    out = build_predictions_rows(eval_rows, xgb_preds, horizon=21)
    assert list(out.columns) == PREDICTIONS_TABLE_COLUMNS
    assert out.loc[0, "xgb_pred"] == pytest.approx(0.0123)
    assert out.loc[0, "lstm_pred"] is None
    assert out.loc[0, "ensemble_pred"] is None
    assert out.loc[0, "actual_return"] is None


def test_build_predictions_rows_no_target_date_leaves_it_unfabricated():
    eval_rows = pd.DataFrame([{"ticker": "AAPL", "date": pd.Timestamp("2021-06-01")}])
    xgb_preds = pd.Series({"AAPL": 0.01})
    out = build_predictions_rows(eval_rows, xgb_preds, horizon=21)
    # No calendar-day guess (date + 21 calendar days) — left NaT.
    assert pd.isna(out.loc[0, "target_date"])


def test_build_predictions_rows_multi_ticker_retains_identity():
    eval_rows = pd.DataFrame(
        [{"ticker": "AAPL", "date": pd.Timestamp("2021-06-01")},
         {"ticker": "MSFT", "date": pd.Timestamp("2021-06-01")}]
    )
    xgb_preds = pd.Series({"AAPL": 0.01, "MSFT": -0.02})
    out = build_predictions_rows(eval_rows, xgb_preds, horizon=21)
    assert out.set_index("ticker")["xgb_pred"].to_dict() == {"AAPL": 0.01, "MSFT": -0.02}


# ---------------------------------------------------------------------------
# Upsert SQL / persistence — table scope
# ---------------------------------------------------------------------------
def test_upsert_sql_touches_only_predictions_table():
    assert "predictions" in PREDICTIONS_UPSERT_SQL
    for forbidden in ("prices_raw", "features", "portfolios", "risk_metrics"):
        assert forbidden not in PREDICTIONS_UPSERT_SQL
    assert "ON CONFLICT (ticker, forecast_date) DO UPDATE" in PREDICTIONS_UPSERT_SQL


def test_upsert_sql_only_updates_xgb_pred_and_target_date_on_conflict():
    # Narrow ON CONFLICT — Phase 4A never overwrites lstm_pred/ensemble_pred/
    # actual_return/directional_correct that a later phase may have written.
    assert "lstm_pred = EXCLUDED" not in PREDICTIONS_UPSERT_SQL
    assert "ensemble_pred = EXCLUDED" not in PREDICTIONS_UPSERT_SQL
    assert "actual_return = EXCLUDED" not in PREDICTIONS_UPSERT_SQL
    assert "xgb_pred = EXCLUDED" in PREDICTIONS_UPSERT_SQL


def test_upsert_predictions_idempotent_no_duplicate_ticker_forecast_date(fake_store):
    eval_rows = pd.DataFrame([{"ticker": "AAPL", "date": pd.Timestamp("2021-06-01")}])
    df = build_predictions_rows(eval_rows, pd.Series({"AAPL": 0.01}), horizon=21)
    conn = FakeConn()
    n1 = upsert_predictions(conn, df)
    assert n1 == 1
    assert len(fake_store) == 1

    df2 = build_predictions_rows(eval_rows, pd.Series({"AAPL": 0.02}), horizon=21)
    n2 = upsert_predictions(conn, df2)
    assert n2 == 1
    assert len(fake_store) == 1  # updated in place, not duplicated
    idx_xgb = PREDICTIONS_TABLE_COLUMNS.index("xgb_pred")
    key = ("AAPL", datetime.date(2021, 6, 1))
    assert fake_store[key][idx_xgb] == pytest.approx(0.02)


def test_upsert_predictions_empty_dataframe_is_noop(fake_store):
    empty = pd.DataFrame(columns=PREDICTIONS_TABLE_COLUMNS)
    conn = FakeConn()
    n = upsert_predictions(conn, empty)
    assert n == 0
    assert fake_store == {}


def test_persist_predictions_requires_explicit_df():
    with pytest.raises(ValueError):
        persist_predictions(conn=FakeConn(), df=None)


def test_persist_predictions_never_opens_a_real_connection_when_conn_given(fake_store, monkeypatch):
    eval_rows = pd.DataFrame([{"ticker": "AAPL", "date": pd.Timestamp("2021-06-01")}])
    df = build_predictions_rows(eval_rows, pd.Series({"AAPL": 0.01}), horizon=21)
    conn = FakeConn()

    def fake_fetch_scalar(conn_, sql, params=None):
        assert "predictions" in sql
        return len(fake_store)

    monkeypatch.setattr(predictions_mod.db, "fetch_scalar", fake_fetch_scalar)

    def fail_get_connection():
        raise AssertionError("persist_predictions must not open a real DB connection when conn is given")
    monkeypatch.setattr(predictions_mod.db, "get_connection", fail_get_connection)

    result = persist_predictions(conn=conn, df=df)
    assert isinstance(result, PersistResult)
    assert result.rows_prepared == 1
    assert result.rows_in_table_after == 1
    assert conn.blocks_committed == 2
