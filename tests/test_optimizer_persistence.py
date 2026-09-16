"""
RiskFecta Phase 7A — `portfolios`/`risk_metrics` persistence tests
(optimizer/persistence.py). Mock-only: NO real database connection is
opened anywhere in this file (task brief §17, §19 — real `portfolios`/
`risk_metrics` must remain empty pre-audit). Mirrors
tests/test_predictions_persistence.py's FakeConn/FakeCursor +
monkeypatched execute_values pattern.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import optimizer.persistence as persist_mod
from optimizer.persistence import (
    PORTFOLIOS_INSERT_SQL,
    PORTFOLIOS_TABLE_COLUMNS,
    RISK_METRICS_INSERT_SQL,
    RISK_METRICS_TABLE_COLUMNS,
    PersistResult,
    build_portfolio_rows,
    build_risk_metric_rows,
    insert_portfolios,
    insert_risk_metrics,
    persist_portfolios,
    persist_risk_metrics,
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
def fake_portfolios_store(monkeypatch):
    store: list = []

    def fake_execute_values(cur, sql, records, page_size=1000):
        assert "portfolios" in sql
        store.extend(records)

    monkeypatch.setattr(persist_mod, "execute_values", fake_execute_values)
    return store


@pytest.fixture
def fake_risk_metrics_store(monkeypatch):
    store: list = []

    def fake_execute_values(cur, sql, records, page_size=1000):
        assert "risk_metrics" in sql
        store.extend(records)

    monkeypatch.setattr(persist_mod, "execute_values", fake_execute_values)
    return store


# ---------------------------------------------------------------------------
# build_portfolio_rows / build_risk_metric_rows
# ---------------------------------------------------------------------------
def test_build_portfolio_rows_shape_matches_schema():
    df = build_portfolio_rows(
        run_id="synthetic-run-1",
        tickers=["AAA", "BBB"],
        weights=[0.6, 0.4],
        target_return=0.02,
        portfolio_vol=0.05,
        sharpe_ratio=0.4,
    )
    assert list(df.columns) == PORTFOLIOS_TABLE_COLUMNS
    assert df["weight"].tolist() == [0.6, 0.4]
    assert (df["run_id"] == "synthetic-run-1").all()
    assert (df["target_return"] == 0.02).all()


def test_build_portfolio_rows_length_mismatch_fails_loudly():
    with pytest.raises(ValueError, match="length mismatch"):
        build_portfolio_rows("r1", ["AAA", "BBB"], [0.5], 0.01, 0.02, 0.3)


def test_build_risk_metric_rows_portfolio_level_ticker_null():
    df = build_risk_metric_rows("r1", {"volatility": 0.12, "var_95": -0.03})
    assert list(df.columns) == RISK_METRICS_TABLE_COLUMNS
    assert df["ticker"].isna().all()
    assert set(df["metric_name"]) == {"volatility", "var_95"}


def test_build_risk_metric_rows_per_asset_ticker_set():
    df = build_risk_metric_rows("r1", {"risk_contribution": 0.05}, ticker="AAA")
    assert df.loc[0, "ticker"] == "AAA"


# ---------------------------------------------------------------------------
# Table scope
# ---------------------------------------------------------------------------
def test_portfolios_sql_touches_only_portfolios_table():
    assert "portfolios" in PORTFOLIOS_INSERT_SQL
    for forbidden in ("prices_raw", "features", "predictions", "risk_metrics"):
        assert forbidden not in PORTFOLIOS_INSERT_SQL


def test_risk_metrics_sql_touches_only_risk_metrics_table():
    assert "risk_metrics" in RISK_METRICS_INSERT_SQL
    for forbidden in ("prices_raw", "features", "predictions", "portfolios"):
        assert forbidden not in RISK_METRICS_INSERT_SQL


# ---------------------------------------------------------------------------
# Insert behavior (append-only; no UNIQUE constraint in schema — see module
# docstring for why true idempotence isn't available here)
# ---------------------------------------------------------------------------
def test_insert_portfolios_appends_expected_row_count(fake_portfolios_store):
    df = build_portfolio_rows("r1", ["AAA", "BBB"], [0.6, 0.4], 0.02, 0.05, 0.4)
    conn = FakeConn()
    n = insert_portfolios(conn, df)
    assert n == 2
    assert len(fake_portfolios_store) == 2


def test_insert_portfolios_empty_dataframe_is_noop(fake_portfolios_store):
    empty = pd.DataFrame(columns=PORTFOLIOS_TABLE_COLUMNS)
    conn = FakeConn()
    n = insert_portfolios(conn, empty)
    assert n == 0
    assert fake_portfolios_store == []


def test_repeated_insert_with_same_run_id_appends_again_documented_behavior(fake_portfolios_store):
    # Documented limitation: no UNIQUE(run_id, ticker) constraint in the
    # frozen schema, so calling insert_portfolios twice with the SAME
    # run_id/df appends the SAME rows a second time (never silently
    # deduplicated by this module — that would be inventing upsert
    # semantics the schema doesn't support).
    df = build_portfolio_rows("r1", ["AAA"], [1.0], 0.02, 0.05, 0.4)
    conn = FakeConn()
    insert_portfolios(conn, df)
    insert_portfolios(conn, df)
    assert len(fake_portfolios_store) == 2


def test_insert_risk_metrics_appends_expected_row_count(fake_risk_metrics_store):
    df = build_risk_metric_rows("r1", {"volatility": 0.1, "sharpe": 0.3})
    conn = FakeConn()
    n = insert_risk_metrics(conn, df)
    assert n == 2
    assert len(fake_risk_metrics_store) == 2


def test_nan_normalized_to_sql_null(fake_portfolios_store):
    df = build_portfolio_rows("r1", ["AAA"], [1.0], target_return=np.nan, portfolio_vol=0.05, sharpe_ratio=np.nan)
    conn = FakeConn()
    insert_portfolios(conn, df)
    idx_target = PORTFOLIOS_TABLE_COLUMNS.index("target_return")
    idx_sharpe = PORTFOLIOS_TABLE_COLUMNS.index("sharpe_ratio")
    assert fake_portfolios_store[0][idx_target] is None
    assert fake_portfolios_store[0][idx_sharpe] is None


# ---------------------------------------------------------------------------
# persist_* — never open a real connection when one is supplied; require df
# ---------------------------------------------------------------------------
def test_persist_portfolios_requires_explicit_df():
    with pytest.raises(ValueError):
        persist_portfolios(conn=FakeConn(), df=None)


def test_persist_risk_metrics_requires_explicit_df():
    with pytest.raises(ValueError):
        persist_risk_metrics(conn=FakeConn(), df=None)


def test_persist_portfolios_never_opens_real_connection_when_conn_given(fake_portfolios_store, monkeypatch):
    df = build_portfolio_rows("r1", ["AAA"], [1.0], 0.02, 0.05, 0.4)
    conn = FakeConn()

    def fake_fetch_scalar(conn_, sql, params=None):
        assert "portfolios" in sql
        return len(fake_portfolios_store)

    monkeypatch.setattr(persist_mod.db, "fetch_scalar", fake_fetch_scalar)

    def fail_get_connection():
        raise AssertionError("persist_portfolios must not open a real DB connection when conn is given")

    monkeypatch.setattr(persist_mod.db, "get_connection", fail_get_connection)

    result = persist_portfolios(conn=conn, df=df)
    assert isinstance(result, PersistResult)
    assert result.rows_prepared == 1
    assert result.rows_in_table_after == 1
    assert conn.blocks_committed == 2


def test_persist_risk_metrics_never_opens_real_connection_when_conn_given(fake_risk_metrics_store, monkeypatch):
    df = build_risk_metric_rows("r1", {"volatility": 0.1})
    conn = FakeConn()

    def fake_fetch_scalar(conn_, sql, params=None):
        assert "risk_metrics" in sql
        return len(fake_risk_metrics_store)

    monkeypatch.setattr(persist_mod.db, "fetch_scalar", fake_fetch_scalar)

    def fail_get_connection():
        raise AssertionError("persist_risk_metrics must not open a real DB connection when conn is given")

    monkeypatch.setattr(persist_mod.db, "get_connection", fail_get_connection)

    result = persist_risk_metrics(conn=conn, df=df)
    assert isinstance(result, PersistResult)
    assert result.rows_prepared == 1
    assert result.rows_in_table_after == 1
