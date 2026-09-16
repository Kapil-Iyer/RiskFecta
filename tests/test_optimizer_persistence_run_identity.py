"""
RiskFecta Phase 7B — experiment/run identity tests
(optimizer/persistence.py's build_run_id/parse_run_id/run_id_exists/
assert_run_id_available/persist_portfolio_run/persist_risk_metrics_run).
Mock-only: NO real database connection is opened anywhere in this file.
"""
from __future__ import annotations

import pandas as pd
import pytest

import optimizer.persistence as persist_mod
from optimizer.persistence import (
    PORTFOLIOS_TABLE_COLUMNS,
    build_portfolio_rows,
    build_run_id,
    parse_run_id,
    persist_portfolio_run,
    persist_risk_metrics_run,
    run_id_exists,
    assert_run_id_available,
)


class FakeCursor:
    def __init__(self, existing_run_ids):
        self.existing_run_ids = existing_run_ids
        self.last_query = None

    def execute(self, sql, params=None):
        self.last_query = (sql, params)

    def fetchone(self):
        run_id = self.last_query[1][0]
        return (1,) if run_id in self.existing_run_ids else None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeConn:
    def __init__(self, existing_run_ids=()):
        self.existing_run_ids = set(existing_run_ids)
        self.blocks_committed = 0

    def cursor(self):
        return FakeCursor(self.existing_run_ids)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.blocks_committed += 1
        return False


# ---------------------------------------------------------------------------
# build_run_id / parse_run_id
# ---------------------------------------------------------------------------
def test_build_run_id_round_trips_through_parse_run_id():
    run_id = build_run_id("p7b1", "SAMPLE_MAXSHARPE", pd.Timestamp("2022-02-25"))
    parsed = parse_run_id(run_id)
    assert parsed == {"experiment_id": "p7b1", "strategy": "SAMPLE_MAXSHARPE", "formation_date": "2022-02-25"}


def test_build_run_id_fits_varchar_50_for_longest_strategy():
    run_id = build_run_id("phase7bexp", "SAMPLE_MAXSHARPE", pd.Timestamp("2022-02-25"))
    assert len(run_id) <= 50


def test_build_run_id_rejects_unknown_strategy():
    with pytest.raises(ValueError, match="unknown strategy"):
        build_run_id("p7b1", "MYSTERY_STRATEGY", pd.Timestamp("2022-02-25"))


def test_build_run_id_rejects_underscore_in_experiment_id():
    with pytest.raises(ValueError, match="underscore|_"):
        build_run_id("p7b_1", "EQUAL_WEIGHT", pd.Timestamp("2022-02-25"))


def test_build_run_id_rejects_whitespace_experiment_id():
    with pytest.raises(ValueError):
        build_run_id("p7b 1", "EQUAL_WEIGHT", pd.Timestamp("2022-02-25"))


def test_build_run_id_rejects_overlong_result():
    with pytest.raises(ValueError, match="VARCHAR"):
        build_run_id("x" * 40, "SAMPLE_MAXSHARPE", pd.Timestamp("2022-02-25"))


def test_parse_run_id_rejects_unrecognized_strategy():
    with pytest.raises(ValueError, match="known strategy"):
        parse_run_id("p7b1_NOT_A_STRATEGY_2022-02-25")


# ---------------------------------------------------------------------------
# run_id_exists / assert_run_id_available (mock-only)
# ---------------------------------------------------------------------------
def test_run_id_exists_true_when_present():
    conn = FakeConn(existing_run_ids={"p7b1_EQUAL_WEIGHT_2022-02-25"})
    assert run_id_exists(conn, "portfolios", "p7b1_EQUAL_WEIGHT_2022-02-25") is True


def test_run_id_exists_false_when_absent():
    conn = FakeConn(existing_run_ids=set())
    assert run_id_exists(conn, "portfolios", "p7b1_EQUAL_WEIGHT_2022-02-25") is False


def test_run_id_exists_rejects_unsupported_table():
    conn = FakeConn()
    with pytest.raises(ValueError, match="unsupported table"):
        run_id_exists(conn, "prices_raw", "anything")


def test_assert_run_id_available_fails_loudly_on_collision():
    conn = FakeConn(existing_run_ids={"p7b1_EQUAL_WEIGHT_2022-02-25"})
    with pytest.raises(ValueError, match="already exists"):
        assert_run_id_available(conn, "portfolios", "p7b1_EQUAL_WEIGHT_2022-02-25")


def test_assert_run_id_available_passes_when_free():
    conn = FakeConn(existing_run_ids=set())
    assert_run_id_available(conn, "portfolios", "p7b1_EQUAL_WEIGHT_2022-02-25")  # must not raise


# ---------------------------------------------------------------------------
# persist_portfolio_run / persist_risk_metrics_run — collision aborts BEFORE
# any insert; no destructive overwrite.
# ---------------------------------------------------------------------------
def test_persist_portfolio_run_aborts_before_insert_on_collision(monkeypatch):
    insert_calls = []
    monkeypatch.setattr(persist_mod, "insert_portfolios", lambda conn, df: insert_calls.append(df) or 999)

    conn = FakeConn(existing_run_ids={"p7b1_EQUAL_WEIGHT_2022-02-25"})
    df = build_portfolio_rows("p7b1_EQUAL_WEIGHT_2022-02-25", ["AAA"], [1.0], 0.01, 0.05, 0.3)
    with pytest.raises(ValueError, match="already exists"):
        persist_portfolio_run(conn, "p7b1_EQUAL_WEIGHT_2022-02-25", df)
    assert insert_calls == []  # never reached the insert path


def test_persist_portfolio_run_succeeds_when_run_id_free(monkeypatch):
    def fake_fetch_scalar(conn_, sql, params=None):
        return 1

    monkeypatch.setattr(persist_mod.db, "fetch_scalar", fake_fetch_scalar)
    monkeypatch.setattr(persist_mod, "execute_values", lambda cur, sql, records, page_size=1000: None)

    conn = FakeConn(existing_run_ids=set())
    df = build_portfolio_rows("p7b1_EQUAL_WEIGHT_2022-02-25", ["AAA"], [1.0], 0.01, 0.05, 0.3)
    result = persist_portfolio_run(conn, "p7b1_EQUAL_WEIGHT_2022-02-25", df)
    assert result.rows_prepared == 1
