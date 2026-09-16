"""
Phase 2A local-only integration tests against the REAL Supabase-hosted
PostgreSQL database, exercised through the actual FastAPI app (not mocks).
Follows the same pattern as tests/test_db_integration.py: every test here is
skipped automatically when DATABASE_URL is not set (e.g. in CI). Run locally
with: pytest -m db

These tests never print, log, or assert on the DATABASE_URL value itself.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

import config
from app.main import app
from pipeline import db

pytestmark = pytest.mark.db

_skip_no_db = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL"),
    reason="DATABASE_URL not set in the environment (expected in CI / clean checkouts)",
)

# Downstream research tables the Phase 2A read-only API must never mutate.
# `features` (Phase 3) and `predictions` (Phase 4) were already populated
# even at Phase 2A. `portfolios`/`risk_metrics` (Phase 7) were empty when
# this test was first written but now legitimately hold the official,
# frozen Phase 7B experiment (11,500 / 685 rows) — so, like
# features/predictions, the invariant is "row count unchanged across API
# calls", never a hard-coded emptiness assumption
# (see test_future_phase_tables_remain_unchanged_after_api_use).
UNCHANGED_DOWNSTREAM_TABLES = ["features", "predictions", "portfolios", "risk_metrics"]
EXPECTED_ROW_COUNT = 62_800  # 50 tickers x 1256 valid sessions each (Phase 1A audit)


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


@_skip_no_db
def test_health_ready_reports_connected(client):
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "connected"}


@_skip_no_db
def test_universe_matches_config_universe(client):
    resp = client.get("/api/universe")
    assert resp.status_code == 200
    body = resp.json()
    assert {row["ticker"] for row in body} == set(config.TICKER_UNIVERSE)


@_skip_no_db
def test_prices_real_ticker_returns_full_ordered_history(client):
    resp = client.get("/api/prices/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["count"] == 1256
    dates = [row["date"] for row in body["prices"]]
    assert dates == sorted(dates)


@_skip_no_db
def test_prices_date_filter_narrows_range(client):
    resp = client.get("/api/prices/AAPL", params={"start": "2025-01-01", "end": "2025-01-31"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] > 0
    assert all("2025-01-01" <= row["date"] <= "2025-01-31" for row in body["prices"])


@_skip_no_db
def test_prices_unknown_ticker_returns_404(client):
    resp = client.get("/api/prices/ZZZNOTAREALTICKER")
    assert resp.status_code == 404


@_skip_no_db
def test_prices_invalid_date_range_returns_400(client):
    resp = client.get("/api/prices/AAPL", params={"start": "2025-02-01", "end": "2025-01-01"})
    assert resp.status_code == 400


@_skip_no_db
def test_market_summary_matches_known_totals(client):
    resp = client.get("/api/market/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker_count"] == 50
    assert body["price_row_count"] == EXPECTED_ROW_COUNT
    assert body["first_date"] == "2021-03-01"
    assert body["last_date"] == "2026-02-27"


@_skip_no_db
def test_future_phase_tables_remain_unchanged_after_api_use(client):
    """Phase 2A read-only API must not mutate later-phase research tables.

    Every downstream table's row count must be identical before and after
    a batch of ordinary read-only API calls — never a hard-coded
    emptiness assumption, which breaks the moment a later phase
    legitimately populates that table (as Phase 7B's frozen 46-period
    experiment now has for `portfolios`/`risk_metrics`).
    """
    conn = db.get_connection()
    try:
        before = {t: db.fetch_scalar(conn, f"SELECT COUNT(*) FROM {t}") for t in UNCHANGED_DOWNSTREAM_TABLES}
    finally:
        conn.close()

    client.get("/health")
    client.get("/health/ready")
    client.get("/api/universe")
    client.get("/api/prices/AAPL")
    client.get("/api/prices/AAPL", params={"start": "2025-01-01", "end": "2025-01-31"})
    client.get("/api/market/summary")

    conn = db.get_connection()
    try:
        for table, before_count in before.items():
            after_count = db.fetch_scalar(conn, f"SELECT COUNT(*) FROM {table}")
            assert after_count == before_count, (
                f"{table} row count changed across Phase 2A API use ({before_count} -> {after_count})"
            )
    finally:
        conn.close()
