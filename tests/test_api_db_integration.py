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

# Still-empty downstream research tables the Phase 2A API must never touch.
# `features` (Phase 3) and `predictions` (Phase 4) are intentionally omitted:
# both may already hold legitimate rows; the API invariant for them is
# "row count unchanged across API calls"
# (see test_future_phase_tables_remain_empty_after_api_use).
EMPTY_DOWNSTREAM_TABLES = ["portfolios", "risk_metrics"]
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
def test_future_phase_tables_remain_empty_after_api_use(client):
    """Phase 2A read-only API must not mutate later-phase research tables.

    `features` (Phase 3) and `predictions` (Phase 4) may already be populated —
    assert their row counts are unchanged across the API calls rather than
    hard-coding emptiness or a fixed size. Still-unused tables
    (portfolios/risk_metrics) must remain empty.
    """
    conn = db.get_connection()
    try:
        features_before = db.fetch_scalar(conn, "SELECT COUNT(*) FROM features")
        predictions_before = db.fetch_scalar(conn, "SELECT COUNT(*) FROM predictions")
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
        features_after = db.fetch_scalar(conn, "SELECT COUNT(*) FROM features")
        predictions_after = db.fetch_scalar(conn, "SELECT COUNT(*) FROM predictions")
        assert features_after == features_before, (
            f"features row count changed across Phase 2A API use "
            f"({features_before} -> {features_after})"
        )
        assert predictions_after == predictions_before, (
            f"predictions row count changed across Phase 2A API use "
            f"({predictions_before} -> {predictions_after})"
        )
        for table in EMPTY_DOWNSTREAM_TABLES:
            n = db.fetch_scalar(conn, f"SELECT COUNT(*) FROM {table}")
            assert n == 0, f"{table} is not empty after Phase 2A API use"
    finally:
        conn.close()
