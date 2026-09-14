"""
Phase 2A CI-safe unit tests for the FastAPI skeleton (app/). These never
require a live database — the `get_db` dependency is overridden with an
in-memory fake connection/cursor via FastAPI's dependency_overrides. Real
database behavior (against the live Supabase Postgres) is covered separately
in tests/test_api_db_integration.py, which follows the existing `db` marker
pattern (tests/test_db_integration.py) and is skipped when DATABASE_URL is
unset (e.g. in CI).
"""
from __future__ import annotations

import pathlib
from datetime import date

import psycopg2
import pytest
from fastapi.testclient import TestClient

import config
from app.db import get_db
from app.main import app


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    def execute(self, sql, params=None):
        pass

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnection:
    def __init__(self, rows):
        self._rows = rows

    def cursor(self):
        return _FakeCursor(self._rows)

    def close(self):
        pass


PRICE_ROWS = [
    (date(2024, 1, 2), 100.0, 101.0, 99.0, 100.5, 1_000_000, 250.0),
    (date(2024, 1, 3), 100.5, 102.0, 100.0, 101.5, 1_100_000, 252.0),
    (date(2024, 1, 4), 101.5, 103.0, 101.0, 102.5, 1_050_000, 254.0),
]


def _get_db_returning(rows):
    def _fake_get_db():
        yield _FakeConnection(rows)

    return _fake_get_db


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_readiness_reports_unavailable_when_db_unreachable(client, monkeypatch):
    import app.routes.health as health_module

    monkeypatch.setattr(health_module, "check_connectivity", lambda: False)
    resp = client.get("/health/ready")
    assert resp.status_code == 503
    assert resp.json() == {"status": "error", "database": "unavailable"}


def test_readiness_reports_ok_when_db_reachable(client, monkeypatch):
    import app.routes.health as health_module

    monkeypatch.setattr(health_module, "check_connectivity", lambda: True)
    resp = client.get("/health/ready")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "database": "connected"}


# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
def test_universe_returns_config_universe(client):
    tickers = [(t,) for t in sorted(config.TICKER_UNIVERSE)]
    app.dependency_overrides[get_db] = _get_db_returning(tickers)
    resp = client.get("/api/universe")
    assert resp.status_code == 200
    body = resp.json()
    assert {row["ticker"] for row in body} == set(config.TICKER_UNIVERSE)
    assert all("sector" in row for row in body)  # present (possibly null)


# ---------------------------------------------------------------------------
# Prices
# ---------------------------------------------------------------------------
def test_prices_valid_ticker_returns_chronological_rows(client):
    app.dependency_overrides[get_db] = _get_db_returning(PRICE_ROWS)
    resp = client.get("/api/prices/AAPL")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker"] == "AAPL"
    assert body["count"] == len(PRICE_ROWS)
    dates = [row["date"] for row in body["prices"]]
    assert dates == sorted(dates)


def test_prices_case_insensitive_uses_canonical_symbol(client):
    app.dependency_overrides[get_db] = _get_db_returning(PRICE_ROWS)
    resp = client.get("/api/prices/aapl")
    assert resp.status_code == 200
    assert resp.json()["ticker"] == "AAPL"


def test_prices_unknown_ticker_returns_404(client):
    app.dependency_overrides[get_db] = _get_db_returning([])
    resp = client.get("/api/prices/NOT_A_REAL_TICKER")
    assert resp.status_code == 404


def test_prices_invalid_date_range_returns_400(client):
    app.dependency_overrides[get_db] = _get_db_returning(PRICE_ROWS)
    resp = client.get("/api/prices/AAPL", params={"start": "2024-02-01", "end": "2024-01-01"})
    assert resp.status_code == 400


def test_prices_date_filtering_reflected_in_response(client):
    app.dependency_overrides[get_db] = _get_db_returning(PRICE_ROWS[:1])
    resp = client.get("/api/prices/AAPL", params={"start": "2024-01-02", "end": "2024-01-02"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 1
    assert body["start"] == "2024-01-02"
    assert body["end"] == "2024-01-02"


def test_prices_includes_required_fields(client):
    app.dependency_overrides[get_db] = _get_db_returning(PRICE_ROWS[:1])
    resp = client.get("/api/prices/AAPL")
    row = resp.json()["prices"][0]
    for field in ("date", "open", "high", "low", "close", "volume", "total_return_idx"):
        assert field in row


# ---------------------------------------------------------------------------
# Market summary
# ---------------------------------------------------------------------------
def test_market_summary_shape(client):
    app.dependency_overrides[get_db] = _get_db_returning(
        [(50, 62800, date(2021, 3, 1), date(2026, 2, 27))]
    )
    resp = client.get("/api/market/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ticker_count"] == 50
    assert body["price_row_count"] == 62800
    assert body["first_date"] == "2021-03-01"
    assert body["last_date"] == "2026-02-27"


# ---------------------------------------------------------------------------
# Credential / error-handling safety
# ---------------------------------------------------------------------------
def test_no_credential_leakage_on_db_failure(client):
    """A DB connection failure must surface as a generic 503 — never the raw
    psycopg2 message, which can carry host/user/connection details."""

    def _broken_get_db():
        raise psycopg2.OperationalError(
            'connection to server at "db.supabase.internal-secret-host" (10.0.0.1), '
            "port 5432 failed: FATAL: password authentication failed for user \"postgres\""
        )
        yield  # pragma: no cover - unreachable, keeps this a generator function

    app.dependency_overrides[get_db] = _broken_get_db
    resp = client.get("/api/market/summary")
    assert resp.status_code == 503
    assert resp.json() == {"detail": "Database unavailable"}
    assert "secret-host" not in resp.text
    assert "password" not in resp.text.lower()
    assert "postgres" not in resp.text.lower()


# ---------------------------------------------------------------------------
# Future-phase tables must never be referenced by Phase 2A app code
# ---------------------------------------------------------------------------
def test_future_phase_tables_never_referenced_in_app_code():
    forbidden_tables = ["features", "predictions", "portfolios", "risk_metrics"]
    app_dir = pathlib.Path(__file__).resolve().parent.parent / "app"
    for path in app_dir.rglob("*.py"):
        text = path.read_text().lower()
        for table in forbidden_tables:
            assert table not in text, f"{path} references future-phase table '{table}'"
