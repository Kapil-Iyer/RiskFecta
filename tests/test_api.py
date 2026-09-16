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
import re
from datetime import date

import pandas as pd
import psycopg2
import pytest
from fastapi.testclient import TestClient

import config
from app.db import get_db
from app.main import app

_SQL_MUTATION_KEYWORDS = re.compile(r"\b(insert\s+into|update\s+\w+\s+set|delete\s+from|drop\s+table|truncate|alter\s+table)\b")


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
# Predictions (Phase 4-6 frozen walk-forward forecasts — Phase 8B)
# ---------------------------------------------------------------------------
PREDICTION_ROWS = [
    ("AAPL", date(2022, 3, 25), 0.03, 0.05, 0.04, 0.06, True),
    ("MSFT", date(2022, 3, 25), -0.01, 0.02, 0.005, -0.02, False),
    ("NVDA", date(2022, 3, 25), 0.08, 0.07, 0.075, 0.09, True),
]


def test_prediction_dates_returns_list_of_dates(client):
    rows = [(date(2022, 2, 25),), (date(2022, 3, 28),), (date(2026, 1, 2),)]
    app.dependency_overrides[get_db] = _get_db_returning(rows)
    resp = client.get("/api/predictions/dates")
    assert resp.status_code == 200
    assert resp.json() == ["2022-02-25", "2022-03-28", "2026-01-02"]


def test_predictions_explicit_formation_date_returns_all_model_fields(client):
    app.dependency_overrides[get_db] = _get_db_returning(PREDICTION_ROWS)
    resp = client.get("/api/predictions", params={"formation_date": "2022-02-25"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["formation_date"] == "2022-02-25"
    assert body["target_date"] == "2022-03-25"
    assert body["count"] == 3
    for row in body["predictions"]:
        for field in ("ticker", "xgb_pred", "lstm_pred", "ensemble_pred", "actual_return", "directional_correct"):
            assert field in row


def test_predictions_unknown_formation_date_returns_404(client):
    app.dependency_overrides[get_db] = _get_db_returning([])
    resp = client.get("/api/predictions", params={"formation_date": "2099-01-01"})
    assert resp.status_code == 404


def test_predictions_malformed_formation_date_returns_422(client):
    resp = client.get("/api/predictions", params={"formation_date": "not-a-date"})
    assert resp.status_code == 422


class _FakeCursorDefaultDate:
    """Distinguishes the MAX(forecast_date) lookup from the cross-section
    SELECT within one request — the shared `_FakeCursor` above returns the
    same fixed rows for every call, which can't model this two-query path."""

    def __init__(self, max_date, cross_section_rows):
        self._max_date = max_date
        self._cross_section_rows = cross_section_rows

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        return (self._max_date,)

    def fetchall(self):
        return self._cross_section_rows

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


class _FakeConnDefaultDate:
    def __init__(self, max_date, cross_section_rows):
        self._cursor = _FakeCursorDefaultDate(max_date, cross_section_rows)

    def cursor(self):
        return self._cursor

    def close(self):
        pass


def test_predictions_defaults_to_latest_formation_date_when_omitted(client):
    def _fake_get_db():
        yield _FakeConnDefaultDate(date(2026, 1, 2), PREDICTION_ROWS)

    app.dependency_overrides[get_db] = _fake_get_db
    resp = client.get("/api/predictions")
    assert resp.status_code == 200
    assert resp.json()["formation_date"] == "2026-01-02"


# Note: deterministic base ordering (`ORDER BY ticker ASC`) is a real-SQL
# guarantee that a mocked cursor can't meaningfully verify (it returns
# whatever rows it's given, regardless of the query text) — it's covered by
# the live-DB test in test_api_db_integration.py, and separately by a pure
# unit test of the frontend's client-side ranking function
# (frontend/src/pages/forecastRanking.test.ts).


# ---------------------------------------------------------------------------
# Model comparison (Phase 4-6 frozen historical evaluation — Phase 8B)
# ---------------------------------------------------------------------------
def _synthetic_aligned_predictions() -> pd.DataFrame:
    """A tiny, hand-computable stand-in for `load_aligned_predictions`'s
    real (2,350-row) output — this test exercises the ROUTE's wiring
    (provenance labels, metadata, schema), not metric arithmetic, which is
    already covered by tests/test_metrics.py and tests/test_ensemble.py."""
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "forecast_date": pd.to_datetime(["2022-02-25", "2022-02-25", "2022-03-28", "2022-03-28"]),
            "target_date": pd.to_datetime(["2022-03-25", "2022-03-25", "2022-04-27", "2022-04-27"]),
            "xgb_pred": [0.01, -0.02, 0.03, 0.00],
            "lstm_pred": [0.02, -0.01, 0.01, 0.02],
            "actual_return": [0.015, -0.01, 0.02, -0.01],
            "directional_correct": [True, True, True, False],
        }
    )


def test_model_comparison_shape_and_provenance(client, monkeypatch):
    import app.routes.models as models_route

    monkeypatch.setattr(models_route, "load_aligned_predictions", lambda conn: _synthetic_aligned_predictions())
    app.dependency_overrides[get_db] = _get_db_returning([])  # unused by the monkeypatched loader

    resp = client.get("/api/models/comparison")
    assert resp.status_code == 200
    body = resp.json()

    assert body["experiment_type"] == "historical_walk_forward_oos"
    assert body["target_horizon_sessions"] == 21
    assert body["fold_count"] == 2
    assert body["prediction_count"] == 4
    assert body["formation_date_start"] == "2022-02-25"
    assert body["formation_date_end"] == "2022-03-28"

    models_by_key = {m["model"]: m for m in body["models"]}
    assert set(models_by_key) == {"historical_mean", "momentum_3m", "ridge", "xgb_pred", "lstm_pred", "ensemble_pred"}
    for key in ("historical_mean", "momentum_3m", "ridge"):
        assert models_by_key[key]["source"] == "frozen_baseline_constant"
    for key in ("xgb_pred", "lstm_pred", "ensemble_pred"):
        assert models_by_key[key]["source"] == "computed_from_persisted_predictions"

    # Frozen baseline values are the exact authoritative constants, never derived.
    assert models_by_key["historical_mean"]["mae"] == 0.07312
    assert models_by_key["momentum_3m"]["directional_accuracy"] == 0.5077
    assert models_by_key["ridge"]["spearman_corr"] == -0.03153

    metric_keys = {m["key"] for m in body["metric_definitions"]}
    assert metric_keys == {"mae", "rmse", "directional_accuracy", "pearson_corr", "spearman_corr"}
    directions = {m["key"]: m["direction"] for m in body["metric_definitions"]}
    assert directions["mae"] == "lower_is_better"
    assert directions["rmse"] == "lower_is_better"
    assert directions["directional_accuracy"] == "higher_is_better"

    assert body["disagreement"]["source"] == "computed_from_persisted_predictions"
    assert body["disagreement"]["n_total"] == 4


def test_model_comparison_no_overall_winner_field():
    """The response must never carry a single ranked/"best model" field —
    the evidence is metric-mixed by design (§5)."""
    import app.schemas as schemas

    field_names = set(schemas.ModelComparisonResponse.model_fields.keys())
    assert not field_names & {"best_model", "winner", "overall_score", "overall_rank"}


# ---------------------------------------------------------------------------
# Research API is read-only: no mutating SQL keyword anywhere in app/routes/
# ---------------------------------------------------------------------------
def test_research_routes_never_issue_mutating_sql():
    routes_dir = pathlib.Path(__file__).resolve().parent.parent / "app" / "routes"
    for path in routes_dir.glob("*.py"):
        text = path.read_text().lower()
        assert not _SQL_MUTATION_KEYWORDS.search(text), f"{path} appears to contain mutating SQL"


# ---------------------------------------------------------------------------
# Not-yet-authorized research tables must still never be referenced in app/
# code. `predictions` was authorized for Phase 8B (Forecast Rankings) and is
# deliberately removed from this list — see app/routes/predictions.py.
# `features`, `portfolios`, and `risk_metrics` remain unauthorized until
# their own future Phase 8 slices (Model Comparison, Portfolio Construction,
# Efficient Frontier, Risk Analytics, Historical Evidence).
# ---------------------------------------------------------------------------
def test_unauthorized_future_phase_tables_never_referenced_in_app_code():
    forbidden_tables = ["features", "portfolios", "risk_metrics"]
    app_dir = pathlib.Path(__file__).resolve().parent.parent / "app"
    for path in app_dir.rglob("*.py"):
        text = path.read_text().lower()
        for table in forbidden_tables:
            assert table not in text, f"{path} references not-yet-authorized table '{table}'"
